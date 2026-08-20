"""Fast WebQSP grounding on FB+CVT-REV via shared CompactRelIndex.

Rules (数据处理说明2):
- authoritative workload = Parses[].Sparql (complete train only)
- multi-parse: query-level f_v/f_e (+1 per question, not per parse)
- task hyperedges: one witness per successful parse; identical pin-sets dedupe
- shared Freebase index under data/Freebase/index/freebase_rel_index.pkl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.kg.compact_rel_index import (
    CompactRelIndex,
    collect_relation_ids_from_sparqls,
    load_mid2vid_subset,
    merge_missing_relations,
)
from src.workload.freebase_normalize import (
    canonicalize_query_triples,
    load_reverse_properties,
    normalize_ns_token,
)
from src.workload.trace_cwq_fb_fast import ground_one, is_var


def load_webqsp_questions(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("Questions") or [])
    return list(data)


def iter_train_sparqls(questions: Iterable[dict]) -> Iterable[str]:
    for q in questions:
        for parse in q.get("Parses") or []:
            sparql = parse.get("Sparql") or ""
            if sparql.strip():
                yield sparql


def collect_constant_mids_from_questions(
    questions: List[dict],
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> Set[str]:
    mids: Set[str] = set()
    for q in questions:
        for parse in q.get("Parses") or []:
            topic = parse.get("TopicEntityMid")
            if topic:
                # m.06w2sn5 -> /m/06w2sn5
                mids.add(normalize_ns_token("ns:" + topic if not str(topic).startswith(("ns:", "/", "http")) else str(topic)))
            sparql = parse.get("Sparql") or ""
            canon = canonicalize_query_triples(sparql, relation2id, reverse_map)
            for t in canon["triples"]:
                for n in (t["s"], t["o"]):
                    if not is_var(n):
                        mids.add(n)
    return mids


def ensure_freebase_rel_index(
    processed_graph_dir: Path,
    needed_rids: Set[int],
    index_dir: Path,
    seed_cache: Optional[Path] = None,
) -> CompactRelIndex:
    """Load/merge shared Freebase compact index covering needed_rids."""
    index_dir.mkdir(parents=True, exist_ok=True)
    unified = index_dir / "freebase_rel_index.pkl"
    edges = processed_graph_dir / "edges.tsv"

    candidates = [
        unified,
        seed_cache,
        processed_graph_dir / "cwq_rel_index.pkl",
    ]
    index: Optional[CompactRelIndex] = None
    for c in candidates:
        if c and c.exists():
            print(f"Loading compact index {c} ...", flush=True)
            index = CompactRelIndex.load(c)
            break

    if index is None:
        print("Building compact Freebase index from scratch ...", flush=True)
        index = CompactRelIndex.build_from_edges(edges, needed_rids)
    else:
        merge_missing_relations(index, edges, needed_rids)

    print(f"Saving unified index -> {unified} ...", flush=True)
    index.save(unified)
    return index


def run_webqsp_coverage_fast(
    train_json: Path,
    processed_graph_dir: Path,
    reverse_properties_path: Path,
    out_dir: Path,
    index_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    relation2id = json.loads((processed_graph_dir / "relation2id.json").read_text(encoding="utf-8"))
    reverse_map = load_reverse_properties(reverse_properties_path)
    questions = load_webqsp_questions(train_json)
    if limit is not None:
        questions = questions[:limit]

    print("Collecting WebQSP relation ids / constant MIDs ...", flush=True)
    rids = collect_relation_ids_from_sparqls(iter_train_sparqls(questions), relation2id, reverse_map)
    const_mids = collect_constant_mids_from_questions(questions, relation2id, reverse_map)
    print(f"  relations={len(rids)} constants={len(const_mids)}", flush=True)

    print("Loading mid2vid subset from vertices.tsv ...", flush=True)
    mid2vid = load_mid2vid_subset(processed_graph_dir / "vertices.tsv", const_mids)
    print(f"  mapped {len(mid2vid)}/{len(const_mids)}", flush=True)

    idx_dir = index_dir or (processed_graph_dir.parent / "index")
    index = ensure_freebase_rel_index(
        processed_graph_dir=processed_graph_dir,
        needed_rids=rids,
        index_dir=idx_dir,
        seed_cache=processed_graph_dir / "cwq_rel_index.pkl",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "query_traces.jsonl"
    failed_path = out_dir / "failed_queries.jsonl"
    normalized_path = out_dir / "normalized_queries.jsonl"

    coverage = {
        "total_train_questions": len(questions),
        "successful_questions": 0,
        "partial_questions": 0,
        "failed_questions": 0,
        "success_rate": 0.0,
        "total_parses": 0,
        "successful_parses": 0,
        "entity_missing": 0,
        "relation_missing": 0,
        "reverse_relation_rewritten": 0,
        "metadata_relation_resolved": 0,
        "empty_result": 0,
        "unsupported_sparql": 0,
        "raw_task_hyperedges": 0,
        "unique_task_hyperedges": 0,
    }

    seen_witness: Set[str] = set()

    with traces_path.open("w", encoding="utf-8") as ft, failed_path.open(
        "w", encoding="utf-8"
    ) as ff, normalized_path.open("w", encoding="utf-8") as fn:
        for i, q in enumerate(questions):
            qid = q.get("QuestionId", f"WebQTrn-{i}")
            parses = q.get("Parses") or []
            q_vertices: Set[int] = set()
            q_edges: Set[Tuple[int, str, int]] = set()
            parse_results = []
            any_success = False
            any_partial = False
            witnesses: List[List[int]] = []

            for parse in parses:
                sparql = parse.get("Sparql") or ""
                coverage["total_parses"] += 1
                result = ground_one(sparql, index, mid2vid, relation2id, reverse_map)
                parse_results.append(
                    {
                        "parse_id": parse.get("ParseId"),
                        "success": result["success"],
                        "partial": result["partial"],
                        "reason": result.get("reason"),
                        "coverage": result["coverage"],
                        "n_vertices": len(result["vertices"]),
                        "n_edges": len(result["edges"]),
                    }
                )
                fn.write(
                    json.dumps(
                        {
                            "qid": qid,
                            "parse_id": parse.get("ParseId"),
                            "sparql": sparql,
                            "topic_mid": parse.get("TopicEntityMid"),
                            "inferential_chain": parse.get("InferentialChain"),
                            "grounding": parse_results[-1],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                c = result["coverage"]
                # parse-level flags accumulate as question-level presence below
                if result["success"]:
                    coverage["successful_parses"] += 1
                    any_success = True
                    if result["partial"]:
                        any_partial = True
                    q_vertices.update(int(v) for v in result["vertices"])
                    for e in result["edges"]:
                        q_edges.add((int(e[0]), str(e[1]), int(e[2])))
                    for wit in result.get("witnesses") or []:
                        pins = sorted({int(x) for x in wit})
                        if len(pins) < 2:
                            continue
                        coverage["raw_task_hyperedges"] += 1
                        key = "|".join(map(str, pins))
                        if key not in seen_witness:
                            seen_witness.add(key)
                            coverage["unique_task_hyperedges"] += 1
                        witnesses.append(pins)

            # query-level coverage flags (presence)
            q_cov_flags = {
                "entity_missing": 0,
                "relation_missing": 0,
                "reverse_relation_rewritten": 0,
                "metadata_relation_resolved": 0,
                "empty_result": 0,
                "unsupported_sparql": 0,
            }
            for pr in parse_results:
                c = pr["coverage"]
                for k in q_cov_flags:
                    if int(c.get(k, 0) or 0) > 0:
                        q_cov_flags[k] = 1

            for k, v in q_cov_flags.items():
                coverage[k] += v

            record = {
                "qid": qid,
                "question": q.get("RawQuestion") or q.get("ProcessedQuestion"),
                "n_parses": len(parses),
                "success": any_success,
                "partial": bool(any_success and any_partial),
                "vertices": sorted(q_vertices),
                "edges": [[h, r, t] for h, r, t in sorted(q_edges)],
                "witnesses": witnesses,
                "parse_results": parse_results,
                "coverage": q_cov_flags,
                "reason": None if any_success else "all_parses_failed",
            }
            ft.write(json.dumps(record, ensure_ascii=False) + "\n")

            if any_success:
                coverage["successful_questions"] += 1
                if any_partial:
                    coverage["partial_questions"] += 1
            else:
                coverage["failed_questions"] += 1
                ff.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 500 == 0:
                print(
                    f"  WebQSP {i+1}/{len(questions)} "
                    f"success={coverage['successful_questions']} fail={coverage['failed_questions']}",
                    flush=True,
                )

    total = coverage["total_train_questions"] or 1
    coverage["success_rate"] = coverage["successful_questions"] / total
    (out_dir / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return coverage
