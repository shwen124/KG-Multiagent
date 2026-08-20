"""Fast GrailQA grounding on shared FB+CVT-REV.

Primary topology: graph_query (entity/class/literal).
Grounding: structural entity–entity edges via CompactRelIndex
(class → variables; literal edges skipped; functions ignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.kg.compact_rel_index import (
    CompactRelIndex,
    collect_relation_ids_from_sparqls,
    load_mid2vid_subset,
)
from src.workload.freebase_normalize import (
    canonicalize_triple,
    is_metadata_relation,
    load_reverse_properties,
    normalize_ns_token,
)
from src.workload.trace_cwq_fb_fast import ground_one, is_var
from src.workload.trace_webqsp_fb_fast import ensure_freebase_rel_index


def entity_id_to_mid(eid: str) -> str:
    eid = str(eid).strip()
    if eid.startswith("ns:") or eid.startswith(":"):
        return normalize_ns_token(eid)
    if eid.startswith("/"):
        return eid
    if eid.startswith(("m.", "g.", "en.")):
        return "/" + eid.replace(".", "/", 1)
    return "/" + eid.replace(".", "/")


def graph_query_to_pseudo_sparql(graph_query: dict) -> str:
    """Encode structural graph_query edges as ns: triple patterns for ground_one."""
    nodes = {int(n["nid"]): n for n in (graph_query or {}).get("nodes") or []}
    lines: List[str] = []
    for e in (graph_query or {}).get("edges") or []:
        sn = nodes.get(int(e["start"]))
        en = nodes.get(int(e["end"]))
        if not sn or not en:
            continue
        if sn.get("node_type") == "literal" or en.get("node_type") == "literal":
            continue
        rel = str(e.get("relation") or "")
        if not rel:
            continue
        r_path = "/" + rel.replace(".", "/") if not rel.startswith("/") else rel
        if is_metadata_relation(r_path):
            continue

        def term(node: dict) -> str:
            if node.get("node_type") == "entity":
                mid = entity_id_to_mid(node["id"])
                # /m/abc -> ns:m.abc
                body = mid.lstrip("/").replace("/", ".", 1)
                return "ns:" + body
            return f"?n{int(node['nid'])}"

        rel_ns = "ns:" + rel.replace("/", ".")
        lines.append(f"{term(sn)} {rel_ns} {term(en)} .")
    return "\n".join(lines)


def collect_entity_mids(samples: Iterable[dict]) -> Set[str]:
    mids: Set[str] = set()
    for s in samples:
        for n in (s.get("graph_query") or {}).get("nodes") or []:
            if n.get("node_type") == "entity":
                mids.add(entity_id_to_mid(n["id"]))
        # also constants appearing in sparql_query
        sparql = s.get("sparql_query") or ""
        for tok in sparql.replace("\n", " ").split():
            if tok.startswith(":m.") or tok.startswith("ns:m.") or tok.startswith(":g."):
                mids.add(normalize_ns_token(tok.rstrip(".")))
    return mids


def collect_grailqa_relation_ids(
    samples: Iterable[dict],
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> Set[int]:
    rids: Set[int] = set()
    sparqls: List[str] = []
    for s in samples:
        gq = s.get("graph_query") or {}
        nodes = {int(n["nid"]): n for n in gq.get("nodes") or []}
        for e in gq.get("edges") or []:
            sn = nodes.get(int(e["start"]))
            en = nodes.get(int(e["end"]))
            if not sn or not en:
                continue
            if sn.get("node_type") == "literal" or en.get("node_type") == "literal":
                continue
            rel = str(e.get("relation") or "")
            r = "/" + rel.replace(".", "/") if rel and not rel.startswith("/") else rel
            if not r or is_metadata_relation(r):
                continue
            _s, cr, _o, status = canonicalize_triple("?s", r, "?o", relation2id, reverse_map)
            if status in {"ok", "reverse_rewritten"}:
                rid = relation2id.get(cr)
                if rid is not None:
                    rids.add(rid)
        sparqls.append(graph_query_to_pseudo_sparql(gq))
        if s.get("sparql_query"):
            sparqls.append(s["sparql_query"])
    rids |= collect_relation_ids_from_sparqls(sparqls, relation2id, reverse_map)
    return rids


def ground_grailqa_sample(
    sample: dict,
    index: CompactRelIndex,
    mid2vid: Dict[str, int],
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> dict:
    gq = sample.get("graph_query") or {}
    pseudo = graph_query_to_pseudo_sparql(gq)
    result = ground_one(pseudo, index, mid2vid, relation2id, reverse_map)

    # Fallback: sparql_query structural patterns if graph_query failed
    if not result["success"]:
        sparql = sample.get("sparql_query") or ""
        alt = ground_one(sparql, index, mid2vid, relation2id, reverse_map)
        if alt["success"]:
            alt["coverage"]["used_sparql_fallback"] = 1
            result = alt
            result["reason"] = None
        else:
            # merge coverage flags
            for k, v in alt["coverage"].items():
                if isinstance(v, (int, float)) and v:
                    result["coverage"][k] = max(int(result["coverage"].get(k, 0) or 0), int(v))
            if not result.get("reason"):
                result["reason"] = alt.get("reason") or "empty_result"

    # Annotate node-type stats (not vertices)
    ntypes = {}
    for n in gq.get("nodes") or []:
        t = n.get("node_type") or "?"
        ntypes[t] = ntypes.get(t, 0) + 1
    result["node_type_counts"] = ntypes
    result["function"] = sample.get("function") or "none"
    result["domains"] = sample.get("domains")
    return result


def run_grailqa_coverage_fast(
    train_json: Path,
    processed_graph_dir: Path,
    reverse_properties_path: Path,
    out_dir: Path,
    index_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    dev_json: Optional[Path] = None,
) -> dict:
    relation2id = json.loads((processed_graph_dir / "relation2id.json").read_text(encoding="utf-8"))
    reverse_map = load_reverse_properties(reverse_properties_path)
    samples = json.loads(train_json.read_text(encoding="utf-8"))
    if limit is not None:
        samples = samples[:limit]

    print("Collecting GrailQA relation ids / entity MIDs ...", flush=True)
    rids = collect_grailqa_relation_ids(samples, relation2id, reverse_map)
    const_mids = collect_entity_mids(samples)
    print(f"  relations={len(rids)} entity_mids={len(const_mids)}", flush=True)

    print("Loading mid2vid subset from vertices.tsv ...", flush=True)
    mid2vid = load_mid2vid_subset(processed_graph_dir / "vertices.tsv", const_mids)
    print(f"  mapped {len(mid2vid)}/{len(const_mids)}", flush=True)

    idx_dir = index_dir or (processed_graph_dir.parent / "index")
    index = ensure_freebase_rel_index(
        processed_graph_dir=processed_graph_dir,
        needed_rids=rids,
        index_dir=idx_dir,
        seed_cache=idx_dir / "freebase_rel_index.pkl",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "query_traces.jsonl"
    failed_path = out_dir / "failed_queries.jsonl"
    normalized_path = out_dir / "normalized_queries.jsonl"

    coverage = {
        "total_train_questions": len(samples),
        "successful_questions": 0,
        "partial_questions": 0,
        "failed_questions": 0,
        "success_rate": 0.0,
        "entity_missing": 0,
        "relation_missing": 0,
        "reverse_relation_rewritten": 0,
        "metadata_relation_resolved": 0,
        "empty_result": 0,
        "unsupported_sparql": 0,
        "no_entity_anchor": 0,
        "sparql_fallback_success": 0,
        "raw_task_hyperedges": 0,
        "unique_task_hyperedges": 0,
        "by_function": {},
    }
    seen_witness: Set[str] = set()

    with traces_path.open("w", encoding="utf-8") as ft, failed_path.open(
        "w", encoding="utf-8"
    ) as ff, normalized_path.open("w", encoding="utf-8") as fn:
        for i, sample in enumerate(samples):
            has_entity = any(
                n.get("node_type") == "entity"
                for n in (sample.get("graph_query") or {}).get("nodes") or []
            )
            if not has_entity:
                coverage["no_entity_anchor"] += 1

            result = ground_grailqa_sample(sample, index, mid2vid, relation2id, reverse_map)
            fn.write(
                json.dumps(
                    {
                        "qid": sample.get("qid"),
                        "pseudo_sparql": graph_query_to_pseudo_sparql(sample.get("graph_query") or {}),
                        "function": sample.get("function"),
                        "domains": sample.get("domains"),
                        "grounding": {
                            "success": result["success"],
                            "partial": result["partial"],
                            "reason": result.get("reason"),
                            "coverage": result["coverage"],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            witnesses: List[List[int]] = []
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

            record = {
                "qid": sample.get("qid"),
                "question": sample.get("question"),
                "function": sample.get("function") or "none",
                "domains": sample.get("domains"),
                "success": result["success"],
                "partial": result["partial"],
                "vertices": result["vertices"],
                "edges": result["edges"],
                "witnesses": witnesses,
                "coverage": result["coverage"],
                "node_type_counts": result.get("node_type_counts"),
                "reason": result.get("reason"),
            }
            ft.write(json.dumps(record, ensure_ascii=False) + "\n")

            c = result["coverage"]
            for k in (
                "entity_missing",
                "relation_missing",
                "reverse_relation_rewritten",
                "metadata_relation_resolved",
                "empty_result",
                "unsupported_sparql",
            ):
                coverage[k] += int(int(c.get(k, 0) or 0) > 0)
            if int(c.get("used_sparql_fallback", 0) or 0) > 0 and result["success"]:
                coverage["sparql_fallback_success"] += 1

            fn_key = str(sample.get("function") or "none")
            bucket = coverage["by_function"].setdefault(
                fn_key, {"total": 0, "success": 0, "failed": 0}
            )
            bucket["total"] += 1

            if result["success"]:
                coverage["successful_questions"] += 1
                bucket["success"] += 1
                if result["partial"]:
                    coverage["partial_questions"] += 1
            else:
                coverage["failed_questions"] += 1
                bucket["failed"] += 1
                ff.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 2000 == 0:
                print(
                    f"  GrailQA {i+1}/{len(samples)} "
                    f"success={coverage['successful_questions']} fail={coverage['failed_questions']}",
                    flush=True,
                )

    total = coverage["total_train_questions"] or 1
    coverage["success_rate"] = coverage["successful_questions"] / total

    # Optional: official dev unseen-vertex coverage (stats only; never mutate train graph)
    if dev_json and dev_json.exists() and limit is None:
        print("Computing official-dev vertex coverage (stats only) ...", flush=True)
        coverage["dev_eval"] = _dev_vertex_coverage(
            dev_json=dev_json,
            index=index,
            relation2id=relation2id,
            reverse_map=reverse_map,
            processed_graph_dir=processed_graph_dir,
            train_vertex_set=_load_train_vertices(traces_path),
        )

    (out_dir / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return coverage


def _load_train_vertices(traces_path: Path) -> Set[int]:
    verts: Set[int] = set()
    with traces_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("success"):
                continue
            verts.update(int(v) for v in rec.get("vertices") or [])
    return verts


def _dev_vertex_coverage(
    dev_json: Path,
    index: CompactRelIndex,
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
    processed_graph_dir: Path,
    train_vertex_set: Set[int],
) -> dict:
    samples = json.loads(dev_json.read_text(encoding="utf-8"))
    const_mids = collect_entity_mids(samples)
    mid2vid = load_mid2vid_subset(processed_graph_dir / "vertices.tsv", const_mids)

    # ensure index covers any new dev relations (read-only for hypergraph; OK to extend index)
    rids = collect_grailqa_relation_ids(samples, relation2id, reverse_map)
    ensure_freebase_rel_index(
        processed_graph_dir=processed_graph_dir,
        needed_rids=rids,
        index_dir=processed_graph_dir.parent / "index",
        seed_cache=processed_graph_dir.parent / "index" / "freebase_rel_index.pkl",
    )
    # reload in case merged
    index = CompactRelIndex.load(
        processed_graph_dir.parent / "index" / "freebase_rel_index.pkl"
    )

    dev_total = len(samples)
    groundable = 0
    witness_verts: Set[int] = set()
    fully = 0
    partial = 0

    for sample in samples:
        result = ground_grailqa_sample(sample, index, mid2vid, relation2id, reverse_map)
        if not result["success"]:
            continue
        groundable += 1
        vs = {int(v) for v in result["vertices"]}
        witness_verts |= vs
        if vs and vs <= train_vertex_set:
            fully += 1
        elif vs & train_vertex_set:
            partial += 1

    seen = witness_verts & train_vertex_set
    unseen = witness_verts - train_vertex_set
    return {
        "dev_total_queries": dev_total,
        "dev_groundable_queries": groundable,
        "dev_total_witness_vertices": len(witness_verts),
        "dev_seen_vertices": len(seen),
        "dev_unseen_vertices": len(unseen),
        "vertex_coverage": (len(seen) / len(witness_verts)) if witness_verts else 0.0,
        "fully_covered_query_rate": (fully / groundable) if groundable else 0.0,
        "partially_covered_query_rate": (partial / groundable) if groundable else 0.0,
        "note": "dev used only for coverage stats; not injected into train hypergraph",
    }
