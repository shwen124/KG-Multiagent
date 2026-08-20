"""Fast CWQ grounding using CompactRelIndex (CWQ-relation filtered, in-memory)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.kg.compact_rel_index import (
    CompactRelIndex,
    collect_cwq_relation_ids,
    load_mid2vid_subset,
)
from src.workload.freebase_normalize import (
    canonicalize_query_triples,
    load_reverse_properties,
)


def is_var(x: str) -> bool:
    return x.startswith("?")


def collect_constant_mids(train_json: Path, relation2id: Dict[str, int], reverse_map: Dict[str, str]) -> Set[str]:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    mids: Set[str] = set()
    for sample in data:
        canon = canonicalize_query_triples(sample.get("sparql") or "", relation2id, reverse_map)
        for t in canon["triples"]:
            for n in (t["s"], t["o"]):
                if not is_var(n):
                    mids.add(n)
    return mids


def ground_one(
    sparql: str,
    index: CompactRelIndex,
    mid2vid: Dict[str, int],
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> dict:
    canon = canonicalize_query_triples(sparql, relation2id, reverse_map)
    coverage = {
        "entity_missing": 0,
        "relation_missing": canon["stats"]["relation_missing"],
        "reverse_relation_rewritten": canon["stats"]["reverse_rewritten"],
        "metadata_relation_resolved": canon["stats"]["metadata_skipped"],
        "unsupported_sparql": 0,
        "empty_result": 0,
    }
    triples = canon["triples"]
    if not triples:
        coverage["unsupported_sparql"] = int(canon["stats"]["raw_triples"] == 0)
        return {
            "success": False,
            "partial": False,
            "vertices": [],
            "edges": [],
            "witnesses": [],
            "coverage": coverage,
            "reason": "no_structural_triples",
        }

    patterns: List[Tuple[str, int, str, str]] = []
    for t in triples:
        rid = relation2id.get(t["r"])
        if rid is None:
            coverage["relation_missing"] += 1
            continue
        s, o = t["s"], t["o"]
        if not is_var(s) and s not in mid2vid:
            coverage["entity_missing"] += 1
        if not is_var(o) and o not in mid2vid:
            coverage["entity_missing"] += 1
        patterns.append((s, rid, o, t["r"]))

    if not patterns:
        return {
            "success": False,
            "partial": False,
            "vertices": [],
            "edges": [],
            "witnesses": [],
            "coverage": coverage,
            "reason": "relation_missing",
        }

    # domains: var -> set of vids
    variables = set()
    for s, _r, o, _ in patterns:
        if is_var(s):
            variables.add(s)
        if is_var(o):
            variables.add(o)
    domain: Dict[str, Optional[Set[int]]] = {v: None for v in variables}

    def const_vid(term: str) -> Optional[int]:
        if is_var(term):
            return None
        return mid2vid.get(term)

    # Seed + propagate (few rounds)
    for _ in range(6):
        changed = False
        for s, rid, o, _rname in patterns:
            if not is_var(s) and is_var(o):
                hv = const_vid(s)
                if hv is None:
                    continue
                cand = set(index.tails(hv, rid))
                old = domain[o]
                new = cand if old is None else old & cand
                if new != old:
                    domain[o] = new
                    changed = True
            elif is_var(s) and not is_var(o):
                tv = const_vid(o)
                if tv is None:
                    continue
                cand = set(index.heads(tv, rid))
                old = domain[s]
                new = cand if old is None else old & cand
                if new != old:
                    domain[s] = new
                    changed = True
            elif is_var(s) and is_var(o):
                # expand from whichever side has a domain
                if domain[s]:
                    cand_o: Set[int] = set()
                    for hv in list(domain[s])[:300]:
                        cand_o.update(index.tails(hv, rid))
                    old = domain[o]
                    new = cand_o if old is None else old & cand_o
                    if new != old:
                        domain[o] = new
                        changed = True
                if domain[o]:
                    cand_s: Set[int] = set()
                    for tv in list(domain[o])[:300]:
                        cand_s.update(index.heads(tv, rid))
                    old = domain[s]
                    new = cand_s if old is None else old & cand_s
                    if new != old:
                        domain[s] = new
                        changed = True
            else:
                # both const: just check later
                pass
        if not changed:
            break

    # Pick one binding
    binding: Dict[str, int] = {}
    for v in variables:
        vals = domain.get(v)
        if vals:
            binding[v] = next(iter(vals))

    edge_set: Set[Tuple[int, str, int]] = set()
    vertex_set: Set[int] = set()

    def resolve(term: str) -> Optional[int]:
        if is_var(term):
            return binding.get(term)
        return const_vid(term)

    matched_patterns = 0
    for s, rid, o, rname in patterns:
        hv, tv = resolve(s), resolve(o)
        if hv is None or tv is None:
            continue
        if index.has_edge(hv, rid, tv):
            matched_patterns += 1
            edge_set.add((hv, rname, tv))
            vertex_set.add(hv)
            vertex_set.add(tv)

    # constant-constant facts even without vars
    for s, rid, o, rname in patterns:
        if not is_var(s) and not is_var(o):
            hv, tv = const_vid(s), const_vid(o)
            if hv is not None and tv is not None and index.has_edge(hv, rid, tv):
                edge_set.add((hv, rname, tv))
                vertex_set.add(hv)
                vertex_set.add(tv)
                matched_patterns += 1

    success = len(edge_set) > 0
    unbound = [v for v in variables if v not in binding]
    partial = success and (len(unbound) > 0 or matched_patterns < len(patterns) or coverage["entity_missing"] > 0)
    if not success:
        coverage["empty_result"] = 1

    witnesses = [sorted(vertex_set)] if vertex_set else []
    return {
        "success": success,
        "partial": partial,
        "vertices": sorted(vertex_set),
        "edges": [[h, r, t] for h, r, t in sorted(edge_set)],
        "witnesses": witnesses,
        "coverage": coverage,
        "reason": None if success else "empty_result",
    }


def run_cwq_coverage_fast(
    train_json: Path,
    processed_graph_dir: Path,
    reverse_properties_path: Path,
    out_dir: Path,
    limit: Optional[int] = None,
    index_cache: Optional[Path] = None,
) -> dict:
    relation2id = json.loads((processed_graph_dir / "relation2id.json").read_text(encoding="utf-8"))
    reverse_map = load_reverse_properties(reverse_properties_path)

    print("Collecting CWQ relation ids / constant MIDs ...", flush=True)
    rids = collect_cwq_relation_ids(train_json, relation2id, reverse_map)
    const_mids = collect_constant_mids(train_json, relation2id, reverse_map)
    print(f"  relations={len(rids)} constants={len(const_mids)}", flush=True)

    print("Loading mid2vid subset from vertices.tsv ...", flush=True)
    mid2vid = load_mid2vid_subset(processed_graph_dir / "vertices.tsv", const_mids)
    print(f"  mapped {len(mid2vid)}/{len(const_mids)}", flush=True)

    cache = index_cache or (processed_graph_dir / "cwq_rel_index.pkl")
    if cache.exists():
        print(f"Loading compact index {cache} ...", flush=True)
        index = CompactRelIndex.load(cache)
    else:
        print("Building compact relation index (one scan of edges.tsv) ...", flush=True)
        index = CompactRelIndex.build_from_edges(processed_graph_dir / "edges.tsv", rids)
        print(f"Saving {cache} ...", flush=True)
        index.save(cache)

    data = json.loads(train_json.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "query_traces.jsonl"
    failed_path = out_dir / "failed_queries.jsonl"

    coverage = {
        "total_train_queries": len(data),
        "success": 0,
        "partial": 0,
        "failed": 0,
        "entity_missing": 0,
        "relation_missing": 0,
        "reverse_relation_rewritten": 0,
        "metadata_relation_resolved": 0,
        "empty_result": 0,
        "unsupported_sparql": 0,
    }

    with traces_path.open("w", encoding="utf-8") as ft, failed_path.open("w", encoding="utf-8") as ff:
        for i, sample in enumerate(data):
            result = ground_one(sample.get("sparql") or "", index, mid2vid, relation2id, reverse_map)
            record = {
                "qid": sample.get("ID", i),
                "question": sample.get("question"),
                "compositionality_type": sample.get("compositionality_type"),
                **result,
            }
            ft.write(json.dumps(record, ensure_ascii=False) + "\n")
            c = result["coverage"]
            coverage["entity_missing"] += int(c.get("entity_missing", 0) > 0)
            coverage["relation_missing"] += int(c.get("relation_missing", 0) > 0)
            coverage["reverse_relation_rewritten"] += int(c.get("reverse_relation_rewritten", 0) > 0)
            coverage["metadata_relation_resolved"] += int(c.get("metadata_relation_resolved", 0) > 0)
            coverage["empty_result"] += int(c.get("empty_result", 0) > 0)
            coverage["unsupported_sparql"] += int(c.get("unsupported_sparql", 0) > 0)
            if result["success"]:
                coverage["success"] += 1
                if result["partial"]:
                    coverage["partial"] += 1
            else:
                coverage["failed"] += 1
                ff.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 2000 == 0:
                print(
                    f"  CWQ fast {i+1}/{len(data)} success={coverage['success']} fail={coverage['failed']}",
                    flush=True,
                )

    (out_dir / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    return coverage
