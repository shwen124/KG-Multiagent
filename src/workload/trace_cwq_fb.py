"""CWQ SPARQL grounding against FB+CVT-REV via SQLite store."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.kg.index_fb_sqlite import FreebaseSQLiteStore
from src.workload.freebase_normalize import (
    canonicalize_query_triples,
    load_reverse_properties,
)


def is_var(x: str) -> bool:
    return x.startswith("?")


def load_maps(processed_dir: Path) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str]]:
    mid2vid = json.loads((processed_dir / "mid2vid.json").read_text(encoding="utf-8"))
    relation2id = json.loads((processed_dir / "relation2id.json").read_text(encoding="utf-8"))
    # rebuild id2rel from relations.tsv if needed
    id2rel: Dict[int, str] = {v: k for k, v in relation2id.items()}
    return mid2vid, relation2id, id2rel


def ground_cwq_query(
    sparql: str,
    store: FreebaseSQLiteStore,
    mid2vid: Dict[str, int],
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
    max_bindings: int = 200,
) -> dict:
    """Ground canonical structural triples; return witness vertices/edges.

    Strategy:
    - Build CSP over variables from known constants using outgoing/incoming edges.
    - Collect successful bindings' entity sets as witnesses (capped).
    """
    canon = canonicalize_query_triples(sparql, relation2id, reverse_map)
    triples = canon["triples"]
    coverage = {
        "entity_missing": 0,
        "relation_missing": canon["stats"]["relation_missing"],
        "reverse_relation_rewritten": canon["stats"]["reverse_rewritten"],
        "metadata_relation_resolved": canon["stats"]["metadata_skipped"],
        "unsupported_sparql": 0,
        "empty_result": 0,
    }

    if not triples:
        if canon["stats"]["raw_triples"] == 0:
            coverage["unsupported_sparql"] = 1
        return {
            "success": False,
            "partial": False,
            "vertices": [],
            "edges": [],
            "witnesses": [],
            "coverage": coverage,
            "reason": "no_structural_triples",
        }

    # check entities present
    for t in triples:
        for node in (t["s"], t["o"]):
            if not is_var(node) and node not in mid2vid:
                coverage["entity_missing"] += 1

    if coverage["entity_missing"] and all(
        (is_var(t["s"]) or t["s"] not in mid2vid) and (is_var(t["o"]) or t["o"] not in mid2vid)
        for t in triples
    ):
        return {
            "success": False,
            "partial": False,
            "vertices": [],
            "edges": [],
            "witnesses": [],
            "coverage": coverage,
            "reason": "entity_missing",
        }

    # Simple iterative binding: start from grounded endpoints
    # Represent each triple as (s_term, rid, o_term)
    patterns = []
    for t in triples:
        rid = relation2id.get(t["r"])
        if rid is None:
            coverage["relation_missing"] += 1
            continue
        patterns.append((t["s"], rid, t["o"], t["r"]))

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

    # Initialize domains
    variables: Set[str] = set()
    for s, _rid, o, _r in patterns:
        if is_var(s):
            variables.add(s)
        if is_var(o):
            variables.add(o)

    assignment: Dict[str, Set[int]] = {}
    for v in variables:
        assignment[v] = set()

    # Seed: for patterns with one constant entity, expand neighbors into var domain
    seeded = False
    for s, rid, o, _rname in patterns:
        if not is_var(s) and is_var(o) and s in mid2vid:
            vid = mid2vid[s]
            tails = store.tails(vid, rid)
            assignment[o].update(tails)
            seeded = True
        elif is_var(s) and not is_var(o) and o in mid2vid:
            vid = mid2vid[o]
            heads = store.heads(vid, rid)
            assignment[s].update(heads)
            seeded = True
        elif not is_var(s) and not is_var(o):
            seeded = True

    if not seeded and variables:
        coverage["empty_result"] = 1
        return {
            "success": False,
            "partial": True,
            "vertices": [],
            "edges": [],
            "witnesses": [],
            "coverage": coverage,
            "reason": "cannot_seed",
        }

    # Propagate a few rounds
    for _ in range(5):
        changed = False
        for s, rid, o, _rname in patterns:
            if not is_var(s) and is_var(o) and s in mid2vid:
                allowed = set(store.tails(mid2vid[s], rid))
                new = assignment[o] & allowed if assignment[o] else allowed
                if new != assignment[o]:
                    assignment[o] = new
                    changed = True
            elif is_var(s) and not is_var(o) and o in mid2vid:
                allowed = set(store.heads(mid2vid[o], rid))
                new = assignment[s] & allowed if assignment[s] else allowed
                if new != assignment[s]:
                    assignment[s] = new
                    changed = True
            elif is_var(s) and is_var(o):
                # prune o based on s candidates (cap)
                if assignment[s]:
                    allowed_o: Set[int] = set()
                    for hv in list(assignment[s])[:500]:
                        allowed_o.update(store.tails(hv, rid))
                    if assignment[o]:
                        new = assignment[o] & allowed_o
                    else:
                        new = allowed_o
                    if new != assignment[o]:
                        assignment[o] = new
                        changed = True
        if not changed:
            break

    # Collect grounded edges for constants + sample variable bindings
    edge_set: Set[Tuple[str, str, str]] = set()
    vertex_set: Set[str] = set()
    vid2mid = {v: k for k, v in mid2vid.items()}

    def add_edge(h_mid: str, rname: str, t_mid: str) -> None:
        vertex_set.add(h_mid)
        vertex_set.add(t_mid)
        edge_set.add((h_mid, rname, t_mid))

    # constant-constant facts
    for s, rid, o, rname in patterns:
        if not is_var(s) and not is_var(o) and s in mid2vid and o in mid2vid:
            if store.has_edge(mid2vid[s], rid, mid2vid[o]):
                add_edge(s, rname, o)

    # sample witnesses from variable assignments
    witnesses: List[List[str]] = []
    # pick one binding combination greedily
    binding: Dict[str, int] = {}
    for v in variables:
        if assignment.get(v):
            binding[v] = next(iter(assignment[v]))

    if binding or any(not is_var(s) and not is_var(o) for s, _, o, _ in patterns):
        for s, rid, o, rname in patterns:
            hs = mid2vid[s] if not is_var(s) and s in mid2vid else binding.get(s)
            ts = mid2vid[o] if not is_var(o) and o in mid2vid else binding.get(o)
            if hs is None or ts is None:
                continue
            if store.has_edge(hs, rid, ts):
                h_mid = s if not is_var(s) else vid2mid.get(hs, f"vid:{hs}")
                t_mid = o if not is_var(o) else vid2mid.get(ts, f"vid:{ts}")
                add_edge(h_mid, rname, t_mid)
        if vertex_set:
            witnesses.append(sorted(vertex_set))

    # More witness samples (limited)
    if variables and all(assignment.get(v) for v in variables):
        # take up to max_bindings product samples along first var
        first_var = next(iter(variables))
        for alt in list(assignment[first_var])[1 : max_bindings]:
            local = dict(binding)
            local[first_var] = alt
            verts: Set[str] = set()
            ok = True
            local_edges: Set[Tuple[str, str, str]] = set()
            for s, rid, o, rname in patterns:
                hs = mid2vid[s] if not is_var(s) and s in mid2vid else local.get(s)
                ts = mid2vid[o] if not is_var(o) and o in mid2vid else local.get(o)
                if hs is None or ts is None or not store.has_edge(hs, rid, ts):
                    ok = False
                    break
                h_mid = s if not is_var(s) else vid2mid.get(hs, f"vid:{hs}")
                t_mid = o if not is_var(o) else vid2mid.get(ts, f"vid:{ts}")
                verts.update([h_mid, t_mid])
                local_edges.add((h_mid, rname, t_mid))
            if ok and verts:
                witnesses.append(sorted(verts))
                edge_set.update(local_edges)
                vertex_set.update(verts)
            if len(witnesses) >= 20:
                break

    success = len(edge_set) > 0 or len(vertex_set) > 0
    # partial: some vars unbound or some patterns unused
    unbound = [v for v in variables if v not in binding]
    partial = success and (len(unbound) > 0 or coverage["entity_missing"] > 0)
    if not success:
        coverage["empty_result"] = 1

    return {
        "success": success,
        "partial": partial,
        "vertices": sorted(vertex_set),
        "edges": [list(e) for e in sorted(edge_set)],
        "witnesses": witnesses[:50],
        "coverage": coverage,
        "reason": None if success else "empty_result",
    }


def run_cwq_coverage(
    train_json: Path,
    processed_graph_dir: Path,
    reverse_properties_path: Path,
    out_dir: Path,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> dict:
    mid2vid, relation2id, _ = load_maps(processed_graph_dir)
    reverse_map = load_reverse_properties(reverse_properties_path)
    db = db_path or (processed_graph_dir / "fb_cvt_rev.sqlite")
    store = FreebaseSQLiteStore(db)

    data = json.loads(train_json.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "query_traces.jsonl"
    failed_path = out_dir / "failed_queries.jsonl"
    norm_path = out_dir / "normalized_queries.jsonl"

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

    with traces_path.open("w", encoding="utf-8") as ft, failed_path.open(
        "w", encoding="utf-8"
    ) as ff, norm_path.open("w", encoding="utf-8") as fn:
        for i, sample in enumerate(data):
            sparql = sample.get("sparql") or ""
            result = ground_cwq_query(sparql, store, mid2vid, relation2id, reverse_map)
            record = {
                "qid": sample.get("ID", i),
                "question": sample.get("question"),
                "compositionality_type": sample.get("compositionality_type"),
                **result,
            }
            ft.write(json.dumps(record, ensure_ascii=False) + "\n")

            canon = canonicalize_query_triples(sparql, relation2id, reverse_map)
            fn.write(
                json.dumps(
                    {
                        "qid": sample.get("ID", i),
                        "triples": canon["triples"],
                        "stats": canon["stats"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            c = result["coverage"]
            coverage["entity_missing"] += int(c.get("entity_missing", 0) > 0)
            coverage["relation_missing"] += int(c.get("relation_missing", 0) > 0)
            coverage["reverse_relation_rewritten"] += int(c.get("reverse_relation_rewritten", 0) > 0)
            coverage["metadata_relation_resolved"] += int(c.get("metadata_relation_resolved", 0) > 0)
            coverage["empty_result"] += int(c.get("empty_result", 0) > 0)
            coverage["unsupported_sparql"] += int(c.get("unsupported_sparql", 0) > 0)

            if result["success"] and not result["partial"]:
                coverage["success"] += 1
            elif result["success"] and result["partial"]:
                coverage["partial"] += 1
                coverage["success"] += 1  # still executable partial
            else:
                coverage["failed"] += 1
                ff.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (i + 1) % 500 == 0:
                print(f"  CWQ grounding {i+1}/{len(data)} ...", flush=True)

    store.close()
    (out_dir / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return coverage
