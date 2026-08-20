"""Ground SPARQL-like patterns against a local adjacency index."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from src.workload.sparql_utils import (
    extract_triple_patterns,
    extract_values_bindings,
    is_var,
    normalize_term,
)

Adj = Dict[str, Dict[str, Set[str]]]
Binding = Dict[str, str]
Edge = Tuple[str, str, str]


def _fallback_from_constants(
    patterns: List[Tuple[str, str, str]],
    answer_mids: Optional[Set[str]],
) -> dict:
    verts: Set[str] = set(answer_mids or [])
    edges: Set[Edge] = set()
    for s, p, o in patterns:
        s, p, o = normalize_term(s), normalize_term(p), normalize_term(o)
        if not is_var(s) and not is_var(o):
            edges.add((s, p, o))
            verts.update([s, o])
        else:
            if not is_var(s):
                verts.add(s)
            if not is_var(o):
                verts.add(o)
    return {
        "success": len(verts) > 0 or len(edges) > 0,
        "partial": True,
        "vertices": sorted(verts),
        "edges": [list(e) for e in sorted(edges)],
        "witnesses": [sorted(verts)] if len(verts) >= 2 else [],
    }


def ground_patterns(
    patterns: List[Tuple[str, str, str]],
    adj: Adj,
    seed_bindings: Optional[Binding] = None,
    answer_mids: Optional[Set[str]] = None,
    max_bindings: int = 128,
    radj: Optional[Adj] = None,
) -> List[Tuple[Set[str], Set[Edge]]]:
    """Join only from already-bound / constant endpoints. Never enumerate full V."""
    seed_bindings = dict(seed_bindings or {})
    radj = radj or {}
    if not patterns:
        verts = set(seed_bindings.values())
        if answer_mids:
            verts |= set(answer_mids)
        return [(verts, set())] if verts else []

    def score(pat):
        s, p, o = pat
        return (0 if is_var(s) else 1) + (0 if is_var(o) else 1)

    ordered = sorted(patterns, key=score, reverse=True)
    bindings: List[Binding] = [seed_bindings]
    matched_edges: List[Set[Edge]] = [set()]

    for s_raw, p_raw, o_raw in ordered:
        s, p, o = normalize_term(s_raw), normalize_term(p_raw), normalize_term(o_raw)
        new_b: List[Binding] = []
        new_e: List[Set[Edge]] = []
        for b, edges in zip(bindings, matched_edges):
            s_bound = (not is_var(s)) or (s in b)
            o_bound = (not is_var(o)) or (o in b)
            if not s_bound and not o_bound:
                new_b.append(b)
                new_e.append(edges)
                continue

            if s_bound:
                heads = [b[s]] if is_var(s) else [s]
                for hs in heads:
                    tails = adj.get(hs, {}).get(p, set())
                    if o_bound:
                        target = b[o] if is_var(o) else o
                        if target in tails:
                            nb = dict(b)
                            ne = set(edges)
                            ne.add((hs, p, target))
                            new_b.append(nb)
                            new_e.append(ne)
                    else:
                        for ts in tails:
                            nb = dict(b)
                            nb[o] = ts
                            ne = set(edges)
                            ne.add((hs, p, ts))
                            new_b.append(nb)
                            new_e.append(ne)
                            if len(new_b) >= max_bindings:
                                break
            else:
                target = b[o] if is_var(o) else o
                heads = radj.get(target, {}).get(p, set())
                for hs in heads:
                    nb = dict(b)
                    nb[s] = hs
                    ne = set(edges)
                    ne.add((hs, p, target))
                    new_b.append(nb)
                    new_e.append(ne)
                    if len(new_b) >= max_bindings:
                        break

            if len(new_b) >= max_bindings:
                break

        if new_b:
            bindings, matched_edges = new_b[:max_bindings], new_e[:max_bindings]

    results = []
    for b, edges in zip(bindings, matched_edges):
        verts = set(b.values())
        for h, _r, t in edges:
            verts.add(h)
            verts.add(t)
        if answer_mids:
            verts |= set(answer_mids)
        results.append((verts, edges))
    return results


def ground_sparql(
    sparql: str,
    adj: Adj,
    answer_mids: Optional[Set[str]] = None,
    radj: Optional[Adj] = None,
) -> dict:
    patterns = extract_triple_patterns(sparql)
    values = extract_values_bindings(sparql)
    seed: Binding = {}
    for var, mids in values.items():
        if len(mids) == 1:
            seed[var] = mids[0]

    grounded = ground_patterns(
        patterns, adj, seed_bindings=seed, answer_mids=answer_mids, radj=radj
    )
    if not grounded or (
        len(grounded) == 1
        and not grounded[0][1]
        and len(grounded[0][0]) <= len(answer_mids or [])
    ):
        fb = _fallback_from_constants(patterns, answer_mids)
        if grounded:
            fb_verts = set(fb["vertices"]) | grounded[0][0]
            fb["vertices"] = sorted(fb_verts)
            if len(fb_verts) >= 2 and not fb["witnesses"]:
                fb["witnesses"] = [sorted(fb_verts)]
        return fb

    verts: Set[str] = set()
    edges: Set[Edge] = set()
    witnesses: List[List[str]] = []
    for vset, eset in grounded:
        verts |= vset
        edges |= eset
        if len(vset) >= 2:
            witnesses.append(sorted(vset))

    for s, p, o in patterns:
        s, p, o = normalize_term(s), normalize_term(p), normalize_term(o)
        if not is_var(s) and not is_var(o):
            edges.add((s, p, o))
            verts.update([s, o])

    uniq, seen = [], set()
    for w in witnesses:
        key = "|".join(w)
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    if not uniq and len(verts) >= 2:
        uniq = [sorted(verts)]

    return {
        "success": True,
        "partial": len(edges) == 0,
        "vertices": sorted(verts),
        "edges": [list(e) for e in sorted(edges)],
        "witnesses": uniq[:32],
    }
