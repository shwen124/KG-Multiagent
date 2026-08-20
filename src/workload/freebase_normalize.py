"""Freebase namespace + reverse-relation normalization for CWQ SPARQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


NS_TOKEN_RE = re.compile(r"\bns:([A-Za-z0-9_./\-]+)")
TRIPLE_RE = re.compile(
    r"(ns:[A-Za-z0-9_./\-]+|\?[A-Za-z0-9_]+)\s+"
    r"(ns:[A-Za-z0-9_./\-]+)\s+"
    r"(ns:[A-Za-z0-9_./\-]+|\?[A-Za-z0-9_]+)\s*\."
)

METADATA_PREFIXES = (
    "/type/",
    "/common/",
    "/kg/",
    "/base/",
    "/freebase/",
    "/dataworld/",
    "/user/",
    "/pipeline/",
    "/atom/",
    "/topic_server/",
)


def ns_token_to_mid_or_rel(token: str) -> str:
    """ns:m.012abc -> /m/012abc ; ns:people.person.place_of_birth -> /people/person/place_of_birth"""
    if token.startswith("ns:"):
        token = token[3:]
    if token.startswith("m.") or token.startswith("g.") or token.startswith("en."):
        # entity MID: only first dot becomes /
        return "/" + token.replace(".", "/", 1) if not token.startswith("/") else token
    # relation / type path: all dots -> /
    if not token.startswith("/"):
        return "/" + token.replace(".", "/")
    return token


def normalize_ns_token(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("?"):
        return raw
    if raw.startswith("ns:"):
        return ns_token_to_mid_or_rel(raw)
    if raw.startswith("http://rdf.freebase.com/ns/"):
        rest = raw[len("http://rdf.freebase.com/ns/") :]
        return ns_token_to_mid_or_rel("ns:" + rest)
    if raw.startswith("/"):
        return raw
    return raw


def is_metadata_relation(rel: str) -> bool:
    return any(rel.startswith(p) for p in METADATA_PREFIXES)


def load_reverse_properties(path: Path) -> Dict[str, str]:
    """Load bidirectional reverse map. File lines: rel_a\\trel_b (often without leading /)."""
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]

            def canon(x: str) -> str:
                if not x.startswith("/"):
                    x = "/" + x.replace(".", "/")
                else:
                    # already path-like
                    pass
                # if still dotted without slashes after first char
                if x.count("/") == 1 and "." in x:
                    x = "/" + x.lstrip("/").replace(".", "/")
                return x

            a, b = canon(a), canon(b)
            mapping[a] = b
            mapping[b] = a
    return mapping


def extract_structural_triples(sparql: str) -> List[Tuple[str, str, str]]:
    """Extract (s,p,o) after ns: normalization. Skips FILTER-only content."""
    triples = []
    for m in TRIPLE_RE.finditer(sparql):
        s, p, o = m.group(1), m.group(2), m.group(3)
        s_n = normalize_ns_token(s)
        p_n = normalize_ns_token(p)
        o_n = normalize_ns_token(o)
        triples.append((s_n, p_n, o_n))
    return triples


CanonicalTriple = Tuple[str, str, str, str]
# (s, r, o, status) status in {ok, reverse_rewritten, metadata_skipped, relation_missing}


def canonicalize_triple(
    s: str,
    r: str,
    o: str,
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> CanonicalTriple:
    if is_metadata_relation(r):
        return s, r, o, "metadata_skipped"

    if r in relation2id:
        return s, r, o, "ok"

    rev = reverse_map.get(r)
    if rev and rev in relation2id:
        # (s, r_rev, o) ≡ (o, r_canon, s)
        return o, rev, s, "reverse_rewritten"

    return s, r, o, "relation_missing"


def canonicalize_query_triples(
    sparql: str,
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> dict:
    raw = extract_structural_triples(sparql)
    canonical = []
    stats = {
        "raw_triples": len(raw),
        "ok": 0,
        "reverse_rewritten": 0,
        "metadata_skipped": 0,
        "relation_missing": 0,
    }
    for s, r, o in raw:
        cs, cr, co, status = canonicalize_triple(s, r, o, relation2id, reverse_map)
        stats[status] = stats.get(status, 0) + 1
        if status in {"ok", "reverse_rewritten"}:
            canonical.append({"s": cs, "r": cr, "o": co, "status": status})
    return {"triples": canonical, "stats": stats}
