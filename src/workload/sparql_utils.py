"""SPARQL structural triple-pattern utilities for Freebase workloads."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

TriplePat = Tuple[str, str, str]  # may contain ?vars


_PREFIX_NS = re.compile(r"PREFIX\s+\w+:\s*<[^>]+>", re.I)
_MID = re.compile(r"^(?:ns:)?(?:m|g)\.[0-9a-zA-Z_]+$")
_VAR = re.compile(r"^\?\w+$")


def normalize_term(term: str) -> str:
    term = term.strip().rstrip(".")
    if term.startswith("<") and term.endswith(">"):
        term = term[1:-1]
    term = term.replace("http://rdf.freebase.com/ns/", "")
    if term.startswith(":"):
        term = term[1:]
    if term.startswith("ns:"):
        term = term[3:]
    # m.xxx / g.xxx stay as mid-like ids with dot form for consistency
    if term.startswith("m/") or term.startswith("g/"):
        term = term[0] + "." + term[2:]
    if "/" in term and not term.startswith("?"):
        # people.person.gender already dotted; education/school -> dotted
        if term.count("/") >= 1 and not term.startswith("http"):
            # type.object.type style may use /
            pass
    return term


def is_var(term: str) -> bool:
    return bool(_VAR.match(term))


def is_entity_mid(term: str) -> bool:
    t = normalize_term(term)
    return bool(re.match(r"^(?:m|g)\.[0-9a-zA-Z_]+$", t))


def extract_where_body(sparql: str) -> str:
    text = _PREFIX_NS.sub("", sparql)
    m = re.search(r"WHERE\s*\{", text, re.I)
    if not m:
        return text
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1]


def extract_triple_patterns(sparql: str) -> List[TriplePat]:
    """Extract structural triple patterns; ignore FILTER/OPTIONAL/VALUES-only lines."""
    body = extract_where_body(sparql)
    # drop FILTER blocks roughly
    body = re.sub(r"FILTER\s*\((?:[^()]|\([^()]*\))*\)", " ", body, flags=re.I)
    body = re.sub(r"OPTIONAL\s*\{[^{}]*\}", " ", body, flags=re.I)
    # VALUES ?x { :m.xx } -> bind later separately
    triples: List[TriplePat] = []
    # token-based scan for s p o .
    # replace ; and , chains poorly; handle simple whitespace triples
    cleaned = body.replace("\n", " ")
    # expand VALUES into fake type triples skipped; capture VALUES mids separately
    for m in re.finditer(
        r"(ns:[^\s]+|:[^\s]+|\?[A-Za-z0-9_]+)\s+(ns:[^\s]+|:[^\s]+|\?[A-Za-z0-9_]+)\s+(ns:[^\s]+|:[^\s]+|\?[A-Za-z0-9_]+)\s*\.",
        cleaned,
    ):
        s, p, o = normalize_term(m.group(1)), normalize_term(m.group(2)), normalize_term(m.group(3))
        # skip rdf/rdfs meta if needed but keep type.object.type
        triples.append((s, p, o))
    return triples


def extract_values_bindings(sparql: str) -> dict:
    """Parse simple VALUES ?x { :m.a :m.b }."""
    bindings = {}
    for m in re.finditer(
        r"VALUES\s+(\?\w+)\s*\{([^}]+)\}", sparql, flags=re.I
    ):
        var = m.group(1)
        mids = [normalize_term(tok) for tok in m.group(2).split() if tok.strip()]
        bindings[var] = mids
    return bindings


def grounded_constant_triples(sparql: str) -> List[Tuple[str, str, str]]:
    out = []
    for s, p, o in extract_triple_patterns(sparql):
        if not is_var(s) and not is_var(o) and not is_var(p):
            out.append((s, p, o))
    return out
