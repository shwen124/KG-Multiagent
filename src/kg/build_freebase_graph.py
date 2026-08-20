"""Build shared Simplified Freebase from gold SPARQL / graph_query constants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

from src.workload.sparql_utils import (
    extract_triple_patterns,
    grounded_constant_triples,
    is_entity_mid,
    is_var,
    normalize_term,
)

Edge = Tuple[str, str, str]


def _add_edge(edges: Set[Edge], h: str, r: str, t: str) -> None:
    if not h or not r or not t:
        return
    if is_var(h) or is_var(r) or is_var(t):
        return
    edges.add((h, r, t))


def collect_from_webqsp(train_json: Path, edges: Set[Edge], vertices: Set[str]) -> None:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    questions = data["Questions"] if isinstance(data, dict) else data
    for q in questions:
        for parse in q.get("Parses", []):
            topic = parse.get("TopicEntityMid")
            if topic:
                vertices.add(normalize_term(topic))
            for ans in parse.get("Answers") or []:
                arg = ans.get("AnswerArgument")
                if arg and is_entity_mid(arg):
                    vertices.add(normalize_term(arg))
            for c in parse.get("Constraints") or []:
                arg = c.get("Argument")
                if arg and is_entity_mid(str(arg)):
                    vertices.add(normalize_term(str(arg)))
            sparql = parse.get("Sparql") or ""
            if sparql:
                for s, p, o in grounded_constant_triples(sparql):
                    _add_edge(edges, s, p, o)
                    vertices.update([s, o])
                for s, p, o in extract_triple_patterns(sparql):
                    for term in (s, o):
                        if not is_var(term):
                            vertices.add(term)


def collect_from_cwq(train_json: Path, edges: Set[Edge], vertices: Set[str]) -> None:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    for q in data:
        sparql = q.get("sparql") or ""
        if sparql:
            for s, p, o in grounded_constant_triples(sparql):
                _add_edge(edges, s, p, o)
                vertices.update([s, o])
            for s, p, o in extract_triple_patterns(sparql):
                for term in (s, o):
                    if not is_var(term):
                        vertices.add(term)
        for ans in q.get("answers") or []:
            aid = ans.get("answer_id")
            if aid and is_entity_mid(str(aid)):
                vertices.add(normalize_term(str(aid)))


def collect_from_grailqa(train_json: Path, edges: Set[Edge], vertices: Set[str]) -> None:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    for q in data:
        gq = q.get("graph_query") or {}
        nodes = {n["nid"]: n for n in gq.get("nodes") or []}
        for n in nodes.values():
            nid = normalize_term(str(n.get("id", "")))
            if n.get("node_type") == "entity" and is_entity_mid(nid):
                vertices.add(nid)
            elif n.get("node_type") == "class":
                vertices.add(nid)
        for e in gq.get("edges") or []:
            sn, en = nodes.get(e["start"]), nodes.get(e["end"])
            if not sn or not en:
                continue
            s = normalize_term(str(sn["id"]))
            t = normalize_term(str(en["id"]))
            r = normalize_term(str(e["relation"]))
            _add_edge(edges, s, r, t)
            vertices.update([s, t])
        sparql = q.get("sparql_query") or ""
        if sparql:
            for s, p, o in grounded_constant_triples(sparql):
                _add_edge(edges, s, p, o)
                vertices.update([s, o])
        for ans in q.get("answer") or []:
            arg = ans.get("answer_argument")
            if arg and is_entity_mid(str(arg)):
                vertices.add(normalize_term(str(arg)))


def write_graph(vertices: Set[str], edges: Set[Edge], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    # ensure edge endpoints exist
    for h, _r, t in edges:
        vertices.add(h)
        vertices.add(t)
    vertex_ids: Dict[str, int] = {}
    for i, oid in enumerate(sorted(vertices), start=1):
        vertex_ids[oid] = i

    with (out_dir / "vertices.tsv").open("w", encoding="utf-8") as f:
        f.write("vid\toriginal_id\n")
        for oid, vid in sorted(vertex_ids.items(), key=lambda x: x[1]):
            f.write(f"{vid}\t{oid}\n")

    with (out_dir / "edges.tsv").open("w", encoding="utf-8") as f:
        f.write("eid\thead\trelation\ttail\n")
        for eid, (h, r, t) in enumerate(sorted(edges), start=1):
            f.write(f"{eid}\t{vertex_ids[h]}\t{r}\t{vertex_ids[t]}\n")

    return {"num_vertices": len(vertex_ids), "num_edges": len(edges), "vertex_ids": vertex_ids}


def build_shared_freebase(
    webqsp_train: Path,
    cwq_train: Path,
    grailqa_train: Path,
    out_dir: Path,
) -> dict:
    edges: Set[Edge] = set()
    vertices: Set[str] = set()
    collect_from_webqsp(webqsp_train, edges, vertices)
    collect_from_cwq(cwq_train, edges, vertices)
    collect_from_grailqa(grailqa_train, edges, vertices)
    stats = write_graph(vertices, edges, out_dir)
    meta = {
        "source": "gold SPARQL / graph_query grounded constants from CWQ+WebQSP+GrailQA train",
        "note": (
            "GraftNet freebase_prepro.tgz unavailable (502); Freebase Easy download optional. "
            "This is a dataset-derived Simplified Freebase covering gold structural facts."
        ),
        "num_vertices": stats["num_vertices"],
        "num_edges": stats["num_edges"],
    }
    (out_dir / "graph_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**meta, "vertex_ids": stats["vertex_ids"]}


def load_adjacency(graph_dir: Path):
    from collections import defaultdict

    original = {}
    with (graph_dir / "vertices.tsv").open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            vid, oid = line.rstrip("\n").split("\t")
            original[int(vid)] = oid
    adj = defaultdict(lambda: defaultdict(set))
    radj = defaultdict(lambda: defaultdict(set))  # radj[tail][rel] = {heads}
    with (graph_dir / "edges.tsv").open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            _eid, h, r, t = line.rstrip("\n").split("\t")
            hs, ts = original[int(h)], original[int(t)]
            adj[hs][r].add(ts)
            radj[ts][r].add(hs)
    return adj, radj
