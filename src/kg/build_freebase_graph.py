"""Build a shared Freebase KG from RoG subgraphs + WebQSP structural hints.

This produces a practical Simplified Freebase covering CWQ/WebQSP workloads
when the full GraftNet Freebase dump is unavailable.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pyarrow.parquet as pq

Edge = Tuple[str, str, str]


def _norm_mid(x: str) -> str:
    x = x.strip()
    if x.startswith("ns:"):
        x = x[3:]
    x = x.replace("http://rdf.freebase.com/ns/", "")
    if x.startswith("m.") or x.startswith("g."):
        return x.replace(".", "/", 1) if x[1] == "." and "/" not in x else x
    if x.startswith("/m/") or x.startswith("/g/"):
        return x
    if re.fullmatch(r"m\.[0-9a-z_]+", x) or re.fullmatch(r"g\.[0-9a-z_]+", x):
        return "/" + x.replace(".", "/", 1)
    return x


def _norm_relation(r: str) -> str:
    r = r.strip()
    if r.startswith("ns:"):
        r = r[3:]
    r = r.replace("http://rdf.freebase.com/ns/", "")
    if not r.startswith("/") and "." in r:
        r = "/" + r.replace(".", "/")
    return r


def iter_rog_parquet_edges(parquet_dir: Path) -> Iterable[Edge]:
    for path in sorted(parquet_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["graph"])
        for graph in table.column("graph").to_pylist():
            if not graph:
                continue
            for triple in graph:
                if not triple or len(triple) < 3:
                    continue
                h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
                yield h, _norm_relation(r), t


def extract_webqsp_edges(train_json: Path) -> Iterable[Edge]:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    questions = data["Questions"] if isinstance(data, dict) else data
    for q in questions:
        for parse in q.get("Parses", []):
            topic = parse.get("TopicEntityMid")
            chain = parse.get("InferentialChain") or []
            if not topic or not chain:
                continue
            # InferentialChain alone is not enough for concrete edges without CVT
            # binding; keep topic as vertex seed via self-loop marker skipped later.
            _ = (topic, chain)


def write_graph(edges: Set[Edge], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices: Dict[str, int] = {}
    next_vid = 1
    for h, _r, t in edges:
        if h not in vertices:
            vertices[h] = next_vid
            next_vid += 1
        if t not in vertices:
            vertices[t] = next_vid
            next_vid += 1

    with (out_dir / "vertices.tsv").open("w", encoding="utf-8") as f:
        f.write("vid\toriginal_id\n")
        for oid, vid in sorted(vertices.items(), key=lambda x: x[1]):
            f.write(f"{vid}\t{oid}\n")

    with (out_dir / "edges.tsv").open("w", encoding="utf-8") as f:
        f.write("eid\thead\trelation\ttail\n")
        for eid, (h, r, t) in enumerate(sorted(edges), start=1):
            f.write(f"{eid}\t{vertices[h]}\t{r}\t{vertices[t]}\n")

    return {"num_vertices": len(vertices), "num_edges": len(edges), "vertex_ids": vertices}


def build_freebase_from_rog(
    rog_cwq_dir: Path,
    rog_webqsp_dir: Path,
    out_dir: Path,
) -> dict:
    edges: Set[Edge] = set()
    for src in (rog_cwq_dir, rog_webqsp_dir):
        if src.exists():
            for e in iter_rog_parquet_edges(src):
                edges.add(e)
    stats = write_graph(edges, out_dir)
    meta = {
        "source": "union(RoG-CWQ, RoG-WebQSP)",
        "note": "Practical Simplified Freebase for CWQ/WebQSP when GraftNet dump unavailable.",
        **{k: stats[k] for k in ("num_vertices", "num_edges")},
    }
    (out_dir / "graph_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {**meta, "vertex_ids": stats["vertex_ids"]}


def load_adjacency(graph_dir: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Return adj[head][relation] = {tails} using original ids."""
    original = {}
    with (graph_dir / "vertices.tsv").open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            vid, oid = line.rstrip("\n").split("\t")
            original[int(vid)] = oid
    adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    with (graph_dir / "edges.tsv").open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            _eid, h, r, t = line.rstrip("\n").split("\t")
            adj[original[int(h)]][r].add(original[int(t)])
    return adj
