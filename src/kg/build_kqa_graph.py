"""Build vertices.tsv and edges.tsv from KQA Pro kb.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


EdgeTriple = Tuple[str, str, str]


def load_kb(kb_path: Path) -> dict:
    return json.loads(kb_path.read_text(encoding="utf-8"))


def iter_kqa_edges(kb: dict) -> Iterable[EdgeTriple]:
    for entity_id, entity in kb["entities"].items():
        for concept_id in entity.get("instanceOf", []):
            yield entity_id, "instanceOf", concept_id
        for rel in entity.get("relations", []):
            predicate = rel["predicate"]
            obj = rel["object"]
            direction = rel.get("direction", "forward")
            if direction == "forward":
                yield entity_id, predicate, obj
            else:
                yield obj, predicate, entity_id

    for concept_id, concept in kb.get("concepts", {}).items():
        for rel in concept.get("relations", []):
            predicate = rel["predicate"]
            obj = rel["object"]
            direction = rel.get("direction", "forward")
            if direction == "forward":
                yield concept_id, predicate, obj
            else:
                yield obj, predicate, concept_id


def build_vertex_map(kb: dict) -> Dict[str, int]:
    vertex_ids: Dict[str, int] = {}
    next_id = 1
    for concept_id in kb["concepts"]:
        vertex_ids[concept_id] = next_id
        next_id += 1
    for entity_id in kb["entities"]:
        if entity_id not in vertex_ids:
            vertex_ids[entity_id] = next_id
            next_id += 1
    return vertex_ids


def write_vertices(out_path: Path, vertex_ids: Dict[str, int]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("vid\toriginal_id\n")
        for original_id, vid in sorted(vertex_ids.items(), key=lambda item: item[1]):
            f.write(f"{vid}\t{original_id}\n")


def write_edges(out_path: Path, vertex_ids: Dict[str, int], edges: Iterable[EdgeTriple]) -> Dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_ids: Dict[str, int] = {}
    next_eid = 1
    with out_path.open("w", encoding="utf-8") as f:
        f.write("eid\thead\trelation\ttail\n")
        seen = set()
        for head, relation, tail in edges:
            if head not in vertex_ids or tail not in vertex_ids:
                continue
            key = f"{head}\t{relation}\t{tail}"
            if key in seen:
                continue
            seen.add(key)
            edge_ids[key] = next_eid
            f.write(
                f"{next_eid}\t{vertex_ids[head]}\t{relation}\t{vertex_ids[tail]}\n"
            )
            next_eid += 1
    return edge_ids


def build_kqa_graph(kb_path: Path, graph_dir: Path) -> dict:
    kb = load_kb(kb_path)
    vertex_ids = build_vertex_map(kb)
    write_vertices(graph_dir / "vertices.tsv", vertex_ids)
    edge_ids = write_edges(graph_dir / "edges.tsv", vertex_ids, iter_kqa_edges(kb))
    return {
        "num_vertices": len(vertex_ids),
        "num_edges": len(edge_ids),
        "vertex_ids": vertex_ids,
        "edge_ids": edge_ids,
    }
