"""Generic hypergraph builder for any dataset following the unified format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from src.common.weights import edge_key, task_hyperedge_weight


def load_vertices(vertices_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    original_to_vid: Dict[str, int] = {}
    vid_to_original: Dict[int, str] = {}
    with vertices_path.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            vid_str, original_id = line.rstrip("\n").split("\t")
            vid = int(vid_str)
            original_to_vid[original_id] = vid
            vid_to_original[vid] = original_id
    return original_to_vid, vid_to_original


def load_edge_weights(path: Path) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    if not path.exists():
        return weights
    with path.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            head, relation, tail, _freq, weight = line.rstrip("\n").split("\t")
            weights[edge_key(head, relation, tail)] = int(weight)
    return weights


def load_vertex_weights(path: Path) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    if not path.exists():
        return weights
    with path.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            original_id, _freq, weight = line.rstrip("\n").split("\t")
            weights[original_id] = int(weight)
    return weights


def load_witnesses(trace_path: Path) -> List[List[str]]:
    witnesses: List[List[str]] = []
    seen: Set[str] = set()
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not record.get("success"):
                continue
            for witness in record.get("witnesses", []):
                key = "|".join(witness)
                if key in seen:
                    continue
                seen.add(key)
                witnesses.append(witness)
    return witnesses


def build_hypergraph(
    graph_dir: Path,
    weights_dir: Path,
    workload_dir: Path,
    hypergraph_dir: Path,
    hgr_name: str = "graph.hgr",
) -> dict:
    original_to_vid, vid_to_original = load_vertices(graph_dir / "vertices.tsv")
    vertex_weights = load_vertex_weights(weights_dir / "vertex_weights.tsv")
    edge_weights = load_edge_weights(weights_dir / "structural_edge_weights.tsv")
    task_weight = task_hyperedge_weight()

    hyperedges: List[Tuple[int, List[int]]] = []
    hyperedge_rows: List[str] = []

    with (graph_dir / "edges.tsv").open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            _eid, head_vid, relation, tail_vid = line.rstrip("\n").split("\t")
            head_vid_i = int(head_vid)
            tail_vid_i = int(tail_vid)
            original_head = vid_to_original[head_vid_i]
            original_tail = vid_to_original[tail_vid_i]
            weight = edge_weights.get(edge_key(original_head, relation, original_tail), 1)
            hyperedges.append((weight, [head_vid_i, tail_vid_i]))
            hyperedge_rows.append(
                f"structural\t{head_vid_i},{tail_vid_i}\t{relation}\t{weight}"
            )

    witnesses = load_witnesses(workload_dir / "query_traces.jsonl")
    task_count = 0
    for witness in witnesses:
        pins = sorted({original_to_vid[v] for v in witness if v in original_to_vid})
        if len(pins) < 2:
            continue
        hyperedges.append((task_weight, pins))
        hyperedge_rows.append(
            f"task\t{','.join(str(v) for v in pins)}\twitness\t{task_weight}"
        )
        task_count += 1

    hypergraph_dir.mkdir(parents=True, exist_ok=True)
    with (hypergraph_dir / "hyperedges.tsv").open("w", encoding="utf-8") as f:
        f.write("type\tpins\tmeta\tweight\n")
        for row in hyperedge_rows:
            f.write(row + "\n")

    num_vertices = len(original_to_vid)
    num_hyperedges = len(hyperedges)
    hgr_path = hypergraph_dir / hgr_name
    with hgr_path.open("w", encoding="utf-8") as f:
        f.write(f"{num_vertices} {num_hyperedges} 3\n")
        for vid in range(1, num_vertices + 1):
            original_id = vid_to_original[vid]
            f.write(f"{vertex_weights.get(original_id, 1)}\n")
        for weight, pins in hyperedges:
            f.write(f"{weight} {len(pins)} {' '.join(str(p) for p in pins)}\n")

    return {
        "num_vertices": num_vertices,
        "num_hyperedges": num_hyperedges,
        "num_task_hyperedges": task_count,
        "hgr_path": str(hgr_path),
    }
