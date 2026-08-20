"""Shared helpers to write workload traces, weights from train-only frequencies."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from src.common.weights import assign_edge_weights, assign_vertex_weights, edge_key


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def accumulate_and_write(
    traces: List[dict],
    output_dir: Path,
    train_meta_rows: List[dict],
) -> dict:
    vertex_freq: Counter[str] = Counter()
    edge_freq: Counter[str] = Counter()
    witness_freq: Counter[str] = Counter()
    success = 0
    partial = 0

    for record in traces:
        if record.get("success"):
            success += 1
            if record.get("partial"):
                partial += 1
            for v in set(record.get("vertices") or []):
                vertex_freq[v] += 1
            for e in {tuple(x) for x in (record.get("edges") or [])}:
                edge_freq[edge_key(*e)] += 1
            for w in record.get("witnesses") or []:
                witness_freq["|".join(w)] += 1

    workload_dir = output_dir / "workload"
    weights_dir = output_dir / "weights"
    write_jsonl(workload_dir / "train_queries.jsonl", train_meta_rows)
    write_jsonl(workload_dir / "query_traces.jsonl", traces)

    vw = assign_vertex_weights(dict(vertex_freq))
    ew = assign_edge_weights(dict(edge_freq))
    weights_dir.mkdir(parents=True, exist_ok=True)
    with (weights_dir / "vertex_weights.tsv").open("w", encoding="utf-8") as f:
        f.write("original_id\tfrequency\tweight\n")
        for oid, freq in sorted(vertex_freq.items()):
            f.write(f"{oid}\t{freq}\t{vw[oid]}\n")
    with (weights_dir / "structural_edge_weights.tsv").open("w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\tfrequency\tweight\n")
        for ek, freq in sorted(edge_freq.items()):
            h, r, t = ek.split("\t")
            f.write(f"{h}\t{r}\t{t}\t{freq}\t{ew[ek]}\n")

    stats = {
        "total_train_queries": len(traces),
        "executable_queries": success,
        "partial_queries": partial,
        "non_executable_queries": len(traces) - success,
        "unique_vertices_in_workload": len(vertex_freq),
        "unique_edges_in_workload": len(edge_freq),
        "unique_witnesses": len(witness_freq),
    }
    (output_dir / "workload_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats
