"""Generate query traces and frequency statistics from KQA Pro train split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from src.common.weights import assign_edge_weights, assign_vertex_weights, edge_key
from src.workload.kopl_trace_executor import TraceRuleExecutor


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_train_queries(train_path: Path) -> List[dict]:
    return json.loads(train_path.read_text(encoding="utf-8"))


def generate_query_traces(
    train_path: Path,
    kb_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict:
    executor = TraceRuleExecutor(kb_path)
    train_data = load_train_queries(train_path)
    if limit is not None:
        train_data = train_data[:limit]

    traces = []
    vertex_freq: Counter[str] = Counter()
    edge_freq: Counter[str] = Counter()
    witness_freq: Counter[str] = Counter()

    success_count = 0
    for qid, sample in enumerate(train_data):
        trace = executor.execute_with_trace(sample["program"], gold_answer=sample.get("answer"))
        record = {
            "qid": qid,
            "question": sample["question"],
            "success": trace["success"],
            "vertices": trace["vertices"],
            "edges": trace["edges"],
            "witnesses": trace["witnesses"],
        }
        if not trace["success"]:
            record["error"] = trace.get("error")
            traces.append(record)
            continue

        success_count += 1
        for vertex in set(trace["vertices"]):
            vertex_freq[vertex] += 1
        for edge in {tuple(edge) for edge in trace["edges"]}:
            edge_freq[edge_key(*edge)] += 1
        for witness in trace["witnesses"]:
            witness_key = "|".join(witness)
            witness_freq[witness_key] += 1
        traces.append(record)

    workload_dir = output_dir / "workload"
    weights_dir = output_dir / "weights"
    write_jsonl(workload_dir / "train_queries.jsonl", train_data[: len(traces)])
    write_jsonl(workload_dir / "query_traces.jsonl", traces)

    vertex_weights = assign_vertex_weights(dict(vertex_freq))
    edge_weights = assign_edge_weights(dict(edge_freq))

    weights_dir.mkdir(parents=True, exist_ok=True)
    with (weights_dir / "vertex_weights.tsv").open("w", encoding="utf-8") as f:
        f.write("original_id\tfrequency\tweight\n")
        for original_id, freq in sorted(vertex_freq.items()):
            f.write(f"{original_id}\t{freq}\t{vertex_weights[original_id]}\n")

    with (weights_dir / "structural_edge_weights.tsv").open("w", encoding="utf-8") as f:
        f.write("head\trelation\ttail\tfrequency\tweight\n")
        for edge, freq in sorted(edge_freq.items()):
            head, relation, tail = edge.split("\t")
            f.write(f"{head}\t{relation}\t{tail}\t{freq}\t{edge_weights[edge]}\n")

    stats = {
        "total_train_queries": len(train_data),
        "executable_queries": success_count,
        "non_executable_queries": len(train_data) - success_count,
        "unique_vertices_in_workload": len(vertex_freq),
        "unique_edges_in_workload": len(edge_freq),
        "unique_witnesses": len(witness_freq),
    }
    (output_dir / "workload_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **stats,
        "vertex_freq": vertex_freq,
        "edge_freq": edge_freq,
        "witness_freq": witness_freq,
        "traces": traces,
    }
