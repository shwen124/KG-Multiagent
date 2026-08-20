"""End-to-end KQA Pro preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hypergraph.build_from_traces import build_hypergraph
from src.kg.build_kqa_graph import build_kqa_graph
from src.workload.trace_kqa_train import generate_query_traces


def run_pipeline(
    kb_path: Path,
    train_path: Path,
    output_dir: Path,
    limit: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = output_dir / "graph"
    graph_stats = build_kqa_graph(kb_path, graph_dir)
    trace_stats = generate_query_traces(train_path, kb_path, output_dir, limit=limit)
    hypergraph_stats = build_hypergraph(
        graph_dir=graph_dir,
        weights_dir=output_dir / "weights",
        workload_dir=output_dir / "workload",
        hypergraph_dir=output_dir / "hypergraph",
    )
    summary = {
        "dataset": "KQAPro",
        "graph": {
            "num_vertices": graph_stats["num_vertices"],
            "num_edges": graph_stats["num_edges"],
        },
        "workload": {
            "total_train_queries": trace_stats["total_train_queries"],
            "executable_queries": trace_stats["executable_queries"],
            "non_executable_queries": trace_stats["non_executable_queries"],
            "unique_vertices_in_workload": trace_stats["unique_vertices_in_workload"],
            "unique_edges_in_workload": trace_stats["unique_edges_in_workload"],
            "unique_witnesses": trace_stats["unique_witnesses"],
        },
        "hypergraph": hypergraph_stats,
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Process KQA Pro into hypergraph inputs.")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path("data/kqa-pro/KQAPro/kb.json"),
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/kqa-pro/KQAPro/train.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/kqa-pro/processed"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    args = parser.parse_args()
    summary = run_pipeline(args.kb_path, args.train_path, args.output_dir, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
