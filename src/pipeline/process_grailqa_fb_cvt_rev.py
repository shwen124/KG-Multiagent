"""GrailQA end-to-end on shared FB+CVT-REV (coverage + λ hypergraph)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hypergraph.build_fb_workload_hypergraph import build_workload_hypergraph
from src.workload.trace_grailqa_fb_fast import run_grailqa_coverage_fast


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument(
        "--stage",
        choices=["all", "coverage", "hypergraph"],
        default="all",
    )
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--skip-dev-eval", action="store_true")
    args = parser.parse_args()

    root = args.root
    processed_dir = root / "data" / "Freebase" / "processed"
    index_dir = root / "data" / "Freebase" / "index"
    ontology_rev = root / "data" / "Freebase" / "ontology" / "reverse_properties"
    train_json = root / "data" / "GrailQA" / "origin" / "grailqa_v1.0_train.json"
    dev_json = root / "data" / "GrailQA" / "origin" / "grailqa_v1.0_dev.json"
    grail_out = root / "data" / "GrailQA" / "processed"
    hgr_out = root / "data" / "hypergraph" / "GrailQA"

    if not train_json.exists():
        alt = root / "data" / "GrailQA" / "GrailQA_v1.0" / "grailqa_v1.0_train.json"
        if alt.exists():
            train_json = alt
            dev_json = root / "data" / "GrailQA" / "GrailQA_v1.0" / "grailqa_v1.0_dev.json"
        else:
            raise FileNotFoundError(train_json)

    summary: dict = {}

    if args.stage in {"all", "coverage"}:
        print("Running GrailQA coverage on FB+CVT-REV ...", flush=True)
        cov = run_grailqa_coverage_fast(
            train_json=train_json,
            processed_graph_dir=processed_dir,
            reverse_properties_path=ontology_rev,
            out_dir=grail_out,
            index_dir=index_dir,
            limit=args.limit,
            dev_json=None if args.skip_dev_eval else dev_json,
        )
        summary["coverage"] = cov
        print(json.dumps(cov, ensure_ascii=False, indent=2), flush=True)

    if args.stage in {"all", "hypergraph"}:
        traces = grail_out / "query_traces.jsonl"
        if not traces.exists():
            raise FileNotFoundError(traces)
        expand = None if args.no_expand else (index_dir / "freebase_rel_index.pkl")
        print("Building GrailQA workload hypergraph ...", flush=True)
        hg = build_workload_hypergraph(
            dataset="GrailQA",
            trace_path=traces,
            out_dir=hgr_out,
            lam=args.lam,
            expand_index=expand,
        )
        summary["hypergraph"] = hg

    grail_out.mkdir(parents=True, exist_ok=True)
    (grail_out / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
