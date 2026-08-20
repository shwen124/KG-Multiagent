"""WebQSP end-to-end on shared FB+CVT-REV (coverage + λ hypergraph)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hypergraph.build_fb_workload_hypergraph import build_workload_hypergraph
from src.workload.trace_webqsp_fb_fast import run_webqsp_coverage_fast


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
    args = parser.parse_args()

    root = args.root
    processed_dir = root / "data" / "Freebase" / "processed"
    index_dir = root / "data" / "Freebase" / "index"
    ontology_rev = root / "data" / "Freebase" / "ontology" / "reverse_properties"
    train_json = root / "data" / "WebQSP" / "origin" / "WebQSP.train.json"
    webqsp_out = root / "data" / "WebQSP" / "processed"
    hgr_out = root / "data" / "hypergraph" / "WebQSP"

    if not train_json.exists():
        alt = root / "data" / "WebQSP" / "WebQSP" / "data" / "WebQSP.train.json"
        if alt.exists():
            train_json = alt
        else:
            raise FileNotFoundError(train_json)

    summary: dict = {}

    if args.stage in {"all", "coverage"}:
        print("Running WebQSP coverage on FB+CVT-REV ...", flush=True)
        cov = run_webqsp_coverage_fast(
            train_json=train_json,
            processed_graph_dir=processed_dir,
            reverse_properties_path=ontology_rev,
            out_dir=webqsp_out,
            index_dir=index_dir,
            limit=args.limit,
        )
        summary["coverage"] = cov
        print(json.dumps(cov, ensure_ascii=False, indent=2), flush=True)

    if args.stage in {"all", "hypergraph"}:
        traces = webqsp_out / "query_traces.jsonl"
        if not traces.exists():
            raise FileNotFoundError(traces)
        expand = None if args.no_expand else (index_dir / "freebase_rel_index.pkl")
        if expand and not expand.exists():
            # fallback to CWQ seed index
            alt = processed_dir / "cwq_rel_index.pkl"
            expand = alt if alt.exists() else None
        print("Building WebQSP workload hypergraph ...", flush=True)
        hg = build_workload_hypergraph(
            dataset="WebQSP",
            trace_path=traces,
            out_dir=hgr_out,
            lam=args.lam,
            expand_index=expand,
        )
        summary["hypergraph"] = hg

    (webqsp_out / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
