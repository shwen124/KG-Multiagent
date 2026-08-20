"""End-to-end CWQ pipeline on FB+CVT-REV (download/extract/build/index/coverage)."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from src.kg.build_fb_cvt_rev import build_fb_cvt_rev_graph, find_fb_cvt_rev_dir
from src.kg.index_fb_sqlite import build_sqlite_index
from src.workload.trace_cwq_fb import run_cwq_coverage


ROOT = Path(__file__).resolve().parents[2]


def extract_fb_cvt_rev(zip_path: Path, source_out: Path) -> Path:
    source_out.mkdir(parents=True, exist_ok=True)
    print(f"Listing zip members matching FB+CVT-REV in {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        matches = [n for n in z.namelist() if "fb+cvt-rev" in n.lower().replace(" ", "")]
        if not matches:
            # broader search
            matches = [
                n
                for n in z.namelist()
                if "cvt" in n.lower() and "rev" in n.lower() and "+cvt" in n.lower().replace(" ", "")
            ]
            # exclude +REV (with reverse) when possible: FB+CVT+REV vs FB+CVT-REV
            matches = [n for n in matches if "fb+cvt-rev" in n.lower().replace(" ", "") or "/fb+cvt-rev" in n.lower()]
        if not matches:
            # print some CVT names for debugging
            sample = [n for n in z.namelist() if "cvt" in n.lower()][:40]
            raise FileNotFoundError(
                "No FB+CVT-REV members found in zip. Sample CVT entries:\n" + "\n".join(sample)
            )
        print(f"Extracting {len(matches)} files ...", flush=True)
        for i, name in enumerate(matches):
            z.extract(name, source_out)
            if (i + 1) % 20 == 0:
                print(f"  extracted {i+1}/{len(matches)}", flush=True)
    return find_fb_cvt_rev_dir(source_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Path to idirlab-freebases.zip",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="CWQ train limit for debug")
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "graph", "index", "coverage"],
        default="all",
    )
    args = parser.parse_args()

    root = args.root
    fb_raw = root / "data" / "Freebase" / "raw" / "idirlab-freebases.zip"
    zip_path = args.zip_path or fb_raw
    source_dir = root / "data" / "Freebase" / "source"
    processed_dir = root / "data" / "Freebase" / "processed"
    ontology_rev = root / "data" / "Freebase" / "ontology" / "reverse_properties"
    cwq_train = root / "data" / "CWQ" / "origin" / "ComplexWebQuestions_train.json"
    cwq_out = root / "data" / "CWQ" / "processed"

    stages = {
        "extract": args.stage in {"all", "extract"},
        "graph": args.stage in {"all", "graph"},
        "index": args.stage in {"all", "index"},
        "coverage": args.stage in {"all", "coverage"},
    }
    if args.skip_extract:
        stages["extract"] = False
    if args.skip_graph:
        stages["graph"] = False
    if args.skip_index:
        stages["index"] = False
    if args.skip_coverage:
        stages["coverage"] = False

    summary = {}

    if stages["extract"]:
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing zip: {zip_path}. Wait for Zenodo download.")
        size_gb = zip_path.stat().st_size / (1024**3)
        print(f"Zip size: {size_gb:.2f} GB", flush=True)
        fb_dir = extract_fb_cvt_rev(zip_path, source_dir)
        summary["extracted"] = str(fb_dir)
        print(f"FB+CVT-REV at {fb_dir}", flush=True)

    if stages["graph"]:
        print("Building vertices/edges/relations ...", flush=True)
        meta = build_fb_cvt_rev_graph(source_dir, processed_dir)
        summary["graph"] = meta
        print(json.dumps(meta, indent=2), flush=True)

    if stages["index"]:
        edges = processed_dir / "edges.tsv"
        db = processed_dir / "fb_cvt_rev.sqlite"
        print(f"Building SQLite index -> {db} ...", flush=True)
        idx = build_sqlite_index(edges, db)
        summary["index"] = idx
        print(json.dumps(idx, indent=2), flush=True)

    if stages["coverage"]:
        if not cwq_train.exists():
            raise FileNotFoundError(cwq_train)
        if not ontology_rev.exists():
            raise FileNotFoundError(ontology_rev)
        print("Running CWQ coverage / grounding ...", flush=True)
        cov = run_cwq_coverage(
            train_json=cwq_train,
            processed_graph_dir=processed_dir,
            reverse_properties_path=ontology_rev,
            out_dir=cwq_out,
            limit=args.limit,
            db_path=processed_dir / "fb_cvt_rev.sqlite",
        )
        summary["coverage"] = cov
        print(json.dumps(cov, indent=2), flush=True)

    (processed_dir / "cwq_pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
