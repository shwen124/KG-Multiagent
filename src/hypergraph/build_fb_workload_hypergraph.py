"""Build workload-aware Freebase hypergraph from unified query_traces.jsonl.

Used by CWQ / WebQSP / GrailQA after grounding. Closure only adds edges whose
both endpoints are already in the train workload vertex set (no k-hop growth).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.common.weights import (
    assign_edge_weights,
    assign_vertex_weights,
    scale_task_weights_by_lambda,
)
from src.kg.compact_rel_index import CompactRelIndex


ROOT = Path(__file__).resolve().parents[2]
Edge = Tuple[int, str, int]


def accumulate_from_traces(trace_path: Path) -> dict:
    v_freq: Counter = Counter()
    e_freq: Counter = Counter()
    witnesses: List[List[int]] = []
    seen_w: Set[str] = set()
    success = 0
    total = 0
    raw_task = 0

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total += 1
            if not rec.get("success"):
                continue
            success += 1
            verts = [int(v) for v in rec.get("vertices") or []]
            for v in set(verts):
                v_freq[v] += 1
            edge_keys = set()
            for e in rec.get("edges") or []:
                h, r, t = int(e[0]), str(e[1]), int(e[2])
                edge_keys.add((h, r, t))
            for ek in edge_keys:
                e_freq[ek] += 1
            for wit in rec.get("witnesses") or []:
                pins = sorted({int(x) for x in wit})
                if len(pins) < 2:
                    continue
                raw_task += 1
                key = "|".join(map(str, pins))
                if key in seen_w:
                    continue
                seen_w.add(key)
                witnesses.append(pins)

    return {
        "total_train_questions": total,
        "successful_questions": success,
        "failed_questions": total - success,
        "success_rate": (success / total) if total else 0.0,
        "v_freq": dict(v_freq),
        "e_freq_tuples": dict(e_freq),
        "witnesses": witnesses,
        "raw_task_hyperedges": raw_task,
        "unique_task_hyperedges": len(witnesses),
        "vertex_set": set(v_freq.keys()),
    }


def expand_structural_from_index(
    index: CompactRelIndex,
    vertex_set: Set[int],
    existing: Dict[Edge, int],
) -> Tuple[Dict[Edge, int], int]:
    out = dict(existing)
    added = 0
    for rid, (heads, tails) in index.out.items():
        rname = f"rid:{rid}"
        i = 0
        n = len(heads)
        while i < n:
            h = heads[i]
            if h not in vertex_set:
                i += 1
                continue
            j = i
            while j < n and heads[j] == h:
                t = tails[j]
                if t in vertex_set:
                    ek = (h, rname, t)
                    if ek not in out:
                        out[ek] = 0
                        added += 1
                j += 1
            i = j
    return out, added


def dense_remap(vids: Set[int]) -> Dict[int, int]:
    return {v: i + 1 for i, v in enumerate(sorted(vids))}


def build_workload_hypergraph(
    dataset: str,
    trace_path: Path,
    out_dir: Path,
    lam: float = 1.0,
    expand_index: Optional[Path] = None,
    raw_task_w: int = 5,
) -> dict:
    print(f"[{dataset}] Accumulating frequencies from traces ...", flush=True)
    stats = accumulate_from_traces(trace_path)
    v_freq = stats["v_freq"]
    e_freq_tuples: Dict[Edge, int] = stats["e_freq_tuples"]
    witnesses: List[List[int]] = stats["witnesses"]
    vertex_set: Set[int] = set(stats["vertex_set"])
    trace_structural = len(e_freq_tuples)

    closure_added = 0
    if expand_index and expand_index.exists():
        print(f"[{dataset}] Expanding induced closure via {expand_index} ...", flush=True)
        index = CompactRelIndex.load(expand_index)
        e_freq_tuples, closure_added = expand_structural_from_index(
            index, vertex_set, e_freq_tuples
        )
        print(f"  closure +{closure_added} (total {len(e_freq_tuples)})", flush=True)

    e_freq_str = {f"{h}\t{r}\t{t}": c for (h, r, t), c in e_freq_tuples.items()}
    v_weights = assign_vertex_weights({str(k): v for k, v in v_freq.items()})
    positive = {k: c for k, c in e_freq_str.items() if c > 0}
    e_weights_pos = assign_edge_weights(positive)
    e_weights_str = {k: e_weights_pos.get(k, 1) for k in e_freq_str}

    structural: List[Tuple[int, List[int], str]] = []
    for (h, r, t), _c in e_freq_tuples.items():
        w = e_weights_str[f"{h}\t{r}\t{t}"]
        structural.append((w, [h, t], r))

    raw_tasks = [raw_task_w for _ in witnesses]
    struct_w_list = [w for w, _, _ in structural]
    scaled_tasks, lam_meta = scale_task_weights_by_lambda(struct_w_list, raw_tasks, lam)

    all_vids = set(vertex_set)
    for wit in witnesses:
        all_vids.update(wit)
    remap = dense_remap(all_vids)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "vertex_weights.tsv").open("w", encoding="utf-8") as f:
        f.write("dense_vid\tfreebase_vid\tfrequency\tweight\n")
        for fb_vid in sorted(all_vids):
            f.write(
                f"{remap[fb_vid]}\t{fb_vid}\t{v_freq.get(fb_vid, 0)}\t"
                f"{v_weights.get(str(fb_vid), 1)}\n"
            )

    with (out_dir / "structural_hyperedges.tsv").open("w", encoding="utf-8") as f:
        f.write("dense_h\tdense_t\trelation\tfrequency\tweight\n")
        for (h, r, t), c in sorted(e_freq_tuples.items()):
            key = f"{h}\t{r}\t{t}"
            f.write(f"{remap[h]}\t{remap[t]}\t{r}\t{c}\t{e_weights_str[key]}\n")

    with (out_dir / "task_hyperedges.tsv").open("w", encoding="utf-8") as f:
        f.write("dense_pins\traw_weight\tscaled_weight\n")
        for wit, raw_w, sc_w in zip(witnesses, raw_tasks, scaled_tasks):
            pins = ",".join(str(remap[v]) for v in wit)
            f.write(f"{pins}\t{raw_w}\t{sc_w}\n")

    num_v = len(remap)
    hyperedges_out: List[Tuple[int, List[int]]] = []
    for w, pins, _r in structural:
        hyperedges_out.append((w, sorted({remap[p] for p in pins})))
    for wit, sc_w in zip(witnesses, scaled_tasks):
        hyperedges_out.append((sc_w, sorted({remap[p] for p in wit})))

    hgr_path = out_dir / "graph.hgr"
    with hgr_path.open("w", encoding="utf-8") as f:
        f.write(f"{num_v} {len(hyperedges_out)} 11\n")
        for w, pins in hyperedges_out:
            f.write(f"{w} {' '.join(map(str, pins))}\n")
        for fb_vid in sorted(all_vids):
            f.write(f"{v_weights.get(str(fb_vid), 1)}\n")

    summary = {
        "dataset": dataset,
        "total_train_questions": stats["total_train_questions"],
        "successful_questions": stats["successful_questions"],
        "failed_questions": stats["failed_questions"],
        "success_rate": stats["success_rate"],
        "vertices": num_v,
        "trace_structural_edges": trace_structural,
        "closure_structural_edges": closure_added,
        "total_structural_edges": len(structural),
        "raw_task_hyperedges": stats["raw_task_hyperedges"],
        "unique_task_hyperedges": stats["unique_task_hyperedges"],
        "num_hyperedges_total": len(hyperedges_out),
        "lambda": lam,
        "gamma": lam_meta.get("gamma"),
        "structural_mass": lam_meta.get("M_s"),
        "workload_mass_before_scale": lam_meta.get("M_w_raw"),
        "workload_mass_after_scale": lam_meta.get("M_w_scaled"),
        "Mw_over_Ms_after_scale": lam_meta.get("ratio_Mw_over_Ms"),
        "lambda_meta": lam_meta,
        "hgr_path": str(hgr_path),
        "note": (
            "workload-induced Freebase subgraph; closure only within V_train; "
            "task hyperedges are unique witnesses (raw count also reported)."
        ),
    }
    (out_dir / "hypergraph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument(
        "--expand-index",
        type=Path,
        default=ROOT / "data" / "Freebase" / "index" / "freebase_rel_index.pkl",
    )
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--raw-task-weight", type=int, default=5)
    args = parser.parse_args()
    build_workload_hypergraph(
        dataset=args.dataset,
        trace_path=args.traces,
        out_dir=args.out_dir,
        lam=args.lam,
        expand_index=None if args.no_expand else args.expand_index,
        raw_task_w=args.raw_task_weight,
    )


if __name__ == "__main__":
    main()
