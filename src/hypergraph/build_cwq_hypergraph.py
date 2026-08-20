"""Build CWQ workload-aware hypergraph on FB+CVT-REV (λ-scaled task edges).

Does NOT dump all 134M Freebase edges. Uses:
- structural: edges from successful CWQ train traces (+ optional compact-index
  edges whose both endpoints are in the workload vertex set)
- task: unique witnesses from successful traces, raw weight 5 then λ-scaled
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
    task_hyperedge_weight,
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

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
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
                key = "|".join(map(str, pins))
                if key in seen_w:
                    continue
                seen_w.add(key)
                witnesses.append(pins)

    return {
        "success_queries": success,
        "v_freq": dict(v_freq),
        "e_freq": {f"{h}\t{r}\t{t}": c for (h, r, t), c in e_freq.items()},
        "e_freq_tuples": dict(e_freq),
        "witnesses": witnesses,
        "vertex_set": set(v_freq.keys()),
    }


def expand_structural_from_index(
    index: CompactRelIndex,
    vertex_set: Set[int],
    existing: Dict[Edge, int],
) -> Dict[Edge, int]:
    """Add compact-index edges with both ends in workload vertex set (weight freq 0 → later w=1)."""
    out = dict(existing)
    vset = vertex_set
    added = 0
    for rid, (heads, tails) in index.out.items():
        # relation name unknown here; keep rid as string token "rid:{id}"
        rname = f"rid:{rid}"
        i = 0
        n = len(heads)
        while i < n:
            h = heads[i]
            if h not in vset:
                i += 1
                continue
            j = i
            while j < n and heads[j] == h:
                t = tails[j]
                if t in vset:
                    ek = (h, rname, t)
                    if ek not in out:
                        out[ek] = 0
                        added += 1
                j += 1
            i = j
    print(f"  expanded structural edges +{added} (total {len(out)})", flush=True)
    return out


def dense_remap(vids: Set[int]) -> Dict[int, int]:
    return {v: i + 1 for i, v in enumerate(sorted(vids))}


def build_cwq_hypergraph(
    trace_path: Path,
    out_dir: Path,
    lam: float = 1.0,
    expand_index: Optional[Path] = None,
    raw_task_w: int = 5,
) -> dict:
    print("Accumulating frequencies from CWQ traces ...", flush=True)
    stats = accumulate_from_traces(trace_path)
    v_freq = stats["v_freq"]
    e_freq_tuples: Dict[Edge, int] = stats["e_freq_tuples"]
    witnesses: List[List[int]] = stats["witnesses"]
    vertex_set: Set[int] = set(stats["vertex_set"])

    if expand_index and expand_index.exists():
        print(f"Expanding structural edges via {expand_index} ...", flush=True)
        index = CompactRelIndex.load(expand_index)
        e_freq_tuples = expand_structural_from_index(index, vertex_set, e_freq_tuples)
        # vertices only from workload; expansion does not add new verts by construction

    # string keys for assign_edge_weights
    e_freq_str = {f"{h}\t{r}\t{t}": c for (h, r, t), c in e_freq_tuples.items()}
    # vertices with freq 0 should not exist; structural-only verts N/A
    v_weights = assign_vertex_weights({str(k): v for k, v in v_freq.items()})
    # edges with freq 0 get normalized weight 0 → structural weight 1
    e_weights_str = {}
    positive = {k: c for k, c in e_freq_str.items() if c > 0}
    e_weights_pos = assign_edge_weights(positive)
    for k, c in e_freq_str.items():
        e_weights_str[k] = e_weights_pos.get(k, 1)

    structural: List[Tuple[int, List[int], str]] = []  # weight, pins, meta
    for (h, r, t), _c in e_freq_tuples.items():
        w = e_weights_str[f"{h}\t{r}\t{t}"]
        structural.append((w, [h, t], r))

    raw_tasks = [raw_task_w for _ in witnesses]
    struct_w_list = [w for w, _, _ in structural]
    scaled_tasks, lam_meta = scale_task_weights_by_lambda(struct_w_list, raw_tasks, lam)

    # dense remap for hMETIS / Mt-KaHyPar
    all_vids = set(vertex_set)
    for wit in witnesses:
        all_vids.update(wit)
    remap = dense_remap(all_vids)

    out_dir.mkdir(parents=True, exist_ok=True)

    # sidecars
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

    # .hgr format 11? format 3 = vertex weights + hyperedge weights
    # Mt-KaHyPar / hMETIS: first line "num_vertices num_hyperedges 11" or "3"
    # Our KQA used: weight on hyperedge line first, then count? Looking at KQA:
    #   f.write(f"{weight} {len(pins)} {' '.join(...)}\n")
    # Standard hMETIS format with edge+vertex weights (fmt=11):
    #   each hyperedge: weight v1 v2 ...
    #   then n vertex weights
    # Actually hMETIS fmt:
    #  0: no weights
    #  1: edge weights
    # 10: vertex weights  
    # 11: both
    # hyperedge line with edge weights: <weight> <v1> <v2> ...
    # Then N lines of vertex weights

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
        "dataset": "CWQ",
        "lambda": lam,
        "success_queries": stats["success_queries"],
        "num_vertices": num_v,
        "num_structural_hyperedges": len(structural),
        "num_task_hyperedges": len(witnesses),
        "num_hyperedges_total": len(hyperedges_out),
        "lambda_meta": lam_meta,
        "hgr_path": str(hgr_path),
        "note": (
            "Structural edges from CWQ train traces"
            + (" + compact-index expansion within workload verts" if expand_index else "")
            + "; task weights λ-scaled. Dense remap for hgr."
        ),
    }
    (out_dir / "hypergraph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        type=Path,
        default=ROOT / "data" / "CWQ" / "processed" / "query_traces.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "hypergraph" / "CWQ",
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    parser.add_argument(
        "--expand-index",
        type=Path,
        default=ROOT / "data" / "Freebase" / "processed" / "cwq_rel_index.pkl",
    )
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--raw-task-weight", type=int, default=5)
    args = parser.parse_args()
    build_cwq_hypergraph(
        trace_path=args.traces,
        out_dir=args.out_dir,
        lam=args.lam,
        expand_index=None if args.no_expand else args.expand_index,
        raw_task_w=args.raw_task_weight,
    )


if __name__ == "__main__":
    main()
