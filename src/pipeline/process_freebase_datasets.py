"""Process CWQ / WebQSP / GrailQA with shared Freebase graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Set

from src.hypergraph.build_from_traces import build_hypergraph
from src.kg.build_freebase_graph import build_shared_freebase, load_adjacency
from src.workload.ground_sparql import ground_sparql
from src.workload.sparql_utils import is_entity_mid, normalize_term
from src.workload.write_workload import accumulate_and_write


ROOT = Path(__file__).resolve().parents[2]


def _answer_mids_cwq(sample: dict) -> Set[str]:
    out = set()
    for a in sample.get("answers") or []:
        aid = a.get("answer_id")
        if aid and is_entity_mid(str(aid)):
            out.add(normalize_term(str(aid)))
    return out


def _answer_mids_webqsp(parse: dict) -> Set[str]:
    out = set()
    for a in parse.get("Answers") or []:
        arg = a.get("AnswerArgument")
        if arg and is_entity_mid(str(arg)):
            out.add(normalize_term(str(arg)))
    return out


def _answer_mids_grailqa(sample: dict) -> Set[str]:
    out = set()
    for a in sample.get("answer") or []:
        arg = a.get("answer_argument")
        if arg and is_entity_mid(str(arg)):
            out.add(normalize_term(str(arg)))
    return out


def process_cwq_with_graph(
    adj, graph_dir: Path, train_path: Path, out_dir: Path, limit: Optional[int] = None, radj=None
) -> dict:
    data = json.loads(train_path.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]
    traces, metas = [], []
    for i, sample in enumerate(data):
        sparql = sample.get("sparql") or ""
        g = ground_sparql(sparql, adj, answer_mids=_answer_mids_cwq(sample), radj=radj)
        traces.append(
            {
                "qid": sample.get("ID", i),
                "question": sample.get("question"),
                "compositionality_type": sample.get("compositionality_type"),
                **g,
            }
        )
        metas.append({"qid": sample.get("ID", i), "question": sample.get("question"), "sparql": sparql})
    stats = accumulate_and_write(traces, out_dir, metas)
    hg = build_hypergraph(
        graph_dir=graph_dir,
        weights_dir=out_dir / "weights",
        workload_dir=out_dir / "workload",
        hypergraph_dir=out_dir / "hypergraph",
        hgr_name="CWQ.hgr",
    )
    summary = {"dataset": "CWQ", "workload": stats, "hypergraph": hg}
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def process_webqsp_with_graph(
    adj, graph_dir: Path, train_path: Path, out_dir: Path, limit: Optional[int] = None, radj=None
) -> dict:
    data = json.loads(train_path.read_text(encoding="utf-8"))
    questions = data["Questions"] if isinstance(data, dict) else data
    if limit is not None:
        questions = questions[:limit]
    traces, metas = [], []
    for q in questions:
        qid = q.get("QuestionId")
        # primary complete parses only; frequency +1 per question
        merged_v, merged_e, witnesses = set(), set(), []
        success = False
        partial = False
        used_sparql = None
        for parse in q.get("Parses") or []:
            sparql = parse.get("Sparql") or ""
            if not sparql:
                continue
            used_sparql = sparql
            g = ground_sparql(sparql, adj, answer_mids=_answer_mids_webqsp(parse), radj=radj)
            if g["success"]:
                success = True
                partial = partial or bool(g.get("partial"))
                merged_v.update(g["vertices"])
                merged_e.update(tuple(e) for e in g["edges"])
                witnesses.extend(g.get("witnesses") or [])
            topic = parse.get("TopicEntityMid")
            if topic:
                merged_v.add(normalize_term(topic))
            for a in parse.get("Answers") or []:
                arg = a.get("AnswerArgument")
                if arg and is_entity_mid(str(arg)):
                    merged_v.add(normalize_term(str(arg)))
        # dedupe witnesses
        uniq_w, seen = [], set()
        for w in witnesses:
            key = "|".join(w)
            if key not in seen and len(w) >= 2:
                seen.add(key)
                uniq_w.append(w)
        if not uniq_w and len(merged_v) >= 2:
            uniq_w = [sorted(merged_v)]
        traces.append(
            {
                "qid": qid,
                "question": q.get("RawQuestion") or q.get("ProcessedQuestion"),
                "success": success or len(merged_v) > 0,
                "partial": partial or not success,
                "vertices": sorted(merged_v),
                "edges": [list(e) for e in sorted(merged_e)],
                "witnesses": uniq_w[:32],
            }
        )
        metas.append({"qid": qid, "question": traces[-1]["question"], "sparql": used_sparql})
    stats = accumulate_and_write(traces, out_dir, metas)
    hg = build_hypergraph(
        graph_dir=graph_dir,
        weights_dir=out_dir / "weights",
        workload_dir=out_dir / "workload",
        hypergraph_dir=out_dir / "hypergraph",
        hgr_name="WebQSP.hgr",
    )
    summary = {"dataset": "WebQSP", "workload": stats, "hypergraph": hg}
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def process_grailqa_with_graph(
    adj, graph_dir: Path, train_path: Path, out_dir: Path, limit: Optional[int] = None, radj=None
) -> dict:
    data = json.loads(train_path.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]
    traces, metas = [], []
    for sample in data:
        sparql = sample.get("sparql_query") or ""
        g = ground_sparql(sparql, adj, answer_mids=_answer_mids_grailqa(sample), radj=radj)
        # enrich with graph_query entity nodes
        gq = sample.get("graph_query") or {}
        nodes = {n["nid"]: n for n in gq.get("nodes") or []}
        verts = set(g.get("vertices") or [])
        edges = {tuple(e) for e in (g.get("edges") or [])}
        for n in nodes.values():
            if n.get("node_type") in {"entity", "class"}:
                verts.add(normalize_term(str(n["id"])))
        for e in gq.get("edges") or []:
            sn, en = nodes.get(e["start"]), nodes.get(e["end"])
            if not sn or not en:
                continue
            s = normalize_term(str(sn["id"]))
            t = normalize_term(str(en["id"]))
            r = normalize_term(str(e["relation"]))
            edges.add((s, r, t))
            verts.update([s, t])
        answers = _answer_mids_grailqa(sample)
        verts |= answers
        witnesses = g.get("witnesses") or []
        if not witnesses and len(verts) >= 2:
            witnesses = [sorted(verts)]
        traces.append(
            {
                "qid": sample.get("qid"),
                "question": sample.get("question"),
                "level": sample.get("level") if "level" in sample else None,
                "success": bool(verts),
                "partial": bool(g.get("partial", True)),
                "vertices": sorted(verts),
                "edges": [list(e) for e in sorted(edges)],
                "witnesses": witnesses[:32],
            }
        )
        metas.append(
            {"qid": sample.get("qid"), "question": sample.get("question"), "sparql": sparql}
        )
    stats = accumulate_and_write(traces, out_dir, metas)
    hg = build_hypergraph(
        graph_dir=graph_dir,
        weights_dir=out_dir / "weights",
        workload_dir=out_dir / "workload",
        hypergraph_dir=out_dir / "hypergraph",
        hgr_name="GrailQA.hgr",
    )
    summary = {"dataset": "GrailQA", "workload": stats, "hypergraph": hg}
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only",
        choices=["all", "cwq", "webqsp", "grailqa", "graph"],
        default="all",
    )
    args = parser.parse_args()

    webqsp_train = ROOT / "data/WebQSP/WebQSP/data/WebQSP.train.json"
    cwq_train = ROOT / "data/CWQ/ComplexWebQuestions_train.json"
    grailqa_train = ROOT / "data/GrailQA/GrailQA_v1.0/grailqa_v1.0_train.json"
    graph_dir = ROOT / "data/Freebase/graph"

    print("Building shared Freebase graph...", flush=True)
    if args.only != "all" and (graph_dir / "vertices.tsv").exists() and args.only != "graph":
        gstats = json.loads((graph_dir / "graph_meta.json").read_text(encoding="utf-8"))
        print("Reusing existing graph:", gstats.get("num_vertices"), gstats.get("num_edges"), flush=True)
    else:
        gstats = build_shared_freebase(webqsp_train, cwq_train, grailqa_train, graph_dir)
        print(
            json.dumps(
                {k: gstats[k] for k in ("num_vertices", "num_edges", "source")},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    if args.only == "graph":
        return

    adj, radj = load_adjacency(graph_dir)
    results = {"freebase_graph": {"num_vertices": gstats["num_vertices"], "num_edges": gstats["num_edges"]}}

    if args.only in ("all", "cwq"):
        print("Processing CWQ...", flush=True)
        results["CWQ"] = process_cwq_with_graph(
            adj, graph_dir, cwq_train, ROOT / "data/CWQ/processed", limit=args.limit, radj=radj
        )
        print(json.dumps(results["CWQ"], ensure_ascii=False, indent=2), flush=True)

    if args.only in ("all", "webqsp"):
        print("Processing WebQSP...", flush=True)
        results["WebQSP"] = process_webqsp_with_graph(
            adj, graph_dir, webqsp_train, ROOT / "data/WebQSP/processed", limit=args.limit, radj=radj
        )
        print(json.dumps(results["WebQSP"], ensure_ascii=False, indent=2), flush=True)

    if args.only in ("all", "grailqa"):
        print("Processing GrailQA...", flush=True)
        results["GrailQA"] = process_grailqa_with_graph(
            adj, graph_dir, grailqa_train, ROOT / "data/GrailQA/processed", limit=args.limit, radj=radj
        )
        print(json.dumps(results["GrailQA"], ensure_ascii=False, indent=2), flush=True)
    (ROOT / "data/Freebase/pipeline_all_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
