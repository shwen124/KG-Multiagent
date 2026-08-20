"""Build fixed Freebase graph from IDIRLab FB+CVT-REV source files.

Merges train.txt + valid.txt + test.txt (link-prediction splits) into one KG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


def find_fb_cvt_rev_dir(source_root: Path) -> Path:
    """Locate the extracted FB+CVT-REV directory under source/."""
    for p in source_root.rglob("entity2id.txt"):
        parent = p.parent
        # Exact name match preferred
        if parent.name == "FB+CVT-REV":
            needed = ["train.txt", "valid.txt", "test.txt", "relation2id.txt"]
            if all((parent / f).exists() for f in needed):
                return parent
    raise FileNotFoundError(
        f"Cannot find FB+CVT-REV under {source_root}. "
        "Extract only that variant from idirlab-freebases.zip."
    )


def parse_entity2id(path: Path) -> Dict[str, int]:
    """Return MID -> source_id (0-based)."""
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: /m/xxx, 0   or /m/xxx\t0
            if ", " in line:
                mid, sid = line.rsplit(", ", 1)
            elif "," in line:
                mid, sid = line.rsplit(",", 1)
            else:
                parts = line.split()
                mid, sid = parts[0], parts[-1]
            mapping[mid.strip()] = int(sid.strip())
    return mapping


def parse_relation2id(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ", " in line:
                rel, sid = line.rsplit(", ", 1)
            elif "," in line:
                rel, sid = line.rsplit(",", 1)
            else:
                parts = line.split()
                rel, sid = parts[0], parts[-1]
            mapping[rel.strip()] = int(sid.strip())
    return mapping


def iter_triples(paths: Iterable[Path]) -> Iterator[Tuple[int, int, int]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ", " in line:
                    parts = [p.strip() for p in line.split(",")]
                elif "," in line:
                    parts = [p.strip() for p in line.split(",")]
                else:
                    parts = line.split()
                if len(parts) != 3:
                    continue
                yield int(parts[0]), int(parts[1]), int(parts[2])


def check_id_continuity(entity2id: Dict[str, int]) -> dict:
    ids = list(entity2id.values())
    min_id = min(ids)
    max_id = max(ids)
    n = len(ids)
    unique = len(set(ids))
    ok = min_id == 0 and max_id == n - 1 and unique == n
    return {
        "num_entities": n,
        "min_id": min_id,
        "max_id": max_id,
        "unique_id_count": unique,
        "contiguous_0_to_n_minus_1": ok,
    }


def build_fb_cvt_rev_graph(source_dir: Path, processed_dir: Path) -> dict:
    """Merge train/valid/test into vertices.tsv / edges.tsv / relations.tsv.

    Mt-KaHyPar vid = source_entity_id + 1
    """
    fb_dir = find_fb_cvt_rev_dir(source_dir)
    entity2id = parse_entity2id(fb_dir / "entity2id.txt")
    relation2id = parse_relation2id(fb_dir / "relation2id.txt")
    id_check = check_id_continuity(entity2id)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # vertices.tsv: vid, original_id, source_id
    with (processed_dir / "vertices.tsv").open("w", encoding="utf-8") as f:
        f.write("vid\toriginal_id\tsource_id\n")
        for mid, sid in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{sid + 1}\t{mid}\t{sid}\n")

    # relations.tsv
    id2rel = {v: k for k, v in relation2id.items()}
    with (processed_dir / "relations.tsv").open("w", encoding="utf-8") as f:
        f.write("relation_id\trelation\n")
        for rid in sorted(id2rel):
            f.write(f"{rid}\t{id2rel[rid]}\n")

    # edges.tsv from all three splits
    triple_files = [fb_dir / "train.txt", fb_dir / "valid.txt", fb_dir / "test.txt"]
    num_edges = 0
    with (processed_dir / "edges.tsv").open("w", encoding="utf-8") as f:
        f.write("eid\thead\trelation\ttail\n")
        eid = 1
        for h, r, t in iter_triples(triple_files):
            f.write(f"{eid}\t{h + 1}\t{r}\t{t + 1}\n")
            eid += 1
            num_edges += 1

    meta = {
        "source_dir": str(fb_dir),
        "variant": "FB+CVT-REV",
        "num_vertices": len(entity2id),
        "num_relations": len(relation2id),
        "num_edges": num_edges,
        "id_check": id_check,
        "note": "Merged train+valid+test link-prediction splits into one fixed KG.",
    }
    (processed_dir / "graph_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # save mid->vid and relation->rid for grounding
    mid2vid = {mid: sid + 1 for mid, sid in entity2id.items()}
    (processed_dir / "mid2vid.json").write_text(
        json.dumps(mid2vid), encoding="utf-8"
    )
    (processed_dir / "relation2id.json").write_text(
        json.dumps(relation2id), encoding="utf-8"
    )
    return meta
