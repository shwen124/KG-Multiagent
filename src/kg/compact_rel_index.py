"""Compact per-relation adjacency for CWQ-needed Freebase relations.

Stores outgoing/incoming neighbors in sorted parallel arrays for O(log n) lookups.
Much faster than per-query SQLite on the full 134M-edge table.
"""

from __future__ import annotations

import array
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


class CompactRelIndex:
    def __init__(self):
        # rid -> (heads_sorted, tails_aligned)
        self.out: Dict[int, Tuple[array.array, array.array]] = {}
        self.inn: Dict[int, Tuple[array.array, array.array]] = {}

    @staticmethod
    def _sort_parallel(heads: array.array, tails: array.array) -> Tuple[array.array, array.array]:
        order = sorted(range(len(heads)), key=lambda i: (heads[i], tails[i]))
        h2 = array.array("I", (heads[i] for i in order))
        t2 = array.array("I", (tails[i] for i in order))
        return h2, t2

    @classmethod
    def build_from_edges(
        cls,
        edges_tsv: Path,
        needed_rids: Set[int],
        report_every: int = 20_000_000,
    ) -> "CompactRelIndex":
        out_h: Dict[int, array.array] = defaultdict(lambda: array.array("I"))
        out_t: Dict[int, array.array] = defaultdict(lambda: array.array("I"))
        in_t: Dict[int, array.array] = defaultdict(lambda: array.array("I"))
        in_h: Dict[int, array.array] = defaultdict(lambda: array.array("I"))

        n = 0
        kept = 0
        with edges_tsv.open("r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                n += 1
                _eid, h, r, t = line.rstrip("\n").split("\t")
                rid = int(r)
                if rid not in needed_rids:
                    continue
                hv, tv = int(h), int(t)
                out_h[rid].append(hv)
                out_t[rid].append(tv)
                in_t[rid].append(tv)
                in_h[rid].append(hv)
                kept += 1
                if n % report_every == 0:
                    print(f"  scan {n:,} kept {kept:,}", flush=True)

        print(f"  scanned {n:,} kept {kept:,}; sorting...", flush=True)
        idx = cls()
        for rid in needed_rids:
            if rid not in out_h:
                continue
            idx.out[rid] = cls._sort_parallel(out_h[rid], out_t[rid])
            idx.inn[rid] = cls._sort_parallel(in_t[rid], in_h[rid])
        print(f"  relations indexed: {len(idx.out)}", flush=True)
        return idx

    @staticmethod
    def _lookup(keys: array.array, vals: array.array, key: int) -> List[int]:
        # binary search range for key in sorted keys
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(keys) or keys[lo] != key:
            return []
        start = lo
        while lo < len(keys) and keys[lo] == key:
            lo += 1
        return vals[start:lo].tolist()

    def tails(self, head: int, relation: int) -> List[int]:
        pair = self.out.get(relation)
        if not pair:
            return []
        return self._lookup(pair[0], pair[1], head)

    def heads(self, tail: int, relation: int) -> List[int]:
        pair = self.inn.get(relation)
        if not pair:
            return []
        return self._lookup(pair[0], pair[1], tail)

    def has_edge(self, head: int, relation: int, tail: int) -> bool:
        pair = self.out.get(relation)
        if not pair:
            return False
        keys, vals = pair
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] < head:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        while i < len(keys) and keys[i] == head:
            if vals[i] == tail:
                return True
            if vals[i] > tail:
                return False
            i += 1
        return False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"out": self.out, "inn": self.inn}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "CompactRelIndex":
        with path.open("rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.out = data["out"]
        obj.inn = data["inn"]
        return obj


def collect_relation_ids_from_sparqls(
    sparqls: Iterable[str],
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> Set[int]:
    from src.workload.freebase_normalize import canonicalize_query_triples

    rids: Set[int] = set()
    for sparql in sparqls:
        canon = canonicalize_query_triples(sparql or "", relation2id, reverse_map)
        for t in canon["triples"]:
            rid = relation2id.get(t["r"])
            if rid is not None:
                rids.add(rid)
    return rids


def collect_cwq_relation_ids(
    train_json: Path,
    relation2id: Dict[str, int],
    reverse_map: Dict[str, str],
) -> Set[int]:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    return collect_relation_ids_from_sparqls(
        (sample.get("sparql") or "" for sample in data),
        relation2id,
        reverse_map,
    )


def merge_missing_relations(
    index: CompactRelIndex,
    edges_tsv: Path,
    needed_rids: Set[int],
) -> CompactRelIndex:
    """Scan edges.tsv for rids not yet in index and merge them in-place."""
    missing = {rid for rid in needed_rids if rid not in index.out}
    if not missing:
        print(f"  index already covers all {len(needed_rids)} needed relations", flush=True)
        return index
    print(f"  merging {len(missing)} missing relations into compact index ...", flush=True)
    extra = CompactRelIndex.build_from_edges(edges_tsv, missing)
    index.out.update(extra.out)
    index.inn.update(extra.inn)
    print(f"  index now has {len(index.out)} relations", flush=True)
    return index


def load_mid2vid_subset(vertices_tsv: Path, mids: Iterable[str]) -> Dict[str, int]:
    need = set(mids)
    out: Dict[str, int] = {}
    with vertices_tsv.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            vid_s, mid, _sid = line.rstrip("\n").split("\t")
            if mid in need:
                out[mid] = int(vid_s)
                if len(out) == len(need):
                    break
    return out
