"""Shared log-normalized weight helpers for hypergraph construction."""

from __future__ import annotations

import math
from typing import Dict, Iterable


def log_normalize(count: int, max_count: int) -> float:
    if count <= 0 or max_count <= 0:
        return 0.0
    return math.log1p(count) / math.log1p(max_count)


def vertex_weight(normalized_freq: float) -> int:
    return 1 + round(4 * normalized_freq)


def structural_edge_weight(normalized_freq: float) -> int:
    return 1 + round(9 * normalized_freq)


def task_hyperedge_weight() -> int:
    return 5


def assign_vertex_weights(freq: Dict[str, int]) -> Dict[str, int]:
    if not freq:
        return {}
    max_count = max(freq.values())
    return {
        key: vertex_weight(log_normalize(count, max_count))
        for key, count in freq.items()
    }


def assign_edge_weights(freq: Dict[str, int]) -> Dict[str, int]:
    if not freq:
        return {}
    max_count = max(freq.values())
    return {
        key: structural_edge_weight(log_normalize(count, max_count))
        for key, count in freq.items()
    }


def edge_key(head: str, relation: str, tail: str) -> str:
    return f"{head}\t{relation}\t{tail}"


def unique_query_level_increment(items: Iterable[str]) -> Dict[str, int]:
    return {item: 1 for item in set(items)}
