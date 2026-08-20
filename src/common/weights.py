"""Shared log-normalized weight helpers for hypergraph construction."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


def log_normalize(count: int, max_count: int) -> float:
    if count <= 0 or max_count <= 0:
        return 0.0
    return math.log1p(count) / math.log1p(max_count)


def vertex_weight(normalized_freq: float) -> int:
    return 1 + round(4 * normalized_freq)


def structural_edge_weight(normalized_freq: float) -> int:
    return 1 + round(9 * normalized_freq)


def task_hyperedge_weight() -> int:
    """Raw task weight before λ scaling (legacy / KQA Pro)."""
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


def scale_task_weights_by_lambda(
    structural_weights: Sequence[int],
    raw_task_weights: Sequence[int],
    lam: float,
) -> Tuple[List[int], dict]:
    """Scale task hyperedge weights so workload mass ≈ λ * structural mass.

    w_H = max(1, round(γ * ŵ_H)),  γ = λ * M_s / M_w
    """
    m_s = float(sum(structural_weights)) if structural_weights else 0.0
    m_w = float(sum(raw_task_weights)) if raw_task_weights else 0.0
    if lam <= 0 or m_w <= 0 or m_s <= 0:
        scaled = (
            [1 for _ in raw_task_weights]
            if lam <= 0
            else [max(1, int(w)) for w in raw_task_weights]
        )
        meta = {
            "lambda": lam,
            "M_s": m_s,
            "M_w_raw": m_w,
            "gamma": 0.0 if lam <= 0 else (lam * m_s / m_w if m_w else 0.0),
            "M_w_scaled": float(sum(scaled)),
        }
        return scaled, meta

    gamma = lam * m_s / m_w
    scaled = [max(1, int(round(gamma * w))) for w in raw_task_weights]
    meta = {
        "lambda": lam,
        "M_s": m_s,
        "M_w_raw": m_w,
        "gamma": gamma,
        "M_w_scaled": float(sum(scaled)),
        "ratio_Mw_over_Ms": (float(sum(scaled)) / m_s) if m_s else None,
    }
    return scaled, meta
