"""检索质量指标计算（任务 25.7）。

实现了 ``Recall@K``、``MRR``（Mean Reciprocal Rank）和 ``NDCG@K``
（Normalized Discounted Cumulative Gain），均为信息检索领域的标准指标。

设计要点
--------

- 函数纯 Python 实现，无外部依赖，便于在 CI 中运行
- 支持二值相关性（默认）与等级相关性（``relevance_grades`` 参数）
- 所有函数都对空结果集 / 空 ground truth 给出确定性返回值
- 提供 ``EvalSample`` 与 ``EvalResult`` 数据类作为契约边界，
  便于 ``run_eval.py`` 与各种检索后端对接
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


# ─── 数据类 ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalSample:
    """单个评估样本：query + 相关 chunk_id 列表。"""

    query_id: str
    query: str
    # 相关 chunk_id 列表，可按相关性递减排序；二值相关性时全部为 1
    relevant_chunk_ids: list[str]
    # 可选：每个 chunk_id 的相关性等级（0..N），未提供则全为 1
    relevance_grades: dict[str, float] | None = None


@dataclass
class EvalResult:
    """聚合的评估结果。"""

    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    num_samples: int = 0


# ─── 单样本指标 ───────────────────────────────────────────────────────


def recall_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int
) -> float:
    """单样本 Recall@K。

    Args:
        retrieved: 检索系统返回的结果（按相关性递减排序）
        relevant: 标注的相关 chunk_id 集合
        k: 截断位置（>=1）

    Returns:
        命中相关项的比例，∈ [0, 1]。relevant 为空时返回 0.0。
    """
    if k <= 0:
        raise ValueError("k 必须 ≥ 1")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(
    retrieved: Sequence[str], relevant: Iterable[str]
) -> float:
    """单样本 Reciprocal Rank。

    Returns:
        第一个命中相关项的位置倒数 ``1/rank``，若没有命中则返回 0.0。
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for idx, cid in enumerate(retrieved, start=1):
        if cid in relevant_set:
            return 1.0 / idx
    return 0.0


def dcg_at_k(
    retrieved: Sequence[str],
    grades: dict[str, float],
    k: int,
) -> float:
    """单样本 DCG@K（基于 log2 折扣）。

    使用 ``DCG = sum(rel_i / log2(i+1))`` 公式（标准实现，
    位置 i 从 1 开始）。
    """
    if k <= 0:
        raise ValueError("k 必须 ≥ 1")
    dcg = 0.0
    for idx, cid in enumerate(retrieved[:k], start=1):
        rel = grades.get(cid, 0.0)
        if rel > 0:
            dcg += rel / math.log2(idx + 1)
    return dcg


def ndcg_at_k(
    retrieved: Sequence[str],
    grades: dict[str, float],
    k: int,
) -> float:
    """单样本 NDCG@K。

    Args:
        retrieved: 检索结果（按相关性递减排序）
        grades: chunk_id → 相关性等级（≥0）的映射
        k: 截断位置

    Returns:
        归一化 DCG，∈ [0, 1]。理想 DCG 为 0 时返回 0.0。
    """
    if k <= 0:
        raise ValueError("k 必须 ≥ 1")
    if not grades:
        return 0.0
    actual = dcg_at_k(retrieved, grades, k)
    # 理想排序：把 grades 中所有正分按递减顺序取前 k 个
    ideal_grades = sorted([g for g in grades.values() if g > 0], reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    if ideal == 0:
        return 0.0
    return actual / ideal


# ─── 聚合指标 ─────────────────────────────────────────────────────────


def evaluate(
    samples: Iterable[EvalSample],
    retrieve_fn,
    *,
    k_values: Sequence[int] = (1, 5, 10, 20),
) -> EvalResult:
    """对一批样本计算聚合指标。

    Args:
        samples: 待评估样本
        retrieve_fn: 同步函数 ``(query: str) -> list[str]``，返回 chunk_id
            列表（按相关性递减排序）
        k_values: 计算 Recall@K / NDCG@K 的截断位置集合

    Returns:
        EvalResult：包含 Recall@K（每个 K 一个数值）、MRR、NDCG@K
        以及样本总数。
    """
    sample_list = list(samples)
    if not sample_list:
        return EvalResult()

    recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
    ndcg_sums: dict[int, float] = {k: 0.0 for k in k_values}
    rr_sum = 0.0

    for sample in sample_list:
        retrieved = retrieve_fn(sample.query) or []
        grades = sample.relevance_grades or {
            cid: 1.0 for cid in sample.relevant_chunk_ids
        }
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, grades.keys(), k)
            ndcg_sums[k] += ndcg_at_k(retrieved, grades, k)
        rr_sum += reciprocal_rank(retrieved, grades.keys())

    n = len(sample_list)
    return EvalResult(
        recall_at_k={k: recall_sums[k] / n for k in k_values},
        mrr=rr_sum / n,
        ndcg_at_k={k: ndcg_sums[k] / n for k in k_values},
        num_samples=n,
    )
