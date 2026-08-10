"""RRF 融合算法（k=60，合并去重，Top 100 候选集）聚焦单元测试。

对应任务 14.5：实现 RRF 融合算法（k=60，合并去重，生成 Top 100 候选集）。

设计依据：
- requirements.md 需求 6.2：使用 RRF 算法（k=60）融合三路召回结果，
  生成不超过 100 个候选文档块的统一候选集。
- design.md "Search Service - RRF 融合"：candidates = self.rrf_fusion(recalls, k=60)。

公式：score(d) = Σ 1/(k + rank_i(d))，其中 k=60，rank 从 1 开始计。

测试范围：
- 公式正确性（已知输入对应已知 RRF 分数）
- 去重：同一 chunk_id 在多路命中时分数为各路贡献之和
- Top 100 限制
- k=60 常量保证
- 按 RRF 分数降序排列
- 空输入返回 []
- 单路召回输入（保留原顺序）

并通过 Hypothesis 验证以下属性：
- 候选数上界：len(fused) ≤ min(3 × TOP_K_PER_RETRIEVER, RRF_CANDIDATE_LIMIT)
- 排名一致性：输出按 RRF 分数严格非升序排列
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from app.services.search_service import (
    RRF_CANDIDATE_LIMIT,
    RRF_K,
    TOP_K_PER_RETRIEVER,
    SearchHit,
    SearchService,
)


# ─── Fixtures & Helpers ────────────────────────────────────────────────


@pytest.fixture
def search_service() -> SearchService:
    """无 embedding 依赖的 SearchService（RRF 融合是纯计算函数）。"""
    return SearchService(embedding_service=MagicMock())


def _make_hit(
    *,
    chunk_id: str | None = None,
    score: float = 1.0,
    content: str = "test content",
) -> SearchHit:
    """构造一个 SearchHit。"""
    return SearchHit(
        chunk_id=chunk_id or str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        space_id=str(uuid.uuid4()),
        chunk_index=0,
        title_chain="A > B",
        source_file="file.pdf",
        content=content,
        score=score,
    )


def _rrf_expected(rank: int) -> float:
    """RRF 单路贡献：1 / (k + rank)。"""
    return 1.0 / (RRF_K + rank)


# ─── 常量保证 ─────────────────────────────────────────────────────────


class TestRRFConstants:
    """k=60 与 Top 100 是设计契约，须在常量层固定。"""

    def test_k_equals_60(self) -> None:
        """RRF_K 必须严格等于 60（需求 6.2 显式约定）。"""
        assert RRF_K == 60

    def test_candidate_limit_is_100(self) -> None:
        """候选集上界必须为 100。"""
        assert RRF_CANDIDATE_LIMIT == 100


# ─── 公式正确性 ───────────────────────────────────────────────────────


class TestRRFFormula:
    """验证 score(d) = Σ 1/(k + rank_i(d))。"""

    def test_single_hit_rank1(self, search_service: SearchService) -> None:
        """单路、单条命中时，分数应严格等于 1/(60+1)。"""
        hit = _make_hit(chunk_id="c1")
        results = search_service._rrf_fusion([[hit]])

        assert len(results) == 1
        assert results[0].chunk_id == "c1"
        assert math.isclose(results[0].score, _rrf_expected(1), rel_tol=1e-12)

    def test_single_retriever_rank_decay(
        self, search_service: SearchService
    ) -> None:
        """单路下，第 N 名分数应等于 1/(60+N)。"""
        hits = [_make_hit(chunk_id=f"c{i}") for i in range(1, 6)]
        results = search_service._rrf_fusion([hits])

        for idx, hit in enumerate(results):
            expected_rank = idx + 1
            assert math.isclose(
                hit.score, _rrf_expected(expected_rank), rel_tol=1e-12
            ), f"rank={expected_rank} 分数应为 {_rrf_expected(expected_rank)}"

    def test_three_retrievers_known_score(
        self, search_service: SearchService
    ) -> None:
        """三路召回中 chunk 分别位于不同名次时，RRF 分数应为各贡献之和。"""
        chunk_id = "shared"
        # 在三路中分别排第 1、第 3、第 7
        r1 = [_make_hit(chunk_id=chunk_id)] + [
            _make_hit(chunk_id=f"r1_{i}") for i in range(2, 6)
        ]
        r2 = [_make_hit(chunk_id=f"r2_{i}") for i in range(1, 3)] + [
            _make_hit(chunk_id=chunk_id)
        ] + [_make_hit(chunk_id=f"r2_{i}") for i in range(4, 6)]
        r3 = [_make_hit(chunk_id=f"r3_{i}") for i in range(1, 7)] + [
            _make_hit(chunk_id=chunk_id)
        ]

        results = search_service._rrf_fusion([r1, r2, r3])

        target = next(h for h in results if h.chunk_id == chunk_id)
        expected = _rrf_expected(1) + _rrf_expected(3) + _rrf_expected(7)
        assert math.isclose(target.score, expected, rel_tol=1e-12)


# ─── 合并去重 ─────────────────────────────────────────────────────────


class TestRRFDeduplication:
    """同一 chunk_id 来自多路召回时应被合并，分数为各路贡献之和。"""

    def test_same_chunk_three_retrievers_summed(
        self, search_service: SearchService
    ) -> None:
        """同一 chunk 在三路均位列第 1 时，分数应是 3 × 1/(60+1)。"""
        chunk_id = "alpha"
        r1 = [_make_hit(chunk_id=chunk_id)]
        r2 = [_make_hit(chunk_id=chunk_id)]
        r3 = [_make_hit(chunk_id=chunk_id)]

        results = search_service._rrf_fusion([r1, r2, r3])

        assert len(results) == 1, "重复 chunk 应被合并为一条结果"
        assert results[0].chunk_id == chunk_id
        assert math.isclose(
            results[0].score, 3.0 * _rrf_expected(1), rel_tol=1e-12
        )

    def test_distinct_chunks_not_merged(
        self, search_service: SearchService
    ) -> None:
        """不同 chunk_id 不应被合并。"""
        r1 = [_make_hit(chunk_id=f"c{i}") for i in range(5)]
        r2 = [_make_hit(chunk_id=f"d{i}") for i in range(5)]

        results = search_service._rrf_fusion([r1, r2])

        chunk_ids = {h.chunk_id for h in results}
        assert len(chunk_ids) == 10
        assert chunk_ids == {f"c{i}" for i in range(5)} | {
            f"d{i}" for i in range(5)
        }

    def test_partial_overlap_boosts_shared(
        self, search_service: SearchService
    ) -> None:
        """部分重叠时，共享 chunk 因贡献叠加应排在仅出现在单路的 chunk 之前。"""
        shared = "shared"
        r1 = [_make_hit(chunk_id=shared), _make_hit(chunk_id="r1_only")]
        r2 = [_make_hit(chunk_id=shared), _make_hit(chunk_id="r2_only")]

        results = search_service._rrf_fusion([r1, r2])

        # 共享 chunk 应排第一
        assert results[0].chunk_id == shared
        shared_score = results[0].score
        # 仅在单路出现的 chunk 分数应严格小于共享 chunk
        for h in results[1:]:
            assert h.score < shared_score


# ─── Top 100 限制 ──────────────────────────────────────────────────────


class TestRRFCandidateLimit:
    """生成的候选集长度不应超过 RRF_CANDIDATE_LIMIT (=100)。"""

    def test_limit_with_three_full_retrievers(
        self, search_service: SearchService
    ) -> None:
        """三路各 50 条且全部互不重叠时（理论 150），应被截断到 100。"""
        r1 = [_make_hit(chunk_id=f"r1_{i}") for i in range(TOP_K_PER_RETRIEVER)]
        r2 = [_make_hit(chunk_id=f"r2_{i}") for i in range(TOP_K_PER_RETRIEVER)]
        r3 = [_make_hit(chunk_id=f"r3_{i}") for i in range(TOP_K_PER_RETRIEVER)]

        results = search_service._rrf_fusion([r1, r2, r3])

        assert len(results) == RRF_CANDIDATE_LIMIT == 100

    def test_limit_keeps_highest_scoring(
        self, search_service: SearchService
    ) -> None:
        """截断时保留的应是 RRF 分数最高的前 100 条（即各路靠前的 chunk）。"""
        # 让 chunk_0 在三路中均排第 1，理应稳定保留
        target = "must_keep"
        r1 = [_make_hit(chunk_id=target)] + [
            _make_hit(chunk_id=f"r1_{i}") for i in range(1, TOP_K_PER_RETRIEVER)
        ]
        r2 = [_make_hit(chunk_id=target)] + [
            _make_hit(chunk_id=f"r2_{i}") for i in range(1, TOP_K_PER_RETRIEVER)
        ]
        r3 = [_make_hit(chunk_id=target)] + [
            _make_hit(chunk_id=f"r3_{i}") for i in range(1, TOP_K_PER_RETRIEVER)
        ]

        results = search_service._rrf_fusion([r1, r2, r3])

        ids = [h.chunk_id for h in results]
        assert target in ids
        assert ids[0] == target  # 命中三路的 chunk 排第一


# ─── 排序 ────────────────────────────────────────────────────────────


class TestRRFOrdering:
    """输出必须按 RRF 分数降序排列。"""

    def test_sorted_descending(self, search_service: SearchService) -> None:
        """三路混合后输出应严格非升序（允许并列）。"""
        r1 = [_make_hit(chunk_id=f"a{i}") for i in range(10)]
        r2 = [_make_hit(chunk_id=f"a{i}") for i in range(5, 15)]
        r3 = [_make_hit(chunk_id=f"a{i}") for i in range(8, 18)]

        results = search_service._rrf_fusion([r1, r2, r3])

        for prev, nxt in zip(results, results[1:]):
            assert prev.score >= nxt.score


# ─── 边界与单路 ───────────────────────────────────────────────────────


class TestRRFEdgeCases:
    """空输入、单路输入等边界。"""

    def test_empty_input_returns_empty(self, search_service: SearchService) -> None:
        """无召回结果时返回空列表。"""
        assert search_service._rrf_fusion([]) == []

    def test_all_retrievers_empty(self, search_service: SearchService) -> None:
        """每一路都是空列表时返回 []。"""
        assert search_service._rrf_fusion([[], [], []]) == []

    def test_single_retriever_preserves_order(
        self, search_service: SearchService
    ) -> None:
        """仅一路召回时，输出顺序应与输入一致（因 1/(k+rank) 单调递减）。"""
        hits = [_make_hit(chunk_id=f"c{i}") for i in range(7)]
        results = search_service._rrf_fusion([hits])

        assert [h.chunk_id for h in results] == [h.chunk_id for h in hits]

    def test_one_empty_one_full(self, search_service: SearchService) -> None:
        """一路空、一路非空时，按非空路单独计算 RRF 分数。"""
        hits = [_make_hit(chunk_id=f"c{i}") for i in range(3)]
        results = search_service._rrf_fusion([hits, []])

        assert len(results) == 3
        for idx, h in enumerate(results):
            assert math.isclose(h.score, _rrf_expected(idx + 1), rel_tol=1e-12)


# ─── 属性测试（Hypothesis）───────────────────────────────────────────


# 单路命中数量上界：与生产保持一致（TOP_K_PER_RETRIEVER=50）
_PER_RETRIEVER_MAX = TOP_K_PER_RETRIEVER


def _retriever_strategy(chunk_pool: list[str]) -> st.SearchStrategy[list[str]]:
    """从给定 chunk_id 池中无放回抽样出一路有序结果。

    单路内 chunk_id 不重复（与真实召回一致），最多 _PER_RETRIEVER_MAX 条。
    """
    return st.lists(
        st.sampled_from(chunk_pool),
        min_size=0,
        max_size=min(_PER_RETRIEVER_MAX, len(chunk_pool)),
        unique=True,
    )


@st.composite
def _recall_results_strategy(draw) -> list[list[SearchHit]]:
    """生成多路召回结果。

    - 1..3 路（覆盖单路、双路、三路）
    - 共享一个有限 chunk_id 池，使 chunk 在不同路间可能重叠
    - 单路内不重复、多路间可能重复
    """
    pool_size = draw(st.integers(min_value=1, max_value=80))
    chunk_pool = [f"c{i}" for i in range(pool_size)]

    n_retrievers = draw(st.integers(min_value=1, max_value=3))
    retrievers: list[list[SearchHit]] = []
    for _ in range(n_retrievers):
        chunk_ids = draw(_retriever_strategy(chunk_pool))
        retrievers.append([_make_hit(chunk_id=cid) for cid in chunk_ids])
    return retrievers


class TestRRFProperties:
    """Hypothesis 属性测试：覆盖输入空间，验证 RRF 算法的不变量。

    Validates: Requirements 6.2
    """

    @given(recall_results=_recall_results_strategy())
    @hyp_settings(
        max_examples=150,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_candidate_count_upper_bound(
        self,
        search_service: SearchService,
        recall_results: list[list[SearchHit]],
    ) -> None:
        """属性：候选数 ≤ min(3 × TOP_K_PER_RETRIEVER, RRF_CANDIDATE_LIMIT)。

        Validates: Requirements 6.2
        """
        results = search_service._rrf_fusion(recall_results)

        # 全局上界：100
        assert len(results) <= RRF_CANDIDATE_LIMIT

        # 输入侧上界：最多 3 路 × 每路 50 条 = 150 条命中，
        # 去重后 ≤ 唯一 chunk_id 总数；同时仍受 RRF_CANDIDATE_LIMIT 约束。
        unique_ids = {hit.chunk_id for r in recall_results for hit in r}
        assert len(results) <= min(
            3 * TOP_K_PER_RETRIEVER,
            RRF_CANDIDATE_LIMIT,
            len(unique_ids),
        )

        # 输出 chunk_id 不重复
        out_ids = [h.chunk_id for h in results]
        assert len(out_ids) == len(set(out_ids))

    @given(recall_results=_recall_results_strategy())
    @hyp_settings(
        max_examples=150,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_ranking_consistent_with_rrf_scores(
        self,
        search_service: SearchService,
        recall_results: list[list[SearchHit]],
    ) -> None:
        """属性：输出按 RRF 分数严格非升序排列，且每条分数等于公式重算值。

        Validates: Requirements 6.2
        """
        results = search_service._rrf_fusion(recall_results)

        # 1) 排名一致性：score 非升序
        for prev, nxt in zip(results, results[1:]):
            assert prev.score >= nxt.score - 1e-12

        # 2) 每条候选的 RRF 分数应等于按公式独立重算的值
        recomputed: dict[str, float] = {}
        for retriever in recall_results:
            for rank, hit in enumerate(retriever, start=1):
                recomputed[hit.chunk_id] = recomputed.get(hit.chunk_id, 0.0) + (
                    1.0 / (RRF_K + rank)
                )

        for hit in results:
            assert math.isclose(
                hit.score, recomputed[hit.chunk_id], rel_tol=1e-9, abs_tol=1e-12
            )
