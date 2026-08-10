"""权限 Filter 构建逻辑测试。

对应任务 14.4：实现权限 Filter 构建逻辑（根据用户可访问空间列表构建
Qdrant/OpenSearch filter）。

来源：``app.services.search_service.SearchService``
- ``_build_qdrant_filter(user_id, allowed_space_ids) -> dict``
- ``_build_opensearch_filter(user_id, allowed_space_ids) -> dict``
- 辅助：``_dict_to_qdrant_filter(filter_dict) -> qdrant_client.models.Filter``

ABAC 语义（_Requirements: 6.5_）：
当 ``allowed_user_ids`` 命中当前用户 ID **或** ``space_id`` 命中任一可访问
空间时，文档块对该用户可见。两个搜索后端必须保持等价的 OR 语义：
- Qdrant 通过 ``Filter(should=[...])``（无 ``must``）达到至少匹配一项
- OpenSearch 通过 ``bool.minimum_should_match=1`` 显式声明
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.services.search_service import SearchService


@pytest.fixture
def search_service() -> SearchService:
    """无 embedding 依赖的 SearchService（filter 构建是纯函数行为）。"""
    return SearchService(embedding_service=MagicMock())


# ─── Qdrant Filter ────────────────────────────────────────────────────


class TestQdrantFilterBuilder:
    """``_build_qdrant_filter`` 的结构与语义。"""

    def test_returns_dict_with_should_clauses(
        self, search_service: SearchService
    ) -> None:
        """返回 dict 顶层包含 ``should`` 列表。"""
        f = search_service._build_qdrant_filter("u1", ["s1"])
        assert "should" in f
        assert isinstance(f["should"], list)

    def test_contains_user_id_match_value_clause(
        self, search_service: SearchService
    ) -> None:
        """``allowed_user_ids`` 字段使用 MatchValue 精确匹配当前用户 ID。"""
        user_id = str(uuid.uuid4())
        f = search_service._build_qdrant_filter(user_id, ["s1"])
        user_clauses = [
            c for c in f["should"] if c["key"] == "allowed_user_ids"
        ]
        assert len(user_clauses) == 1
        assert user_clauses[0]["match"] == {"value": user_id}

    def test_contains_space_id_match_any_clause(
        self, search_service: SearchService
    ) -> None:
        """``space_id`` 字段使用 MatchAny 匹配可访问空间 ID 集合。"""
        space_ids = [str(uuid.uuid4()) for _ in range(3)]
        f = search_service._build_qdrant_filter("u1", space_ids)
        space_clauses = [c for c in f["should"] if c["key"] == "space_id"]
        assert len(space_clauses) == 1
        assert space_clauses[0]["match"] == {"any": space_ids}

    def test_no_must_or_must_not_clauses(
        self, search_service: SearchService
    ) -> None:
        """ABAC 仅使用 OR 语义，不应包含 ``must``/``must_not`` 限制。

        若存在 ``must``，则两条 should 子句都必须命中才能放行，与设计中
        的 OR 关系不一致，会丢失所有继承自空间权限的文档块。
        """
        f = search_service._build_qdrant_filter("u1", ["s1"])
        assert "must" not in f
        assert "must_not" not in f

    def test_empty_space_ids_still_includes_user_clause(
        self, search_service: SearchService
    ) -> None:
        """``allowed_space_ids`` 为空时仍应保留 user_id 子句，避免误锁。"""
        user_id = "user-empty-spaces"
        f = search_service._build_qdrant_filter(user_id, [])
        user_clauses = [
            c for c in f["should"] if c["key"] == "allowed_user_ids"
        ]
        assert len(user_clauses) == 1
        assert user_clauses[0]["match"] == {"value": user_id}

    def test_dict_to_qdrant_filter_converts_to_filter_object(
        self, search_service: SearchService
    ) -> None:
        """dict 形式可被 ``_dict_to_qdrant_filter`` 转换为 Qdrant Filter 对象。

        转换后：
        - 类型为 ``Filter``，``must`` 为空（保持 OR 语义）
        - ``should`` 含 2 个 ``FieldCondition``
        - 一个使用 ``MatchValue(user_id)``，一个使用 ``MatchAny(space_ids)``
        """
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        user_id = "u1"
        space_ids = ["s1", "s2"]

        d = search_service._build_qdrant_filter(user_id, space_ids)
        qf = search_service._dict_to_qdrant_filter(d)

        assert isinstance(qf, Filter)
        assert not qf.must  # None 或 空列表
        assert not qf.must_not
        assert qf.should is not None
        assert len(qf.should) == 2

        match_values = [
            c.match
            for c in qf.should
            if isinstance(c, FieldCondition)
            and isinstance(c.match, MatchValue)
        ]
        assert len(match_values) == 1
        assert match_values[0].value == user_id

        match_anys = [
            c.match
            for c in qf.should
            if isinstance(c, FieldCondition)
            and isinstance(c.match, MatchAny)
        ]
        assert len(match_anys) == 1
        assert match_anys[0].any == space_ids


# ─── OpenSearch Filter ────────────────────────────────────────────────


class TestOpenSearchFilterBuilder:
    """``_build_opensearch_filter`` 的结构与语义。"""

    def test_returns_bool_should_with_min_match_one(
        self, search_service: SearchService
    ) -> None:
        """返回结构：``{"bool": {"should": [...], "minimum_should_match": 1}}``。"""
        f = search_service._build_opensearch_filter("u1", ["s1"])
        assert "bool" in f
        assert "should" in f["bool"]
        assert isinstance(f["bool"]["should"], list)
        # OR 语义需要 minimum_should_match=1（OpenSearch 在有 must 同级时
        # should 默认 0，不强制可能放行无权限文档）
        assert f["bool"]["minimum_should_match"] == 1

    def test_uses_term_for_user_id_match(
        self, search_service: SearchService
    ) -> None:
        """``allowed_user_ids`` 字段使用 ``term`` 精确匹配当前用户 ID。"""
        user_id = str(uuid.uuid4())
        f = search_service._build_opensearch_filter(user_id, ["s1"])
        term_clauses = [c for c in f["bool"]["should"] if "term" in c]
        assert len(term_clauses) == 1
        assert term_clauses[0]["term"] == {"allowed_user_ids": user_id}

    def test_uses_terms_for_space_id_set_match(
        self, search_service: SearchService
    ) -> None:
        """``space_id`` 字段使用 ``terms`` 匹配可访问空间集合。"""
        space_ids = [str(uuid.uuid4()) for _ in range(3)]
        f = search_service._build_opensearch_filter("u1", space_ids)
        terms_clauses = [c for c in f["bool"]["should"] if "terms" in c]
        assert len(terms_clauses) == 1
        assert terms_clauses[0]["terms"] == {"space_id": space_ids}

    def test_empty_space_ids_still_includes_user_term(
        self, search_service: SearchService
    ) -> None:
        """空间列表为空时仍保留 user_id term，且 ``minimum_should_match=1``
        保证仅靠 user_id 即可匹配。"""
        user_id = "user-empty"
        f = search_service._build_opensearch_filter(user_id, [])

        term_clauses = [c for c in f["bool"]["should"] if "term" in c]
        assert len(term_clauses) == 1
        assert term_clauses[0]["term"] == {"allowed_user_ids": user_id}
        assert f["bool"]["minimum_should_match"] == 1


# ─── 双后端语义等价 ──────────────────────────────────────────────────


class TestFilterSemanticEquivalence:
    """Qdrant 与 OpenSearch filter 在 ABAC 语义上保持等价。"""

    def test_both_have_two_should_clauses(
        self, search_service: SearchService
    ) -> None:
        """两个后端都生成恰好 2 条 should 子句（user_id + space_id）。"""
        user_id = "u1"
        space_ids = ["s1", "s2"]
        q = search_service._build_qdrant_filter(user_id, space_ids)
        o = search_service._build_opensearch_filter(user_id, space_ids)
        assert len(q["should"]) == 2
        assert len(o["bool"]["should"]) == 2

    def test_both_implement_at_least_one_match(
        self, search_service: SearchService
    ) -> None:
        """OR 语义在两端都体现为 “至少命中一条 should”。

        - OpenSearch：显式 ``minimum_should_match=1``。
        - Qdrant：``Filter`` 仅有 ``should`` 而无 ``must``，等价于至少一项命中。
        """
        from qdrant_client.models import Filter

        user_id = "u1"
        space_ids = ["s1"]

        o = search_service._build_opensearch_filter(user_id, space_ids)
        assert o["bool"]["minimum_should_match"] == 1

        q = search_service._build_qdrant_filter(user_id, space_ids)
        qf = search_service._dict_to_qdrant_filter(q)
        assert isinstance(qf, Filter)
        # 没有 must 子句意味着只要 should 命中即可放行
        assert not qf.must

    def test_both_extract_user_id_from_same_input(
        self, search_service: SearchService
    ) -> None:
        """两端都用相同的 user_id 做精确匹配。"""
        user_id = "user-xyz"
        space_ids = ["sa", "sb"]
        q = search_service._build_qdrant_filter(user_id, space_ids)
        o = search_service._build_opensearch_filter(user_id, space_ids)

        q_user_vals = [
            c["match"]["value"]
            for c in q["should"]
            if c["key"] == "allowed_user_ids"
        ]
        o_user_vals = [
            c["term"]["allowed_user_ids"]
            for c in o["bool"]["should"]
            if "term" in c
        ]
        assert q_user_vals == [user_id]
        assert o_user_vals == [user_id]

    def test_both_extract_space_ids_from_same_input(
        self, search_service: SearchService
    ) -> None:
        """两端都用相同的 space_ids 集合做集合匹配。"""
        user_id = "u"
        space_ids = ["a", "b", "c"]
        q = search_service._build_qdrant_filter(user_id, space_ids)
        o = search_service._build_opensearch_filter(user_id, space_ids)

        q_spaces = [
            c["match"]["any"]
            for c in q["should"]
            if c["key"] == "space_id"
        ]
        o_spaces = [
            c["terms"]["space_id"]
            for c in o["bool"]["should"]
            if "terms" in c
        ]
        assert q_spaces == [space_ids]
        assert o_spaces == [space_ids]
