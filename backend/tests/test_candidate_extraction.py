"""候选术语自动提取测试（任务 13.9）。

覆盖 ``DictionaryService.extract_candidate_terms`` 与
``POST /api/admin/dictionaries/candidates/extract`` 端点：

- 频次阈值（``min_frequency``）下限
- n-gram 长度区间 [``min_length``, ``max_length``]
- 已启用词典中的 terms / stop_words 必须从候选中剔除
- 返回结果按频次降序排列
- ``top_n`` 上限被遵守
- 空文档列表 / 全非中文输入返回空列表
- 控制字符等非法术语被剔除
- 属性测试：返回的所有候选频次 ≥ ``min_frequency``
- 路由鉴权：未登录 401、非管理员 403

Validates: Requirements 20.6
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from app.api.admin_dictionaries import router as admin_dictionaries_router
from app.api.auth import require_admin
from app.core.database import get_db
from app.core.exceptions import (
    ForbiddenException,
    UnauthorizedException,
    register_exception_handlers,
)
from app.models.domain_dictionary import DomainDictionary
from app.services.dictionary_service import DictionaryService


# ─── Helpers ───────────────────────────────────────────────────────────


def _build_dictionary(
    *,
    terms: list | None = None,
    stop_words: list | None = None,
    enabled: bool = True,
) -> DomainDictionary:
    """构造一个填好字段的 DomainDictionary，用于 mock execute 返回。"""
    d = DomainDictionary(
        name=f"test-{uuid.uuid4().hex[:8]}",
        description=None,
        terms=terms if terms is not None else [],
        synonyms=[],
        stop_words=stop_words if stop_words is not None else [],
        enabled=enabled,
    )
    d.id = uuid.uuid4()
    d.created_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    d.updated_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return d


def _mock_db_with_dictionaries(*dictionaries: DomainDictionary) -> AsyncMock:
    """构造一个 ``execute(...)`` 返回指定启用词典列表的 AsyncSession mock。"""
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(dictionaries)
    db.execute = AsyncMock(return_value=result)
    return db


# ─── Service-level extraction tests ────────────────────────────────────


class TestExtractCandidateTerms:
    """``DictionaryService.extract_candidate_terms`` 行为测试。

    全部用 mock DB —— 不依赖真实 PostgreSQL 即可校验过滤/排序逻辑。
    """

    @pytest.mark.asyncio
    async def test_min_frequency_threshold_respected(self):
        """``min_frequency`` 是严格下限，频次低于该值的候选必须被过滤。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        # "水泥" 出现 4 次（4 个 2-gram 滑窗），"钢材" 出现 1 次。
        text = "水泥水泥水泥水泥钢材"
        candidates = await service.extract_candidate_terms(
            documents_content=[text],
            min_frequency=3,
            min_length=2,
            max_length=2,
            top_n=10,
        )

        words = {c["word"] for c in candidates}
        assert "水泥" in words
        assert "钢材" not in words
        assert all(c["frequency"] >= 3 for c in candidates)

    @pytest.mark.asyncio
    async def test_ngram_length_range_respected(self):
        """超出 [``min_length``, ``max_length``] 的 n-gram 必须不出现。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        # 重复一段长文本，让 3-char n-gram 频次足够。
        text = "回转窑设备回转窑设备回转窑设备回转窑设备"
        candidates = await service.extract_candidate_terms(
            documents_content=[text],
            min_frequency=2,
            min_length=3,
            max_length=3,
            top_n=20,
        )

        for c in candidates:
            assert 3 <= len(c["word"]) <= 3, c

    @pytest.mark.asyncio
    async def test_existing_dictionary_terms_excluded(self):
        """命中已启用词典的 ``terms`` / ``stop_words`` 必须从候选中剔除。"""
        existing = _build_dictionary(
            terms=[{"word": "水泥", "pos": "n", "weight": 1.0}],
            stop_words=["钢材"],
        )
        db = _mock_db_with_dictionaries(existing)
        service = DictionaryService(db)

        # 三个 2-char 候选都出现 ≥ 3 次。
        text = "水泥水泥水泥水泥钢材钢材钢材钢材石灰石灰石灰石灰"
        candidates = await service.extract_candidate_terms(
            documents_content=[text],
            min_frequency=3,
            min_length=2,
            max_length=2,
            top_n=20,
        )

        words = {c["word"] for c in candidates}
        # 已存在的术语和停用词被排除，新术语保留
        assert "水泥" not in words
        assert "钢材" not in words
        assert "石灰" in words

    @pytest.mark.asyncio
    async def test_results_sorted_by_frequency_desc(self):
        """结果按频次降序排列。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        # 让 "水泥" 频次显著高于 "石灰"，"钢材" 居中。
        text = (
            "水泥" * 10
            + "石灰" * 5
            + "钢材" * 7
        )
        candidates = await service.extract_candidate_terms(
            documents_content=[text],
            min_frequency=2,
            min_length=2,
            max_length=2,
            top_n=10,
        )

        freqs = [c["frequency"] for c in candidates]
        assert freqs == sorted(freqs, reverse=True), candidates

    @pytest.mark.asyncio
    async def test_top_n_cap_respected(self):
        """返回数量不超过 ``top_n``。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        # 用 8 个互不重叠的 2-char 词汇，每个重复 3 次。
        words = ["水泥", "钢材", "石灰", "黏土", "煤炭", "矿渣", "炉渣", "粉煤"]
        text = "".join(w * 3 for w in words)
        candidates = await service.extract_candidate_terms(
            documents_content=[text],
            min_frequency=3,
            min_length=2,
            max_length=2,
            top_n=3,
        )

        assert len(candidates) <= 3
        # 也应该都满足频次门槛
        assert all(c["frequency"] >= 3 for c in candidates)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self):
        """``documents_content`` 为空列表 → 返回空候选列表。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        candidates = await service.extract_candidate_terms(
            documents_content=[],
            min_frequency=1,
            min_length=2,
            max_length=4,
            top_n=10,
        )
        assert candidates == []

    @pytest.mark.asyncio
    async def test_empty_strings_in_input_return_empty_list(self):
        """全空字符串与 None 元素混合时也返回空候选列表。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        candidates = await service.extract_candidate_terms(
            documents_content=["", "", None],  # type: ignore[list-item]
            min_frequency=1,
            min_length=2,
            max_length=4,
            top_n=10,
        )
        assert candidates == []

    @pytest.mark.asyncio
    async def test_non_chinese_text_returns_empty(self):
        """全非中文文本 → 当前 n-gram 提取实现仅匹配中文字符段，应返回空。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        candidates = await service.extract_candidate_terms(
            documents_content=["this is purely english text", "12345 67890"],
            min_frequency=1,
            min_length=2,
            max_length=5,
            top_n=10,
        )
        assert candidates == []

    @pytest.mark.asyncio
    async def test_invalid_terms_with_control_chars_excluded(self):
        """含控制字符的候选会被 ``validate_term`` 拒掉（理论上 n-gram
        提取只匹配 [\\u4e00-\\u9fff]，不会产生控制字符；但为了防御性，
        我们直接构造一个会出现"看起来像"控制字符的场景：
        在中文段中夹一个 NULL 字节，确认它不会被当成单词的一部分。"""
        db = _mock_db_with_dictionaries()
        service = DictionaryService(db)

        # 中文段被 \\x00 切断，"水泥" 仍能从两侧各自的中文段里提出。
        text = "水泥水泥水泥水泥\x00水泥水泥水泥水泥"
        candidates = await service.extract_candidate_terms(
            documents_content=[text],
            min_frequency=3,
            min_length=2,
            max_length=2,
            top_n=10,
        )
        # 候选词本身不应含控制字符
        for c in candidates:
            assert "\x00" not in c["word"]


# ─── Property: returned candidates always satisfy min_frequency ────────


class TestCandidateExtractionProperty:
    """属性测试：``extract_candidate_terms`` 的输出永远不破坏频次约束。"""

    @hyp_settings(
        max_examples=40,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        # 仅生成中文字符段，避免 Hypothesis 触发非常长的非中文文本路径。
        chinese_block=st.text(
            alphabet=st.characters(min_codepoint=0x4E00, max_codepoint=0x9FFF),
            min_size=0,
            max_size=80,
        ),
        min_freq=st.integers(min_value=1, max_value=8),
        min_len=st.integers(min_value=2, max_value=4),
        max_len_offset=st.integers(min_value=0, max_value=4),
        top_n=st.integers(min_value=1, max_value=20),
    )
    def test_property_never_returns_below_min_frequency(
        self,
        chinese_block: str,
        min_freq: int,
        min_len: int,
        max_len_offset: int,
        top_n: int,
    ) -> None:
        """对任意输入文本和参数组合，返回的所有候选频次 ≥ ``min_frequency``，
        且每个候选的字符长度落在 [``min_length``, ``max_length``] 内。

        Validates: Requirements 20.6
        """
        import asyncio

        max_len = min_len + max_len_offset

        async def _run() -> list[dict]:
            db = _mock_db_with_dictionaries()
            service = DictionaryService(db)
            return await service.extract_candidate_terms(
                documents_content=[chinese_block],
                min_frequency=min_freq,
                min_length=min_len,
                max_length=max_len,
                top_n=top_n,
            )

        candidates = asyncio.run(_run())

        # 永远不超过 top_n
        assert len(candidates) <= top_n
        for c in candidates:
            assert c["frequency"] >= min_freq, c
            assert min_len <= len(c["word"]) <= max_len, c

        # 排序不变量：返回结果按频次降序
        freqs = [c["frequency"] for c in candidates]
        assert freqs == sorted(freqs, reverse=True)


# ─── HTTP route tests: auth on POST /candidates/extract ────────────────


def _build_app_with_auth_failure(mock_db: AsyncMock, exc: Exception) -> TestClient:
    """构造一个 ``require_admin`` 抛 *exc* 的 FastAPI app。"""
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(admin_dictionaries_router)

    async def _override_get_db():
        yield mock_db

    async def _override_require_admin():
        raise exc

    application.dependency_overrides[get_db] = _override_get_db
    application.dependency_overrides[require_admin] = _override_require_admin
    return TestClient(application)


class TestCandidatesExtractAuth:
    """``POST /api/admin/dictionaries/candidates/extract`` 鉴权守门测试。"""

    def test_unauthenticated_returns_401(self, mock_db: AsyncMock) -> None:
        """未登录 → 401。"""
        client = _build_app_with_auth_failure(
            mock_db, UnauthorizedException("缺少认证令牌")
        )
        response = client.post(
            "/api/admin/dictionaries/candidates/extract",
            json={"documents_content": ["水泥水泥水泥"]},
        )
        assert response.status_code == 401, response.text

    def test_non_admin_returns_403(self, mock_db: AsyncMock) -> None:
        """非管理员 → 403。"""
        client = _build_app_with_auth_failure(
            mock_db, ForbiddenException("需要管理员权限")
        )
        response = client.post(
            "/api/admin/dictionaries/candidates/extract",
            json={"documents_content": ["水泥水泥水泥"]},
        )
        assert response.status_code == 403, response.text

    def test_admin_can_call_extract(self, mock_db: AsyncMock) -> None:
        """管理员鉴权通过后路由能成功调用服务层并返回候选列表。"""
        application = FastAPI()
        register_exception_handlers(application)
        application.include_router(admin_dictionaries_router)

        admin_user = MagicMock()
        admin_user.id = uuid.uuid4()
        admin_user.email = "admin@wikforge.local"

        async def _override_get_db():
            yield mock_db

        async def _override_require_admin():
            return admin_user

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin

        # 让 DictionaryService.extract_candidate_terms 的 DB 查询返回空启用列表。
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=result)

        client = TestClient(application)
        response = client.post(
            "/api/admin/dictionaries/candidates/extract",
            json={
                "documents_content": ["水泥水泥水泥水泥"],
                "min_frequency": 3,
                "min_length": 2,
                "max_length": 2,
                "top_n": 5,
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert isinstance(body, list)
        # "水泥" 在 8 个字符的串里有 7 个 2-gram，其中 "水泥" 出现 4 次。
        words = {item["word"] for item in body}
        assert "水泥" in words
        for item in body:
            assert item["frequency"] >= 3
