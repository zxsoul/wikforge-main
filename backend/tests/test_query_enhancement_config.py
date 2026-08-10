"""任务 15.6 / 需求 7.6：查询增强开关配置测试。

覆盖以下行为：

- 三个环境变量驱动的开关默认 True，与 ``QueryEnhancerConfig`` 默认值对齐
- ``QueryEnhancerConfig.from_settings()`` 能从任意 ``Settings``-like 对象
  读取并映射到三个开关
- ``build_query_enhancer()`` 工厂可注入测试 ``Settings``，并返回带正确
  ``config`` 的 :class:`QueryEnhancer` 实例
- 端到端：通过环境变量关闭某个子模块后，``QueryEnhancer.enhance()``
  不会再触发对应的 LLM / Embedding 调用
- ``app/api/search.py`` 中的依赖工厂 :func:`get_query_enhancer` 也消费
  同一套配置（与运行时入口保持一致）
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.services.query_enhancer import (
    QueryEnhancer,
    QueryEnhancerConfig,
    build_query_enhancer,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_gateway() -> AsyncMock:
    """LLM 网关 mock，单次完成时返回固定文本。"""
    gateway = AsyncMock()
    response = MagicMock()
    response.content = "改写A\n改写B"
    gateway.complete = AsyncMock(return_value=response)
    return gateway


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """嵌入服务 mock，dense_vector 为 1024 维占位。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(
        return_value=MagicMock(
            dense_vector=[0.01] * 1024,
            sparse_indices=[1, 2, 3],
            sparse_values=[0.5, 0.3, 0.2],
        )
    )
    return service


# ─── Settings 默认值 / 字段存在性 ─────────────────────────────────────


class TestSettingsDefaults:
    """Settings 中的查询增强开关默认值。"""

    def test_settings_has_three_toggle_fields(self):
        """Settings 应暴露三个 ``QUERY_ENHANCEMENT_ENABLE_*`` 字段。"""
        settings = Settings(_env_file=None)

        assert hasattr(settings, "QUERY_ENHANCEMENT_ENABLE_REWRITE")
        assert hasattr(settings, "QUERY_ENHANCEMENT_ENABLE_HYDE")
        assert hasattr(settings, "QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION")

    def test_settings_defaults_all_true(self):
        """三个开关默认全部 True，符合需求 7.1/7.2/7.3 的"默认启用"语义。"""
        settings = Settings(_env_file=None)

        assert settings.QUERY_ENHANCEMENT_ENABLE_REWRITE is True
        assert settings.QUERY_ENHANCEMENT_ENABLE_HYDE is True
        assert settings.QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION is True

    @pytest.mark.parametrize(
        "rewrite, hyde, decompose",
        [
            (False, True, True),
            (True, False, True),
            (True, True, False),
            (False, False, False),
        ],
    )
    def test_settings_reads_env_vars(
        self, monkeypatch, rewrite: bool, hyde: bool, decompose: bool
    ):
        """三个开关应分别由对应的环境变量驱动，互不干扰。"""
        monkeypatch.setenv(
            "QUERY_ENHANCEMENT_ENABLE_REWRITE", "true" if rewrite else "false"
        )
        monkeypatch.setenv(
            "QUERY_ENHANCEMENT_ENABLE_HYDE", "true" if hyde else "false"
        )
        monkeypatch.setenv(
            "QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION",
            "true" if decompose else "false",
        )

        settings = Settings(_env_file=None)

        assert settings.QUERY_ENHANCEMENT_ENABLE_REWRITE is rewrite
        assert settings.QUERY_ENHANCEMENT_ENABLE_HYDE is hyde
        assert settings.QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION is decompose


# ─── QueryEnhancerConfig.from_settings ────────────────────────────────


class TestQueryEnhancerConfigFromSettings:
    """``QueryEnhancerConfig.from_settings`` 的字段映射。"""

    def test_maps_all_three_fields(self):
        """给定 SimpleNamespace 也能映射出对应配置。"""
        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=False,
            QUERY_ENHANCEMENT_ENABLE_HYDE=True,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=False,
        )

        config = QueryEnhancerConfig.from_settings(fake_settings)

        assert isinstance(config, QueryEnhancerConfig)
        assert config.enable_rewrite is False
        assert config.enable_hyde is True
        assert config.enable_decomposition is False

    def test_missing_fields_default_to_true(self):
        """Settings 上若没有相应字段（极端兼容路径），应回退为 True。"""
        empty = SimpleNamespace()

        config = QueryEnhancerConfig.from_settings(empty)

        assert config.enable_rewrite is True
        assert config.enable_hyde is True
        assert config.enable_decomposition is True

    def test_uses_real_settings_when_no_arg(self, monkeypatch):
        """未传 settings 时应通过 :func:`get_settings` 读取真实环境。"""
        # 显式覆盖三个环境变量，并清掉 lru_cache，确保读到测试值
        monkeypatch.setenv("QUERY_ENHANCEMENT_ENABLE_REWRITE", "false")
        monkeypatch.setenv("QUERY_ENHANCEMENT_ENABLE_HYDE", "false")
        monkeypatch.setenv("QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION", "false")

        from app.core.config import get_settings

        get_settings.cache_clear()
        try:
            config = QueryEnhancerConfig.from_settings()
        finally:
            get_settings.cache_clear()

        assert config.enable_rewrite is False
        assert config.enable_hyde is False
        assert config.enable_decomposition is False


# ─── build_query_enhancer 工厂 ─────────────────────────────────────────


class TestBuildQueryEnhancer:
    """``build_query_enhancer`` 工厂行为。"""

    def test_returns_query_enhancer_with_config_from_settings(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """工厂应把 Settings 上的开关注入到返回的 ``QueryEnhancer.config``。"""
        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=True,
            QUERY_ENHANCEMENT_ENABLE_HYDE=False,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=True,
        )

        enhancer = build_query_enhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            settings=fake_settings,
        )

        assert isinstance(enhancer, QueryEnhancer)
        assert enhancer.config.enable_rewrite is True
        assert enhancer.config.enable_hyde is False
        assert enhancer.config.enable_decomposition is True

    def test_returns_query_enhancer_with_all_disabled_when_settings_off(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """三个开关全部 False 时，返回的增强器三项都关闭。"""
        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=False,
            QUERY_ENHANCEMENT_ENABLE_HYDE=False,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=False,
        )

        enhancer = build_query_enhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            settings=fake_settings,
        )

        assert enhancer.config.enable_rewrite is False
        assert enhancer.config.enable_hyde is False
        assert enhancer.config.enable_decomposition is False


# ─── 端到端：开关真正影响 enhance() 行为 ──────────────────────────────


class TestEnhanceRespectsSettingsToggles:
    """通过工厂注入 Settings 关闭子模块后，``enhance()`` 不再触发对应调用。"""

    @pytest.mark.asyncio
    async def test_all_disabled_no_llm_or_embedding_calls(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """全部关闭时 ``enhance()`` 既不调用 LLM 也不调用 Embedding。"""
        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=False,
            QUERY_ENHANCEMENT_ENABLE_HYDE=False,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=False,
        )
        enhancer = build_query_enhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            settings=fake_settings,
        )

        result = await enhancer.enhance("你好世界")

        # 仅原始查询保留（需求 7.4）
        assert result.original == "你好世界"
        assert result.variants == []
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        # 关键断言：所有外部依赖均未被调用
        mock_llm_gateway.complete.assert_not_called()
        mock_embedding_service.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_only_rewrite_via_settings(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """通过 Settings 仅启用改写时，不会触发 HyDE 嵌入。"""
        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=True,
            QUERY_ENHANCEMENT_ENABLE_HYDE=False,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=False,
        )
        enhancer = build_query_enhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            settings=fake_settings,
        )

        result = await enhancer.enhance("机器学习")

        assert result.variants == ["改写A", "改写B"]
        assert result.hyde_embeddings == []
        assert result.sub_queries == []
        # 仅 rewrite 一次 LLM 调用
        assert mock_llm_gateway.complete.await_count == 1
        # HyDE 关闭意味着不会执行 embed_query
        mock_embedding_service.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_disable_only_hyde_via_settings(
        self, mock_llm_gateway, mock_embedding_service
    ):
        """关闭 HyDE 后，rewrite 与 decomposition 仍执行；embed_query 不被调用。"""
        # 让分解模块返回 ≥ 2 条子查询，否则会被 ``< 2`` 的去抖逻辑视为单一问题
        decompose_response = MagicMock()
        decompose_response.content = "子查询1\n子查询2\n子查询3"
        rewrite_response = MagicMock()
        rewrite_response.content = "改写A\n改写B"

        # complete 被多个并发任务调用，使用 side_effect 兜两次返回不同内容
        mock_llm_gateway.complete = AsyncMock(
            side_effect=[rewrite_response, decompose_response]
        )

        fake_settings = SimpleNamespace(
            QUERY_ENHANCEMENT_ENABLE_REWRITE=True,
            QUERY_ENHANCEMENT_ENABLE_HYDE=False,
            QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION=True,
        )
        enhancer = build_query_enhancer(
            llm_gateway=mock_llm_gateway,
            embedding_service=mock_embedding_service,
            settings=fake_settings,
        )

        result = await enhancer.enhance("查询A及查询B")

        assert result.original == "查询A及查询B"
        assert result.variants == ["改写A", "改写B"]
        # HyDE 已关闭：embed_query 一次都不应被调用
        mock_embedding_service.embed_query.assert_not_called()
        assert result.hyde_embeddings == []
        # 分解仍运行
        assert len(result.sub_queries) == 3
        # rewrite + decompose 共两次 LLM 调用
        assert mock_llm_gateway.complete.await_count == 2


# ─── search API 依赖工厂 ───────────────────────────────────────────────


class TestSearchApiDependency:
    """``app/api/search.py`` 的依赖工厂同样消费 Settings 配置。"""

    @pytest.mark.asyncio
    async def test_get_query_enhancer_returns_configured_instance(
        self, monkeypatch
    ):
        """``get_query_enhancer`` 应返回按当前 Settings 配置好的增强器。"""
        # 通过环境变量驱动，避免直接打补丁 build_query_enhancer 的内部
        monkeypatch.setenv("QUERY_ENHANCEMENT_ENABLE_REWRITE", "false")
        monkeypatch.setenv("QUERY_ENHANCEMENT_ENABLE_HYDE", "true")
        monkeypatch.setenv("QUERY_ENHANCEMENT_ENABLE_DECOMPOSITION", "false")

        from app.api.search import get_query_enhancer
        from app.core.config import get_settings

        get_settings.cache_clear()
        try:
            enhancer = await get_query_enhancer()
        finally:
            get_settings.cache_clear()

        assert isinstance(enhancer, QueryEnhancer)
        assert enhancer.config.enable_rewrite is False
        assert enhancer.config.enable_hyde is True
        assert enhancer.config.enable_decomposition is False
