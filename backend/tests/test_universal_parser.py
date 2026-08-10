"""Tests for the Universal Parser and LLM Gateway.

Tests:
- LLMGateway initialization and configuration
- UniversalParser.parse with mocked LLM responses
- Degradation on LLM failure (fallback to plain text)
- Page merging logic (cross-page tables, deduplication)
- Candidate profile suggestion
- Trigger condition evaluation
"""

import asyncio
import inspect
import re
import subprocess
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Inject a stub ``pdf2image`` module before importing the parser so test machines
# without Poppler/pdf2image installed can still resolve the patches we install
# via ``patch("pdf2image.convert_from_path", ...)``. The real module — when
# present — is preserved unchanged; this only fills in the gap.
if "pdf2image" not in sys.modules:
    _pdf2image_stub = ModuleType("pdf2image")
    _pdf2image_stub.convert_from_path = lambda *a, **kw: []  # type: ignore[attr-defined]
    sys.modules["pdf2image"] = _pdf2image_stub

from app.services.llm_gateway import LLMGateway, LLMGatewayError, LLMResponse  # noqa: E402
from app.services.parsers.base import Block, ParsedDocument  # noqa: E402
from app.services.universal_parser import PageResult, UniversalParser  # noqa: E402

# ─── LLMGateway Tests ─────────────────────────────────────────────────


class TestLLMGatewayInit:
    """Test LLMGateway initialization and configuration."""

    @patch("app.services.llm_gateway.get_settings")
    def test_default_initialization(self, mock_settings):
        """LLMGateway uses settings defaults when no args provided."""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="https://api.example.com",
            LITELLM_API_KEY="test-key-123",
        )

        gateway = LLMGateway()

        assert gateway.model == "gpt-4o"
        assert gateway.api_base == "https://api.example.com"
        assert gateway.api_key == "test-key-123"
        assert gateway.timeout == 60.0

    @patch("app.services.llm_gateway.get_settings")
    def test_custom_initialization(self, mock_settings):
        """LLMGateway uses provided args over settings."""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="",
            LITELLM_API_KEY="",
        )

        gateway = LLMGateway(
            model="qwen-vl-max",
            api_base="https://custom.api.com",
            api_key="custom-key",
            timeout=30.0,
        )

        assert gateway.model == "qwen-vl-max"
        assert gateway.api_base == "https://custom.api.com"
        assert gateway.api_key == "custom-key"
        assert gateway.timeout == 30.0

    @patch("app.services.llm_gateway.get_settings")
    def test_multimodal_models_list(self, mock_settings):
        """LLMGateway has a list of known multimodal models."""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            LITELLM_API_BASE="",
            LITELLM_API_KEY="",
        )

        gateway = LLMGateway()
        assert "gpt-4o" in gateway.MULTIMODAL_MODELS
        assert "qwen-vl-max" in gateway.MULTIMODAL_MODELS
        assert "claude-3-5-sonnet-20241022" in gateway.MULTIMODAL_MODELS


# ─── UniversalParser Tests ────────────────────────────────────────────


class TestUniversalParserScaffolding:
    """任务 10.1：UniversalParser 组件骨架与依赖注入。

    这些测试只验证组件最基础的契约，是后续 10.2 ~ 10.10 子任务实现的脚手架：
    - 类的存在与导入路径
    - 公共 API 方法存在并具备正确的签名（async + 类型提示）
    - LLM_Gateway 通过构造函数注入
    - 默认模型与超时来自全局 ``Settings`` / 显式参数
    - 内部子例程已经声明（10.2 ~ 10.8 实现），不抛 AttributeError
    """

    @patch("app.services.universal_parser.get_settings")
    def test_constructor_injects_llm_gateway(self, mock_settings):
        """构造函数接受 LLMGateway 依赖注入。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
        )
        gateway = MagicMock(spec=LLMGateway)

        parser = UniversalParser(llm_gateway=gateway)

        assert parser.llm is gateway
        assert parser.model == "gpt-4o"
        assert parser.page_timeout == 60.0

    @patch("app.services.universal_parser.LLMGateway")
    @patch("app.services.universal_parser.get_settings")
    def test_constructor_creates_default_gateway(self, mock_settings, mock_gateway_cls):
        """未传入 gateway 时，使用 Settings 中的默认模型创建 LLMGateway。"""
        mock_settings.return_value = MagicMock(LITELLM_MODEL="qwen-vl-max")
        sentinel_gateway = MagicMock(spec=LLMGateway)
        mock_gateway_cls.return_value = sentinel_gateway

        parser = UniversalParser()

        # 默认模型来自 settings
        assert parser.model == "qwen-vl-max"
        # 默认走 LLMGateway 工厂构造，并把模型 / 超时透传过去
        mock_gateway_cls.assert_called_once_with(model="qwen-vl-max", timeout=60.0)
        assert parser.llm is sentinel_gateway

    @patch("app.services.universal_parser.get_settings")
    def test_constructor_supports_explicit_model_and_timeout(self, mock_settings):
        """显式传入的模型与超时优先于 Settings。"""
        mock_settings.return_value = MagicMock(LITELLM_MODEL="gpt-4o")
        gateway = MagicMock(spec=LLMGateway)

        parser = UniversalParser(
            llm_gateway=gateway,
            model="minicpm-v",
            vision_model="qwen-vl-max",
            page_timeout=30.0,
        )

        assert parser.model == "minicpm-v"
        assert parser.vision_model == "qwen-vl-max"
        assert parser.page_timeout == 30.0

    @patch("app.services.universal_parser.get_settings")
    def test_public_api_signatures(self, mock_settings):
        """公共 API parse / suggest_profile / should_trigger 必须存在且签名正确。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        # parse 必须是协程，签名为 (parsed_doc) -> ProcessedDocument
        assert inspect.iscoroutinefunction(parser.parse)
        sig = inspect.signature(parser.parse)
        assert list(sig.parameters) == ["parsed_doc"]

        # suggest_profile 必须是协程
        assert inspect.iscoroutinefunction(parser.suggest_profile)
        sig = inspect.signature(parser.suggest_profile)
        assert list(sig.parameters) == ["result"]

        # should_trigger 必须是同步函数（在管线决策时使用）
        assert not inspect.iscoroutinefunction(parser.should_trigger)
        sig = inspect.signature(parser.should_trigger)
        assert {"profile_matched", "quality_score", "threshold"} <= set(sig.parameters)

    @patch("app.services.universal_parser.get_settings")
    def test_internal_subroutines_exist(self, mock_settings):
        """内部子例程（10.2 ~ 10.8 实现位）已声明，确保后续子任务无重大重构。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        # 10.2 按页转图
        assert callable(parser._get_page_image)
        assert callable(parser._pdf_page_to_image)
        assert callable(parser._office_page_to_image)
        # 10.3 逐页多模态调用
        assert inspect.iscoroutinefunction(parser._parse_page)
        # 10.4 页面结果合并
        assert callable(parser._merge_page_results)
        assert callable(parser._merge_cross_page_tables)
        # 10.5 候选 Profile 生成
        assert callable(parser._extract_heading_patterns)
        assert callable(parser._extract_noise_patterns)
        assert callable(parser._recommend_chunking)
        # 10.8 失败降级
        assert callable(parser._degrade_to_plain_text)

    def test_structured_document_alias_exported(self):
        """``StructuredDocument`` 别名指向 Task 9 的 ProcessedDocument，
        让后续子任务可以使用 design.md 中的命名。"""
        from app.services.document_processor import ProcessedDocument
        from app.services.universal_parser import StructuredDocument

        assert StructuredDocument is ProcessedDocument


class TestUniversalParserParse:
    """Test UniversalParser.parse with mocked LLM responses."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mocked LLM gateway."""
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock()
        gateway.complete_multimodal = AsyncMock()
        return gateway

    @pytest.fixture
    def sample_parsed_doc(self):
        """Create a sample ParsedDocument for testing."""
        return ParsedDocument(
            blocks=[
                Block(type="heading", text="Chapter 1: Introduction", page_number=1),
                Block(type="paragraph", text="This is the introduction text.", page_number=1),
                Block(type="paragraph", text="More content here.", page_number=1),
                Block(type="heading", text="Chapter 2: Methods", page_number=2),
                Block(type="paragraph", text="Method description goes here.", page_number=2),
            ],
            metadata={"file_type": "docx", "page_count": 2},
            assets=[],
        )

    @pytest.mark.asyncio
    async def test_parse_success(self, mock_gateway, sample_parsed_doc):
        """UniversalParser.parse returns structured document on LLM success."""
        # Mock LLM responses for each page
        mock_gateway.complete.side_effect = [
            LLMResponse(
                content="# Chapter 1: Introduction\n\nThis is the introduction text.\n\nMore content here.",
                model="gpt-4o",
                usage={"total_tokens": 100},
            ),
            LLMResponse(
                content="# Chapter 2: Methods\n\nMethod description goes here.",
                model="gpt-4o",
                usage={"total_tokens": 80},
            ),
        ]

        parser = UniversalParser(llm_gateway=mock_gateway)
        result = await parser.parse(sample_parsed_doc)

        assert result.markdown != ""
        assert result.headings_detected >= 2
        assert len(result.blocks) > 0
        # Verify LLM was called for each page
        assert mock_gateway.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_parse_empty_document(self, mock_gateway):
        """UniversalParser.parse handles empty documents."""
        empty_doc = ParsedDocument(blocks=[], metadata={"file_type": "pdf"})

        parser = UniversalParser(llm_gateway=mock_gateway)
        result = await parser.parse(empty_doc)

        assert result.blocks == []
        assert result.markdown == ""
        mock_gateway.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_parse_with_multimodal(self, mock_gateway):
        """UniversalParser uses multimodal call when page image is available."""
        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="Some text", page_number=1),
            ],
            metadata={"file_type": "pdf", "file_path": "/tmp/test.pdf"},
        )

        mock_gateway.complete_multimodal.return_value = LLMResponse(
            content="# Title\n\nSome text content.",
            model="gpt-4o",
        )

        parser = UniversalParser(llm_gateway=mock_gateway)

        # Patch the page-to-image extraction to return fake image bytes,
        # bypassing the on-disk PDF and pdf2image dependency.
        with patch(
            "app.services.universal_parser.UniversalParser._get_page_image",
            return_value=b"fake-png-data",
        ):
            result = await parser.parse(doc)

        # Should have used multimodal call
        mock_gateway.complete_multimodal.assert_called_once()
        assert result.markdown != ""


class TestUniversalParserDegradation:
    """Test degradation behavior on LLM failure."""

    @pytest.fixture
    def failing_gateway(self):
        """Create a gateway that always fails."""
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock(
            side_effect=LLMGatewayError("Service unavailable", reason="timeout")
        )
        gateway.complete_multimodal = AsyncMock(
            side_effect=LLMGatewayError("Service unavailable", reason="timeout")
        )
        return gateway

    @pytest.mark.asyncio
    async def test_degrade_on_llm_failure(self, failing_gateway):
        """On LLM failure, falls back to plain text + fixed-size chunks."""
        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="A" * 600, page_number=1),
                Block(type="paragraph", text="B" * 400, page_number=2),
            ],
            metadata={"file_type": "docx"},
        )

        parser = UniversalParser(llm_gateway=failing_gateway)
        result = await parser.parse(doc)

        # Should still produce output (degraded)
        assert result.blocks != []
        assert result.markdown != ""
        # Headings won't be detected in degraded mode
        assert result.headings_detected == 0

    @pytest.mark.asyncio
    async def test_degrade_preserves_text(self, failing_gateway):
        """Degraded mode preserves all text content."""
        original_text = "Important document content that must be preserved."
        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text=original_text, page_number=1),
            ],
            metadata={},
        )

        parser = UniversalParser(llm_gateway=failing_gateway)
        result = await parser.parse(doc)

        assert original_text in result.markdown


class TestPageMerging:
    """Test page result merging logic."""

    @pytest.fixture
    def mock_gateway(self):
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock()
        return gateway

    @pytest.mark.asyncio
    async def test_merge_cross_page_tables(self, mock_gateway):
        """Tables with same header on adjacent pages are merged."""
        # Simulate LLM returning tables that span pages
        mock_gateway.complete.side_effect = [
            LLMResponse(
                content="# Data\n\n| Name | Value |\n| --- | --- |\n| A | 1 |",
                model="gpt-4o",
            ),
            LLMResponse(
                content="| Name | Value |\n| --- | --- |\n| B | 2 |\n| C | 3 |",
                model="gpt-4o",
            ),
        ]

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="Data table", page_number=1),
                Block(type="paragraph", text="Table continued", page_number=2),
            ],
            metadata={},
        )

        parser = UniversalParser(llm_gateway=mock_gateway)
        result = await parser.parse(doc)

        # Find table blocks
        table_blocks = [b for b in result.blocks if b.type == "table"]
        # The two tables with same header should be merged into one
        assert len(table_blocks) == 1
        # Merged table should contain data from both pages
        assert "A" in table_blocks[0].text
        assert "B" in table_blocks[0].text
        assert "C" in table_blocks[0].text

    @pytest.mark.asyncio
    async def test_deduplicate_repeated_paragraphs(self, mock_gateway):
        """Short repeated paragraphs (noise) are deduplicated."""
        # Simulate repeated header/footer text across pages
        mock_gateway.complete.side_effect = [
            LLMResponse(
                content="Company Confidential\n\n# Chapter 1\n\nContent page 1.",
                model="gpt-4o",
            ),
            LLMResponse(
                content="Company Confidential\n\n# Chapter 2\n\nContent page 2.",
                model="gpt-4o",
            ),
        ]

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="Page 1 text", page_number=1),
                Block(type="paragraph", text="Page 2 text", page_number=2),
            ],
            metadata={},
        )

        parser = UniversalParser(llm_gateway=mock_gateway)
        result = await parser.parse(doc)

        # "Company Confidential" should appear only once
        confidential_count = sum(
            1 for b in result.blocks
            if "company confidential" in b.text.lower()
        )
        assert confidential_count == 1

    @pytest.mark.asyncio
    async def test_preserve_page_numbers(self, mock_gateway):
        """Page numbers are preserved in merged results."""
        mock_gateway.complete.side_effect = [
            LLMResponse(content="# Page 1 Heading\n\nPage 1 content.", model="gpt-4o"),
            LLMResponse(content="# Page 2 Heading\n\nPage 2 content.", model="gpt-4o"),
            LLMResponse(content="# Page 3 Heading\n\nPage 3 content.", model="gpt-4o"),
        ]

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="Text 1", page_number=1),
                Block(type="paragraph", text="Text 2", page_number=2),
                Block(type="paragraph", text="Text 3", page_number=3),
            ],
            metadata={},
        )

        parser = UniversalParser(llm_gateway=mock_gateway)
        result = await parser.parse(doc)

        # Blocks should have page numbers from 1 to 3
        page_numbers = {b.page_number for b in result.blocks}
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers


class TestSuggestProfile:
    """Test candidate profile suggestion (基础 envelope 契约)."""

    @pytest.fixture
    def mock_gateway(self):
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock()
        return gateway

    @pytest.mark.asyncio
    async def test_suggest_profile_basic(self, mock_gateway):
        """suggest_profile returns the candidate envelope with profile + metadata."""
        from app.services.document_processor import ProcessedBlock, ProcessedDocument

        result = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、总则", heading_level=1, page_number=1),
                ProcessedBlock(type="paragraph", text="Content under heading.", page_number=1),
                ProcessedBlock(type="heading", text="二、范围", heading_level=1, page_number=2),
                ProcessedBlock(type="paragraph", text="More content.", page_number=2),
            ],
            metadata={},
            markdown="# 一、总则\n\nContent.\n\n# 二、范围\n\nMore content.",
        )

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(result)

        # Top-level envelope shape (任务 10.5 → 10.6 storage contract).
        assert set(candidate.keys()) == {"profile", "metadata"}

        profile = candidate["profile"]
        assert profile["enabled"] is False
        assert "heading_rules" in profile
        assert "boilerplate" in profile
        assert "chunking" in profile
        assert "tables" in profile

        meta = candidate["metadata"]
        assert meta["status"] == "pending_approval"
        assert meta["source"] == "universal_parser"

    @pytest.mark.asyncio
    async def test_suggest_profile_detects_chinese_numbering(self, mock_gateway):
        """suggest_profile detects Chinese numbering patterns."""
        from app.services.document_processor import ProcessedBlock, ProcessedDocument

        result = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、概述", heading_level=1, page_number=1),
                ProcessedBlock(type="heading", text="二、设计", heading_level=1, page_number=2),
                ProcessedBlock(type="heading", text="三、实施", heading_level=1, page_number=3),
            ],
            metadata={},
            markdown="",
        )

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(result)

        # Should detect Chinese numbering pattern
        heading_rules = candidate["profile"]["heading_rules"]
        assert len(heading_rules) > 0
        # The pattern should match Chinese numbering
        assert any("一二三" in rule.get("pattern", "") for rule in heading_rules)

    @pytest.mark.asyncio
    async def test_suggest_profile_chunking_recommendation(self, mock_gateway):
        """suggest_profile recommends appropriate chunking parameters."""
        from app.services.document_processor import ProcessedBlock, ProcessedDocument

        # Document with many headings
        blocks = []
        for i in range(25):
            blocks.append(
                ProcessedBlock(type="heading", text=f"Section {i}", heading_level=2, page_number=i + 1)
            )
            blocks.append(
                ProcessedBlock(type="paragraph", text=f"Content {i}", page_number=i + 1)
            )

        result = ProcessedDocument(blocks=blocks, metadata={}, markdown="")

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(result)

        # With many headings, should recommend respect_heading_level = 1
        assert candidate["profile"]["chunking"]["respect_heading_level"] == 1


class TestSuggestProfileCandidate:
    """任务 10.5：候选 Profile 生成的细化契约。

    这些测试覆盖 envelope 形状、与 ``profile_matcher`` 的字典互转、
    标题/噪声/分块/表格启发式以及 Profile 命名约定。
    """

    @pytest.fixture
    def mock_gateway(self):
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock()
        return gateway

    @staticmethod
    def _make_doc(blocks, metadata=None):
        from app.services.document_processor import ProcessedDocument

        return ProcessedDocument(
            blocks=list(blocks), metadata=dict(metadata or {}), markdown=""
        )

    @staticmethod
    def _heading(text, level, page):
        from app.services.document_processor import ProcessedBlock

        return ProcessedBlock(
            type="heading", text=text, heading_level=level, page_number=page
        )

    @staticmethod
    def _para(text, page):
        from app.services.document_processor import ProcessedBlock

        return ProcessedBlock(type="paragraph", text=text, page_number=page)

    @staticmethod
    def _table(text, page):
        from app.services.document_processor import ProcessedBlock

        return ProcessedBlock(type="table", text=text, page_number=page)

    # ── Envelope shape ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_envelope_top_level_keys(self, mock_gateway):
        """候选返回值必须有 profile + metadata 两个顶层键。"""
        doc = self._make_doc(
            [
                self._heading("一、总则", 1, 1),
                self._para("内容", 1),
            ],
            metadata={"file_type": "pdf"},
        )
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        assert set(candidate.keys()) == {"profile", "metadata"}
        meta = candidate["metadata"]
        assert meta["status"] == "pending_approval"
        assert meta["source"] == "universal_parser"
        evidence = meta["evidence"]
        # 关键证据字段必须存在，方便审核 UI 展示。
        assert {
            "page_count",
            "heading_count",
            "table_count",
            "boilerplate_candidates",
            "avg_block_chars",
        } <= set(evidence.keys())
        assert evidence["page_count"] == 1
        assert evidence["heading_count"] == 1
        assert isinstance(evidence["avg_block_chars"], float)

    # ── Round-trip with profile_matcher ───────────────────────────

    @pytest.mark.asyncio
    async def test_profile_dict_round_trips_through_profile_matcher(
        self, mock_gateway
    ):
        """候选 profile 字典必须能通过 profile_from_dict / profile_to_dict 完整往返。"""
        from app.services.profile_matcher import profile_from_dict, profile_to_dict

        doc = self._make_doc(
            [
                self._heading("一、总则", 1, 1),
                self._heading("二、范围", 1, 2),
                self._para("正文段落，内容较长 " * 20, 1),
                self._table(
                    "| Name | Value |\n| --- | --- |\n| A | 1 |", 1
                ),
            ],
            metadata={"file_type": "pdf", "page_count": 2},
        )
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        profile_dict = candidate["profile"]
        # 不应混入 storage-only 字段（status 现在归 metadata envelope 管）。
        assert "status" not in profile_dict

        # 真正的 round-trip：dict → DocumentProfileConfig → dict。
        config = profile_from_dict(profile_dict)
        round_tripped = profile_to_dict(config)

        # 比较候选字典上的关键字段（profile_to_dict 还会带 id/version 等
        # storage 字段，所以做子集断言）。
        for key in (
            "name",
            "description",
            "priority",
            "enabled",
            "match_rules",
            "heading_rules",
            "boilerplate",
            "tables",
            "chunking",
            "domain_dictionary_id",
        ):
            assert round_tripped[key] == profile_dict[key], (
                f"round-trip drift on {key}: "
                f"{round_tripped[key]!r} vs {profile_dict[key]!r}"
            )

    # ── Heading patterns ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_heading_chinese_numbering_emitted(self, mock_gateway):
        """5 个中文编号 H1 → 发出中文编号规则。"""
        blocks = [
            self._heading(f"{ch}、章节", 1, i + 1)
            for i, ch in enumerate("一二三四五")
        ]
        doc = self._make_doc(blocks)
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        rules = candidate["profile"]["heading_rules"]
        assert len(rules) == 1
        assert rules[0]["level"] == 1
        assert "一二三" in rules[0]["pattern"]
        assert rules[0]["strip_pattern"] is False

    @pytest.mark.asyncio
    async def test_heading_mixed_levels_chapter_and_numeric(self, mock_gateway):
        """混合层级：H1 全部为 ``第N章``，H2 全部为 ``N.``，两个规则都要发出。"""
        h1 = [self._heading(f"第{i}章 引言", 1, i) for i in range(1, 5)]
        h2 = [self._heading(f"{i}. 小节", 2, i) for i in range(1, 7)]
        doc = self._make_doc(h1 + h2)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        rules = {rule["level"]: rule for rule in candidate["profile"]["heading_rules"]}
        assert set(rules) == {1, 2}
        # H1 命中 ``Chapter|第 \d+ 章`` 模式（catalog 中最长的，兼容 Chapter/第N章）。
        assert "第" in rules[1]["pattern"] or "Chapter" in rules[1]["pattern"]
        # H2 命中数字编号。
        assert r"\d+" in rules[2]["pattern"]

    @pytest.mark.asyncio
    async def test_heading_below_threshold_emits_no_rule(self, mock_gateway):
        """10 个 H1 中只有 3 个匹配编号 → 不发规则（60% 阈值）。"""
        # 3 个数字编号 + 7 个无编号
        matched = [self._heading(f"{i}. 标题", 1, i) for i in range(1, 4)]
        unmatched = [
            self._heading(f"自由文本标题 {i}", 1, i + 10) for i in range(7)
        ]
        doc = self._make_doc(matched + unmatched)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        assert candidate["profile"]["heading_rules"] == []

    # ── Boilerplate / noise patterns ──────────────────────────────

    @pytest.mark.asyncio
    async def test_boilerplate_emitted_when_present_on_all_pages(
        self, mock_gateway
    ):
        """4 页中相同短文本出现 4 次 → 作为锚定的转义 regex 发出。"""
        blocks = []
        for page in range(1, 5):
            blocks.append(self._para("Wikforge Inc. (Confidential)", page))
            blocks.append(self._para(f"page {page} body", page))
        doc = self._make_doc(blocks)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        manual = candidate["profile"]["boilerplate"]["manual_patterns"]
        # 必须 anchored + escaped，匹配原文本。
        expected = "^" + re.escape("Wikforge Inc. (Confidential)") + "$"
        assert expected in manual

    @pytest.mark.asyncio
    async def test_boilerplate_single_page_short_text_not_emitted(
        self, mock_gateway
    ):
        """只在第 1 页出现的短文本不应进入 boilerplate。"""
        blocks = [
            self._para("only-on-first-page-marker", 1),
            self._para("page 1 body", 1),
            self._para("page 2 body", 2),
            self._para("page 3 body", 3),
        ]
        doc = self._make_doc(blocks)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        manual = candidate["profile"]["boilerplate"]["manual_patterns"]
        assert all("only-on-first-page-marker" not in p for p in manual)

    @pytest.mark.asyncio
    async def test_boilerplate_page_marker_shortcut_two_pages(self, mock_gateway):
        """``Page 5`` / ``5/12`` 这类页码即使只出现 2 页也应捕获。"""
        blocks = [
            self._para("Page 1", 1),
            self._para("Page 2", 2),
            self._para("3/10", 1),
            self._para("3/10", 2),
            self._para("正文不应触发 shortcut " * 5, 1),
        ]
        doc = self._make_doc(blocks)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        manual = candidate["profile"]["boilerplate"]["manual_patterns"]
        # ``Page N`` 和 ``N/M`` 各自的 anchored escaped pattern 都应出现。
        # 但 "Page 1" 和 "Page 2" 是不同文本，shortcut 只在重复出现的同一文本上触发。
        # 因此用 3/10 这条断言路径。
        assert any("3/10" in p for p in manual)

    @pytest.mark.asyncio
    async def test_boilerplate_capped_at_twenty_and_sorted(self, mock_gateway):
        """超过 20 个候选时只保留前 20 个，按页数倒序。"""
        # 25 条短文本，分别在不同页数上重复（确保跨页且排名差异化）。
        blocks = []
        # 总页数 30 → 阈值 max(3, ceil(0.3*30))=9。
        for idx in range(25):
            text = f"footer-{idx:02d}"
            page_count = 30 - idx  # 30, 29, 28, ..., 6
            for page in range(1, page_count + 1):
                blocks.append(self._para(text, page))
        # 加几行真正的正文
        for page in range(1, 31):
            blocks.append(self._para(f"unique body for page {page}", page))

        doc = self._make_doc(blocks)
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        manual = candidate["profile"]["boilerplate"]["manual_patterns"]
        assert len(manual) <= 20
        # 排在前面的 footer 出现页数更多。
        # footer-00 (30 pages) > footer-01 (29 pages) > ...
        assert manual[0] == "^footer\\-00$"
        assert manual[1] == "^footer\\-01$"

    # ── Chunking heuristics ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_chunking_dense_document(self, mock_gateway):
        """heading_count > 20 → respect=1, min=384, max=1024。"""
        blocks = []
        for i in range(25):
            blocks.append(self._heading(f"Section {i}", 2, i + 1))
            # 长段落保证 avg_block_chars ≥ 100。
            blocks.append(self._para("这是一个比较长的段落。" * 20, i + 1))
        doc = self._make_doc(blocks)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        chunk = candidate["profile"]["chunking"]
        assert chunk["respect_heading_level"] == 1
        assert chunk["min_tokens"] == 384
        assert chunk["max_tokens"] == 1024
        assert chunk["overlap_tokens"] == 96

    @pytest.mark.asyncio
    async def test_chunking_sparse_document(self, mock_gateway):
        """heading_count < 5 → respect=3, min=256, max=800。"""
        blocks = [
            self._heading("Title 1", 1, 1),
            self._heading("Title 2", 1, 2),
            self._heading("Title 3", 1, 3),
        ]
        for i in range(10):
            blocks.append(self._para("这是一个比较长的段落。" * 20, (i % 3) + 1))
        doc = self._make_doc(blocks)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        chunk = candidate["profile"]["chunking"]
        assert chunk["respect_heading_level"] == 3
        assert chunk["min_tokens"] == 256
        assert chunk["max_tokens"] == 800

    @pytest.mark.asyncio
    async def test_chunking_short_block_override_halves_tokens(
        self, mock_gateway
    ):
        """avg_block_chars < 100 + dense → 减半 tokens, respect=1 不变。"""
        blocks = []
        for i in range(25):
            blocks.append(self._heading(f"S{i}", 2, i + 1))
            # 短段落（< 100 字符）。
            blocks.append(self._para(f"short {i}", i + 1))
        doc = self._make_doc(blocks)

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        chunk = candidate["profile"]["chunking"]
        assert chunk["respect_heading_level"] == 1  # 保留 dense 决策
        assert chunk["min_tokens"] == 192  # 384 // 2
        assert chunk["max_tokens"] == 512  # 1024 // 2

    @pytest.mark.asyncio
    async def test_chunking_protect_patterns_always_present(self, mock_gateway):
        """protect_patterns 永远非空，且包含单位 / 公式正则。"""
        doc = self._make_doc(
            [self._para("正文" * 50, 1)],
            metadata={"file_type": "pdf"},
        )
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        protect = candidate["profile"]["chunking"]["protect_patterns"]
        assert isinstance(protect, list) and len(protect) > 0
        joined = " ".join(protect)
        # 单位（mm/cm/...）与公差（±）。
        assert "mm" in joined
        assert "±" in joined

    # ── Tables ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_tables_row_level_chunking_when_large_table_present(
        self, mock_gateway
    ):
        """1 个表格有 30 行 → row_level_chunking=True。"""
        rows = "\n".join(f"| r{i} | v{i} |" for i in range(30))
        table = "| Name | Value |\n| --- | --- |\n" + rows
        doc = self._make_doc(
            [
                self._para("intro", 1),
                self._table(table, 1),
            ],
            metadata={"file_type": "pdf"},
        )
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        tables = candidate["profile"]["tables"]
        assert tables["row_level_chunking"] is True
        assert tables["cross_page_merge"] is True
        assert tables["collapse_merged_cells"] == "describe"

    @pytest.mark.asyncio
    async def test_tables_row_level_chunking_off_for_small_tables(
        self, mock_gateway
    ):
        """所有表格都很小 → row_level_chunking=False。"""
        small = "| Name | Value |\n| --- | --- |\n| A | 1 |\n| B | 2 |"
        doc = self._make_doc(
            [
                self._table(small, 1),
                self._table(small, 2),
            ],
            metadata={"file_type": "pdf"},
        )
        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        assert candidate["profile"]["tables"]["row_level_chunking"] is False

    # ── Profile naming ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_profile_name_with_file_type(self, mock_gateway):
        """file_type=pdf, 12 页 → name='auto-generated-pdf-12p'。"""
        blocks = [self._para(f"body {p}", p) for p in range(1, 13)]
        doc = self._make_doc(blocks, metadata={"file_type": "pdf"})

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        assert candidate["profile"]["name"] == "auto-generated-pdf-12p"

    @pytest.mark.asyncio
    async def test_profile_name_falls_back_to_generic(self, mock_gateway):
        """缺少 file_type → name 使用 'generic'。"""
        blocks = [self._para(f"body {p}", p) for p in range(1, 4)]
        doc = self._make_doc(blocks, metadata={})

        parser = UniversalParser(llm_gateway=mock_gateway)
        candidate = await parser.suggest_profile(doc)

        assert candidate["profile"]["name"] == "auto-generated-generic-3p"


class TestTriggerCondition:
    """Test trigger condition evaluation."""

    @patch("app.services.universal_parser.get_settings")
    def test_trigger_when_no_profile_matched(self, mock_settings):
        """Triggers when no profile was matched."""
        mock_settings.return_value = MagicMock(QUALITY_FALLBACK_THRESHOLD=0.7)

        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        assert parser.should_trigger(profile_matched=False) is True

    @patch("app.services.universal_parser.get_settings")
    def test_trigger_when_quality_below_threshold(self, mock_settings):
        """Triggers when quality score is below threshold."""
        mock_settings.return_value = MagicMock(QUALITY_FALLBACK_THRESHOLD=0.7)

        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        assert parser.should_trigger(profile_matched=True, quality_score=0.5) is True

    @patch("app.services.universal_parser.get_settings")
    def test_no_trigger_when_profile_matched_and_quality_ok(self, mock_settings):
        """Does not trigger when profile matched and quality is good."""
        mock_settings.return_value = MagicMock(QUALITY_FALLBACK_THRESHOLD=0.7)

        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        assert parser.should_trigger(profile_matched=True, quality_score=0.85) is False

    @patch("app.services.universal_parser.get_settings")
    def test_no_trigger_when_profile_matched_no_score(self, mock_settings):
        """Does not trigger when profile matched and no score computed yet."""
        mock_settings.return_value = MagicMock(QUALITY_FALLBACK_THRESHOLD=0.7)

        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        assert parser.should_trigger(profile_matched=True, quality_score=None) is False

    @patch("app.services.universal_parser.get_settings")
    def test_trigger_with_custom_threshold(self, mock_settings):
        """Respects custom threshold parameter."""
        mock_settings.return_value = MagicMock(QUALITY_FALLBACK_THRESHOLD=0.7)

        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        # Score 0.6 is below custom threshold 0.8
        assert parser.should_trigger(
            profile_matched=True, quality_score=0.6, threshold=0.8
        ) is True
        # Score 0.6 is above custom threshold 0.5
        assert parser.should_trigger(
            profile_matched=True, quality_score=0.6, threshold=0.5
        ) is False


# ─── 任务 10.2: Page-to-Image Pipeline Tests ─────────────────────────
#
# These tests focus narrowly on the page rasterization helpers introduced /
# hardened in task 10.2. They mock every external dependency (subprocess,
# shutil.which, pdf2image, os.path.exists) so they pass on machines without
# Poppler or LibreOffice installed.


class _FakePilImage:
    """Tiny stand-in for ``PIL.Image.Image`` that records PNG saves."""

    def __init__(self, payload: bytes = b"\x89PNG\r\n\x1a\nfake-png"):
        self._payload = payload
        self.saved_format: str | None = None

    def save(self, buffer, format: str):  # noqa: A002 — match PIL signature
        self.saved_format = format
        buffer.write(self._payload)


class _ProcStub:
    """Minimal stand-in for ``subprocess.CompletedProcess``."""

    def __init__(self, returncode: int = 0, stderr: bytes = b""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = b""


def _make_parser(dpi: int = 150, lo_timeout: int = 60) -> UniversalParser:
    """Build a UniversalParser with a mocked LLM gateway and pinned settings."""
    with patch("app.services.universal_parser.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=dpi,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=lo_timeout,
        )
        gateway = MagicMock(spec=LLMGateway)
        return UniversalParser(llm_gateway=gateway)


class TestPdfPageToImage:
    """``_pdf_page_to_image`` should be exception-safe and DPI-aware."""

    def test_returns_png_bytes_on_success(self):
        parser = _make_parser(dpi=200)

        fake_image = _FakePilImage(payload=b"\x89PNG\r\n\x1a\npage1")
        with patch(
            "pdf2image.convert_from_path", return_value=[fake_image]
        ) as mock_convert:
            result = parser._pdf_page_to_image("/tmp/sample.pdf", page_number=3)

        assert result == b"\x89PNG\r\n\x1a\npage1"
        # DPI from settings must propagate into pdf2image, and page bounds must be
        # passed through unchanged.
        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args.kwargs
        assert kwargs["first_page"] == 3
        assert kwargs["last_page"] == 3
        assert kwargs["dpi"] == 200
        assert kwargs["fmt"] == "png"
        assert fake_image.saved_format == "PNG"

    def test_returns_none_when_pdf2image_missing(self):
        parser = _make_parser()

        # Simulate ``import pdf2image`` failing inside the helper.
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pdf2image":
                raise ImportError("pdf2image not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = parser._pdf_page_to_image("/tmp/sample.pdf", page_number=1)

        assert result is None

    def test_returns_none_for_invalid_page_number(self):
        parser = _make_parser()
        with patch("pdf2image.convert_from_path") as mock_convert:
            assert parser._pdf_page_to_image("/tmp/sample.pdf", page_number=0) is None
            assert parser._pdf_page_to_image("/tmp/sample.pdf", page_number=-5) is None
        mock_convert.assert_not_called()

    def test_returns_none_when_convert_raises(self):
        parser = _make_parser()
        with patch(
            "pdf2image.convert_from_path", side_effect=RuntimeError("Poppler missing")
        ):
            result = parser._pdf_page_to_image("/tmp/sample.pdf", page_number=1)
        assert result is None

    def test_returns_none_when_no_images_produced(self):
        parser = _make_parser()
        with patch("pdf2image.convert_from_path", return_value=[]):
            assert (
                parser._pdf_page_to_image("/tmp/sample.pdf", page_number=1) is None
            )


class TestOfficePageToImage:
    """``_office_page_to_image`` shells out to LibreOffice + pdf2image safely."""

    def test_success_invokes_correct_soffice_args(self):
        parser = _make_parser(dpi=150, lo_timeout=45)

        fake_image = _FakePilImage(payload=b"office-png-bytes")
        proc = _ProcStub(returncode=0)

        with (
            patch(
                "app.services.universal_parser.shutil.which",
                return_value="/usr/bin/soffice",
            ),
            patch(
                "app.services.universal_parser.subprocess.run", return_value=proc
            ) as mock_run,
            patch(
                "app.services.universal_parser.os.makedirs"
            ),
            patch(
                "app.services.universal_parser.os.path.exists", return_value=True
            ),
            patch(
                "app.services.universal_parser.tempfile.mkdtemp",
                return_value="/tmp/wikforge-fake",
            ),
            patch(
                "app.services.universal_parser.shutil.rmtree"
            ),
            patch(
                "pdf2image.convert_from_path", return_value=[fake_image]
            ) as mock_convert,
        ):
            with parser._office_pdf_scope():
                result = parser._office_page_to_image(
                    "/data/report.docx", page_number=2
                )

        assert result == b"office-png-bytes"

        # LibreOffice CLI: must be headless, must request PDF, must use a
        # private user profile, and must respect the configured timeout.
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "/usr/bin/soffice"
        assert "--headless" in cmd
        assert "--convert-to" in cmd
        pdf_idx = cmd.index("--convert-to")
        assert cmd[pdf_idx + 1] == "pdf"
        assert "--outdir" in cmd
        assert "/data/report.docx" in cmd
        assert any(
            arg.startswith("-env:UserInstallation=file://") for arg in cmd
        )
        assert mock_run.call_args.kwargs["timeout"] == 45

        # pdf2image was invoked with the requested page and configured DPI.
        kwargs = mock_convert.call_args.kwargs
        assert kwargs["first_page"] == 2
        assert kwargs["last_page"] == 2
        assert kwargs["dpi"] == 150

    def test_returns_none_when_libreoffice_missing(self, caplog):
        parser = _make_parser()
        with (
            patch(
                "app.services.universal_parser.shutil.which", return_value=None
            ),
            patch("pdf2image.convert_from_path") as mock_convert,
            caplog.at_level("WARNING", logger="app.services.universal_parser"),
        ):
            with parser._office_pdf_scope():
                result = parser._office_page_to_image(
                    "/data/report.docx", page_number=1
                )

        assert result is None
        mock_convert.assert_not_called()
        assert any(
            "LibreOffice" in record.message for record in caplog.records
        )

    def test_returns_none_on_libreoffice_timeout(self):
        parser = _make_parser()

        with (
            patch(
                "app.services.universal_parser.shutil.which",
                return_value="/usr/bin/soffice",
            ),
            patch(
                "app.services.universal_parser.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="soffice", timeout=60),
            ),
            patch("app.services.universal_parser.os.makedirs"),
            patch(
                "app.services.universal_parser.tempfile.mkdtemp",
                return_value="/tmp/wikforge-fake",
            ),
            patch("app.services.universal_parser.shutil.rmtree"),
        ):
            with parser._office_pdf_scope():
                result = parser._office_page_to_image(
                    "/data/report.docx", page_number=1
                )

        assert result is None  # Did not raise out of the helper.

    def test_returns_none_on_nonzero_exit(self):
        parser = _make_parser()

        with (
            patch(
                "app.services.universal_parser.shutil.which",
                return_value="/usr/bin/soffice",
            ),
            patch(
                "app.services.universal_parser.subprocess.run",
                return_value=_ProcStub(returncode=77, stderr=b"corrupt input"),
            ),
            patch("app.services.universal_parser.os.makedirs"),
            patch(
                "app.services.universal_parser.tempfile.mkdtemp",
                return_value="/tmp/wikforge-fake",
            ),
            patch("app.services.universal_parser.shutil.rmtree"),
        ):
            with parser._office_pdf_scope():
                result = parser._office_page_to_image(
                    "/data/report.docx", page_number=1
                )

        assert result is None

    def test_pdf_cache_avoids_repeated_conversion(self):
        """Within a parse() scope the same Office file should convert exactly once."""
        parser = _make_parser()

        fake_image = _FakePilImage()
        with (
            patch(
                "app.services.universal_parser.shutil.which",
                return_value="/usr/bin/soffice",
            ),
            patch(
                "app.services.universal_parser.subprocess.run",
                return_value=_ProcStub(returncode=0),
            ) as mock_run,
            patch("app.services.universal_parser.os.makedirs"),
            patch(
                "app.services.universal_parser.os.path.exists",
                return_value=True,
            ),
            patch(
                "app.services.universal_parser.tempfile.mkdtemp",
                return_value="/tmp/wikforge-fake",
            ),
            patch("app.services.universal_parser.shutil.rmtree"),
            patch("pdf2image.convert_from_path", return_value=[fake_image]),
        ):
            with parser._office_pdf_scope():
                parser._office_page_to_image("/data/report.docx", page_number=1)
                parser._office_page_to_image("/data/report.docx", page_number=2)
                parser._office_page_to_image("/data/report.docx", page_number=3)

        # LibreOffice should run only once for the three pages.
        assert mock_run.call_count == 1


class TestGetPageImageRouting:
    """``_get_page_image`` must route by file_type / extension and stay safe."""

    def test_routes_pdf_to_pdf_helper(self):
        parser = _make_parser()
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="x", page_number=1)],
            metadata={"file_type": "pdf", "file_path": "/data/foo.pdf"},
        )

        with (
            patch(
                "app.services.universal_parser.os.path.exists", return_value=True
            ),
            patch.object(
                UniversalParser, "_pdf_page_to_image", return_value=b"pdf-png"
            ) as mock_pdf,
            patch.object(
                UniversalParser, "_office_page_to_image"
            ) as mock_office,
        ):
            result = parser._get_page_image(doc, page_number=1)

        assert result == b"pdf-png"
        mock_pdf.assert_called_once_with("/data/foo.pdf", 1)
        mock_office.assert_not_called()

    def test_routes_docx_to_office_helper(self):
        parser = _make_parser()
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="x", page_number=1)],
            metadata={"file_type": "docx", "file_path": "/data/foo.docx"},
        )

        with (
            patch(
                "app.services.universal_parser.os.path.exists", return_value=True
            ),
            patch.object(
                UniversalParser, "_pdf_page_to_image"
            ) as mock_pdf,
            patch.object(
                UniversalParser,
                "_office_page_to_image",
                return_value=b"docx-png",
            ) as mock_office,
        ):
            result = parser._get_page_image(doc, page_number=1)

        assert result == b"docx-png"
        mock_office.assert_called_once_with("/data/foo.docx", 1)
        mock_pdf.assert_not_called()

    def test_extension_only_routing_when_filetype_missing(self):
        parser = _make_parser()
        # No file_type provided — must route from the path's extension.
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="x", page_number=1)],
            metadata={"file_path": "/data/Report.PPTX"},
        )

        with (
            patch(
                "app.services.universal_parser.os.path.exists", return_value=True
            ),
            patch.object(
                UniversalParser,
                "_office_page_to_image",
                return_value=b"pptx-png",
            ) as mock_office,
        ):
            assert parser._get_page_image(doc, page_number=1) == b"pptx-png"
        mock_office.assert_called_once()

    def test_returns_none_for_unsupported_format(self):
        parser = _make_parser()
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="x", page_number=1)],
            metadata={"file_type": "epub", "file_path": "/data/foo.epub"},
        )

        with (
            patch(
                "app.services.universal_parser.os.path.exists", return_value=True
            ),
            patch.object(UniversalParser, "_pdf_page_to_image") as mock_pdf,
            patch.object(
                UniversalParser, "_office_page_to_image"
            ) as mock_office,
        ):
            assert parser._get_page_image(doc, page_number=1) is None
        mock_pdf.assert_not_called()
        mock_office.assert_not_called()

    def test_returns_none_when_file_path_missing(self):
        parser = _make_parser()
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="x", page_number=1)],
            metadata={"file_type": "pdf"},  # no file_path
        )

        # ``os.path.exists`` must NOT be called when no path is provided.
        with patch(
            "app.services.universal_parser.os.path.exists"
        ) as mock_exists:
            assert parser._get_page_image(doc, page_number=1) is None
        mock_exists.assert_not_called()

    def test_returns_none_when_file_does_not_exist(self):
        parser = _make_parser()
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="x", page_number=1)],
            metadata={"file_type": "pdf", "file_path": "/data/missing.pdf"},
        )

        with (
            patch(
                "app.services.universal_parser.os.path.exists", return_value=False
            ),
            patch.object(UniversalParser, "_pdf_page_to_image") as mock_pdf,
        ):
            assert parser._get_page_image(doc, page_number=1) is None
        mock_pdf.assert_not_called()


# ─── 任务 10.3: 逐页多模态 LLM 解析（图片 + 原始文本 → 结构化 Markdown） ─────
#
# 这些测试聚焦在 ``_parse_page`` 单页协程的契约上，刻意绕开 ``parse()`` 的页面
# 合并逻辑，让单页失败/截断/降级路径能被独立断言。统一使用 ``_make_parser``
# 注入的 mock LLMGateway，配合 ``AsyncMock`` 模拟成功 / 超时 / 空响应。


def _make_parse_page_parser(
    *,
    vision_model: str | None = None,
    page_timeout: float = 60.0,
    max_raw_text_chars: int | None = None,
) -> tuple[UniversalParser, MagicMock]:
    """构造 ``_parse_page`` 测试专用的 parser，并返回 (parser, gateway)。

    与 ``_make_parser`` 区别：默认提供 ``UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS``
    设置，确保新 settings 字段被读取；同时把 LLMGateway 暴露给测试用例做断言。
    """
    with patch("app.services.universal_parser.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
        )
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock()
        gateway.complete_multimodal = AsyncMock()
        parser = UniversalParser(
            llm_gateway=gateway,
            vision_model=vision_model,
            page_timeout=page_timeout,
            max_raw_text_chars=max_raw_text_chars,
        )
    return parser, gateway


class TestParsePageMultimodalDispatch:
    """有图像时走 ``complete_multimodal``，无图像时回退到 ``complete``。"""

    @pytest.mark.asyncio
    async def test_uses_multimodal_when_image_provided(self):
        parser, gateway = _make_parse_page_parser()

        gateway.complete_multimodal.return_value = LLMResponse(
            content="# Page 1\n\nbody text",
            model="gpt-4o",
        )

        result = await parser._parse_page(
            page_number=7,
            raw_text="raw extracted text",
            page_image=b"fake-png-bytes",
        )

        # 多模态路径被调用且只调用一次
        gateway.complete_multimodal.assert_awaited_once()
        gateway.complete.assert_not_called()

        kwargs = gateway.complete_multimodal.await_args.kwargs
        # images 列表必须包含传入的页面图像
        assert kwargs["images"] == [b"fake-png-bytes"]
        # 系统提示非空，并体现强制 Markdown 规则
        assert isinstance(kwargs["system_prompt"], str)
        assert kwargs["system_prompt"].strip() != ""
        # 用户提示包含页码与原始文本
        assert "page 7" in kwargs["prompt"].lower()
        assert "raw extracted text" in kwargs["prompt"]
        # 默认部署不传 ``model`` 覆盖，让 gateway 使用其自身配置
        assert "model" not in kwargs

        # 返回结构正确
        assert isinstance(result, PageResult)
        assert result.page_number == 7
        assert result.markdown.startswith("# Page 1")
        assert result.headings == [{"level": 1, "text": "Page 1"}]

    @pytest.mark.asyncio
    async def test_passes_vision_model_override_when_configured(self):
        """vision_model 被显式配置时，多模态调用必须带上 ``model`` 覆盖参数（任务 10.7 接口）。"""
        parser, gateway = _make_parse_page_parser(vision_model="qwen-vl-max")

        gateway.complete_multimodal.return_value = LLMResponse(
            content="# Header\n\ncontent",
            model="qwen-vl-max",
        )

        await parser._parse_page(
            page_number=1,
            raw_text="text",
            page_image=b"png",
        )

        kwargs = gateway.complete_multimodal.await_args.kwargs
        assert kwargs["model"] == "qwen-vl-max"

    @pytest.mark.asyncio
    async def test_falls_back_to_text_only_when_no_image(self):
        parser, gateway = _make_parse_page_parser()

        gateway.complete.return_value = LLMResponse(
            content="# Title\n\nplain text page",
            model="gpt-4o",
        )

        result = await parser._parse_page(
            page_number=2,
            raw_text="some raw text",
            page_image=None,
        )

        # 文本路径被调用，多模态路径完全没有触发
        gateway.complete.assert_awaited_once()
        gateway.complete_multimodal.assert_not_called()

        kwargs = gateway.complete.await_args.kwargs
        assert "page 2" in kwargs["prompt"].lower()
        assert "some raw text" in kwargs["prompt"]
        assert isinstance(kwargs["system_prompt"], str)
        assert kwargs["system_prompt"].strip() != ""

        assert result.markdown == "# Title\n\nplain text page"
        assert result.success is True


class TestParsePageTimeoutAndErrors:
    """超时与异常响应必须抛出 LLMGatewayError，让外层 parse() 走降级路径。"""

    @pytest.mark.asyncio
    async def test_raises_llm_gateway_error_on_timeout(self):
        parser, gateway = _make_parse_page_parser(page_timeout=0.05)

        async def slow_call(*_args, **_kwargs):
            # 故意 sleep 远长于配置的 page_timeout，强制 asyncio.wait_for 触发 TimeoutError
            await asyncio.sleep(0.5)
            return LLMResponse(content="late", model="gpt-4o")

        gateway.complete_multimodal.side_effect = slow_call

        with pytest.raises(LLMGatewayError) as exc_info:
            await parser._parse_page(
                page_number=3,
                raw_text="x",
                page_image=b"png",
            )

        assert exc_info.value.reason == "timeout"
        assert "page 3" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raises_on_empty_content(self):
        """空字符串响应必须抛错，避免将空页静默并入文档。"""
        parser, gateway = _make_parse_page_parser()
        gateway.complete_multimodal.return_value = LLMResponse(
            content="   \n  ",  # whitespace only — 等价于空响应
            model="gpt-4o",
        )

        with pytest.raises(LLMGatewayError) as exc_info:
            await parser._parse_page(
                page_number=4,
                raw_text="raw",
                page_image=b"png",
            )

        assert exc_info.value.reason == "empty"

    @pytest.mark.asyncio
    async def test_raises_on_non_string_content(self):
        """非字符串响应同样视为硬失败。"""
        parser, gateway = _make_parse_page_parser()
        gateway.complete.return_value = LLMResponse(content=None, model="gpt-4o")  # type: ignore[arg-type]

        with pytest.raises(LLMGatewayError):
            await parser._parse_page(
                page_number=1,
                raw_text="raw",
                page_image=None,
            )


class TestParsePagePostProcessing:
    """后处理：去围栏、抽取标题/表格、截断原文。"""

    @pytest.mark.asyncio
    async def test_strips_surrounding_markdown_fence(self):
        parser, gateway = _make_parse_page_parser()
        # LLM 不听话，把整页结果包在 ```markdown ... ``` 里
        gateway.complete_multimodal.return_value = LLMResponse(
            content="```markdown\n# Title\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n```",
            model="gpt-4o",
        )

        result = await parser._parse_page(
            page_number=1,
            raw_text="raw",
            page_image=b"png",
        )

        # 围栏被剥离，且 markdown 仍可被正确解析为标题 + 表格
        assert not result.markdown.startswith("```")
        assert not result.markdown.endswith("```")
        assert result.markdown.startswith("# Title")
        assert result.headings == [{"level": 1, "text": "Title"}]
        assert len(result.tables) == 1
        assert "| A | B |" in result.tables[0]

    @pytest.mark.asyncio
    async def test_strips_plain_fence_without_language_tag(self):
        parser, gateway = _make_parse_page_parser()
        gateway.complete_multimodal.return_value = LLMResponse(
            content="```\n# Heading\n\nbody\n```",
            model="gpt-4o",
        )

        result = await parser._parse_page(
            page_number=1,
            raw_text="raw",
            page_image=b"png",
        )

        assert result.markdown == "# Heading\n\nbody"

    @pytest.mark.asyncio
    async def test_truncates_raw_text_to_configured_budget(self):
        """超长 raw_text 必须被硬截断到 max_raw_text_chars，避免提示词溢出。"""
        parser, gateway = _make_parse_page_parser(max_raw_text_chars=50)
        gateway.complete_multimodal.return_value = LLMResponse(
            content="# ok\n\nbody",
            model="gpt-4o",
        )

        long_raw = "A" * 5000  # 远超 50 字符预算
        await parser._parse_page(
            page_number=1,
            raw_text=long_raw,
            page_image=b"png",
        )

        prompt: str = gateway.complete_multimodal.await_args.kwargs["prompt"]
        # 提示词中出现的连续 ``A`` 必须不超过预算长度。这里直接断言整个提示
        # 中 'A' 的总数 ≤ 50；模板本身不含 'A'，所以这等价于裁剪后的 raw_text 长度。
        a_count = prompt.count("A")
        assert a_count <= 50
        # 同时确保 5000 条原文里的尾部内容真的被丢弃了
        assert long_raw not in prompt

    @pytest.mark.asyncio
    async def test_extracts_headings_and_tables_from_response(self):
        """成功响应必须填充 PageResult.headings 与 PageResult.tables。"""
        parser, gateway = _make_parse_page_parser()
        gateway.complete.return_value = LLMResponse(
            content=(
                "# H1\n"
                "## H2\n"
                "para text\n"
                "\n"
                "| col1 | col2 |\n"
                "| --- | --- |\n"
                "| a | b |\n"
            ),
            model="gpt-4o",
        )

        result = await parser._parse_page(
            page_number=1,
            raw_text="raw",
            page_image=None,
        )

        assert result.headings == [
            {"level": 1, "text": "H1"},
            {"level": 2, "text": "H2"},
        ]
        assert len(result.tables) == 1
        assert "| col1 | col2 |" in result.tables[0]
        assert "| a | b |" in result.tables[0]


class TestUniversalParserSettingsPlumbing:
    """构造函数把 ``UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS`` 设置接入。"""

    @patch("app.services.universal_parser.get_settings")
    def test_reads_max_raw_text_chars_from_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=1234,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)

        assert parser.max_raw_text_chars == 1234

    @patch("app.services.universal_parser.get_settings")
    def test_constructor_override_wins_over_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway, max_raw_text_chars=42)

        assert parser.max_raw_text_chars == 42

    @patch("app.services.universal_parser.get_settings")
    def test_max_raw_text_chars_floor_is_one(self, mock_settings):
        """0 / 负值不允许，避免空提示词被发到 LLM 触发 'empty content' 误判。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=0,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway, max_raw_text_chars=-50)

        assert parser.max_raw_text_chars == 1


# ─── 任务 10.7: LLM 模型选择配置（GPT-4o / Qwen-VL / MiniCPM-V 等） ─────
#
# 这些测试覆盖：
# - 构造函数从 settings 读取 ``UNIVERSAL_PARSER_VISION_MODEL`` /
#   ``UNIVERSAL_PARSER_TEXT_MODEL``。
# - 显式构造参数优先于 settings。
# - 旧 ``model=`` 参数在新参数缺席时回填 ``self.text_model``。
# - ``_parse_page`` 在多模态 / 文本路径上正确传 / 不传 ``model`` 覆盖。
# - ``is_known_vision_model`` / ``is_known_text_model`` 的纯字符串校验逻辑
#   （前缀匹配、provider 前缀剥离、tag 剥离、未知模型不通过）。


class TestModelSelection:
    """任务 10.7：LLM 模型选择契约。"""

    # ─── settings 读取 / 显式参数优先级 ───────────────────────────

    @patch("app.services.universal_parser.get_settings")
    def test_constructor_reads_vision_model_from_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_VISION_MODEL="qwen-vl-max",
            UNIVERSAL_PARSER_TEXT_MODEL="",
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)
        assert parser.vision_model == "qwen-vl-max"
        # text 路径未设置 → 保持 None（让 gateway 用自己的默认模型）
        assert parser.text_model is None

    @patch("app.services.universal_parser.get_settings")
    def test_constructor_reads_text_model_from_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_VISION_MODEL="",
            UNIVERSAL_PARSER_TEXT_MODEL="qwen-max",
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)
        assert parser.text_model == "qwen-max"
        assert parser.vision_model is None

    @patch("app.services.universal_parser.get_settings")
    def test_explicit_vision_model_overrides_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_VISION_MODEL="qwen-vl-max",
            UNIVERSAL_PARSER_TEXT_MODEL="",
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(
            llm_gateway=gateway,
            vision_model="ollama/minicpm-v:latest",
        )
        assert parser.vision_model == "ollama/minicpm-v:latest"

    @patch("app.services.universal_parser.get_settings")
    def test_explicit_text_model_overrides_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_VISION_MODEL="",
            UNIVERSAL_PARSER_TEXT_MODEL="qwen-max",
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway, text_model="deepseek-chat")
        assert parser.text_model == "deepseek-chat"

    @patch("app.services.universal_parser.get_settings")
    def test_legacy_model_kwarg_falls_back_to_text_model(self, mock_settings):
        """旧 API ``model=...`` 在新参数缺席时承担 text_model 的角色。"""
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_VISION_MODEL="",
            UNIVERSAL_PARSER_TEXT_MODEL="",
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway, model="minicpm-v")
        # 兼容老调用：model 同时被记为 self.model 与 self.text_model 兜底。
        assert parser.model == "minicpm-v"
        assert parser.text_model == "minicpm-v"

    @patch("app.services.universal_parser.get_settings")
    def test_explicit_text_model_wins_over_legacy_model(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_VISION_MODEL="",
            UNIVERSAL_PARSER_TEXT_MODEL="",
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(
            llm_gateway=gateway,
            model="minicpm-v",
            text_model="qwen-max",
        )
        assert parser.text_model == "qwen-max"

    # ─── _parse_page 上的 model 透传 ─────────────────────────────

    @pytest.mark.asyncio
    async def test_parse_page_passes_text_model_when_configured(self):
        """text_model 配置后，文本路径必须透传 ``model=`` 给 gateway。"""
        with patch("app.services.universal_parser.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                QUALITY_FALLBACK_THRESHOLD=0.7,
                UNIVERSAL_PARSER_PAGE_DPI=150,
                UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
                UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
                UNIVERSAL_PARSER_VISION_MODEL="",
                UNIVERSAL_PARSER_TEXT_MODEL="",
            )
            gateway = MagicMock(spec=LLMGateway)
            gateway.complete = AsyncMock(
                return_value=LLMResponse(content="# h\n\nbody", model="qwen-max")
            )
            gateway.complete_multimodal = AsyncMock()
            parser = UniversalParser(llm_gateway=gateway, text_model="qwen-max")

        await parser._parse_page(page_number=1, raw_text="raw", page_image=None)
        kwargs = gateway.complete.await_args.kwargs
        assert kwargs["model"] == "qwen-max"
        gateway.complete_multimodal.assert_not_called()

    @pytest.mark.asyncio
    async def test_parse_page_omits_text_model_when_not_configured(self):
        """text_model 未配置时，文本路径不应传 ``model=``。"""
        with patch("app.services.universal_parser.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                QUALITY_FALLBACK_THRESHOLD=0.7,
                UNIVERSAL_PARSER_PAGE_DPI=150,
                UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
                UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
                UNIVERSAL_PARSER_VISION_MODEL="",
                UNIVERSAL_PARSER_TEXT_MODEL="",
            )
            gateway = MagicMock(spec=LLMGateway)
            gateway.complete = AsyncMock(
                return_value=LLMResponse(content="# h\n\nbody", model="gpt-4o")
            )
            gateway.complete_multimodal = AsyncMock()
            # 不传 model / text_model；构造器应保持 self.text_model = None
            parser = UniversalParser(llm_gateway=gateway)
        assert parser.text_model is None

        await parser._parse_page(page_number=1, raw_text="raw", page_image=None)
        kwargs = gateway.complete.await_args.kwargs
        assert "model" not in kwargs

    @pytest.mark.asyncio
    async def test_parse_page_omits_vision_model_when_not_configured(self):
        """vision_model 未配置时，多模态路径不应传 ``model=``（与 10.3 行为一致）。"""
        with patch("app.services.universal_parser.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                QUALITY_FALLBACK_THRESHOLD=0.7,
                UNIVERSAL_PARSER_PAGE_DPI=150,
                UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
                UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
                UNIVERSAL_PARSER_VISION_MODEL="",
                UNIVERSAL_PARSER_TEXT_MODEL="",
            )
            gateway = MagicMock(spec=LLMGateway)
            gateway.complete = AsyncMock()
            gateway.complete_multimodal = AsyncMock(
                return_value=LLMResponse(content="# h\n\nbody", model="gpt-4o")
            )
            parser = UniversalParser(llm_gateway=gateway)
        assert parser.vision_model is None

        await parser._parse_page(
            page_number=1, raw_text="raw", page_image=b"png-bytes"
        )
        kwargs = gateway.complete_multimodal.await_args.kwargs
        assert "model" not in kwargs

    @pytest.mark.asyncio
    async def test_parse_page_passes_vision_model_from_settings(self):
        """settings 设置的 vision_model 在多模态路径上必须生效。"""
        with patch("app.services.universal_parser.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                LITELLM_MODEL="gpt-4o",
                QUALITY_FALLBACK_THRESHOLD=0.7,
                UNIVERSAL_PARSER_PAGE_DPI=150,
                UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
                UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
                UNIVERSAL_PARSER_VISION_MODEL="minicpm-v",
                UNIVERSAL_PARSER_TEXT_MODEL="",
            )
            gateway = MagicMock(spec=LLMGateway)
            gateway.complete = AsyncMock()
            gateway.complete_multimodal = AsyncMock(
                return_value=LLMResponse(content="# h\n\nbody", model="minicpm-v")
            )
            parser = UniversalParser(llm_gateway=gateway)

        await parser._parse_page(
            page_number=1, raw_text="raw", page_image=b"png-bytes"
        )
        kwargs = gateway.complete_multimodal.await_args.kwargs
        assert kwargs["model"] == "minicpm-v"

    # ─── 已知模型目录的字符串校验 ───────────────────────────────

    def test_is_known_vision_model_basic(self):
        assert UniversalParser.is_known_vision_model("gpt-4o") is True
        assert UniversalParser.is_known_vision_model("GPT-4o") is True
        assert UniversalParser.is_known_vision_model("qwen-vl-max") is True
        assert UniversalParser.is_known_vision_model("minicpm-v") is True

    def test_is_known_vision_model_provider_prefixed(self):
        # provider 前缀 + tag 应该被剥离后命中
        assert (
            UniversalParser.is_known_vision_model("ollama/minicpm-v:latest") is True
        )
        assert UniversalParser.is_known_vision_model("openrouter/gpt-4o") is True

    def test_is_known_vision_model_versioned_variant(self):
        # ``gpt-4o-2024-05-13`` 是 ``gpt-4o`` 系列变体（前缀匹配）。
        assert UniversalParser.is_known_vision_model("gpt-4o-2024-05-13") is True

    def test_is_known_vision_model_rejects_unknown(self):
        assert UniversalParser.is_known_vision_model("totally-fake-model") is False
        assert UniversalParser.is_known_vision_model("") is False
        assert UniversalParser.is_known_vision_model("   ") is False

    def test_is_known_text_model_basic(self):
        assert UniversalParser.is_known_text_model("gpt-4o-mini") is True
        assert UniversalParser.is_known_text_model("qwen-max") is True
        assert (
            UniversalParser.is_known_text_model("claude-3-5-sonnet-20241022") is True
        )

    def test_is_known_text_model_provider_prefixed(self):
        assert UniversalParser.is_known_text_model("ollama/llama3.1:latest") is True

    def test_is_known_text_model_rejects_unknown(self):
        assert UniversalParser.is_known_text_model("not-a-real-model") is False
        assert UniversalParser.is_known_text_model("") is False


# ─── 任务 10.4: 页面结果合并（保留页码、合并跨页表格、去重重复段落） ─────
#
# 这些测试聚焦在 ``_merge_page_results`` / ``_merge_cross_page_tables`` 两个
# 同步入口上：直接构造 ``PageResult`` 列表喂入合并器，跳过 LLM 调用，让单测
# 在不依赖网络/LiteLLM 的情况下覆盖所有合并语义边界。


def _merge_parser() -> UniversalParser:
    """构造一个仅用于合并逻辑的 UniversalParser。

    该 parser 不会真正调用 LLM；这里只需要 ``_merge_page_results``、
    ``_markdown_to_blocks``、``_merge_cross_page_tables`` 这几个同步辅助方法
    就足够了。沿用 ``_make_parser`` 的 settings 模拟避免读取真实环境。
    """
    return _make_parser()


class TestMergePageResults:
    """``_merge_page_results`` 的页面合并契约。"""

    # ─── 页码保留 ────────────────────────────────────────────────────

    def test_page_numbers_preserved_across_pages(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown="# Page 1 Heading\n\nFirst page paragraph.",
            ),
            PageResult(
                page_number=2,
                markdown="## Page 2 Heading\n\nSecond page paragraph.",
            ),
            PageResult(
                page_number=3,
                markdown="### Page 3 Heading\n\nThird page paragraph.",
            ),
        ]

        merged = parser._merge_page_results(results, metadata={"file_type": "pdf"})

        page_numbers = {b.page_number for b in merged.blocks}
        assert page_numbers == {1, 2, 3}

        # 每个 heading 的 page_number 应该等于它的来源页
        headings = [b for b in merged.blocks if b.type == "heading"]
        assert len(headings) == 3
        assert headings[0].page_number == 1
        assert headings[0].text == "Page 1 Heading"
        assert headings[-1].page_number == 3

        # 最后一块（无论类型）应当来自 page 3
        assert merged.blocks[-1].page_number == 3

    def test_unsorted_input_is_sorted_defensively(self):
        """传入乱序的 PageResult 列表，合并仍按页码顺序输出块。"""
        parser = _merge_parser()
        results = [
            PageResult(page_number=3, markdown="page-3-only"),
            PageResult(page_number=1, markdown="page-1-only"),
            PageResult(page_number=2, markdown="page-2-only"),
        ]

        merged = parser._merge_page_results(results, metadata={})

        observed_pages = [b.page_number for b in merged.blocks]
        # 必须按 1 → 2 → 3 出现
        assert observed_pages == sorted(observed_pages)
        assert observed_pages[0] == 1
        assert observed_pages[-1] == 3

    # ─── 跳过失败 / 空页 ─────────────────────────────────────────────

    def test_skips_failed_and_empty_pages(self):
        """failed PageResult 与空 markdown 都应当被静默跳过，不抛异常。"""
        parser = _merge_parser()
        results = [
            PageResult(page_number=1, markdown="real content"),
            PageResult(page_number=2, markdown="bogus", success=False),
            PageResult(page_number=3, markdown="   \n   "),  # whitespace only
            PageResult(page_number=4, markdown=""),  # 完全空
            PageResult(page_number=5, markdown="more real content"),
        ]

        merged = parser._merge_page_results(results, metadata={})

        page_numbers = {b.page_number for b in merged.blocks}
        # 仅页码 1 与 5 的内容会被保留
        assert page_numbers == {1, 5}
        assert any("real content" in b.text for b in merged.blocks)
        assert any("more real content" in b.text for b in merged.blocks)

    # ─── 短重复段落去重（噪声） ──────────────────────────────────────

    def test_short_repeated_paragraphs_deduplicated(self):
        """跨页重复且长度 < 100 的段落只保留首次出现。"""
        parser = _merge_parser()
        boilerplate = "Confidential — do not distribute"
        results = [
            PageResult(
                page_number=1,
                markdown=f"{boilerplate}\n\n# Section 1\n\nReal page-1 content.",
            ),
            PageResult(
                page_number=2,
                markdown=f"{boilerplate}\n\n# Section 2\n\nReal page-2 content.",
            ),
            PageResult(
                page_number=3,
                markdown=f"{boilerplate}\n\n# Section 3\n\nReal page-3 content.",
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        boilerplate_blocks = [
            b for b in merged.blocks if boilerplate.lower() in b.text.lower()
        ]
        # 跨 3 页只应保留 1 次，且必须停留在首次出现的页（page 1）
        assert len(boilerplate_blocks) == 1
        assert boilerplate_blocks[0].page_number == 1

        # 噪声计数 = 后两页被丢弃的次数
        assert merged.noise_removed_count == 2

    def test_long_repeated_paragraphs_preserved(self):
        """长度 ≥ 100 的重复段落是正文，必须每页都保留。"""
        parser = _merge_parser()
        long_paragraph = "A" * 200  # 远超 100 字符阈值
        results = [
            PageResult(page_number=1, markdown=f"{long_paragraph}\n\nPage 1 tail."),
            PageResult(page_number=2, markdown=f"{long_paragraph}\n\nPage 2 tail."),
        ]

        merged = parser._merge_page_results(results, metadata={})

        long_blocks = [b for b in merged.blocks if b.text == long_paragraph]
        # 长段落跨页应该仍然出现两次（每页一次）
        assert len(long_blocks) == 2
        assert {b.page_number for b in long_blocks} == {1, 2}
        # 长段落不计入噪声
        assert merged.noise_removed_count == 0

    # ─── 跨页表格合并 ────────────────────────────────────────────────

    def test_cross_page_table_merge_two_pages(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "# Inventory\n\n"
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| Widget | 10 |\n"
                ),
            ),
            PageResult(
                page_number=2,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| Gadget | 20 |\n"
                    "| Sprocket | 30 |\n"
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        tables = [b for b in merged.blocks if b.type == "table"]
        assert len(tables) == 1
        # 锚定在第一页
        assert tables[0].page_number == 1
        # 行内容来自两页
        assert "Widget" in tables[0].text
        assert "Gadget" in tables[0].text
        assert "Sprocket" in tables[0].text
        # 表头只出现一次（中间不应再有 "| --- | --- |"）
        assert tables[0].text.count("| --- | --- |") == 1

    def test_cross_page_table_merge_three_or_more_pages(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| A | 1 |\n"
                ),
            ),
            PageResult(
                page_number=2,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| B | 2 |\n"
                ),
            ),
            PageResult(
                page_number=3,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| C | 3 |\n"
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        tables = [b for b in merged.blocks if b.type == "table"]
        assert len(tables) == 1
        assert tables[0].page_number == 1
        for row_id in ("A", "B", "C"):
            assert f"| {row_id} |" in tables[0].text

    def test_cross_page_table_not_merged_on_header_mismatch(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| A | 1 |\n"
                ),
            ),
            PageResult(
                page_number=2,
                markdown=(
                    "| Name | Total |\n"  # 表头不同
                    "| --- | --- |\n"
                    "| B | 2 |\n"
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        tables = [b for b in merged.blocks if b.type == "table"]
        assert len(tables) == 2
        assert tables[0].page_number == 1
        assert tables[1].page_number == 2

    def test_cross_page_table_not_merged_on_non_adjacent_pages(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| A | 1 |\n"
                ),
            ),
            # page 2 缺失 / 内容不同（这里干脆没有这一页）
            PageResult(
                page_number=3,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| C | 3 |\n"
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        tables = [b for b in merged.blocks if b.type == "table"]
        # 不相邻的两张同表头表格不能合并
        assert len(tables) == 2
        assert tables[0].page_number == 1
        assert tables[1].page_number == 3

    # ─── 头部空白容忍 ────────────────────────────────────────────────

    def test_header_match_is_whitespace_and_case_insensitive(self):
        """单元格内空白与大小写差异不应阻断合并。"""
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "|  Item  |  Qty  |\n"
                    "| --- | --- |\n"
                    "| A | 1 |\n"
                ),
            ),
            PageResult(
                page_number=2,
                markdown=(
                    "| ITEM | qty |\n"
                    "| --- | --- |\n"
                    "| B | 2 |\n"
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})
        tables = [b for b in merged.blocks if b.type == "table"]
        assert len(tables) == 1
        assert "A" in tables[0].text
        assert "B" in tables[0].text

    # ─── 混合内容（标题 + 段落 + 表格） ──────────────────────────────

    def test_mixed_content_keeps_headings_and_merges_tables(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "# Chapter 1\n\n"
                    "Intro paragraph for chapter 1.\n"
                    "\n"
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| A | 1 |\n"
                ),
            ),
            PageResult(
                page_number=2,
                markdown=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| B | 2 |\n"
                    "\n"
                    "## Chapter 1 (cont.)\n\n"
                    "Continuation paragraph."
                ),
            ),
            PageResult(
                page_number=3,
                markdown=(
                    "# Chapter 2\n\n"
                    "Another section paragraph."
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        headings = [b for b in merged.blocks if b.type == "heading"]
        # 三个标题各自停留在原页
        assert {h.text for h in headings} == {
            "Chapter 1",
            "Chapter 1 (cont.)",
            "Chapter 2",
        }
        page_by_heading = {h.text: h.page_number for h in headings}
        assert page_by_heading["Chapter 1"] == 1
        assert page_by_heading["Chapter 1 (cont.)"] == 2
        assert page_by_heading["Chapter 2"] == 3

        # 跨页表格被合并为一个，锚定在 page 1
        tables = [b for b in merged.blocks if b.type == "table"]
        assert len(tables) == 1
        assert tables[0].page_number == 1
        assert "A" in tables[0].text and "B" in tables[0].text

        # 段落顺序保持（每页内部相对顺序不变）
        paragraph_texts = [b.text for b in merged.blocks if b.type == "paragraph"]
        assert "Intro paragraph for chapter 1." in paragraph_texts
        assert "Continuation paragraph." in paragraph_texts
        assert "Another section paragraph." in paragraph_texts

    # ─── 计数与 markdown 输出 ────────────────────────────────────────

    def test_headings_detected_counter_matches_actual(self):
        parser = _merge_parser()
        results = [
            PageResult(page_number=1, markdown="# A\n## B\n### C\n\npara"),
            PageResult(page_number=2, markdown="# D\n\npara2"),
        ]
        merged = parser._merge_page_results(results, metadata={})
        actual = sum(1 for b in merged.blocks if b.type == "heading")
        assert merged.headings_detected == actual == 4

    def test_merged_markdown_round_trip(self):
        parser = _merge_parser()
        results = [
            PageResult(
                page_number=1,
                markdown=(
                    "# Surviving Heading One\n\n"
                    "Body paragraph one.\n"
                    "\n"
                    "| Col1 | Col2 |\n"
                    "| --- | --- |\n"
                    "| x | y |\n"
                ),
            ),
            PageResult(
                page_number=2,
                markdown=(
                    "## Surviving Heading Two\n\n"
                    "Body paragraph two."
                ),
            ),
        ]

        merged = parser._merge_page_results(results, metadata={})

        assert merged.markdown.strip() != ""
        # 所有保留的 heading 文本都进入合并后的 markdown
        assert "Surviving Heading One" in merged.markdown
        assert "Surviving Heading Two" in merged.markdown
        # 表头仍然出现
        assert "| Col1 | Col2 |" in merged.markdown


class TestParseIntegrationCrossPageTables:
    """端到端 parse() 的跨页表格场景：连续页合并、非连续不合并。"""

    @pytest.mark.asyncio
    async def test_four_page_doc_merges_only_consecutive_tables(self):
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock()
        gateway.complete_multimodal = AsyncMock()

        # Page 1 起始表格，Page 2 同表头延续 → 必须合并；
        # Page 3 是无关内容（标题 + 段落）打断；
        # Page 4 又出现同表头表格 → 与 Page 1/2 不相邻，必须保持独立。
        gateway.complete.side_effect = [
            LLMResponse(
                content=(
                    "# Inventory\n\n"
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| A | 1 |\n"
                ),
                model="gpt-4o",
            ),
            LLMResponse(
                content=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| B | 2 |\n"
                ),
                model="gpt-4o",
            ),
            LLMResponse(
                content=(
                    "# Notes\n\n"
                    "Some unrelated narrative paragraph on page three."
                ),
                model="gpt-4o",
            ),
            LLMResponse(
                content=(
                    "| Item | Qty |\n"
                    "| --- | --- |\n"
                    "| C | 99 |\n"
                ),
                model="gpt-4o",
            ),
        ]

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw page 1", page_number=1),
                Block(type="paragraph", text="raw page 2", page_number=2),
                Block(type="paragraph", text="raw page 3", page_number=3),
                Block(type="paragraph", text="raw page 4", page_number=4),
            ],
            metadata={"file_type": "txt"},
        )

        parser = UniversalParser(llm_gateway=gateway)
        result = await parser.parse(doc)

        # LLM 被四次调用（一页一次）
        assert gateway.complete.call_count == 4

        tables = [b for b in result.blocks if b.type == "table"]
        # 仅有两张表：合并表（page 1+2）+ page 4 独立表
        assert len(tables) == 2

        merged_table = next(t for t in tables if t.page_number == 1)
        assert "A" in merged_table.text
        assert "B" in merged_table.text
        # page 4 的行不应混入 merged 表
        assert "99" not in merged_table.text

        page4_table = next(t for t in tables if t.page_number == 4)
        assert "C" in page4_table.text
        assert "99" in page4_table.text

        # 页号集合覆盖 1, 3, 4（page 2 的所有内容都属于合并表里）
        page_numbers = {b.page_number for b in result.blocks}
        assert {1, 3, 4}.issubset(page_numbers)

        # 标题在原页保留
        headings = {(h.text, h.page_number) for h in result.blocks if h.type == "heading"}
        assert ("Inventory", 1) in headings
        assert ("Notes", 3) in headings

        # markdown 包含两个表头出现两次（每张表各一次）
        assert result.markdown.count("| Item | Qty |") == 2


class TestPartialFailureDegradation:
    """任务 10.8：单页失败不丢全文，整体超时/失败回退到固定分块。

    覆盖契约：
    - 单页 LLM 失败时，成功页保留 LLM 解析结果；失败页用原始 ``parsed_doc`` 的
      纯文本按 ``fallback_chunk_chars`` 切分降级，``page_number`` 保持原页。
    - ``LLMGatewayError.reason`` 透传到 ``metadata["universal_parser"]["page_errors"]``
      （``timeout`` / ``empty`` / ``rate_limit`` 等），其它异常归入 ``"unknown"``。
    - 全部页都失败 → 整文件回退到 ``_degrade_to_plain_text``，仍打上元数据信封。
    - ``fallback_chunk_chars`` 经构造函数 / settings 接入，floor=1。
    """

    @staticmethod
    def _make_gateway_per_page(side_effects):
        """构造一个 ``complete``/``complete_multimodal`` 都按列表派发的 gateway。

        允许列表里混合 ``LLMResponse`` / ``LLMGatewayError`` / 其他异常类型。
        """
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock(side_effect=list(side_effects))
        # 多模态走 complete_multimodal；当前测试都不带页面图像，所以仅 complete 被调。
        gateway.complete_multimodal = AsyncMock(side_effect=list(side_effects))
        return gateway

    # ── 1) 单页失败：成功页保留，失败页降级 ───────────────────────────

    @pytest.mark.asyncio
    async def test_single_page_failure_preserves_successes(self):
        """三页文档中第二页超时：另两页 LLM 结果保留，第二页降级。"""
        gateway = self._make_gateway_per_page([
            LLMResponse(content="# Page 1\n\nFirst page body.", model="gpt-4o"),
            LLMGatewayError("LLM call timed out", reason="timeout"),
            LLMResponse(content="# Page 3\n\nThird page body.", model="gpt-4o"),
        ])

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw text page 1", page_number=1),
                Block(type="paragraph", text="raw text page 2 was lost", page_number=2),
                Block(type="paragraph", text="raw text page 3", page_number=3),
            ],
            metadata={"file_type": "pdf"},
        )

        parser = UniversalParser(llm_gateway=gateway)
        result = await parser.parse(doc)

        # 三页都被尝试调用了一次（旧实现遇到第一个失败就 break，不会调到 page 3）
        assert gateway.complete.call_count == 3

        # 三个页号都应在最终结果中出现
        page_numbers = {b.page_number for b in result.blocks}
        assert {1, 2, 3}.issubset(page_numbers)

        # page 1 / page 3 是 LLM 解析（含 heading），page 2 是降级段落（无 heading）
        page1_kinds = {b.type for b in result.blocks if b.page_number == 1}
        page3_kinds = {b.type for b in result.blocks if b.page_number == 3}
        page2_kinds = {b.type for b in result.blocks if b.page_number == 2}
        assert "heading" in page1_kinds
        assert "heading" in page3_kinds
        # 降级输出仅有 paragraph
        assert page2_kinds == {"paragraph"}
        # 降级文本来自原始 parsed_doc 而非 LLM
        page2_text = next(b.text for b in result.blocks if b.page_number == 2)
        assert "raw text page 2 was lost" in page2_text

        # 元数据信封记录失败状态
        meta = result.metadata["universal_parser"]
        assert meta["failed_pages"] == [2]
        assert meta["successful_pages"] == [1, 3]
        assert meta["degraded_pages"] == [2]
        assert meta["page_errors"][2] == "timeout"
        assert meta["whole_doc_degraded"] is False

    # ── 2) 全页失败：回退到 _degrade_to_plain_text ────────────────────

    @pytest.mark.asyncio
    async def test_all_pages_failure_falls_back_whole_doc(self):
        """每一页都失败时，整文件走 ``_degrade_to_plain_text`` 路径。"""
        gateway = self._make_gateway_per_page([
            LLMGatewayError("Service unavailable", reason="timeout"),
            LLMGatewayError("Service unavailable", reason="timeout"),
        ])

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="A" * 600, page_number=1),
                Block(type="paragraph", text="B" * 400, page_number=2),
            ],
            metadata={"file_type": "pdf"},
        )

        parser = UniversalParser(llm_gateway=gateway)
        result = await parser.parse(doc)

        # 全页失败 → 旧降级形状：仅有 paragraph 块、无 heading
        assert result.blocks != []
        assert result.headings_detected == 0
        assert all(b.type == "paragraph" for b in result.blocks)

        # 元数据信封仍然存在，用于运维定位
        meta = result.metadata["universal_parser"]
        assert meta["successful_pages"] == []
        assert meta["failed_pages"] == [1, 2]
        assert meta["degraded_pages"] == []  # 整文件兜底，不是按页降级
        assert meta["whole_doc_degraded"] is True
        assert meta["page_errors"][1] == "timeout"
        assert meta["page_errors"][2] == "timeout"

    # ── 3) 降级页按 fallback_chunk_chars 分段 ─────────────────────────

    @pytest.mark.asyncio
    async def test_degraded_page_respects_fallback_chunk_chars(self):
        """page 1（1500 字符）失败时，``fallback_chunk_chars=500`` → 3 个段落。"""
        gateway = self._make_gateway_per_page([
            LLMGatewayError("rate limited", reason="rate_limit"),
            LLMResponse(content="# Page 2\n\nbody", model="gpt-4o"),
        ])

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="a" * 1500, page_number=1),
                Block(type="paragraph", text="page 2 raw", page_number=2),
            ],
            metadata={"file_type": "pdf"},
        )

        parser = UniversalParser(llm_gateway=gateway, fallback_chunk_chars=500)
        result = await parser.parse(doc)

        page1_paras = [
            b for b in result.blocks
            if b.page_number == 1 and b.type == "paragraph"
        ]
        # 1500 / 500 = 3 个降级段落
        assert len(page1_paras) == 3
        for block in page1_paras:
            assert len(block.text) == 500
            assert block.text == "a" * 500

        meta = result.metadata["universal_parser"]
        assert meta["page_errors"][1] == "rate_limit"
        assert meta["degraded_pages"] == [1]

    # ── 4) timeout reason 透传 ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_reason_is_propagated(self):
        gateway = self._make_gateway_per_page([
            LLMGatewayError("page 1 timed out", reason="timeout"),
            LLMResponse(content="# ok", model="gpt-4o"),
        ])
        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw 1", page_number=1),
                Block(type="paragraph", text="raw 2", page_number=2),
            ],
            metadata={},
        )
        parser = UniversalParser(llm_gateway=gateway)
        result = await parser.parse(doc)

        assert result.metadata["universal_parser"]["page_errors"][1] == "timeout"

    # ── 5) 非 LLM 异常归入 unknown ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_non_llm_exception_bucketed_as_unknown(self):
        gateway = self._make_gateway_per_page([
            RuntimeError("boom"),
            LLMResponse(content="# ok", model="gpt-4o"),
        ])
        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw 1", page_number=1),
                Block(type="paragraph", text="raw 2", page_number=2),
            ],
            metadata={},
        )
        parser = UniversalParser(llm_gateway=gateway)
        result = await parser.parse(doc)

        meta = result.metadata["universal_parser"]
        assert meta["page_errors"][1] == "unknown"
        assert meta["failed_pages"] == [1]
        # page 2 仍然成功
        assert meta["successful_pages"] == [2]

    # ── 6) 空内容 reason 透传 ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_content_reason_is_propagated(self):
        """LLM 返回空白时 ``_parse_page`` 抛 ``LLMGatewayError(reason='empty')``。"""
        # 模拟 LLM 返回纯空白 markdown：_parse_page 内部会把它包装为 reason='empty'
        gateway = MagicMock(spec=LLMGateway)
        gateway.complete = AsyncMock(side_effect=[
            LLMResponse(content="   \n   ", model="gpt-4o"),
            LLMResponse(content="# Page 2\n\nbody", model="gpt-4o"),
        ])
        gateway.complete_multimodal = AsyncMock()

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw 1", page_number=1),
                Block(type="paragraph", text="raw 2", page_number=2),
            ],
            metadata={},
        )
        parser = UniversalParser(llm_gateway=gateway)
        result = await parser.parse(doc)

        assert result.metadata["universal_parser"]["page_errors"][1] == "empty"
        assert result.metadata["universal_parser"]["successful_pages"] == [2]

    # ── 7) 失败页若无原始块则不产出降级输出 ───────────────────────────

    @pytest.mark.asyncio
    async def test_failed_page_without_original_blocks_emits_nothing(self):
        """failed page 没有原始 block 时不产出降级段落，且不计入 degraded_pages。"""
        gateway = self._make_gateway_per_page([
            LLMResponse(content="# Page 1", model="gpt-4o"),
            # page 2 在 ``parsed_doc`` 里没有任何块（仅出现在 page_image 那种边角场景）
        ])
        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw 1", page_number=1),
            ],
            metadata={},
        )
        parser = UniversalParser(llm_gateway=gateway)

        # 直接验证 _degrade_pages 在没有 block 的页号上返回空列表
        degraded = parser._degrade_pages(doc, [99], parser.fallback_chunk_chars)
        assert degraded == []

        # 也验证 ``_degraded_pages_with_blocks`` 不会把它标进去
        assert parser._degraded_pages_with_blocks(doc, [99]) == []

    @pytest.mark.asyncio
    async def test_failed_page_without_blocks_via_parse(self):
        """端到端：失败页号在 parsed_doc 中没有 block 时不出现在 degraded_pages。"""
        gateway = MagicMock(spec=LLMGateway)
        # 通过 patch ``_process_pages`` 让 page=99 出现在 page_errors 但 parsed_doc 里没有它的块
        gateway.complete = AsyncMock()
        gateway.complete_multimodal = AsyncMock()

        doc = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="raw 1", page_number=1),
            ],
            metadata={},
        )
        parser = UniversalParser(llm_gateway=gateway)

        # 把 _process_pages 替换成一个失败页号根本不在原文档里的 stub
        async def fake_process(_doc, _pages):
            # page 1 假装成功，并伪造一个页号 99 的失败（doc 里没有这个页号）
            page1_result = PageResult(
                page_number=1,
                markdown="# Page 1\n\nbody",
                headings=[{"level": 1, "text": "Page 1"}],
                tables=[],
                success=True,
            )
            return [page1_result], {99: "timeout"}

        parser._process_pages = fake_process  # type: ignore[assignment]
        result = await parser.parse(doc)

        meta = result.metadata["universal_parser"]
        # failed_pages 仍然记录 99（这是 page_errors 的回声）
        assert meta["failed_pages"] == [99]
        # 但 degraded_pages 为空，因为没有原始块可降级
        assert meta["degraded_pages"] == []
        # 不应有 page_number=99 的块被注入
        assert all(b.page_number != 99 for b in result.blocks)

    # ── 8) fallback_chunk_chars 构造函数覆盖 ──────────────────────────

    @patch("app.services.universal_parser.get_settings")
    def test_constructor_fallback_chunk_chars_wins_over_settings(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS=500,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway, fallback_chunk_chars=200)
        assert parser.fallback_chunk_chars == 200

    # ── 9) 0 / 负值 floor 至 1 ─────────────────────────────────────────

    @patch("app.services.universal_parser.get_settings")
    @pytest.mark.parametrize("override", [0, -1, -100])
    def test_fallback_chunk_chars_floor_is_one(self, mock_settings, override):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS=0,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway, fallback_chunk_chars=override)
        assert parser.fallback_chunk_chars == 1

    # ── 10) settings 接入：未传构造参数时读 settings ───────────────────

    @patch("app.services.universal_parser.get_settings")
    def test_settings_fallback_chunk_chars_is_read(self, mock_settings):
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS=750,
        )
        gateway = MagicMock(spec=LLMGateway)
        parser = UniversalParser(llm_gateway=gateway)
        assert parser.fallback_chunk_chars == 750
