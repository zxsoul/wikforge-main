"""Universal Parser「无 Profile 场景」端到端单元测试（任务 10.10）。

本文件是 Task 10 的最后一块拼图：在 10.1 ~ 10.9 已经覆盖了内部子例程的
基础上，把所有部件按真实管线的顺序串起来跑一遍——

1. ``ProfileMatcher`` 在没有规则命中时回落到 ``generic-text`` 兜底；
2. 触发器 ``should_run_universal_parser`` 命中 ``no_profile_match``；
3. ``UniversalParser.parse`` 端到端跑完每页 LLM 调用 + 合并；
4. ``UniversalParser.suggest_profile`` 输出可与 ``profile_matcher`` 直接互转的
   候选 envelope；
5. 即使个别页 LLM 失败，整篇文档仍然产出 blocks 并把失败原因写进 metadata。

所有外部依赖（pdf2image / LibreOffice / litellm / Postgres）都被 mock；
测试在 CI 上无需任何系统级二进制即可通过。

Validates: Requirements 16
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 与 ``test_universal_parser.py`` 一致的 pdf2image stub 注入：避免无 Poppler
# 的开发机在 import 阶段就失败。
if "pdf2image" not in sys.modules:
    _pdf2image_stub = ModuleType("pdf2image")
    _pdf2image_stub.convert_from_path = lambda *a, **kw: []  # type: ignore[attr-defined]
    sys.modules["pdf2image"] = _pdf2image_stub

from app.services.llm_gateway import (  # noqa: E402
    LLMGateway,
    LLMGatewayError,
    LLMResponse,
)
from app.services.profile_matcher import (  # noqa: E402
    ProfileMatcher,
    profile_from_dict,
    profile_to_dict,
)
from app.services.universal_parser import UniversalParser  # noqa: E402
from app.services.universal_parser_trigger import (  # noqa: E402
    TRIGGER_NO_PROFILE_MATCH,
    should_run_universal_parser,
)
from tests.fixtures.universal_parser import (  # noqa: E402
    make_chinese_unknown_layout_document,
    make_scanned_pdf_like_document,
    make_unknown_format_document,
)


# ─── 共用工具 ──────────────────────────────────────────────────────────


def _make_parser(*, llm_gateway=None) -> UniversalParser:
    """Build a UniversalParser with mocked settings + injected gateway.

    与 ``test_universal_parser.py`` 的 ``_make_parser`` 保持同样的形状：
    把 ``get_settings`` 和 ``UNIVERSAL_PARSER_*`` 字段都固定下来，避免本机
    环境变量污染测试。
    """
    with patch("app.services.universal_parser.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LITELLM_MODEL="gpt-4o",
            QUALITY_FALLBACK_THRESHOLD=0.7,
            UNIVERSAL_PARSER_PAGE_DPI=150,
            UNIVERSAL_PARSER_LIBREOFFICE_TIMEOUT=60,
            UNIVERSAL_PARSER_MAX_RAW_TEXT_CHARS=3000,
            UNIVERSAL_PARSER_FALLBACK_CHUNK_CHARS=500,
            UNIVERSAL_PARSER_VISION_MODEL="",
            UNIVERSAL_PARSER_TEXT_MODEL="",
        )
        gateway = llm_gateway or _build_mock_gateway()
        return UniversalParser(llm_gateway=gateway)


def _build_mock_gateway() -> MagicMock:
    """Create a typed LLMGateway mock with async-callable surfaces."""
    gateway = MagicMock(spec=LLMGateway)
    gateway.complete = AsyncMock()
    gateway.complete_multimodal = AsyncMock()
    return gateway


def _markdown_for_page(page_number: int, fixture_name: str) -> str:
    """生成一页形状真实的 Markdown 输出。

    每页都返回「heading + 一段正文」，让后续断言能稳定地从输出里找到该页
    的 marker（``page-{N}-{fixture}``）。
    """
    return (
        f"# Section for page {page_number}\n\n"
        f"Body content for {fixture_name} page {page_number}, marker "
        f"page-{page_number}-{fixture_name}."
    )


def _gateway_returning_per_page(fixture_name: str) -> MagicMock:
    """Build a mock gateway whose ``complete`` / ``complete_multimodal`` returns
    per-page Markdown derived from the call count.

    每次被调用时按调用顺序返回 ``page 1`` / ``page 2`` / ...，调用方不需要
    预先知道每页的 raw_text 长度。
    """
    gateway = _build_mock_gateway()

    counter = {"complete": 0, "multimodal": 0}

    async def complete_side_effect(*args, **kwargs):
        counter["complete"] += 1
        page_number = counter["complete"]
        return LLMResponse(
            content=_markdown_for_page(page_number, fixture_name),
            model="gpt-4o",
        )

    async def multimodal_side_effect(*args, **kwargs):
        counter["multimodal"] += 1
        page_number = counter["multimodal"]
        return LLMResponse(
            content=_markdown_for_page(page_number, fixture_name),
            model="gpt-4o",
        )

    gateway.complete.side_effect = complete_side_effect
    gateway.complete_multimodal.side_effect = multimodal_side_effect
    return gateway


# ─── Scenario A — ProfileMatcher 兜底 + 触发器命中 ────────────────────


class TestProfileMatcherFallsBackOnUnknownDocuments:
    """没有可命中规则时 ``ProfileMatcher`` 必须回落到 ``generic-text``，
    并被触发器识别为「应运行 LLM 兜底」。"""

    @pytest.mark.parametrize(
        "fixture_factory, name",
        [
            (make_unknown_format_document, "unknown-format"),
            (make_scanned_pdf_like_document, "scanned-pdf"),
            (make_chinese_unknown_layout_document, "chinese-unknown"),
        ],
    )
    def test_profile_matcher_returns_generic_text_fallback(
        self, fixture_factory, name
    ):
        """在三种无 Profile 场景下，``ProfileMatcher`` 都返回 generic-text 兜底。

        因为我们没有给 matcher 注入任何具体 Profile，``_get_default_profile``
        会构造硬编码兜底（``id='default'``, ``name='generic-text'``）。
        """
        matcher = _matcher_with_no_profiles()
        doc = fixture_factory()

        matched = matcher.match(doc, filename=doc.metadata.get("file_path", ""))

        assert matched.name == "generic-text"
        # 硬编码兜底 ID 是 ``"default"``。
        assert matched.id == "default"

    @pytest.mark.parametrize(
        "fixture_factory",
        [
            make_unknown_format_document,
            make_scanned_pdf_like_document,
            make_chinese_unknown_layout_document,
        ],
    )
    def test_trigger_says_run_universal_parser(self, fixture_factory):
        """``should_run_universal_parser`` 在 generic-text 兜底场景下命中
        ``no_profile_match``。"""
        # 模拟 ``profile_match`` 任务的产出：在 pipeline 里 generic-text 兜底
        # 会被改写成 ``profile_id=None``，但触发器对 "id is None" 与
        # "name == generic-text" 任一命中都返回 True；此处验证「name 命中」。
        ok, reasons = should_run_universal_parser(
            profile_id=None,
            profile_name="generic-text",
            quality_score=None,
        )
        assert ok is True
        assert reasons == [TRIGGER_NO_PROFILE_MATCH]

        # 顺带验证：当 profile_id 已被改写成 None 时同样命中。
        ok2, reasons2 = should_run_universal_parser(
            profile_id=None,
            profile_name=None,
            quality_score=None,
        )
        assert ok2 is True
        assert reasons2 == [TRIGGER_NO_PROFILE_MATCH]


def _matcher_with_no_profiles() -> ProfileMatcher:
    """构造一个空 Profile 列表的 ``ProfileMatcher``，让 ``match`` 强制走兜底。"""
    return ProfileMatcher(profiles=[])


# ─── Scenario B — UniversalParser.parse 端到端 ────────────────────────


class TestUniversalParserEndToEndOnNoProfileFixtures:
    """在三种「无 Profile」场景下，``UniversalParser.parse`` 必须产出
    覆盖所有原始页码的 ``ProcessedDocument``，并且把每页都标记为
    ``successful_pages``。"""

    @pytest.mark.asyncio
    async def test_unknown_format_document_full_parse(self):
        doc = make_unknown_format_document()
        gateway = _gateway_returning_per_page("unknown-format")
        parser = _make_parser(llm_gateway=gateway)

        # 无图像路径：``_get_page_image`` 返回 None → 走纯文本调用。
        with patch(
            "app.services.universal_parser.UniversalParser._get_page_image",
            return_value=None,
        ):
            result = await parser.parse(doc)

        # 输出非空。
        assert result.blocks != []
        assert result.markdown != ""

        # 所有原始页码都出现在输出里。
        original_pages = sorted({b.page_number for b in doc.blocks})
        output_pages = sorted({b.page_number for b in result.blocks})
        assert set(original_pages) <= set(output_pages)

        # metadata.universal_parser 全部页都成功。
        envelope = result.metadata["universal_parser"]
        assert envelope["successful_pages"] == original_pages
        assert envelope["failed_pages"] == []
        assert envelope["whole_doc_degraded"] is False

        # 标题数 ≥ LLM 实际声明的 heading 数（每页 1 个 heading,
        # mock 返回了 ``# Section for page N``）。
        assert result.headings_detected >= len(original_pages)

        # 走的是纯文本通道。
        assert gateway.complete.await_count == len(original_pages)
        assert gateway.complete_multimodal.await_count == 0

    @pytest.mark.asyncio
    async def test_scanned_pdf_like_document_uses_multimodal_path(self):
        """扫描件场景：``_get_page_image`` 返回伪造 PNG → 走 multimodal。"""
        doc = make_scanned_pdf_like_document()
        gateway = _gateway_returning_per_page("scanned-pdf")
        parser = _make_parser(llm_gateway=gateway)

        with patch(
            "app.services.universal_parser.UniversalParser._get_page_image",
            return_value=b"fake-png-data",
        ):
            result = await parser.parse(doc)

        # 即便原始 raw_text 大多为空 / "???"，LLM 仍然从图像里产出 Markdown。
        assert result.blocks != []
        assert result.markdown != ""

        original_pages = sorted({b.page_number for b in doc.blocks})
        envelope = result.metadata["universal_parser"]
        assert envelope["successful_pages"] == original_pages
        assert envelope["failed_pages"] == []
        assert envelope["whole_doc_degraded"] is False

        # 走的是 multimodal 通道。
        assert gateway.complete_multimodal.await_count == len(original_pages)
        assert gateway.complete.await_count == 0

    @pytest.mark.asyncio
    async def test_chinese_unknown_layout_full_parse(self):
        doc = make_chinese_unknown_layout_document()
        gateway = _gateway_returning_per_page("chinese-unknown")
        parser = _make_parser(llm_gateway=gateway)

        with patch(
            "app.services.universal_parser.UniversalParser._get_page_image",
            return_value=None,
        ):
            result = await parser.parse(doc)

        original_pages = sorted({b.page_number for b in doc.blocks})
        envelope = result.metadata["universal_parser"]
        assert envelope["successful_pages"] == original_pages
        assert envelope["failed_pages"] == []
        assert envelope["whole_doc_degraded"] is False
        assert result.headings_detected >= len(original_pages)


# ─── Scenario C — suggest_profile 候选 envelope 形状 ───────────────────


class TestSuggestProfileOnNoProfileFixtures:
    """对每个场景 fixture 跑完 ``parse`` 之后，``suggest_profile`` 输出的
    envelope 必须满足设计契约（10.5 / 10.6 已建立的形状）。"""

    @pytest.mark.parametrize(
        "fixture_factory, expected_file_type",
        [
            (make_unknown_format_document, "txt"),
            (make_scanned_pdf_like_document, "pdf"),
            (make_chinese_unknown_layout_document, "docx"),
        ],
    )
    @pytest.mark.asyncio
    async def test_envelope_shape_after_parse(
        self, fixture_factory, expected_file_type
    ):
        doc = fixture_factory()
        gateway = _gateway_returning_per_page(expected_file_type)
        parser = _make_parser(llm_gateway=gateway)

        # multimodal 路径在测试里不重要，关键是 parse 走通；统一返回 None。
        with patch(
            "app.services.universal_parser.UniversalParser._get_page_image",
            return_value=None if expected_file_type != "pdf" else b"fake-png",
        ):
            processed = await parser.parse(doc)

        envelope = await parser.suggest_profile(processed)

        # ── 顶层结构 ─────────────────────────────────────────────────
        assert set(envelope.keys()) == {"profile", "metadata"}

        meta = envelope["metadata"]
        assert meta["status"] == "pending_approval"
        assert meta["source"] == "universal_parser"

        # ── profile 子字典 ───────────────────────────────────────────
        profile = envelope["profile"]
        assert profile["enabled"] is False
        assert isinstance(profile.get("description"), str)
        assert profile["description"]  # 非空字符串

        # 命名约定：``auto-generated-{file_type}-{N}p``。
        n_pages = max(b.page_number for b in processed.blocks if b.page_number)
        assert profile["name"] == f"auto-generated-{expected_file_type}-{n_pages}p"

        # protect_patterns 默认非空（至少包含单位 / 公差正则）。
        protect = profile["chunking"]["protect_patterns"]
        assert isinstance(protect, list) and len(protect) > 0

        # ── round-trip 与 profile_matcher 互通 ──────────────────────
        config = profile_from_dict(profile)
        round_tripped = profile_to_dict(config)
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
            assert round_tripped[key] == profile[key], (
                f"round-trip drift on key {key!r}: "
                f"{round_tripped[key]!r} vs {profile[key]!r}"
            )


# ─── Scenario D — 部分失败场景下仍然产出 ───────────────────────────────


class TestPartialFailureOnNoProfileDocument:
    """单页 LLM 失败时，``parse`` 必须把成功页 + 失败页的降级 chunk 都返回，
    并把失败原因写进 metadata。"""

    @pytest.mark.asyncio
    async def test_one_page_timeout_other_pages_succeed(self):
        doc = make_unknown_format_document()
        gateway = _build_mock_gateway()

        # 第 2 页（call index 2，1-based）让 LLM 抛 timeout，其余页正常。
        call_count = {"n": 0}

        async def complete_side_effect(*args, **kwargs):
            call_count["n"] += 1
            page_number = call_count["n"]
            if page_number == 2:
                raise LLMGatewayError("upstream timeout", reason="timeout")
            return LLMResponse(
                content=f"# Page {page_number}\n\nBody for page {page_number}.",
                model="gpt-4o",
            )

        gateway.complete.side_effect = complete_side_effect

        parser = _make_parser(llm_gateway=gateway)

        with patch(
            "app.services.universal_parser.UniversalParser._get_page_image",
            return_value=None,
        ):
            result = await parser.parse(doc)

        original_pages = sorted({b.page_number for b in doc.blocks})
        output_pages = {b.page_number for b in result.blocks}

        # 所有原始页码仍然出现在输出里（成功页来自 LLM，失败页来自降级文本）。
        assert set(original_pages) <= output_pages

        envelope = result.metadata["universal_parser"]
        # 第 2 页的 reason 必须是 ``timeout``。
        assert envelope["page_errors"][2] == "timeout"
        # 整篇没降级（部分成功路径）。
        assert envelope["whole_doc_degraded"] is False
        # 失败页清单仅包含第 2 页。
        assert envelope["failed_pages"] == [2]
        # 成功页是其它三页。
        assert envelope["successful_pages"] == [
            p for p in original_pages if p != 2
        ]


# ─── Scenario E — 触发条件 + ProfileMatcher 协同（smoke） ─────────────


class TestPipelineTriggerSmoke:
    """对 ``ProfileMatcher.match`` 的兜底返回值跑一遍触发器，验证「没有具体
    Profile → 应运行 Universal Parser」的端到端决策路径。"""

    @pytest.mark.asyncio
    async def test_unknown_format_would_trigger_llm_path(self):
        doc = make_unknown_format_document()

        matcher = ProfileMatcher(profiles=[])
        matched = matcher.match(doc, filename=doc.metadata.get("file_path", ""))

        # 兜底名 + 默认 ID。
        assert matched.name == "generic-text"

        # pipeline 在拿到 generic-text 兜底后会把 profile_id 改写为 None；
        # 这里两种写法都应该命中触发器。
        ok, reasons = should_run_universal_parser(
            profile_id=None,
            profile_name=matched.name,
            quality_score=None,
        )
        assert ok is True
        assert TRIGGER_NO_PROFILE_MATCH in reasons
