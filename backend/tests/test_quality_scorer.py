"""Unit tests for QualityScorer: multi-dimensional quality scoring and review queue logic.

Tests cover:
- ParseQualityScore data structure (overall, components, issues)
- Text retention rate scoring
- Heading detection rate scoring
- Table completeness scoring (cell fill rate, cross-page merge)
- Numeric protection rate scoring
- Boilerplate removal rate scoring
- Weighted overall score calculation (30% text + 25% heading + 20% table + 15% numeric + 10% noise)
- Review queue enqueue logic (score < 0.7 triggers review)
"""

import uuid

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from app.services.document_processor import ProcessedBlock, ProcessedDocument
from app.services.parsers.base import Block, ParsedDocument
from app.services.profile_matcher import (
    BoilerplateConfig,
    ChunkingConfig,
    DocumentProfileConfig,
    HeadingRule,
    MatchRules,
    TableConfig,
)
from app.services.quality_scorer import (
    DEFAULT_REVIEW_THRESHOLD,
    WEIGHT_BOILERPLATE_REMOVAL,
    WEIGHT_HEADING_DETECTION,
    WEIGHT_NUMERIC_PROTECTION,
    WEIGHT_TABLE_COMPLETENESS,
    WEIGHT_TEXT_RETENTION,
    ParseQualityScore,
    QualityScorer,
    _extract_numbers,
    _visible_char_count,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def scorer() -> QualityScorer:
    """Default quality scorer with standard weights."""
    return QualityScorer()


@pytest.fixture
def default_profile() -> DocumentProfileConfig:
    """Default generic-text profile."""
    return DocumentProfileConfig(
        id="default",
        name="generic-text",
        description="通用文本文档",
        priority=0,
        enabled=True,
        match_rules=MatchRules(),
        heading_rules=[
            HeadingRule(pattern=r"^#{1,6}\s+", level=0, strip_pattern=False),
        ],
        boilerplate=BoilerplateConfig(),
        tables=TableConfig(),
        chunking=ChunkingConfig(),
    )


@pytest.fixture
def chinese_spec_profile() -> DocumentProfileConfig:
    """Chinese technical specification profile with heading rules."""
    return DocumentProfileConfig(
        id="chinese-spec",
        name="chinese-technical-spec",
        description="中式技术规范",
        priority=10,
        enabled=True,
        match_rules=MatchRules(),
        heading_rules=[
            HeadingRule(pattern=r"^[一二三四五六七八九十]+[、．.]", level=1, strip_pattern=False),
            HeadingRule(pattern=r"^\([一二三四五六七八九十]+\)", level=2, strip_pattern=False),
            HeadingRule(pattern=r"^\d+[、．.]", level=3, strip_pattern=False),
        ],
        boilerplate=BoilerplateConfig(detection_mode="both"),
        tables=TableConfig(),
        chunking=ChunkingConfig(),
    )


# ─── Test ParseQualityScore Data Structure ─────────────────────────────


class TestParseQualityScore:
    """Tests for ParseQualityScore data structure."""

    def test_default_values(self):
        """ParseQualityScore has sensible defaults."""
        score = ParseQualityScore()
        assert score.overall == 0.0
        assert score.components == {}
        assert score.issues == []

    def test_to_dict(self):
        """to_dict produces a JSON-serializable dictionary."""
        score = ParseQualityScore(
            overall=0.85,
            components={
                "text_retention": 0.95,
                "heading_detection": 0.80,
                "table_completeness": 0.70,
                "numeric_protection": 1.0,
                "boilerplate_removal": 0.90,
            },
            issues=["标题识别率偏低"],
        )
        d = score.to_dict()
        assert d["overall"] == 0.85
        assert d["components"]["text_retention"] == 0.95
        assert d["issues"] == ["标题识别率偏低"]

    def test_to_dict_rounds_values(self):
        """to_dict rounds float values to 4 decimal places."""
        score = ParseQualityScore(
            overall=0.123456789,
            components={"text_retention": 0.987654321},
        )
        d = score.to_dict()
        assert d["overall"] == 0.1235
        assert d["components"]["text_retention"] == 0.9877

    def test_construction_preserves_all_fields(self):
        """构造时给定全部字段 → 字段被原样保留。"""
        score = ParseQualityScore(
            overall=0.73,
            components={
                "text_retention": 0.9,
                "heading_detection": 0.6,
                "table_completeness": 0.8,
                "numeric_protection": 0.95,
                "boilerplate_removal": 0.4,
            },
            issues=["标题识别率偏低", "噪声去除率偏低"],
        )
        assert score.overall == 0.73
        assert score.components["text_retention"] == 0.9
        assert score.components["heading_detection"] == 0.6
        assert len(score.issues) == 2
        assert "标题识别率偏低" in score.issues

    def test_missing_components_and_issues_default_to_empty(self):
        """缺省 components / issues 时落入空 dict、空 list。"""
        score = ParseQualityScore(overall=0.5)
        assert score.components == {}
        assert score.issues == []

    def test_explicit_none_components_and_issues_coerced_to_empty(self):
        """显式传 None 也会被 __post_init__ 兜底为默认空集合。"""
        score = ParseQualityScore(overall=0.5, components=None, issues=None)  # type: ignore[arg-type]
        assert score.components == {}
        assert score.issues == []

    def test_overall_clamped_below_zero(self):
        """overall < 0 时被夹紧到 0.0。"""
        score = ParseQualityScore(overall=-0.5)
        assert score.overall == 0.0

    def test_overall_clamped_above_one(self):
        """overall > 1 时被夹紧到 1.0。"""
        score = ParseQualityScore(overall=1.5)
        assert score.overall == 1.0

    def test_overall_clamp_preserves_in_range_values(self):
        """overall ∈ [0, 1] 的值不被改写。"""
        for value in (0.0, 0.25, 0.5, 0.7, 1.0):
            assert ParseQualityScore(overall=value).overall == value

    def test_to_dict_from_dict_round_trip(self):
        """to_dict() → from_dict() 往返得到等价实例。"""
        original = ParseQualityScore(
            overall=0.85,
            components={
                "text_retention": 0.95,
                "heading_detection": 0.80,
                "table_completeness": 0.70,
                "numeric_protection": 1.0,
                "boilerplate_removal": 0.90,
            },
            issues=["标题识别率偏低"],
        )
        rebuilt = ParseQualityScore.from_dict(original.to_dict())
        # to_dict 会按 4 位小数取整，于是比较时按相同精度做
        assert rebuilt.overall == pytest.approx(original.overall, abs=1e-4)
        assert set(rebuilt.components.keys()) == set(original.components.keys())
        for key, value in original.components.items():
            assert rebuilt.components[key] == pytest.approx(value, abs=1e-4)
        assert rebuilt.issues == original.issues

    def test_from_dict_handles_none(self):
        """from_dict(None) 回退为默认实例（对应 JSONB 列为 NULL 的情况）。"""
        score = ParseQualityScore.from_dict(None)
        assert score.overall == 0.0
        assert score.components == {}
        assert score.issues == []

    def test_from_dict_handles_partial_dict(self):
        """from_dict 容忍缺失键。"""
        score = ParseQualityScore.from_dict({"overall": 0.42})
        assert score.overall == 0.42
        assert score.components == {}
        assert score.issues == []

    def test_importable_from_services_module(self):
        """ParseQualityScore 可从 app.services.quality_scorer 模块导入。"""
        from app.services import quality_scorer as qs_module

        assert hasattr(qs_module, "ParseQualityScore")
        assert qs_module.ParseQualityScore is ParseQualityScore


# ─── Test Helper Functions ─────────────────────────────────────────────


class TestHelperFunctions:
    """Tests for helper utility functions."""

    def test_visible_char_count_basic(self):
        """Counts non-whitespace characters."""
        assert _visible_char_count("hello world") == 10
        assert _visible_char_count("  spaces  ") == 6
        assert _visible_char_count("") == 0
        assert _visible_char_count("   ") == 0

    def test_visible_char_count_newlines(self):
        """Handles newlines and tabs as whitespace."""
        assert _visible_char_count("a\nb\tc") == 3

    def test_extract_numbers_basic(self):
        """Extracts basic numeric values."""
        numbers = _extract_numbers("温度为 25°C，压力 100kPa")
        assert "25°C" in numbers
        assert "100kPa" in numbers

    def test_extract_numbers_with_units(self):
        """Extracts numbers with complex units."""
        text = "偏差 0.05mm/m，直径 0.002D，公差 ±10mm"
        numbers = _extract_numbers(text)
        assert "0.05mm/m" in numbers
        assert "0.002D" in numbers

    def test_extract_numbers_ranges(self):
        """Extracts numeric ranges."""
        numbers = _extract_numbers("角度 55°~65°")
        # The range may be extracted as one token or two separate tokens
        all_text = " ".join(numbers)
        assert "55" in all_text and "65" in all_text

    def test_extract_numbers_empty(self):
        """Returns empty list for text without numbers."""
        assert _extract_numbers("这是一段没有数字的文本") == []


# ─── Test Text Retention Scoring ───────────────────────────────────────


class TestTextRetentionScoring:
    """Tests for text retention rate scoring."""

    def test_perfect_retention(self, scorer, default_profile):
        """Full score when all text is preserved."""
        original = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="这是第一段文本内容"),
                Block(type="paragraph", text="这是第二段文本内容"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text="这是第一段文本内容"),
                ProcessedBlock(type="paragraph", text="这是第二段文本内容"),
            ]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == 1.0

    def test_partial_retention(self, scorer, default_profile):
        """Partial score when some text is lost."""
        original = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="这是一段很长的文本内容,包含很多字符"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text="这是一段文本"),
            ]
        )
        score = scorer.score(original, processed, default_profile)
        assert 0.0 < score.components["text_retention"] < 1.0

    def test_noise_blocks_excluded(self, scorer, default_profile):
        """Noise blocks in processed output are excluded from retention calc."""
        original = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="正文内容"),
                Block(type="paragraph", text="页眉噪声"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text="正文内容", is_noise=False),
                ProcessedBlock(type="paragraph", text="页眉噪声", is_noise=True),
            ]
        )
        score = scorer.score(original, processed, default_profile)
        # Only "正文内容" counts, which is half of original
        assert score.components["text_retention"] < 1.0

    # ─── Focused tests for text_retention algorithm (Task 11.2) ────────

    def test_text_retention_full_preservation(self, scorer, default_profile):
        """original == processed visible chars → score 1.0."""
        text = "原文与清洗后内容完全一致"
        original = ParsedDocument(blocks=[Block(type="paragraph", text=text)])
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=text)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == 1.0

    def test_text_retention_partial_loss(self, scorer, default_profile):
        """processed has 80% of original visible chars → score ≈ 0.8."""
        # 10 visible chars → 8 visible chars; whitespace surrounding does
        # not count, so the ratio is exactly 0.8.
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="ABCDEFGHIJ")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="ABCDEFGH")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == pytest.approx(0.8, abs=0.01)

    def test_text_retention_total_loss(self, scorer, default_profile):
        """processed empty, original has content → 0.0 with 'text_lost' issue."""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="原文有实质内容")]
        )
        processed = ProcessedDocument(blocks=[])
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == 0.0
        assert "text_lost" in score.issues

    def test_text_retention_original_empty(self, scorer, default_profile):
        """both empty → 1.0 with 'original_empty' issue (vacuous retention)."""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="   \n\t")]
        )
        processed = ProcessedDocument(blocks=[])
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == 1.0
        assert "original_empty" in score.issues
        # Sanity: text_lost must not also fire — original was empty, not lost.
        assert "text_lost" not in score.issues

    def test_text_retention_capped_at_one(self, scorer, default_profile):
        """processed has more visible chars than original → capped at 1.0.

        Markdown rendering can legitimately add characters (e.g. heading
        ``#`` markers, ``**bold**`` wrappers). Since this dimension
        measures *preservation*, growth must not read as ">100%".
        """
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="标题")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="heading", text="# **标题**", heading_level=1)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == 1.0

    def test_text_retention_ignores_whitespace(self, scorer, default_profile):
        """Same visible chars but more whitespace in processed → still 1.0.

        ``_visible_char_count`` strips all whitespace, so reflowing or
        re-indenting text must not affect the retention score.
        """
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="一二三四五")]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(
                    type="paragraph",
                    text="  一  二\n三\t四   五  \n",
                )
            ]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["text_retention"] == 1.0


# ─── Test Heading Detection Scoring ───────────────────────────────────


class TestHeadingDetectionScoring:
    """Tests for heading detection rate scoring."""

    def test_all_headings_detected(self, scorer, chinese_spec_profile):
        """Full score when all expected headings are detected."""
        original = ParsedDocument(
            blocks=[
                Block(type="heading", text="一、总则", style={"heading_level": 1}),
                Block(type="heading", text="二、范围", style={"heading_level": 1}),
                Block(type="paragraph", text="正文内容"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、总则", heading_level=1),
                ProcessedBlock(type="heading", text="二、范围", heading_level=1),
                ProcessedBlock(type="paragraph", text="正文内容"),
            ],
            headings_detected=2,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        assert score.components["heading_detection"] == 1.0

    def test_no_headings_expected(self, scorer, default_profile):
        """Full score when no headings are expected."""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="纯文本内容，没有标题")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="纯文本内容，没有标题")],
            headings_detected=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["heading_detection"] == 1.0

    def test_partial_heading_detection(self, scorer, chinese_spec_profile):
        """Partial score when some headings are missed."""
        original = ParsedDocument(
            blocks=[
                Block(type="heading", text="一、总则", style={"heading_level": 1}),
                Block(type="heading", text="二、范围", style={"heading_level": 1}),
                Block(type="heading", text="三、定义", style={"heading_level": 1}),
                Block(type="paragraph", text="正文"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、总则", heading_level=1),
                ProcessedBlock(type="paragraph", text="二、范围"),  # Missed as heading
                ProcessedBlock(type="paragraph", text="三、定义"),  # Missed as heading
                ProcessedBlock(type="paragraph", text="正文"),
            ],
            headings_detected=1,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        # 1 detected out of 3 expected
        assert score.components["heading_detection"] == pytest.approx(1 / 3, abs=0.01)

    # ─── Focused tests for heading_detection algorithm (Task 11.3) ─────

    def test_heading_detection_total_loss(self, scorer, chinese_spec_profile):
        """expected > 0, detected = 0 → score 0.0 与"标题识别率偏低" issue。"""
        original = ParsedDocument(
            blocks=[
                Block(type="heading", text="一、总则", style={"heading_level": 1}),
                Block(type="heading", text="二、范围", style={"heading_level": 1}),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text="一、总则"),
                ProcessedBlock(type="paragraph", text="二、范围"),
            ],
            headings_detected=0,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        assert score.components["heading_detection"] == 0.0
        # 有具体的"标题识别率偏低"说明文本被记录到 issues
        assert any("标题识别率偏低" in issue for issue in score.issues)

    def test_heading_detection_capped_at_one(self, scorer, chinese_spec_profile):
        """detected > expected → 截断到 1.0（避免出现 >100% 的"识别率"）。"""
        original = ParsedDocument(
            blocks=[
                Block(type="heading", text="一、总则", style={"heading_level": 1}),
            ]
        )
        # processed 报告检测到的标题比 original 多（例如把段落首行误判为标题）
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、总则", heading_level=1),
                ProcessedBlock(type="heading", text="额外标题", heading_level=2),
                ProcessedBlock(type="heading", text="另一个标题", heading_level=2),
            ],
            headings_detected=3,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        assert score.components["heading_detection"] == 1.0

    def test_heading_detection_uses_profile_rules_when_block_type_not_heading(
        self, scorer, chinese_spec_profile
    ):
        """原文 block.type 都是 paragraph，但内容匹配 profile 正则 → 仍被计入预期标题。"""
        original = ParsedDocument(
            blocks=[
                # 这两段文字与 chinese_spec_profile 的正则匹配，应被计为预期标题
                Block(type="paragraph", text="一、总则"),
                Block(type="paragraph", text="(一)适用范围"),
                # 不匹配任何标题规则的普通段落
                Block(type="paragraph", text="本文档适用于通用情况。"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、总则", heading_level=1),
                ProcessedBlock(type="heading", text="(一)适用范围", heading_level=2),
                ProcessedBlock(type="paragraph", text="本文档适用于通用情况。"),
            ],
            headings_detected=2,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        # 2 个匹配规则的预期标题，全部识别 → 1.0
        assert score.components["heading_detection"] == 1.0

    def test_heading_detection_rule_match_counts_each_block_once(
        self, scorer, chinese_spec_profile
    ):
        """同一段文字若同时匹配多条规则，只算 1 个预期标题（首匹配后 break）。"""
        # "1、" 同时匹配 chinese_spec_profile 中的 r"^\d+[、．.]"，但只该计数一次。
        # 这里构造内容让某些段落能匹配多条规则中的一条；不应重复计数。
        original = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="1、第一条"),
                Block(type="paragraph", text="2、第二条"),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="1、第一条", heading_level=3),
                ProcessedBlock(type="heading", text="2、第二条", heading_level=3),
            ],
            headings_detected=2,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        # 预期 = 2，识别 = 2 → 1.0；不会因"块同时匹配多条规则"而被算成 4
        assert score.components["heading_detection"] == 1.0

    def test_heading_detection_below_threshold_records_issue(
        self, scorer, chinese_spec_profile
    ):
        """ratio < 0.7 时 issues 列表中包含'标题识别率偏低'相关文本。"""
        original = ParsedDocument(
            blocks=[
                Block(type="heading", text="一、总则", style={"heading_level": 1}),
                Block(type="heading", text="二、范围", style={"heading_level": 1}),
                Block(type="heading", text="三、定义", style={"heading_level": 1}),
                Block(type="heading", text="四、要求", style={"heading_level": 1}),
                Block(type="heading", text="五、附录", style={"heading_level": 1}),
            ]
        )
        # 5 个预期，仅识别 1 个 → ratio = 0.2，远低于 0.7
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="heading", text="一、总则", heading_level=1),
            ],
            headings_detected=1,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        assert score.components["heading_detection"] == pytest.approx(0.2, abs=0.01)
        assert any("标题识别率偏低" in issue for issue in score.issues)

    def test_heading_detection_at_threshold_no_issue(self, scorer, chinese_spec_profile):
        """ratio == 0.7 时不记录"标题识别率偏低"（与阈值文档一致：低于 0.7 才提示）。"""
        # 10 个预期标题，识别 7 个 → 比率正好 0.7
        original_blocks = [
            Block(
                type="heading", text=f"标题 {i}", style={"heading_level": 1}
            )
            for i in range(10)
        ]
        original = ParsedDocument(blocks=original_blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(
                    type="heading", text=f"标题 {i}", heading_level=1
                )
                for i in range(7)
            ],
            headings_detected=7,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        assert score.components["heading_detection"] == pytest.approx(0.7, abs=0.01)
        # 边界：恰好 0.7 不应触发"识别率偏低"提示
        assert not any("标题识别率偏低" in issue for issue in score.issues)


# ─── Test Table Completeness Scoring ──────────────────────────────────


class TestTableCompletenessScoring:
    """Tests for table completeness scoring."""

    def test_no_tables(self, scorer, default_profile):
        """Full score when document has no tables."""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="纯文本")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="纯文本")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] == 1.0

    def test_table_preserved_with_full_cells(self, scorer, default_profile):
        """High score when table is preserved with all cells filled."""
        table_md = "| 名称 | 数值 | 单位 |\n| --- | --- | --- |\n| 温度 | 25 | °C |\n| 压力 | 100 | kPa |"
        original = ParsedDocument(
            blocks=[Block(type="table", text=table_md, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=table_md, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] > 0.5

    def test_table_lost(self, scorer, default_profile):
        """Zero score when original tables are lost in processing."""
        original = ParsedDocument(
            blocks=[Block(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="A B 1 2")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] == 0.0

    def test_cross_page_table_merge(self, scorer, default_profile):
        """Merge score when adjacent-page tables are merged."""
        original = ParsedDocument(
            blocks=[
                Block(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |", page_number=1),
                Block(type="table", text="| A | B |\n| --- | --- |\n| 3 | 4 |", page_number=2),
            ]
        )
        # After merge: single table
        merged_table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=merged_table, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] > 0.7

    # ─── Focused tests for table_completeness algorithm (Task 11.4) ────
    #
    # 算法（参见 quality_scorer._score_table_completeness）：
    #   score = 0.6 * cell_fill_rate + 0.4 * cross_page_merge_score
    # 其中：
    #   - cell_fill_rate = 已填充单元格 / 总单元格（跨所有处理后的表格汇总）
    #   - cross_page_merge_score = min(已合并对数 / 相邻页表格对数, 1.0)
    #
    # 边界规则：
    #   - 原文无表格 → 该维度直接得 1.0
    #   - 处理后表格完全丢失 → 0.0，并记录"原始文档包含表格但处理后未保留任何表格"
    #   - score < 0.7 → 记录"表格完整率偏低"提示，并附带两路子分百分比
    #
    # 这组用例以确定性的算术验证子分与综合公式，避免基础测试中只用
    # ">" / "<" 阈值断言带来的回归覆盖盲区。

    def test_table_completeness_full_fill_single_page(self, scorer, default_profile):
        """单表 + 单元格全填充 + 无跨页对 → 1.0。"""
        full = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        original = ParsedDocument(
            blocks=[Block(type="table", text=full, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=full, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] == pytest.approx(1.0)
        # 不应触发"表格完整率偏低"提示
        assert not any("表格完整率偏低" in issue for issue in score.issues)

    def test_table_completeness_partial_fill_no_merge_needed(
        self, scorer, default_profile
    ):
        """单表 7/9 填充 + 单页（无跨页对） → 0.6 * (7/9) + 0.4 * 1.0 ≈ 0.867。

        既验证 cell_fill_rate 的精确计算（``_count_table_cells`` 把表头与
        数据行一视同仁，仅跳过 ``| --- |`` 分隔行），也验证当无相邻页对时
        merge_score 默认 1.0 不会拖低综合分。
        """
        # 3 列 × (1 表头 + 2 数据) = 9 个单元格；填充 7 个（表头 3 + 数据 4）
        partial = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n|  | 5 |  |"
        original = ParsedDocument(
            blocks=[Block(type="table", text=partial, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=partial, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        # 0.6 * (7/9) + 0.4 * 1.0 ≈ 0.4667 + 0.4 = 0.8667
        expected = 0.6 * (7 / 9) + 0.4 * 1.0
        assert score.components["table_completeness"] == pytest.approx(expected, abs=0.01)
        # 高于 0.7 阈值 → 不应记录"表格完整率偏低"
        assert not any("表格完整率偏低" in issue for issue in score.issues)

    def test_table_completeness_low_fill_records_issue(self, scorer, default_profile):
        """单元格填充率过低（综合分 < 0.7）→ 记录"表格完整率偏低"，
        且提示文本包含 cell_fill / merge 两路子分百分比。

        ``_count_table_cells`` 会跳过纯 ``-``/``|``/空白的行（用于过滤
        分隔行），因此为了构造一个"稀疏但仍有数据行"的表，需要每行
        至少有一个非空非分隔字符。
        """
        # 表头 3 列 + 5 行数据（每行只有 1 个填充） → 8/18 ≈ 0.444
        sparse = (
            "| A | B | C |\n"
            "| --- | --- | --- |\n"
            "| . |  |  |\n"
            "| . |  |  |\n"
            "| . |  |  |\n"
            "| . |  |  |\n"
            "| . |  |  |"
        )
        original = ParsedDocument(
            blocks=[Block(type="table", text=sparse, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=sparse, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        # 0.6 * (8/18) + 0.4 * 1.0 ≈ 0.667 < 0.7
        expected = 0.6 * (8 / 18) + 0.4 * 1.0
        assert score.components["table_completeness"] == pytest.approx(expected, abs=0.01)
        # issue 包含核心提示与两路子分百分比
        table_issues = [issue for issue in score.issues if "表格完整率偏低" in issue]
        assert len(table_issues) == 1
        assert "单元格填充率" in table_issues[0]
        assert "跨页合并率" in table_issues[0]

    def test_table_completeness_failed_cross_page_merge(self, scorer, default_profile):
        """相邻页有 2 张表但处理后未合并 → merge_score = 0/1 = 0.0；
        即便单元格全填充，综合分也只有 0.6。"""
        full_a = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        full_b = "| A | B |\n| --- | --- |\n| 3 | 4 |"
        original = ParsedDocument(
            blocks=[
                Block(type="table", text=full_a, page_number=1),
                Block(type="table", text=full_b, page_number=2),
            ]
        )
        # 处理后两张表仍然分开 → 没有合并发生
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="table", text=full_a, page_number=1),
                ProcessedBlock(type="table", text=full_b, page_number=2),
            ]
        )
        score = scorer.score(original, processed, default_profile)
        # 0.6 * 1.0 + 0.4 * 0.0 = 0.6 < 0.7 → 触发提示
        assert score.components["table_completeness"] == pytest.approx(0.6, abs=0.01)
        assert any("表格完整率偏低" in issue for issue in score.issues)

    def test_table_completeness_successful_cross_page_merge(
        self, scorer, default_profile
    ):
        """3 张相邻页表全部被合并为 1 张 → merge_score = 2/2 = 1.0；
        单元格全填充时综合分 = 1.0。"""
        a = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        b = "| A | B |\n| --- | --- |\n| 3 | 4 |"
        c = "| A | B |\n| --- | --- |\n| 5 | 6 |"
        original = ParsedDocument(
            blocks=[
                Block(type="table", text=a, page_number=1),
                Block(type="table", text=b, page_number=2),
                Block(type="table", text=c, page_number=3),
            ]
        )
        merged = (
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=merged, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] == pytest.approx(1.0)

    def test_table_completeness_non_adjacent_tables_skip_merge_penalty(
        self, scorer, default_profile
    ):
        """非相邻页（page 1 与 page 5）的多张表 → 不存在"应该合并"的对，
        即便没有合并也不应被扣分。"""
        full_a = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        full_b = "| A | B |\n| --- | --- |\n| 3 | 4 |"
        original = ParsedDocument(
            blocks=[
                Block(type="table", text=full_a, page_number=1),
                Block(type="table", text=full_b, page_number=5),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="table", text=full_a, page_number=1),
                ProcessedBlock(type="table", text=full_b, page_number=5),
            ]
        )
        score = scorer.score(original, processed, default_profile)
        # 单元格全填充 + 没有应该合并的对 → 综合分 1.0
        assert score.components["table_completeness"] == pytest.approx(1.0)

    def test_table_completeness_records_issue_when_all_tables_lost(
        self, scorer, default_profile
    ):
        """原文有表格但处理后完全没有表格 → 0.0，且记录中文提示。"""
        original = ParsedDocument(
            blocks=[
                Block(
                    type="table",
                    text="| A | B |\n| --- | --- |\n| 1 | 2 |",
                    page_number=1,
                )
            ]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="A B 1 2")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["table_completeness"] == 0.0
        # 期望记录"原始文档包含表格但处理后未保留任何表格"
        assert any(
            "原始文档包含表格但处理后未保留任何表格" in issue
            for issue in score.issues
        )

    def test_table_completeness_combined_formula_60_40_split(
        self, scorer, default_profile
    ):
        """显式校验综合公式 0.6*cell_fill + 0.4*merge：
        2 张相邻页表合并成功（merge=1.0），合并后单元格填充率 0.5
        → 综合分 = 0.6 * 0.5 + 0.4 * 1.0 = 0.7。"""
        a = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |"
        b = "| A | B | C |\n| --- | --- | --- |\n| 4 | 5 | 6 |"
        original = ParsedDocument(
            blocks=[
                Block(type="table", text=a, page_number=1),
                Block(type="table", text=b, page_number=2),
            ]
        )
        # 合并后表格：表头 3 列 + 3 行数据，每行只保留首列
        # → 总计 12 单元格，填充 6 个（表头 3 + 各数据行 1） → fill_rate = 0.5
        merged_partial = (
            "| A | B | C |\n"
            "| --- | --- | --- |\n"
            "| 1 |  |  |\n"
            "| 2 |  |  |\n"
            "| 3 |  |  |"
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="table", text=merged_partial, page_number=1)]
        )
        score = scorer.score(original, processed, default_profile)
        # cell_fill = 6/12 = 0.5；merge = 1/1 = 1.0
        # 综合 = 0.6 * 0.5 + 0.4 * 1.0 = 0.7
        assert score.components["table_completeness"] == pytest.approx(0.7, abs=0.01)
        # 0.7 是阈值边界：算法判定 score < 0.7 时才提示，刚好 0.7 不应触发
        assert not any("表格完整率偏低" in issue for issue in score.issues)

    def test_count_table_cells_filled_and_total(self, scorer):
        """直接覆盖 _count_table_cells：
        - 表头与数据行一视同仁地参与计数（仅跳过 ``| --- |`` 分隔行）
        - 空白单元格不计入 ``filled``。
        """
        # 表头 2 单元格 + 数据 2 单元格 = 4 个，全部填充
        full = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        assert scorer._count_table_cells(full) == {"total": 4, "filled": 4}

        # 表头 3 + 数据 6 = 9 单元格；表头 3 + 数据 4 = 7 个填充
        partial = "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n|  | 5 |  |"
        assert scorer._count_table_cells(partial) == {"total": 9, "filled": 7}

    def test_evaluate_cross_page_merge_no_adjacent_pairs(self, scorer):
        """直接覆盖 _evaluate_cross_page_merge：不相邻 → 1.0。"""
        a = Block(type="table", text="...", page_number=1)
        b = Block(type="table", text="...", page_number=5)
        proc_a = ProcessedBlock(type="table", text="...", page_number=1)
        proc_b = ProcessedBlock(type="table", text="...", page_number=5)
        assert scorer._evaluate_cross_page_merge([a, b], [proc_a, proc_b]) == 1.0

    def test_evaluate_cross_page_merge_full_success(self, scorer):
        """相邻 2 表合并成 1 表 → merged_count=1, pairs=1 → 1.0。"""
        a = Block(type="table", text="...", page_number=1)
        b = Block(type="table", text="...", page_number=2)
        merged = ProcessedBlock(type="table", text="...", page_number=1)
        assert scorer._evaluate_cross_page_merge([a, b], [merged]) == pytest.approx(1.0)

    def test_evaluate_cross_page_merge_full_failure(self, scorer):
        """相邻 2 表未合并 → merged_count=0, pairs=1 → 0.0。"""
        a = Block(type="table", text="...", page_number=1)
        b = Block(type="table", text="...", page_number=2)
        proc_a = ProcessedBlock(type="table", text="...", page_number=1)
        proc_b = ProcessedBlock(type="table", text="...", page_number=2)
        assert scorer._evaluate_cross_page_merge([a, b], [proc_a, proc_b]) == 0.0


# ─── Test Numeric Protection Scoring ──────────────────────────────────


class TestNumericProtectionScoring:
    """Tests for numeric protection rate scoring."""

    def test_all_numbers_preserved(self, scorer, default_profile):
        """Full score when all numbers are preserved."""
        text = "温度 25°C，压力 100kPa，偏差 0.05mm/m"
        original = ParsedDocument(blocks=[Block(type="paragraph", text=text)])
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=text)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 1.0

    def test_no_numbers(self, scorer, default_profile):
        """Full score when document has no numbers."""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="这是纯文字内容")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="这是纯文字内容")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 1.0

    def test_numbers_lost(self, scorer, default_profile):
        """Low score when numbers are lost during processing."""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="数值 123.45mm 和 67.89kPa 以及 0.01m/s")]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="数值和以及")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 0.0

    # ─── Focused tests for numeric_protection algorithm (Task 11.5) ────
    #
    # 算法（参见 quality_scorer._score_numeric_protection）：
    #   ratio = preserved / sample_size
    # 其中：
    #   - 原文用 ``_extract_numbers`` 抽取所有数值 token（含单位/范围/±/百分号）
    #   - 抽样前先按出现顺序去重（dict.fromkeys），避免同一数值重复占名额
    #   - 抽样上限为 ``NUMERIC_SAMPLE_SIZE`` (=50)
    #   - 处理后文本排除 ``is_noise=True`` 的块
    #
    # 边界规则：
    #   - 原文无数值 → 直接 1.0（vacuous）
    #   - ratio < ``NUMERIC_PROTECTION_ISSUE_THRESHOLD`` (=0.9) → 记录中文提示，
    #     提示中含比率、样本量、丢失数量，并附最多 ``NUMERIC_ISSUE_EXAMPLE_LIMIT``
    #     (=3) 个具体丢失数值示例
    #
    # 这组用例以确定性的算术校验子分公式与 issue 消息格式，覆盖：
    # 完美保留、部分丢失（精确比率）、阈值边界 0.9、空原文、含单位/范围/±/百分
    # 号的数值形式、抽样上限、去重、issue 内容、噪声块排除。

    def test_numeric_protection_full_preservation(self, scorer, default_profile):
        """全部抽样数值都在处理后文本中找到 → 1.0，且不记录 issue。"""
        text = "测量值 25.5°C，压力 100.2kPa，偏差 ±0.05mm/m"
        original = ParsedDocument(blocks=[Block(type="paragraph", text=text)])
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=text)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 1.0
        assert not any("数值保护率偏低" in issue for issue in score.issues)

    def test_numeric_protection_partial_loss_exact_ratio(self, scorer, default_profile):
        """5 个唯一数值丢 1 → 4/5 = 0.8 (< 0.9 阈值，记录 issue)。"""
        original = ParsedDocument(
            blocks=[
                Block(
                    type="paragraph",
                    text="测量值 10mm 20mm 30mm 40mm 50mm",
                )
            ]
        )
        # 处理后丢失 50mm
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="测量值 10mm 20mm 30mm 40mm")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == pytest.approx(0.8, abs=0.01)
        # 0.8 < 0.9 → 必须记录 issue
        assert any("数值保护率偏低" in issue for issue in score.issues)

    def test_numeric_protection_threshold_boundary_at_0_9(self, scorer, default_profile):
        """ratio == 0.9 时不应记录 issue（算法严格判定 < 0.9 才提示）。"""
        # 10 个唯一数值，丢 1 个 → 9/10 = 0.9 恰好等于阈值
        # 注意：使用空格分隔避免 _extract_numbers 贪婪吞噬后续字母
        nums = [f"{10 + i}.0mm" for i in range(10)]
        text = " ".join(nums)
        original = ParsedDocument(blocks=[Block(type="paragraph", text=text)])
        # 仅丢失最后一个 19.0mm
        processed_text = " ".join(nums[:-1])
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=processed_text)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == pytest.approx(0.9, abs=0.01)
        # 边界：恰好 0.9 不触发"数值保护率偏低"
        assert not any("数值保护率偏低" in issue for issue in score.issues)

    def test_numeric_protection_no_numbers_returns_one(self, scorer, default_profile):
        """原文不含任何数值 → 1.0（vacuous，且不记录 issue）。"""
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="这是一段没有任何数字的纯文字")]
        )
        # 处理后即使空也无所谓——没有数值需要保护
        processed = ProcessedDocument(blocks=[])
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 1.0
        assert not any("数值保护率偏低" in issue for issue in score.issues)

    def test_numeric_protection_complex_units_preserved(self, scorer, default_profile):
        """带复杂单位/范围/±/百分号/千分号的数值被原样保留 → 1.0。

        覆盖 ``_extract_numbers`` 实际匹配的形式：
        - 复合单位：mm/m、m/s²
        - 容差：±10mm
        - 范围：55°~65°
        - 百分号/千分号：5%、0.1‰
        """
        text = (
            "公差 ±10mm，比率 0.05mm/m，温度区间 55°~65°，"
            "误差率 5%，杂质 0.1‰"
        )
        original = ParsedDocument(blocks=[Block(type="paragraph", text=text)])
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=text)]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 1.0

    def test_numeric_protection_dedupes_repeated_values(self, scorer, default_profile):
        """重复出现的同一数值在抽样阶段被去重，不会因重复而占用样本配额。

        构造：原文重复出现 ``100kPa`` 三次，再加上 ``200kPa``、``300kPa``。
        若不去重，样本会是 [100, 100, 100, 200, 300] 共 5 个。
        去重后样本只剩 [100, 200, 300] 共 3 个唯一数值。
        处理后丢失 200kPa：
          - 不去重：preserved = 3+0+1 = 4，ratio = 4/5 = 0.8
          - 去重：preserved = 1+0+1 = 2，ratio = 2/3 ≈ 0.667
        断言去重路径生效（即得到 0.667）。
        """
        original = ParsedDocument(
            blocks=[
                Block(
                    type="paragraph",
                    text="测量 100kPa 又一次 100kPa 还是 100kPa，对比 200kPa 与 300kPa",
                )
            ]
        )
        # 丢失 200kPa
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="100kPa 与 300kPa")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == pytest.approx(2 / 3, abs=0.01)

    def test_numeric_protection_caps_sample_at_limit(self, scorer, default_profile):
        """原文唯一数值数量超过 ``NUMERIC_SAMPLE_SIZE`` 时，仅按上限抽样。

        构造 60 个唯一整数，处理后只保留前 50 个：
          - 抽样上限 50（=NUMERIC_SAMPLE_SIZE），样本是前 50 个唯一值
          - 这 50 个全部出现在处理后文本中 → ratio = 1.0
        若不截断，分母会是 60，分子也会是 50 → ratio = 5/6 ≈ 0.833 < 0.9，
        会触发 issue。这条断言验证抽样确实按上限截断。
        """
        from app.services.quality_scorer import NUMERIC_SAMPLE_SIZE

        # 60 个唯一数值（带单位避免被合并）
        unique_count = NUMERIC_SAMPLE_SIZE + 10
        all_nums = [f"{i + 1}.0mm" for i in range(unique_count)]
        original_text = " ".join(all_nums)
        # 处理后只保留前 NUMERIC_SAMPLE_SIZE 个
        processed_text = " ".join(all_nums[:NUMERIC_SAMPLE_SIZE])

        original = ParsedDocument(
            blocks=[Block(type="paragraph", text=original_text)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=processed_text)]
        )
        score = scorer.score(original, processed, default_profile)
        # 抽样停在前 50，被截断的尾部 10 个不参与评分 → 1.0
        assert score.components["numeric_protection"] == 1.0
        assert not any("数值保护率偏低" in issue for issue in score.issues)

    def test_numeric_protection_issue_message_includes_examples(
        self, scorer, default_profile
    ):
        """触发 issue 时，消息中包含比率、样本量、丢失数量与具体丢失示例。"""
        # 5 个唯一数值，丢失前 3 个 → ratio = 2/5 = 0.4 < 0.9
        original = ParsedDocument(
            blocks=[
                Block(
                    type="paragraph",
                    text="数值 11.1mm 22.2mm 33.3mm 44.4mm 55.5mm",
                )
            ]
        )
        # 仅保留 44.4mm 与 55.5mm
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="保留 44.4mm 与 55.5mm")]
        )
        score = scorer.score(original, processed, default_profile)
        numeric_issues = [
            issue for issue in score.issues if "数值保护率偏低" in issue
        ]
        assert len(numeric_issues) == 1
        msg = numeric_issues[0]
        # 比率（40.0%）和样本量/丢失数量必须出现
        assert "40.0%" in msg
        assert "抽样 5 个" in msg
        assert "3 个" in msg
        # 至少出现一个具体丢失示例（最多 3 个）
        assert "11.1mm" in msg
        assert "22.2mm" in msg
        assert "33.3mm" in msg
        # 第 4 个被丢失的数值不应再出现（截断到 NUMERIC_ISSUE_EXAMPLE_LIMIT=3）
        # 这里 5 个里只丢了 3 个，刚好等于上限，所以这条隐含成立。

    def test_numeric_protection_issue_examples_capped_at_three(
        self, scorer, default_profile
    ):
        """丢失数值数量超过 ``NUMERIC_ISSUE_EXAMPLE_LIMIT`` 时，
        issue 消息中最多出现前 3 个示例，第 4 个不应出现。
        """
        from app.services.quality_scorer import NUMERIC_ISSUE_EXAMPLE_LIMIT

        # 5 个唯一数值，全部丢失 → ratio = 0.0
        # 注意：使用空格而非 = 分隔，避免 _extract_numbers 把字母吞进数值 token
        original = ParsedDocument(
            blocks=[
                Block(
                    type="paragraph",
                    text="数值: 11.1mm; 22.2mm; 33.3mm; 44.4mm; 55.5mm",
                )
            ]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="无任何数值")]
        )
        score = scorer.score(original, processed, default_profile)
        numeric_issues = [
            issue for issue in score.issues if "数值保护率偏低" in issue
        ]
        assert len(numeric_issues) == 1
        msg = numeric_issues[0]
        # 前 3 个丢失示例必须出现
        assert "11.1mm" in msg
        assert "22.2mm" in msg
        assert "33.3mm" in msg
        # 第 4、第 5 个不应出现（截断到 3）
        assert "44.4mm" not in msg
        assert "55.5mm" not in msg
        # 简单 sanity check：上限常量保持为 3
        assert NUMERIC_ISSUE_EXAMPLE_LIMIT == 3

    def test_numeric_protection_excludes_noise_blocks_from_processed(
        self, scorer, default_profile
    ):
        """处理后块若标记为 ``is_noise=True``，其中的数值不计入"已保留"。

        构造：原文包含 ``42mm``；处理后 ``42mm`` 仅出现在 noise 块中。
        噪声块应被排除，因此 ``42mm`` 视为丢失 → ratio = 0/1 = 0.0。
        """
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="关键尺寸 42mm")]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text="正文已被清空"),
                # 数值仅残留在被识别为噪声的页眉里
                ProcessedBlock(type="paragraph", text="页眉 42mm", is_noise=True),
            ]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["numeric_protection"] == 0.0
        # 也应触发 issue（因为 0.0 < 0.9）
        assert any("数值保护率偏低" in issue for issue in score.issues)


# ─── Test Boilerplate Removal Scoring ─────────────────────────────────


class TestBoilerplateRemovalScoring:
    """Tests for boilerplate removal rate scoring."""

    def test_no_noise_expected(self, scorer, default_profile):
        """Full score when document has no expected noise (few pages)."""
        original = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="内容A", page_number=1),
                Block(type="paragraph", text="内容B", page_number=2),
            ]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text="内容A"),
                ProcessedBlock(type="paragraph", text="内容B"),
            ],
            noise_removed_count=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0

    def test_noise_fully_removed(self, scorer, default_profile):
        """Full score when all expected noise is removed."""
        # Create a document with repeated headers across 4 pages
        blocks = []
        for page in range(1, 5):
            blocks.append(Block(type="paragraph", text="公司机密", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文内容第{page}页", page_number=page))
            blocks.append(Block(type="paragraph", text="第X页", page_number=page))

        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文内容第{p}页")
                for p in range(1, 5)
            ],
            noise_removed_count=8,  # 4 headers + 4 footers removed
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0

    def test_noise_partially_removed(self, scorer, default_profile):
        """Partial score when some noise remains."""
        blocks = []
        for page in range(1, 5):
            blocks.append(Block(type="paragraph", text="水印文字", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文{page}", page_number=page))

        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文{p}")
                for p in range(1, 5)
            ],
            noise_removed_count=2,  # Only removed 2 out of 4 expected
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 0.5

    # ─── Focused tests for boilerplate_removal algorithm (Task 11.6) ────
    #
    # 算法（参见 quality_scorer._score_boilerplate_removal）：
    #   ratio = min(noise_removed_count / expected_noise_blocks, 1.0)
    # 其中 expected_noise_blocks 由"每页首/末块文本归一化后频率 ≥
    # BOILERPLATE_DETECTION_THRESHOLD (0.5)"统计得到。
    #
    # 边界规则：
    #   - 原文 0 块                                 → 1.0（vacuous）
    #   - total_pages < MIN_PAGES_FOR_DETECTION (3) → 1.0（频率信号不可靠）
    #   - 检测不到任何重复 boilerplate              → 1.0（vacuous）
    #   - ratio < BOILERPLATE_ISSUE_THRESHOLD (0.7) → 记录"噪声去除率偏低"提示
    #
    # 这组用例锁定常量、归一化和精确比率公式，覆盖：完整去除、精确部分
    # 去除比率、无预期噪声、零噪声原文、少于 3 页跳过、ratio 截断到 1.0、
    # issue 消息格式、阈值边界 0.7、文本归一化（大小写/空白）。

    def test_boilerplate_removal_full_removal(self, scorer, default_profile):
        """4 页统一页眉，全部去除 → 1.0，且不记录 issue。"""
        blocks = []
        for page in range(1, 5):
            blocks.append(Block(type="paragraph", text="公司机密", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文 {page}", page_number=page))
            blocks.append(Block(type="paragraph", text="第 X 页", page_number=page))
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文 {p}")
                for p in range(1, 5)
            ],
            noise_removed_count=8,  # 4 headers + 4 footers
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0
        assert not any("噪声去除率偏低" in issue for issue in score.issues)

    def test_boilerplate_removal_partial_exact_ratio(self, scorer, default_profile):
        """5 页统一页眉/页脚，预期 10 个噪声，仅去除 6 → 0.6（< 0.7 触发 issue）。"""
        blocks = []
        for page in range(1, 6):
            blocks.append(Block(type="paragraph", text="机密文档", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文 {page}", page_number=page))
            blocks.append(Block(type="paragraph", text="版权所有", page_number=page))
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文 {p}")
                for p in range(1, 6)
            ],
            noise_removed_count=6,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == pytest.approx(0.6, abs=0.01)
        assert any("噪声去除率偏低" in issue for issue in score.issues)

    def test_boilerplate_removal_no_expected_noise(self, scorer, default_profile):
        """3+ 页但页眉/页脚均不重复 → 检测不到任何 boilerplate，得 1.0。"""
        blocks = []
        for page in range(1, 5):
            blocks.append(
                Block(type="paragraph", text=f"独特首段 {page}", page_number=page)
            )
            blocks.append(
                Block(type="paragraph", text=f"独特尾段 {page}", page_number=page)
            )
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"独特首段 {p}")
                for p in range(1, 5)
            ],
            noise_removed_count=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0
        assert not any("噪声去除率偏低" in issue for issue in score.issues)

    def test_boilerplate_removal_no_noise_no_removal(self, scorer, default_profile):
        """无预期噪声且未去除任何块 → 1.0（vacuous full score）。"""
        # 文档有正常多页内容，但每页内容全不相同 → 无 boilerplate
        blocks = [
            Block(type="paragraph", text=f"正文段 {p}", page_number=p)
            for p in range(1, 6)
        ]
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文段 {p}")
                for p in range(1, 6)
            ],
            noise_removed_count=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0

    def test_boilerplate_removal_skips_few_pages(self, scorer, default_profile):
        """页数 < MIN_PAGES_FOR_DETECTION 时直接 1.0，不做频率统计。"""
        from app.services.quality_scorer import MIN_PAGES_FOR_DETECTION

        # 构造 (MIN_PAGES_FOR_DETECTION - 1) 页，每页都有"看似 boilerplate"的
        # 重复首块。如果检测照常进行，会算出 noise_removed=0 / expected>0 → 0.0
        # 并触发 issue。但页数不足，应跳过统计直接得 1.0。
        page_count = MIN_PAGES_FOR_DETECTION - 1
        assert page_count >= 1, "MIN_PAGES_FOR_DETECTION 必须 ≥ 2"
        blocks = []
        for page in range(1, page_count + 1):
            blocks.append(Block(type="paragraph", text="重复页眉", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文 {page}", page_number=page))
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文 {p}")
                for p in range(1, page_count + 1)
            ],
            noise_removed_count=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0
        assert not any("噪声去除率偏低" in issue for issue in score.issues)

    def test_boilerplate_removal_capped_at_one(self, scorer, default_profile):
        """noise_removed_count > expected_noise → 截断到 1.0（不能 >100%）。"""
        # 4 页统一页眉，预期噪声 = 4；处理器报告去除了 99 个噪声块（例如把
        # 整页内容也全部当作噪声删了）。覆盖率应被截断为 1.0。
        blocks = []
        for page in range(1, 5):
            blocks.append(Block(type="paragraph", text="重复页眉", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文 {page}", page_number=page))
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="正文")],
            noise_removed_count=99,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == 1.0

    def test_boilerplate_removal_threshold_boundary_at_0_7(
        self, scorer, default_profile
    ):
        """ratio == 0.7 时不应记录 issue（算法严格判定 < 0.7 才提示）。"""
        # 10 页统一页眉/页脚 → expected_noise = 20；去除 14 → ratio 恰好 0.7
        blocks = []
        for page in range(1, 11):
            blocks.append(Block(type="paragraph", text="页眉重复", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文 {page}", page_number=page))
            blocks.append(Block(type="paragraph", text="页脚重复", page_number=page))
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文 {p}")
                for p in range(1, 11)
            ],
            noise_removed_count=14,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.components["boilerplate_removal"] == pytest.approx(0.7, abs=0.01)
        # 边界：恰好 0.7 不触发 issue
        assert not any("噪声去除率偏低" in issue for issue in score.issues)

    def test_boilerplate_removal_issue_message_format(self, scorer, default_profile):
        """触发 issue 时，消息包含比率百分比、预期噪声数和实际去除数。"""
        # 4 页统一页眉/页脚 → expected = 8；去除 0 → ratio = 0.0 必触发 issue
        blocks = []
        for page in range(1, 5):
            blocks.append(Block(type="paragraph", text="水印", page_number=page))
            blocks.append(Block(type="paragraph", text=f"正文 {page}", page_number=page))
            blocks.append(Block(type="paragraph", text="页脚", page_number=page))
        original = ParsedDocument(blocks=blocks)
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"正文 {p}")
                for p in range(1, 5)
            ],
            noise_removed_count=0,
        )
        score = scorer.score(original, processed, default_profile)
        boilerplate_issues = [
            issue for issue in score.issues if "噪声去除率偏低" in issue
        ]
        assert len(boilerplate_issues) == 1
        msg = boilerplate_issues[0]
        # 比率（0.0%）、预期数（8）和实际数（0）必须出现
        assert "0.0%" in msg
        assert "8 个" in msg
        assert "实际去除 0 个" in msg

    def test_boilerplate_removal_normalizes_whitespace_and_case(
        self, scorer, default_profile
    ):
        """重复 boilerplate 仅在大小写/空白上有差异时仍被识别为同一文本。

        若不归一化：  ``"Page 1"``、``" page  1 "``、``"PAGE 1"`` 会被视作
        三个不同字符串，频率均 < 50% 阈值，expected_noise 漏判 → 算法误判
        为"无 boilerplate"，得 1.0。

        归一化后三者折叠为同一字符串，频率 = 4/4 = 100% ≥ 50% → expected
        计为 4。处理器声称去除了全部 4 个 → ratio = 1.0。
        虽然结果都是 1.0，但通过把 noise_removed_count 调到 0 来区分两条
        路径：归一化生效时 expected=4, removed=0 → ratio=0.0；不归一化时
        expected=0 → vacuous 1.0。本用例断言归一化生效（ratio = 0.0）。
        """
        blocks = []
        # 4 页，首块文本仅在大小写/空白上不同；末块都是独特正文
        variants = ["Header Text", " header  text ", "HEADER TEXT", "header text"]
        for page, header in enumerate(variants, start=1):
            blocks.append(Block(type="paragraph", text=header, page_number=page))
            blocks.append(
                Block(type="paragraph", text=f"独特正文 {page}", page_number=page)
            )
        original = ParsedDocument(blocks=blocks)
        # 处理器没有去除任何噪声 → 归一化生效时 ratio = 0/4 = 0.0
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=f"独特正文 {p}")
                for p in range(1, 5)
            ],
            noise_removed_count=0,
        )
        score = scorer.score(original, processed, default_profile)
        # 归一化生效 → expected=4 > 0，noise_removed=0 → ratio=0.0
        assert score.components["boilerplate_removal"] == 0.0
        assert any("噪声去除率偏低" in issue for issue in score.issues)


# ─── Test Overall Score Calculation ───────────────────────────────────


class TestOverallScoreCalculation:
    """Tests for weighted overall score calculation."""

    def test_perfect_score(self, scorer, default_profile):
        """Overall score is 1.0 when all dimensions are perfect."""
        text = "这是一段完整的文本内容"
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text=text, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=text)],
            noise_removed_count=0,
            headings_detected=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.overall == pytest.approx(1.0, abs=0.01)

    def test_weights_sum_to_one(self):
        """Scoring weights sum to 1.0."""
        total = (
            WEIGHT_TEXT_RETENTION
            + WEIGHT_HEADING_DETECTION
            + WEIGHT_TABLE_COMPLETENESS
            + WEIGHT_NUMERIC_PROTECTION
            + WEIGHT_BOILERPLATE_REMOVAL
        )
        assert total == pytest.approx(1.0)

    def test_weighted_calculation(self):
        """Overall score is correctly weighted."""
        scorer = QualityScorer()
        # Manually verify: if text_retention=0.5, all others=1.0
        # overall = 0.30*0.5 + 0.25*1.0 + 0.20*1.0 + 0.15*1.0 + 0.10*1.0
        #         = 0.15 + 0.25 + 0.20 + 0.15 + 0.10 = 0.85
        text = "ABCDEFGHIJ"  # 10 chars
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text=text, page_number=1)]
        )
        # Processed has only half the text
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="ABCDE")],
            noise_removed_count=0,
            headings_detected=0,
        )
        profile = _make_default_profile()
        score = scorer.score(original, processed, profile)
        # text_retention = 5/10 = 0.5
        assert score.components["text_retention"] == 0.5
        # overall = 0.30*0.5 + 0.25*1.0 + 0.20*1.0 + 0.15*1.0 + 0.10*1.0 = 0.85
        assert score.overall == pytest.approx(0.85, abs=0.01)

    # ─── Constants ────────────────────────────────────────────────

    def test_weight_constants_exact_values(self):
        """权重常量必须严格等于设计文档中规定的数值。

        设计文档（design.md "Quality Scorer + Review Queue"）和任务 11.7
        均明确规定加权配比为 30% / 25% / 20% / 15% / 10%。一旦有人
        意外修改了某个常量，加权综合分就会偏离规范，因此这里做精确
        相等断言（不用 approx）。
        """
        assert WEIGHT_TEXT_RETENTION == 0.30
        assert WEIGHT_HEADING_DETECTION == 0.25
        assert WEIGHT_TABLE_COMPLETENESS == 0.20
        assert WEIGHT_NUMERIC_PROTECTION == 0.15
        assert WEIGHT_BOILERPLATE_REMOVAL == 0.10

    # ─── All-zero edge case ───────────────────────────────────────

    def test_all_zero_dimensions_yield_zero_overall(self, chinese_spec_profile):
        """5 个维度同时为 0 时综合分为 0 并伴随相应 issues。

        构造一份原始文档使得每个维度在处理后都"该有的没保住"：

        - 文本：原文有大量可见字符，但 processed 把所有正文都丢失
          → text_retention=0（issue: ``text_lost``）
        - 标题：原文里有匹配 chinese_spec_profile heading_rules 的标题
          块，但 processed.headings_detected=0
          → heading_detection=0（issue: 标题识别率偏低）
        - 表格：原文里有 table 块，但 processed 没有任何 table
          → table_completeness=0（issue: 原始文档包含表格但处理后未保留）
        - 数值：原文里有数值串，但 processed 文本里完全没有这些串
          → numeric_protection=0（issue: 数值保护率偏低）
        - 噪声：原文 ≥ 3 页且每页首块都是同一段重复内容（≥50% 阈值），
          但 processed.noise_removed_count=0
          → boilerplate_removal=0（issue: 噪声去除率偏低）
        """
        # 4 页：每页首块为重复"页眉"，第二块为含标题/表格/数值的内容
        blocks: list[Block] = []
        for page in range(1, 5):
            blocks.append(
                Block(type="paragraph", text="重复页眉文字", page_number=page)
            )
            blocks.append(
                Block(
                    type="heading",
                    text=f"一、章节{page}",
                    page_number=page,
                    style={"heading_level": 1},
                )
            )
            blocks.append(
                Block(
                    type="paragraph",
                    text=f"参数 {page * 10}.5mm 偏差控制",
                    page_number=page,
                )
            )
            blocks.append(
                Block(
                    type="table",
                    text="| A | B |\n| --- | --- |\n| 1 | 2 |",
                    page_number=page,
                )
            )
        original = ParsedDocument(blocks=blocks)

        # 处理结果：完全没有保留任何内容（空 blocks），
        # 没有标题、没有表格、不含原文里的任何数值、未声明去除噪声
        processed = ProcessedDocument(
            blocks=[],
            noise_removed_count=0,
            headings_detected=0,
        )

        scorer = QualityScorer()
        score = scorer.score(original, processed, chinese_spec_profile)

        # 5 个维度全部归零
        assert score.components["text_retention"] == 0.0
        assert score.components["heading_detection"] == 0.0
        assert score.components["table_completeness"] == 0.0
        assert score.components["numeric_protection"] == 0.0
        assert score.components["boilerplate_removal"] == 0.0
        # 加权综合分为 0
        assert score.overall == 0.0
        # 每个失分维度都应留下可读 issue（用关键字断言宽松匹配）
        joined = "\n".join(score.issues)
        assert "text_lost" in joined
        assert "标题识别率偏低" in joined
        assert "原始文档包含表格" in joined
        assert "数值保护率偏低" in joined
        assert "噪声去除率偏低" in joined

    # ─── Single-dimension-zero / inverse-weight tests ─────────────

    @pytest.mark.parametrize(
        "zero_method, expected_overall",
        [
            ("_score_text_retention", 1.0 - WEIGHT_TEXT_RETENTION),
            ("_score_heading_detection", 1.0 - WEIGHT_HEADING_DETECTION),
            ("_score_table_completeness", 1.0 - WEIGHT_TABLE_COMPLETENESS),
            ("_score_numeric_protection", 1.0 - WEIGHT_NUMERIC_PROTECTION),
            ("_score_boilerplate_removal", 1.0 - WEIGHT_BOILERPLATE_REMOVAL),
        ],
    )
    def test_single_dimension_zero_yields_inverse_weight(
        self,
        monkeypatch,
        default_profile,
        zero_method: str,
        expected_overall: float,
    ):
        """单一维度归零、其余满分时 overall = 1 - 该维度权重。

        通过 monkeypatch 把每个维度评分函数替换为可控常量（白盒测试
        加权公式本身），独立于具体维度算法。这能精确验证：
        ``WEIGHT_X * 0 + ΣWEIGHT_其余 * 1 == 1 - WEIGHT_X``。
        """
        scorer = QualityScorer()
        for method in (
            "_score_text_retention",
            "_score_heading_detection",
            "_score_table_completeness",
            "_score_numeric_protection",
            "_score_boilerplate_removal",
        ):
            value = 0.0 if method == zero_method else 1.0
            monkeypatch.setattr(
                scorer, method, lambda *a, _v=value, **kw: _v
            )

        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="占位", page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="占位")]
        )
        score = scorer.score(original, processed, default_profile)

        assert score.overall == pytest.approx(expected_overall, abs=1e-9)

    # ─── Constructor parameter propagation ────────────────────────

    def test_custom_weight_injection_through_init(
        self, monkeypatch, default_profile
    ):
        """自定义权重通过 ``__init__`` 正确传播到加权求和。

        构造一组刻意非默认且互不相同的权重，使得"是否生效"在数值上
        毫无歧义：若构造器没有正确接管，结果会落在默认 0.30/0.25/0.20/
        0.15/0.10 上而显著偏离断言值。
        """
        custom = QualityScorer(
            weight_text_retention=0.50,
            weight_heading_detection=0.20,
            weight_table_completeness=0.15,
            weight_numeric_protection=0.10,
            weight_boilerplate_removal=0.05,
        )
        # 字段必须原样存储
        assert custom.weight_text_retention == 0.50
        assert custom.weight_heading_detection == 0.20
        assert custom.weight_table_completeness == 0.15
        assert custom.weight_numeric_protection == 0.10
        assert custom.weight_boilerplate_removal == 0.05

        # 让各维度返回不同已知值，验证加权确实使用了自定义权重
        components = {
            "_score_text_retention": 0.8,
            "_score_heading_detection": 0.6,
            "_score_table_completeness": 0.4,
            "_score_numeric_protection": 0.2,
            "_score_boilerplate_removal": 0.0,
        }
        for method, value in components.items():
            monkeypatch.setattr(custom, method, lambda *a, _v=value, **kw: _v)

        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="占位", page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="占位")]
        )
        score = custom.score(original, processed, default_profile)

        expected = (
            0.50 * 0.8 + 0.20 * 0.6 + 0.15 * 0.4 + 0.10 * 0.2 + 0.05 * 0.0
        )
        assert score.overall == pytest.approx(expected, abs=1e-9)

    # ─── Clamping inside ParseQualityScore ────────────────────────

    def test_overall_clamped_to_unit_interval_against_float_drift(self):
        """ParseQualityScore.__post_init__ 必须把越界 overall 夹紧到 [0,1]。

        即便维度评分实现层出现微小浮点漂移（例如 1.0 + 1e-12 这种因
        IEEE-754 累加误差产生的"略大于 1"），下游也不应看到非法值。
        同时验证略小于 0 的情况会被夹到 0.0。
        """
        # 略大于 1.0 的浮点漂移 → 夹回 1.0
        upper = ParseQualityScore(overall=1.0 + 1e-12)
        assert upper.overall == 1.0

        # 远超 1.0 的脏数据也要夹住
        far_upper = ParseQualityScore(overall=2.5)
        assert far_upper.overall == 1.0

        # 略小于 0.0 → 夹回 0.0
        lower = ParseQualityScore(overall=-1e-12)
        assert lower.overall == 0.0

        # 远小于 0 的脏数据也要夹住
        far_lower = ParseQualityScore(overall=-3.7)
        assert far_lower.overall == 0.0

    # ─── Numerical precision (parametrised) ───────────────────────

    @pytest.mark.parametrize(
        "components, expected",
        [
            # 全为 0 的快速路径
            ((0.0, 0.0, 0.0, 0.0, 0.0), 0.0),
            # 全为 1 的快速路径
            ((1.0, 1.0, 1.0, 1.0, 1.0), 1.0),
            # 任务说明里的 0.85 用例（text=0.5 其余=1）
            ((0.5, 1.0, 1.0, 1.0, 1.0), 0.85),
            # 阶梯输入：每个维度都不同
            (
                (0.9, 0.8, 0.7, 0.6, 0.5),
                0.30 * 0.9
                + 0.25 * 0.8
                + 0.20 * 0.7
                + 0.15 * 0.6
                + 0.10 * 0.5,
            ),
            # 略带浮点尾数的输入
            (
                (0.123, 0.456, 0.789, 0.321, 0.654),
                0.30 * 0.123
                + 0.25 * 0.456
                + 0.20 * 0.789
                + 0.15 * 0.321
                + 0.10 * 0.654,
            ),
            # 接近 1.0 边界的输入：验证不会越过夹紧上限
            ((0.99, 0.99, 0.99, 0.99, 0.99), 0.99),
        ],
    )
    def test_overall_precision_parametrised(
        self,
        monkeypatch,
        default_profile,
        components: tuple[float, float, float, float, float],
        expected: float,
    ):
        """参数化校验加权求和在不同输入下的数值精度。

        ``expected`` 用 Python 表达式直接表达加权式，避免在断言里写死
        预先算好的浮点字面量；用 ``pytest.approx`` 容忍合理浮点误差，
        但容差设到 1e-9 以确保不放过真实漂移。
        """
        scorer = QualityScorer()
        text, heading, table, numeric, boilerplate = components
        monkeypatch.setattr(
            scorer, "_score_text_retention", lambda *a, **kw: text
        )
        monkeypatch.setattr(
            scorer, "_score_heading_detection", lambda *a, **kw: heading
        )
        monkeypatch.setattr(
            scorer, "_score_table_completeness", lambda *a, **kw: table
        )
        monkeypatch.setattr(
            scorer, "_score_numeric_protection", lambda *a, **kw: numeric
        )
        monkeypatch.setattr(
            scorer, "_score_boilerplate_removal", lambda *a, **kw: boilerplate
        )

        original = ParsedDocument(
            blocks=[Block(type="paragraph", text="占位", page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="占位")]
        )
        score = scorer.score(original, processed, default_profile)
        assert score.overall == pytest.approx(expected, abs=1e-9)


# ─── Test Review Queue Logic ──────────────────────────────────────────


class TestReviewQueueLogic:
    """Tests for review queue enqueue logic."""

    def test_needs_review_below_threshold(self, scorer):
        """Documents with score below 0.7 need review."""
        score = ParseQualityScore(overall=0.65)
        assert scorer.needs_review(score) is True

    def test_no_review_above_threshold(self, scorer):
        """Documents with score at or above 0.7 don't need review."""
        score = ParseQualityScore(overall=0.70)
        assert scorer.needs_review(score) is False

        score = ParseQualityScore(overall=0.95)
        assert scorer.needs_review(score) is False

    def test_needs_review_at_boundary(self, scorer):
        """Boundary: score exactly at threshold does not need review."""
        score = ParseQualityScore(overall=DEFAULT_REVIEW_THRESHOLD)
        assert scorer.needs_review(score) is False

    def test_custom_threshold(self):
        """Custom threshold works correctly."""
        scorer = QualityScorer(review_threshold=0.8)
        assert scorer.needs_review(ParseQualityScore(overall=0.75)) is True
        assert scorer.needs_review(ParseQualityScore(overall=0.85)) is False


# ─── Test Integration: Full Scoring Pipeline ──────────────────────────


class TestFullScoringPipeline:
    """Integration tests for the complete scoring pipeline."""

    def test_high_quality_document(self, scorer, default_profile):
        """High quality document gets high overall score."""
        text = "这是一份高质量的文档，包含完整的文本内容和结构信息。数值 25.5mm 保留完好。"
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text=text, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text=text)],
            noise_removed_count=0,
            headings_detected=0,
        )
        score = scorer.score(original, processed, default_profile)
        assert score.overall >= 0.9
        assert not scorer.needs_review(score)

    def test_low_quality_document(self, scorer, chinese_spec_profile):
        """Low quality document gets low score and needs review."""
        original = ParsedDocument(
            blocks=[
                Block(type="heading", text="一、总则", style={"heading_level": 1}),
                Block(type="heading", text="二、范围", style={"heading_level": 1}),
                Block(type="paragraph", text="详细内容包含数值 100.5mm/m 和 25°C"),
                Block(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |"),
            ]
        )
        # Processed loses most content
        processed = ProcessedDocument(
            blocks=[ProcessedBlock(type="paragraph", text="内容")],
            noise_removed_count=0,
            headings_detected=0,
        )
        score = scorer.score(original, processed, chinese_spec_profile)
        assert score.overall < 0.7
        assert scorer.needs_review(score)
        assert len(score.issues) > 0


# Helper for parametrized test
def _make_default_profile():
    return DocumentProfileConfig(
        id="default",
        name="generic-text",
        description="通用文本文档",
        priority=0,
        enabled=True,
        match_rules=MatchRules(),
        heading_rules=[],
        boilerplate=BoilerplateConfig(),
        tables=TableConfig(),
        chunking=ChunkingConfig(),
    )


# ─── Property-based Invariants for QualityScorer.score() ──────────────


class TestQualityScorerProperties:
    """``QualityScorer.score()`` 全局不变量（基于 Hypothesis 的属性测试）。

    既有 Tests 都是「示例 + 边界」式的，覆盖了每个维度的 happy path、
    issue branch 和阈值边界。本组测试用 Hypothesis 在更广的输入空间上
    锁定 ``score()`` 的几条全局不变量，避免某次维度算法改动让综合分
    悄悄越界、键变形、或失去确定性。

    锁定的不变量（参见 design.md「Quality Scorer + Review Queue」）：

    1. ``overall ∈ [0, 1]``：综合分必须落在单位区间内（既有
       ``ParseQualityScore.__post_init__`` 也会兜底，但这里同时验证打分
       路径自身就给出合法值，不依赖兜底）。
    2. ``components`` 始终包含 5 个固定键：``text_retention``、
       ``heading_detection``、``table_completeness``、``numeric_protection``、
       ``boilerplate_removal``。任何子分缺失都会导致前端审核界面渲染破。
    3. 每个子分也必须落在 ``[0, 1]``。
    4. 确定性：相同输入连续两次调用 ``score()`` 必须返回相等结果（综合分、
       全部子分、issues 顺序）。任何隐式的随机性（例如 set 顺序漏出、
       未排序 dict 迭代）都会破坏审核队列的可重复性。
    5. 当输入「平凡满分」（原文与处理结果完全一致、无表格、无标题、无
       数值、单页）时综合分必须是 ``1.0``。这把"算法的零风险输入应该满
       分"这条公约写成可执行断言。

    Validates: Requirements 17
    """

    # 文本字符策略：限制在常见可打印 ASCII 与 CJK，避免 Hypothesis
    # 生成把 ``_extract_numbers`` 正则吞进非预期 token 的 Unicode 控制字符
    # 噪声样本——这些不属于本属性测试要锁的不变量。
    _safe_char = st.characters(
        whitelist_categories=("L", "N", "P", "Zs"),
        blacklist_characters="\x00\r\n\t|",
    )
    _safe_text = st.text(_safe_char, min_size=0, max_size=80)

    # ParsedDocument.Block 策略：随机 type / 文本 / 页号；保留 page_number
    # 在 [1, 6] 区间，让"≥3 页"的 boilerplate 检测分支也有机会被触发。
    _block_strategy = st.builds(
        Block,
        type=st.sampled_from(["paragraph", "heading", "table", "image", "formula"]),
        text=_safe_text,
        bbox=st.none(),
        page_number=st.integers(min_value=1, max_value=6),
        style=st.just({}),
        raw=st.just({}),
    )

    # ProcessedBlock 策略：与 Block 类似，但允许 ``is_noise`` 真假各半。
    _processed_block_strategy = st.builds(
        ProcessedBlock,
        type=st.sampled_from(["paragraph", "heading", "table", "image", "formula"]),
        text=_safe_text,
        heading_level=st.integers(min_value=0, max_value=6),
        page_number=st.integers(min_value=1, max_value=6),
        is_noise=st.booleans(),
        asset_ids=st.just([]),
        original_text=st.just(""),
    )

    _parsed_doc_strategy = st.builds(
        ParsedDocument,
        blocks=st.lists(_block_strategy, min_size=0, max_size=12),
        metadata=st.just({}),
        assets=st.just([]),
    )

    _processed_doc_strategy = st.builds(
        ProcessedDocument,
        blocks=st.lists(_processed_block_strategy, min_size=0, max_size=12),
        metadata=st.just({}),
        markdown=st.just(""),
        noise_removed_count=st.integers(min_value=0, max_value=20),
        headings_detected=st.integers(min_value=0, max_value=10),
    )

    @hyp_settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    @given(original=_parsed_doc_strategy, processed=_processed_doc_strategy)
    def test_score_returns_valid_overall_in_unit_interval(
        self, original, processed
    ):
        """``overall`` 永远落在 ``[0, 1]``，且综合分本身就合法（不依赖夹紧）。

        通过手工再算一次 0.30 / 0.25 / 0.20 / 0.15 / 0.10 加权和并要求
        与 ``score.overall`` 几乎相等，把"打分路径自身合法"与
        "``__post_init__`` 兜底"两条边界拆开。

        **Validates: Requirements 17**
        """
        scorer = QualityScorer()
        profile = _make_default_profile()
        score = scorer.score(original, processed, profile)

        assert 0.0 <= score.overall <= 1.0
        # 同时检查没有走到夹紧分支（即子分加权和已经在 [0,1] 内）。
        weighted = (
            WEIGHT_TEXT_RETENTION * score.components["text_retention"]
            + WEIGHT_HEADING_DETECTION * score.components["heading_detection"]
            + WEIGHT_TABLE_COMPLETENESS * score.components["table_completeness"]
            + WEIGHT_NUMERIC_PROTECTION * score.components["numeric_protection"]
            + WEIGHT_BOILERPLATE_REMOVAL * score.components["boilerplate_removal"]
        )
        # round-trip 通过 to_dict 会四舍五入到 4 位小数，但 score.overall
        # 自身没有四舍五入，所以这里只允许极小浮点误差。
        assert abs(score.overall - weighted) <= 1e-9

    @hyp_settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    @given(original=_parsed_doc_strategy, processed=_processed_doc_strategy)
    def test_score_components_always_have_five_keys(self, original, processed):
        """``components`` 永远包含五个固定键，且每个子分都在 ``[0, 1]``。

        **Validates: Requirements 17**
        """
        scorer = QualityScorer()
        profile = _make_default_profile()
        score = scorer.score(original, processed, profile)

        expected_keys = {
            "text_retention",
            "heading_detection",
            "table_completeness",
            "numeric_protection",
            "boilerplate_removal",
        }
        assert set(score.components.keys()) == expected_keys
        for key, value in score.components.items():
            assert 0.0 <= value <= 1.0, f"component {key!r}={value!r} out of [0,1]"

    @hyp_settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.function_scoped_fixture,
        ],
    )
    @given(original=_parsed_doc_strategy, processed=_processed_doc_strategy)
    def test_score_is_deterministic(self, original, processed):
        """同样输入连续两次 ``score()`` → 综合分、全部子分、issues 顺序完全相同。

        审核队列依赖打分稳定：不稳定的分数会让管理员每次刷新看到不同的
        排序，文档误入/逃出审核队列。

        **Validates: Requirements 17**
        """
        scorer = QualityScorer()
        profile = _make_default_profile()
        first = scorer.score(original, processed, profile)
        second = scorer.score(original, processed, profile)

        assert first.overall == second.overall
        assert first.components == second.components
        # issues 是 list[str]，顺序也必须稳定（前端按顺序展示）。
        assert first.issues == second.issues

    def test_trivial_perfect_input_yields_overall_one(self):
        """当输入「平凡满分」时综合分为 1.0。

        平凡满分场景：

        - 单段纯文字段落（无任何数值 token，处理后字符全保留）
        - 无标题（``profile.heading_rules=[]``，预期标题数为 0）
        - 无表格
        - 处理结果为 1 页（< MIN_PAGES_FOR_DETECTION=3，跳过 boilerplate 检测）
        - ``noise_removed_count=0``、``headings_detected=0``

        在这种刻意构造的场景下五个维度都会返回 1.0，加权和也必然是 1.0。
        这是对算法的"零风险输入必须满分"公约的可执行断言。

        **Validates: Requirements 17**
        """
        scorer = QualityScorer()
        profile = _make_default_profile()
        # 不含任何数字 token；空格只用于 _visible_char_count 不计入字符数。
        text = "这是一段没有任何数字的纯文字内容用于属性测试"
        original = ParsedDocument(
            blocks=[Block(type="paragraph", text=text, page_number=1)]
        )
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(type="paragraph", text=text, page_number=1)
            ],
            noise_removed_count=0,
            headings_detected=0,
        )
        score = scorer.score(original, processed, profile)

        assert score.overall == pytest.approx(1.0, abs=1e-9)
        for key, value in score.components.items():
            assert value == pytest.approx(1.0, abs=1e-9), f"component {key} != 1.0"


# ─── End-to-End Integration: Scorer → ReviewQueue → Admin endpoints ───


class TestEndToEndScoringAndReview:
    """E2E 集成：``QualityScorer`` → 真实 ``ReviewQueue.enqueue`` → 管理员
    ``list`` / ``preview`` / ``correct`` 的全链路一致性。

    既有的单元测试把每个组件独立 stub 验证（``test_review_queue.py`` mock
    了 DB；``test_admin_reviews_*.py`` 各自 mock 了 review）。本测试把这些
    层连起来：用一个真实 ``QualityScorer`` 算分、塞进真实
    ``ReviewQueue.enqueue``（写入 mock DB session 中持久化的 ``DocumentReview``
    实例）、然后让 admin 路由从同一个 mock DB 中读到同一行，验证：

    - 低质量文档被打分 → 自动入队（``status='pending'``）
    - admin list 能看到这条 pending review，``quality_score`` 字段透传完整
    - admin preview 能拿到 ``parsed_markdown``、各维度子分与 issues
    - admin correct 提交修正后状态变为 ``corrected``，``corrected_markdown``
      和 ``original_markdown`` 都落到 JSONB

    Validates: Requirements 17
    """

    @pytest.mark.asyncio
    async def test_low_quality_doc_flows_through_full_review_pipeline(self):
        """打分 → 入队 → 列表 → 预览 → 修正 一条龙跑通且数据一致。"""
        from datetime import datetime, timezone
        from unittest.mock import AsyncMock, MagicMock, patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.admin_reviews import router as admin_reviews_router
        from app.api.auth import require_admin
        from app.core.database import get_db
        from app.models.document_review import DocumentReview, ReviewStatus
        from app.services.review_queue import ReviewQueue

        # ─── 1. 构造真实的低质量 ParsedDocument / ProcessedDocument ──
        # 设计：3 页 + 重复页眉 + 数值丢失 + 标题丢失 + 表格丢失，
        # 让综合分明显低于 0.7 阈值，触发入队。
        original = ParsedDocument(
            blocks=[
                # 第 1 页：页眉 + 标题 + 数值段落 + 表格
                Block(type="paragraph", text="公司机密文档", page_number=1),
                Block(
                    type="heading",
                    text="一、技术指标",
                    page_number=1,
                    style={"heading_level": 1},
                ),
                Block(
                    type="paragraph",
                    text="工作温度 25.5°C，压力 100kPa，偏差 ±0.05mm/m",
                    page_number=1,
                ),
                Block(
                    type="table",
                    text=(
                        "| 参数 | 数值 | 单位 |\n"
                        "| --- | --- | --- |\n"
                        "| 温度 | 25 | °C |\n"
                        "| 压力 | 100 | kPa |"
                    ),
                    page_number=1,
                ),
                # 第 2 页：页眉重复 + 标题 + 段落
                Block(type="paragraph", text="公司机密文档", page_number=2),
                Block(
                    type="heading",
                    text="二、应用范围",
                    page_number=2,
                    style={"heading_level": 1},
                ),
                Block(
                    type="paragraph",
                    text="角度 55°~65°，长度 1500mm",
                    page_number=2,
                ),
                # 第 3 页：页眉重复 + 标题
                Block(type="paragraph", text="公司机密文档", page_number=3),
                Block(
                    type="heading",
                    text="三、参考资料",
                    page_number=3,
                    style={"heading_level": 1},
                ),
                Block(
                    type="paragraph",
                    text="质量等级 0.002D，公差 ±10mm",
                    page_number=3,
                ),
            ]
        )

        # 处理结果：模拟低质量解析——仅保留少量正文，未识别标题、表格丢失、
        # 数值丢失、噪声未去除。
        processed = ProcessedDocument(
            blocks=[
                ProcessedBlock(
                    type="paragraph",
                    text="技术指标 工作温度",
                    page_number=1,
                ),
                ProcessedBlock(
                    type="paragraph",
                    text="应用范围",
                    page_number=2,
                ),
            ],
            metadata={"file_type": "pdf"},
            markdown="技术指标 工作温度\n\n应用范围",
            noise_removed_count=0,
            headings_detected=0,
        )

        profile = _make_default_profile()

        # ─── 2. 真实 QualityScorer 算分 → 应当低于阈值 ─────────────
        scorer = QualityScorer()
        score = scorer.score(original, processed, profile)

        assert score.overall < DEFAULT_REVIEW_THRESHOLD, (
            f"测试 fixture 设计错误：打出的综合分 {score.overall:.4f} "
            f"应低于阈值 {DEFAULT_REVIEW_THRESHOLD}"
        )
        assert scorer.needs_review(score) is True
        # 五个维度都应留下可读 issue（由各维度阈值触发）
        assert len(score.issues) > 0

        # ─── 3. 真实 ReviewQueue.enqueue → 写入 mock DB ────────────
        # 让 mock DB 保存 add() 进来的 DocumentReview 实例，并让后续
        # SELECT * FROM document_reviews WHERE id = ? 能拿到同一实例。
        document_id = uuid.uuid4()
        added_reviews: list[DocumentReview] = []

        db = AsyncMock()
        db.add = MagicMock(side_effect=lambda obj: added_reviews.append(obj))
        db.flush = AsyncMock()

        async def _refresh(obj):
            # 模拟 DB 在 flush 后填充自动列；这里只填 id / created_at。
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            if getattr(obj, "created_at", None) is None:
                obj.created_at = datetime.now(timezone.utc)

        db.refresh = AsyncMock(side_effect=_refresh)

        # 第一次 enqueue：``_find_pending`` 返回 None。
        first_select_result = MagicMock()
        first_select_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=first_select_result)

        # 把 ProcessedDocument.markdown 一并写入 JSONB（任务 11.10），让
        # admin preview 能直接拿到 parsed_markdown。
        queue = ReviewQueue(db)
        review = await queue.enqueue(
            document_id,
            score,
            extra_payload={"parsed_markdown": processed.markdown},
        )

        assert isinstance(review, DocumentReview)
        assert review.document_id == document_id
        assert review.status == ReviewStatus.pending
        assert review.quality_score["overall"] == pytest.approx(
            score.overall, abs=1e-4
        )
        assert review.quality_score["parsed_markdown"] == processed.markdown
        # 5 个子分键都在 JSONB 里
        assert set(review.quality_score["components"].keys()) == {
            "text_retention",
            "heading_detection",
            "table_completeness",
            "numeric_protection",
            "boilerplate_removal",
        }
        # 入队记录在 mock DB 中
        assert added_reviews == [review]

        # ─── 4. 装配 FastAPI test client，复用同一份 review 数据 ───
        # 用一个独立的 admin_db mock：list 走 (count, items) 两次 execute；
        # preview 走单次 select review；correct 走单次 select review。
        admin_db = AsyncMock()
        admin_db.flush = AsyncMock()
        admin_db.refresh = AsyncMock()

        # 给 review.document 挂上 selectin 加载出来的 Document mock。
        document = MagicMock()
        document.id = document_id
        document.title = "技术规范测试样例.pdf"
        document.storage_path = f"spaces/test/{document_id}.pdf"
        document.space_id = uuid.uuid4()
        document.matched_profile_id = uuid.uuid4()
        review.document = document

        admin_user = MagicMock()
        admin_user.id = uuid.uuid4()
        admin_user.email = "admin@wikforge.local"

        application = FastAPI()
        application.include_router(admin_reviews_router)

        async def _override_get_db():
            yield admin_db

        async def _override_require_admin():
            return admin_user

        application.dependency_overrides[get_db] = _override_get_db
        application.dependency_overrides[require_admin] = _override_require_admin
        client = TestClient(application)

        # ─── 5. GET /api/admin/reviews → 列表能看到这条 pending ────
        # list_reviews 先 count，再主查询。
        count_result = MagicMock()
        count_result.scalar.return_value = 1
        list_result = MagicMock()
        list_result.all.return_value = [
            (
                review,
                document.title,
                document.space_id,
                document.matched_profile_id,
                "generic-text",
            )
        ]
        admin_db.execute = AsyncMock(side_effect=[count_result, list_result])

        list_resp = client.get("/api/admin/reviews")
        assert list_resp.status_code == 200, list_resp.text
        list_body = list_resp.json()
        assert list_body["total"] == 1
        assert len(list_body["items"]) == 1
        item = list_body["items"][0]
        assert item["review_id"] == str(review.id)
        assert item["document_id"] == str(document_id)
        assert item["status"] == "pending"
        # quality_score JSONB 透传，五个子分都可见
        assert set(item["quality_score"]["components"].keys()) == {
            "text_retention",
            "heading_detection",
            "table_completeness",
            "numeric_protection",
            "boilerplate_removal",
        }
        assert item["quality_score"]["overall"] == pytest.approx(
            score.overall, abs=1e-4
        )

        # ─── 6. GET /api/admin/reviews/{id}/preview → 拿到完整解析 ──
        preview_result = MagicMock()
        preview_result.scalar_one_or_none.return_value = review
        admin_db.execute = AsyncMock(return_value=preview_result)

        with patch(
            "app.api.admin_reviews.generate_presigned_get_url",
            return_value="https://minio.test/preview-url",
        ):
            preview_resp = client.get(f"/api/admin/reviews/{review.id}/preview")
        assert preview_resp.status_code == 200, preview_resp.text
        preview_body = preview_resp.json()
        assert preview_body["review_id"] == str(review.id)
        assert preview_body["document_id"] == str(document_id)
        assert preview_body["document_title"] == "技术规范测试样例.pdf"
        assert preview_body["original_file_url"] == "https://minio.test/preview-url"
        # parsed_markdown 来自打分时塞进去的 extra_payload
        assert preview_body["parsed_markdown"] == processed.markdown
        assert preview_body["status"] == "pending"
        # quality_score 各维度在响应中可见
        assert set(preview_body["quality_score"]["components"].keys()) == {
            "text_retention",
            "heading_detection",
            "table_completeness",
            "numeric_protection",
            "boilerplate_removal",
        }
        assert preview_body["quality_score"]["overall"] == pytest.approx(
            score.overall, abs=1e-4
        )
        assert preview_body["quality_score"]["issues"] == score.issues

        # ─── 7. POST /api/admin/reviews/{id}/correct → 状态翻转 ────
        correct_result = MagicMock()
        correct_result.scalar_one_or_none.return_value = review
        admin_db.execute = AsyncMock(return_value=correct_result)

        corrected_md = (
            "# 技术规范测试样例\n\n"
            "## 一、技术指标\n\n"
            "工作温度 25.5°C，压力 100kPa，偏差 ±0.05mm/m\n\n"
            "| 参数 | 数值 | 单位 |\n| --- | --- | --- |\n"
            "| 温度 | 25 | °C |\n| 压力 | 100 | kPa |\n"
        )
        with patch(
            "app.api.admin_reviews.submit_reprocess_from_markdown",
            return_value=True,
        ) as mock_reprocess:
            correct_resp = client.post(
                f"/api/admin/reviews/{review.id}/correct",
                json={
                    "corrected_markdown": corrected_md,
                    "reviewer_note": "标题/数值/表格已补回",
                },
            )

        assert correct_resp.status_code == 200, correct_resp.text
        correct_body = correct_resp.json()
        assert correct_body["review_id"] == str(review.id)
        assert correct_body["status"] == "corrected"

        # 同一 review 实例在内存里被路由就地修改 → 确认状态机切换
        assert review.status == ReviewStatus.corrected
        assert review.reviewer_note == "标题/数值/表格已补回"
        assert review.reviewed_at is not None
        # JSONB 留底完整
        assert review.quality_score["corrected_markdown"] == corrected_md
        # original_markdown 来自 enqueue 阶段写入的 parsed_markdown
        assert review.quality_score["original_markdown"] == processed.markdown
        assert "correction_timestamp" in review.quality_score
        # 五个子分仍然保留（不会因为修正而被擦掉，便于后续 sample 分析）
        assert set(review.quality_score["components"].keys()) == {
            "text_retention",
            "heading_detection",
            "table_completeness",
            "numeric_protection",
            "boilerplate_removal",
        }

        # Celery reprocess 链被触发，参数与 (document_id, corrected_md) 一致
        mock_reprocess.assert_called_once()
        args = mock_reprocess.call_args.args
        assert args[0] == str(document_id)
        assert args[1] == corrected_md
