"""Unit tests for Profile system and feature matching.

Tests cover:
- Data classes: MatchRules, HeadingRule, BoilerplateConfig, TableConfig, ChunkingConfig
- DocumentProfileConfig and serialization (profile_from_dict, profile_to_dict)
- ProfileMatcher.extract_features (filename patterns, numbering, header/footer, table density)
- ProfileMatcher.match (priority matching, tie-breaking by updated_at, default fallback)
- Default profile fallback (generic-text when no match)
- Edge cases (empty documents, invalid regex, disabled profiles)
"""

from datetime import datetime, timedelta

import pytest

from app.services.parsers.base import Block, ParsedDocument
from app.services.profile_matcher import (
    BoilerplateConfig,
    ChunkingConfig,
    DocumentFeatures,
    DocumentProfileConfig,
    HeadingRule,
    MatchRules,
    ProfileMatcher,
    TableConfig,
    profile_from_dict,
    profile_to_dict,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def generic_profile() -> DocumentProfileConfig:
    """Generic-text fallback profile (no match rules)."""
    return DocumentProfileConfig(
        id="profile-generic",
        name="generic-text",
        description="通用文本文档",
        priority=0,
        enabled=True,
        match_rules=MatchRules(filename_regex=[], content_regex=[], min_content_match_count=1),
    )


@pytest.fixture
def chinese_spec_profile() -> DocumentProfileConfig:
    """Chinese technical spec profile."""
    return DocumentProfileConfig(
        id="profile-chinese",
        name="chinese-technical-spec",
        description="中式技术规范文档",
        priority=10,
        enabled=True,
        match_rules=MatchRules(
            filename_regex=[r".*规范.*", r".*标准.*", r".*规程.*"],
            content_regex=[
                r"^[一二三四五六七八九十]+[、．.]",
                r"^\([一二三四五六七八九十]+\)",
                r"^\d+\.\d+",
                r"^第[一二三四五六七八九十百]+[章节条款]",
            ],
            min_content_match_count=2,
        ),
        heading_rules=[
            HeadingRule(pattern=r"^第[一二三四五六七八九十百]+[章]", level=1),
            HeadingRule(pattern=r"^[一二三四五六七八九十]+[、．.]", level=2),
        ],
        updated_at=datetime(2024, 6, 1),
    )


@pytest.fixture
def scanned_pdf_profile() -> DocumentProfileConfig:
    """Scanned PDF profile."""
    return DocumentProfileConfig(
        id="profile-scanned",
        name="scanned-pdf",
        description="扫描版 PDF 文档",
        priority=5,
        enabled=True,
        match_rules=MatchRules(
            filename_regex=[r".*扫描.*", r".*scan.*"],
            content_regex=[],
            min_content_match_count=1,
        ),
        updated_at=datetime(2024, 5, 1),
    )


@pytest.fixture
def all_profiles(generic_profile, chinese_spec_profile, scanned_pdf_profile):
    """All three default profiles."""
    return [generic_profile, chinese_spec_profile, scanned_pdf_profile]


@pytest.fixture
def chinese_spec_document() -> ParsedDocument:
    """A parsed document that looks like a Chinese technical spec."""
    blocks = [
        Block(type="heading", text="第一章 总则", page_number=1),
        Block(type="paragraph", text="1.1 本规范适用于水泥生产线的设计与施工。", page_number=1),
        Block(type="paragraph", text="一、基本要求", page_number=1),
        Block(type="paragraph", text="(一) 设计应符合国家标准。", page_number=1),
        Block(type="paragraph", text="二、技术指标", page_number=2),
        Block(type="paragraph", text="2.1 强度等级不低于 42.5。", page_number=2),
        Block(type="table", text="| 参数 | 值 |\n| 强度 | 42.5 |", page_number=3),
    ]
    return ParsedDocument(blocks=blocks, metadata={"page_count": 3})


@pytest.fixture
def plain_text_document() -> ParsedDocument:
    """A simple plain text document with no special patterns."""
    blocks = [
        Block(type="paragraph", text="This is a simple document.", page_number=1),
        Block(type="paragraph", text="It has no special numbering.", page_number=1),
        Block(type="paragraph", text="Just plain text content.", page_number=1),
    ]
    return ParsedDocument(blocks=blocks, metadata={"page_count": 1})


@pytest.fixture
def scanned_document() -> ParsedDocument:
    """A document that appears to be scanned (very little text)."""
    blocks = [
        Block(type="paragraph", text="", page_number=1),
        Block(type="paragraph", text="a", page_number=2),
        Block(type="paragraph", text="", page_number=3),
        Block(type="paragraph", text="b", page_number=4),
        Block(type="paragraph", text="", page_number=5),
    ]
    return ParsedDocument(blocks=blocks, metadata={"page_count": 5})


# ─── Data Class Tests ──────────────────────────────────────────────────


class TestDataClasses:
    """Tests for Pydantic/dataclass models for JSONB fields."""

    def test_match_rules_defaults(self):
        """MatchRules has sensible defaults."""
        rules = MatchRules()
        assert rules.filename_regex == []
        assert rules.content_regex == []
        assert rules.min_content_match_count == 1

    def test_heading_rule_creation(self):
        """HeadingRule stores pattern, level, and strip_pattern."""
        rule = HeadingRule(pattern=r"^#{1,6}\s+", level=1, strip_pattern=True)
        assert rule.pattern == r"^#{1,6}\s+"
        assert rule.level == 1
        assert rule.strip_pattern is True

    def test_boilerplate_config_defaults(self):
        """BoilerplateConfig has correct defaults."""
        config = BoilerplateConfig()
        assert config.detection_mode == "statistical"
        assert config.statistical_threshold == 0.5
        assert config.manual_patterns == []

    def test_table_config_defaults(self):
        """TableConfig has correct defaults."""
        config = TableConfig()
        assert config.cross_page_merge is True
        assert config.row_level_chunking is False
        assert config.collapse_merged_cells == "describe"

    def test_chunking_config_defaults(self):
        """ChunkingConfig has correct defaults."""
        config = ChunkingConfig()
        assert config.min_tokens == 256
        assert config.max_tokens == 800
        assert config.overlap_tokens == 80
        assert config.respect_heading_level == 1
        assert config.protect_patterns == []

    def test_document_profile_config_defaults(self):
        """DocumentProfileConfig has correct defaults."""
        profile = DocumentProfileConfig(id="test", name="test-profile")
        assert profile.priority == 0
        assert profile.enabled is True
        assert profile.version == 1
        assert profile.domain_dictionary_id is None


# ─── Serialization Tests ───────────────────────────────────────────────


class TestSerialization:
    """Tests for profile_from_dict and profile_to_dict."""

    def test_profile_from_dict_full(self):
        """profile_from_dict correctly deserializes all fields."""
        data = {
            "id": "abc-123",
            "name": "test-profile",
            "description": "A test profile",
            "priority": 5,
            "enabled": True,
            "match_rules": {
                "filename_regex": [r".*test.*"],
                "content_regex": [r"^hello"],
                "min_content_match_count": 2,
            },
            "heading_rules": [
                {"pattern": r"^#\s+", "level": 1, "strip_pattern": True},
            ],
            "boilerplate": {
                "detection_mode": "both",
                "statistical_threshold": 0.6,
                "manual_patterns": [r"^Page \d+"],
            },
            "tables": {
                "cross_page_merge": False,
                "row_level_chunking": True,
                "collapse_merged_cells": "repeat",
            },
            "chunking": {
                "min_tokens": 128,
                "max_tokens": 1024,
                "overlap_tokens": 64,
                "respect_heading_level": 2,
                "protect_patterns": [r"\d+mm"],
            },
            "domain_dictionary_id": "dict-001",
            "version": 3,
        }

        profile = profile_from_dict(data)

        assert profile.id == "abc-123"
        assert profile.name == "test-profile"
        assert profile.description == "A test profile"
        assert profile.priority == 5
        assert profile.match_rules.filename_regex == [r".*test.*"]
        assert profile.match_rules.min_content_match_count == 2
        assert len(profile.heading_rules) == 1
        assert profile.heading_rules[0].level == 1
        assert profile.boilerplate.detection_mode == "both"
        assert profile.tables.row_level_chunking is True
        assert profile.chunking.max_tokens == 1024
        assert profile.domain_dictionary_id == "dict-001"
        assert profile.version == 3

    def test_profile_from_dict_minimal(self):
        """profile_from_dict handles minimal data with defaults."""
        data = {"id": "min", "name": "minimal"}
        profile = profile_from_dict(data)

        assert profile.id == "min"
        assert profile.name == "minimal"
        assert profile.priority == 0
        assert profile.match_rules.filename_regex == []
        assert profile.heading_rules == []
        assert profile.chunking.max_tokens == 800

    def test_profile_to_dict_roundtrip(self):
        """profile_to_dict produces a dict that can be deserialized back."""
        original = DocumentProfileConfig(
            id="roundtrip",
            name="roundtrip-test",
            description="Testing roundtrip",
            priority=7,
            enabled=True,
            match_rules=MatchRules(
                filename_regex=[r".*\.pdf"],
                content_regex=[r"^Chapter"],
                min_content_match_count=1,
            ),
            heading_rules=[HeadingRule(pattern=r"^Chapter", level=1)],
            boilerplate=BoilerplateConfig(detection_mode="manual", manual_patterns=[r"^Footer"]),
            tables=TableConfig(cross_page_merge=False),
            chunking=ChunkingConfig(max_tokens=512, protect_patterns=[r"\d+kg"]),
            domain_dictionary_id="dict-x",
            version=2,
        )

        serialized = profile_to_dict(original)
        restored = profile_from_dict(serialized)

        assert restored.name == original.name
        assert restored.priority == original.priority
        assert restored.match_rules.filename_regex == original.match_rules.filename_regex
        assert restored.heading_rules[0].pattern == original.heading_rules[0].pattern
        assert restored.chunking.max_tokens == 512
        assert restored.domain_dictionary_id == "dict-x"


# ─── Feature Extraction Tests ──────────────────────────────────────────


class TestFeatureExtraction:
    """Tests for ProfileMatcher.extract_features."""

    def test_extract_features_chinese_spec(self, all_profiles, chinese_spec_document):
        """Feature extraction detects Chinese numbering patterns."""
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(chinese_spec_document, "水泥规范.pdf")

        assert features.filename == "水泥规范.pdf"
        assert features.page_count == 3
        assert "chinese_chapter" in features.numbering_patterns
        assert "chinese_numbering" in features.numbering_patterns
        assert "decimal_section" in features.numbering_patterns

    def test_extract_features_plain_text(self, all_profiles, plain_text_document):
        """Feature extraction for plain text has no special patterns."""
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(plain_text_document, "readme.txt")

        assert features.filename == "readme.txt"
        assert features.numbering_patterns == []
        assert features.table_density == 0.0

    def test_extract_features_table_density(self, all_profiles):
        """Feature extraction calculates table density correctly."""
        blocks = [
            Block(type="table", text="| A | B |", page_number=1),
            Block(type="paragraph", text="text", page_number=1),
            Block(type="table", text="| C | D |", page_number=1),
            Block(type="paragraph", text="more text", page_number=1),
        ]
        doc = ParsedDocument(blocks=blocks, metadata={})
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(doc, "tables.pdf")

        assert features.table_density == 0.5  # 2 tables out of 4 blocks

    def test_extract_features_empty_document(self, all_profiles):
        """Feature extraction handles empty documents gracefully."""
        doc = ParsedDocument(blocks=[], metadata={})
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(doc, "empty.txt")

        assert features.filename == "empty.txt"
        assert features.sample_text == ""
        assert features.numbering_patterns == []
        assert features.page_count == 0

    def test_extract_features_scanned_detection(self, all_profiles, scanned_document):
        """Feature extraction detects scanned documents (low text per page)."""
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(scanned_document, "scan_001.pdf")

        assert features.appears_scanned is True
        assert features.avg_text_per_page < 50

    def test_extract_features_header_footer_repetition(self, all_profiles):
        """Feature extraction detects repeated headers/footers."""
        blocks = []
        for page in range(1, 6):
            blocks.append(Block(type="paragraph", text="Company Header", page_number=page))
            blocks.append(Block(type="paragraph", text=f"Content on page {page}", page_number=page))
            blocks.append(Block(type="paragraph", text="Page Footer", page_number=page))

        doc = ParsedDocument(blocks=blocks, metadata={})
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(doc, "report.pdf")

        # All pages have same first and last block text
        assert features.header_footer_repetition == 1.0

    def test_extract_features_samples_first_n_pages(self, all_profiles):
        """Feature extraction only samples first N pages."""
        blocks = []
        # Pages 1-5: Chinese numbering
        for page in range(1, 6):
            blocks.append(Block(type="paragraph", text="一、内容", page_number=page))
        # Pages 6-10: English content (should not be sampled)
        for page in range(6, 11):
            blocks.append(Block(type="paragraph", text="English content only", page_number=page))

        doc = ParsedDocument(blocks=blocks, metadata={})
        matcher = ProfileMatcher(profiles=all_profiles)
        features = matcher.extract_features(doc, "mixed.pdf")

        # Chinese patterns should be detected from first 5 pages
        assert "chinese_numbering" in features.numbering_patterns


# ─── Profile Matching Tests ────────────────────────────────────────────


class TestProfileMatching:
    """Tests for ProfileMatcher.match."""

    def test_match_chinese_spec_by_filename(self, all_profiles):
        """Matches chinese-technical-spec by filename pattern."""
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="Some content", page_number=1)],
            metadata={},
        )
        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(doc, "建筑施工规范2024.pdf")

        assert result.name == "chinese-technical-spec"

    def test_match_chinese_spec_by_content(self, all_profiles, chinese_spec_document):
        """Matches chinese-technical-spec by content patterns."""
        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(chinese_spec_document, "document.pdf")

        assert result.name == "chinese-technical-spec"

    def test_match_scanned_pdf_by_filename(self, all_profiles):
        """Matches scanned-pdf by filename pattern."""
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="Some text", page_number=1)],
            metadata={},
        )
        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(doc, "合同扫描件.pdf")

        assert result.name == "scanned-pdf"

    def test_match_scanned_pdf_english_filename(self, all_profiles):
        """Matches scanned-pdf by English filename pattern."""
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="Some text", page_number=1)],
            metadata={},
        )
        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(doc, "contract_scan_v2.pdf")

        assert result.name == "scanned-pdf"

    def test_match_fallback_to_generic(self, all_profiles, plain_text_document):
        """Falls back to generic-text when no profile matches."""
        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(plain_text_document, "readme.txt")

        assert result.name == "generic-text"

    def test_match_priority_ordering(self):
        """Higher priority profile wins when multiple match."""
        low_priority = DocumentProfileConfig(
            id="low",
            name="low-priority",
            priority=1,
            match_rules=MatchRules(filename_regex=[r".*\.pdf"]),
            updated_at=datetime(2024, 1, 1),
        )
        high_priority = DocumentProfileConfig(
            id="high",
            name="high-priority",
            priority=10,
            match_rules=MatchRules(filename_regex=[r".*\.pdf"]),
            updated_at=datetime(2024, 1, 1),
        )
        generic = DocumentProfileConfig(id="gen", name="generic-text", priority=0)

        matcher = ProfileMatcher(profiles=[low_priority, high_priority, generic])
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "test.pdf")

        assert result.name == "high-priority"

    def test_match_tiebreak_by_updated_at(self):
        """Same priority profiles are tie-broken by most recently updated."""
        older = DocumentProfileConfig(
            id="older",
            name="older-profile",
            priority=5,
            match_rules=MatchRules(filename_regex=[r".*\.pdf"]),
            updated_at=datetime(2024, 1, 1),
        )
        newer = DocumentProfileConfig(
            id="newer",
            name="newer-profile",
            priority=5,
            match_rules=MatchRules(filename_regex=[r".*\.pdf"]),
            updated_at=datetime(2024, 6, 15),
        )
        generic = DocumentProfileConfig(id="gen", name="generic-text", priority=0)

        matcher = ProfileMatcher(profiles=[older, newer, generic])
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "report.pdf")

        assert result.name == "newer-profile"

    def test_match_disabled_profiles_skipped(self, generic_profile):
        """Disabled profiles are not considered for matching."""
        disabled = DocumentProfileConfig(
            id="disabled",
            name="disabled-profile",
            priority=100,
            enabled=False,
            match_rules=MatchRules(filename_regex=[r".*"]),
        )

        matcher = ProfileMatcher(profiles=[disabled, generic_profile])
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "anything.pdf")

        assert result.name == "generic-text"

    def test_match_empty_profiles_list(self):
        """Returns hardcoded default when no profiles are loaded."""
        matcher = ProfileMatcher(profiles=[])
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "test.txt")

        assert result.name == "generic-text"
        assert result.id == "default"

    def test_match_content_regex_min_count(self):
        """Content regex requires min_content_match_count hits."""
        profile = DocumentProfileConfig(
            id="strict",
            name="strict-profile",
            priority=5,
            match_rules=MatchRules(
                content_regex=[r"^Pattern_A", r"^Pattern_B", r"^Pattern_C"],
                min_content_match_count=2,
            ),
        )
        generic = DocumentProfileConfig(id="gen", name="generic-text", priority=0)

        # Only 1 pattern matches - should NOT match
        doc_one_match = ParsedDocument(
            blocks=[Block(type="paragraph", text="Pattern_A found here", page_number=1)],
            metadata={},
        )
        matcher = ProfileMatcher(profiles=[profile, generic])
        result = matcher.match(doc_one_match, "test.txt")
        assert result.name == "generic-text"

        # 2 patterns match - should match
        doc_two_matches = ParsedDocument(
            blocks=[
                Block(type="paragraph", text="Pattern_A found here", page_number=1),
                Block(type="paragraph", text="Pattern_B also here", page_number=1),
            ],
            metadata={},
        )
        result = matcher.match(doc_two_matches, "test.txt")
        assert result.name == "strict-profile"

    def test_match_invalid_regex_handled_gracefully(self, generic_profile):
        """Invalid regex patterns don't crash the matcher."""
        bad_profile = DocumentProfileConfig(
            id="bad",
            name="bad-regex",
            priority=5,
            match_rules=MatchRules(
                filename_regex=[r"[invalid(regex"],
                content_regex=[r"(unclosed"],
            ),
        )

        matcher = ProfileMatcher(profiles=[bad_profile, generic_profile])
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        # Should not raise, just fall back to generic
        result = matcher.match(doc, "test.txt")
        assert result.name == "generic-text"

    def test_match_filename_case_insensitive(self):
        """Filename matching is case-insensitive."""
        profile = DocumentProfileConfig(
            id="case",
            name="case-test",
            priority=5,
            match_rules=MatchRules(filename_regex=[r".*SCAN.*"]),
        )
        generic = DocumentProfileConfig(id="gen", name="generic-text", priority=0)

        matcher = ProfileMatcher(profiles=[profile, generic])
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "document_scan_v1.pdf")
        assert result.name == "case-test"


# ─── Default Fallback Tests ────────────────────────────────────────────


class TestDefaultFallback:
    """Tests for default profile fallback behavior."""

    def test_fallback_returns_generic_text_from_list(self, all_profiles):
        """Fallback returns the generic-text profile from the loaded list."""
        matcher = ProfileMatcher(profiles=all_profiles)
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="nothing special", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "random_file.txt")

        assert result.name == "generic-text"
        assert result.id == "profile-generic"

    def test_fallback_hardcoded_when_no_generic_in_list(self):
        """Fallback uses hardcoded default when generic-text is not in list."""
        profiles = [
            DocumentProfileConfig(
                id="only",
                name="only-profile",
                priority=5,
                match_rules=MatchRules(filename_regex=[r".*specific.*"]),
            ),
        ]
        matcher = ProfileMatcher(profiles=profiles)
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="content", page_number=1)],
            metadata={},
        )
        result = matcher.match(doc, "unmatched.txt")

        assert result.name == "generic-text"
        assert result.id == "default"

    def test_generic_text_profile_does_not_actively_match(self, all_profiles):
        """generic-text with empty match rules doesn't actively match documents."""
        matcher = ProfileMatcher(profiles=all_profiles)

        # A document that could match generic-text if it were active
        doc = ParsedDocument(
            blocks=[Block(type="paragraph", text="anything", page_number=1)],
            metadata={},
        )
        # The generic-text profile should only be returned as fallback,
        # not as an active match
        result = matcher.match(doc, "test.txt")
        assert result.name == "generic-text"


# ─── Integration-style Tests ───────────────────────────────────────────


class TestProfileMatcherIntegration:
    """Integration tests combining feature extraction and matching."""

    def test_full_flow_chinese_spec(self, all_profiles):
        """Full flow: Chinese spec document → chinese-technical-spec profile."""
        blocks = [
            Block(type="heading", text="第一章 总则", page_number=1),
            Block(type="paragraph", text="一、适用范围", page_number=1),
            Block(type="paragraph", text="(一) 本标准适用于...", page_number=1),
            Block(type="paragraph", text="1.1 基本要求", page_number=2),
            Block(type="paragraph", text="二、技术要求", page_number=2),
        ]
        doc = ParsedDocument(blocks=blocks, metadata={"page_count": 2})

        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(doc, "技术标准_v2.pdf")

        assert result.name == "chinese-technical-spec"
        assert result.priority == 10

    def test_full_flow_scanned_pdf(self, all_profiles):
        """Full flow: Scanned filename → scanned-pdf profile."""
        blocks = [Block(type="paragraph", text="OCR text", page_number=1)]
        doc = ParsedDocument(blocks=blocks, metadata={})

        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(doc, "合同扫描件_2024.pdf")

        assert result.name == "scanned-pdf"

    def test_full_flow_generic_fallback(self, all_profiles):
        """Full flow: Plain English document → generic-text fallback."""
        blocks = [
            Block(type="paragraph", text="Introduction to the project.", page_number=1),
            Block(type="paragraph", text="This document describes the architecture.", page_number=1),
        ]
        doc = ParsedDocument(blocks=blocks, metadata={})

        matcher = ProfileMatcher(profiles=all_profiles)
        result = matcher.match(doc, "architecture_overview.md")

        assert result.name == "generic-text"

    def test_profiles_setter(self, all_profiles, chinese_spec_document):
        """ProfileMatcher.profiles can be set after initialization."""
        matcher = ProfileMatcher()
        assert matcher.profiles == []

        matcher.profiles = all_profiles
        result = matcher.match(chinese_spec_document, "规范.pdf")
        assert result.name == "chinese-technical-spec"
