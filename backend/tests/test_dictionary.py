"""Unit tests for domain dictionary management.

Tests cover:
- Term validation (length, control characters)
- IK dictionary file generation
- CSV/JSON import/export
- Candidate term extraction
- Enable/disable logic
- Preset dictionary
"""

import csv
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.services.dictionary_service import (
    CHINESE_STOP_WORDS,
    DictionaryService,
    Term,
    SynonymGroup,
    generate_ik_dict_content,
    generate_ik_stopword_content,
    validate_term,
    _extract_chinese_words,
)


# ─── Term Validation Tests ─────────────────────────────────────────────


class TestTermValidation:
    """Tests for term format validation."""

    def test_valid_term_chinese(self):
        """Chinese term within length limit passes."""
        is_valid, msg = validate_term("大齿圈")
        assert is_valid is True
        assert msg == ""

    def test_valid_term_english(self):
        """English term within length limit passes."""
        is_valid, msg = validate_term("bearing")
        assert is_valid is True
        assert msg == ""

    def test_valid_term_mixed(self):
        """Mixed Chinese-English term passes."""
        is_valid, msg = validate_term("API接口")
        assert is_valid is True

    def test_valid_term_single_char(self):
        """Single character term passes (min length 1)."""
        is_valid, msg = validate_term("钢")
        assert is_valid is True

    def test_valid_term_30_chars(self):
        """Term with exactly 30 characters passes."""
        word = "a" * 30
        is_valid, msg = validate_term(word)
        assert is_valid is True

    def test_invalid_term_empty(self):
        """Empty term fails."""
        is_valid, msg = validate_term("")
        assert is_valid is False
        assert "不能为空" in msg

    def test_invalid_term_whitespace_only(self):
        """Whitespace-only term fails."""
        is_valid, msg = validate_term("   ")
        assert is_valid is False
        assert "不能为空" in msg

    def test_invalid_term_too_long(self):
        """Term exceeding 30 characters fails."""
        word = "a" * 31
        is_valid, msg = validate_term(word)
        assert is_valid is False
        assert "1-30" in msg

    def test_invalid_term_control_char_null(self):
        """Term with null byte fails."""
        is_valid, msg = validate_term("hello\x00world")
        assert is_valid is False
        assert "控制字符" in msg

    def test_invalid_term_control_char_bell(self):
        """Term with bell character fails."""
        is_valid, msg = validate_term("test\x07term")
        assert is_valid is False
        assert "控制字符" in msg

    def test_invalid_term_control_char_escape(self):
        """Term with escape character fails."""
        is_valid, msg = validate_term("test\x1bterm")
        assert is_valid is False
        assert "控制字符" in msg

    def test_valid_term_with_normal_whitespace(self):
        """Term with normal spaces is valid (stripped)."""
        is_valid, msg = validate_term(" 水泥 ")
        assert is_valid is True

    def test_valid_term_with_newline_in_content(self):
        """Term with tab/newline - these are not control chars in our pattern."""
        # Tab (\x09), newline (\x0a), carriage return (\x0d) are NOT blocked
        is_valid, msg = validate_term("hello\tworld")
        assert is_valid is True

    def test_invalid_term_c1_control_char(self):
        """Term with C1 control character (0x80-0x9f) fails."""
        is_valid, msg = validate_term("test\x80term")
        assert is_valid is False
        assert "控制字符" in msg


# ─── IK Dictionary Generation Tests ───────────────────────────────────


class TestIKDictGeneration:
    """Tests for IK dictionary file content generation."""

    def test_generate_ik_dict_content_basic(self):
        """Generates one word per line."""
        terms = [Term(word="大齿圈"), Term(word="回转窑"), Term(word="水泥")]
        content = generate_ik_dict_content(terms)
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert "大齿圈" in lines
        assert "回转窑" in lines
        assert "水泥" in lines

    def test_generate_ik_dict_content_deduplicates(self):
        """Duplicate terms are deduplicated."""
        terms = [Term(word="水泥"), Term(word="水泥"), Term(word="钢材")]
        content = generate_ik_dict_content(terms)
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_generate_ik_dict_content_sorted(self):
        """Output is sorted."""
        terms = [Term(word="钢材"), Term(word="水泥"), Term(word="大齿圈")]
        content = generate_ik_dict_content(terms)
        lines = content.strip().split("\n")
        assert lines == sorted(lines)

    def test_generate_ik_dict_content_empty(self):
        """Empty term list produces empty content."""
        content = generate_ik_dict_content([])
        assert content == ""

    def test_generate_ik_dict_content_skips_empty_words(self):
        """Terms with empty words are skipped."""
        terms = [Term(word="水泥"), Term(word=""), Term(word="  ")]
        content = generate_ik_dict_content(terms)
        lines = content.strip().split("\n")
        assert len(lines) == 1
        assert "水泥" in lines

    def test_generate_ik_stopword_content(self):
        """Generates stopword file content."""
        stop_words = ["的", "了", "是"]
        content = generate_ik_stopword_content(stop_words)
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert "的" in lines

    def test_generate_ik_stopword_deduplicates(self):
        """Duplicate stop words are deduplicated."""
        stop_words = ["的", "的", "了"]
        content = generate_ik_stopword_content(stop_words)
        lines = content.strip().split("\n")
        assert len(lines) == 2


# ─── CSV Import/Export Tests ───────────────────────────────────────────


class TestCSVImportExport:
    """Tests for CSV import and export functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db = AsyncMock()
        self.service = DictionaryService(self.db)

    def test_export_csv_basic(self):
        """Export produces valid CSV with header."""
        dictionary = MagicMock()
        dictionary.name = "test"
        dictionary.terms = [
            {"word": "水泥", "pos": "n", "weight": 1.0},
            {"word": "钢材", "pos": "n", "weight": 0.8},
        ]

        csv_content = self.service.export_as_csv(dictionary)
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        assert rows[0] == ["word", "pos", "weight"]
        assert rows[1] == ["水泥", "n", "1.0"]
        assert rows[2] == ["钢材", "n", "0.8"]

    def test_export_csv_string_terms(self):
        """Export handles string-only terms."""
        dictionary = MagicMock()
        dictionary.name = "test"
        dictionary.terms = ["水泥", "钢材"]

        csv_content = self.service.export_as_csv(dictionary)
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        assert rows[1] == ["水泥", "", "1.0"]

    def test_import_csv_basic(self):
        """Import parses CSV with header."""
        csv_content = "word,pos,weight\n水泥,n,1.0\n钢材,n,0.8\n"
        terms = self.service.import_from_csv(csv_content)

        assert len(terms) == 2
        assert terms[0]["word"] == "水泥"
        assert terms[0]["pos"] == "n"
        assert terms[0]["weight"] == 1.0
        assert terms[1]["word"] == "钢材"
        assert terms[1]["weight"] == 0.8

    def test_import_csv_no_header(self):
        """Import handles CSV without standard header."""
        csv_content = "水泥,n,1.0\n钢材,n,0.8\n"
        terms = self.service.import_from_csv(csv_content)

        assert len(terms) == 2
        assert terms[0]["word"] == "水泥"

    def test_import_csv_minimal_columns(self):
        """Import handles CSV with only word column."""
        csv_content = "word\n水泥\n钢材\n"
        terms = self.service.import_from_csv(csv_content)

        assert len(terms) == 2
        assert terms[0]["word"] == "水泥"
        assert terms[0]["pos"] is None
        assert terms[0]["weight"] == 1.0

    def test_import_csv_skips_invalid_terms(self):
        """Import skips terms that fail validation."""
        long_word = "a" * 31
        csv_content = f"word,pos,weight\n水泥,n,1.0\n{long_word},n,1.0\n"
        terms = self.service.import_from_csv(csv_content)

        assert len(terms) == 1
        assert terms[0]["word"] == "水泥"

    def test_import_csv_empty_rows(self):
        """Import skips empty rows."""
        csv_content = "word,pos,weight\n\n水泥,n,1.0\n\n"
        terms = self.service.import_from_csv(csv_content)

        assert len(terms) == 1


# ─── JSON Import/Export Tests ──────────────────────────────────────────


class TestJSONImportExport:
    """Tests for JSON import and export functionality."""

    def setup_method(self):
        self.db = AsyncMock()
        self.service = DictionaryService(self.db)

    def test_export_json(self):
        """Export produces complete JSON structure."""
        dictionary = MagicMock()
        dictionary.name = "水泥行业术语"
        dictionary.description = "水泥行业专业术语"
        dictionary.terms = [{"word": "大齿圈", "pos": "n", "weight": 1.0}]
        dictionary.synonyms = [{"primary": "大齿圈", "synonyms": ["齿圈"]}]
        dictionary.stop_words = ["的", "了"]
        dictionary.enabled = True

        result = self.service.export_as_json(dictionary)

        assert result["name"] == "水泥行业术语"
        assert result["description"] == "水泥行业专业术语"
        assert len(result["terms"]) == 1
        assert len(result["synonyms"]) == 1
        assert len(result["stop_words"]) == 2
        assert result["enabled"] is True

    def test_import_json_basic(self):
        """Import parses valid JSON data."""
        json_data = {
            "terms": [
                {"word": "水泥", "pos": "n", "weight": 1.0},
                {"word": "钢材", "pos": "n", "weight": 0.8},
            ],
            "synonyms": [{"primary": "大齿圈", "synonyms": ["齿圈"]}],
            "stop_words": ["的", "了"],
        }
        result = self.service.import_from_json(json_data)

        assert len(result["terms"]) == 2
        assert len(result["synonyms"]) == 1
        assert len(result["stop_words"]) == 2

    def test_import_json_string_terms(self):
        """Import handles string-only terms in JSON."""
        json_data = {
            "terms": ["水泥", "钢材"],
            "synonyms": [],
            "stop_words": [],
        }
        result = self.service.import_from_json(json_data)

        assert len(result["terms"]) == 2
        assert result["terms"][0] == {"word": "水泥", "pos": None, "weight": 1.0}

    def test_import_json_filters_invalid(self):
        """Import filters out invalid terms."""
        json_data = {
            "terms": [
                {"word": "水泥", "pos": "n", "weight": 1.0},
                {"word": "a" * 31, "pos": "n", "weight": 1.0},  # too long
                {"word": "", "pos": "n", "weight": 1.0},  # empty
            ],
            "synonyms": [],
            "stop_words": [],
        }
        result = self.service.import_from_json(json_data)

        assert len(result["terms"]) == 1
        assert result["terms"][0]["word"] == "水泥"


# ─── Candidate Term Extraction Tests ──────────────────────────────────


class TestCandidateExtraction:
    """Tests for candidate term extraction from documents."""

    def test_extract_chinese_words_basic(self):
        """Extracts Chinese character sequences as n-grams."""
        text = "水泥回转窑是水泥生产的核心设备"
        words = _extract_chinese_words(text, min_length=2, max_length=4)

        # Should contain various n-grams
        assert "水泥" in words
        assert "回转" in words
        assert "回转窑" in words

    def test_extract_chinese_words_min_length(self):
        """Respects minimum length parameter."""
        text = "水泥回转窑"
        words = _extract_chinese_words(text, min_length=3, max_length=5)

        # Should not contain 2-char words
        assert "水泥" not in words
        assert "回转窑" in words

    def test_extract_chinese_words_max_length(self):
        """Respects maximum length parameter."""
        text = "水泥回转窑设备"
        words = _extract_chinese_words(text, min_length=2, max_length=3)

        # Should not contain words longer than 3
        four_char_words = [w for w in words if len(w) > 3]
        assert len(four_char_words) == 0

    def test_extract_chinese_words_no_chinese(self):
        """Returns empty for non-Chinese text."""
        text = "This is English text only"
        words = _extract_chinese_words(text, min_length=2, max_length=4)
        assert len(words) == 0

    def test_extract_chinese_words_mixed(self):
        """Handles mixed Chinese-English text."""
        text = "使用API接口调用LLM模型"
        words = _extract_chinese_words(text, min_length=2, max_length=4)

        assert "使用" in words
        assert "接口" in words
        assert "调用" in words


# ─── Preset Dictionary Tests ──────────────────────────────────────────


class TestPresetDictionary:
    """Tests for preset dictionary content."""

    def test_chinese_stop_words_not_empty(self):
        """Preset stop words list is not empty."""
        assert len(CHINESE_STOP_WORDS) > 0

    def test_chinese_stop_words_contains_common(self):
        """Preset contains common Chinese stop words."""
        assert "的" in CHINESE_STOP_WORDS
        assert "了" in CHINESE_STOP_WORDS
        assert "是" in CHINESE_STOP_WORDS
        assert "在" in CHINESE_STOP_WORDS
        assert "和" in CHINESE_STOP_WORDS

    def test_chinese_stop_words_no_duplicates(self):
        """Preset stop words have no duplicates."""
        assert len(CHINESE_STOP_WORDS) == len(set(CHINESE_STOP_WORDS))

    def test_chinese_stop_words_all_valid(self):
        """All preset stop words pass validation."""
        for word in CHINESE_STOP_WORDS:
            is_valid, msg = validate_term(word)
            assert is_valid, f"Stop word '{word}' failed validation: {msg}"


# ─── Data Structure Tests ─────────────────────────────────────────────


class TestDataStructures:
    """Tests for Term and SynonymGroup data structures."""

    def test_term_defaults(self):
        """Term has correct defaults."""
        term = Term(word="水泥")
        assert term.word == "水泥"
        assert term.pos is None
        assert term.weight == 1.0

    def test_term_with_all_fields(self):
        """Term accepts all fields."""
        term = Term(word="水泥", pos="n", weight=0.8)
        assert term.word == "水泥"
        assert term.pos == "n"
        assert term.weight == 0.8

    def test_synonym_group_defaults(self):
        """SynonymGroup has correct defaults."""
        sg = SynonymGroup(primary="大齿圈")
        assert sg.primary == "大齿圈"
        assert sg.synonyms == []

    def test_synonym_group_with_synonyms(self):
        """SynonymGroup accepts synonym list."""
        sg = SynonymGroup(primary="大齿圈", synonyms=["齿圈", "主齿圈"])
        assert sg.primary == "大齿圈"
        assert len(sg.synonyms) == 2
        assert "齿圈" in sg.synonyms


# ─── IK Sync Integration Tests ────────────────────────────────────────


class TestIKSyncIntegration:
    """Integration tests for IK dictionary sync logic."""

    @pytest.mark.asyncio
    async def test_sync_ik_dictionaries_enabled_only(self):
        """Only enabled dictionaries are synced to IK."""
        mock_db = AsyncMock()

        # Mock enabled dictionary
        enabled_dict = MagicMock()
        enabled_dict.terms = [{"word": "水泥", "pos": "n", "weight": 1.0}]
        enabled_dict.stop_words = ["的"]
        enabled_dict.enabled = True

        # Mock query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [enabled_dict]
        mock_db.execute = AsyncMock(return_value=mock_result)

        from app.services.dictionary_service import sync_ik_dictionaries

        with patch("app.services.dictionary_service.IK_DICT_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
            result = await sync_ik_dictionaries(mock_db)

        assert result["terms"] == 1
        assert result["stop_words"] == 1

    @pytest.mark.asyncio
    async def test_sync_ik_dictionaries_empty(self):
        """Empty dictionary list produces empty files."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        from app.services.dictionary_service import sync_ik_dictionaries

        with patch("app.services.dictionary_service.IK_DICT_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock()
            mock_dir.__truediv__ = MagicMock(return_value=MagicMock())
            result = await sync_ik_dictionaries(mock_db)

        assert result["terms"] == 0
        assert result["stop_words"] == 0

    @pytest.mark.asyncio
    async def test_sync_handles_write_failure(self):
        """Sync handles file write failures gracefully."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        from app.services.dictionary_service import sync_ik_dictionaries

        with patch("app.services.dictionary_service.IK_DICT_DIR") as mock_dir:
            mock_dir.mkdir = MagicMock(side_effect=OSError("Permission denied"))
            # Should not raise, just log warning
            result = await sync_ik_dictionaries(mock_db)

        assert result["terms"] == 0
        assert result["stop_words"] == 0
