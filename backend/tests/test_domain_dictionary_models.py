"""``app.services.domain_dictionary`` 模块单元测试。

覆盖：
- ``Term`` / ``SynonymGroup`` / ``DomainDictionary`` 默认值
- JSONB 往返保持（to_dict / from_dict）
- 术语长度（1-30）与控制字符校验（需求 20.5）
- ``SynonymGroup`` 主术语不可出现在同义词列表，且同义词不可重复
- ``Term.weight`` 默认 1.0
"""

from __future__ import annotations

import pytest

from app.services.domain_dictionary import (
    WORD_MAX_LENGTH,
    DomainDictionary,
    SynonymGroup,
    Term,
    validate_word,
)


# ─── Term ──────────────────────────────────────────────────────────────


class TestTermDefaults:
    """Term dataclass 默认值与基础行为。"""

    def test_default_weight_is_one(self):
        term = Term(word="水泥")
        assert term.weight == 1.0

    def test_default_pos_is_none(self):
        term = Term(word="水泥")
        assert term.pos is None

    def test_explicit_weight_preserved(self):
        term = Term(word="大齿圈", weight=0.5)
        assert term.weight == 0.5

    def test_int_weight_normalized_to_float(self):
        term = Term(word="水泥", weight=2)
        assert isinstance(term.weight, float)
        assert term.weight == 2.0

    def test_word_stripped(self):
        term = Term(word="  水泥  ")
        assert term.word == "水泥"


class TestTermValidation:
    """Term 创建时的合法性校验。"""

    def test_empty_word_rejected(self):
        with pytest.raises(ValueError, match="不能为空"):
            Term(word="")

    def test_whitespace_only_word_rejected(self):
        with pytest.raises(ValueError, match="不能为空"):
            Term(word="   ")

    def test_too_long_word_rejected(self):
        with pytest.raises(ValueError, match="1-30"):
            Term(word="a" * (WORD_MAX_LENGTH + 1))

    def test_max_length_word_accepted(self):
        word = "a" * WORD_MAX_LENGTH
        term = Term(word=word)
        assert term.word == word

    def test_control_char_rejected(self):
        with pytest.raises(ValueError, match="控制字符"):
            Term(word="abc\x00def")

    def test_c1_control_char_rejected(self):
        with pytest.raises(ValueError, match="控制字符"):
            Term(word="abc\x80def")


class TestTermRoundTrip:
    """Term 的 JSONB 序列化往返。"""

    def test_to_dict_full_fields(self):
        term = Term(word="水泥", pos="n", weight=0.8)
        assert term.to_dict() == {"word": "水泥", "pos": "n", "weight": 0.8}

    def test_round_trip_preserves_value(self):
        original = Term(word="大齿圈", pos="n", weight=0.75)
        restored = Term.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_with_string(self):
        term = Term.from_dict("水泥")
        assert term.word == "水泥"
        assert term.pos is None
        assert term.weight == 1.0

    def test_from_dict_invalid_type_raises(self):
        with pytest.raises(TypeError):
            Term.from_dict(123)  # type: ignore[arg-type]

    def test_from_dict_missing_optional_fields(self):
        term = Term.from_dict({"word": "水泥"})
        assert term.pos is None
        assert term.weight == 1.0


# ─── SynonymGroup ──────────────────────────────────────────────────────


class TestSynonymGroupDefaults:
    def test_default_synonyms_empty_list(self):
        sg = SynonymGroup(primary="大齿圈")
        assert sg.synonyms == []

    def test_synonyms_preserved(self):
        sg = SynonymGroup(primary="大齿圈", synonyms=["齿圈", "主齿圈"])
        assert sg.synonyms == ["齿圈", "主齿圈"]

    def test_default_lists_are_independent(self):
        a = SynonymGroup(primary="水泥")
        b = SynonymGroup(primary="钢材")
        a.synonyms.append("洋灰")
        # 防止默认 list 被共享
        assert b.synonyms == []


class TestSynonymGroupValidation:
    def test_primary_in_synonyms_rejected(self):
        with pytest.raises(ValueError, match="不能与主术语相同"):
            SynonymGroup(primary="大齿圈", synonyms=["大齿圈", "齿圈"])

    def test_duplicate_synonyms_rejected(self):
        with pytest.raises(ValueError, match="重复"):
            SynonymGroup(primary="大齿圈", synonyms=["齿圈", "齿圈"])

    def test_empty_primary_rejected(self):
        with pytest.raises(ValueError, match="主术语"):
            SynonymGroup(primary="")

    def test_invalid_synonym_rejected(self):
        with pytest.raises(ValueError, match="同义词"):
            SynonymGroup(primary="大齿圈", synonyms=["x" * 31])

    def test_synonyms_stripped(self):
        sg = SynonymGroup(primary="大齿圈", synonyms=["  齿圈  "])
        assert sg.synonyms == ["齿圈"]


class TestSynonymGroupRoundTrip:
    def test_to_dict_basic(self):
        sg = SynonymGroup(primary="大齿圈", synonyms=["齿圈", "主齿圈"])
        assert sg.to_dict() == {
            "primary": "大齿圈",
            "synonyms": ["齿圈", "主齿圈"],
        }

    def test_round_trip_preserves_value(self):
        original = SynonymGroup(primary="大齿圈", synonyms=["齿圈", "主齿圈"])
        restored = SynonymGroup.from_dict(original.to_dict())
        assert restored == original

    def test_to_dict_returns_independent_list(self):
        sg = SynonymGroup(primary="大齿圈", synonyms=["齿圈"])
        out = sg.to_dict()
        out["synonyms"].append("污染数据")
        # 修改导出结果不应影响内部状态
        assert sg.synonyms == ["齿圈"]


# ─── DomainDictionary ──────────────────────────────────────────────────


class TestDomainDictionaryDefaults:
    def test_minimal_construction(self):
        d = DomainDictionary(id="abc-123", name="水泥行业术语")
        assert d.description == ""
        assert d.terms == []
        assert d.synonyms == []
        assert d.stop_words == []
        assert d.enabled is True

    def test_default_lists_independent(self):
        a = DomainDictionary(id="1", name="A")
        b = DomainDictionary(id="2", name="B")
        a.terms.append(Term(word="水泥"))
        a.synonyms.append(SynonymGroup(primary="大齿圈"))
        a.stop_words.append("的")
        assert b.terms == []
        assert b.synonyms == []
        assert b.stop_words == []


class TestDomainDictionaryValidation:
    def test_invalid_stop_word_rejected(self):
        with pytest.raises(ValueError, match="停用词"):
            DomainDictionary(id="1", name="A", stop_words=["a" * 31])

    def test_control_char_in_stop_word_rejected(self):
        with pytest.raises(ValueError, match="控制字符"):
            DomainDictionary(id="1", name="A", stop_words=["bad\x00word"])


class TestDomainDictionaryRoundTrip:
    def _build_sample(self) -> DomainDictionary:
        return DomainDictionary(
            id="dict-001",
            name="水泥行业术语",
            description="水泥行业专业术语词典",
            terms=[
                Term(word="水泥", pos="n", weight=1.0),
                Term(word="大齿圈", pos="n", weight=0.9),
            ],
            synonyms=[
                SynonymGroup(primary="大齿圈", synonyms=["齿圈", "主齿圈"]),
            ],
            stop_words=["的", "了"],
            enabled=True,
        )

    def test_to_dict_contains_all_fields(self):
        d = self._build_sample()
        payload = d.to_dict()
        assert payload["id"] == "dict-001"
        assert payload["name"] == "水泥行业术语"
        assert payload["description"] == "水泥行业专业术语词典"
        assert len(payload["terms"]) == 2
        assert payload["terms"][0] == {"word": "水泥", "pos": "n", "weight": 1.0}
        assert payload["synonyms"][0] == {
            "primary": "大齿圈",
            "synonyms": ["齿圈", "主齿圈"],
        }
        assert payload["stop_words"] == ["的", "了"]
        assert payload["enabled"] is True

    def test_round_trip_preserves_value(self):
        original = self._build_sample()
        restored = DomainDictionary.from_dict(original.to_dict())
        assert restored == original

    def test_jsonb_payload_has_only_jsonb_columns(self):
        d = self._build_sample()
        payload = d.to_jsonb_payload()
        assert set(payload.keys()) == {"terms", "synonyms", "stop_words"}
        assert payload["terms"][0]["word"] == "水泥"
        assert payload["synonyms"][0]["primary"] == "大齿圈"
        assert payload["stop_words"] == ["的", "了"]

    def test_from_dict_handles_string_terms(self):
        d = DomainDictionary.from_dict(
            {
                "id": "1",
                "name": "x",
                "terms": ["水泥", "钢材"],
                "synonyms": [],
                "stop_words": [],
            }
        )
        assert [t.word for t in d.terms] == ["水泥", "钢材"]
        assert all(t.weight == 1.0 for t in d.terms)

    def test_from_dict_handles_missing_optional_keys(self):
        d = DomainDictionary.from_dict({"id": "1", "name": "x"})
        assert d.description == ""
        assert d.terms == []
        assert d.synonyms == []
        assert d.stop_words == []
        assert d.enabled is True

    def test_from_dict_rejects_non_dict(self):
        with pytest.raises(TypeError):
            DomainDictionary.from_dict("not a dict")  # type: ignore[arg-type]


class TestDomainDictionaryFromORM:
    """from_orm 用于把 SQLAlchemy 模型行映射回领域对象。"""

    def test_maps_jsonb_lists_into_domain_objects(self):
        class _FakeRow:
            id = "dict-007"
            name = "水泥行业术语"
            description = "desc"
            terms = [{"word": "水泥", "pos": "n", "weight": 1.0}]
            synonyms = [{"primary": "大齿圈", "synonyms": ["齿圈"]}]
            stop_words = ["的"]
            enabled = True

        d = DomainDictionary.from_orm(_FakeRow())
        assert d.id == "dict-007"
        assert isinstance(d.terms[0], Term)
        assert d.terms[0].word == "水泥"
        assert isinstance(d.synonyms[0], SynonymGroup)
        assert d.synonyms[0].primary == "大齿圈"

    def test_handles_none_jsonb_columns(self):
        class _FakeRow:
            id = "dict-008"
            name = "空词典"
            description = None
            terms = None
            synonyms = None
            stop_words = None
            enabled = False

        d = DomainDictionary.from_orm(_FakeRow())
        assert d.description == ""
        assert d.terms == []
        assert d.synonyms == []
        assert d.stop_words == []
        assert d.enabled is False


# ─── validate_word 直接测试（公开 API） ────────────────────────────────


class TestValidateWord:
    def test_returns_true_for_valid(self):
        ok, msg = validate_word("水泥")
        assert ok is True
        assert msg == ""

    def test_returns_false_for_non_string(self):
        ok, msg = validate_word(123)
        assert ok is False

    def test_returns_false_for_too_long(self):
        ok, msg = validate_word("a" * 31)
        assert ok is False
        assert "1-30" in msg
