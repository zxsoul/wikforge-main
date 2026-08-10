"""术语格式校验的 Hypothesis 属性测试 (任务 13.6)。

针对 ``app.services.dictionary_service.validate_term`` 与
``app.services.domain_dictionary.validate_word`` 两个公开校验函数，验证以下
不变式（需求 20.5）：

属性 P1 - 一致性：两个实现在同一字符串输入下结果等价（``is_valid`` 相同；
  错误消息可不同，但都为非空字符串）。
属性 P2 - 接受合法术语：满足 (strip 后长度 ∈ [1, 30] 且不含 C0/C1 控制字符)
  的输入必须 ``is_valid is True`` 且错误消息为空。
属性 P3 - 拒绝过短/过长：strip 后长度为 0 或 > 30 的输入必须 ``is_valid is
  False`` 且错误消息非空。
属性 P4 - 拒绝控制字符：包含至少一个 C0/C1 控制字符（剔除 \\t/\\n/\\r）的
  非空输入必须 ``is_valid is False`` 且错误消息非空。

这些属性是该任务的"硬性合同"。

Validates: Requirements 20.5
"""

from __future__ import annotations

import re

import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from app.services.dictionary_service import validate_term
from app.services.domain_dictionary import (
    WORD_MAX_LENGTH,
    WORD_MIN_LENGTH,
    validate_word,
)

# ─── 字符策略 ──────────────────────────────────────────────────────────

# 控制字符（C0 排除 \t \n \r；DEL/C1 0x7F-0x9F）。与 production 正则一致。
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# 合法字符：从 BMP 中挑可见字符；显式排除控制字符与代理对（Cs）。
_NON_CONTROL_CHAR = st.characters(
    blacklist_categories=("Cs", "Cc"),
    # Cc 已经覆盖 C0/C1，再额外保险地排除 0x7F-0x9F 区间不必要——Cc 已含。
    min_codepoint=0x20,
    max_codepoint=0xFFFF,
)

# 控制字符策略：直接采样 0x00-0x08、0x0B、0x0C、0x0E-0x1F、0x7F-0x9F。
_CONTROL_CHAR = st.sampled_from(
    [chr(c) for c in range(0x00, 0x09)]
    + [chr(0x0B), chr(0x0C)]
    + [chr(c) for c in range(0x0E, 0x20)]
    + [chr(c) for c in range(0x7F, 0xA0)]
)


@st.composite
def valid_terms(draw: st.DrawFn) -> str:
    """生成 strip 后长度 ∈ [1, 30] 且不含控制字符的合法术语。

    通过限制内部字符集为可见字符并避免首尾空白，绕开 strip 引入的复杂性。
    """
    size = draw(st.integers(min_value=WORD_MIN_LENGTH, max_value=WORD_MAX_LENGTH))
    text = draw(
        st.text(
            alphabet=_NON_CONTROL_CHAR,
            min_size=size,
            max_size=size,
        )
    )
    # 业务上 strip 后再判长度；为了保证 strip 后仍 ∈ [1, 30]，要求首尾非空白。
    if not text.strip() or text != text.strip():
        # 让 hypothesis 尝试其它例子
        from hypothesis import assume

        assume(False)
    return text


@st.composite
def invalid_length_terms(draw: st.DrawFn) -> str:
    """生成 strip 后长度为 0 或 > 30 的非法术语（不含控制字符）。"""
    choice = draw(st.sampled_from(["empty", "too_long"]))
    if choice == "empty":
        # 0 长度或全空白
        return draw(
            st.one_of(
                st.just(""),
                st.text(alphabet=" \t", min_size=1, max_size=10),
            )
        )
    # too_long: strip 后长度严格 > 30
    size = draw(st.integers(min_value=WORD_MAX_LENGTH + 1, max_value=WORD_MAX_LENGTH + 50))
    text = draw(
        st.text(
            alphabet=_NON_CONTROL_CHAR,
            min_size=size,
            max_size=size,
        )
    )
    from hypothesis import assume

    # 保证 strip 后还是过长
    assume(len(text.strip()) > WORD_MAX_LENGTH)
    return text


@st.composite
def control_char_terms(draw: st.DrawFn) -> str:
    """生成至少含一个控制字符、strip 后非空、长度 ≤ 30 的输入。"""
    # 头尾各取一段普通字符，中间插入控制字符；保证 strip 后仍含控制字符。
    prefix_size = draw(st.integers(min_value=1, max_value=5))
    suffix_size = draw(st.integers(min_value=0, max_value=5))
    prefix = draw(
        st.text(alphabet=_NON_CONTROL_CHAR, min_size=prefix_size, max_size=prefix_size)
    )
    suffix = draw(
        st.text(alphabet=_NON_CONTROL_CHAR, min_size=suffix_size, max_size=suffix_size)
    )
    ctrl = draw(_CONTROL_CHAR)
    word = prefix + ctrl + suffix
    from hypothesis import assume

    # 长度 ≤ 30 才能真正命中"控制字符拒绝"路径，否则会先被长度规则拒。
    assume(len(word.strip()) <= WORD_MAX_LENGTH)
    # strip 后必须仍含控制字符（避免控制字符意外位于首尾被某些实现剥离）。
    assume(_CONTROL_CHAR_RE.search(word.strip()) is not None)
    return word


# ─── 属性测试 ──────────────────────────────────────────────────────────


SETTINGS = hyp_settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


class TestTermValidationProperties:
    """术语校验函数的 Hypothesis 属性测试。

    Validates: Requirements 20.5
    """

    @SETTINGS
    @given(word=st.text())
    def test_property_consistency_between_implementations(self, word: str) -> None:
        """属性 P1：``validate_term`` 与 ``validate_word`` 对 str 输入结果一致。

        Validates: Requirements 20.5
        """
        ok_term, msg_term = validate_term(word)
        ok_word, msg_word = validate_word(word)
        assert ok_term is ok_word, (
            f"两个校验函数对同一输入给出不一致结果：term={ok_term} word={ok_word} "
            f"input={word!r}"
        )
        # 两端都拒绝时各自必须给出非空错误消息
        if not ok_term:
            assert msg_term, "validate_term 拒绝时应返回非空错误消息"
            assert msg_word, "validate_word 拒绝时应返回非空错误消息"
        else:
            assert msg_term == "" and msg_word == "", "通过校验时错误消息必须为空"

    @SETTINGS
    @given(word=valid_terms())
    def test_property_accepts_valid_terms(self, word: str) -> None:
        """属性 P2：满足合法条件的输入必须通过两个校验函数。

        Validates: Requirements 20.5
        """
        # 前置不变量
        stripped = word.strip()
        assert WORD_MIN_LENGTH <= len(stripped) <= WORD_MAX_LENGTH
        assert _CONTROL_CHAR_RE.search(stripped) is None

        ok_term, msg_term = validate_term(word)
        ok_word, msg_word = validate_word(word)
        assert ok_term is True, f"validate_term 拒绝合法术语 {word!r}: {msg_term}"
        assert ok_word is True, f"validate_word 拒绝合法术语 {word!r}: {msg_word}"
        assert msg_term == ""
        assert msg_word == ""

    @SETTINGS
    @given(word=invalid_length_terms())
    def test_property_rejects_invalid_length(self, word: str) -> None:
        """属性 P3：strip 后长度为 0 或 > 30 的输入必须被拒。

        Validates: Requirements 20.5
        """
        stripped = word.strip()
        assert len(stripped) == 0 or len(stripped) > WORD_MAX_LENGTH

        ok_term, msg_term = validate_term(word)
        ok_word, msg_word = validate_word(word)
        assert ok_term is False, f"validate_term 应拒绝长度非法的 {word!r}"
        assert ok_word is False, f"validate_word 应拒绝长度非法的 {word!r}"
        assert msg_term and msg_word

    @SETTINGS
    @given(word=control_char_terms())
    def test_property_rejects_control_chars(self, word: str) -> None:
        """属性 P4：strip 后仍含 C0/C1 控制字符的输入必须被拒。

        Validates: Requirements 20.5
        """
        # 前置：确实含控制字符且不会因长度先被拒
        assert _CONTROL_CHAR_RE.search(word.strip()) is not None
        assert WORD_MIN_LENGTH <= len(word.strip()) <= WORD_MAX_LENGTH

        ok_term, msg_term = validate_term(word)
        ok_word, msg_word = validate_word(word)
        assert ok_term is False, f"validate_term 应拒绝含控制字符的 {word!r}"
        assert ok_word is False, f"validate_word 应拒绝含控制字符的 {word!r}"
        assert "控制字符" in msg_term
        assert "控制字符" in msg_word

    @SETTINGS
    @given(
        word=st.text(
            alphabet=_NON_CONTROL_CHAR,
            min_size=WORD_MAX_LENGTH + 1,
            max_size=WORD_MAX_LENGTH + 100,
        )
    )
    def test_property_length_boundary_30_chars_strict(self, word: str) -> None:
        """属性 P3 加固：仅由非控制字符构成、长度 > 30 的输入必拒。

        Validates: Requirements 20.5
        """
        from hypothesis import assume

        assume(len(word.strip()) > WORD_MAX_LENGTH)
        ok_term, _ = validate_term(word)
        ok_word, _ = validate_word(word)
        assert ok_term is False
        assert ok_word is False
