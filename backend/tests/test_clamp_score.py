"""SearchService._clamp_score 静态方法单元测试。

被测方法：SearchService._clamp_score(score: float) -> float

功能：将任意输入夹紧到 [0.0, 1.0] 区间，并保留 4 位小数。
非法输入（无法转 float、NaN）兜底为 0.0。

测试策略：
- 正常值（0~1之间）：原样返回，保留4位小数
- 边界值：刚好等于 0.0 或 1.0
- 越界值：负数夹到 0.0，大于1夹到 1.0
- 四舍五入：输入精度超过4位小数时正确舍入
- 非法输入：None、字符串、NaN、列表等 → 返回 0.0
"""

from __future__ import annotations

import math

import pytest

from app.services.search_service import SearchService


# ═══════════════════════════════════════════════════════════════
# 一、正常值测试（Happy Path）
# ═══════════════════════════════════════════════════════════════


class TestClampNormalValues:
    """输入在 [0.0, 1.0] 区间内时的行为。"""

    def test_zero_point_five(self) -> None:
        """0.5 是正常值，应该原样返回（保留4位小数后还是 0.5）。"""
        result = SearchService._clamp_score(0.5)
        assert result == 0.5

    def test_zero(self) -> None:
        """刚好 0.0，边界值，返回 0.0。"""
        result = SearchService._clamp_score(0.0)
        assert result == 0.0

    def test_one(self) -> None:
        """刚好 1.0，边界值，返回 1.0。"""
        result = SearchService._clamp_score(1.0)
        assert result == 1.0

    def test_small_positive(self) -> None:
        """很小的正数，正常返回。"""
        result = SearchService._clamp_score(0.001)
        assert result == 0.001


# ═══════════════════════════════════════════════════════════════
# 二、越界值测试（Out of Range）
# ═══════════════════════════════════════════════════════════════


class TestClampOutOfRange:
    """输入超出 [0.0, 1.0] 区间时的夹紧行为。"""

    def test_negative(self) -> None:
        """负数应该被夹到 0.0。"""
        result = SearchService._clamp_score(-0.5)
        assert result == 0.0

    def test_large_negative(self) -> None:
        """很大的负数同样夹到 0.0。"""
        result = SearchService._clamp_score(-999.0)
        assert result == 0.0

    def test_greater_than_one(self) -> None:
        """大于 1.0 应该被夹到 1.0。"""
        result = SearchService._clamp_score(1.5)
        assert result == 1.0

    def test_very_large(self) -> None:
        """极大的数同样夹到 1.0。"""
        result = SearchService._clamp_score(999999.0)
        assert result == 1.0


# ═══════════════════════════════════════════════════════════════
# 三、四舍五入测试（Rounding）
# ═══════════════════════════════════════════════════════════════


class TestClampRounding:
    """验证保留 4 位小数的舍入规则。"""

    def test_round_down(self) -> None:
        """第5位小数小于5，舍去。"""
        result = SearchService._clamp_score(0.12341)
        assert result == 0.1234

    def test_round_up(self) -> None:
        """第5位小数大于等于5，进位。"""
        result = SearchService._clamp_score(0.12345)
        assert result == 0.1235

    def test_round_up_boundary(self) -> None:
        """刚好 0.99995 应该进位到 1.0。"""
        result = SearchService._clamp_score(0.99995)
        assert result == 1.0


# ═══════════════════════════════════════════════════════════════
# 四、非法输入测试（Invalid Input）
# ═══════════════════════════════════════════════════════════════


class TestClampInvalidInput:
    """无法转为 float 或特殊浮点值的兜底行为。"""

    def test_none(self) -> None:
        """None 无法转 float，返回 0.0。"""
        result = SearchService._clamp_score(None)
        assert result == 0.0

    def test_string(self) -> None:
        """非数字字符串无法转 float，返回 0.0。"""
        result = SearchService._clamp_score("hello")
        assert result == 0.0

    def test_nan(self) -> None:
        """NaN（Not a Number）应该返回 0.0。"""
        result = SearchService._clamp_score(float("nan"))
        assert result == 0.0

    def test_list(self) -> None:
        """列表无法转 float，返回 0.0。"""
        result = SearchService._clamp_score([1, 2, 3])
        assert result == 0.0

    def test_empty_string(self) -> None:
        """空字符串无法转 float，返回 0.0。"""
        result = SearchService._clamp_score("")
        assert result == 0.0


# ═══════════════════════════════════════════════════════════════
# 五、参数化测试（pytest.mark.parametrize）
# ═══════════════════════════════════════════════════════════════
# 知识点：当很多测试的"结构完全相同，只有输入和预期输出不同"时，
# 用参数化可以大幅压缩代码量。


class TestClampParameterized:
    """用 pytest.mark.parametrize 批量测试。"""

    @pytest.mark.parametrize(
        "input_score, expected",
        [
            # 正常值
            (0.5, 0.5),
            (0.0, 0.0),
            (1.0, 1.0),
            # 越界
            (-1.0, 0.0),
            (-999.0, 0.0),
            (2.0, 1.0),
            (999999.0, 1.0),
            # 四舍五入
            (0.12341, 0.1234),
            (0.12345, 0.1235),
            (0.99995, 1.0),
        ],
    )
    def test_valid_numbers(self, input_score: float, expected: float) -> None:
        """参数化：输入是合法数字的各种情况。"""
        result = SearchService._clamp_score(input_score)
        assert result == expected

    @pytest.mark.parametrize(
        "bad_input",
        [
            None,
            "hello",
            "",
            [1, 2, 3],
            {"key": "value"},
            float("nan"),
        ],
    )
    def test_invalid_inputs_return_zero(self, bad_input) -> None:
        """参数化：所有非法输入都应返回 0.0。"""
        result = SearchService._clamp_score(bad_input)
        assert result == 0.0
