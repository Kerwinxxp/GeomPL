"""Eq.6 性质测试：W(et|li) = exp[alpha * ln2 * (c - 3)]，不反推论文未给的 (c, alpha)。"""
import math

import pytest

from geobayes.core.likelihood import support_score


def test_neutral_rating_gives_unit_weight():
    # c=3 是中性档，任何置信度下 W=1
    for alpha in (0.0, 0.3, 1.0):
        assert support_score(3, alpha) == pytest.approx(1.0)


def test_zero_confidence_gives_unit_weight():
    for c in (1, 2, 3, 4, 5):
        assert support_score(c, 0.0) == pytest.approx(1.0)


def test_opposite_polarity_cancels():
    # 同 alpha 下 c=5 与 c=1 的 W 互为倒数（对称性，论文 Eq.6 设计意图）
    for alpha in (0.2, 0.5, 1.0):
        assert support_score(5, alpha) * support_score(1, alpha) == pytest.approx(1.0)
        assert support_score(4, alpha) * support_score(2, alpha) == pytest.approx(1.0)


def test_monotonic_in_c():
    alpha = 0.7
    ws = [support_score(c, alpha) for c in (1, 2, 3, 4, 5)]
    assert all(a < b for a, b in zip(ws, ws[1:]))


def test_bounds():
    # W ∈ [1/4, 4]：alpha=1 时 c=5 → 2^2=4，c=1 → 2^-2=0.25
    assert support_score(5, 1.0) == pytest.approx(4.0)
    assert support_score(1, 1.0) == pytest.approx(0.25)


def test_exact_formula():
    # W = exp(alpha * ln2 * (c-3))
    assert support_score(5, 0.45) == pytest.approx(math.exp(0.45 * math.log(2) * 2))
    assert support_score(2, 0.89) == pytest.approx(math.exp(0.89 * math.log(2) * (-1)))


def test_out_of_range_inputs_are_clamped():
    # judge 输出越界时钳制：c → {1..5} 整数，alpha → [0,1]
    assert support_score(9, 0.5) == pytest.approx(support_score(5, 0.5))
    assert support_score(0, 0.5) == pytest.approx(support_score(1, 0.5))
    assert support_score(4, 1.7) == pytest.approx(support_score(4, 1.0))
    assert support_score(4, -0.3) == pytest.approx(support_score(4, 0.0))
    # 非整数 c 四舍五入
    assert support_score(4.4, 0.5) == pytest.approx(support_score(4, 0.5))


def test_string_inputs_tolerated():
    # judge 偶尔把数字输出成 JSON 字符串——容错强转
    assert support_score("4", "0.5") == pytest.approx(support_score(4, 0.5))
