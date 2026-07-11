"""Eq.5 性质测试：P0(li) = exp(min(si, tau_p)/T) / sum_j exp(min(sj, tau_p)/T)。"""
import math

import pytest

from geobayes.core.prior import compute_prior

TAU_P = 0.6
T = 1.5


def test_normalized():
    p = compute_prior({"A": 0.9, "B": 0.4, "C": 0.1})
    assert sum(p.values()) == pytest.approx(1.0)
    assert set(p) == {"A", "B", "C"}


def test_truncation_effective():
    # 全部 si ≥ tau_p → 截断后相等 → 精确均匀
    p = compute_prior({"A": 0.8, "B": 0.9, "C": 1.0})
    for v in p.values():
        assert v == pytest.approx(1 / 3)


def test_temperature_and_ratio_formula():
    # 未截断区间内，两候选概率比 = exp((s1-s2)/T)
    p = compute_prior({"A": 0.6, "B": 0.2})
    assert p["A"] / p["B"] == pytest.approx(math.exp((0.6 - 0.2) / T))


def test_ratio_cap():
    # 任意两候选比值 ≤ e^(tau_p/T) = e^0.4 ≈ 1.4918（先验被刻意压平，论文设计）
    cap = math.exp(TAU_P / T)
    p = compute_prior({"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.31, "E": 0.77})
    vals = sorted(p.values())
    assert vals[-1] / vals[0] <= cap + 1e-9


def test_fig3_style_prior_within_cap():
    # Fig.3 先验 {0.387, 0.317, 0.296} 的 max/min = 1.31 < 1.49；
    # 用 si = {0.6, 0.28, 0.2} 应重现同数量级的平坦分布
    p = compute_prior({"UK": 0.6, "US": 0.28, "SE": 0.2})
    assert max(p.values()) / min(p.values()) < math.exp(TAU_P / T)
    assert max(p.values()) < 0.45  # 平坦性：3 候选下最大值远低于 1


def test_raw_scores_clamped_to_unit_interval():
    # si 钳制 [0,1] 后再进 Eq.5 [assumption, map §3.1]
    assert compute_prior({"A": -0.5, "B": 0.0}) == pytest.approx(
        compute_prior({"A": 0.0, "B": 0.0})
    )
    # 上界钳制必须用 tau_p > 1 才能与截断区分开（tau_p=0.6 时 1.7 与 1.0 同被截断，测不出钳制）
    assert compute_prior({"A": 1.7, "B": 0.3}, tau_p=2.0) == pytest.approx(
        compute_prior({"A": 1.0, "B": 0.3}, tau_p=2.0)
    )


def test_single_candidate_degenerate():
    assert compute_prior({"A": 0.4}) == {"A": pytest.approx(1.0)}


def test_empty_raises():
    with pytest.raises(ValueError):
        compute_prior({})
