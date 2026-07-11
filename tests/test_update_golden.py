"""Eq.7 更新测试 + Fig.3 三层五步全链黄金测试（论文已发表数字，断言 atol=2e-3）。

全链同时锁定两个实现细节：每步更新后归一化；层内不剪枝。
"""
import pytest

from geobayes.core.update import bayes_step

ATOL = 2e-3


def approx_dist(expected):
    return {k: pytest.approx(v, abs=ATOL) for k, v in expected.items()}


def test_step_normalizes():
    p = bayes_step({"A": 0.5, "B": 0.5}, {"A": 4.0, "B": 0.25})
    assert sum(p.values()) == pytest.approx(1.0)


def test_no_pruning_keeps_tiny_hypotheses():
    p = {"A": 0.98, "B": 0.02}
    for _ in range(5):
        p = bayes_step(p, {"A": 4.0, "B": 0.25})
    assert set(p) == {"A", "B"}
    assert p["B"] > 0.0


def test_unit_weights_are_identity():
    p0 = {"A": 0.7, "B": 0.3}
    assert bayes_step(p0, {"A": 1.0, "B": 1.0}) == pytest.approx(p0)


def test_mismatched_keys_raise():
    with pytest.raises(KeyError):
        bayes_step({"A": 1.0}, {"B": 2.0})


def test_zero_mass_raises():
    with pytest.raises(ValueError):
        bayes_step({"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0})


# ---------- Fig.3 黄金全链 ----------

def test_golden_country_chain():
    p = {"UK": 0.387, "US": 0.317, "SE": 0.296}

    p = bayes_step(p, {"UK": 1.87, "US": 0.54, "SE": 0.56})
    assert p == approx_dist({"UK": 0.682, "US": 0.161, "SE": 0.156})

    p = bayes_step(p, {"UK": 0.55, "US": 3.48, "SE": 0.55})
    assert p == approx_dist({"UK": 0.367, "US": 0.549, "SE": 0.084})

    p = bayes_step(p, {"UK": 0.57, "US": 1.86, "SE": 0.29})
    assert p == approx_dist({"UK": 0.167, "US": 0.813, "SE": 0.019})


def test_golden_city_chain():
    p = {"San Francisco": 0.574, "Los Angeles": 0.425}
    p = bayes_step(p, {"San Francisco": 1.74, "Los Angeles": 0.56})
    assert p == approx_dist({"San Francisco": 0.809, "Los Angeles": 0.191})


def test_golden_street_chain():
    p = {"Franklin Street": 0.517, "Hayes Street": 0.483}
    p = bayes_step(p, {"Franklin Street": 1.62, "Hayes Street": 0.66})
    assert p == approx_dist({"Franklin Street": 0.724, "Hayes Street": 0.276})
