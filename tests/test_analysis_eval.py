"""analysis/belief.py（熵/KL/WoE）与 eval/metrics.py、eval/geocode.py 离线部分。"""
import math

import pytest

from geobayes.analysis.belief import entropy_bits, kl_bits, weight_of_evidence_bits
from geobayes.eval.geocode import canonicalize_country
from geobayes.eval.metrics import haversine_km, threshold_accuracy


# ---------- belief ----------

def test_entropy_uniform():
    p = {c: 0.25 for c in "ABCD"}
    assert entropy_bits(p) == pytest.approx(2.0)


def test_entropy_delta_zero():
    assert entropy_bits({"A": 1.0, "B": 0.0}) == pytest.approx(0.0)


def test_kl_self_zero():
    p = {"A": 0.6, "B": 0.4}
    assert kl_bits(p, p) == pytest.approx(0.0)


def test_kl_known_value():
    p = {"A": 0.75, "B": 0.25}
    q = {"A": 0.5, "B": 0.5}
    expected = 0.75 * math.log2(0.75 / 0.5) + 0.25 * math.log2(0.25 / 0.5)
    assert kl_bits(p, q) == pytest.approx(expected)


def test_woe_is_log2_w():
    w = {"UK": 1.87, "US": 0.54}
    woe = weight_of_evidence_bits(w)
    assert woe["UK"] == pytest.approx(math.log2(1.87))
    assert woe["US"] == pytest.approx(math.log2(0.54))


# ---------- metrics ----------

def test_haversine_london_paris():
    d = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert d == pytest.approx(344, abs=5)


def test_haversine_zero():
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)


def test_threshold_accuracy():
    dists = [0.5, 30.0, 100.0, 900.0]
    acc = threshold_accuracy(dists, thresholds=(1, 25, 200, 750, 2500))
    assert acc[1] == pytest.approx(0.25)     # 只有 0.5km 命中
    assert acc[25] == pytest.approx(0.25)
    assert acc[200] == pytest.approx(0.75)
    assert acc[750] == pytest.approx(0.75)
    assert acc[2500] == pytest.approx(1.0)


def test_threshold_accuracy_counts_none_as_miss_in_denominator():
    # 无法地理编码的答案（None）= 未命中，但仍计入分母（公平口径，审计修复）
    dists = [0.5, None, None, 900.0]
    acc = threshold_accuracy(dists, thresholds=(1, 25, 750, 2500))
    assert acc[1] == pytest.approx(0.25)     # 0.5km 命中 / 4 张
    assert acc[750] == pytest.approx(0.25)
    assert acc[2500] == pytest.approx(0.5)   # 0.5 与 900 命中，两个 None 未命中 / 4


# ---------- Top-K recall / 支撑覆盖率（Table 3 口径） ----------

def test_top_k_recall():
    from geobayes.eval.metrics import top_k_recall
    # 每条：候选按先验概率降序 + GT 国家；规范化后比较（UK 别名、Scotland 并入）
    records = [
        (["United Kingdom", "Ireland", "France"], "Scotland"),   # Top-1 命中（经规范化）
        (["France", "USA", "Italy"], "United States"),           # Top-2 命中
        (["Japan", "China", "Korea"], "Brazil"),                 # 未命中
    ]
    r = top_k_recall(records, ks=(1, 3, 5))
    assert r[1] == pytest.approx(1 / 3)
    assert r[3] == pytest.approx(2 / 3)
    assert r[5] == pytest.approx(2 / 3)  # 候选不足 5 时按实际支撑算


def test_top_k_recall_orders_by_probability_not_input_order():
    from geobayes.eval.metrics import top_k_recall_from_prior
    # 直接吃 {location: prob} 字典，按概率排序后计 recall
    prior = {"France": 0.2, "United Kingdom": 0.5, "Ireland": 0.3}
    r = top_k_recall_from_prior([(prior, "UK")], ks=(1,))
    assert r[1] == pytest.approx(1.0)


# ---------- geocode（离线规范化） ----------

def test_canonicalize_aliases():
    assert canonicalize_country("UK") == "United Kingdom"
    assert canonicalize_country("united states") == "United States"
    assert canonicalize_country("USA") == "United States"
    # 论文自己把 Scotland 当国家 → recall 计算时归入 United Kingdom
    assert canonicalize_country("Scotland") == "United Kingdom"
    assert canonicalize_country("England") == "United Kingdom"
    assert canonicalize_country("Türkiye") == "Turkey"  # Nominatim 官方名 → 模型常用名


def test_canonicalize_passthrough_and_whitespace():
    assert canonicalize_country("  France ") == "France"
    assert canonicalize_country("Atlantis") == "Atlantis"  # 未知名词原样保留
