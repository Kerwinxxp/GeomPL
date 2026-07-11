"""闭集全集清洗：把地理近重复候选合并为一个（概率相加），得到干净的固定全集。

用于消除 mixed-granularity/近重复标签（如 Westminster vs Greater London 1.3km）导致的
mPL 爆炸与冗余候选。
"""
import pytest

from geobayes.eval.candidates import cluster_representatives, merge_distribution


def test_cluster_merges_points_within_threshold():
    coords = {"Westminster": (51.50, -0.13), "GreaterLondon": (51.51, -0.12),
              "Paris": (48.85, 2.35)}
    rep = cluster_representatives(coords, min_dist_km=25)
    # 伦敦两点合并到同一代表；巴黎自成一类
    assert rep["Westminster"] == rep["GreaterLondon"]
    assert rep["Paris"] != rep["Westminster"]
    assert len(set(rep.values())) == 2


def test_cluster_keeps_distant_points_separate():
    coords = {"A": (0.0, 0.0), "B": (1.0, 1.0), "C": (40.0, 40.0)}  # A-B ~157km
    rep = cluster_representatives(coords, min_dist_km=25)
    assert len(set(rep.values())) == 3


def test_cluster_representative_is_deterministic():
    coords = {"Z_far": (51.50, -0.13), "A_near": (51.505, -0.125)}
    r1 = cluster_representatives(coords, min_dist_km=25)
    r2 = cluster_representatives(coords, min_dist_km=25)
    assert r1 == r2
    # 代表在簇内唯一
    assert r1["Z_far"] == r1["A_near"]


def test_merge_distribution_sums_probabilities():
    dist = {"Westminster": 0.3, "GreaterLondon": 0.2, "Paris": 0.5}
    rep = {"Westminster": "London", "GreaterLondon": "London", "Paris": "Paris"}
    merged = merge_distribution(dist, rep)
    assert merged == pytest.approx({"London": 0.5, "Paris": 0.5})


def test_merge_distribution_preserves_mass():
    dist = {"A": 0.4, "B": 0.35, "C": 0.25}
    rep = {"A": "R", "B": "R", "C": "C"}
    merged = merge_distribution(dist, rep)
    assert sum(merged.values()) == pytest.approx(1.0)
    assert set(merged) == {"R", "C"}
