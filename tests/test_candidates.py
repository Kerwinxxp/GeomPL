"""闭集候选构造：真值 + K 个硬负样本（地理最近的其它 GT 点）。"""
import pytest

from geobayes.eval.candidates import build_hard_negative_set


def _pool():
    # label, lat, lon —— 距原点由近及远
    return [
        {"label": "Near1", "lat": 0.1, "lon": 0.1},
        {"label": "Near2", "lat": 0.2, "lon": 0.0},
        {"label": "Mid", "lat": 5.0, "lon": 5.0},
        {"label": "Far1", "lat": 40.0, "lon": 40.0},
        {"label": "Far2", "lat": -50.0, "lon": 80.0},
    ]


def test_picks_k_nearest_plus_true():
    r = build_hard_negative_set("TRUE", 0.0, 0.0, _pool(), k=2, seed=1)
    assert set(r["candidates"]) == {"TRUE", "Near1", "Near2"}   # 两个最近的硬负样本
    assert r["true_label"] == "TRUE"
    assert len(r["candidates"]) == 3


def test_true_always_in_candidates():
    r = build_hard_negative_set("TRUE", 0.0, 0.0, _pool(), k=4, seed=7)
    assert "TRUE" in r["candidates"]
    assert len(r["candidates"]) == 5


def test_excludes_pool_item_matching_true_label():
    pool = _pool() + [{"label": "TRUE", "lat": 0.05, "lon": 0.05}]
    r = build_hard_negative_set("TRUE", 0.0, 0.0, pool, k=2, seed=1)
    # 真值不该作为它自己的负样本重复出现
    assert r["candidates"].count("TRUE") == 1


def test_dedupes_pool_labels():
    pool = [{"label": "Dup", "lat": 0.1, "lon": 0.1},
            {"label": "Dup", "lat": 0.3, "lon": 0.3},
            {"label": "Other", "lat": 1.0, "lon": 1.0}]
    r = build_hard_negative_set("TRUE", 0.0, 0.0, pool, k=2, seed=1)
    assert r["candidates"].count("Dup") == 1


def test_deterministic_given_seed():
    a = build_hard_negative_set("TRUE", 0.0, 0.0, _pool(), k=3, seed=42)
    b = build_hard_negative_set("TRUE", 0.0, 0.0, _pool(), k=3, seed=42)
    assert a["candidates"] == b["candidates"]   # 顺序随机但可复现


def test_k_larger_than_pool_uses_all():
    r = build_hard_negative_set("TRUE", 0.0, 0.0, _pool(), k=99, seed=1)
    assert len(r["candidates"]) == 6            # TRUE + 全部 5 个
