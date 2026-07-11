"""距离评测支撑（论文 Table 1 口径）：前向地理编码 + 层级地名组装。"""
import pytest

from geobayes.eval.geocode import forward_geocode
from geobayes.eval.scoring import assemble_name, hierarchical_name


# ---------- 前向地理编码（可注入 transport） ----------

def test_forward_geocode_uses_transport_and_caches():
    calls = []
    def tr(name):
        calls.append(name)
        return [48.85, 2.35]
    cache = {}
    assert forward_geocode("Paris, France", transport=tr, cache=cache) == [48.85, 2.35]
    assert forward_geocode("Paris, France", transport=tr, cache=cache) == [48.85, 2.35]
    assert len(calls) == 1  # 第二次命中缓存


def test_forward_geocode_none_on_failure():
    def tr(name):
        raise RuntimeError("network down")
    assert forward_geocode("Somewhere", transport=tr, cache={}) is None


def test_forward_geocode_none_when_no_result():
    assert forward_geocode("xyzzy", transport=lambda n: None, cache={}) is None


def test_forward_geocode_caches_none_to_avoid_refetch():
    calls = []
    def tr(name):
        calls.append(name)
        return None
    cache = {}
    forward_geocode("nowhere", transport=tr, cache=cache)
    forward_geocode("nowhere", transport=tr, cache=cache)
    assert len(calls) == 1  # 失败也缓存，避免重复请求


# ---------- 名称组装（最细在前，跳过空层） ----------

def test_assemble_name_full_hierarchy():
    assert assemble_name(["Franklin Street", "San Francisco", "United States"]) \
        == "Franklin Street, San Francisco, United States"


def test_assemble_name_skips_empty_and_none():
    assert assemble_name(["", "San Francisco", None]) == "San Francisco"
    assert assemble_name([None, None, "France"]) == "France"


def test_assemble_name_empty_returns_empty_string():
    assert assemble_name([None, "", "  "]) == ""


# ---------- 从 GeoBayes 结果重建层级地名 ----------

def test_hierarchical_name_multi_level_from_argmax():
    r = {"levels": [
        {"level": "country", "posterior": {"United States": 0.8, "UK": 0.2}},
        {"level": "city", "posterior": {"San Francisco": 0.7, "Los Angeles": 0.3}},
        {"level": "street", "posterior": {"Franklin Street": 0.6, "Hayes Street": 0.4}}],
        "final_posterior": {"level": "street", "hypotheses": {"Franklin Street": 0.6}}}
    assert hierarchical_name(r) == "Franklin Street, San Francisco, United States"


def test_hierarchical_name_country_only():
    r = {"levels": [{"level": "country", "posterior": {"France": 0.6, "Italy": 0.4}}],
         "final_posterior": {"level": "country", "hypotheses": {"France": 0.6, "Italy": 0.4}}}
    assert hierarchical_name(r) == "France"


def test_hierarchical_name_falls_back_to_final_when_no_levels():
    r = {"final_posterior": {"level": "city", "hypotheses": {"Tokyo": 0.9, "Osaka": 0.1}}}
    assert hierarchical_name(r) == "Tokyo"
