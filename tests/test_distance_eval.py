"""距离阈值评测（论文 Table 1 口径）：层级地名重建 + 前向地理编码 + 距离。"""
import pytest

from geobayes.eval.geocode import forward_geocode, hierarchical_name
from geobayes.eval.metrics import localization_distance_km


# ---------- 层级地名重建 ----------

def test_hierarchical_name_full_path():
    result = {"levels": [
        {"level": "country", "posterior": {"United States": 0.8, "UK": 0.2}},
        {"level": "city", "posterior": {"San Francisco": 0.7, "Los Angeles": 0.3}},
        {"level": "street", "posterior": {"Franklin Street": 0.6, "Hayes Street": 0.4}},
    ], "final_posterior": {"level": "street", "hypotheses": {"Franklin Street": 0.6}}}
    assert hierarchical_name(result) == "Franklin Street, San Francisco, United States"


def test_hierarchical_name_country_only():
    result = {"levels": [{"level": "country", "posterior": {"France": 0.6, "Italy": 0.4}}],
              "final_posterior": {"level": "country", "hypotheses": {"France": 0.6}}}
    assert hierarchical_name(result) == "France"


def test_hierarchical_name_no_levels_uses_final():
    # 纯国家层结果无 levels 字段时退回 final_posterior argmax
    result = {"final_posterior": {"level": "country", "hypotheses": {"Japan": 0.7, "China": 0.3}}}
    assert hierarchical_name(result) == "Japan"


def test_hierarchical_name_zero_shot_dict():
    # zero-shot 直接给三段
    result = {"zero_shot": {"country": "Spain", "city": "Barcelona", "street": None}}
    assert hierarchical_name(result) == "Barcelona, Spain"


# ---------- 前向地理编码（可注入 transport） ----------

def test_forward_geocode_uses_transport_and_caches():
    calls = []
    def transport(name):
        calls.append(name)
        return [48.8566, 2.3522]
    cache = {}
    a = forward_geocode("Paris, France", cache=cache, transport=transport)
    b = forward_geocode("Paris, France", cache=cache, transport=transport)
    assert a == [48.8566, 2.3522]
    assert b == a
    assert calls == ["Paris, France"]   # 第二次命中缓存


def test_forward_geocode_none_on_failure():
    def transport(name):
        raise RuntimeError("nominatim down")
    assert forward_geocode("Nowhere", cache={}, transport=transport) is None


def test_forward_geocode_empty_result_none():
    assert forward_geocode("Atlantis", cache={}, transport=lambda n: None) is None


# ---------- 距离评测 ----------

def test_localization_distance_km():
    result = {"final_posterior": {"level": "country", "hypotheses": {"France": 1.0}}}
    # 假 geocoder：France → 巴黎附近
    geo = lambda name: [48.85, 2.35]
    d = localization_distance_km(result, 48.85, 2.35, geocoder=geo)
    assert d == pytest.approx(0.0, abs=1.0)


def test_localization_distance_none_when_ungeocodable():
    result = {"final_posterior": {"level": "country", "hypotheses": {"Atlantis": 1.0}}}
    assert localization_distance_km(result, 10.0, 20.0, geocoder=lambda n: None) is None
