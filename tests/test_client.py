"""MLLMClient 离线单元测试：JSON 提取 / 磁盘缓存 / 解析失败重试 / judge 缺失标签兜底。

transport 可注入（Callable[[messages], str]），不发真实网络请求。
"""
import json

import pytest
from PIL import Image

from geobayes.mllm.client import MLLMClient, extract_json


def img():
    return Image.new("RGB", (64, 64), "gray")


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def make_client(tmp_path, transport):
    return MLLMClient(model="mock-model", cache_dir=str(tmp_path / "cache"),
                      transport=transport)


# ---------- JSON 提取 ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_markdown_fence_and_prose():
    text = 'Sure! Here is the result:\n```json\n{"a": [1, 2]}\n```\nHope it helps.'
    assert extract_json(text) == {"a": [1, 2]}


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_extract_json_rejects_non_object_toplevel():
    # 数组包裹单对象（常见 LLM 失误）→ 提取内部对象；纯数组/标量 → 报错（防缓存污染）
    assert extract_json('[{"a": 1}]') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")
    with pytest.raises(ValueError):
        extract_json("null")


# ---------- 缓存 ----------

def test_identical_calls_hit_disk_cache(tmp_path):
    t = FakeTransport(['{"ratings": {"UK": {"c": 3, "alpha": 0.5}}}'])
    client = make_client(tmp_path, t)
    r1 = client.judge("some evidence", ["UK"])
    r2 = client.judge("some evidence", ["UK"])
    assert r1 == r2
    assert len(t.calls) == 1  # 第二次走磁盘缓存

    # 新进程（新实例）也命中缓存
    t2 = FakeTransport(['{"ratings": {"UK": {"c": 5, "alpha": 0.9}}}'])
    client2 = make_client(tmp_path, t2)
    r3 = client2.judge("some evidence", ["UK"])
    assert r3 == r1
    assert len(t2.calls) == 0


def test_different_inputs_miss_cache(tmp_path):
    t = FakeTransport(['{"ratings": {"UK": {"c": 3, "alpha": 0.5}}}'])
    client = make_client(tmp_path, t)
    client.judge("evidence A", ["UK"])
    client.judge("evidence B", ["UK"])
    assert len(t.calls) == 2


# ---------- 解析失败重试 ----------

def test_retry_on_invalid_json_then_success(tmp_path):
    t = FakeTransport(["oops, not json",
                       '{"ratings": {"UK": {"c": 4, "alpha": 0.7}}}'])
    client = make_client(tmp_path, t)
    r = client.judge("evidence", ["UK"])
    assert r["ratings"]["UK"]["c"] == 4
    assert len(t.calls) == 2
    # 重试时附带了报错提示（消息数量比首轮多）
    assert len(t.calls[1]) > len(t.calls[0])


def test_persistent_parse_failure_raises(tmp_path):
    t = FakeTransport(["garbage"])
    client = make_client(tmp_path, t)
    with pytest.raises(ValueError):
        client.judge("evidence", ["UK"])


# ---------- judge 标签集契约 ----------

def test_judge_missing_label_filled_neutral(tmp_path):
    # 模型始终漏掉 SE → 针对性重试确实发生 → 兜底 c=3, alpha=0 (W=1)，标记 fallback_labels
    t = FakeTransport(['{"ratings": {"UK": {"c": 5, "alpha": 0.8}, "US": {"c": 2, "alpha": 0.5}}}'])
    client = make_client(tmp_path, t)
    r = client.judge("evidence", ["UK", "US", "SE"])
    assert set(r["ratings"]) == {"UK", "US", "SE"}
    assert r["ratings"]["SE"] == {"c": 3, "alpha": 0.0}
    assert r["fallback_labels"] == ["SE"]
    assert len(t.calls) == 2  # 删除重试逻辑应使本断言失败


def test_judge_retry_merges_instead_of_replacing(tmp_path):
    # 重试只补回缺失的 SE → 首轮 UK/US 打分必须保留，不得整体替换后兜底
    t = FakeTransport([
        '{"ratings": {"UK": {"c": 5, "alpha": 0.8}, "US": {"c": 2, "alpha": 0.5}}}',
        '{"ratings": {"SE": {"c": 4, "alpha": 0.6}}}',
    ])
    client = make_client(tmp_path, t)
    r = client.judge("evidence", ["UK", "US", "SE"])
    assert r["ratings"]["UK"] == {"c": 5, "alpha": 0.8}
    assert r["ratings"]["US"] == {"c": 2, "alpha": 0.5}
    assert r["ratings"]["SE"] == {"c": 4, "alpha": 0.6}
    assert "fallback_labels" not in r


def test_judge_extra_labels_trigger_retry(tmp_path):
    # 多余标签（幻觉 Norway）也要触发重试；重试后仍多余则丢弃并记 extra_labels
    good = '{"ratings": {"UK": {"c": 5, "alpha": 0.8}, "US": {"c": 2, "alpha": 0.5}}}'
    with_extra = '{"ratings": {"UK": {"c": 5, "alpha": 0.8}, "US": {"c": 2, "alpha": 0.5}, "Norway": {"c": 4, "alpha": 0.4}}}'
    t = FakeTransport([with_extra, good])
    client = make_client(tmp_path, t)
    r = client.judge("evidence", ["UK", "US"])
    assert set(r["ratings"]) == {"UK", "US"}
    assert len(t.calls) == 2


def test_judge_retry_garbage_degrades_to_fallback(tmp_path):
    # 重试对话本身返回垃圾 → 不得抛异常炸掉整个 run，按 map §3.3 兜底
    t = FakeTransport([
        '{"ratings": {"UK": {"c": 5, "alpha": 0.8}, "US": {"c": 2, "alpha": 0.5}}}',
        "garbage",
    ])
    client = make_client(tmp_path, t)
    r = client.judge("evidence", ["UK", "US", "SE"])
    assert r["ratings"]["SE"] == {"c": 3, "alpha": 0.0}
    assert r["fallback_labels"] == ["SE"]


# ---------- 缓存对图像字节敏感 ----------

def test_cache_keys_on_image_bytes(tmp_path):
    t = FakeTransport(['{"observation": "x", "geo_clues": []}'])
    client = make_client(tmp_path, t)
    task = {"desc": "d", "reason": "r"}
    img_a = Image.new("RGB", (64, 64), "red")
    img_b = Image.new("RGB", (64, 64), "blue")
    client.verify(img_a, task)
    client.verify(img_b, task)   # 同 prompt 不同图 → 未命中
    client.verify(img_a, task)   # 同图重复 → 命中
    assert len(t.calls) == 2


def test_cache_key_includes_temperature(tmp_path):
    t = FakeTransport(['{"ratings": {"UK": {"c": 3, "alpha": 0.5}}}'])
    c0 = MLLMClient(model="m", cache_dir=str(tmp_path / "c"), transport=t, temperature=0.0)
    c0.judge("e", ["UK"])
    t2 = FakeTransport(['{"ratings": {"UK": {"c": 5, "alpha": 0.9}}}'])
    c7 = MLLMClient(model="m", cache_dir=str(tmp_path / "c"), transport=t2, temperature=0.7)
    r = c7.judge("e", ["UK"])
    assert r["ratings"]["UK"]["c"] == 5  # 换温度不得复用旧缓存
    assert len(t2.calls) == 1


# ---------- hypothesize 候选清洗 ----------

def test_subhypotheses_parses_and_cleans(tmp_path):
    payload = {"level": "city",
               "candidates": [{"location": "San Francisco", "confidence": 0.6},
                              {"location": "Los Angeles", "confidence": 0.4}],
               "verification_tasks": [{"desc": "cables", "reason": "r", "bbox": [1, 2, 30, 40]}]}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.subhypotheses("United States", "city", ["trolley"],
                             [{"title": "SF", "content": "trolley in San Francisco", "url": "u"}])
    assert [c["location"] for c in r["candidates"]] == ["San Francisco", "Los Angeles"]
    # 城市层不做国家规范化
    flat = json.dumps(t.calls[0])
    assert "United States" in flat and "trolley" in flat


def test_hypothesize_city_granularity_prompt_and_no_country_canon(tmp_path):
    payload = {"level": "city", "scene_summary": "s",
               "candidates": [{"location": "San Francisco", "confidence": 0.6},
                              {"location": "Oakland", "confidence": 0.3}],
               "verification_tasks": []}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.hypothesize(img(), granularity="city")
    assert "city" in json.dumps(t.calls[0]).lower()          # prompt 要城市
    assert r["level"] == "city"
    # 城市名不被国家规范化（"United States" 别名表不该动 "San Francisco"/"Oakland"）
    assert [c["location"] for c in r["candidates"]] == ["San Francisco", "Oakland"]


def test_hypothesize_place_granularity(tmp_path):
    payload = {"level": "place", "scene_summary": "s",
               "candidates": [{"location": "Marina Piccola, Capri, Italy", "confidence": 0.7}],
               "verification_tasks": []}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.hypothesize(img(), granularity="place")
    assert r["level"] == "place"
    assert r["candidates"][0]["location"] == "Marina Piccola, Capri, Italy"


def test_hypothesize_default_granularity_is_country(tmp_path):
    payload = {"scene_summary": "s",
               "candidates": [{"location": "USA", "confidence": 0.6}],
               "verification_tasks": []}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.hypothesize(img())
    assert r["level"] == "country"
    assert r["candidates"][0]["location"] == "United States"   # 默认仍规范化国家


def test_vision_json_prepares_image_and_parses(tmp_path):
    t = FakeTransport(['{"boxes": [[1, 2, 30, 40]]}'])
    client = make_client(tmp_path, t)
    r = client.vision_json("find things at {width}x{height}", img())
    assert r["boxes"] == [[1, 2, 30, 40]]
    flat = json.dumps(t.calls[0])
    assert "data:image" in flat          # 看图
    assert "56x56" in flat               # prompt 已用预处理后尺寸(64→smart_resize 到 28 的倍数 56)


def test_score_candidates_normalizes_over_given_labels(tmp_path):
    payload = {"scores": {"Paris, France": 0.8, "Lyon, France": 0.2, "Rome, Italy": 0.0}}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    labels = ["Paris, France", "Lyon, France", "Rome, Italy"]
    r = client.score_candidates(img(), labels)
    assert set(r["prior"]) == set(labels)
    assert sum(r["prior"].values()) == pytest.approx(1.0)
    assert r["prior"]["Paris, France"] == pytest.approx(0.8, abs=0.03)   # 轻度平滑后仍≈0.8
    assert r["raw_scores"]["Paris, France"] == 0.8                        # 原始分保真
    assert all(p > 0 for p in r["prior"].values())                       # 闭集：无硬零（贝叶斯可更新）
    assert "data:image" in json.dumps(t.calls[0])


def test_score_candidates_all_zero_is_uniform(tmp_path):
    t = FakeTransport(['{"scores": {"A": 0, "B": 0}}'])
    client = make_client(tmp_path, t)
    r = client.score_candidates(img(), ["A", "B"])
    assert r["prior"]["A"] == pytest.approx(0.5)
    assert r["prior"]["B"] == pytest.approx(0.5)


def test_score_candidates_missing_label_gets_floor_not_zero(tmp_path):
    t = FakeTransport(['{"scores": {"A": 0.9}}'])   # B 缺失
    client = make_client(tmp_path, t)
    r = client.score_candidates(img(), ["A", "B"])
    assert set(r["prior"]) == {"A", "B"}
    assert r["prior"]["B"] > 0                        # 平滑保证非零
    assert r["prior"]["A"] > r["prior"]["B"]


def test_plan_verification_tasks_mention_candidates(tmp_path):
    payload = {"verification_tasks": [{"desc": "check sign", "reason": "language", "bbox": [1, 2, 30, 40]}]}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.plan_verification(img(), ["Paris, France", "Rome, Italy"])
    assert r["verification_tasks"][0]["desc"] == "check sign"
    flat = json.dumps(t.calls[0])
    assert "Paris, France" in flat and "Rome, Italy" in flat


def test_zero_shot_assembles_hierarchical_name(tmp_path):
    payload = {"country": "United States", "city": "San Francisco", "street": "Franklin Street"}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.zero_shot(img())
    assert r["name"] == "Franklin Street, San Francisco, United States"
    assert "data:image" in json.dumps(t.calls[0])   # zero-shot 看图（单次调用）


def test_zero_shot_skips_empty_levels(tmp_path):
    payload = {"country": "France", "city": "", "street": None}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.zero_shot(img())
    assert r["name"] == "France"


def test_subhypotheses_no_candidates_raises(tmp_path):
    t = FakeTransport(['{"level": "city", "candidates": []}'])
    client = make_client(tmp_path, t)
    with pytest.raises(ValueError):
        client.subhypotheses("United States", "city", ["x"], [])


def test_hypothesize_canonicalizes_and_dedupes(tmp_path):
    payload = {"level": "country", "scene_summary": "s",
               "candidates": [
                   {"location": "USA", "confidence": 0.5},
                   {"location": "United States", "confidence": 0.7},
                   {"location": "UK", "confidence": 0.6},
               ],
               "verification_tasks": []}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.hypothesize(img())
    locs = [c["location"] for c in r["candidates"]]
    assert locs == ["United States", "United Kingdom"]
    # 同一地点重复 → 取最大 si（si 是独立信念）
    us = next(c for c in r["candidates"] if c["location"] == "United States")
    assert us["confidence"] == 0.7


# ---------- hypothesize 走通编码路径（transport 假） ----------

def test_hypothesize_smoke(tmp_path):
    payload = {"level": "country", "scene_summary": "s",
               "candidates": [{"location": "France", "confidence": 0.5}],
               "verification_tasks": [{"desc": "d", "reason": "r", "bbox": [1, 2, 30, 40]}]}
    t = FakeTransport([json.dumps(payload)])
    client = make_client(tmp_path, t)
    r = client.hypothesize(img())
    assert r["candidates"][0]["location"] == "France"
    # 消息里包含图像（base64 data URL）
    flat = json.dumps(t.calls[0])
    assert "data:image" in flat
