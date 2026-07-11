"""v2 控制流测试（mock MLLM + mock 搜索）：层级转移 / Enhance / Backtrack / 街道停止。

论文机制（map §2.2）：
- maxP >= tau_transition 且 level < street 且搜索可用 → 转移：WebSearch(key evidence) →
  MLLM subhypotheses → Eq.5 下层先验 + 新计划
- ΔP < tau_enhance 且搜索可用 → Enhance：WebSearch 取外部线索 → 作为证据 judge+update
- 下层某步全部假设 c<=2 → Backtrack 回 country 层
- street 层 maxP>=tau 或计划耗尽 → 停止
"""
import pytest
from PIL import Image

from geobayes.core.controller import Controller
from geobayes.mllm.mock_client import MockClient
from geobayes.search.client import MockSearchClient


def img():
    return Image.new("RGB", (200, 150), "gray")


def hyp(cands, tasks, level="country"):
    return {"level": level, "scene_summary": "s",
            "candidates": [{"location": l, "confidence": s} for l, s in cands],
            "verification_tasks": [{"desc": f"t{i}", "reason": "r", "bbox": [10, 10, 90, 70]}
                                   for i in range(tasks)]}


def ratings(spec):
    return {"ratings": {l: {"c": c, "alpha": a} for l, (c, a) in spec.items()}}


V2_CFG = {"enable_hierarchy": True, "enable_enhance": True, "max_replace": 0}


# ---------- 层级转移 ----------

def test_transition_country_to_city_when_confident():
    # 一步就把 US 顶到 >=0.7 → 转移到 city，city 候选来自搜索+subhypotheses
    client = MockClient({
        "hypothesize": hyp([("United States", 0.6), ("UK", 0.1), ("Sweden", 0.1)], tasks=1),
        "verify": [{"observation": "a red trolley", "geo_clues": ["trolley"]},
                   {"observation": "dense cables", "geo_clues": ["cables"]}],
        "judge": [ratings({"United States": (5, 1.0), "UK": (1, 1.0), "Sweden": (1, 1.0)}),
                  ratings({"San Francisco": (5, 1.0), "Los Angeles": (1, 1.0)})],
        "subhypotheses": [
            {"level": "city",
             "candidates": [{"location": "San Francisco", "confidence": 0.6},
                            {"location": "Los Angeles", "confidence": 0.4}],
             "verification_tasks": [{"desc": "check cables", "reason": "r", "bbox": [5, 5, 80, 80]}]},
        ],
    })
    search = MockSearchClient({"text": [[{"title": "SF trams", "content": "trolley in San Francisco", "url": "u"}]]})
    result = Controller(client, V2_CFG, search_client=search).run(img())

    assert any(e["type"] == "transition" and e["to_level"] == "city" for e in result["events"])
    assert result["final_posterior"]["level"] == "city"
    assert set(result["final_posterior"]["hypotheses"]) == {"San Francisco", "Los Angeles"}
    # 转移查询用了 key evidence 对象 + 父层 US
    assert any("United States" in q for q in search.text_calls)
    # prior 仍是 country 初始（v1 兼容）
    assert result["prior"]["level"] == "country"
    assert set(result["prior"]["hypotheses"]) == {"United States", "UK", "Sweden"}
    # 每层分布分别留档
    levels = {L["level"]: L for L in result["levels"]}
    assert set(levels) == {"country", "city"}
    assert levels["city"]["parent"] == "United States"


def test_no_transition_when_hierarchy_disabled():
    # 与 v1 一致：即使 maxP>=0.7 也不转移
    client = MockClient({
        "hypothesize": hyp([("United States", 0.6), ("UK", 0.1), ("Sweden", 0.1)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [ratings({"United States": (5, 1.0), "UK": (1, 1.0), "Sweden": (1, 1.0)})],
    })
    result = Controller(client, {"enable_hierarchy": False, "max_replace": 0},
                        search_client=MockSearchClient()).run(img())
    assert result["final_posterior"]["level"] == "country"
    assert not [e for e in result["events"] if e["type"] == "transition"]


def test_no_transition_without_search_client():
    # 开了 hierarchy 但没接搜索 → 无法生成下层假设，停在 country
    client = MockClient({
        "hypothesize": hyp([("United States", 0.6), ("UK", 0.1), ("Sweden", 0.1)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [ratings({"United States": (5, 1.0), "UK": (1, 1.0), "Sweden": (1, 1.0)})],
    })
    result = Controller(client, {"enable_hierarchy": True, "max_replace": 0}).run(img())
    assert result["final_posterior"]["level"] == "country"


def test_transition_to_street_then_stop():
    client = MockClient({
        "hypothesize": hyp([("United States", 0.6), ("UK", 0.1), ("Sweden", 0.1)], tasks=1),
        "verify": [{"observation": "trolley", "geo_clues": ["trolley"]},
                   {"observation": "hill", "geo_clues": ["hill"]},
                   {"observation": "sign", "geo_clues": ["sign"]}],
        "judge": [ratings({"United States": (5, 1.0), "UK": (1, 1.0), "Sweden": (1, 1.0)}),
                  ratings({"San Francisco": (5, 1.0), "Los Angeles": (1, 1.0)}),
                  ratings({"Franklin Street": (5, 1.0), "Hayes Street": (1, 1.0)})],
        "subhypotheses": [
            {"level": "city", "candidates": [{"location": "San Francisco", "confidence": 0.6},
                                             {"location": "Los Angeles", "confidence": 0.4}],
             "verification_tasks": [{"desc": "c", "reason": "r", "bbox": [5, 5, 80, 80]}]},
            {"level": "street", "candidates": [{"location": "Franklin Street", "confidence": 0.6},
                                               {"location": "Hayes Street", "confidence": 0.4}],
             "verification_tasks": [{"desc": "b", "reason": "r", "bbox": [5, 5, 80, 80]}]},
        ],
    })
    search = MockSearchClient({"text": [[{"title": "t", "content": "clue", "url": "u"}]]})
    result = Controller(client, V2_CFG, search_client=search).run(img())
    assert result["final_posterior"]["level"] == "street"
    assert result["map_estimate"] == "Franklin Street"
    transitions = [e for e in result["events"] if e["type"] == "transition"]
    assert [e["to_level"] for e in transitions] == ["city", "street"]
    assert {L["level"] for L in result["levels"]} == {"country", "city", "street"}


# ---------- Enhance ----------

def test_enhance_injects_external_evidence_when_belief_stalls():
    # 第 1 步中性（ΔP≈0）→ Enhance 触发搜索，取回线索作为额外证据 judge+update
    client = MockClient({
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("Sweden", 0.3)], tasks=1),
        "verify": [{"observation": "blurry", "geo_clues": ["sign"]}],
        "judge": [ratings({"UK": (3, 0.5), "US": (3, 0.5), "Sweden": (3, 0.5)}),   # 原始证据中性
                  ratings({"UK": (5, 0.9), "US": (2, 0.5), "Sweden": (2, 0.5)})],  # 增强证据判分
    })
    search = MockSearchClient({"text": [[{"title": "Royal Mail", "content": "red pillar box is British", "url": "u"}]]})
    result = Controller(client, V2_CFG, search_client=search).run(img())
    enh = [e for e in result["events"] if e["type"] == "enhance"]
    assert len(enh) >= 1
    # 增强步进入了轨迹并标注来源
    enhanced_steps = [s for s in result["trajectory"] if s.get("source") == "websearch"]
    assert len(enhanced_steps) >= 1
    # 增强证据把 UK 推上去了
    assert result["final_posterior"]["hypotheses"]["UK"] > result["prior"]["hypotheses"]["UK"]


def test_enhance_skipped_when_search_returns_nothing():
    client = MockClient({
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("Sweden", 0.3)], tasks=1),
        "verify": [{"observation": "blurry", "geo_clues": []}],
        "judge": [ratings({"UK": (3, 0.5), "US": (3, 0.5), "Sweden": (3, 0.5)})],
    })
    search = MockSearchClient({"text": [[]]})   # 空结果
    result = Controller(client, V2_CFG, search_client=search).run(img())
    # 无外部线索 → 不产生 websearch 证据步；分布不变
    assert not [s for s in result["trajectory"] if s.get("source") == "websearch"]
    assert result["final_posterior"]["hypotheses"] == pytest.approx(result["prior"]["hypotheses"])


# ---------- Backtrack ----------

def test_backtrack_when_all_refuted_at_finer_level():
    # 转移到 city 后，某步所有 city 假设 c<=2 → Backtrack 回 country
    client = MockClient({
        "hypothesize": hyp([("United States", 0.6), ("UK", 0.1), ("Sweden", 0.1)], tasks=1),
        "verify": [{"observation": "trolley", "geo_clues": ["trolley"]},
                   {"observation": "contradicts", "geo_clues": []}],
        "judge": [ratings({"United States": (5, 1.0), "UK": (1, 1.0), "Sweden": (1, 1.0)}),
                  ratings({"San Francisco": (1, 0.9), "Los Angeles": (2, 0.9)})],  # 全反对
        "subhypotheses": [
            {"level": "city", "candidates": [{"location": "San Francisco", "confidence": 0.6},
                                             {"location": "Los Angeles", "confidence": 0.4}],
             "verification_tasks": [{"desc": "c", "reason": "r", "bbox": [5, 5, 80, 80]}]},
        ],
    })
    search = MockSearchClient({"text": [[{"title": "t", "content": "c", "url": "u"}]]})
    result = Controller(client, V2_CFG, search_client=search).run(img())
    assert any(e["type"] == "backtrack" for e in result["events"])
    # 回到 country 层输出
    assert result["final_posterior"]["level"] == "country"
