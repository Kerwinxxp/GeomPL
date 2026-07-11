"""v1 控制流测试（mock MLLM，验证控制逻辑与模型能力解耦）。

覆盖 map §2.1：全任务执行不早停 / enhance_flag / threshold_crossed /
key_evidence / Replace 上限与支撑集变更 / judge 不得见概率 / 输出 schema。
"""
import json

import pytest
from PIL import Image

from geobayes.core.controller import Controller
from geobayes.mllm.mock_client import MockClient


def img():
    return Image.new("RGB", (200, 150), "gray")


def hyp(candidates, tasks, level="country"):
    return {
        "level": level,
        "scene_summary": "mock scene",
        "candidates": [{"location": l, "confidence": s} for l, s in candidates],
        "verification_tasks": [
            {"desc": f"task{i}", "reason": "r", "bbox": [10, 10, 90, 70]}
            for i in range(tasks)
        ],
    }


def ratings(spec):
    # spec: {label: (c, alpha)}
    return {"ratings": {l: {"c": c, "alpha": a} for l, (c, a) in spec.items()}}


STRONG_US = ratings({"UK": (1, 1.0), "US": (5, 1.0), "SE": (1, 1.0)})
NEUTRAL_3 = ratings({"UK": (3, 0.5), "US": (3, 0.5), "SE": (3, 0.5)})


def make_controller(script, **cfg_overrides):
    cfg = {"max_replace": 2, "max_tasks_per_level": 6}
    cfg.update(cfg_overrides)
    return Controller(MockClient(script), cfg)


# ---------- 输出 schema 与 happy path ----------

def test_result_schema_and_no_early_stop():
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=2),
        "verify": [{"observation": "a london bus", "geo_clues": ["bus"]}] * 2,
        "judge": [STRONG_US, STRONG_US],
    }
    result = make_controller(script).run(img())

    # schema
    for key in ("prior", "trajectory", "final_posterior", "map_estimate", "events"):
        assert key in result
    assert result["prior"]["level"] == "country"
    assert result["prior"]["raw_scores"] == {"UK": 0.6, "US": 0.2, "SE": 0.2}
    assert sum(result["prior"]["hypotheses"].values()) == pytest.approx(1.0)

    # 不早停：第 1 步后 maxP 已 ≥0.7，但 2 个任务全部执行
    assert len(result["trajectory"]) == 2
    assert any(e["type"] == "threshold_crossed" for e in result["events"])

    # 轨迹条目完整
    step = result["trajectory"][0]
    for key in ("task", "evidence", "judgments", "W", "posterior", "log2_woe"):
        assert key in step
    assert sum(step["posterior"].values()) == pytest.approx(1.0)

    # MAP 输出
    assert result["map_estimate"] == "US"
    assert max(result["final_posterior"]["hypotheses"].values()) > 0.7

    # 强证据 (W=4, log2=2 bits ≥ 1) → key evidence
    assert any(e["type"] == "key_evidence" for e in result["events"])

    # 全程 JSON 可序列化
    json.dumps(result)


def test_judge_never_sees_probabilities():
    client = MockClient({
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [STRONG_US],
    })
    Controller(client, {"max_replace": 0}).run(img())
    # judge 只收到 (evidence_text, labels)：labels 是字符串列表，无任何数值
    assert len(client.judge_calls) == 1
    evidence_text, labels = client.judge_calls[0]
    assert isinstance(evidence_text, str)
    assert set(labels) == {"UK", "US", "SE"}
    assert all(isinstance(l, str) for l in labels)


# ---------- enhance flag ----------

def test_neutral_evidence_sets_enhance_flag_each_step():
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=2),
        "verify": [{"observation": "blurry", "geo_clues": []}] * 2,
        "judge": [NEUTRAL_3, NEUTRAL_3],
    }
    result = make_controller(script, max_replace=0).run(img())
    flags = [e for e in result["events"] if e["type"] == "enhance_flag"]
    assert len(flags) == 2  # ΔP=0 < 0.05，每步都记 flag（v1 不执行 Enhance）
    # 中性证据不改变分布
    assert result["final_posterior"]["hypotheses"] == pytest.approx(
        result["prior"]["hypotheses"]
    )


# ---------- Replace ----------

def test_replace_capped_and_support_change_recorded():
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=2),
        "verify": [{"observation": "x", "geo_clues": []}] * 10,
        "judge": [NEUTRAL_3, NEUTRAL_3,
                  ratings({"France": (3, 0.5), "Italy": (3, 0.5)}),
                  ratings({"Japan": (3, 0.5), "Chile": (3, 0.5)})],
        "replace": [
            hyp([("France", 0.5), ("Italy", 0.4)], tasks=1),
            hyp([("Japan", 0.6), ("Chile", 0.1)], tasks=1),
        ],
    }
    result = make_controller(script).run(img())

    replaces = [e for e in result["events"] if e["type"] == "replace"]
    changes = [e for e in result["events"] if e["type"] == "support_changed"]
    assert len(replaces) == 2  # 恰好 max_replace=2 次后停止
    assert len(changes) == 2
    # 轨迹 = 首轮 2 任务 + 两轮 replace 各 1 任务
    assert len(result["trajectory"]) == 4
    # 最终后验在最后一个假设集上；初始先验快照保持原支撑
    assert set(result["final_posterior"]["hypotheses"]) == {"Japan", "Chile"}
    assert set(result["prior"]["hypotheses"]) == {"UK", "US", "SE"}


def test_no_replace_when_confident():
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [STRONG_US],
    }
    result = make_controller(script).run(img())
    assert not [e for e in result["events"] if e["type"] == "replace"]


# ---------- 坐标空间一致性（高危回归） ----------

class PreparingClient(MockClient):
    """模拟真实客户端：prepare 把工作图缩放到模型看到的空间，并记录 verify 收到的 crop。"""

    def __init__(self, script, prepared_size):
        super().__init__(script)
        self.prepared_size = prepared_size
        self.verify_crop_sizes = []

    def prepare(self, image):
        return image.resize(self.prepared_size)

    def verify(self, crop_image, task):
        self.verify_crop_sizes.append(crop_image.size)
        return super().verify(crop_image, task)


def test_controller_crops_in_prepared_coordinate_space():
    # 原图 400x300，模型空间 200x150；bbox=[0,0,200,150]（模型空间全幅）
    script = {
        "hypothesize": {
            "level": "country", "scene_summary": "s",
            "candidates": [{"location": "UK", "confidence": 0.6},
                           {"location": "US", "confidence": 0.2},
                           {"location": "SE", "confidence": 0.2}],
            "verification_tasks": [{"desc": "t", "reason": "r", "bbox": [0, 0, 200, 150]}],
        },
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [STRONG_US],
    }
    client = PreparingClient(script, prepared_size=(200, 150))
    Controller(client, {"max_replace": 0}).run(Image.new("RGB", (400, 300)))
    # 若控制器在原图空间裁剪，padding 后会得到 (220, 165)；正确行为是模型空间全幅 (200, 150)
    assert client.verify_crop_sizes == [(200, 150)]


# ---------- k / max_tasks 配置贯通 ----------

def test_k_and_max_tasks_reach_hypothesize():
    client = MockClient({
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [STRONG_US],
    })
    Controller(client, {"max_replace": 0, "k": 3, "max_tasks_per_level": 4}).run(img())
    assert client.hyp_kwargs["k"] == 3
    assert client.hyp_kwargs["max_tasks"] == 4


# ---------- judge 兜底事件贯通 ----------

def test_judge_fallback_surfaces_as_event():
    judged = ratings({"UK": (5, 0.8), "US": (2, 0.5), "SE": (3, 0.0)})
    judged["fallback_labels"] = ["SE"]
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [judged],
    }
    result = make_controller(script, max_replace=0).run(img())
    ev = [e for e in result["events"] if e["type"] == "judge_fallback"]
    assert len(ev) == 1 and ev[0]["labels"] == ["SE"]


# ---------- Replace 失败降级 ----------

class FailingReplaceClient(MockClient):
    def replace(self, context):
        raise ValueError("model returned no candidates")


def test_replace_failure_degrades_gracefully():
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [NEUTRAL_3],
    }
    result = Controller(FailingReplaceClient(script), {"max_replace": 2}).run(img())
    # 已算出的两个分布不得丢失
    assert result["prior"]["hypotheses"]
    assert result["final_posterior"]["hypotheses"]
    assert result["map_estimate"]
    assert any(e["type"] == "replace_failed" for e in result["events"])


# ---------- 空验证计划 ----------

def test_empty_plan_stops_without_burning_replace():
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=0),
        # 不提供 replace 脚本：若控制器误触 Replace，MockClient 会抛 AssertionError
    }
    result = make_controller(script).run(img())
    assert result["trajectory"] == []
    assert result["final_posterior"]["hypotheses"] == pytest.approx(
        result["prior"]["hypotheses"]
    )
    assert any(e["type"] == "empty_plan" for e in result["events"])
    assert not [e for e in result["events"] if e["type"] == "replace"]


# ---------- Replace 证伪门 ----------

def test_replace_gated_when_evidence_supported_a_hypothesis():
    # 领先者拿到过支持性证据 (c>=4) 但 maxP<0.7：Replace 不得触发
    #（论文 Replace 语义是"所有已验证候选无效"；有 c=5 支持时该前提不成立）
    supported_but_below_tau = ratings({"UK": (5, 0.5), "US": (2, 0.3), "SE": (2, 0.3)})
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [supported_but_below_tau],
        # 不提供 replace 脚本：若误触发，MockClient 抛 AssertionError
    }
    result = make_controller(script).run(img())
    assert max(result["final_posterior"]["hypotheses"].values()) < 0.7
    assert not [e for e in result["events"] if e["type"] == "replace"]
    assert any(e["type"] == "replace_gated" for e in result["events"])
    assert set(result["final_posterior"]["hypotheses"]) == {"UK", "US", "SE"}


def test_replace_still_fires_without_any_support():
    # 全程无支持性证据（c<=3）→ Replace 照常触发（既有行为保持）
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}] * 3,
        "judge": [NEUTRAL_3,
                  ratings({"France": (3, 0.5), "Italy": (3, 0.5)}),
                  ratings({"Japan": (3, 0.5), "Chile": (3, 0.5)})],
        "replace": [
            hyp([("France", 0.5), ("Italy", 0.4)], tasks=1),
            hyp([("Japan", 0.6), ("Chile", 0.1)], tasks=1),
        ],
    }
    result = make_controller(script).run(img())
    assert len([e for e in result["events"] if e["type"] == "replace"]) == 2


def test_support_ledger_resets_after_replace():
    # Replace 后新支撑集重新计账：新集里拿到 c>=4 → 第二次 Replace 被门挡住
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}] * 2,
        "judge": [NEUTRAL_3,
                  ratings({"France": (4, 0.4), "Italy": (2, 0.4)})],
        "replace": [hyp([("France", 0.5), ("Italy", 0.4)], tasks=1)],
    }
    result = make_controller(script).run(img())
    assert len([e for e in result["events"] if e["type"] == "replace"]) == 1
    assert any(e["type"] == "replace_gated" for e in result["events"])
    assert set(result["final_posterior"]["hypotheses"]) == {"France", "Italy"}


# ---------- 缺 desc + 退化 bbox 的健壮性 ----------

def test_task_missing_desc_and_bbox_does_not_crash():
    script = {
        "hypothesize": {
            "level": "country", "scene_summary": "s",
            "candidates": [{"location": "UK", "confidence": 0.6},
                           {"location": "US", "confidence": 0.2},
                           {"location": "SE", "confidence": 0.2}],
            "verification_tasks": [{"reason": "r"}],  # 无 desc、无 bbox
        },
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [STRONG_US],
    }
    result = make_controller(script, max_replace=0).run(img())
    assert any(e["type"] == "bbox_fallback" for e in result["events"])
    assert len(result["trajectory"]) == 1


# ---------- judge 标签守卫 ----------

def test_controller_rejects_wrong_judge_labels():
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [ratings({"France": (3, 0.5), "Italy": (3, 0.5), "Spain": (3, 0.5)})],
    }
    with pytest.raises(KeyError):
        make_controller(script, max_replace=0).run(img())


# ---------- ΔP 的 L∞ 语义锚定 ----------

def test_enhance_flag_uses_linf_norm():
    # 构造：每个假设位移 <0.05 但总位移 >0.05 → L∞ 下应打 flag（L1 实现会漏）
    small_shift = ratings({"UK": (4, 0.1375), "US": (2, 0.152), "SE": (3, 0.0)})
    script = {
        "hypothesize": hyp([("UK", 0.5), ("US", 0.4), ("SE", 0.3)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [small_shift],
    }
    result = make_controller(script, max_replace=0).run(img())
    step = result["trajectory"][0]
    deltas = [abs(step["posterior"][l] - result["prior"]["hypotheses"][l])
              for l in step["posterior"]]
    assert max(deltas) < 0.05 < sum(deltas)  # 前提成立才有区分力
    assert any(e["type"] == "enhance_flag" for e in result["events"])


def test_no_enhance_flag_on_large_shift():
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=1),
        "verify": [{"observation": "x", "geo_clues": []}],
        "judge": [STRONG_US],
    }
    result = make_controller(script, max_replace=0).run(img())
    assert not [e for e in result["events"] if e["type"] == "enhance_flag"]


# ---------- Eq.8 状态审计（Ct history / Eq.9 status 生命周期） ----------

def test_state_records_history_and_task_completion():
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.2), ("SE", 0.2)], tasks=2),
        "verify": [{"observation": "x", "geo_clues": []}] * 2,
        "judge": [STRONG_US, STRONG_US],
    }
    result = make_controller(script, max_replace=0).run(img())
    state = result["state"]
    assert len(state["context"]["history"]) == 2
    assert all(t["status"] == "Completed" for t in state["plan"])


# ---------- 单层细粒度：同支撑先验↔后验（用户研究目标形态） ----------

def test_single_level_fine_grained_same_support():
    client = MockClient({
        "hypothesize": {"level": "city", "scene_summary": "s",
            "candidates": [{"location": "San Francisco", "confidence": 0.6},
                           {"location": "Oakland", "confidence": 0.4},
                           {"location": "Berkeley", "confidence": 0.3}],
            "verification_tasks": [{"desc": "t", "reason": "r", "bbox": [10, 10, 90, 70]}]},
        "verify": [{"observation": "a cable car", "geo_clues": ["cable car"]}],
        "judge": [ratings({"San Francisco": (5, 0.8), "Oakland": (2, 0.5), "Berkeley": (2, 0.5)})],
    })
    result = Controller(client, {"hypothesis_granularity": "city",
                                 "enable_hierarchy": False, "max_replace": 0}).run(img())
    # 先验与最终后验在同一批城市候选上（同支撑，可直接算 KL/熵）
    keys = {"San Francisco", "Oakland", "Berkeley"}
    assert set(result["prior"]["hypotheses"]) == keys
    assert set(result["final_posterior"]["hypotheses"]) == keys
    assert result["prior"]["level"] == "city"
    assert result["final_posterior"]["level"] == "city"
    # 证据把 SF 推高
    assert result["final_posterior"]["hypotheses"]["San Francisco"] > \
        result["prior"]["hypotheses"]["San Francisco"]
    # granularity 透传到 hypothesize
    assert client.hyp_kwargs["granularity"] == "city"


# ---------- 闭集模式：真值+硬负样本上的先验↔后验 ----------

def test_run_closed_set_same_support_and_updates():
    labels = ["Paris, France", "Lyon, France", "Rome, Italy"]
    client = MockClient({
        "score_candidates": {"prior": {"Paris, France": 0.5, "Lyon, France": 0.3, "Rome, Italy": 0.2},
                             "raw_scores": {"Paris, France": 0.8, "Lyon, France": 0.5, "Rome, Italy": 0.3}},
        "plan_verification": {"verification_tasks": [{"desc": "sign", "reason": "r", "bbox": [10, 10, 90, 70]}]},
        "verify": [{"observation": "Italian text on the sign", "geo_clues": ["italian"]}],
        "judge": [ratings({"Paris, France": (1, 0.8), "Lyon, France": (1, 0.8), "Rome, Italy": (5, 0.9)})],
    })
    result = Controller(client, {"max_replace": 0}).run_closed_set(img(), labels)

    assert result["prior"]["level"] == "closed_set"
    assert result["prior"]["raw_scores"]["Paris, France"] == 0.8
    # 先验与后验在同一批候选上（同支撑，可直接算 KL）
    assert set(result["prior"]["hypotheses"]) == set(labels)
    assert set(result["final_posterior"]["hypotheses"]) == set(labels)
    # 意大利语线索把 Rome 推上去
    assert result["final_posterior"]["hypotheses"]["Rome, Italy"] > \
        result["prior"]["hypotheses"]["Rome, Italy"]
    assert result["map_estimate"] == "Rome, Italy"
    assert len(result["trajectory"]) == 1
    assert client.score_calls[0] == labels
    assert sum(result["final_posterior"]["hypotheses"].values()) == pytest.approx(1.0)


def test_run_closed_set_no_tasks_prior_equals_posterior():
    labels = ["A", "B", "C"]
    client = MockClient({
        "score_candidates": {"prior": {"A": 0.5, "B": 0.3, "C": 0.2}, "raw_scores": {"A": 1, "B": 1, "C": 1}},
        "plan_verification": {"verification_tasks": []},
    })
    result = Controller(client, {"max_replace": 0}).run_closed_set(img(), labels)
    assert result["trajectory"] == []
    assert result["final_posterior"]["hypotheses"] == pytest.approx(result["prior"]["hypotheses"])


# ---------- 概率不变式 ----------

def test_every_step_normalized_and_keys_stable():
    script = {
        "hypothesize": hyp([("UK", 0.6), ("US", 0.3), ("SE", 0.1)], tasks=3),
        "verify": [{"observation": "x", "geo_clues": []}] * 3,
        "judge": [ratings({"UK": (4, 0.6), "US": (2, 0.4), "SE": (3, 0.1)}),
                  ratings({"UK": (2, 0.9), "US": (5, 0.7), "SE": (1, 0.2)}),
                  ratings({"UK": (3, 0.0), "US": (4, 0.3), "SE": (2, 0.8)})],
    }
    result = make_controller(script, max_replace=0).run(img())
    keys = set(result["prior"]["hypotheses"])
    for step in result["trajectory"]:
        assert set(step["posterior"]) == keys
        assert sum(step["posterior"].values()) == pytest.approx(1.0)
