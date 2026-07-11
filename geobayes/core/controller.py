"""GeoBayes 主循环。v1（国家单层）与 v2（层级+搜索）共用一套代码，由 config 开关切换。

v1（map §2.1）：Hypothesize → (Verify → Judge → Update)* → 输出。
v2 增量（map §2.2）：
- Enhance（in-loop）：ΔP < tau_enhance 且搜索可用 → WebSearch 取外部线索作为额外证据 judge+update
- 层级转移：maxP >= tau_transition 且 level < street 且搜索可用 →
  WebSearch(key evidence) → MLLM subhypotheses → Eq.5 下层先验 + 新计划
- Backtrack：下层某步全部假设 c<=2 → 回 country 层输出
- 停止（Eq.11）：street 层 maxP>=tau 或计划耗尽；非 street 层无法转移即停

控制流不变式（v1 保持）：
- 全任务执行，不因 maxP>=tau 早停；judge 只收 (evidence_text, labels)，不得见概率
- 搜索/转移失败一律降级为"停在当前层输出"，绝不丢弃已算出的分布
"""
from ..analysis.belief import weight_of_evidence_bits
from ..mllm.imaging import crop_with_padding
from .likelihood import support_score
from .prior import compute_prior
from .state import State
from .update import bayes_step

LEVEL_SEQUENCE = ["country", "city", "street"]

DEFAULT_CONFIG = {
    "k": 5,
    "tau_p": 0.6,
    "temperature_scaling": 1.5,
    "tau_stop": 0.7,           # = tau_transition [assumption]
    "tau_transition": 0.7,
    "tau_enhance": 0.05,
    "max_replace": 2,
    "replace_requires_refutation": True,
    "max_tasks_per_level": 6,
    "bbox_padding": 0.10,
    "key_evidence_bits": 1.0,
    # 单层粒度 [assumption]：country / city / place。粒度与层级解耦——
    # 设 city/place + enable_hierarchy:false 即得同支撑的细粒度先验↔后验。
    "hypothesis_granularity": "country",
    # v2 开关
    "enable_hierarchy": False,
    "enable_enhance": False,
    "max_enhance_per_level": 3,
}


class Controller:
    def __init__(self, client, config: dict | None = None, search_client=None):
        self.client = client
        self.search = search_client
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ---------- 顶层编排 ----------

    def run(self, image) -> dict:
        if hasattr(self.client, "prepare"):
            image = self.client.prepare(image)
        hyp = self.client.hypothesize(
            image, k=self.cfg["k"], max_tasks=self.cfg["max_tasks_per_level"],
            granularity=self.cfg["hypothesis_granularity"]
        )
        level = hyp.get("level", "country")
        raw_scores = {c["location"]: c["confidence"] for c in hyp["candidates"]}
        posterior = compute_prior(raw_scores, self.cfg["tau_p"], self.cfg["temperature_scaling"])
        prior_snapshot = {"level": level, "hypotheses": dict(posterior),
                          "raw_scores": dict(raw_scores)}

        state = State(
            hypotheses=posterior,
            plan=self._take_tasks(hyp),
            context={"level": level, "scene_summary": hyp.get("scene_summary", ""),
                     "parent": None, "history": []},
        )

        ctx = {"trajectory": [], "events": [], "step": 0, "memory": state.memory,
               "levels": [], "country_snapshot": None}

        while True:
            level_prior = dict(state.hypotheses)
            outcome = self._run_level(image, state, ctx)
            ctx["levels"].append({
                "level": state.context["level"], "parent": state.context.get("parent"),
                "prior": level_prior, "posterior": dict(state.hypotheses),
            })
            if state.context["level"] == "country":
                ctx["country_snapshot"] = {"level": "country",
                                           "hypotheses": dict(state.hypotheses),
                                           "context": dict(state.context)}

            if outcome == "backtrack":
                snap = ctx["country_snapshot"]
                if snap:
                    state.hypotheses = dict(snap["hypotheses"])
                    state.context = dict(snap["context"])
                break

            nxt = self._maybe_transition(image, state, ctx)
            if nxt is None:
                break
            state = nxt

        final = dict(state.hypotheses)
        return {
            "prior": prior_snapshot,
            "trajectory": ctx["trajectory"],
            "levels": ctx["levels"],
            "final_posterior": {"level": state.context["level"], "hypotheses": final},
            "map_estimate": max(final, key=final.get),
            "events": ctx["events"],
            "state": state.to_dict(),
        }

    # ---------- 闭集模式：外部给定候选集上的先验↔后验（同支撑） ----------

    def run_closed_set(self, image, candidate_labels) -> dict:
        """候选集外部给定（真值+硬负样本）：闭集打分→先验；对同一候选集跑
        Verify→Judge→Update→后验。先验与后验天然同支撑，可直接算 KL/熵。"""
        labels = list(candidate_labels)
        if hasattr(self.client, "prepare"):
            image = self.client.prepare(image)
        scored = self.client.score_candidates(image, labels)
        posterior = dict(scored["prior"])
        prior_snapshot = {"level": "closed_set", "hypotheses": dict(posterior),
                          "raw_scores": scored.get("raw_scores", {}),
                          "candidates": labels}
        plan = self._take_tasks(self.client.plan_verification(
            image, labels, max_tasks=self.cfg["max_tasks_per_level"]))
        state = State(hypotheses=posterior, plan=plan,
                      context={"level": "closed_set", "candidates": labels, "history": []})
        ctx = {"trajectory": [], "events": [], "step": 0, "memory": state.memory,
               "levels": [], "country_snapshot": None}

        for task in state.plan:
            self._verify_task(image, task, state, ctx, source="verify")

        final = dict(state.hypotheses)
        return {
            "prior": prior_snapshot,
            "trajectory": ctx["trajectory"],
            "final_posterior": {"level": "closed_set", "hypotheses": final},
            "map_estimate": max(final, key=final.get),
            "events": ctx["events"],
            "state": state.to_dict(),
        }

    # ---------- 单层循环（含 Enhance / Replace / Backtrack 检测） ----------

    def _run_level(self, image, state, ctx) -> str:
        """返回停止原因：'stop'（正常）或 'backtrack'。"""
        events, trajectory = ctx["events"], ctx["trajectory"]
        replace_count = 0
        support_seen = False
        enhance_used = 0
        is_finer = state.context["level"] != "country"

        while True:
            if not state.plan:
                events.append({"type": "empty_plan", "level": state.context["level"]})
                return "stop"
            for task in state.plan:
                crop_ev = self._verify_task(image, task, state, ctx, source="verify")
                weights = crop_ev["weights"]
                judgments = crop_ev["judgments"]

                # Backtrack：finer 层某步全部假设 c<=2
                if is_finer and all(int(round(float(r["c"]))) <= 2 for r in judgments.values()):
                    events.append({"type": "backtrack", "level": state.context["level"],
                                   "step": ctx["step"]})
                    return "backtrack"

                if any(int(round(float(r["c"]))) >= 4 and float(r["alpha"]) > 0
                       for r in judgments.values()):
                    support_seen = True

                # Enhance：信念停滞且搜索可用 → 注入外部证据
                if (crop_ev["delta_p"] < self.cfg["tau_enhance"]
                        and self.cfg["enable_enhance"] and self.search
                        and enhance_used < self.cfg["max_enhance_per_level"]):
                    if self._enhance(image, state, task, crop_ev, ctx):
                        enhance_used += 1
                    else:
                        events.append({"type": "enhance_flag", "step": ctx["step"],
                                       "delta_p": crop_ev["delta_p"]})
                elif crop_ev["delta_p"] < self.cfg["tau_enhance"]:
                    events.append({"type": "enhance_flag", "step": ctx["step"],
                                   "delta_p": crop_ev["delta_p"]})

            if max(state.hypotheses.values()) >= self.cfg["tau_stop"]:
                return "stop"
            if replace_count >= self.cfg["max_replace"]:
                return "stop"
            if support_seen and self.cfg["replace_requires_refutation"]:
                events.append({"type": "replace_gated", "level": state.context["level"]})
                return "stop"
            replace_count += 1
            if not self._replace(state, ctx, replace_count):
                return "stop"
            support_seen = False

    # ---------- 一次 Verify→Judge→Update ----------

    def _verify_task(self, image, task, state, ctx, source, override_evidence=None):
        ctx["step"] += 1
        step_idx = ctx["step"]
        events, trajectory = ctx["events"], ctx["trajectory"]

        if override_evidence is not None:
            evidence_text, geo_clues, used_bbox = override_evidence, [], None
        else:
            crop, used_bbox, fallback = crop_with_padding(
                image, task.get("bbox"), self.cfg["bbox_padding"])
            if fallback:
                events.append({"type": "bbox_fallback", "step": step_idx,
                               "task": task.get("desc", "")})
            evidence = self.client.verify(crop, task)
            evidence_text = evidence["observation"]
            geo_clues = evidence.get("geo_clues", [])

        labels = list(state.hypotheses)
        judged = self.client.judge(evidence_text, labels)
        judgments = judged["ratings"]
        if set(judgments) != set(labels):
            raise KeyError(f"judge returned {sorted(judgments)}, expected {sorted(labels)}")
        if judged.get("fallback_labels"):
            events.append({"type": "judge_fallback", "step": step_idx,
                           "labels": judged["fallback_labels"]})
        weights = {loc: support_score(r["c"], r["alpha"]) for loc, r in judgments.items()}

        prev = state.hypotheses
        state.hypotheses = bayes_step(prev, weights)
        woe = weight_of_evidence_bits(weights)
        if override_evidence is None:
            task["status"] = "Completed"
        state.context["history"].append({"step": step_idx, "task": task.get("desc", ""),
                                          "posterior": dict(state.hypotheses)})

        entry = {"step": step_idx, "level": state.context["level"], "source": source,
                 "task": dict(task, bbox_used=used_bbox), "evidence": evidence_text,
                 "geo_clues": geo_clues,
                 "judgments": {l: [judgments[l]["c"], judgments[l]["alpha"]] for l in judgments},
                 "W": weights, "posterior": dict(state.hypotheses), "log2_woe": woe}
        trajectory.append(entry)

        delta_p = max(abs(state.hypotheses[l] - prev[l]) for l in state.hypotheses)
        if max(abs(v) for v in woe.values()) >= self.cfg["key_evidence_bits"]:
            obj = geo_clues[0] if geo_clues else task.get("desc", evidence_text[:40])
            state.memory.append({"evidence": evidence_text, "object": obj,
                                 "W": weights, "step": step_idx})
            events.append({"type": "key_evidence", "step": step_idx, "object": obj})
        if max(state.hypotheses.values()) >= self.cfg["tau_stop"]:
            if not any(e["type"] == "threshold_crossed" for e in events):
                events.append({"type": "threshold_crossed", "step": step_idx})
        return {"weights": weights, "judgments": judgments, "delta_p": delta_p,
                "geo_clues": geo_clues, "evidence": evidence_text, "task": task}

    # ---------- Enhance：外部检索 → 额外证据 ----------

    def _enhance(self, image, state, task, crop_ev, ctx) -> bool:
        from ..search.client import build_level_query
        obj = (crop_ev["geo_clues"][0] if crop_ev["geo_clues"]
               else task.get("desc", state.context.get("scene_summary", "")))
        query = build_level_query(obj, state.context["level"], state.context.get("parent"))
        results = self.search.text_search(query)
        if not results:
            return False
        snippet = results[0].get("content") or results[0].get("title") or ""
        if not snippet.strip():
            return False
        evidence_text = f"[WebSearch on '{obj}'] {snippet}"
        self._verify_task(image, task, state, ctx, source="websearch",
                          override_evidence=evidence_text)
        ctx["events"].append({"type": "enhance", "step": ctx["step"], "query": query})
        return True

    # ---------- 层级转移 ----------

    def _maybe_transition(self, image, state, ctx):
        level = state.context["level"]
        if not (self.cfg["enable_hierarchy"] and self.search):
            return None
        if level == "street":
            return None
        if max(state.hypotheses.values()) < self.cfg["tau_transition"]:
            return None

        from ..search.client import build_level_query
        top = max(state.hypotheses, key=state.hypotheses.get)
        next_level = LEVEL_SEQUENCE[LEVEL_SEQUENCE.index(level) + 1]
        objects = [m["object"] for m in state.memory if m.get("object")][-3:]
        if not objects:
            objects = [state.context.get("scene_summary", "")]

        snippets = []
        for obj in objects:
            snippets += self.search.text_search(build_level_query(obj, next_level, top))
        try:
            sub = self.client.subhypotheses(top, next_level, objects, snippets)
            cands = sub.get("candidates") or []
        except (ValueError, KeyError, AssertionError):
            cands = []
        if not cands:
            return None

        new_raw = {c["location"]: c["confidence"] for c in cands}
        ctx["events"].append({"type": "transition", "from_level": level,
                              "to_level": next_level, "parent": top,
                              "objects": objects})
        new_state = State(
            hypotheses=compute_prior(new_raw, self.cfg["tau_p"], self.cfg["temperature_scaling"]),
            plan=self._take_tasks(sub),
            memory=state.memory,   # key evidence 跨层复用（Mt）
            context={"level": next_level, "parent": top,
                     "scene_summary": state.context.get("scene_summary", ""), "history": []},
        )
        return new_state

    # ---------- Replace ----------

    def _replace(self, state, ctx, replace_count) -> bool:
        try:
            rep = self.client.replace({
                "level": state.context["level"],
                "failed_hypotheses": list(state.hypotheses),
                "memory": list(state.memory),
                "scene_summary": state.context.get("scene_summary", ""),
            })
        except (ValueError, KeyError) as err:
            ctx["events"].append({"type": "replace_failed", "round": replace_count,
                                  "error": str(err)})
            return False
        new_raw = {c["location"]: c["confidence"] for c in rep["candidates"]}
        ctx["events"].append({"type": "replace", "round": replace_count})
        ctx["events"].append({"type": "support_changed",
                              "from": list(state.hypotheses), "to": list(new_raw)})
        state.hypotheses = compute_prior(new_raw, self.cfg["tau_p"],
                                         self.cfg["temperature_scaling"])
        state.plan = self._take_tasks(rep)
        return True

    def _take_tasks(self, hyp_result: dict) -> list:
        tasks = hyp_result.get("verification_tasks", [])
        return [dict(t, status="Pending") for t in tasks[: self.cfg["max_tasks_per_level"]]]
