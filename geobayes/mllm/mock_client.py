"""Mock MLLM 客户端：按脚本返回固定 JSON，用于脱离模型验证控制流。

script 键：
  hypothesize: dict                    — Hypothesize 返回值
  verify:      list[dict]              — 依次弹出；耗尽后重复最后一个
  judge:       list[dict]              — 同上
  replace:     list[dict]              — 依次弹出；耗尽即报错（测试应精确编排）
记录 judge_calls = [(evidence_text, labels), ...] 供"judge 不得见概率"断言。
"""
import copy


class MockClient:
    def __init__(self, script: dict):
        self._hypothesize = script.get("hypothesize")   # 闭集模式不需要
        self._verify = list(script.get("verify", []))
        self._judge = list(script.get("judge", []))
        self._replace = list(script.get("replace", []))
        self._subhyp = list(script.get("subhypotheses", []))
        self._score = script.get("score_candidates")
        self._plan = script.get("plan_verification")
        self.judge_calls = []
        self.verify_calls = 0
        self.subhyp_calls = []
        self.score_calls = []
        self.hyp_kwargs = None

    def hypothesize(self, image, k: int = 5, max_tasks: int = 6, granularity: str = "country"):
        self.hyp_kwargs = {"k": k, "max_tasks": max_tasks, "granularity": granularity}
        return copy.deepcopy(self._hypothesize)

    def verify(self, crop_image, task):
        self.verify_calls += 1
        item = self._verify.pop(0) if len(self._verify) > 1 else self._verify[0]
        return copy.deepcopy(item)

    def judge(self, evidence_text: str, labels: list):
        self.judge_calls.append((evidence_text, list(labels)))
        item = self._judge.pop(0) if len(self._judge) > 1 else self._judge[0]
        return copy.deepcopy(item)

    def replace(self, context: dict):
        if not self._replace:
            raise AssertionError("MockClient: unexpected replace() call")
        return copy.deepcopy(self._replace.pop(0))

    def score_candidates(self, image, candidate_labels, smoothing=0.02):
        self.score_calls.append(list(candidate_labels))
        if self._score is None:
            raise AssertionError("MockClient: unexpected score_candidates() call")
        return copy.deepcopy(self._score)

    def plan_verification(self, image, candidate_labels, max_tasks=6):
        if self._plan is None:
            return {"verification_tasks": []}
        return copy.deepcopy(self._plan)

    def subhypotheses(self, parent, level, objects, snippets):
        self.subhyp_calls.append({"parent": parent, "level": level,
                                  "objects": list(objects)})
        if not self._subhyp:
            raise AssertionError("MockClient: unexpected subhypotheses() call")
        return copy.deepcopy(self._subhyp.pop(0))
