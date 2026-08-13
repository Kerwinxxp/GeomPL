"""【实验性 · 可整体删除】三种"从 VLM 引出位置信念"的方式,用于诊断现有打分的伪影。

背景:现有 geobayes/mllm/client.score_candidates 让模型对每个候选**独立**打 [0,1] 分再归一化。
实测问题(London 样本,遮掉全部 3 条线索):
  - 真值 City of Westminster 0.90 → 0.90(纹丝不动)
  - 干扰项 Paris/Vienna/Budapest 全部 −0.10(灰块 → 图像残损 → 模型砍掉弱选项)
  - 分数总和 16.30 → 11.20 ⇒ 归一化后真值反而升高 0.055 → 0.080
  - 所有变化都是 ±0.10 的整数倍 ⇒ 量化在 0.1 网格,细微信念变化不可见

三种方法:
  A independent  = 现有方案(基线,原样复刻,便于对照)
  B allocate     = 强制把 100 分配到候选上、总和必须 100 → 模型端自己归一化,
                   "没排除"无法混入质量
  C logprob_yesno= 对每个候选问 Yes/No,从 token logprob 取 P(Yes) → 连续、无量化

本模块自带 OpenAI 调用(不复用 client 的 transport,因为要拿 logprobs)。
"""
import base64
import io
import json
import math
import os

_CLIENT = None


def _openai():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        _CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),
                         base_url="https://api.openai.com/v1")
    return _CLIENT


def _img_url(image):
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _msg(text, image):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _img_url(image)}},
        {"type": "text", "text": text},
    ]}]


def _norm(d, smoothing=0.02):
    """归一化 + 轻度平滑(与现有 client 一致,保证可比)。"""
    n = len(d)
    tot = sum(d.values())
    base = {k: (v / tot if tot > 0 else 1.0 / n) for k, v in d.items()}
    out = {k: (1 - smoothing) * v + smoothing / n for k, v in base.items()}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


# ---------- A: 现有方案(基线) ----------
# 必须用**线上管线的原 prompt**,否则就是在跟稻草人比。
# 注意它明确要求 "a clearly-wrong one ~0.05" —— 这正是"没排除"地板的来源。
from geobayes.mllm.prompts import SCORE_CANDIDATES as A_PROMPT


def score_independent(image, labels, model="gpt-4o"):
    r = _openai().chat.completions.create(
        model=model, temperature=0.0,
        messages=_msg(A_PROMPT.format(candidates=json.dumps(labels, ensure_ascii=False)), image))
    txt = r.choices[0].message.content
    s = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("scores", {})
    raw = {l: min(1.0, max(0.0, float(s.get(l, 0.0) or 0.0))) for l in labels}
    return {"prior": _norm(raw), "raw": raw}


# ---------- B: 强制分配 100 分 ----------

B_PROMPT = """You are a geolocation expert. Look at this image.
Distribute EXACTLY 100 probability points across the candidate locations below,
reflecting how likely each is to be where this photo was taken.
The points MUST sum to exactly 100. Give 0 to candidates you consider implausible.
Do NOT spread points evenly out of caution - commit to what the image actually shows.
Candidates: {candidates}
Reply with ONLY a JSON object: {{"points": {{"<label>": <number>, ...}}}}"""


def score_allocate(image, labels, model="gpt-4o"):
    r = _openai().chat.completions.create(
        model=model, temperature=0.0,
        messages=_msg(B_PROMPT.format(candidates=json.dumps(labels, ensure_ascii=False)), image))
    txt = r.choices[0].message.content
    s = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]).get("points", {})
    raw = {l: max(0.0, float(s.get(l, 0.0) or 0.0)) for l in labels}
    return {"prior": _norm(raw), "raw": raw, "points_sum": sum(raw.values())}


# ---------- C: 逐候选 Yes/No + logprob ----------

C_PROMPT = """Was this photo taken in {label}?
Consider the visual evidence carefully. Answer with exactly one word: Yes or No."""


def _p_yes(image, label, model):
    r = _openai().chat.completions.create(
        model=model, temperature=0.0, max_tokens=1,
        logprobs=True, top_logprobs=20,
        messages=_msg(C_PROMPT.format(label=label), image))
    top = r.choices[0].logprobs.content[0].top_logprobs
    ly = ln = None
    for t in top:
        tok = t.token.strip().lower()
        if tok in ("yes", "y") and ly is None:
            ly = t.logprob
        elif tok in ("no", "n") and ln is None:
            ln = t.logprob
    if ly is None and ln is None:
        return 0.5
    if ly is None:
        return 1.0 - math.exp(min(0.0, ln))
    if ln is None:
        return math.exp(min(0.0, ly))
    ey, en = math.exp(ly), math.exp(ln)          # 只在 Yes/No 两者间归一化,稳健
    return ey / (ey + en)


def score_logprob_yesno(image, labels, model="gpt-4o", progress=None):
    raw = {}
    for i, l in enumerate(labels):
        raw[l] = _p_yes(image, l, model)
        if progress:
            progress(i + 1, len(labels), l, raw[l])
    return {"prior": _norm(raw), "raw": raw}


METHODS = {"independent": score_independent,
           "allocate": score_allocate,
           "logprob": score_logprob_yesno}
