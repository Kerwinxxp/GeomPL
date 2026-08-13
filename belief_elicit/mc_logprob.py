"""【实验性 · 可整体删除】多选题 + 字母 token logprob 的信念引出。

动机:嘴报 0-1 分会被量化在 0.1 网格、且 prompt 逼出 0.05 地板 → 对遮蔽失明。
改为:候选列成 A) B) C)…,模型只输出一个字母,读该字母 token 的 logprob → softmax
     → 闭集上的**连续**分布(不量化)。这才是"从 logprob 读位置信念"的可行形式。

为什么不用"续写地名再读 logprob":
  - 多 token 地名总 logprob 受长度惩罚,跨候选不可比;
  - gpt-4o chat 接口不暴露 echo 打分,无法给任意字符串评分。
单字母 MC 规避了这两点(每个字母都是单 token,长度一致)。

防偏置:字母位置有已知偏好 → 用 n_perm 个随机排列打乱"候选↔字母"再平均。
候选数须 ≤ 26(单字母)。top_logprobs 上限 20,缺席字母给地板值。
"""
import base64
import io
import math
import os
import random

_CLIENT = None
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


PROMPT = """You are a geolocation expert. Exactly one option below is the true location of this photo.
Weigh the visual evidence (architecture, script/signage, vegetation, vehicles, climate, terrain).
{block}
Answer with ONLY the single capital letter of the most likely option. Output just one letter, nothing else."""


def _one_pass(image, labels, letter_of, model):
    """一次 MC 调用 → {letter: logprob}(仅前 top_logprobs 个;缺席由调用方兜底)。"""
    block = "\n".join(f"{letter_of[l]}) {l}" for l in labels)
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _img_url(image)}},
        {"type": "text", "text": PROMPT.format(block=block)},
    ]}]
    r = _openai().chat.completions.create(
        model=model, temperature=0.0, max_tokens=1, logprobs=True, top_logprobs=20,
        messages=msg)
    top = r.choices[0].logprobs.content[0].top_logprobs
    lp = {}
    for t in top:
        s = t.token.strip()
        if len(s) == 1 and s.upper() in LETTERS and s.upper() not in lp:
            lp[s.upper()] = t.logprob
    return lp


def score_mc_logprob(image, labels, model="gpt-4o", n_perm=3, seed=0):
    """返回 {prior, raw_per_perm}。prior = 各排列下 softmax 概率对候选求平均。"""
    if len(labels) > 26:
        raise ValueError(f"MC 单字母最多 26 候选,给了 {len(labels)}")
    rng = random.Random(seed)
    used = LETTERS[:len(labels)]
    acc = {l: 0.0 for l in labels}
    per_perm = []
    for _ in range(n_perm):
        perm = labels[:]
        rng.shuffle(perm)
        letter_of = {perm[i]: used[i] for i in range(len(perm))}
        lp = _one_pass(image, labels, letter_of, model)
        floor = (min(lp.values()) if lp else -15.0) - 5.0     # 缺席字母:比最小可见值再低 5 nats
        vals = {L: lp.get(L, floor) for L in used}
        m = max(vals.values())
        exps = {L: math.exp(v - m) for L, v in vals.items()}
        Z = sum(exps.values())
        probs = {L: exps[L] / Z for L in used}                # softmax over 字母
        for l in labels:
            acc[l] += probs[letter_of[l]]
        per_perm.append({l: probs[letter_of[l]] for l in labels})
    prior = {l: acc[l] / n_perm for l in labels}
    z = sum(prior.values())
    prior = {l: v / z for l, v in prior.items()}
    return {"prior": prior, "raw_per_perm": per_perm}
