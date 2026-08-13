"""【实验性 · 可整体删除】证伪"幂次缩放能修好 mPL"这个想法。零 API 调用。

推导:幂次缩放 p_i ∝ base_i^β 会让
    llr'_i = β·llr_i + C   (C 与 i 无关)
  ⇒ llr'_i − llr'_j = β·(llr_i − llr_j)
  ⇒ mPL' = β · mPL
即**只是把 mPL 整体乘以 β**(噪声和伪影一起乘),不改变任何排序、不产生新信息。

但推导忽略了两件事,必须实测确认:
  (a) 平滑是**加性**的 prior=(1−s)·base+s/n,破坏严格比例;
  (b) 25km 聚类会**求和**簇内概率,而 Σ(x^β) ≠ (Σx)^β,也破坏比例。
若实测 mPL(β)/mPL(1) ≈ β,则该想法确认无效。

base 可从已存的 prior 反解(零 API):base_i = (prior_i − s/n)/(1−s)。
用法：python -m belief_elicit.test_power_scaling
"""
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from belief_elicit.run_noise import build_geometry, mpl

COMBODIR = os.path.join(ROOT, "clue_leak", "combo2_sam3_results")
SMOOTHING = 0.02


def unsmooth(prior):
    """从存下来的 prior 反解归一化前的相对分数 base(零 API)。"""
    n = len(prior)
    base = {k: max(0.0, (v - SMOOTHING / n) / (1 - SMOOTHING)) for k, v in prior.items()}
    t = sum(base.values())
    return {k: v / t for k, v in base.items()} if t > 0 else prior


def rescale(base, beta):
    """幂次缩放 + 按原样重新平滑(与线上 client 一致)。"""
    n = len(base)
    p = {k: v ** beta for k, v in base.items()}
    t = sum(p.values())
    p = {k: (v / t if t > 0 else 1.0 / n) for k, v in p.items()}
    p = {k: (1 - SMOOTHING) * v + SMOOTHING / n for k, v in p.items()}
    z = sum(p.values())
    return {k: v / z for k, v in p.items()}


def main():
    gallery = json.load(open(os.path.join(ROOT, "data", "gallery_labels.json"), encoding="utf-8"))
    rep, clusters, dist = build_geometry(gallery)
    betas = [1.0, 2.0, 3.0, 5.0, 8.0]

    print("mPL 随 β 的变化(若 ≈ β 倍,则幂次缩放无效)\n")
    ratios = []
    for f in sorted(glob.glob(os.path.join(COMBODIR, "*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        place = r["true_label"].split(",")[0]
        post_b = unsmooth(r["posterior"])
        # 取该图 mPL 最大的那个子集(最有信号的),避免全零子集干扰
        best, best_v = None, -1
        for c in r["combos"]:
            v = mpl(c["prior"], r["posterior"], rep, clusters, dist)
            if v > best_v:
                best, best_v = c, v
        if best_v <= 1e-9:
            print(f"{place:12s} 全部子集 mPL≈0,跳过")
            continue
        pri_b = unsmooth(best["prior"])
        line = f"{place:12s} S={str(best['subset']):12s}"
        vals = []
        for b in betas:
            v = mpl(rescale(pri_b, b), rescale(post_b, b), rep, clusters, dist)
            vals.append(v)
            line += f"  β={b:.0f}:{v:.4f}"
        print(line)
        rel = [v / vals[0] for v in vals]
        print(f"{'':12s} {'':12s}  比值 mPL(β)/mPL(1) = " +
              "  ".join(f"{x:.2f}" for x in rel) + f"   (理论应为 {betas})")
        ratios.append(rel)

    if ratios:
        m = np.array(ratios).mean(axis=0)
        print("\n=== 汇总:各图平均比值 ===")
        for b, x in zip(betas, m):
            tag = "≈β,纯换单位" if abs(x - b) / b < 0.15 else ("塌缩" if x < b * 0.5 else "偏离")
            print(f"  β={b:.0f}  实测倍数={x:.2f}   理论={b:.0f}   偏差={abs(x-b)/b*100:5.1f}%  {tag}")
        lo = m[1] / betas[1]        # β=2 处是否贴合理论
        hi = m[-1] / betas[-1]      # β=8 处是否塌缩
        print(f"\n结论(随数据):")
        print(f"  低 β(=2):实测/理论 = {lo:.2f} ⇒ " +
              ("推导成立,幂次缩放只是给 mPL 换单位,不产生新信息。"
               if lo > 0.85 else "推导不成立,需重查。"))
        print(f"  高 β(=8):实测/理论 = {hi:.2f} ⇒ " +
              ("mPL 朝 0 塌缩:分布过尖 → 非头部候选被压到平滑地板 s/n → "
               "遮蔽前后同值 → llr=0。这与 allocate(≈β→∞ 极限)的失败是同一机制。"
               if hi < 0.5 else "未见塌缩。"))
        print("  两个区间都无用 ⇒ 幂次缩放证伪。")


if __name__ == "__main__":
    main()
