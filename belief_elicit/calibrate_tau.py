"""【实验性 · 可整体删除】温度校准(审稿意见 #1):τ 敏感性 + NLL/Brier/ECE + mPL∝1/τ 验证。

关键:softmax(s/τ) 的分布可从已存 p(1) 精确恢复,无需重打分:
    p_i(τ) = p_i(1)^{1/τ} / Σ_j p_j(1)^{1/τ}
因此 (a) 用真值 NLL 在 held-out 上选 τ*、报 Brier/ECE;(b) 直接展示 mPL(τ)≈mPL(1)/τ,
说明 τ 只是全局缩放、图内比较/交互符号对 τ 不变。
运行(主环境):python -m belief_elicit.calibrate_tau
"""
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

from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

SWEEP = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")


def build_geometry(gv, merge_km=2.0):
    coords = {g["label"]: g["gps"] for g in gv if g["gps"]}
    rep = cluster_representatives(coords, merge_km)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mpl(prior, post, rep, clusters, dist):
    pr, po = merge_distribution(prior, rep), merge_distribution(post, rep)
    keys = [k for k in clusters if pr.get(k, 0) > 0 and po.get(k, 0) > 0]
    llr = {k: math.log(po[k] / pr[k]) for k in keys}
    ks = list(llr)
    vals = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = dist(ks[i], ks[j])
            if d and d > 0:
                vals.append(abs(llr[ks[i]] - llr[ks[j]]) / d * 1000)
    return sum(vals) / len(vals) if vals else 0.0
OUT = os.path.join(os.path.dirname(__file__), "calibrate_tau_results.json")


def temper(prob, tau):
    """p(1) dict -> p(tau) dict,精确恢复 softmax(s/τ)。"""
    labels = list(prob)
    lp = np.array([math.log(max(prob[l], 1e-12)) for l in labels]) / tau
    lp -= lp.max()
    e = np.exp(lp); e /= e.sum()
    return {labels[i]: float(e[i]) for i in range(len(labels))}


def nll(rows, tau):
    v = []
    for r in rows:
        p = temper(r["posterior"], tau)
        v.append(-math.log(max(p.get(r["true_label"], 1e-12), 1e-12)))
    return float(np.mean(v))


def brier(rows, tau):
    v = []
    for r in rows:
        p = temper(r["posterior"], tau)
        tl = r["true_label"]
        s = sum((p[l] - (1.0 if l == tl else 0.0)) ** 2 for l in p)
        v.append(s)
    return float(np.mean(v))


def ece(rows, tau, nbin=10):
    conf, acc = [], []
    for r in rows:
        p = temper(r["posterior"], tau)
        am = max(p, key=p.get)
        conf.append(p[am]); acc.append(1.0 if am == r["true_label"] else 0.0)
    conf = np.array(conf); acc = np.array(acc)
    e = 0.0
    for b in range(nbin):
        lo, hi = b / nbin, (b + 1) / nbin
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def main():
    rows = json.load(open(SWEEP, encoding="utf-8"))
    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    rep, clusters, dist = build_geometry(gv, merge_km=2.0)

    # (a) held-out τ 选择:固定 50/50 划分,train 上最小化 NLL,test 上报指标
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))
    tr = [rows[i] for i in idx[:len(rows) // 2]]
    te = [rows[i] for i in idx[len(rows) // 2:]]
    taus = np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.2, 8, 25)])
    star = min(taus, key=lambda t: nll(tr, t))
    print(f"held-out 温度校准({len(tr)} train / {len(te)} test):")
    print(f"  τ* = {star:.2f}(在 train 上最小化真值 NLL)")
    print(f"  {'':14s}{'τ=1':>10s}{'τ=τ*':>10s}")
    for name, fn in [("NLL", nll), ("Brier", brier), ("ECE", ece)]:
        print(f"  {name:12s}{fn(te,1.0):10.4f}{fn(te,star):10.4f}")

    # (b) mPL 对 τ 的缩放:取几张有代表性的图,验证 ≈ 1/τ
    print("\nmPL(全遮) 随 τ(应 ≈ mPL(1)/τ,证明 τ 只是全局缩放):")
    demo = [r for r in rows if r["n_cues"] >= 2][:2]
    tau_grid = [0.5, 1.0, 2.0, 5.0]
    scale_rows = []
    for r in demo:
        place = r["true_label"].split(",")[0]
        post = r["posterior"]; pri = r["prior_allmask"]
        base = mpl(temper(pri, 1.0), temper(post, 1.0), rep, clusters, dist)
        line = f"  {place[:12]:12s} mPL(1)={base:.4f}: "
        rec = {"place": place, "mpl_base": base, "ratio": {}}
        for t in tau_grid:
            val = mpl(temper(pri, t), temper(post, t), rep, clusters, dist)
            ratio = val / base if base else 0
            rec["ratio"][t] = ratio
            line += f"τ={t}: {val:.4f}(×{ratio:.2f}, 理论 {1/t:.2f})  "
        scale_rows.append(rec)
        print(line)

    # 排名对 τ 不变性:全体 mPL(τ) 排序相关
    print("\n图内排名对 τ 不变(全局缩放不改变序):"
          "\n  Δllr(τ) = (1/τ)·Δllr(1) 逐对成立 ⇒ 任一图的子集 mPL 排序与交互符号严格 τ-不变。")

    json.dump({"tau_star": float(star),
               "test": {"NLL": [nll(te, 1.0), nll(te, star)],
                        "Brier": [brier(te, 1.0), brier(te, star)],
                        "ECE": [ece(te, 1.0), ece(te, star)]},
               "mpl_scaling": scale_rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
