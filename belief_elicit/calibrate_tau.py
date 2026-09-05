"""Temperature calibration: tau sensitivity + NLL/Brier/ECE + the mPL ~ 1/tau check.

The key point is that softmax(s/tau) can be recovered exactly from the stored p(1), so
nothing has to be re-scored:
    p_i(tau) = p_i(1)^(1/tau) / sum_j p_j(1)^(1/tau)
So (a) tau* is chosen on a held-out split by minimising the true-label NLL, with Brier/ECE
reported, and (b) mPL(tau) ~ mPL(1)/tau is shown directly — tau is only a global scale, so
within-image comparisons and interaction signs are tau-invariant.

Run (main environment): python -m belief_elicit.calibrate_tau
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

from belief_elicit.geometry import build_geometry, mpl
from belief_elicit.results import CALIBRATE_TAU as OUT, SWEEP, load_gallery, load_sweep


def temper(prob, tau):
    """p(1) dict -> p(tau) dict; exact recovery of softmax(s/tau)."""
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
    rows = load_sweep(SWEEP)
    gv = load_gallery()
    rep, clusters, dist = build_geometry(gv, merge_km=2.0)

    # (a) held-out tau selection: fixed 50/50 split, minimise NLL on train, report on test
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

    # (b) how mPL scales with tau: a couple of representative images, expect ~ 1/tau
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

    # ranking is tau-invariant
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
