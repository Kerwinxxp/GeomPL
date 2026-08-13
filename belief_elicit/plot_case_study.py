"""【实验性 · 可整体删除】逐图案例:原始单条 mPL vs Shapley φ 对比(+ SII 交互标注)。
左=原图+彩色线索掩码;中=single-cue mPL(基线,重复计数);右=Shapley φ(矫正,Σφ=v(N))。
中右同序同色 → 直观看共享质量如何被重分配。底部标注最强交互对(overlap/backup)。
用法：python -m belief_elicit.plot_case_study 158307292 754780171
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cue_extract.rle import rle_to_mask

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
DATA = os.path.join(os.path.dirname(__file__), "shapley_v2_results.json")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]


def cue_unions(iid):
    rec = json.load(open(os.path.join(ROOT, "cue_extract", "results_sam3", iid + ".json"),
                         encoding="utf-8"))
    W, H = rec["image_size"]
    out = []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        good = [i for i in c["instances"] if not i.get("degenerate") and i.get("mask_rle")]
        if not good:
            continue
        u = np.zeros((H, W), bool)
        for i in good:
            m = rle_to_mask(i["mask_rle"])
            if m.shape == (H, W):
                u |= m
        out.append(u)
    return out, (W, H)


def load_img(iid, size):
    import glob
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return Image.open(p).resize(size).convert("RGB")


def plot_one(r):
    iid = r["image_id"]
    cues = r["cues"]
    m = len(cues)
    unions, (W, H) = cue_unions(iid)
    img = load_img(iid, (W, H))

    fig = plt.figure(figsize=(15, max(4.2, 0.7 * m + 2.6)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28)
    axL = fig.add_subplot(gs[0]); axM = fig.add_subplot(gs[1]); axR = fig.add_subplot(gs[2])

    # 左:原图 + 掩码
    ov = np.asarray(img).astype(float).copy()
    for k, u in enumerate(unions):
        col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
        ov[u] = 0.45 * ov[u] + 0.55 * col
    axL.imshow(ov.astype(np.uint8)); axL.set_xticks([]); axL.set_yticks([])
    for k, u in enumerate(unions):
        ys, xs = np.nonzero(u)
        if len(xs):
            axL.text(xs.min() + 3, ys.min() + 3, str(k + 1), color="white", fontsize=12,
                     fontweight="bold", va="top",
                     bbox=dict(boxstyle="round,pad=0.15", fc=COLORS[k % len(COLORS)], ec="none"))
    axL.set_title(f"{r['place']} — {m} cues", fontsize=11, loc="left")

    # 布局:m 条线索柱 + 1 空行 + 2 条汇总柱(Σ 与 v(N)),汇总用深灰
    SUM, JOINT, GRAY = "#78909C", "#37474F", "#B0BEC5"
    v = [cues[k]["v_single"] for k in range(m)]
    phi = [cues[k]["phi"] for k in range(m)]
    sum_v = sum(v)
    cols = [COLORS[k % len(COLORS)] for k in range(m)]
    ncue = [f"{k+1}. {cues[k]['cue'][:22]}" for k in range(m)]
    # y 位置:线索在上,汇总在下(留一行间隔)
    y_cue = np.arange(m)[::-1] + 2.0
    y_sum, y_joint = 0.9, 0.0
    xmax = max(max(v), max(phi), sum_v, r["vN"]) * 1.30

    # ---- 中栏:单条 + Σv({k}) + v(N) ----
    axM.barh(y_cue, v, color=cols, edgecolor="white", zorder=3)
    for yi, val in zip(y_cue, v):
        axM.text(val + xmax * 0.015, yi, f"{val:.3f}", va="center", fontsize=8.5)
    axM.barh(y_sum, sum_v, color=SUM, edgecolor="white", zorder=3)
    axM.text(sum_v + xmax * 0.015, y_sum, f"{sum_v:.3f}", va="center", fontsize=8.5,
             fontweight="bold")
    axM.barh(y_joint, r["vN"], color=JOINT, edgecolor="white", zorder=3)
    axM.text(r["vN"] + xmax * 0.015, y_joint, f"{r['vN']:.3f}", va="center", fontsize=8.5,
             fontweight="bold")
    axM.set_yticks(list(y_cue) + [y_sum, y_joint])
    axM.set_yticklabels(ncue + ["Σ v({k})  (naive sum)", "v(N)  (mask ALL)"], fontsize=9.3)
    axM.axhline(1.5, color="#CCC", lw=0.8, ls=":")
    axM.set_xlim(0, xmax)
    axM.set_title("Single-cue mPL  v({k})\nΣv({k}) ≠ v(N)  →  NOT additive", fontsize=10)
    axM.set_xlabel("nats/1000 km"); axM.grid(axis="x", color="#EEE", zorder=0)
    axM.set_axisbelow(True)

    # ---- 右栏:φ + Σφ(=v(N)) ----
    axR.barh(y_cue, phi, color=cols, edgecolor="white", zorder=3)
    for yi, val, vv in zip(y_cue, phi, v):
        arrow = "↓" if val < vv - 1e-6 else ("↑" if val > vv + 1e-6 else "=")
        axR.text(val + xmax * 0.015, yi, f"{val:.3f} {arrow}", va="center", fontsize=8.5,
                 color="#1565C0")
    axR.barh(y_joint, sum(phi), color=JOINT, edgecolor="white", zorder=3)
    axR.text(sum(phi) + xmax * 0.015, y_joint, f"{sum(phi):.3f}", va="center", fontsize=8.5,
             fontweight="bold")
    axR.set_yticks(list(y_cue) + [y_joint])
    axR.set_yticklabels([]); axR.set_xlim(0, xmax)
    axR.axhline(1.5, color="#CCC", lw=0.8, ls=":")
    axR.text(sum(phi) + xmax * 0.015, y_sum, "Σφ = v(N)  (additive) ✓", va="center",
             fontsize=9, color=JOINT, fontweight="bold")
    axR.set_title(f"Shapley $\\varphi_k$ (corrected)\n$\\Sigma\\varphi = v(N) = {r['vN']:.3f}$  →  additive",
                  fontsize=10)
    axR.set_xlabel("nats/1000 km"); axR.grid(axis="x", color="#EEE", zorder=0)
    axR.set_axisbelow(True)

    # 交互标注:最强 overlap 与 backup 对
    sii = [(v, kl) for kl, v in r["sii"].items()]
    lo = min(sii); hi = max(sii)
    def pair_txt(val, kl):
        k, l = map(int, kl.split(","))
        kind = "overlap" if val < 0 else "backup"
        return f"{kind}: cue {k+1} × cue {l+1},  $I^{{SII}}$ = {val:+.3f}"
    note = "Strongest interaction — " + pair_txt(*lo)
    if hi[0] > 0.01:
        note += "     " + pair_txt(*hi)
    fig.text(0.5, 0.005, note, ha="center", fontsize=9.5, color="#455A64")

    fig.suptitle("Per-cue leakage: raw single-cue baseline  vs  Shapley attribution",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, f"case_{r['place'].lower().replace(' ','')}_{iid.split('_')[0]}.png")
    fig.savefig(out, bbox_inches="tight", dpi=130); plt.close(fig)
    print("saved", out)


def main():
    R = {r["image_id"][:len(a)]: r for r in json.load(open(DATA, encoding="utf-8"))
         for a in [r["image_id"]]}
    prefs = sys.argv[1:] or ["158307292", "754780171"]
    allrows = json.load(open(DATA, encoding="utf-8"))
    for pref in prefs:
        r = next((x for x in allrows if x["image_id"].startswith(pref)), None)
        if r:
            plot_one(r)
        else:
            print("not found:", pref)


if __name__ == "__main__":
    main()
