"""每图的"净泄露"对照图(主指标 = 真线索 mPL − 等面积对照 mPL)。
左 = 原图 + SAM 掩码(编号);右 = 每条线索:真线索(蓝)vs 等面积对照(灰)双柱 + 净 Δ。
Δ>0 = 泄露内容特异(经得起对照);Δ≤0 = 只是"遮大块"伪影。
数据来自 clue_leak/control_area_results.json(先跑 control_area.py)。
用法：python -m clue_leak.plot_net_mpl
"""
import glob
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

CTRL = os.path.join(os.path.dirname(__file__), "control_area_results.json")
CUEDIR = os.path.join(ROOT, "cue_extract", "results_sam3")
OUTDIR = os.path.join(os.path.dirname(__file__), "figures", "net_mpl_sam3")
COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]
plt.rcParams.update({"font.family": "Microsoft YaHei", "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_subsets():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); out[it["image_id"]] = it   # 覆盖:后加载(subset_sample 高清)优先
    return out


def main():
    data = json.load(open(CTRL, encoding="utf-8"))
    subset = load_subsets()
    by_img = {}
    for o in data:
        by_img.setdefault(o["image"], []).append(o)

    os.makedirs(OUTDIR, exist_ok=True)
    for iid, cues in by_img.items():
        cue_rec = json.load(open(os.path.join(CUEDIR, iid + ".json"), encoding="utf-8"))
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize(tuple(cue_rec["image_size"])).convert("RGB")
        # 掩码并集叠图(按 control 结果里的线索顺序)
        mcues = [c for c in cue_rec["geo_privacy_cues"]
                 if c.get("maskable") and any((not i.get("degenerate")) and i.get("mask_rle") for i in c["instances"])]
        overlay = np.asarray(img).astype(float).copy()
        for k, c in enumerate(mcues):
            col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
            for i in c["instances"]:
                if (not i.get("degenerate")) and i.get("mask_rle"):
                    m = rle_to_mask(i["mask_rle"])
                    if m.shape == overlay.shape[:2]:
                        overlay[m] = 0.5 * overlay[m] + 0.5 * col

        n = len(cues)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, max(4.5, 0.8 * n + 2.2)),
                                       gridspec_kw={"width_ratios": [1.1, 1]})
        axL.imshow(overlay.astype(np.uint8))
        for k, c in enumerate(mcues):
            for i in c["instances"][:1]:
                if i.get("mask_rle"):
                    m = rle_to_mask(i["mask_rle"]); ys, xs = np.nonzero(m)
                    if len(xs):
                        axL.text(xs.min() + 2, ys.min() + 2, str(k + 1), color="white",
                                 fontsize=12, fontweight="bold", va="top",
                                 bbox=dict(boxstyle="round,pad=0.15", fc=COLORS[k % len(COLORS)], ec="none"))
        axL.set_xticks([]); axL.set_yticks([])
        axL.set_title(f"{cues[0]['place']} — {n} cues", fontsize=11)

        y = np.arange(n)[::-1]; h = 0.38
        real = [o["real_mpl"] or 0 for o in cues]
        ctrl = [o["ctrl_mpl"] or 0 for o in cues]
        axR.barh(y + h / 2, real, h, color=[COLORS[k % len(COLORS)] for k in range(n)],
                 edgecolor="white", zorder=3, label="真线索")
        axR.barh(y - h / 2, ctrl, h, color="#CCCCCC", edgecolor="white", zorder=3, label="等面积对照")
        axR.set_yticks(y)
        axR.set_yticklabels([f"{k+1}. {o['cue'][:20]} ({o['area_pct']:.0f}%)" for k, o in enumerate(cues)], fontsize=9)
        vmax = max(real + ctrl + [1e-6])
        for yi, o in zip(y, cues):
            d = (o["real_mpl"] or 0) - (o["ctrl_mpl"] or 0)
            c = "#1a7f37" if d > 0.01 else ("#999999" if abs(d) <= 0.01 else "#c0392b")
            axR.text(max(o["real_mpl"] or 0, o["ctrl_mpl"] or 0) + vmax * 0.02, yi,
                     f"净Δ={d:+.3f}", va="center", fontsize=8.5, color=c, fontweight="bold")
        axR.set_xlabel("mean mPL (nats/1000km)", fontsize=10)
        axR.set_title("真线索 vs 等面积对照(同形掩码移到非线索处)\n净Δ>0=内容特异,≤0=遮大块伪影", fontsize=9.5)
        axR.set_xlim(0, vmax * 1.35); axR.legend(frameon=False, fontsize=8.5, loc="lower right")
        axR.grid(axis="x", color="#EEE", lw=0.5, zorder=0); axR.set_axisbelow(True)
        fig.tight_layout()
        stem = "".join(ch for ch in cues[0]["place"].lower() if ch.isalnum())[:12]
        fig.savefig(os.path.join(OUTDIR, f"{stem}_{iid.split('_')[0]}_net.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"saved {stem}_{iid.split('_')[0]}_net.png")


if __name__ == "__main__":
    main()
