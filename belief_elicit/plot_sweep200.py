"""【实验性 · 可整体删除】GeoCLIP 200 张 sweep 的两张汇总图。
① 精度体检:km 误差分档命中率 + 国家命中率 + p_true 分布 —— 判断 GeoCLIP 够不够用。
② 逐类别泄露:按线索类别聚合的单线索 mPL(只在 GeoCLIP 认得准的图上,country_hit=True)。
运行:cue_extract/.venv/Scripts/python.exe -m belief_elicit.plot_sweep200
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
DATA = os.path.join(os.path.dirname(__file__), "geoclip_sweep200_results.json")
OUT = os.path.join(os.path.dirname(__file__), "figures", "geoclip_sweep200.png")
BLUE, GREEN, ORANGE = "#1E88E5", "#43A047", "#FB8C00"


def main():
    R = json.load(open(DATA, encoding="utf-8"))
    km = np.array([r["km_error"] for r in R])
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    # ① km 误差分档(标准地理定位阈值)
    thr = [(1, "street\n<1km"), (25, "city\n<25km"), (200, "region\n<200km"),
           (750, "country\n<750km"), (2500, "cont.\n<2500km")]
    acc = [(km <= t).mean() * 100 for t, _ in thr]
    b = ax[0].bar(range(len(thr)), acc, color=BLUE, zorder=3)
    for bi, v in zip(b, acc):
        ax[0].text(bi.get_x() + bi.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax[0].set_xticks(range(len(thr))); ax[0].set_xticklabels([n for _, n in thr], fontsize=9)
    ax[0].set_ylabel("% of images within threshold"); ax[0].set_ylim(0, 105)
    ch = np.mean([r["country_hit"] for r in R]) * 100
    ax[0].set_title(f"(a) GeoCLIP accuracy on {len(R)} images\ncountry-hit {ch:.0f}%  ·  "
                    f"median error {np.median(km):.0f} km", fontsize=11)
    ax[0].grid(axis="y", color="#EEE", zorder=0); ax[0].set_axisbelow(True)

    # ② p_true 分布
    pt = np.array([r["p_true"] for r in R])
    ax[1].hist(pt, bins=np.linspace(0, 1, 21), color=GREEN, zorder=3)
    ax[1].axvline(np.median(pt), color=ORANGE, lw=1.5, label=f"median {np.median(pt):.2f}")
    ax[1].set_xlabel("p_true (GeoCLIP posterior on true label)")
    ax[1].set_ylabel("images"); ax[1].legend(fontsize=9)
    ax[1].set_title("(b) Confidence on the true location", fontsize=11)
    ax[1].grid(axis="y", color="#EEE", zorder=0); ax[1].set_axisbelow(True)

    # ③ 逐类别单线索 mPL(只用 country_hit=True 的图)
    good = [r for r in R if r["country_hit"]]
    cat = defaultdict(list)
    for r in good:
        for c in r["per_cue"]:
            cat[c["category"] or "unknown"].append(c["mpl"])
    items = sorted(cat.items(), key=lambda kv: -np.median(kv[1]))
    names = [k for k, _ in items]
    meds = [np.median(v) for _, v in items]
    cnts = [len(v) for _, v in items]
    yb = ax[2].barh(range(len(names))[::-1], meds, color=BLUE, zorder=3)
    for i, (m, c) in enumerate(zip(meds, cnts)):
        ax[2].text(m + max(meds) * 0.02, len(names) - 1 - i, f"{m:.3f} (n={c})",
                   va="center", fontsize=8)
    ax[2].set_yticks(range(len(names))[::-1]); ax[2].set_yticklabels(names, fontsize=9)
    ax[2].set_xlabel("median single-cue mPL (nats/1000km)")
    ax[2].set_title(f"(c) Per-category leakage\n(GeoCLIP-recognized images, n={len(good)})", fontsize=11)
    ax[2].set_xlim(0, max(meds) * 1.35)
    ax[2].grid(axis="x", color="#EEE", zorder=0); ax[2].set_axisbelow(True)

    fig.suptitle("GeoCLIP belief meter on 200 images — accuracy health check + per-cue location leakage",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=120)
    print("saved", OUT)


if __name__ == "__main__":
    main()
