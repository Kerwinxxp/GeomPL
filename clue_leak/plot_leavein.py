"""leave-one-out(边际) vs leave-one-in(单独) 配对条形图。
每条线索两根柱:灰=遮它(边际泄露),彩=只留它(单独泄露 vs uniform)。
in >> out 的线索 = 强但冗余(信号被压在边际外);out > in = 协同/上下文型。
用法：python -m clue_leak.plot_leavein
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

plt.rcParams.update({"font.family": "Microsoft YaHei", "figure.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
DATA = os.path.join(os.path.dirname(__file__), "leavein_results.json")
OUT = os.path.join(os.path.dirname(__file__), "figures", "leavein_vs_out.png")


def main():
    rows = json.load(open(DATA, encoding="utf-8"))
    rows.sort(key=lambda r: r["mpl_in_vs_uniform"])
    labels = [f"{r['place']} · {r['cue'][:20]}" for r in rows]
    out = [r["mpl_out"] or 0 for r in rows]
    inn = [r["mpl_in_vs_uniform"] for r in rows]
    y = np.arange(len(rows))
    h = 0.4

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 1.5))
    ax.barh(y + h / 2, inn, h, color="#1E88E5", zorder=3,
            label="leave-one-IN:只留该线索(单独泄露 vs 无信息)")
    ax.barh(y - h / 2, out, h, color="#B0B7C0", zorder=3,
            label="leave-one-OUT:遮掉该线索(边际泄露)")
    for yi, (o, i) in enumerate(zip(out, inn)):
        ax.text(i + 0.004, yi + h / 2, f"{i:.3f}", va="center", fontsize=7.5, color="#1565C0")
        ax.text(o + 0.004, yi - h / 2, f"{o:.3f}", va="center", fontsize=7.5, color="#555")
        if i > 0.001 and (i - o) > 0.02:                           # 明显冗余标记
            tag = "边际≈0 · 完全冗余" if o < 0.005 else f"×{i / o:.1f} 冗余压制"
            ax.text(max(i, o) + 0.03, yi, tag,
                    va="center", fontsize=7.5, color="#C62828", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("mean mPL  (nats / 1000 km)", fontsize=10)
    ax.set_title("单独泄露(蓝) vs 边际泄露(灰):in >> out 即该线索强但被冗余压制\n"
                 "(Cuba 城墙是反例:out>in,价值来自场景协同而非单独)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", color="#EEEEEE", lw=0.6, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
