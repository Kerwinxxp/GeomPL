"""【实验性 · 可整体删除】GPT-4o 线索清单 vs 固定词表清单:三联图。

(a) 逐图 v(N) 配对条形:两种清单"能移除的全部泄漏"(同图 / 同算子 / 同攻击者)
(b) 逐图逐线索 φ 点图,两列(GPT-4o | vocab);IoU>=0.5 的配对用细线连;
    实心 = 高于伪影零假设阈值(敏感),空心 = 不敏感;阈值 c_img/m 画成短横线
(c) 三组 φ 的中位数条形 + 散点:无对应的词表线索 / 有对应的词表线索 / 无对应的 GPT-4o 线索

数据来自 belief_elicit/vocab_vs_gpt4o.json(先跑 python -m belief_elicit.vocab_vs_gpt4o)。
运行:python -m belief_elicit.plot_vocab_vs_gpt4o
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "vocab_vs_gpt4o.json")
OUT = os.path.join(HERE, "figures", "vocab_vs_gpt4o_leakage.png")
BLUE, RED, GRAY, GREEN = "#1E88E5", "#E53935", "#B0BEC5", "#43A047"


def short(lbl):
    return lbl.split(",")[0]


# ---------------- (a) 逐图 v(N) ----------------

def panel_a(ax, S):
    P = S["per_image"]
    A = S["aggregate"]
    y = np.arange(len(P))[::-1]
    h = 0.36
    g = [x["vN_gpt"] for x in P]
    v = [x["vN_vocab"] for x in P]
    ax.barh(y + h / 2, g, h, color=BLUE, zorder=3, label="GPT-4o-proposed cues")
    ax.barh(y - h / 2, v, h, color=GREEN, zorder=3, label="fixed vocabulary (12 concepts)")
    xmax = max(max(g), max(v))
    for yi, x in zip(y, P):
        ax.text(x["vN_gpt"] + xmax * 0.012, yi + h / 2, f"{x['vN_gpt']:.3f}",
                va="center", ha="left", fontsize=7.5, color="#1565C0")
        ax.text(x["vN_vocab"] + xmax * 0.012, yi - h / 2, f"{x['vN_vocab']:.3f}",
                va="center", ha="left", fontsize=7.5, color="#2E7D32")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short(x['true_label'])}  ({x['m_gpt']}|{x['m_vocab']})"
                        for x in P], fontsize=8.5)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_ylim(-0.7, len(P) - 0.3)
    ax.set_xlabel("v(N) — total removable leakage, mPL (nats / 1000 km)\n"
                  "y labels: image  (m_gpt | m_vocab)", fontsize=9.5)
    ax.set_title("(a) Whole-inventory removal v(N), per image\n"
                 f"sum {A['sum_vN_gpt']:.3f} (GPT-4o) vs {A['sum_vN_vocab']:.3f} (vocab); "
                 f"per-image wins {A['n_win_gpt']} / {A['n_win_vocab']}"
                 + (f" / {A['n_win_tie']} tie" if A.get("n_win_tie") else ""),
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.grid(axis="x", color="#EEE", zorder=0)
    ax.set_axisbelow(True)


# ---------------- (b) 逐线索 φ ----------------

def panel_b(ax, S):
    P = S["per_image"]
    step, gap = 1.0, 0.42           # 每张图占 step,列间距 gap
    xt, xl = [], []
    for i, x in enumerate(P):
        xg, xv = i * step - gap / 2, i * step + gap / 2
        xt.append(i * step)
        xl.append(short(x["true_label"]))
        if i % 2:
            ax.axvspan(i * step - step / 2, i * step + step / 2, color="#F5F7F8",
                       zorder=0, lw=0)
        # 阈值短横
        ax.plot([xg - 0.11, xg + 0.11], [x["thr_gpt"]] * 2, color="#9E9E9E",
                lw=1.4, zorder=2)
        ax.plot([xv - 0.11, xv + 0.11], [x["thr_vocab"]] * 2, color="#9E9E9E",
                lw=1.4, zorder=2)
        # 匹配连线
        for p in x["matched_pairs"]:
            ax.plot([xg, xv], [p["phi_gpt"], p["phi_vocab"]], color="#B0BEC5",
                    lw=0.8, zorder=1)
        for xx, rows, col in ((xg, x["gpt_cues"], BLUE), (xv, x["vocab_cues"], GREEN)):
            # 同一列内 φ 相近的点做微小水平抖动,避免完全叠在一起
            order = np.argsort([r["phi"] for r in rows])
            jit = {}
            prev, run = None, 0
            span = 0.055
            for j in order:
                p = rows[j]["phi"]
                run = run + 1 if (prev is not None and abs(p - prev) < 0.012) else 0
                jit[j] = (run - (run % 2) * 0) * 0.0
                jit[j] = (0 if run == 0 else (span if run % 2 else -span) * ((run + 1) // 2))
                prev = p
            for j, r in enumerate(rows):
                ax.scatter(xx + jit[j], r["phi"], s=34,
                           facecolors=(col if r["sensitive"] else "none"),
                           edgecolors=col, linewidths=1.3, zorder=4)
    ax.axhline(0, color="#CCC", lw=0.9, zorder=0)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=8, rotation=30, ha="right")
    ax.set_xlim(-0.55, (len(P) - 1) * step + 0.55)
    ax.set_ylabel("order-2 anchored φ (nats / 1000 km)")
    ax.set_title("(b) Per-cue attribution φ — left dot column = GPT-4o, right = vocabulary\n"
                 "filled = above the artifact null c_img/m (grey tick), hollow = below; "
                 "thin line = same region (IoU ≥ 0.5)", fontsize=10.5)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=6.5,
               label="GPT-4o cue, sensitive"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=BLUE, ms=6.5,
               label="GPT-4o cue, not sensitive"),
        Line2D([], [], marker="o", ls="", mfc=GREEN, mec=GREEN, ms=6.5,
               label="vocabulary cue, sensitive"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=GREEN, ms=6.5,
               label="vocabulary cue, not sensitive"),
        Line2D([], [], color="#9E9E9E", lw=1.4, label="null threshold c_img / m"),
    ], fontsize=8, loc="upper left", ncol=2, framealpha=0.95)
    ax.grid(axis="y", color="#EEE", zorder=0)
    ax.set_axisbelow(True)


# ---------------- (c) 三组 φ 分布 ----------------

def panel_c(ax, S):
    A = S["aggregate"]
    groups = [("vocabulary cues\nwith NO GPT-4o match", A["vocab_unmatched"], RED),
              ("vocabulary cues\nmatched to GPT-4o", A["vocab_matched"], GREEN),
              ("GPT-4o cues\nwith NO vocabulary match", A["gpt_unmatched"], BLUE)]
    xs = np.arange(len(groups))
    meds = [d["phi_median"] for _, d, _ in groups]
    ax.bar(xs, meds, 0.55, color=[c for _, _, c in groups], alpha=0.32, zorder=2)
    rng = np.random.default_rng(0)
    for i, (_, d, c) in enumerate(groups):
        phi = np.array(d["phi"], float)
        if not len(phi):
            continue
        jx = xs[i] + rng.uniform(-0.17, 0.17, len(phi))
        ax.scatter(jx, phi, s=26, color=c, alpha=0.8, edgecolors="white",
                   linewidths=0.5, zorder=4)
        ax.plot([xs[i] - 0.275, xs[i] + 0.275], [meds[i]] * 2, color=c, lw=2.4, zorder=5)
    ax.axhline(0, color="#CCC", lw=0.9, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\nn={d['n']}, {d['n_sensitive']} sensitive"
                        f"\nmedian = {d['phi_median']:.3f}"
                        for n, d, _ in groups], fontsize=8.5)
    ax.set_ylabel("order-2 anchored φ (nats / 1000 km)")
    ax.set_title("(c) Does either inventory miss leaky regions?\n"
                 "φ of cues the other inventory has no counterpart for", fontsize=10.5)
    ax.grid(axis="y", color="#EEE", zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description="GPT-4o 清单 vs 词表清单 三联图")
    ap.add_argument("--summary", default=SUMMARY)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if not os.path.exists(a.summary):
        print(f"[fatal] 找不到 {a.summary};先跑 python -m belief_elicit.vocab_vs_gpt4o")
        return 1
    S = json.load(open(a.summary, encoding="utf-8"))

    fig = plt.figure(figsize=(15.0, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.40, wspace=0.19)
    panel_a(fig.add_subplot(gs[0, 0]), S)
    panel_c(fig.add_subplot(gs[0, 1]), S)
    panel_b(fig.add_subplot(gs[1, :]), S)

    ns = S["meta"]["null_sources"]
    fig.suptitle("Top-down (GPT-4o-proposed) vs bottom-up (fixed 12-concept vocabulary) cue "
                 "inventories\n"
                 f"{S['meta']['n_images']} images, LaMa inpainting removal, GeoRanker adversary; "
                 "artifact null from "
                 + " + ".join(f"{k} controls ({v} img)" for k, v in ns.items()),
                 fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.085)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=130)
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
