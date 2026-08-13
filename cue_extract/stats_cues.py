"""线索统计分析可视化(跨全数据集)。图内文字全英文(投稿/给导师用)。

6 个面板回答:攻击者到底在看什么、看得准不准、这些线索占多大画面。
  ① 线索类别分布(GPT-4o 自报的 8 类)
  ② 每图线索数 / maskable 数分布
  ③ 定位结局:成功掩码 / 退化(占满画幅) / SAM3 未定位  —— 管线健康度
  ④ 掩码面积占比分布(线索有多"大") + 40% 退化阈值线
  ⑤ GPT-4o 置信度 vs 是否成功定位(高置信线索是否更可抠)
  ⑥ 最高频线索词 top15(攻击者反复依赖的线索类型)
用法：python -m cue_extract.stats_cues [data/subset200_hires.jsonl]
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cue_extract.rle import rle_to_mask

RES = os.path.join(os.path.dirname(__file__), "results_sam3")
BLUE, GREEN, ORANGE, RED, GRAY = "#1E88E5", "#43A047", "#FB8C00", "#E53935", "#B0B7C0"
# 定位结局三态(同时用作字典键与图例文字)
OK, DEGEN, MISS = "Localized", "Degenerate (fills frame)", "Not localized (no SAM3 instance)"


def collect(src):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, src), encoding="utf-8")]
    cues, per_img = [], []
    for it in rows:
        f = os.path.join(RES, it["image_id"] + ".json")
        if not os.path.exists(f):
            continue
        rec = json.load(open(f, encoding="utf-8"))
        W, H = rec.get("image_size", [1, 1])
        n_mask = 0
        for c in rec["geo_privacy_cues"]:
            insts = c.get("instances", [])
            good = [i for i in insts if not i.get("degenerate") and i.get("mask_rle")]
            if not insts:
                outcome = MISS
            elif not good:
                outcome = DEGEN
            else:
                outcome = OK
                n_mask += 1
            area = np.nan
            if good:
                u = np.zeros((H, W), bool)
                for i in good:
                    m = rle_to_mask(i["mask_rle"])
                    if m.shape == (H, W):
                        u |= m
                area = u.sum() / (W * H) * 100
            cues.append({"cue": c["cue"], "category": c.get("category") or "unknown",
                         "is_text": bool(c.get("is_text")), "conf": c.get("confidence"),
                         "outcome": outcome, "area": area,
                         "country": it.get("gt_country")})
        per_img.append({"n_cues": len(rec["geo_privacy_cues"]), "n_mask": n_mask,
                        "country": it.get("gt_country")})
    return cues, per_img, len(rows)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/subset100_hires.jsonl"
    cues, per_img, n_req = collect(src)
    n_img = len(per_img)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    # ① 类别分布
    ax = axes[0, 0]
    cat = Counter(c["category"] for c in cues).most_common()
    ax.barh([k for k, _ in cat][::-1], [v for _, v in cat][::-1], color=BLUE, zorder=3)
    for i, (_, v) in enumerate(cat[::-1]):
        ax.text(v + max(v for _, v in cat) * 0.01, i, str(v), va="center", fontsize=8)
    ax.set_title(f"(a) Cue category distribution ({len(cues)} cues)", fontsize=11)
    ax.set_xlabel("number of cues"); ax.grid(axis="x", color="#EEE", zorder=0)
    ax.set_axisbelow(True)

    # ② 每图线索数 / maskable 数
    ax = axes[0, 1]
    mx = max(max(p["n_cues"] for p in per_img), max(p["n_mask"] for p in per_img))
    bins = np.arange(-0.5, mx + 1.5)
    ax.hist([[p["n_cues"] for p in per_img], [p["n_mask"] for p in per_img]], bins=bins,
            color=[BLUE, GREEN], label=["cues extracted", "maskable cues"], zorder=3)
    ax.set_title(f"(b) Cues per image (n={n_img} images)", fontsize=11)
    ax.set_xlabel("cues per image"); ax.set_ylabel("number of images"); ax.legend(fontsize=9)
    ax.set_xticks(range(mx + 1))
    ax.grid(axis="y", color="#EEE", zorder=0); ax.set_axisbelow(True)

    # ③ 定位结局
    ax = axes[0, 2]
    oc = Counter(c["outcome"] for c in cues)
    order = [OK, DEGEN, MISS]
    short = ["Localized", "Degenerate\n(fills frame)", "Not localized\n(no SAM3 instance)"]
    vals = [oc.get(k, 0) for k in order]
    w = ax.bar(range(3), vals, color=[GREEN, ORANGE, RED], zorder=3)
    for b, v in zip(w, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v/len(cues)*100:.0f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=9)
    ax.set_title("(c) Cue localization outcome (pipeline health)", fontsize=11)
    ax.set_ylabel("number of cues"); ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(axis="y", color="#EEE", zorder=0); ax.set_axisbelow(True)

    # ④ 掩码面积分布
    ax = axes[1, 0]
    areas = [c["area"] for c in cues if not np.isnan(c["area"])]
    ax.hist(areas, bins=np.arange(0, 62, 2.5), color=BLUE, zorder=3)
    ax.axvline(40, color=RED, ls="--", lw=1.2, label="40% degeneracy threshold")
    ax.axvline(float(np.median(areas)), color=GREEN, ls="-", lw=1.4,
               label=f"median {np.median(areas):.1f}%")
    ax.set_title(f"(d) Image area covered by localized cues ({len(areas)} cues)", fontsize=11)
    ax.set_xlabel("mask area / full image (%)"); ax.set_ylabel("number of cues")
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="#EEE", zorder=0); ax.set_axisbelow(True)

    # ⑤ 置信度 vs 定位结局
    ax = axes[1, 1]
    confs = sorted({c["conf"] for c in cues if c["conf"] is not None}, key=lambda x: str(x))
    if confs and all(isinstance(c, (int, float)) for c in confs):
        data = [[c["conf"] for c in cues if c["outcome"] == k and c["conf"] is not None]
                for k in order]
        ax.hist(data, bins=8, color=[GREEN, ORANGE, RED], label=order, zorder=3)
    else:                                   # 离散置信(high/medium/low)
        keys = [k for k in ["high", "medium", "low"] if k in confs] or confs
        x = np.arange(len(keys)); bw = 0.26
        for j, (k, col) in enumerate(zip(order, [GREEN, ORANGE, RED])):
            v = [sum(1 for c in cues if c["conf"] == kk and c["outcome"] == k) for kk in keys]
            ax.bar(x + (j - 1) * bw, v, bw, color=col, label=k, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_xlabel("GPT-4o self-reported confidence")
    ax.set_title("(e) Confidence x localization outcome", fontsize=11)
    ax.set_ylabel("number of cues"); ax.legend(fontsize=8)
    ax.grid(axis="y", color="#EEE", zorder=0); ax.set_axisbelow(True)

    # ⑥ 高频线索词
    ax = axes[1, 2]
    stop = {"on", "in", "the", "a", "of", "with", "and", "style", "an", "at", "for"}
    words = Counter(w for c in cues for w in re.findall(r"[a-z]+", c["cue"].lower())
                    if w not in stop and len(w) > 2)
    top = words.most_common(15)
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color=GREEN, zorder=3)
    for i, (_, v) in enumerate(top[::-1]):
        ax.text(v + top[0][1] * 0.01, i, str(v), va="center", fontsize=8)
    ax.set_title("(f) Most frequent cue terms (top 15)", fontsize=11)
    ax.set_xlabel("occurrences"); ax.grid(axis="x", color="#EEE", zorder=0)
    ax.set_axisbelow(True)

    ntext = sum(1 for c in cues if c["is_text"])
    ncountry = len({p["country"] for p in per_img})
    fig.suptitle(f"Geo-privacy cue statistics — {n_img} images / {ncountry} countries / "
                 f"{len(cues)} cues ({ntext} textual, {ntext/len(cues)*100:.0f}%) — "
                 f"route-B + SAM 3 hi-res pipeline", fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(os.path.dirname(__file__), "figures", f"cue_stats_{n_img}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=115)
    print("saved", out)
    print(f"images {n_img}/{n_req} | cues {len(cues)} | localized {oc.get(OK,0)} "
          f"| degenerate {oc.get(DEGEN,0)} | not localized {oc.get(MISS,0)}")


if __name__ == "__main__":
    main()
