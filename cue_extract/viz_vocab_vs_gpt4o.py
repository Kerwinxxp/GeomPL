"""对照大图:每张图两栏 —— 左 = GPT-4o 线索掩码,右 = 固定词表线索掩码(编号 + 图例)。

输出 cue_extract/figures/vocab_vs_gpt4o.png。人眼 QA 用:看掩码是不是落在对的物体上。
运行:cue_extract/.venv/Scripts/python.exe -m cue_extract.viz_vocab_vs_gpt4o
"""
import glob
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
from PIL import Image

from cue_extract.rle import rle_to_mask
from cue_extract.viz import COLORS

SAM3DIR = os.path.join(os.path.dirname(__file__), "results_sam3")
VOCABDIR = os.path.join(os.path.dirname(__file__), "results_vocab")
OUT = os.path.join(os.path.dirname(__file__), "figures", "vocab_vs_gpt4o.png")


def masks_of(path):
    rec = json.load(open(path, encoding="utf-8"))
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
        out.append((c["cue"], u))
    return out, (W, H)


def panel(ax, img, cues, title):
    ov = np.asarray(img).astype(float).copy()
    for k, (_, m) in enumerate(cues):
        col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
        ov[m] = 0.5 * ov[m] + 0.5 * col
    ax.imshow(ov.astype(np.uint8))
    for k, (_, m) in enumerate(cues):
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        ax.text(xs.mean(), ys.mean(), str(k + 1), color="white", fontsize=9,
                fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.16", fc=COLORS[k % len(COLORS)], ec="white",
                          lw=0.7))
    ax.set_xticks([]); ax.set_yticks([])
    legend = "\n".join(f"{k+1}. {n[:34]}" for k, (n, _) in enumerate(cues)) or "(no cues)"
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xlabel(legend, fontsize=6.6, ha="left", x=0.0, labelpad=3, linespacing=1.25)


def main():
    sweep = {r["image_id"]: r.get("true_label", "?")
             for r in json.load(open(os.path.join(ROOT, "belief_elicit",
                                                  "georanker_sweep_results.json"),
                                     encoding="utf-8"))}
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it

    ids = sorted((os.path.basename(p)[:-5] for p in glob.glob(os.path.join(VOCABDIR, "*.jpg.json"))),
                 key=lambda i: sweep.get(i, "?"))
    ncol = 4                      # 2 图 x 2 栏
    nrow = int(np.ceil(len(ids) / 2))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.3 * nrow))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for k, iid in enumerate(ids):
        r, c0 = divmod(k, 2)
        c0 *= 2
        gc, (W, H) = masks_of(os.path.join(SAM3DIR, iid + ".json"))
        vc, _ = masks_of(os.path.join(VOCABDIR, iid + ".json"))
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).convert("RGB").resize((W, H))
        lab = f"{sweep.get(iid,'?')} [{iid.split('_')[0]}]"
        for ax in (axes[r][c0], axes[r][c0 + 1]):
            ax.axis("on")
        panel(axes[r][c0], img, gc, f"{lab}\nGPT-4o cues ({len(gc)})")
        panel(axes[r][c0 + 1], img, vc, f"{lab}\nfixed vocabulary ({len(vc)})")

    fig.suptitle("GPT-4o-proposed cues (left of each pair) vs fixed generic vocabulary "
                 "+ SAM 3 (right)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
