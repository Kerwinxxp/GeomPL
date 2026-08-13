"""【实验性 · 可整体删除】画 GeoCLIP + raw-mPL 的逐图全子集图。
左=原图+彩色线索掩码;右=每子集一根横条(单条线索色、组合深灰),值=raw mPL。
标题标注 GeoCLIP 后验 p_true 与 argmax(判断该图 GeoCLIP 是否可信)。
柱旁标 cov 与遮后 p_true(应随遮蔽下降)。
运行:cue_extract/.venv/Scripts/python.exe -m belief_elicit.plot_geoclip_ablation
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

DATA = os.path.join(os.path.dirname(__file__), "geoclip_ablation_results.json")
FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]
COMBO = "#54626F"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


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
            mm = rle_to_mask(i["mask_rle"])
            if mm.shape == (H, W):
                u |= mm
        out.append(u)
    return out, (W, H)


def load_img(iid, size):
    subset = {}
    import glob
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return Image.open(p).resize(size).convert("RGB")


def plot_one(r):
    iid, m = r["image_id"], r["n_maskable"]
    unions, (W, H) = cue_unions(iid)
    img = load_img(iid, (W, H))
    rows = sorted(r["combos"], key=lambda c: (len(c["subset"]), c["subset"]))
    n = len(rows)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, max(5.0, 0.5 * n + 1.8)),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    ov = np.asarray(img).astype(float).copy()
    for k, u in enumerate(unions):
        col = np.array(matplotlib.colors.to_rgb(COLORS[k % len(COLORS)])) * 255
        ov[u] = 0.45 * ov[u] + 0.55 * col
    axL.imshow(ov.astype(np.uint8)); axL.set_xticks([]); axL.set_yticks([])
    legend = "\n".join(f"{k+1}. {nm}" for k, nm in enumerate(r["cue_names"]))
    axL.set_title(f"{r['true_label']}  —  {m} cues\n"
                  f"GeoCLIP posterior: p_true={r['post_p_true']:.3f}  "
                  f"(argmax = {r['post_argmax'].split(',')[0]})\n{legend}",
                  fontsize=9.5, loc="left")

    y = np.arange(n)[::-1]
    lab = ["+".join(str(k + 1) for k in row["subset"]) for row in rows]
    val = [row["mpl_raw"] for row in rows]
    cols = [COLORS[row["subset"][0] % len(COLORS)] if len(row["subset"]) == 1 else COMBO
            for row in rows]
    axR.barh(y, val, color=cols, edgecolor="white", zorder=3)
    axR.set_yticks(y); axR.set_yticklabels(lab, fontsize=9)
    vmax = max(val + [1e-9])
    for yi, row in zip(y, rows):
        axR.text(row["mpl_raw"] + vmax * 0.02, yi,
                 f"{row['mpl_raw']:.3f}  (cov {row['cov']*100:.0f}%, p_t {row['p_true']:.2f})",
                 va="center", fontsize=7.5, color="#333")
    sizes = [len(row["subset"]) for row in rows]
    for i in range(1, n):
        if sizes[i] != sizes[i - 1]:
            axR.axhline(y[i] + 0.5, color="#CCC", lw=0.8, ls=":")
    axR.set_xlabel("mPL  (nats / 1000 km)  — GeoCLIP belief, no clamp", fontsize=10)
    axR.set_title("per masked subset: mPL (bar) + coverage + p_true after masking\n"
                  "p_true should DROP as more cues are masked", fontsize=9)
    axR.set_xlim(0, vmax * 1.32)
    axR.grid(axis="x", color="#EEE", zorder=0); axR.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    place = r["place"].lower().replace(" ", "")
    out = os.path.join(FIGDIR, f"geoclip_mpl_{place}_{iid.split('_')[0]}.png")
    fig.savefig(out, bbox_inches="tight", dpi=140); plt.close(fig)
    print("saved", out)


def main():
    for r in json.load(open(DATA, encoding="utf-8")):
        plot_one(r)


if __name__ == "__main__":
    main()
