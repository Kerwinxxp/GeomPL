"""等面积对照的可视化对比:真线索掩码 vs 同形状随机平移掩码。

关键:用与 control_area.py 相同的 seed **精确重放**随机序列,保证图上每块掩码
与 control_area_results.json 里存的 ctrl_samples[i] 一一对应(否则数字对不上图)。
超过真线索 mPL 的对照用红框标出 —— 这就是"随便遮一块也可能泄露更高"的直接证据。
用法：python -m clue_leak.plot_control_example [image_id前缀]   (默认 847733166 Venice)
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

from clue_leak.control_area import sample_control
from cue_extract.rle import rle_to_mask

COMBODIR = os.path.join(os.path.dirname(__file__), "combo2_sam3_results")
CUEDIR = os.path.join(ROOT, "cue_extract", "results_sam3")
RESULTS = os.path.join(os.path.dirname(__file__), "control_area_results.json")
NCTRL, SEED = 5, 42                      # 必须与当初 control_area.py 的运行参数一致
GRAY = 128


def masks_of(iid):
    """该图每条 maskable 线索的掩码并集(顺序与 control_area.py 完全一致)。"""
    rec = json.load(open(os.path.join(CUEDIR, iid + ".json"), encoding="utf-8"))
    W, H = rec["image_size"]
    mcues = [c for c in rec["geo_privacy_cues"]
             if c.get("maskable") and any((not i.get("degenerate")) and i.get("mask_rle")
                                          for i in c["instances"])]
    out = []
    for c in mcues:
        u = np.zeros((H, W), bool)
        for i in c["instances"]:
            if (not i.get("degenerate")) and i.get("mask_rle"):
                m = rle_to_mask(i["mask_rle"])
                if m.shape == (H, W):
                    u |= m
        out.append((c["cue"], u))
    return out, (W, H)


def replay(target_iid):
    """重放 control_area.py 的 rng 序列,取出 target 图每条线索的 NCTRL 个平移掩码。"""
    rng = np.random.default_rng(SEED)
    got = {}
    for f in sorted(glob.glob(os.path.join(COMBODIR, "*.json"))):
        iid = os.path.basename(f)[:-5]
        cues, _ = masks_of(iid)
        union = np.zeros_like(cues[0][1]) if cues else None
        for _, m in cues:
            union |= m
        for ci, (name, m) in enumerate(cues):
            placements = [sample_control(m, union, rng) for _ in range(NCTRL)]
            if iid == target_iid:
                got[ci] = (name, m, placements)
    return got


def apply_mask(img_arr, mask):
    a = img_arr.copy()
    a[mask] = GRAY
    return a


def main():
    pref = sys.argv[1] if len(sys.argv) > 1 else "847733166"
    iid = next(os.path.basename(f)[:-5] for f in sorted(glob.glob(os.path.join(COMBODIR, "*.json")))
               if os.path.basename(f).startswith(pref))
    stored = json.load(open(RESULTS, encoding="utf-8"))
    rows = [o for o in stored if o["image"] == iid]
    got = replay(iid)

    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    _, (W, H) = masks_of(iid)
    p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    img = np.asarray(Image.open(p).resize((W, H)).convert("RGB"))

    # 取 net Δ 最有代表性的一条线索(默认第 0 条 = 拉丁铭文)
    ci = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    name, cue_mask, placements = got[ci]
    rec = rows[ci]
    real, ctrl_samples = rec["real_mpl"], rec["ctrl_samples"]

    def outline(ax, mask, color):
        """掩码很小,画个轮廓框把它圈出来,否则读者找不到灰块在哪。"""
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return
        pad = max(H, W) * 0.03
        ax.add_patch(plt.Rectangle((xs.min() - pad, ys.min() - pad),
                                   xs.max() - xs.min() + 2 * pad,
                                   ys.max() - ys.min() + 2 * pad,
                                   fill=False, edgecolor=color, lw=2.0, ls="--"))

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    n = 2 + len(placements)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 5.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(img); axes[0].set_title("Original\n(full image)", fontsize=11)
    axes[1].imshow(apply_mask(img, cue_mask))
    outline(axes[1], cue_mask, "#1565C0")
    axes[1].set_title(f"REAL cue masked — {name[:26]}\nmPL = {real:.3f}", fontsize=11,
                      color="#1565C0", fontweight="bold")

    for j, (t, v) in enumerate(zip(placements, ctrl_samples)):
        ax = axes[2 + j]
        ax.imshow(apply_mask(img, t) if t is not None else img)
        if t is not None:
            outline(ax, t, "#C62828" if v > real else "#777")
        beats = v > real
        ax.set_title(f"Control #{j+1} (same shape, moved)\nmPL = {v:.3f}"
                     + ("   BEATS the real cue" if beats else ""), fontsize=10.5,
                     color="#C62828" if beats else "#555",
                     fontweight="bold" if beats else "normal")

    # 结论按数据自动生成(不要硬编码某张图的结论)
    nbeat = sum(1 for v in ctrl_samples if v > real)
    lo, hi = min(ctrl_samples), max(ctrl_samples)
    cm, cs = float(np.mean(ctrl_samples)), float(np.std(ctrl_samples))
    if real > hi:
        verdict = (f"The real cue ({real:.3f}) sits ABOVE every control placement "
                   f"({lo:.3f}-{hi:.3f}) -> content-specific leakage, not a gray-patch artifact.")
    elif real < lo:
        verdict = (f"The real cue ({real:.3f}) sits BELOW every control placement "
                   f"({lo:.3f}-{hi:.3f}) -> masking this cue leaks LESS than masking unrelated content.")
    else:
        verdict = (f"The real cue ({real:.3f}) falls INSIDE the control spread ({lo:.3f}-{hi:.3f}) "
                   f"-> mPL cannot separate this cue from an equally-sized patch of unrelated content.")
    fig.suptitle(
        f"Equal-area control — {rec['place']}: \"{name}\"   (mask covers {rec['area_pct']}% of the image)\n"
        f"The SAME mask shape/area, translated to non-cue locations. "
        f"{nbeat}/{len(ctrl_samples)} random placements leak more than the real cue. "
        f"Control mean {cm:.3f} +/- {cs:.3f} (SD, n={len(ctrl_samples)}).\n"
        f"{verdict}",
        fontsize=12, y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(os.path.dirname(__file__), "figures",
                       f"control_example_{rec['place'].lower().replace(' ','')}_{ci}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    print("saved", out)
    print(f"{rec['place']} | {name} | real={real:.3f} | ctrl={[round(v,3) for v in ctrl_samples]} "
          f"| beats={nbeat}/{len(ctrl_samples)}")


if __name__ == "__main__":
    main()
