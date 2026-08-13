"""等面积对照:证明逐线索 mPL 是"内容特异",不是灰块/面积伪影。

对每条真线索,把它的 SAM 掩码**平移到随机非线索位置**(形状+面积完全一样,只是盖到
无关内容上)→ 遮灰 → 打分 → mPL。每条线索取 N 个随机位置求均值。
若 真线索 mPL ≫ 对照 mPL,则泄露来自该线索的内容,而非"遮了这么大一块"。

真线索 mPL 直接复用 combo2 里存好的 prior(零额外 API);只有对照要现打分。
用法：python -m clue_leak.control_area [--nctrl 3]
"""
import argparse
import glob
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
from PIL import Image

from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km
from run import build_client, load_config

COMBODIR = os.path.join(os.path.dirname(__file__), "combo2_sam3_results")
CUEDIR = os.path.join(ROOT, "cue_extract", "results_sam3")
GALLERY = os.path.join(ROOT, "data", "gallery_labels.json")
FWD = os.path.join(ROOT, "data", "forward_geocode_cache.json")
OUT = os.path.join(os.path.dirname(__file__), "control_area_results.json")


def build_geometry(all_labels):
    cache = json.load(open(FWD, encoding="utf-8"))
    coords = {l: cache.get(l) for l in all_labels if cache.get(l)}
    rep = cluster_representatives(coords, 25.0)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mean_mpl(prior, post, rep, clusters, dist):
    pr, po = merge_distribution(prior, rep), merge_distribution(post, rep)
    keys = [k for k in clusters if pr.get(k, 0) > 0 and po.get(k, 0) > 0]
    llr = {k: math.log(po[k] / pr[k]) for k in keys}
    vals = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = dist(keys[a], keys[b])
            if d and d > 0:
                vals.append(abs(llr[keys[a]] - llr[keys[b]]) / d * 1000)
    return sum(vals) / len(vals) if vals else 0.0


def translate_mask(mask, dx, dy):
    """把 mask 里 True 的形状整体平移 (dx,dy),越界部分丢弃;面积可能略减(越界),形状保持。"""
    H, W = mask.shape
    out = np.zeros_like(mask)
    ys, xs = np.nonzero(mask)
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
    out[ny[ok], nx[ok]] = True
    return out


def sample_control(cue_mask, cue_union, rng, tries=40):
    """找一个平移量:平移后与所有线索并集重叠尽量小、且不越界太多。返回平移后的 mask。"""
    H, W = cue_mask.shape
    ys, xs = np.nonzero(cue_mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    area = int(cue_mask.sum())
    best, best_bad = None, 1e18
    for _ in range(tries):
        dx = rng.integers(-x0, W - 1 - x1) if x1 - x0 < W - 1 else 0
        dy = rng.integers(-y0, H - 1 - y1) if y1 - y0 < H - 1 else 0
        t = translate_mask(cue_mask, int(dx), int(dy))
        if t.sum() < 0.9 * area:            # 越界太多,弃
            continue
        overlap = int((t & cue_union).sum())
        if overlap < best_bad:
            best_bad, best = overlap, t
        if overlap == 0:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nctrl", type=int, default=3, help="每条线索的随机对照位置数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ids", default="", help="只跑这些 image_id(逗号分隔);留空=全部")
    ap.add_argument("--out", default=OUT, help="结果输出路径")
    args = ap.parse_args()
    only = {x for x in args.ids.split(",") if x}

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    gallery = json.load(open(GALLERY, encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it   # 覆盖:后加载(subset_sample 高清)优先

    allc = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(COMBODIR, "*.json")))]
    rep, clusters, dist = build_geometry({k for r in allc for k in r["posterior"]})  # 几何用全集,保证可比
    combos = [r for r in allc if (not only) or r["image_id"] in only]
    rng = np.random.default_rng(args.seed)

    out = []
    for r in combos:
        iid = r["image_id"]
        post = r["posterior"]
        cue_rec = json.load(open(os.path.join(CUEDIR, iid + ".json"), encoding="utf-8"))
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = client.prepare(Image.open(p))
        W, Hh = img.size
        # 每条 maskable 线索的掩码并集(= 平移对照要避开的区域)
        mcues = [c for c in cue_rec["geo_privacy_cues"]
                 if c.get("maskable") and any((not i.get("degenerate")) and i.get("mask_rle") for i in c["instances"])]
        cue_masks = [np.zeros((Hh, W), bool) for _ in mcues]
        for ci, c in enumerate(mcues):
            for i in c["instances"]:
                if (not i.get("degenerate")) and i.get("mask_rle"):
                    m = rle_to_mask(i["mask_rle"])
                    if m.shape == (Hh, W):
                        cue_masks[ci] |= m
        union = np.zeros((Hh, W), bool)
        for m in cue_masks:
            union |= m

        # 真线索 mPL(复用 combo 里 subset==[k] 的 prior)
        single = {tuple(c["subset"])[0]: c for c in r["combos"] if len(c["subset"]) == 1}
        print(f"=== {r['true_label'].split(',')[0]} ({len(mcues)} cues) ===", flush=True)
        for ci, c in enumerate(mcues):
            real = mean_mpl(single[ci]["prior"], post, rep, clusters, dist) if ci in single else None
            ctrl_mpls = []
            for _ in range(args.nctrl):
                t = sample_control(cue_masks[ci], union, rng)
                if t is None:
                    continue
                pr = client.score_candidates(mask_solid_from_masks(img, [t]), gallery)["prior"]
                ctrl_mpls.append(mean_mpl(pr, post, rep, clusters, dist))
            cm = float(np.mean(ctrl_mpls)) if ctrl_mpls else None
            area_pct = round(cue_masks[ci].sum() / (W * Hh) * 100, 1)
            out.append({"image": iid, "place": r["true_label"].split(",")[0],
                        "cue": c["cue"], "area_pct": area_pct,
                        "real_mpl": real, "ctrl_mpl": cm, "ctrl_samples": ctrl_mpls})
            rr = f"{real:.3f}" if real is not None else "  -  "
            cc = f"{cm:.3f}" if cm is not None else "  -  "
            print(f"  {c['cue'][:26]:26s} area={area_pct:4.1f}%  real={rr}  ctrl={cc}", flush=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    reals = [o["real_mpl"] for o in out if o["real_mpl"] is not None]
    ctrls = [o["ctrl_mpl"] for o in out if o["ctrl_mpl"] is not None]
    print(f"\n=== 汇总 ({len(out)} cues) ===")
    print(f"  真线索 mPL  中位={np.median(reals):.3f} 均值={np.mean(reals):.3f}")
    print(f"  等面积对照   中位={np.median(ctrls):.3f} 均值={np.mean(ctrls):.3f}")
    higher = sum(1 for o in out if o["real_mpl"] and o["ctrl_mpl"] and o["real_mpl"] > o["ctrl_mpl"])
    print(f"  真线索 > 对照 的线索数: {higher}/{len(out)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
