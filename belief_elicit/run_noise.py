"""【实验性 · 可整体删除】① 噪声地板:同一张图重复打分 N 次,量测量本身的抖动。

核心设计:把**完全相同的图**打分两次,再对这两次算 mPL。
理想仪器下 mPL(同图→同图) 恒等于 0。实测出来是多少,那就是**噪声地板**——
任何"真实"的 mPL 必须显著高过它才算数。

同时报:
  p_true / logodds(真值 vs 最强干扰项) 的重复标准差
  noise mPL  = 同图两次打分之间的 mPL(全部 C(N,2) 对)
  signal mPL = 原图 vs 遮蔽图 之间的 mPL(全部 N×N 对)
若 signal 的分布与 noise 的分布重叠 ⇒ 该方法测不出这张图的线索泄露。

用法：python -m belief_elicit.run_noise --image 171046893 --n 8 --method allocate
"""
import argparse
import glob
import itertools
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

from belief_elicit.elicit import METHODS
from belief_elicit.run_compare import load_image_and_masks
from clue_leak.masking import mask_solid_from_masks
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km
from run import build_client, load_config

OUT = os.path.join(os.path.dirname(__file__), "noise_results.json")


def build_geometry(labels):
    cache = json.load(open(os.path.join(ROOT, "data", "forward_geocode_cache.json"),
                           encoding="utf-8"))
    coords = {l: cache.get(l) for l in labels if cache.get(l)}
    rep = cluster_representatives(coords, 25.0)
    clusters = sorted(set(rep.values()))
    rc = {c: coords[c] for c in clusters}
    dmat = {}
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dmat[(clusters[a], clusters[b])] = haversine_km(*rc[clusters[a]], *rc[clusters[b]])
    return rep, clusters, (lambda i, j: dmat.get((i, j)) or dmat.get((j, i)))


def mpl(prior, post, rep, clusters, dist):
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


def describe(name, arr):
    a = np.array(arr)
    return (f"  {name:14s} n={len(a):3d}  中位={np.median(a):.4f}  均值={a.mean():.4f}  "
            f"SD={a.std():.4f}  范围=[{a.min():.4f}, {a.max():.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="171046893")
    ap.add_argument("--n", type=int, default=8, help="每种条件的重复打分次数")
    ap.add_argument("--method", default="allocate", choices=["independent", "allocate"])
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="线上管线用 0.0;想看采样抖动可调高")
    args = ap.parse_args()

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    gallery = json.load(open(os.path.join(ROOT, "data", "gallery_labels.json"), encoding="utf-8"))
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    iid = next(os.path.basename(f)[:-5]
               for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
               if os.path.basename(f).startswith(args.image))
    img, masks, names, meta = load_image_and_masks(client, iid)
    tl = labels_cache.get(f"{meta['lat']:.5f},{meta['lon']:.5f}")
    masked_img = mask_solid_from_masks(img, masks)
    rep, clusters, dist = build_geometry(gallery)
    fn = METHODS[args.method]

    print(f"image {iid[:16]} | true={tl} | method={args.method} | temp={args.temperature}")
    print(f"gallery {len(gallery)} labels -> {len(clusters)} clusters | {len(names)} cues | n={args.n}\n")

    def rounds(image, tag):
        out = []
        for i in range(args.n):
            r = fn(image, gallery)          # 每次都是独立 API 调用(本模块不走磁盘缓存)
            out.append(r["prior"])
            top = max((l for l in gallery if l != tl), key=lambda l: r["prior"][l])
            print(f"  [{tag} {i+1}/{args.n}] p_true={r['prior'][tl]:.4f} "
                  f"logodds_vs_{top[:14]}={math.log(r['prior'][tl]/r['prior'][top]):+.3f}", flush=True)
        return out

    full_runs = rounds(img, "full  ")
    mask_runs = rounds(masked_img, "masked")

    # 噪声地板:同一条件下、两次独立打分之间的 mPL(理想应为 0)
    noise_full = [mpl(a, b, rep, clusters, dist) for a, b in itertools.combinations(full_runs, 2)]
    noise_mask = [mpl(a, b, rep, clusters, dist) for a, b in itertools.combinations(mask_runs, 2)]
    # 信号:遮蔽 vs 原图(所有交叉配对)
    signal = [mpl(m, f, rep, clusters, dist) for m in mask_runs for f in full_runs]

    pt_f = [p[tl] for p in full_runs]; pt_m = [p[tl] for p in mask_runs]
    print("\n=== 重复稳定性 ===")
    print(describe("p_true(原图)", pt_f))
    print(describe("p_true(遮蔽)", pt_m))
    print("\n=== mPL:信号 vs 噪声地板 ===")
    print(describe("噪声(原图×原图)", noise_full))
    print(describe("噪声(遮蔽×遮蔽)", noise_mask))
    print(describe("信号(遮蔽×原图)", signal))

    nf = np.array(noise_full + noise_mask); sg = np.array(signal)
    floor = float(nf.max()) if len(nf) else 0.0
    print(f"\n噪声地板(所有同条件配对的最大值) = {floor:.4f}")
    print(f"信号中位数 = {np.median(sg):.4f}  →  信噪比 = {np.median(sg)/(floor or 1e-9):.2f}x")
    above = (sg > floor).mean() * 100 if len(sg) else 0
    print(f"信号高于地板的比例: {above:.0f}%")
    print("判读:信噪比 >> 1 且比例接近 100% 才说明该方法真的测到了线索泄露。")

    json.dump({"image_id": iid, "true_label": tl, "method": args.method,
               "temperature": args.temperature, "n": args.n, "cues": names,
               "p_true_full": pt_f, "p_true_masked": pt_m,
               "noise_full": noise_full, "noise_masked": noise_mask, "signal": signal,
               "noise_floor": floor},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
