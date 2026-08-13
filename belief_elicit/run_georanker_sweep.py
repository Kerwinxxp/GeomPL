"""【实验性 · 可整体删除】GeoRanker(变体 B)全量 sweep:逐单线索 mPL + 精度体检。

对每张有 maskable 线索的图:
  原图打分 → argmax / p_true / rank / km误差 / 国家命中;
  每条单线索遮蔽 → mPL;全部线索遮蔽 → mpl_all。
几何(Geo-I 口径):去重半径 2km —— 只合并真别名(Westminster↔Greater London 1.3km),
保留一切真实近邻对(北京各区 2.4km+、巴黎近郊等):按 Geo-indistinguishability,
近距对的 |Δllr|/d 是经验 ε,细粒度可区分 = 最强泄露,不得被合并抹掉。
**完整分布落盘**(posterior + 每条先验):换任何几何都是纯后处理,无需重新打分。
增量保存:每张图完成即写盘;重启自动跳过已完成的图(断点续跑)。
运行:belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_sweep \
      [--src data/subset100_hires.jsonl]
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from belief_elicit.georanker_belief import score_labels
from belief_elicit.run_georanker_check import build_geometry, mpl
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from geobayes.eval.metrics import haversine_km

OUT = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")
MERGE_KM = 2.0        # Geo-I 口径:仅别名去重,保留真实近邻对


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/subset100_hires.jsonl")
    ap.add_argument("--variant", default="B")
    args = ap.parse_args()

    gv = [g for g in json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"),
                                    encoding="utf-8")) if g["gps"]]
    label_gps = {g["label"]: g["gps"] for g in gv}
    label_country = {g["label"]: g["label"].split(",")[-1].strip() for g in gv}
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)

    # 断点续跑:读已有结果,跳过完成的
    results = []
    if os.path.exists(OUT):
        results = json.load(open(OUT, encoding="utf-8"))
        print(f"resume: 已有 {len(results)} 张,跳过它们", flush=True)
    done = {r["image_id"] for r in results}

    ids = [json.loads(l)["image_id"] for l in
           open(os.path.join(ROOT, args.src), encoding="utf-8")]
    t0 = time.time()
    for n, iid in enumerate(ids, 1):
        if iid in done:
            continue
        cf = os.path.join(ROOT, "cue_extract", "results_sam3", iid + ".json")
        if not os.path.exists(cf):
            continue
        rec = json.load(open(cf, encoding="utf-8"))
        W, H = rec["image_size"]
        cues, cmasks = [], []
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
            cues.append(c); cmasks.append(u)
        if not cmasks:
            continue
        tl = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        if not tl or tl not in label_gps:
            continue
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize((W, H)).convert("RGB")

        post, _ = score_labels(img, gv, variant=args.variant, batch_size=4)
        arg = max(post, key=post.get)
        glat, glon = subset[iid]["lat"], subset[iid]["lon"]
        km_err = haversine_km(glat, glon, label_gps[arg][0], label_gps[arg][1])
        srt = sorted(post, key=post.get, reverse=True)

        per_cue = []
        for c, u in zip(cues, cmasks):
            pri, _ = score_labels(mask_solid_from_masks(img, [u]), gv, variant=args.variant,
                                  batch_size=4)
            per_cue.append({"cue": c["cue"], "category": c.get("category"),
                            "cov": float(u.sum() / (W * H)),
                            "p_true_masked": pri.get(tl, 0.0),
                            "mpl": mpl(pri, post, rep, clusters, dist),
                            "prior": pri})                       # 完整分布落盘
        uall = np.zeros((H, W), bool)
        for u in cmasks:
            uall |= u
        pri, _ = score_labels(mask_solid_from_masks(img, [uall]), gv, variant=args.variant,
                              batch_size=4)
        results.append({"image_id": iid, "true_label": tl,
                        "gt_country": subset[iid].get("gt_country"),
                        "argmax": arg, "p_true": post.get(tl, 0.0),
                        "rank_true": srt.index(tl) + 1, "km_error": km_err,
                        "country_hit": label_country.get(arg) == subset[iid].get("gt_country"),
                        "n_cues": len(cues), "per_cue": per_cue,
                        "p_true_allmask": pri.get(tl, 0.0),
                        "mpl_all": mpl(pri, post, rep, clusters, dist),
                        "posterior": post, "prior_allmask": pri,   # 完整分布落盘
                        "merge_km": MERGE_KM})
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        el = time.time() - t0
        print(f"[{n}/{len(ids)}] {tl.split(',')[0][:16]:16s} p_true={post.get(tl,0):.3f} "
              f"rank={srt.index(tl)+1:3d} km={km_err:6.0f} cues={len(cues)} "
              f"({el/60:.0f}min, 完成 {len(results)})", flush=True)

    km = np.array([r["km_error"] for r in results])
    print(f"\n=== GeoRanker 精度体检 ({len(results)} 图) ===")
    for thr, name in [(1, "street<1km"), (25, "city<25km"), (200, "region<200km"),
                      (750, "country<750km"), (2500, "continent<2500km")]:
        print(f"  {name:18s}: {(km<=thr).mean()*100:5.1f}%")
    print(f"  国家命中率: {np.mean([r['country_hit'] for r in results])*100:.1f}%")
    print(f"  p_true 中位: {np.median([r['p_true'] for r in results]):.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
