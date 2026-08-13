"""【实验性 · 可整体删除】GeoCLIP 铺 200 张:逐线索 mPL + 精度体检。

对每张有 maskable 线索的图:
  posterior = GeoCLIP(原图) over gallery_v2(138);
  精度信号:argmax 标签、p_true、真值 rank、km 误差(argmax坐标 vs GT坐标)、国家是否命中;
  逐单线索:遮该线索 → prior → mpl(prior,post)(边际泄露),带 category;
  全遮:所有线索一起遮 → mpl。
聚合:①精度体检(km 误差分档 + 国家命中率 + p_true 分布);②按类别的逐线索泄露。
运行:cue_extract/.venv/Scripts/python.exe -m belief_elicit.run_geoclip_sweep200
"""
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

from belief_elicit.geoclip_belief import get_model, score_gallery
from clue_leak.masking import mask_solid_from_masks
from cue_extract.rle import rle_to_mask
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

OUT = os.path.join(os.path.dirname(__file__), "geoclip_sweep200_results.json")


def build_geometry(gv):
    coords = {g["label"]: g["gps"] for g in gv if g["gps"]}
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


def main():
    gv = json.load(open(os.path.join(ROOT, "data", "gallery_v2.json"), encoding="utf-8"))
    gv = [g for g in gv if g["gps"]]
    labels = [g["label"] for g in gv]
    coords = [g["gps"] for g in gv]
    label_country = {g["label"]: g["label"].split(",")[-1].strip() for g in gv}
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "subset*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            it = json.loads(line); subset[it["image_id"]] = it
    rep, clusters, dist = build_geometry(gv)
    model = get_model()

    ids = [json.loads(l)["image_id"] for l in
           open(os.path.join(ROOT, "data", "subset200_hires.jsonl"), encoding="utf-8")]
    results = []
    t0 = __import__("time").time()
    for n, iid in enumerate(ids, 1):
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
        p = subset[iid]["path"]; p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize((W, H)).convert("RGB")
        tl = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        glat, glon = subset[iid]["lat"], subset[iid]["lon"]

        pd = score_gallery(img, coords, model=model)
        post = {labels[i]: pd[i] for i in range(len(labels))}
        arg = max(post, key=post.get)
        arg_gps = coords[labels.index(arg)]
        km_err = haversine_km(glat, glon, arg_gps[0], arg_gps[1])
        srt = sorted(post, key=post.get, reverse=True)
        rank = srt.index(tl) + 1 if tl in post else -1
        country_hit = (label_country.get(arg) == subset[iid].get("gt_country"))

        per_cue = []
        for c, u in zip(cues, cmasks):
            prd = score_gallery(mask_solid_from_masks(img, [u]), coords, model=model)
            prior = {labels[i]: prd[i] for i in range(len(labels))}
            per_cue.append({"cue": c["cue"], "category": c.get("category"),
                            "cov": float(u.sum() / (W * H)),
                            "mpl": mpl(prior, post, rep, clusters, dist)})
        # 全遮
        uall = np.zeros((H, W), bool)
        for u in cmasks:
            uall |= u
        prd = score_gallery(mask_solid_from_masks(img, [uall]), coords, model=model)
        mpl_all = mpl({labels[i]: prd[i] for i in range(len(labels))}, post, rep, clusters, dist)

        results.append({"image_id": iid, "true_label": tl, "gt_country": subset[iid].get("gt_country"),
                        "argmax": arg, "p_true": post.get(tl, 0.0), "rank_true": rank,
                        "km_error": km_err, "country_hit": country_hit,
                        "n_cues": len(cues), "per_cue": per_cue, "mpl_all": mpl_all})
        if n % 20 == 0 or n == len(ids):
            print(f"[{n}/{len(ids)}] {len(results)} imgs done ({__import__('time').time()-t0:.0f}s)", flush=True)

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    km = np.array([r["km_error"] for r in results])
    print(f"\n=== 精度体检 ({len(results)} 图) ===")
    for thr, name in [(1, "street<1km"), (25, "city<25km"), (200, "region<200km"),
                      (750, "country<750km"), (2500, "continent<2500km")]:
        print(f"  {name:18s}: {(km<=thr).mean()*100:5.1f}%")
    print(f"  国家命中率: {np.mean([r['country_hit'] for r in results])*100:.1f}%")
    print(f"  p_true 中位: {np.median([r['p_true'] for r in results]):.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
