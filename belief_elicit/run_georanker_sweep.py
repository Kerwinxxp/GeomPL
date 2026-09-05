"""GeoRanker (prompt variant B) full sweep: per-single-cue mPL + an accuracy health check.

For every image with maskable cues:
  score the clean image -> argmax / p_true / rank / km error / country hit;
  mask each single cue -> mPL; mask every cue -> mpl_all.
Geometry (Geo-I convention): 2 km dedup radius — merge true aliases only (Westminster <->
Greater London, 1.3 km) and keep every genuine near-neighbour pair (Beijing districts at
2.4 km+, the Paris suburbs, ...): under Geo-indistinguishability the |dllr|/d of a close
pair is the empirical epsilon, so fine-grained distinguishability is the strongest leak and
must not be merged away.
**Full distributions are persisted** (posterior + every prior): changing the geometry is
then pure post-processing, with no re-scoring.
Incremental saves: each image is written as it finishes, and a restart skips what is done.

Run: belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_sweep
     [--src data/subset100_hires.jsonl]
"""
import argparse
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

from belief_elicit.cues import cue_masks_of
from belief_elicit.geometry import build_geometry, haversine_km, mpl
from belief_elicit.georanker_belief import score_labels
from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.results import (SAM3_DIR, SWEEP as OUT, image_path, load_gallery,
                                   load_subsets)

MERGE_KM = 2.0        # Geo-I convention: alias dedup only, genuine near-neighbours kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/subset100_hires.jsonl")
    ap.add_argument("--variant", default="B")
    args = ap.parse_args()

    gv = load_gallery()
    label_gps = {g["label"]: g["gps"] for g in gv}
    label_country = {g["label"]: g["label"].split(",")[-1].strip() for g in gv}
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    subset = load_subsets()
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)

    # resume: read what is already there and skip those images
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
        got = cue_masks_of(SAM3_DIR, iid)
        if got is None:
            continue
        cues, cats, cmasks, (W, H) = got
        if not cmasks:
            continue
        tl = labels_cache.get(f"{subset[iid]['lat']:.5f},{subset[iid]['lon']:.5f}")
        if not tl or tl not in label_gps:
            continue
        img = Image.open(image_path(subset[iid])).resize((W, H)).convert("RGB")

        post, _ = score_labels(img, gv, variant=args.variant, batch_size=4)
        arg = max(post, key=post.get)
        glat, glon = subset[iid]["lat"], subset[iid]["lon"]
        km_err = haversine_km(glat, glon, label_gps[arg][0], label_gps[arg][1])
        srt = sorted(post, key=post.get, reverse=True)

        per_cue = []
        for name, cat, u in zip(cues, cats, cmasks):
            pri, _ = score_labels(mask_solid_from_masks(img, [u]), gv, variant=args.variant,
                                  batch_size=4)
            per_cue.append({"cue": name, "category": cat,
                            "cov": float(u.sum() / (W * H)),
                            "p_true_masked": pri.get(tl, 0.0),
                            "mpl": mpl(pri, post, rep, clusters, dist),
                            "prior": pri})                       # full distribution persisted
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
                        "posterior": post, "prior_allmask": pri,   # full distributions persisted
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
