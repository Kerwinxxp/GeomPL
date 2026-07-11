"""在闭集先验/后验分布上计算 mPL（Chen et al. 2026）。

1. 预计算候选位置两两地理距离（前向地理编码每个标签 → 坐标 → haversine 矩阵，缓存）;
2. 逐图 mPL = sup over 候选对 |LLR_i − LLR_j| / d_ij;
3. 聚合分布 + 保存 data/mpl_results.json。
用法：python scripts/compute_mpl.py [--results results_closedset_gallery]
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from geobayes.analysis.leakage import mpl_sup
from geobayes.eval.geocode import forward_geocode
from geobayes.eval.metrics import haversine_km

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
GEO_CACHE = os.path.join(ROOT, "data", "forward_geocode_cache.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_closedset_gallery")
    ap.add_argument("--out", default="data/mpl_results.json")
    ap.add_argument("--min_dist", type=float, default=0.0,
                    help="排除距离 < min_dist km 的候选对（去除同城近重复标签导致的 mPL 爆炸）")
    args = ap.parse_args()

    subset = {json.loads(l)["image_id"]: json.loads(l) for l in open(SUBSET, encoding="utf-8")}
    rdir = os.path.join(ROOT, args.results)
    R = {i: json.load(open(os.path.join(rdir, i + ".json"), encoding="utf-8"))
         for i in subset if os.path.exists(os.path.join(rdir, i + ".json"))}

    # 全部出现过的候选标签
    labels = sorted({l for r in R.values() for l in r["prior"]["hypotheses"]})
    print(f"{len(R)} images, {len(labels)} unique candidate labels")

    # 1) 前向地理编码 → 坐标（缓存 + 限速）
    cache = json.load(open(GEO_CACHE, encoding="utf-8")) if os.path.exists(GEO_CACHE) else {}
    last = [0.0]
    coord = {}
    for lbl in labels:
        if lbl not in cache:
            w = 1.1 - (time.time() - last[0])
            if w > 0:
                time.sleep(w)
            last[0] = time.time()
        c = forward_geocode(lbl, cache=cache)
        if c:
            coord[lbl] = c
    json.dump(cache, open(GEO_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"geocoded {len(coord)}/{len(labels)} labels")

    # 2) 预计算距离矩阵
    dmat = {}
    ls = list(coord)
    for a in range(len(ls)):
        for b in range(a + 1, len(ls)):
            i, j = ls[a], ls[b]
            dmat[(i, j)] = haversine_km(coord[i][0], coord[i][1], coord[j][0], coord[j][1])
    def distance(i, j):
        d = dmat.get((i, j)) or dmat.get((j, i))
        if d is None or d < args.min_dist:
            return None
        return d

    # 3) 逐图 mPL
    per = []
    for i, r in R.items():
        pr, fp = r["prior"]["hypotheses"], r["final_posterior"]["hypotheses"]
        res = mpl_sup(pr, fp, distance)
        if res["mpl"] is None:
            continue
        pi, pj = res["pair"]
        per.append({"image_id": i, "true": r.get("true_label"),
                    "mpl_nats_per_km": res["mpl"],
                    "mpl_nats_per_1000km": res["mpl"] * 1000,
                    "sup_pair": [pi, pj], "sup_pair_km": round(distance(pi, pj), 1),
                    "n_pairs": res["n_pairs"]})

    vals = [p["mpl_nats_per_km"] for p in per]
    summary = {
        "mpl_definition": "Chen et al. 2026 Def 2.2/2.3; sup over candidate pairs of |dLLR|/d_ij; unit nats/km",
        "results_dir": args.results, "n_images": len(per),
        "mean_mpl_nats_per_km": round(statistics.mean(vals), 6) if vals else None,
        "median_mpl_nats_per_km": round(statistics.median(vals), 6) if vals else None,
        "max_mpl_nats_per_km": round(max(vals), 6) if vals else None,
        "min_mpl_nats_per_km": round(min(vals), 6) if vals else None,
        "mean_mpl_nats_per_1000km": round(statistics.mean(vals) * 1000, 3) if vals else None,
        "per_image": sorted(per, key=lambda p: -p["mpl_nats_per_km"]),
    }
    json.dump(summary, open(os.path.join(ROOT, args.out), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\n===== mPL SUMMARY (nats/km) =====")
    for k in ["n_images", "mean_mpl_nats_per_km", "median_mpl_nats_per_km",
              "max_mpl_nats_per_km", "mean_mpl_nats_per_1000km"]:
        print(f"  {k}: {summary[k]}")
    print("\ntop-5 highest-leakage images (sup pair):")
    for p in summary["per_image"][:5]:
        print(f"  mPL={p['mpl_nats_per_1000km']:.2f}/1000km  true={str(p['true'])[:22]:22s} "
              f"pair=({p['sup_pair'][0][:16]} | {p['sup_pair'][1][:16]}) d={p['sup_pair_km']}km")


if __name__ == "__main__":
    main()
