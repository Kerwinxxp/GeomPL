"""把 47 候选全集按 25km 合并近重复 → 干净固定全集，重算 mPL / 交叉熵 / 分布。

不重跑模型：对已有 results_closedset_gallery 的先验/后验做候选合并（概率相加），
真值标签重映射到其簇代表。输出 data/gallery_clean_summary.json + 逐图干净分布。
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from geobayes.analysis.leakage import mpl_sup
from geobayes.eval.candidates import cluster_representatives, merge_distribution
from geobayes.eval.metrics import haversine_km

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
GEO_CACHE = os.path.join(ROOT, "data", "forward_geocode_cache.json")
RES = os.path.join(ROOT, "results_closedset_gallery")
OUT = os.path.join(ROOT, "data", "gallery_clean_summary.json")
MIN_DIST = 25.0


def ce(dist, true):
    return -math.log(max(dist.get(true, 1e-12), 1e-12))


def main():
    subset = {json.loads(l)["image_id"]: json.loads(l) for l in open(SUBSET, encoding="utf-8")}
    R = {i: json.load(open(os.path.join(RES, i + ".json"), encoding="utf-8"))
         for i in subset if os.path.exists(os.path.join(RES, i + ".json"))}
    cache = json.load(open(GEO_CACHE, encoding="utf-8"))

    labels = sorted({l for r in R.values() for l in r["prior"]["hypotheses"]})
    coords = {l: cache[l] for l in labels if cache.get(l)}
    rep = cluster_representatives(coords, MIN_DIST)
    clusters = sorted(set(rep.values()))
    print(f"raw gallery {len(labels)} labels → {len(clusters)} clean clusters "
          f"(merged {len(labels)-len(clusters)} near-duplicates within {MIN_DIST}km)")
    merged_pairs = {l: rep[l] for l in rep if rep[l] != l}
    for l, r in merged_pairs.items():
        print(f"  merged: {l}  →  {r}")

    rep_coord = {c: coords[c] for c in clusters}
    dmat = {}
    cl = list(clusters)
    for a in range(len(cl)):
        for b in range(a + 1, len(cl)):
            i, j = cl[a], cl[b]
            dmat[(i, j)] = haversine_km(*rep_coord[i], *rep_coord[j])
    def distance(i, j):
        return dmat.get((i, j)) or dmat.get((j, i))

    per, mpls, ce_pr, ce_po, dH = [], [], [], [], []
    prior_hit = post_hit = 0
    for i, r in R.items():
        tl_raw = r.get("true_label")
        tl = rep.get(tl_raw, tl_raw)
        pr = merge_distribution(r["prior"]["hypotheses"], rep)
        po = merge_distribution(r["final_posterior"]["hypotheses"], rep)
        if tl not in pr:      # 真值无坐标/未入簇：跳过
            continue
        m = mpl_sup(pr, po, distance)
        Hp = -sum(v * math.log2(v) for v in pr.values() if v > 0)
        Hf = -sum(v * math.log2(v) for v in po.values() if v > 0)
        prior_hit += (max(pr, key=pr.get) == tl)
        post_hit += (max(po, key=po.get) == tl)
        ce_pr.append(ce(pr, tl)); ce_po.append(ce(po, tl)); dH.append(Hp - Hf)
        rec = {"image_id": i, "true": tl, "prior_prob_true": pr[tl], "post_prob_true": po[tl],
               "prior_entropy": Hp, "post_entropy": Hf}
        if m["mpl"] is not None:
            mpls.append(m["mpl"] * 1000)
            rec["mpl_nats_per_1000km"] = m["mpl"] * 1000
            rec["sup_pair"] = m["pair"]; rec["sup_pair_km"] = round(distance(*m["pair"]), 1)
        per.append(rec)

    n = len(per)
    avg = lambda xs: round(statistics.mean(xs), 4) if xs else None
    summary = {
        "clean_gallery_size": len(clusters), "raw_size": len(labels),
        "merged_near_duplicates": merged_pairs, "min_dist_km": MIN_DIST, "n_images": n,
        "prior_map_accuracy": round(prior_hit / n, 3), "posterior_map_accuracy": round(post_hit / n, 3),
        "random_baseline": round(1 / len(clusters), 3),
        "cross_entropy_prior": avg(ce_pr), "cross_entropy_posterior": avg(ce_po),
        "cross_entropy_uniform": round(math.log(len(clusters)), 3),
        "mean_entropy_drop_bits": avg(dH),
        "mpl_median_nats_per_1000km": round(statistics.median(mpls), 2) if mpls else None,
        "mpl_mean_nats_per_1000km": round(statistics.mean(mpls), 2) if mpls else None,
        "mpl_max_nats_per_1000km": round(max(mpls), 2) if mpls else None,
        "per_image": sorted(per, key=lambda p: -p.get("mpl_nats_per_1000km", 0)),
    }
    json.dump(summary, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n===== CLEAN GALLERY SUMMARY =====")
    for k in ["clean_gallery_size", "n_images", "prior_map_accuracy", "posterior_map_accuracy",
              "cross_entropy_uniform", "cross_entropy_prior", "cross_entropy_posterior",
              "mean_entropy_drop_bits", "mpl_median_nats_per_1000km", "mpl_mean_nats_per_1000km",
              "mpl_max_nats_per_1000km"]:
        print(f"  {k}: {summary[k]}")
    print("\ntop-3 leakage pairs (clean, no floor):")
    for p in summary["per_image"][:3]:
        if "sup_pair" in p:
            print(f"  {p['mpl_nats_per_1000km']:.1f}/1000km  {p['sup_pair'][0][:20]} ↔ "
                  f"{p['sup_pair'][1][:20]} ({p['sup_pair_km']}km)")


if __name__ == "__main__":
    main()
