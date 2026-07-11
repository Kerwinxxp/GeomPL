"""闭集实验：真值 + K 个硬负样本上的先验↔后验信念分布 + 校准。

流程：
  1. 反向地理编码每张图 GT → "City, Country" 标签（池 = 全子集标签）;
  2. 每张图候选集 = 真值 + K 个地理最近的硬负样本（build_hard_negative_set）;
  3. Controller.run_closed_set → 闭集打分先验 + 线索循环后验（同支撑）;
  4. 聚合 data/closedset_summary.json：先验/后验 MAP 命中率、真值排名、
     真值获得的概率（校准）、熵降、KL。
用法：python scripts/run_closedset.py [--k 5] [--limit N]
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image

from geobayes.analysis.belief import entropy_bits, kl_bits
from geobayes.core.controller import Controller
from geobayes.eval.candidates import build_hard_negative_set
from run import build_client, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
OUTDIR = os.path.join(ROOT, "results_closedset")
SUMMARY = os.path.join(ROOT, "data", "closedset_summary.json")
LABELS = os.path.join(ROOT, "data", "gt_labels_cache.json")
UA = {"User-Agent": "GeoBayes-reproduction/0.1 (academic; xinpengxie2000@gmail.com)"}
_last = [0.0]


def reverse_label(lat, lon, cache):
    key = f"{lat:.5f},{lon:.5f}"
    if key in cache:
        return cache[key]
    w = 1.1 - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()
    url = ("https://nominatim.openstreetmap.org/reverse?"
           + urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2",
                                     "zoom": 10, "accept-language": "en"}))
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20).read())
        a = d.get("address", {})
        place = a.get("city") or a.get("town") or a.get("village") or a.get("county") or a.get("state")
        country = a.get("country")
        label = ", ".join(x for x in (place, country) if x) or None
    except Exception:
        label = None
    cache[key] = label
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="硬负样本数（候选集大小 = k+1）")
    ap.add_argument("--gallery", choices=["hard", "full"], default="hard",
                    help="hard=每图真值+K硬负样本；full=所有图共用数据集全集（固定支撑）")
    ap.add_argument("--prior_only", action="store_true",
                    help="只跑闭集打分先验，跳过线索循环（全集模式默认建议开，省算力）")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    outdir = OUTDIR + ("_gallery" if args.gallery == "full" else "")
    summary_path = SUMMARY.replace("closedset_summary", "gallery_summary") if args.gallery == "full" else SUMMARY
    os.makedirs(outdir, exist_ok=True)

    config = load_config(os.path.join(ROOT, "config.yaml"))
    config["enable_hierarchy"] = False
    config["enable_enhance"] = False
    client = build_client(config)
    controller = Controller(client, config)

    subset = [json.loads(l) for l in open(SUBSET, encoding="utf-8")]
    if args.limit:
        subset = subset[: args.limit]

    # 1) 反向地理编码所有 GT → 标签池
    cache = json.load(open(LABELS, encoding="utf-8")) if os.path.exists(LABELS) else {}
    labeled = []
    for it in subset:
        lbl = reverse_label(it["lat"], it["lon"], cache)
        if lbl:
            labeled.append({**it, "label": lbl})
    json.dump(cache, open(LABELS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"labeled {len(labeled)}/{len(subset)} GT points")

    # full 模式：所有图共用同一份去重后的全集标签（固定支撑）
    full_gallery = sorted({o["label"] for o in labeled})
    if args.gallery == "full":
        print(f"fixed gallery size (unique labels): {len(full_gallery)}")

    # 2-3) 每图闭集运行
    recs, errors = [], []
    t0 = time.time()
    for i, it in enumerate(labeled, 1):
        out = os.path.join(outdir, it["image_id"] + ".json")
        try:
            if os.path.exists(out):
                r = json.load(open(out, encoding="utf-8"))
            else:
                if args.gallery == "full":
                    candidates = full_gallery
                else:
                    pool = [{"label": o["label"], "lat": o["lat"], "lon": o["lon"]}
                            for o in labeled if o["image_id"] != it["image_id"]]
                    candidates = build_hard_negative_set(it["label"], it["lat"], it["lon"],
                                                         pool, k=args.k, seed=42)["candidates"]
                if args.prior_only:
                    img = Image.open(it["path"])
                    if hasattr(client, "prepare"):
                        img = client.prepare(img)
                    scored = client.score_candidates(img, candidates)
                    r = {"prior": {"level": "closed_set", "hypotheses": scored["prior"],
                                   "raw_scores": scored["raw_scores"], "candidates": candidates},
                         "trajectory": [], "events": [],
                         "final_posterior": {"level": "closed_set", "hypotheses": scored["prior"]},
                         "map_estimate": max(scored["prior"], key=scored["prior"].get)}
                else:
                    r = controller.run_closed_set(Image.open(it["path"]), candidates)
                r["true_label"] = it["label"]
                r["candidates"] = candidates
                json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            recs.append((it, r))
            pr, fp = r["prior"]["hypotheses"], r["final_posterior"]["hypotheses"]
            tl = it["label"]
            print(f"[{i}/{len(labeled)}] true={tl[:26]:26s} "
                  f"prior_p={pr.get(tl,0):.2f} post_p={fp.get(tl,0):.2f} "
                  f"MAP={'HIT' if max(fp,key=fp.get)==tl else 'miss'} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            errors.append({"image_id": it["image_id"], "error": str(e)})
            print(f"[{i}] ERROR {it['image_id'][:26]}: {e}", flush=True)

    # 4) 聚合
    def rank_of_true(dist, tl):
        return sorted(dist, key=dist.get, reverse=True).index(tl) + 1
    n = len(recs)
    prior_hit = post_hit = 0
    prior_p, post_p, kls, dH, prior_rank, post_rank = [], [], [], [], [], []
    calib = []   # (post prob of MAP, correct?)
    for it, r in recs:
        pr, fp, tl = r["prior"]["hypotheses"], r["final_posterior"]["hypotheses"], it["label"]
        prior_hit += (max(pr, key=pr.get) == tl)
        post_hit += (max(fp, key=fp.get) == tl)
        prior_p.append(pr.get(tl, 0.0)); post_p.append(fp.get(tl, 0.0))
        prior_rank.append(rank_of_true(pr, tl)); post_rank.append(rank_of_true(fp, tl))
        kls.append(kl_bits(fp, pr)); dH.append(entropy_bits(pr) - entropy_bits(fp))
        mp = max(fp, key=fp.get)
        calib.append((fp[mp], mp == tl))
    # 校准分箱
    bins = {}
    for p, ok in calib:
        b = min(4, int(p * 5))
        bins.setdefault(b, []).append(ok)
    reliability = {f"{b*20}-{b*20+20}%": {"n": len(v), "acc": round(sum(v) / len(v), 3)}
                   for b, v in sorted(bins.items())}
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    summary = {
        "gallery_mode": args.gallery, "prior_only": args.prior_only,
        "k_hard_negatives": args.k,
        "candidate_set_size": (len(full_gallery) if args.gallery == "full" else args.k + 1),
        "random_baseline_note": f"1/{len(full_gallery) if args.gallery=='full' else args.k+1}",
        "model": config.get("model"), "n": n, "n_errors": len(errors),
        "prior_map_accuracy": round(prior_hit / n, 3) if n else None,
        "posterior_map_accuracy": round(post_hit / n, 3) if n else None,
        "random_baseline": round(1 / (len(full_gallery) if args.gallery == "full" else args.k + 1), 3),
        "mean_prior_prob_true": avg(prior_p), "mean_posterior_prob_true": avg(post_p),
        "mean_prior_rank_true": avg(prior_rank), "mean_posterior_rank_true": avg(post_rank),
        "mean_kl_post_vs_prior_bits": avg(kls), "mean_entropy_drop_bits": avg(dH),
        "reliability_by_confidence": reliability,
    }
    # 逐图分布落盘（供统计图）
    dist_dump = [{"image_id": it["image_id"], "true": it["label"],
                  "prior_prob_true": r["prior"]["hypotheses"].get(it["label"], 0.0),
                  "post_prob_true": r["final_posterior"]["hypotheses"].get(it["label"], 0.0),
                  "prior_entropy": entropy_bits(r["prior"]["hypotheses"]),
                  "post_entropy": entropy_bits(r["final_posterior"]["hypotheses"]),
                  "prior_rank": rank_of_true(r["prior"]["hypotheses"], it["label"]),
                  "post_rank": rank_of_true(r["final_posterior"]["hypotheses"], it["label"])}
                 for it, r in recs]
    summary["per_image"] = dist_dump
    json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n===== CLOSED-SET SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
