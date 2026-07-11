"""研究目标产出：单层细粒度（城市/地点级）的先验↔后验信念分布。

对每张图输出 results_singlelevel/<id>.json（同支撑的 prior 与 final_posterior）。
聚合 data/belief_summary.json：
  - 每图先验熵、后验熵、信息增益 KL(final||prior)、逐线索 log2 证据权重；
  - 细粒度定位：MAP 候选名 → 前向地理编码 → 与 GT 的 haversine → 五档阈值准确率。
用法：python scripts/run_beliefs.py [--limit N] [--granularity city|place]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from geobayes.analysis.belief import entropy_bits, kl_bits
from geobayes.eval.geocode import forward_geocode
from geobayes.eval.metrics import PAPER_THRESHOLDS_KM, localization_distance_km, threshold_accuracy
from run import load_config, run_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
OUTDIR = os.path.join(ROOT, "results_singlelevel")
SUMMARY = os.path.join(ROOT, "data", "belief_summary.json")
GEO_CACHE = os.path.join(ROOT, "data", "forward_geocode_cache.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--granularity", default=None, help="覆盖 config 的 hypothesis_granularity")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    config = load_config(os.path.join(ROOT, "config.yaml"))
    config["enable_hierarchy"] = False
    config["enable_enhance"] = config.get("enable_enhance", False)
    if args.granularity:
        config["hypothesis_granularity"] = args.granularity
    gran = config.get("hypothesis_granularity", "city")

    subset = [json.loads(l) for l in open(SUBSET, encoding="utf-8")]
    if args.limit:
        subset = subset[: args.limit]

    records, errors = [], []
    t0 = time.time()
    for i, it in enumerate(subset, 1):
        out = os.path.join(OUTDIR, it["image_id"] + ".json")
        try:
            r = (json.load(open(out, encoding="utf-8")) if os.path.exists(out)
                 else run_image(it["path"], config=config, output_path=out, search_client=None))
            records.append((it, r))
            print(f"[{i}/{len(subset)}] {it['image_id'][:26]} "
                  f"MAP={r['map_estimate'][:34]} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            errors.append({"image_id": it["image_id"], "error": str(e)})
            print(f"[{i}] ERROR {it['image_id'][:26]}: {e}", flush=True)

    # ---- 信念度量（同支撑，天然可算） ----
    kls, dH, Hp, Hf = [], [], [], []
    for it, r in records:
        pr, fp = r["prior"]["hypotheses"], r["final_posterior"]["hypotheses"]
        Hp.append(entropy_bits(pr)); Hf.append(entropy_bits(fp))
        if set(pr) == set(fp):
            kls.append(kl_bits(fp, pr)); dH.append(entropy_bits(pr) - entropy_bits(fp))

    # ---- 细粒度定位准确率（论文距离口径） ----
    cache = json.load(open(GEO_CACHE, encoding="utf-8")) if os.path.exists(GEO_CACHE) else {}
    last = [0.0]
    def geocoder(name):
        if name not in cache:
            w = 1.1 - (time.time() - last[0])
            if w > 0: time.sleep(w)
            last[0] = time.time()
        return forward_geocode(name, cache=cache)
    dists = [localization_distance_km(r, it["lat"], it["lon"], geocoder) for it, r in records]
    json.dump(cache, open(GEO_CACHE, "w", encoding="utf-8"))
    acc = {t: round(v * 100, 1) for t, v in threshold_accuracy(dists).items()} if dists else {}

    n = len(records)
    summary = {
        "granularity": gran, "model": config.get("model"),
        "n_run": n, "n_errors": len(errors),
        "mean_prior_entropy_bits": round(sum(Hp) / n, 3) if n else None,
        "mean_posterior_entropy_bits": round(sum(Hf) / n, 3) if n else None,
        "mean_entropy_drop_bits": round(sum(dH) / len(dH), 3) if dH else None,
        "mean_info_gain_kl_bits": round(sum(kls) / len(kls), 3) if kls else None,
        "n_same_support": len(kls),
        "distance_accuracy_pct": acc, "n_distance_scored": len(dists),
    }
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n===== BELIEF SUMMARY (single-level, granularity=%s) =====" % gran)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("distance thresholds order: " + " / ".join(f"<{t}km" for t in PAPER_THRESHOLDS_KM))


if __name__ == "__main__":
    main()
