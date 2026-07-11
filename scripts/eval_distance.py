"""统一距离阈值评测（论文 Table 1 口径）：对多个结果目录用同一 geocoder 打分。

对每个结果目录（zero-shot / 搜索关 v1 / 搜索开 v2）：
  层级地名 → 前向地理编码 → 与 GT 的 haversine → 五档阈值准确率（2500/750/200/25/1km）。
共享 geocode 缓存 + Nominatim 限速 1req/s。输出对照表 + data/distance_eval.json。
用法：python scripts/eval_distance.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geobayes.eval.geocode import forward_geocode
from geobayes.eval.metrics import (PAPER_THRESHOLDS_KM, localization_distance_km,
                                    threshold_accuracy)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
GEO_CACHE = os.path.join(ROOT, "data", "forward_geocode_cache.json")
OUT = os.path.join(ROOT, "data", "distance_eval.json")

DIRS = [("zero-shot (baseline)", "results_zeroshot"),
        ("GeoBayes v1 (search OFF)", "results_baseline"),
        ("GeoBayes v2 (search ON)", "results")]


def make_geocoder(cache):
    last = [0.0]
    def geocoder(name):
        if name not in cache:              # 仅未缓存时限速
            wait = 1.1 - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
        return forward_geocode(name, cache=cache)
    return geocoder


def eval_dir(d, subset, geocoder):
    # 公平口径（审计修复）：无法地理编码 = 未命中，仍计入分母；
    # 只有"该方法没跑出结果文件"才不计入分母。
    dists, n_missing_file, n_ungeocodable = [], 0, 0
    for it in subset:
        p = os.path.join(ROOT, d, it["image_id"] + ".json")
        if not os.path.exists(p):
            n_missing_file += 1
            continue
        r = json.load(open(p, encoding="utf-8"))
        dist = localization_distance_km(r, it["lat"], it["lon"], geocoder)
        if dist is None:
            n_ungeocodable += 1
        dists.append(dist)   # None 也入列，threshold_accuracy 记为未命中
    n = len(dists)
    acc = ({t: round(v * 100, 1) for t, v in threshold_accuracy(dists).items()}
           if n else {})
    finite = sorted(x for x in dists if x is not None)
    return {"n_attempted": n, "n_missing_file": n_missing_file,
            "n_ungeocodable": n_ungeocodable, "accuracy_pct": acc,
            "median_km_geocodable": round(finite[len(finite) // 2], 1) if finite else None}


def main():
    subset = [json.loads(l) for l in open(SUBSET, encoding="utf-8")]
    cache = json.load(open(GEO_CACHE, encoding="utf-8")) if os.path.exists(GEO_CACHE) else {}
    geocoder = make_geocoder(cache)

    report = {}
    for label, d in DIRS:
        if not os.path.isdir(os.path.join(ROOT, d)):
            continue
        report[label] = eval_dir(d, subset, geocoder)
        json.dump(cache, open(GEO_CACHE, "w", encoding="utf-8"))   # 增量落盘

    json.dump(report, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    ths = PAPER_THRESHOLDS_KM
    print(f"{'method':30s} {'n':>4s} " + " ".join(f'<{t}km'.rjust(8) for t in ths))
    for label, r in report.items():
        a = r["accuracy_pct"]
        print(f"{label:30s} {r['n_attempted']:4d} " +
              " ".join(f"{a.get(t,0):7.1f}%" for t in ths) +
              f"   (ungeocodable={r['n_ungeocodable']})")
    print("\nPaper Qwen2.5-VL: zero-shot 83.8/70.4/51.1/31.0/5.1 | "
          "GeoBayes 85.9/73.7/53.6/34.7/6.3 (n=2997)")


if __name__ == "__main__":
    main()
