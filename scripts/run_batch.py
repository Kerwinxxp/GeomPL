"""批量跑子集 + 聚合统计（PLAN §Phase3 步骤 4）。

每图输出 results/<image_id>.json（全轨迹）；聚合输出 data/batch_summary.json：
- 先验 Top-1/3/5 国家 recall（Table 3 口径；注意 gpt-4o 无论文锚点可对拍）
- 支撑覆盖率（GT 国家是否在先验支撑集内）
- 最终 MAP 国家准确率、先验/后验熵、KL、事件计数、(c,alpha) 分布（α 塌缩检测）
用法：python scripts/run_batch.py [--limit N]
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geobayes.analysis.belief import entropy_bits, kl_bits
from geobayes.eval.geocode import canonicalize_country
from geobayes.eval.metrics import top_k_recall_from_prior
from run import load_config, run_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = os.path.join(ROOT, "data", "subset50.jsonl")
RESULTS = os.path.join(ROOT, "results")
SUMMARY = os.path.join(ROOT, "data", "batch_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    subset = [json.loads(l) for l in open(SUBSET, encoding="utf-8")]
    if args.limit:
        subset = subset[: args.limit]
    config = load_config(os.path.join(ROOT, "config.yaml"))

    records, errors = [], []
    t0 = time.time()
    for i, item in enumerate(subset, 1):
        out_path = os.path.join(RESULTS, item["image_id"] + ".json")
        try:
            if os.path.exists(out_path):
                result = json.load(open(out_path, encoding="utf-8"))
            else:
                result = run_image(item["path"], config=config, output_path=out_path)
            records.append((item, result))
            print(f"[{i}/{len(subset)}] {item['image_id'][:36]} gt={item['gt_country']} "
                  f"map={result['map_estimate']} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            errors.append({"image_id": item["image_id"], "error": str(e)})
            print(f"[{i}/{len(subset)}] ERROR {item['image_id'][:36]}: {e}", flush=True)
            traceback.print_exc()

    def country_layer(r):
        """取国家层分布（v2 转移后 final 可能在更细层，用 levels 里的 country 条目对拍）。"""
        for L in r.get("levels", []):
            if L["level"] == "country":
                return L
        if r["prior"]["level"] == "country":
            return {"prior": r["prior"]["hypotheses"], "posterior": r["prior"]["hypotheses"]}
        return None

    # ---- 国家层聚合（v1/v2 可比） ----
    country_recs = [(it, r, country_layer(r)) for it, r in records if country_layer(r)]
    skipped_level = [it["image_id"] for it, r in records if not country_layer(r)]

    recall = top_k_recall_from_prior(
        [(r["prior"]["hypotheses"], it["gt_country"]) for it, r, _ in country_recs]
    ) if country_recs else {}
    coverage = (sum(
        1 for it, r, _ in country_recs
        if canonicalize_country(it["gt_country"])
        in {canonicalize_country(l) for l in r["prior"]["hypotheses"]}
    ) / len(country_recs)) if country_recs else None
    # 国家层 MAP = 国家层 posterior 的 argmax（转移前的国家判定）
    map_acc = (sum(
        1 for it, r, cl in country_recs
        if canonicalize_country(max(cl["posterior"], key=cl["posterior"].get))
        == canonicalize_country(it["gt_country"])
    ) / len(country_recs)) if country_recs else None

    # ---- v2 层级统计 ----
    transitions = [e for it, r in records for e in r["events"] if e["type"] == "transition"]
    final_levels = {}
    for it, r in records:
        lv = r["final_posterior"]["level"]
        final_levels[lv] = final_levels.get(lv, 0) + 1
    n_reached_city = sum(1 for it, r in records
                         if any(L["level"] in ("city", "street") for L in r.get("levels", [])))
    n_enhance = sum(1 for it, r in records for e in r["events"] if e["type"] == "enhance")

    entropies_prior, entropies_final, kls = [], [], []
    event_counts, alphas, cs = {}, [], []
    for it, r in records:
        entropies_prior.append(entropy_bits(r["prior"]["hypotheses"]))
        entropies_final.append(entropy_bits(r["final_posterior"]["hypotheses"]))
        if set(r["prior"]["hypotheses"]) == set(r["final_posterior"]["hypotheses"]):
            kls.append(kl_bits(r["final_posterior"]["hypotheses"], r["prior"]["hypotheses"]))
        for e in r["events"]:
            event_counts[e["type"]] = event_counts.get(e["type"], 0) + 1
        for step in r["trajectory"]:
            for c, a in step["judgments"].values():
                cs.append(c); alphas.append(a)

    summary = {
        "model": config.get("model"),
        "v2": {"enable_hierarchy": config.get("enable_hierarchy"),
               "enable_enhance": config.get("enable_enhance")},
        "n_run": len(records), "n_errors": len(errors), "errors": errors,
        "n_country_level": len(country_recs), "non_country_level": skipped_level,
        "prior_top_k_recall": recall,
        "prior_support_coverage": coverage,
        "country_map_accuracy": map_acc,
        "n_transitions": len(transitions), "n_reached_city_or_street": n_reached_city,
        "final_level_dist": final_levels, "n_enhance_calls": n_enhance,
        "mean_entropy_prior_bits": sum(entropies_prior) / len(entropies_prior) if records else None,
        "mean_entropy_final_bits": sum(entropies_final) / len(entropies_final) if records else None,
        "mean_kl_final_vs_prior_bits": sum(kls) / len(kls) if kls else None,
        "n_same_support": len(kls),
        "event_counts": event_counts,
        "judge_c_hist": {str(v): cs.count(v) for v in sorted(set(cs))} if cs else {},
        "judge_alpha_mean": sum(alphas) / len(alphas) if alphas else None,
        "judge_alpha_min_max": [min(alphas), max(alphas)] if alphas else None,
    }
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
