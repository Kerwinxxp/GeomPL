"""【实验性 · 可整体删除】判定病A:引出方式换成 MC-logprob 后,信念能否随遮蔽而动?

对每张图,候选集 = 原图嘴报打分最强的 N 个(必含真值),原图/遮蔽图共用。
两种引出各测 full vs masked(遮全部 maskable 线索):
  verbalized  = 现有嘴报 0-1 分(基线)
  mc_logprob  = 多选字母 logprob(连续)
每种报:
  conc          集中度 1-H/Hmax
  p_true        真值概率  full->masked
  Δlogodds      真值 vs 最强跨簇干扰项  (应为负 = 遮线索后更不确定)
  TV            全变差距离 |full - masked|/2   ← 分布动了多少;0=完全没动
  mPL           masked→full 的 mPL(与主实验同口径)

判读:若 verbalized 的 TV≈0 而 mc_logprob 的 TV 明显>0 ⇒ 病因是**引出方式(尺子太粗)**,
      换 logprob 就能救;若两者 TV 都≈0 ⇒ 是**冗余**,换引出也没用(该图线索本就可替代)。
用法：python -m belief_elicit.run_logprob --images 261517384,847733166,370717727,181848051 --nlab 20
"""
import argparse
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

from belief_elicit.elicit import score_independent
from belief_elicit.mc_logprob import score_mc_logprob
from belief_elicit.run_compare import load_image_and_masks
from belief_elicit.run_noise import build_geometry, mpl
from clue_leak.masking import mask_solid_from_masks
from run import build_client, load_config

OUT = os.path.join(os.path.dirname(__file__), "logprob_results.json")


def conc(prior, labels):
    v = np.array([prior[l] for l in labels]); v = v / v.sum()
    ent = -(v * np.log2(v + 1e-12)).sum()
    return 1 - ent / math.log2(len(labels))


def tv(p, q, labels):
    return 0.5 * sum(abs(p[l] - q[l]) for l in labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="261517384,847733166,370717727,181848051")
    ap.add_argument("--nlab", type=int, default=20, help="候选数(MC 单字母 ≤26)")
    ap.add_argument("--n_perm", type=int, default=3, help="MC 字母排列平均次数(去位置偏置)")
    args = ap.parse_args()

    client = build_client(load_config(os.path.join(ROOT, "config.yaml")))
    gallery = json.load(open(os.path.join(ROOT, "data", "gallery_labels.json"), encoding="utf-8"))
    labels_cache = json.load(open(os.path.join(ROOT, "data", "gt_labels_cache.json"), encoding="utf-8"))
    rep, clusters, dist = build_geometry(gallery)

    rows = []
    for pref in args.images.split(","):
        iid = next(os.path.basename(f)[:-5]
                   for f in sorted(glob.glob(os.path.join(ROOT, "cue_extract", "results_sam3", "*.json")))
                   if os.path.basename(f).startswith(pref))
        img, masks, names, meta = load_image_and_masks(client, iid)
        tl = labels_cache.get(f"{meta['lat']:.5f},{meta['lon']:.5f}")
        if not masks or not tl:
            print(f"skip {pref} (masks={len(masks)} tl={tl})"); continue
        masked = mask_solid_from_masks(img, masks)
        # 候选集:原图嘴报最强 N-1 + 真值(两条件共用)
        rank = score_independent(img, gallery)["raw"]
        cand = [tl] + sorted((l for l in gallery if l != tl), key=lambda l: -rank[l])[:args.nlab - 1]
        tl_cluster = rep.get(tl, tl)
        out_cluster = [l for l in cand if rep.get(l, l) != tl_cluster]

        print(f"\n=== {tl.split(',')[0]} ({pref}) | 遮 {len(names)} 条线索 | {len(cand)} 候选 ===", flush=True)
        for mname, fn in [("verbalized", lambda im: score_independent(im, cand)["prior"]),
                          ("mc_logprob", lambda im: score_mc_logprob(im, cand, n_perm=args.n_perm)["prior"])]:
            pf, pm = fn(img), fn(masked)
            td = max(out_cluster, key=lambda l: pf[l])
            lo_f = math.log(pf[tl] / pf[td]); lo_m = math.log(pm[tl] / pm[td])
            row = {"image": iid, "place": tl.split(",")[0], "method": mname,
                   "conc_full": conc(pf, cand), "p_true_full": pf[tl], "p_true_masked": pm[tl],
                   "dlogodds": lo_m - lo_f, "tv": tv(pf, pm, cand),
                   "mpl": mpl(pm, pf, rep, clusters, dist),
                   "top_distractor": td}
            rows.append(row)
            print(f"  {mname:11s} conc {row['conc_full']*100:5.1f}%  "
                  f"p_true {pf[tl]:.3f}->{pm[tl]:.3f}  Δlogodds {row['dlogodds']:+.3f}  "
                  f"TV={row['tv']:.4f}  mPL={row['mpl']:.4f}", flush=True)

    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n=== 汇总(中位) ===")
    for mname in ["verbalized", "mc_logprob"]:
        sub = [r for r in rows if r["method"] == mname]
        if not sub:
            continue
        tvv = np.array([r["tv"] for r in sub]); mp = np.array([r["mpl"] for r in sub])
        cc = np.array([r["conc_full"] for r in sub])
        print(f"  {mname:11s} conc={np.median(cc)*100:5.1f}%  TV中位={np.median(tvv):.4f}  "
              f"TV>0.02图数={int((tvv>0.02).sum())}/{len(sub)}  mPL中位={np.median(mp):.4f}")
    print("\n判读:mc_logprob 的 TV 明显>verbalized ⇒ 病因是引出方式(可救);"
          "\n      两者 TV 都≈0 的图 ⇒ 该图是冗余主导(换引出也无用)。")
    print("saved", OUT)


if __name__ == "__main__":
    main()
