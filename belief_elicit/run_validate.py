"""【实验性 · 可整体删除】② 在多张图上验证 allocate vs independent。

对每张图:原图 vs 遮掉全部 maskable 线索,两种方法各打一次(全 77 gallery)。
报四个量:
  conc        集中度 1-H/Hmax(信念有多尖;independent 实测只有 3-12%)
  Δp_true     遮后 − 原图(应为负:遮掉线索,攻击者对真值的信心该下降)
  Δlogodds    真值 vs 最强**跨簇**干扰项的对数几率变化(应为负)
              —— 用跨簇干扰项,避免 London 那种"Westminster vs Greater London"簇内迁移被 mPL 吃掉
  mPL         遮蔽→原图 的 mPL(即线索的总泄露)

判读:好的引出方式应当 conc 高、Δ 为负、mPL 非零。
用法：python -m belief_elicit.run_validate --n_images 6
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

from belief_elicit.elicit import METHODS
from belief_elicit.run_compare import load_image_and_masks
from belief_elicit.run_noise import build_geometry, mpl
from clue_leak.masking import mask_solid_from_masks
from run import build_client, load_config

OUT = os.path.join(os.path.dirname(__file__), "validate_results.json")
# 默认样本:覆盖不同大洲/线索类型;都是已提取线索的图
DEFAULT = ["171046893",   # London  红电话亭/酒吧招牌  (簇内退化案例)
           "847733166",   # Venice  拉丁铭文/雕像
           "370717727",   # Cuba    石头要塞
           "261517384",   # Okazaki 日文/盔甲/樱花     (原方案全 0)
           "311344213",   # Delhi   Qutub Minar
           "181848051"]   # NYC     布鲁克林大桥/天际线


def concentration(prior, labels):
    v = np.array([prior[l] for l in labels]); v = v / v.sum()
    ent = -(v * np.log2(v + 1e-12)).sum()
    return 1 - ent / math.log2(len(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=",".join(DEFAULT))
    ap.add_argument("--methods", default="independent,allocate")
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
        masked_img = mask_solid_from_masks(img, masks)
        tl_cluster = rep.get(tl, tl)
        print(f"\n=== {tl.split(',')[0]} ({pref}) | {len(names)} cues ===", flush=True)
        for mname in args.methods.split(","):
            fn = METHODS[mname]
            pf = fn(img, gallery)["prior"]
            pm = fn(masked_img, gallery)["prior"]
            # 最强**跨簇**干扰项(排除与真值同簇的,如 Greater London vs Westminster)
            out_cluster = [l for l in gallery if rep.get(l, l) != tl_cluster]
            td = max(out_cluster, key=lambda l: pf[l])
            lo_f = math.log(pf[tl] / pf[td]); lo_m = math.log(pm[tl] / pm[td])
            val = mpl(pm, pf, rep, clusters, dist)      # prior=遮蔽, post=原图
            row = {"image": iid, "place": tl.split(",")[0], "method": mname,
                   "n_cues": len(names),
                   "conc_full": concentration(pf, gallery), "conc_masked": concentration(pm, gallery),
                   "p_true_full": pf[tl], "p_true_masked": pm[tl],
                   "top_cross_cluster_distractor": td,
                   "logodds_full": lo_f, "logodds_masked": lo_m, "mpl": val}
            rows.append(row)
            print(f"  {mname:12s} conc {row['conc_full']*100:5.1f}%  "
                  f"p_true {pf[tl]:.3f}->{pm[tl]:.3f} (Δ{pm[tl]-pf[tl]:+.3f})  "
                  f"logodds {lo_f:+.2f}->{lo_m:+.2f} (Δ{lo_m-lo_f:+.2f})  "
                  f"mPL={val:.4f}   [vs {td[:20]}]", flush=True)

    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n=== 汇总 ===")
    for mname in args.methods.split(","):
        sub = [r for r in rows if r["method"] == mname]
        if not sub:
            continue
        dlo = np.array([r["logodds_masked"] - r["logodds_full"] for r in sub])
        mp = np.array([r["mpl"] for r in sub])
        cc = np.array([r["conc_full"] for r in sub])
        print(f"  {mname:12s} 集中度中位={np.median(cc)*100:5.1f}%  "
              f"Δlogodds 中位={np.median(dlo):+.3f}  为负的图数={int((dlo<0).sum())}/{len(sub)}  "
              f"mPL 中位={np.median(mp):.4f}  非零图数={int((mp>1e-6).sum())}/{len(sub)}")
    print("\n判读:Δlogodds 应为负(遮掉线索→对真值信心下降)。"
          "\nmPL 恒为 0 说明信念全落在单一簇内、跨簇无迁移 ⇒ 该图线索在 25km 粒度上无边际泄露。")
    print("saved", OUT)


if __name__ == "__main__":
    main()
