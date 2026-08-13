"""【实验性 · 可整体删除】全量等面积对照分析 + 伪影修正 Shapley(φ' = φ − c_img/m)。
运行:python -m belief_elicit.control_report
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

CTRL = os.path.join(os.path.dirname(__file__), "georanker_control_results.json")
SHAP = os.path.join(os.path.dirname(__file__), "shapley_v2_results.json")


def main():
    C = json.load(open(CTRL, encoding="utf-8"))
    S = {r["image_id"]: r for r in json.load(open(SHAP, encoding="utf-8"))}

    cues, c_img = [], {}
    for r in C:
        allc = [x["mpl"] for c in r["cues"] for x in c["controls"]]
        if allc:
            c_img[r["image_id"]] = float(np.mean(allc))
        for c in r["cues"]:
            if not c["controls"]:
                continue
            cm = [x["mpl"] for x in c["controls"]]
            cues.append(dict(iid=r["image_id"], cue=c["cue"], cat=c["category"],
                             area=c["area_frac"], real=c["real_mpl"],
                             cmean=float(np.mean(cm)), cmax=float(np.max(cm)),
                             hit=r["country_hit"], vN=r["vN"], m=r["n_cues"]))
    real = np.array([x["real"] for x in cues])
    cmean = np.array([x["cmean"] for x in cues])
    cmax = np.array([x["cmax"] for x in cues])
    resolv = real > cmax
    ctrl_all = np.array([x["mpl"] for r in C for c in r["cues"] for x in c["controls"]])
    cvals = np.array(list(c_img.values()))

    print(f"=== 全量等面积对照:{len(C)} 张图 / {len(cues)} 条线索 / {len(ctrl_all)} 次放置 ===\n")
    print("① 伪影地板(全部放置):"
          f" 中位={np.median(ctrl_all):.4f}  P90={np.percentile(ctrl_all,90):.4f}"
          f"  最大={ctrl_all.max():.4f}")
    print(f"② 配对: real>ctrl均值 {int((real>cmean).sum())}/{len(cues)}"
          f" | real>ctrl最大(可分辨) {int(resolv.sum())}/{len(cues)}"
          f" ({resolv.mean()*100:.0f}%) | 配对比中位 {np.median(real/np.maximum(cmean,1e-9)):.2f}x")
    print(f"③ 图级伪影常数 c_img: 中位={np.median(cvals):.4f}"
          f"  范围=[{cvals.min():.4f},{cvals.max():.4f}]"
          f"  与 v(N) 相关 r={np.corrcoef([c_img[r['image_id']] for r in C], [r['vN'] for r in C])[0,1]:+.2f}")
    print(f"④ 伪影与面积相关: r={np.corrcoef([x['area'] for x in cues], cmean)[0,1]:+.2f}\n")

    # 按类别的可分辨率
    by = defaultdict(list)
    for x, rs in zip(cues, resolv):
        by[x["cat"] or "unknown"].append(bool(rs))
    print("⑤ 逐类别可分辨率(real > 自身对照最大):")
    for k in sorted(by, key=lambda k: -np.mean(by[k])):
        print(f"   {k:24s} {np.mean(by[k])*100:5.1f}%  (n={len(by[k])})")

    # 伪影修正 Shapley:φ' = φ − c_img/m
    cat_phi, cat_phic = defaultdict(list), defaultdict(list)
    n_neg = 0
    for iid, s in S.items():
        ci = c_img.get(iid)
        if ci is None:
            continue
        m = s["n_cues"]
        for c in s["cues"]:
            phic = c["phi"] - ci / m
            cat_phi[c["category"] or "unknown"].append(c["phi"])
            cat_phic[c["category"] or "unknown"].append(phic)
            if phic < -0.01:
                n_neg += 1
    print("\n⑥ 伪影修正 Shapley(φ' = φ − c_img/m)逐类别中位:")
    print(f"{'类别':24s} {'φ(未修)':>9s} {'φ′(修正)':>9s} {'n':>4s}")
    rank_u = sorted(cat_phi, key=lambda k: -np.median(cat_phi[k]))
    rank_c = sorted(cat_phic, key=lambda k: -np.median(cat_phic[k]))
    for k in rank_c:
        print(f"{k:24s} {np.median(cat_phi[k]):9.4f} {np.median(cat_phic[k]):9.4f} "
              f"{len(cat_phi[k]):4d}")
    print(f"  未修正排序: {[k[:14] for k in rank_u]}")
    print(f"  修正后排序: {[k[:14] for k in rank_c]}")
    print(f"  排序是否不变: {'是' if rank_u == rank_c else '否'} | φ'<-0.01 线索数: {n_neg}")


if __name__ == "__main__":
    main()
