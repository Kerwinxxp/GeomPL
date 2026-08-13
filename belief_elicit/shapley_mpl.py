"""【实验性 · 可整体删除】Shapley-mPL:对 2 线索图算矫正后的逐线索泄露 φ + 交互项 I。

对每张 2 线索图(sweep 已含 v({1}), v({2}), v(N) = mpl_all):
  φ_k = ½ v({k}) + ½ (v(N) - v({other}))         (m=2 的 Shapley,= ½·LOI + ½·LOO)
  I   = v(N) - v({1}) - v({2})                    (交互:<0 信息重叠 / >0 互为备份)
效率校验:φ_1 + φ_2 == v(N)。
汇总:①按类别的 φ 中位(矫正后逐类别泄露);②交互项符号分布(重叠 vs 备份)。
运行:python -m belief_elicit.shapley_mpl
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")
OUT = os.path.join(os.path.dirname(__file__), "shapley_mpl_results.json")


def main():
    R = json.load(open(DATA, encoding="utf-8"))
    two = [r for r in R if r["n_cues"] == 2]
    rows = []
    for r in two:
        v1, v2 = r["per_cue"][0]["mpl"], r["per_cue"][1]["mpl"]
        vN = r["mpl_all"]
        phi1 = 0.5 * v1 + 0.5 * (vN - v2)
        phi2 = 0.5 * v2 + 0.5 * (vN - v1)
        I = vN - v1 - v2
        rows.append({"image": r["image_id"], "place": r["true_label"].split(",")[0],
                     "country_hit": r["country_hit"],
                     "cue1": r["per_cue"][0]["cue"], "cat1": r["per_cue"][0]["category"],
                     "cue2": r["per_cue"][1]["cue"], "cat2": r["per_cue"][1]["category"],
                     "v1": v1, "v2": v2, "vN": vN,
                     "phi1": phi1, "phi2": phi2, "I": I,
                     "eff_ok": abs(phi1 + phi2 - vN) < 1e-9})
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"=== Shapley-mPL:{len(rows)} 张 2 线索图 ===")
    print(f"效率校验 φ1+φ2==v(N):{sum(r['eff_ok'] for r in rows)}/{len(rows)} 通过\n")

    # 交互项符号分布
    good = [r for r in rows if r["country_hit"]]
    Is = np.array([r["I"] for r in good])
    print(f"交互项 I(在 {len(good)} 张可识别图上):")
    print(f"  I<-0.01 信息重叠(次可加): {(Is < -0.01).sum()} 张")
    print(f"  |I|<=0.01 近似可加:        {(np.abs(Is) <= 0.01).sum()} 张")
    print(f"  I>0.01  互为备份(超可加):  {(Is > 0.01).sum()} 张")
    print(f"  I 中位 {np.median(Is):+.3f}\n")

    # 按类别的 φ 中位(矫正后逐类别泄露)
    cat = defaultdict(list)
    for r in good:
        cat[r["cat1"] or "unknown"].append(r["phi1"])
        cat[r["cat2"] or "unknown"].append(r["phi2"])
    print("矫正后逐类别泄露(φ 中位,仅 2 线索可识别图):")
    for k, v in sorted(cat.items(), key=lambda kv: -np.median(kv[1])):
        print(f"  {k:24s} φ中位={np.median(v):.4f}  (n={len(v)})")

    # 最强交互的几张(讲故事用)
    print("\n交互最强的 5 张(|I| 最大):")
    for r in sorted(good, key=lambda x: -abs(x["I"]))[:5]:
        sign = "重叠" if r["I"] < 0 else "备份"
        print(f"  {r['place'][:14]:14s} I={r['I']:+.3f}({sign})  "
              f"v1={r['v1']:.2f} v2={r['v2']:.2f} vN={r['vN']:.2f}  "
              f"[{r['cue1'][:18]} | {r['cue2'][:18]}]")
    print("saved", OUT)


if __name__ == "__main__":
    main()
