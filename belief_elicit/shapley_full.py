"""【实验性 · 可整体删除】全量 Shapley-mPL:对全部多线索图(m=2..5)精确枚举 φ + 交互。

v(S) 来源:单条+全遮 = sweep;中间子集 = lattice。v(∅)=0。
  φ_k = Σ_{S⊆N\\{k}} |S|!(m-|S|-1)!/m! · [v(S∪{k}) − v(S)]   (精确,m≤5 直接枚举)
  I(k,l) = v({k,l}) − v({k}) − v({l})
效率恒等式 Σφ_k = v(N) 逐图校验。输出 shapley_full_results.json + 汇总统计。
运行(主环境):python -m belief_elicit.shapley_full
"""
import itertools
import json
import math
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

SWEEP = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")
LATTICE = os.path.join(os.path.dirname(__file__), "georanker_lattice_results.json")
OUT = os.path.join(os.path.dirname(__file__), "shapley_full_results.json")


def shapley(v, m):
    """v: dict frozenset->value(含全部子集与空集);返回 [φ_0..φ_{m-1}]。"""
    phis = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        total = 0.0
        for size in range(m):
            w = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for S in itertools.combinations(others, size):
                total += w * (v[frozenset(S) | {k}] - v[frozenset(S)])
        phis.append(total)
    return phis


def main():
    sweep = {r["image_id"]: r for r in json.load(open(SWEEP, encoding="utf-8"))}
    lattice = {r["image_id"]: r for r in json.load(open(LATTICE, encoding="utf-8"))} \
        if os.path.exists(LATTICE) else {}

    rows, skipped = [], 0
    for iid, r in sweep.items():
        m = r["n_cues"]
        if m < 2:
            continue
        v = {frozenset(): 0.0}
        for k, pc in enumerate(r["per_cue"]):
            v[frozenset([k])] = pc["mpl"]
        v[frozenset(range(m))] = r["mpl_all"]
        if m >= 3:
            lat = lattice.get(iid)
            need = 2 ** m - 2 - m
            if not lat or len(lat["combos"]) < need:
                skipped += 1; continue                 # 格未补全
            for c in lat["combos"]:
                v[frozenset(c["subset"])] = c["mpl"]
        phis = shapley(v, m)
        vN = v[frozenset(range(m))]
        assert abs(sum(phis) - vN) < 1e-9, f"效率恒等式失败: {iid}"
        inter = {}
        for k, l in itertools.combinations(range(m), 2):
            inter[f"{k},{l}"] = v[frozenset([k, l])] - v[frozenset([k])] - v[frozenset([l])]
        rows.append({"image_id": iid, "place": r["true_label"].split(",")[0],
                     "country_hit": r["country_hit"], "n_cues": m, "vN": vN,
                     "cues": [{"cue": pc["cue"], "category": pc["category"],
                               "v_single": pc["mpl"], "phi": phis[k]}
                              for k, pc in enumerate(r["per_cue"])],
                     "interactions": inter})
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"=== 全量 Shapley:{len(rows)} 张多线索图(跳过未补全 {skipped})===")
    print(f"效率校验:{len(rows)}/{len(rows)} 全部通过\n")
    good = [r for r in rows if r["country_hit"]]

    # 逐类别 φ(矫正后)与单条 v(未矫正)对照
    cat_phi, cat_v = defaultdict(list), defaultdict(list)
    for r in good:
        for c in r["cues"]:
            cat_phi[c["category"] or "unknown"].append(c["phi"])
            cat_v[c["category"] or "unknown"].append(c["v_single"])
    print(f"逐类别泄露(可识别图 n={len(good)};φ=Shapley 矫正,v=单条 mPL):")
    print(f"{'类别':26s} {'φ中位':>8s} {'v中位':>8s} {'n':>4s}")
    for k in sorted(cat_phi, key=lambda x: -np.median(cat_phi[x])):
        print(f"{k:26s} {np.median(cat_phi[k]):8.4f} {np.median(cat_v[k]):8.4f} "
              f"{len(cat_phi[k]):4d}")

    # 交互结构(全部对)
    Is = [x for r in good for x in r["interactions"].values()]
    Is = np.array(Is)
    print(f"\n交互结构({len(Is)} 对,可识别图):")
    print(f"  重叠 I<-0.01: {(Is < -0.01).sum()}  |  ≈可加: {(np.abs(Is) <= 0.01).sum()}"
          f"  |  备份 I>0.01: {(Is > 0.01).sum()}   中位 I={np.median(Is):+.3f}")

    # 负 φ 统计(诚实报告)
    negs = [c for r in good for c in r["cues"] if c["phi"] < -0.01]
    print(f"\nφ<-0.01 的线索(遮之反而回推信念): {len(negs)} 条")
    print("saved", OUT)


if __name__ == "__main__":
    main()
