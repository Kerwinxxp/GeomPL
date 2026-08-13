"""【实验性 · 可整体删除】审稿修复版归因:全部 maskable 图(m>=1)+ 正式 SII + hit/miss 分层。

修复(审稿意见 3/4/5):
  #3 正式 Shapley Interaction Index(在所有 coalition context 上平均二阶差分),取代 empty-context I;
  #4 全部 95 张 maskable 图纳入(m=1 时 φ=v({1}));single-cue 与 Shapley 用【同一图/线索集】配对;
  #5 不再条件化于 country-hit;输出全体 + hit/miss 分层。
v(S) 来源:sweep(单条/全遮)+ lattice(中间)。空集=0。m<=5 直接枚举。
运行(主环境):python -m belief_elicit.shapley_v2
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
OUT = os.path.join(os.path.dirname(__file__), "shapley_v2_results.json")


def shapley(v, m):
    phis = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        tot = 0.0
        for size in range(m):
            w = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for Sset in itertools.combinations(others, size):
                tot += w * (v[frozenset(Sset) | {k}] - v[frozenset(Sset)])
        phis.append(tot)
    return phis


def sii(v, m, k, l):
    """Shapley Interaction Index:在 N\\{k,l} 的所有 coalition 上平均二阶差分。"""
    others = [i for i in range(m) if i not in (k, l)]
    tot = 0.0
    for size in range(len(others) + 1):
        w = math.factorial(size) * math.factorial(m - size - 2) / math.factorial(m - 1)
        for Sset in itertools.combinations(others, size):
            S = frozenset(Sset)
            delta = (v[S | {k, l}] - v[S | {k}] - v[S | {l}] + v[S])
            tot += w * delta
    return tot


def build_v(r, lattice):
    m = r["n_cues"]
    v = {frozenset(): 0.0}
    for k, pc in enumerate(r["per_cue"]):
        v[frozenset([k])] = pc["mpl"]
    if m == 1:
        return v, True
    v[frozenset(range(m))] = r["mpl_all"]
    if m >= 3:
        lat = lattice.get(r["image_id"])
        if not lat or len(lat["combos"]) < 2 ** m - 2 - m:
            return v, False
        for c in lat["combos"]:
            v[frozenset(c["subset"])] = c["mpl"]
    return v, True


def main():
    sweep = json.load(open(SWEEP, encoding="utf-8"))
    lattice = {r["image_id"]: r for r in json.load(open(LATTICE, encoding="utf-8"))} \
        if os.path.exists(LATTICE) else {}

    rows, skipped = [], 0
    for r in sweep:
        m = r["n_cues"]
        if m < 1:
            continue
        v, ok = build_v(r, lattice)
        if not ok:
            skipped += 1; continue
        phis = shapley(v, m)
        vN = v[frozenset(range(m))] if m >= 1 else 0.0
        if m >= 2:
            assert abs(sum(phis) - vN) < 1e-9, f"效率失败 {r['image_id']}"
        inter_sii = {f"{k},{l}": sii(v, m, k, l)
                     for k, l in itertools.combinations(range(m), 2)}
        inter_empty = {f"{k},{l}": v[frozenset([k, l])] - v[frozenset([k])] - v[frozenset([l])]
                       for k, l in itertools.combinations(range(m), 2)} if m >= 2 else {}
        rows.append({"image_id": r["image_id"], "place": r["true_label"].split(",")[0],
                     "country_hit": r["country_hit"], "km_error": r["km_error"],
                     "n_cues": m, "vN": vN,
                     "cues": [{"cue": pc["cue"], "category": pc["category"],
                               "v_single": pc["mpl"], "phi": phis[k]}
                              for k, pc in enumerate(r["per_cue"])],
                     "sii": inter_sii, "empty_interaction": inter_empty})
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def cat_table(subset, tag):
        cphi, cv = defaultdict(list), defaultdict(list)
        for r in subset:
            for c in r["cues"]:
                cphi[c["category"] or "unknown"].append(c["phi"])
                cv[c["category"] or "unknown"].append(c["v_single"])
        print(f"\n[{tag}] 逐类别(φ=Shapley, v=single-cue;同一图/线索集配对):")
        print(f"{'类别':24s} {'φ中位':>8s} {'v中位':>8s} {'n':>4s}")
        rank = sorted(cphi, key=lambda x: -np.median(cphi[x]))
        for k in rank:
            print(f"{k:24s} {np.median(cphi[k]):8.4f} {np.median(cv[k]):8.4f} {len(cphi[k]):4d}")
        vr = sorted(cv, key=lambda x: -np.median(cv[x]))
        print(f"  φ 排名: {[k[:14] for k in rank]}")
        print(f"  v 排名: {[k[:14] for k in vr]}")

    hit = [r for r in rows if r["country_hit"]]
    miss = [r for r in rows if not r["country_hit"]]
    print(f"=== 归因:{len(rows)} 张 maskable 图(m>=1;跳过未补全 {skipped})===")
    print(f"效率(m>=2):全部通过 | hit {len(hit)} / miss {len(miss)}")
    cat_table(rows, "全体 95 张 [主结果]")
    cat_table(hit, "仅 country-hit [分层]")

    # 交互:SII vs empty-context 对照
    sii_all = np.array([x for r in rows for x in r["sii"].values()])
    emp_all = np.array([x for r in rows for x in r["empty_interaction"].values()])
    print(f"\n交互对照({len(sii_all)} 对):")
    for name, arr in [("SII(正式)", sii_all), ("empty-context(旧)", emp_all)]:
        print(f"  {name:18s} 重叠<-.01 {int((arr<-0.01).sum()):3d} | ~0 "
              f"{int((np.abs(arr)<=0.01).sum()):3d} | 备份>.01 {int((arr>0.01).sum()):3d} "
              f"| 中位 {np.median(arr):+.3f}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
