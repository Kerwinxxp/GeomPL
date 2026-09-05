"""Attribution methods other than Shapley, all read off the existing 2^m lattice (no GPU).

  1. Banzhaf value      : coalitions weighted uniformly (not by permutation as Shapley)
  2. Additive surrogate : least-squares additive fit of v(S); R^2 = how much is additive
  3. Harsanyi dividends : exact Mobius decomposition — which order the non-additivity lives at
  4. Minimal sufficient : smallest cue subset reaching a given share of v(N)

The operators live in `belief_elicit.attribution`; this module is the CLI over them.
Run: python -m belief_elicit.alt_attribution
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

from belief_elicit.attribution import (additive_fit, banzhaf, harsanyi,  # noqa: F401
                                       min_sufficient, shapley, spearman)
from belief_elicit.results import (ALT_ATTRIBUTION as OUT, LATTICE as LAT, SWEEP,
                                   build_v, load_lattice, load_sweep)  # noqa: F401


def main():
    sweep = load_sweep(SWEEP)
    lattice = load_lattice(LAT)
    rows, r2s, minsuf, rho_img = [], [], [], []
    ord_share = defaultdict(list)
    for r in sweep:
        m = r["n_cues"]
        if m < 1: continue
        v, ok = build_v(r, lattice)
        if not ok: continue
        ph = shapley(v, m); bz = banzhaf(v, m)
        add, r2 = additive_fit(v, m)
        if m >= 2:
            rho_img.append(spearman(ph, bz))
            if np.isfinite(r2): r2s.append(r2)
            for o, s in harsanyi(v, m).items(): ord_share[o].append(s)
            ms = min_sufficient(v, m)
            if ms: minsuf.append((ms, m))
        for k, pc in enumerate(r["per_cue"]):
            rows.append(dict(image=r["image_id"], cat=pc["category"] or "unknown",
                             v_single=pc["mpl"], shapley=ph[k], banzhaf=bz[k],
                             additive=add[k], n_cues=m, hit=r["country_hit"]))
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ph = np.array([x["shapley"] for x in rows]); bz = np.array([x["banzhaf"] for x in rows])
    ad = np.array([x["additive"] for x in rows]); vs = np.array([x["v_single"] for x in rows])
    print(f"=== 替代归因法({len(rows)} 条线索 / {len(r2s)} 张多线索图)===\n")
    print("① 逐线索全局相关(Spearman):")
    print(f"   Shapley vs Banzhaf        rho = {spearman(ph,bz):+.3f}")
    print(f"   Shapley vs additive-OLS   rho = {spearman(ph,ad):+.3f}")
    print(f"   Shapley vs single-cue mPL rho = {spearman(ph,vs):+.3f}   <- 基线,应更低")
    print(f"   图内 Shapley vs Banzhaf 秩相关中位 = {np.nanmedian(rho_img):+.3f}\n")
    print("② 可加代理模型的解释力(每图 R^2):")
    print(f"   中位 R^2 = {np.median(r2s):.3f} | 均值 = {np.mean(r2s):.3f} "
          f"| R^2<0.9 的图 {int((np.array(r2s)<0.9).sum())}/{len(r2s)}\n")
    print("③ Harsanyi 分解:非可加性住在几阶(|d| 占比中位):")
    for o in sorted(ord_share):
        print(f"   {o} 阶: {np.median(ord_share[o])*100:5.1f}%")
    print()
    ms = np.array([x[0] for x in minsuf]); mm = np.array([x[1] for x in minsuf])
    print(f"④ 最小充分子集(达到 80% v(N) 所需线索数):中位 {np.median(ms):.0f} / 共 {np.median(mm):.0f} 条")
    print(f"   仅需 1 条即可达 80% 的图: {int((ms==1).sum())}/{len(ms)}")
    print("\n⑤ 逐类别中位(三法对照):")
    cat = defaultdict(lambda: defaultdict(list))
    for x in rows:
        cat[x["cat"]]["s"].append(x["shapley"]); cat[x["cat"]]["b"].append(x["banzhaf"])
        cat[x["cat"]]["a"].append(x["additive"])
    print(f"   {'类别':24s} {'Shapley':>9s} {'Banzhaf':>9s} {'additive':>9s} {'n':>4s}")
    for k in sorted(cat, key=lambda k: -np.median(cat[k]["s"])):
        d = cat[k]
        print(f"   {k:24s} {np.median(d['s']):9.4f} {np.median(d['b']):9.4f} "
              f"{np.median(d['a']):9.4f} {len(d['s']):4d}")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
