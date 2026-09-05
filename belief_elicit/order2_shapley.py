"""Validate the second-order anchored Shapley truncation against the exact gray-block lattice.

The estimator itself lives in `belief_elicit.attribution` (`order2_shapley`,
`order2_from_v`) and is re-exported here for the callers that have always imported it from
this module. This CLI checks it where the full 2^m lattice exists, reporting Spearman,
top-1 agreement and relative magnitude error, stratified by m.

Run: python -m belief_elicit.order2_shapley
"""
import argparse
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

from belief_elicit.attribution import (order2_from_v, order2_shapley,  # noqa: F401
                                       shapley as exact_shapley, spearman)
from belief_elicit.results import (ORDER2_VALIDATION as OUT, SWEEP, LATTICE,
                                   build_v as build_v_gray, load_lattice, load_sweep)


def validate(sweep_path=SWEEP, lattice_path=LATTICE, out_path=OUT, quiet=False):
    sweep = load_sweep(sweep_path)
    lattice = load_lattice(lattice_path)

    per_image, skipped = [], 0
    for r in sweep:
        m = r["n_cues"]
        if m < 2:                       # at m=1, phi = v({1}); both methods trivially agree
            continue
        v, ok = build_v_gray(r, lattice)
        if not ok:
            skipped += 1
            continue
        ex = exact_shapley(v, m)
        o2 = order2_from_v(v, m)
        ap = o2["phi"]
        denom = float(np.sum(np.abs(ex)))
        rel = float(np.sum(np.abs(np.array(ap) - np.array(ex))) / denom) if denom > 1e-12 \
            else float("nan")
        maxrel = float(np.max(np.abs(np.array(ap) - np.array(ex))) / denom) \
            if denom > 1e-12 else float("nan")
        per_image.append({
            "image_id": r["image_id"], "m": m,
            "vN": v[frozenset(range(m))],
            "rho": spearman(ap, ex),
            "top1": int(np.argmax(ap)) == int(np.argmax(ex)),
            "rel_err": rel, "max_rel_err": maxrel,
            "residual_share": o2["residual_share"],
            "phi_exact": ex, "phi_order2": ap,
        })

    if out_path:
        json.dump(per_image, open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    if quiet:
        return per_image

    print(f"=== 二阶截断 Shapley 校验(灰块 2^m 全格;{len(per_image)} 张 m>=2 的图"
          f";跳过未补全 {skipped})===\n")
    by_m = defaultdict(list)
    for x in per_image:
        by_m[x["m"]].append(x)
    print(f"{'m':>3s} {'n图':>4s} {'ρ中位':>8s} {'ρ均值':>8s} {'top-1一致':>10s} "
          f"{'相对幅度误差(中位/均值)':>24s} {'残差占比中位':>12s}")
    for m in sorted(by_m):
        g = by_m[m]
        rho = np.array([x["rho"] for x in g], float)
        rel = np.array([x["rel_err"] for x in g], float)
        rs = np.array([x["residual_share"] for x in g], float)
        t1 = np.mean([x["top1"] for x in g])
        print(f"{m:3d} {len(g):4d} {np.nanmedian(rho):+8.3f} {np.nanmean(rho):+8.3f} "
              f"{t1*100:9.0f}% {np.median(rel)*100:11.1f}% /{np.mean(rel)*100:8.1f}% "
              f"{np.median(rs)*100:11.1f}%")
    allrel = np.array([x["rel_err"] for x in per_image], float)
    allrho = np.array([x["rho"] for x in per_image], float)
    allt1 = np.mean([x["top1"] for x in per_image])
    print(f"{'全体':>3s} {len(per_image):4d} {np.nanmedian(allrho):+8.3f} "
          f"{np.nanmean(allrho):+8.3f} {allt1*100:9.0f}% "
          f"{np.median(allrel)*100:11.1f}% /{np.mean(allrel)*100:8.1f}%")
    exact_m = [m for m in sorted(by_m)
               if np.max([x["rel_err"] for x in by_m[m]]) < 1e-9]
    print(f"\n解析上精确的 m: {exact_m}(m<=3 时二阶截断 = 精确 Shapley)")
    print("相对幅度误差 = Σ_k|φ2_k − φ_k| / Σ_k|φ_k|(逐图),残差占比 = |v(N)−Σφ^(2)|/|v(N)|")
    if out_path:
        print("saved", out_path)
    return per_image


def main():
    ap = argparse.ArgumentParser(description="second-order anchored Shapley + gray-block check")
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--lattice", default=LATTICE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    validate(a.sweep, a.lattice, None if a.no_save else a.out)


if __name__ == "__main__":
    main()
