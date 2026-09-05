"""【实验性 · 可整体删除】二阶截断(anchored)Shapley:只用单条 + 成对 + 全遮的 v。

动机:修复(inpaint)口径下打分极贵,2^m 全格不可行;但 **单条 v({k})、成对 v({k,l})、
全遮 v(N)** 是可以负担的(m 条线索 → m + C(m,2) + 1 次打分)。本模块把 Harsanyi
(Mobius)分解截断到二阶,再用 v(N) 做 efficiency 锚定:

    d_k    = v({k})                                  (一阶 Mobius 分红, v(∅)=0)
    d_kl   = v({k,l}) − v({k}) − v({l})               (二阶分红 = 成对交互)
    φ_k^(2)= d_k + ½ Σ_{l≠k} d_kl                     (截断到二阶的 Shapley)
    φ_k    = φ_k^(2) + [v(N) − Σ_j φ_j^(2)] / m       (锚定:强制 Σφ = v(N))

诊断量 residual_share = |v(N) − Σ_j φ_j^(2)| / |v(N)|:三阶及以上分红被丢掉了多少。

CLI 在**灰块**数据上做校验(那里 2^m 全格已有,精确 Shapley 可算):
    python -m belief_elicit.order2_shapley
逐图报告 Spearman、top-1 一致率、相对幅度误差,并按 m 分层。
"""
import argparse
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

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "georanker_sweep_results.json")
LATTICE = os.path.join(HERE, "georanker_lattice_results.json")
OUT = os.path.join(HERE, "order2_shapley_validation.json")


# ---------------- 库:二阶截断 Shapley ----------------

def order2_shapley(singles, pairs, v_all=None):
    """二阶锚定 Shapley。

    参数
      singles : 长度 m 的序列,singles[k] = v({k})
      pairs   : dict,键 (k, l)(k<l 或任意序,内部归一)→ v({k,l})
      v_all   : v(N)。None 时不做锚定(φ = φ^(2)),residual_share = nan。

    返回 dict:
      phi            : 锚定后的 φ_k(list,长度 m)
      phi2           : 锚定前的 φ_k^(2)
      d              : 一阶分红 d_k = v({k})
      interactions   : {(k,l): d_kl},k<l
      anchor         : [v(N) − Σφ^(2)] / m(逐线索均摊的锚定量)
      residual       : v(N) − Σφ^(2)
      residual_share : |residual| / |v(N)|(截断诊断;v(N)≈0 时为 nan)
      sum_phi2       : Σφ^(2)
    """
    m = len(singles)
    d = [float(x) for x in singles]
    pd = {}
    for (k, l), val in pairs.items():
        a, b = (int(k), int(l)) if int(k) < int(l) else (int(l), int(k))
        pd[(a, b)] = float(val)

    inter = {}
    for a, b in itertools.combinations(range(m), 2):
        if (a, b) not in pd:
            raise KeyError(f"缺少成对值 v({{{a},{b}}})")
        inter[(a, b)] = pd[(a, b)] - d[a] - d[b]

    phi2 = []
    for k in range(m):
        s = d[k]
        for l in range(m):
            if l == k:
                continue
            a, b = (k, l) if k < l else (l, k)
            s += 0.5 * inter[(a, b)]
        phi2.append(s)

    sum2 = float(sum(phi2))
    if v_all is None:
        return {"phi": list(phi2), "phi2": list(phi2), "d": d, "interactions": inter,
                "anchor": 0.0, "residual": float("nan"),
                "residual_share": float("nan"), "sum_phi2": sum2}
    vN = float(v_all)
    resid = vN - sum2
    anchor = resid / m if m else 0.0
    phi = [p + anchor for p in phi2]
    share = abs(resid) / abs(vN) if abs(vN) > 1e-12 else float("nan")
    return {"phi": phi, "phi2": list(phi2), "d": d, "interactions": inter,
            "anchor": anchor, "residual": resid, "residual_share": share,
            "sum_phi2": sum2}


def order2_from_v(v, m):
    """从完整/部分 v 字典(键为 frozenset)取出单条/成对/全遮,调用 order2_shapley。"""
    singles = [v[frozenset([k])] for k in range(m)]
    pairs = {(k, l): v[frozenset([k, l])] for k, l in itertools.combinations(range(m), 2)}
    return order2_shapley(singles, pairs, v.get(frozenset(range(m))))


# ---------------- 校验用:精确 Shapley + 相关系数 ----------------

def exact_shapley(v, m):
    out = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        tot = 0.0
        for size in range(m):
            w = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
            for Sset in itertools.combinations(others, size):
                tot += w * (v[frozenset(Sset) | {k}] - v[frozenset(Sset)])
        out.append(tot)
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def build_v_gray(r, lattice):
    """复刻 shapley_v2.build_v:sweep(单条 + 全遮)+ lattice(中间子集)。"""
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


# ---------------- CLI:灰块数据上的校验 ----------------

def validate(sweep_path=SWEEP, lattice_path=LATTICE, out_path=OUT, quiet=False):
    sweep = json.load(open(sweep_path, encoding="utf-8"))
    lattice = ({r["image_id"]: r for r in json.load(open(lattice_path, encoding="utf-8"))}
               if os.path.exists(lattice_path) else {})

    per_image, skipped = [], 0
    for r in sweep:
        m = r["n_cues"]
        if m < 2:                       # m=1 时 φ = v({1}),两法平凡相同
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
    ap = argparse.ArgumentParser(description="二阶截断 anchored Shapley + 灰块校验")
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--lattice", default=LATTICE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    validate(a.sweep, a.lattice, None if a.no_save else a.out)


if __name__ == "__main__":
    main()
