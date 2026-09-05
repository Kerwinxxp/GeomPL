"""Shapley 之外的替代归因法(全部基于已有 2^m 格,零 GPU)。

  1. Banzhaf value      : 对 coalition 均匀加权(而非 Shapley 的排列加权)
  2. Additive surrogate : 对 v(S) 最小二乘拟合可加模型,R^2 = "可加性能解释多少"
  3. Harsanyi dividends : 精确 Mobius 分解,看非可加性住在几阶
  4. Minimal sufficient : 达到 v(N) 给定比例所需的最小线索子集(隐私可读性强)
运行:python -m belief_elicit.alt_attribution
"""
import itertools, json, math, os, sys
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import numpy as np

SWEEP = os.path.join(os.path.dirname(__file__), "georanker_sweep_results.json")
LAT = os.path.join(os.path.dirname(__file__), "georanker_lattice_results.json")
OUT = os.path.join(os.path.dirname(__file__), "alt_attribution_results.json")

def build_v(r, lattice):
    m = r["n_cues"]; v = {frozenset(): 0.0}
    for k, pc in enumerate(r["per_cue"]): v[frozenset([k])] = pc["mpl"]
    if m == 1: return v, True
    v[frozenset(range(m))] = r["mpl_all"]
    if m >= 3:
        lat = lattice.get(r["image_id"])
        if not lat or len(lat["combos"]) < 2**m - 2 - m: return v, False
        for c in lat["combos"]: v[frozenset(c["subset"])] = c["mpl"]
    return v, True

def shapley(v, m):
    out = []
    for k in range(m):
        others = [i for i in range(m) if i != k]; tot = 0.0
        for sz in range(m):
            w = math.factorial(sz)*math.factorial(m-sz-1)/math.factorial(m)
            for S in itertools.combinations(others, sz):
                tot += w*(v[frozenset(S)|{k}] - v[frozenset(S)])
        out.append(tot)
    return out

def banzhaf(v, m):
    out = []
    for k in range(m):
        others = [i for i in range(m) if i != k]; tot = 0.0
        for sz in range(m):
            for S in itertools.combinations(others, sz):
                tot += v[frozenset(S)|{k}] - v[frozenset(S)]
        out.append(tot / 2**(m-1))
    return out

def additive_fit(v, m):
    """v(S) ~ sum_{k in S} a_k,对全部非空子集 OLS;返回 (系数, R^2)。"""
    subs = [S for sz in range(1, m+1) for S in itertools.combinations(range(m), sz)]
    X = np.zeros((len(subs), m)); y = np.array([v[frozenset(S)] for S in subs])
    for i, S in enumerate(subs):
        for k in S: X[i, k] = 1.0
    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ a
    ss_res = float(((y-pred)**2).sum()); ss_tot = float(((y-y.mean())**2).sum())
    return a.tolist(), (1 - ss_res/ss_tot if ss_tot > 1e-12 else float("nan"))

def harsanyi(v, m):
    """Mobius 分解 d(T);返回按阶聚合的 |d| 占比。"""
    by_order = defaultdict(float)
    for sz in range(1, m+1):
        for T in itertools.combinations(range(m), sz):
            Tf = frozenset(T); d = 0.0
            for sz2 in range(sz+1):
                for S in itertools.combinations(T, sz2):
                    d += (-1)**(sz-sz2) * v[frozenset(S)]
            by_order[sz] += abs(d)
    tot = sum(by_order.values())
    return {k: (val/tot if tot > 1e-12 else 0.0) for k, val in by_order.items()}

def min_sufficient(v, m, frac=0.8):
    """达到 frac*v(N) 所需最小 |S|(masking 视角:至少要遮几条才能造成 80% 的总移动)。"""
    target = frac * v[frozenset(range(m))]
    if target <= 0: return None
    for sz in range(1, m+1):
        for S in itertools.combinations(range(m), sz):
            if v[frozenset(S)] >= target: return sz
    return m

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if len(a) < 2 or ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])

def main():
    sweep = json.load(open(SWEEP, encoding="utf-8"))
    lattice = {r["image_id"]: r for r in json.load(open(LAT, encoding="utf-8"))} if os.path.exists(LAT) else {}
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
