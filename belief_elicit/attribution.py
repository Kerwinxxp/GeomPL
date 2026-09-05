"""Attribution operators on the cue set function v(S): Shapley, SII, and the alternatives.

Exact (full 2^m lattice): `shapley`, `sii`, `empty_interaction`, `banzhaf`, `harsanyi`,
`additive_fit`, `min_sufficient`.  Second-order anchored truncation (for the inpaint arm,
where only singles / pairs / all are affordable): `order2_shapley`, `order2_from_v`.
`spearman` is the rank correlation every caller here needs.

Library module — no CLI.  `belief_elicit.order2_shapley` keeps the gray-block validation
CLI on top of `order2_shapley`.
"""
import itertools
import math
from collections import defaultdict

import numpy as np


# ---------------- exact operators over the full lattice ----------------

def shapley(v, m):
    """Exact Shapley values: permutation-weighted average marginal contribution."""
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
    """Shapley Interaction Index: average second difference over all coalitions in N\\{k,l}."""
    others = [i for i in range(m) if i not in (k, l)]
    tot = 0.0
    for size in range(len(others) + 1):
        w = math.factorial(size) * math.factorial(m - size - 2) / math.factorial(m - 1)
        for Sset in itertools.combinations(others, size):
            S = frozenset(Sset)
            delta = (v[S | {k, l}] - v[S | {k}] - v[S | {l}] + v[S])
            tot += w * delta
    return tot


def empty_interaction(v, m):
    """Empty-context pairwise interaction {"k,l": v({k,l}) - v({k}) - v({l})}."""
    return {f"{k},{l}": v[frozenset([k, l])] - v[frozenset([k])] - v[frozenset([l])]
            for k, l in itertools.combinations(range(m), 2)}


def banzhaf(v, m):
    """Banzhaf value: coalitions weighted uniformly rather than by permutation."""
    out = []
    for k in range(m):
        others = [i for i in range(m) if i != k]
        tot = 0.0
        for sz in range(m):
            for S in itertools.combinations(others, sz):
                tot += v[frozenset(S) | {k}] - v[frozenset(S)]
        out.append(tot / 2 ** (m - 1))
    return out


def additive_fit(v, m):
    """OLS fit of v(S) ~ sum_{k in S} a_k over all non-empty subsets -> (coefficients, R^2)."""
    subs = [S for sz in range(1, m + 1) for S in itertools.combinations(range(m), sz)]
    X = np.zeros((len(subs), m)); y = np.array([v[frozenset(S)] for S in subs])
    for i, S in enumerate(subs):
        for k in S:
            X[i, k] = 1.0
    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ a
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return a.tolist(), (1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))


def harsanyi(v, m):
    """Mobius decomposition d(T), aggregated per order as a share of total |d|."""
    by_order = defaultdict(float)
    for sz in range(1, m + 1):
        for T in itertools.combinations(range(m), sz):
            d = 0.0
            for sz2 in range(sz + 1):
                for S in itertools.combinations(T, sz2):
                    d += (-1) ** (sz - sz2) * v[frozenset(S)]
            by_order[sz] += abs(d)
    tot = sum(by_order.values())
    return {k: (val / tot if tot > 1e-12 else 0.0) for k, val in by_order.items()}


def min_sufficient(v, m, frac=0.8):
    """Smallest |S| reaching frac * v(N) — how many cues must be masked for that much shift."""
    target = frac * v[frozenset(range(m))]
    if target <= 0:
        return None
    for sz in range(1, m + 1):
        for S in itertools.combinations(range(m), sz):
            if v[frozenset(S)] >= target:
                return sz
    return m


# ---------------- second-order anchored truncation ----------------

def order2_shapley(singles, pairs, v_all=None):
    """Second-order anchored Shapley from singles / pairs / v(N) only.

        d_k     = v({k})                                (first-order Mobius dividend)
        d_kl    = v({k,l}) - v({k}) - v({l})             (pairwise interaction)
        phi2_k  = d_k + 0.5 * sum_{l != k} d_kl          (truncated at order 2)
        phi_k   = phi2_k + [v(N) - sum_j phi2_j] / m     (anchored so sum phi = v(N))

    Arguments
      singles : sequence of length m, singles[k] = v({k})
      pairs   : dict {(k, l): v({k,l})}, key order irrelevant
      v_all   : v(N); None skips the anchoring (phi = phi2, residual_share = nan)

    Returns a dict with phi, phi2, d, interactions, anchor, residual, residual_share
    (= |v(N) - sum phi2| / |v(N)|, the truncation diagnostic) and sum_phi2.
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
            raise KeyError(f"missing pair value v({{{a},{b}}})")
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
    """Pull singles / pairs / v(N) out of a v dict (frozenset keys) and run order2_shapley."""
    singles = [v[frozenset([k])] for k in range(m)]
    pairs = {(k, l): v[frozenset([k, l])] for k, l in itertools.combinations(range(m), 2)}
    return order2_shapley(singles, pairs, v.get(frozenset(range(m))))


# ---------------- rank correlation ----------------

def spearman(a, b):
    """Spearman rank correlation; nan when either side is constant or shorter than 2."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
