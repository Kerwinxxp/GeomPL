"""Protection-oriented risk metrics on a released belief q.

`attribution.py` answers *"which cue explains the measured belief shift"* — an unsigned,
backward-looking question.  Protection asks a different one: **after treating a set S,
how much location evidence does the released image still carry, and in which direction?**
Two properties of the data force the distinction:

* 39 % of single-cue removals *raise* the attacker's p(x*): the cue was misleading, and
  removing it helps the attacker.  An unsigned mPL cannot see the sign.
* the Shapley top-k set is not the residual-minimising set in 28/128 (image x k) cases.

So every function here takes the belief `q` the attacker holds *after* the treatment and
compares it against a **reference** rho — either the content-free location prior pi
(`location_prior.json`, "what would the attacker believe with no image at all") or the
all-masked belief q_{I(-)N} ("the best this cue vocabulary can do").  Positive numbers
mean evidence is still there.

Distributions are plain ``{label: probability}`` dicts over the gallery; geometry
arguments are whatever `geometry.build_geometry` returned.

Library module — no CLI.
"""
import math

import numpy as np

from belief_elicit.geometry import build_geometry, haversine_km, merge_distribution

EPS = 1e-12
#: radii (km) at which `eps_local` is reported
DEFAULT_RADII = (25.0, 200.0, 750.0)


# ---------------- pointwise, truth-referenced ----------------

def p_true(q, x_star):
    """Probability the attacker assigns to the true label."""
    return float(q.get(x_star, 0.0))


def rank_true(q, x_star):
    """1-based rank of the true label under q (1 = the attacker's argmax)."""
    return 1 + sum(1 for lab, p in q.items() if p > q.get(x_star, 0.0))


def logit(p):
    """log(p / (1 - p)) with the probability clamped away from 0 and 1."""
    p = min(max(float(p), EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def signed_evidence(q, x_star, rho):
    """logit q(x*) - logit rho(x*), in nats. **The primary risk R1.**

    How much better than the reference belief `rho` the attacker can pin the *true*
    location down.  > 0: location evidence toward the truth survives the treatment;
    = 0: the release is as good as the reference; < 0: the release actively misleads the
    attacker away from the truth (over-protection, or a misleading cue was left in).
    """
    return logit(q.get(x_star, 0.0)) - logit(rho.get(x_star, 0.0))


def err_km(q, x_star, coords):
    """Great-circle km from the attacker's argmax label to the true label."""
    guess = max(q, key=q.get)
    if guess not in coords or x_star not in coords:
        return float("nan")
    return haversine_km(*coords[guess], *coords[x_star])


def exp_err_km(q, x_star, coords):
    """E_q[d(label, x*)] — the attacker's expected localisation error under q."""
    if x_star not in coords:
        return float("nan")
    tot = z = 0.0
    for lab, p in q.items():
        if p <= 0 or lab not in coords:
            continue
        tot += p * haversine_km(*coords[lab], *coords[x_star])
        z += p
    return tot / z if z > 0 else float("nan")


# ---------------- distributional, geometry-aware ----------------

def _llr(q, rho, rep, clusters):
    """Cluster-merged log-odds ratio {cluster: log(q/rho)} over clusters both support."""
    a, b = merge_distribution(q, rep), merge_distribution(rho, rep)
    return {k: math.log(a[k] / b[k])
            for k in clusters if a.get(k, 0.0) > 0 and b.get(k, 0.0) > 0}


def mpl_vs(q, rho, rep, clusters, dist):
    """Unsigned mPL of q against reference rho — `geometry.mpl(rho, q, ...)` verbatim.

    Kept here so a risk table can name one reference for every metric; the numbers are
    identical to the main line's `geometry.mpl`.
    """
    from belief_elicit.geometry import mpl as _mpl
    return _mpl(rho, q, rep, clusters, dist)


def mpl_max_vs(q, rho, rep, clusters, dist):
    """Worst-pair variant of mPL: max (not mean) of |dlog-odds| / d * 1000 over all pairs.

    `geometry.mpl` averages over admissible pairs, which dilutes a single very
    distinguishable pair.  For a protection guarantee the worst pair is what matters.
    """
    llr = _llr(q, rho, rep, clusters)
    ks = list(llr)
    best = 0.0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = dist(ks[i], ks[j])
            if d and d > 0:
                best = max(best, abs(llr[ks[i]] - llr[ks[j]]) / d * 1000)
    return best


def eps_local(q, rho, rep, clusters, dist, radii=DEFAULT_RADII):
    """Local geo-indistinguishability epsilon at several radii -> {r: {mean, max, n}}.

    For each radius r, look only at cluster pairs no more than r km apart and report the
    mean and the max of |dlog-odds(x_i, x_j)| / d(x_i, x_j) * 1000 (nats per 1000 km).
    The max is the empirical epsilon of a geo-indistinguishability guarantee at that
    scale: at r = 25 km it is street/city-level distinguishability, at 750 km
    country-level.  `n` counts the admissible pairs (0 -> mean/max are 0.0).
    """
    llr = _llr(q, rho, rep, clusters)
    ks = list(llr)
    out = {float(r): {"mean": 0.0, "max": 0.0, "n": 0} for r in radii}
    acc = {float(r): [] for r in radii}
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = dist(ks[i], ks[j])
            if not d or d <= 0:
                continue
            val = abs(llr[ks[i]] - llr[ks[j]]) / d * 1000
            for r in radii:
                if d <= r:
                    acc[float(r)].append(val)
    for r, vals in acc.items():
        if vals:
            out[r] = {"mean": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}
    return out


def residual_allmask(q, q_allmask, rep, clusters, dist):
    """Residual leakage relative to the fully-masked release: `geometry.mpl(q_allmask, q)`.

    0 means the treated image is indistinguishable (to this meter) from masking every
    known cue — the floor this cue vocabulary can reach.
    """
    from belief_elicit.geometry import mpl as _mpl
    return _mpl(q_allmask, q, rep, clusters, dist)


# ---------------- convenience ----------------

def gallery_coords(gv):
    """Gallery records -> {label: (lat, lon)} for the km metrics."""
    return {g["label"]: tuple(g["gps"]) for g in gv if g.get("gps")}


def risk_vector(q, x_star, rho, q_allmask, rep, clusters, dist, coords,
                radii=DEFAULT_RADII):
    """Every metric at once for one released belief -> dict (R1..R5 plus diagnostics)."""
    el = eps_local(q, rho, rep, clusters, dist, radii)
    return {
        "R1_signed_evidence": signed_evidence(q, x_star, rho),
        "R2_eps_local_max_200": el[200.0]["max"] if 200.0 in el else float("nan"),
        "R3_p_true": p_true(q, x_star),
        "R4_err_km": err_km(q, x_star, coords),
        "R5_residual_allmask": residual_allmask(q, q_allmask, rep, clusters, dist),
        "rank_true": rank_true(q, x_star),
        "exp_err_km": exp_err_km(q, x_star, coords),
        "mpl_vs_rho": mpl_vs(q, rho, rep, clusters, dist),
        "mpl_max_vs_rho": mpl_max_vs(q, rho, rep, clusters, dist),
        "eps_local": {str(int(r)): v for r, v in el.items()},
    }


# ---------------- vectorised evaluator ----------------

class RiskGeometry:
    """Cached cluster geometry, so thousands of (q, rho) pairs can be scored quickly.

    The selection experiment evaluates every one of the `2^P` subsets of every image, and
    the scalar functions above walk ~9 000 cluster pairs per call.  This class does that
    walk once, in numpy: it holds the cluster order, the pair distance vector and the
    per-radius pair masks, and `metrics(q, rho)` returns exactly the numbers
    `mpl_vs` / `mpl_max_vs` / `eps_local` produce (see `tests/test_risk.py`).

    Also exposes the plain `rep` / `clusters` / `dist` triple and `coords`, so a caller
    only needs this one object.
    """

    def __init__(self, gv, merge_km=2.0, radii=DEFAULT_RADII):
        self.rep, self.clusters, self.dist = build_geometry(gv, merge_km=merge_km)
        self.coords = gallery_coords(gv)
        self.radii = tuple(float(r) for r in radii)
        n = len(self.clusters)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = self.dist(self.clusters[i], self.clusters[j]) or 0.0
        self.iu = np.triu_indices(n, 1)
        self.dpair = D[self.iu]
        self.positive = self.dpair > 0
        self.radius_mask = {r: self.positive & (self.dpair <= r) for r in self.radii}

    def vec(self, q):
        """{label: p} -> cluster-merged probability vector in `self.clusters` order."""
        m = merge_distribution(q, self.rep)
        return np.array([m.get(c, 0.0) for c in self.clusters], float)

    def _pairs(self, qv, rv):
        """-> (per-pair |dlog-odds| / d * 1000, boolean admissibility mask)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = np.log(np.where((qv > 0) & (rv > 0), qv, 1.0) /
                         np.where((qv > 0) & (rv > 0), rv, 1.0))
        good = (qv > 0) & (rv > 0)
        i, j = self.iu
        valid = good[i] & good[j] & self.positive
        out = np.zeros(self.dpair.shape)
        out[valid] = np.abs(llr[i][valid] - llr[j][valid]) / self.dpair[valid] * 1000
        return out, valid

    def metrics(self, qv, rv):
        """Vectors (from `vec`) -> {mpl, mpl_max, eps: {radius: {mean, max, n}}}."""
        e, valid = self._pairs(qv, rv)
        sel = e[valid]
        out = {"mpl": float(sel.mean()) if sel.size else 0.0,
               "mpl_max": float(sel.max()) if sel.size else 0.0,
               "eps": {}}
        for r in self.radii:
            m = valid & self.radius_mask[r]
            v = e[m]
            out["eps"][r] = ({"mean": float(v.mean()), "max": float(v.max()),
                              "n": int(v.size)} if v.size
                             else {"mean": 0.0, "max": 0.0, "n": 0})
        return out
