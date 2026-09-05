"""Unit tests for belief_elicit.risk on a 3-label toy gallery."""
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from belief_elicit import risk                                    # noqa: E402
from belief_elicit.geometry import build_geometry, haversine_km   # noqa: E402
from belief_elicit.results import build_q                         # noqa: E402

# A ~ 0 km, B ~ 111 km north of A, C ~ 1112 km north of A
TOY = [{"label": "A", "gps": [0.0, 0.0]},
       {"label": "B", "gps": [1.0, 0.0]},
       {"label": "C", "gps": [10.0, 0.0]}]


@pytest.fixture
def geom():
    return build_geometry(TOY, merge_km=2.0)


@pytest.fixture
def coords():
    return risk.gallery_coords(TOY)


UNI = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}


# ---------------- pointwise ----------------

def test_p_true_and_rank():
    q = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert risk.p_true(q, "A") == 0.5
    assert risk.rank_true(q, "A") == 1
    assert risk.rank_true(q, "B") == 2
    assert risk.rank_true(q, "C") == 3
    assert risk.p_true(q, "Z") == 0.0        # label outside the gallery


def test_logit_clamps():
    assert risk.logit(0.5) == pytest.approx(0.0)
    assert math.isfinite(risk.logit(0.0)) and risk.logit(0.0) < -20
    assert math.isfinite(risk.logit(1.0)) and risk.logit(1.0) > 20


def test_signed_evidence_sign():
    rho = UNI
    # more mass on the truth than the reference -> positive evidence
    assert risk.signed_evidence({"A": 0.8, "B": 0.1, "C": 0.1}, "A", rho) > 0
    # exactly the reference -> zero
    assert risk.signed_evidence(dict(rho), "A", rho) == pytest.approx(0.0)
    # misleading release: less mass on the truth than the reference -> negative
    assert risk.signed_evidence({"A": 0.05, "B": 0.9, "C": 0.05}, "A", rho) < 0


def test_signed_evidence_value():
    q = {"A": 0.5, "B": 0.25, "C": 0.25}
    rho = {"A": 0.25, "B": 0.5, "C": 0.25}
    assert risk.signed_evidence(q, "A", rho) == pytest.approx(
        math.log(0.5 / 0.5) - math.log(0.25 / 0.75))


def test_err_km_and_exp_err(coords):
    q = {"A": 0.2, "B": 0.7, "C": 0.1}
    d_ab = haversine_km(0, 0, 1, 0)
    d_ac = haversine_km(0, 0, 10, 0)
    assert risk.err_km(q, "A", coords) == pytest.approx(d_ab)      # argmax is B
    assert risk.err_km(q, "B", coords) == pytest.approx(0.0)
    assert risk.exp_err_km(q, "A", coords) == pytest.approx(0.7 * d_ab + 0.1 * d_ac)
    assert math.isnan(risk.err_km(q, "Z", coords))


# ---------------- distributional ----------------

def test_mpl_max_ge_mean(geom):
    rep, clusters, dist = geom
    q = {"A": 0.7, "B": 0.2, "C": 0.1}
    mean = risk.mpl_vs(q, UNI, rep, clusters, dist)
    mx = risk.mpl_max_vs(q, UNI, rep, clusters, dist)
    assert mx >= mean > 0


def test_mpl_max_value(geom):
    rep, clusters, dist = geom
    q = {"A": 0.7, "B": 0.2, "C": 0.1}
    llr = {k: math.log(q[k] / UNI[k]) for k in "ABC"}
    pairs = [(abs(llr[a] - llr[b]) / haversine_km(*TOY["ABC".index(a)]["gps"],
                                                  *TOY["ABC".index(b)]["gps"]) * 1000)
             for a, b in (("A", "B"), ("A", "C"), ("B", "C"))]
    assert risk.mpl_max_vs(q, UNI, rep, clusters, dist) == pytest.approx(max(pairs))
    assert risk.mpl_vs(q, UNI, rep, clusters, dist) == pytest.approx(sum(pairs) / 3)


def test_identical_distribution_is_zero_risk(geom):
    rep, clusters, dist = geom
    q = {"A": 0.7, "B": 0.2, "C": 0.1}
    assert risk.mpl_vs(q, q, rep, clusters, dist) == pytest.approx(0.0)
    assert risk.mpl_max_vs(q, q, rep, clusters, dist) == pytest.approx(0.0)
    assert risk.residual_allmask(q, q, rep, clusters, dist) == pytest.approx(0.0)
    for v in risk.eps_local(q, q, rep, clusters, dist).values():
        assert v["mean"] == pytest.approx(0.0) and v["max"] == pytest.approx(0.0)


def test_eps_local_radii(geom):
    rep, clusters, dist = geom
    q = {"A": 0.7, "B": 0.2, "C": 0.1}
    el = risk.eps_local(q, UNI, rep, clusters, dist, radii=(25.0, 200.0, 750.0))
    # A-B is ~111 km: no pair within 25 km, exactly one within 200 km,
    # and still only that one within 750 km (A-C and B-C are ~1000+ km apart)
    assert el[25.0]["n"] == 0
    assert el[200.0]["n"] == 1
    assert el[750.0]["n"] == 1
    assert el[200.0]["mean"] == pytest.approx(el[200.0]["max"])
    # a wider radius can only admit more pairs
    el2 = risk.eps_local(q, UNI, rep, clusters, dist, radii=(2000.0,))
    assert el2[2000.0]["n"] == 3
    assert el2[2000.0]["max"] >= el[200.0]["max"]


def test_eps_local_scales_with_logodds(geom):
    """Doubling every log-odds against rho doubles epsilon."""
    rep, clusters, dist = geom
    q = {"A": 0.7, "B": 0.2, "C": 0.1}
    q2 = {k: v ** 2 for k, v in q.items()}
    z = sum(q2.values())
    q2 = {k: v / z for k, v in q2.items()}
    rho2 = {k: v ** 2 for k, v in UNI.items()}
    z = sum(rho2.values())
    rho2 = {k: v / z for k, v in rho2.items()}
    a = risk.eps_local(q, UNI, rep, clusters, dist)[200.0]["max"]
    b = risk.eps_local(q2, rho2, rep, clusters, dist)[200.0]["max"]
    assert b == pytest.approx(2 * a)


def test_zero_mass_pairs_are_dropped(geom):
    rep, clusters, dist = geom
    q = {"A": 0.5, "B": 0.5, "C": 0.0}
    assert risk.eps_local(q, UNI, rep, clusters, dist, radii=(2000.0,))[2000.0]["n"] == 1


def test_risk_vector_keys(geom, coords):
    rep, clusters, dist = geom
    q = {"A": 0.7, "B": 0.2, "C": 0.1}
    qall = {"A": 0.4, "B": 0.3, "C": 0.3}
    rv = risk.risk_vector(q, "A", UNI, qall, rep, clusters, dist, coords)
    for k in ("R1_signed_evidence", "R2_eps_local_max_200", "R3_p_true", "R4_err_km",
              "R5_residual_allmask", "rank_true", "exp_err_km", "mpl_vs_rho",
              "mpl_max_vs_rho", "eps_local"):
        assert k in rv
    assert rv["R3_p_true"] == 0.7 and rv["R4_err_km"] == pytest.approx(0.0)


# ---------------- vectorised evaluator ----------------

def test_risk_geometry_matches_scalar(geom):
    rep, clusters, dist = geom
    rg = risk.RiskGeometry(TOY, merge_km=2.0)
    assert rg.clusters == clusters
    for q, rho in [({"A": 0.7, "B": 0.2, "C": 0.1}, UNI),
                   ({"A": 0.1, "B": 0.1, "C": 0.8}, {"A": 0.5, "B": 0.3, "C": 0.2}),
                   ({"A": 0.5, "B": 0.5, "C": 0.0}, UNI)]:
        m = rg.metrics(rg.vec(q), rg.vec(rho))
        assert m["mpl"] == pytest.approx(risk.mpl_vs(q, rho, rep, clusters, dist))
        assert m["mpl_max"] == pytest.approx(risk.mpl_max_vs(q, rho, rep, clusters, dist))
        ref = risk.eps_local(q, rho, rep, clusters, dist)
        for r, v in m["eps"].items():
            assert v["n"] == ref[r]["n"]
            assert v["mean"] == pytest.approx(ref[r]["mean"])
            assert v["max"] == pytest.approx(ref[r]["max"])


# ---------------- build_q ----------------

def test_build_q_lattice():
    rec = {"image_id": "x", "n_cues": 3,
           "posterior": {"A": 0.6, "B": 0.3, "C": 0.1},
           "per_cue": [{"prior": {"A": 0.5, "B": 0.4, "C": 0.1}},
                       {"prior": {"A": 0.4, "B": 0.4, "C": 0.2}},
                       {"prior": {"A": 0.3, "B": 0.4, "C": 0.3}}],
           "prior_allmask": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}}
    lat = {"x": {"combos": [{"subset": [0, 1], "prior": {"A": 0.35, "B": 0.4, "C": 0.25}},
                            {"subset": [0, 2], "prior": {"A": 0.34, "B": 0.4, "C": 0.26}},
                            {"subset": [1, 2], "prior": {"A": 0.33, "B": 0.4, "C": 0.27}}]}}
    q = build_q(rec, lat)
    assert len(q) == 2 ** 3
    assert q[frozenset()]["A"] == 0.6
    assert q[frozenset([1])]["A"] == 0.4
    assert q[frozenset([0, 2])]["A"] == 0.34
    assert q[frozenset([0, 1, 2])]["A"] == pytest.approx(1 / 3)
    # missing lattice -> incomplete, detectable by the length check
    assert len(build_q(rec, {})) < 2 ** 3
