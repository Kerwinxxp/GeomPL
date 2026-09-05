"""Unit tests for the belief_elicit library modules (results / cues / attribution).

Pure CPU logic only: the set function v(S) built from a sweep record, the single cue-mask
reader, and the defining properties of the attribution operators.
"""
import glob
import itertools
import json
import os

import numpy as np
import pytest

from belief_elicit.attribution import (banzhaf, empty_interaction, order2_shapley, shapley,
                                       sii, spearman)
from belief_elicit.cues import cue_masks_of
from belief_elicit.results import SAM3_DIR, build_v

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ---------- results.build_v ----------

def _sweep_record(m, singles, mpl_all=None):
    return {"image_id": "img", "n_cues": m,
            "per_cue": [{"cue": f"c{k}", "category": "x", "mpl": s}
                        for k, s in enumerate(singles)],
            "mpl_all": mpl_all}


def test_build_v_single_cue_needs_no_lattice():
    v, ok = build_v(_sweep_record(1, [0.4]), {})
    assert ok is True
    assert v[frozenset()] == 0.0
    assert v[frozenset([0])] == 0.4


def test_build_v_two_cues_uses_mpl_all_as_vN():
    v, ok = build_v(_sweep_record(2, [0.4, 0.1], mpl_all=0.45), {})
    assert ok is True
    assert v[frozenset([0, 1])] == 0.45
    assert set(v) == {frozenset(), frozenset([0]), frozenset([1]), frozenset([0, 1])}


def test_build_v_three_cues_fills_intermediates_from_lattice():
    rec = _sweep_record(3, [0.4, 0.1, 0.2], mpl_all=0.5)
    lattice = {"img": {"combos": [{"subset": [0, 1], "mpl": 0.42},
                                  {"subset": [0, 2], "mpl": 0.47},
                                  {"subset": [1, 2], "mpl": 0.25}]}}
    v, ok = build_v(rec, lattice)
    assert ok is True
    assert len(v) == 2 ** 3                       # the full lattice, empty set included
    assert v[frozenset([1, 2])] == 0.25


def test_build_v_reports_incomplete_lattice():
    rec = _sweep_record(3, [0.4, 0.1, 0.2], mpl_all=0.5)
    v, ok = build_v(rec, {"img": {"combos": [{"subset": [0, 1], "mpl": 0.42}]}})
    assert ok is False                            # 1 of the 3 intermediate subsets present
    v, ok = build_v(rec, {})
    assert ok is False                            # no lattice entry for this image at all


# ---------- cues.cue_masks_of ----------

def test_cue_masks_of_missing_source_returns_none():
    assert cue_masks_of(SAM3_DIR, "no-such-image-id") is None


@pytest.mark.skipif(not glob.glob(os.path.join(SAM3_DIR, "*.json")),
                    reason="cue_extract/results_sam3 not present")
def test_cue_masks_of_matches_the_filtering_rule_on_a_real_record():
    path = sorted(glob.glob(os.path.join(SAM3_DIR, "*.json")))[0]
    iid = os.path.basename(path)[:-len(".json")]
    cues, cats, masks, (W, H) = cue_masks_of(SAM3_DIR, iid)

    rec = json.load(open(path, encoding="utf-8"))
    assert [W, H] == rec["image_size"]
    expected = [c["cue"] for c in rec["geo_privacy_cues"]
                if c.get("maskable")
                and any(not i.get("degenerate") and i.get("mask_rle") for i in c["instances"])]
    assert cues == expected
    assert len(cats) == len(masks) == len(cues)
    for m in masks:
        assert m.dtype == np.bool_ and m.shape == (H, W)
        assert m.any()                            # empty unions are dropped


# ---------- attribution ----------

def _game(m, values):
    """{tuple: value} -> the frozenset-keyed v dict the operators expect."""
    v = {frozenset(): 0.0}
    v.update({frozenset(k): val for k, val in values.items()})
    assert len(v) == 2 ** m
    return v


def test_shapley_efficiency_and_symmetry_on_a_tiny_game():
    # 3 players; 0 and 1 overlap heavily, 2 is independent and additive
    v = _game(3, {(0,): 0.6, (1,): 0.6, (2,): 0.2,
                  (0, 1): 0.7, (0, 2): 0.8, (1, 2): 0.8,
                  (0, 1, 2): 0.9})
    phi = shapley(v, 3)
    assert sum(phi) == pytest.approx(v[frozenset(range(3))], abs=1e-12)   # efficiency
    assert phi[0] == pytest.approx(phi[1], abs=1e-12)                     # symmetry
    assert phi[2] == pytest.approx(0.2, abs=1e-12)                        # 2 is a null add-on


def test_shapley_of_an_additive_game_is_the_singleton_values():
    a = [0.3, 0.5, 0.11]
    v = _game(3, {S: sum(a[k] for k in S)
                  for size in (1, 2, 3) for S in itertools.combinations(range(3), size)})
    assert shapley(v, 3) == pytest.approx(a, abs=1e-12)
    assert banzhaf(v, 3) == pytest.approx(a, abs=1e-12)
    assert all(abs(x) < 1e-12 for x in empty_interaction(v, 3).values())
    assert all(abs(sii(v, 3, k, l)) < 1e-12 for k, l in itertools.combinations(range(3), 2))


def test_sii_sign_separates_overlap_from_backup():
    over = _game(2, {(0,): 0.5, (1,): 0.5, (0, 1): 0.6})      # sub-additive: overlap
    back = _game(2, {(0,): 0.2, (1,): 0.2, (0, 1): 0.7})      # super-additive: backup
    assert sii(over, 2, 0, 1) == pytest.approx(-0.4, abs=1e-12)
    assert sii(back, 2, 0, 1) == pytest.approx(0.3, abs=1e-12)
    assert empty_interaction(over, 2)["0,1"] == pytest.approx(sii(over, 2, 0, 1), abs=1e-12)


def test_order2_shapley_is_exact_for_three_players_and_anchors_to_vN():
    v = _game(3, {(0,): 0.6, (1,): 0.4, (2,): 0.2,
                  (0, 1): 0.7, (0, 2): 0.75, (1, 2): 0.5,
                  (0, 1, 2): 0.85})
    exact = shapley(v, 3)
    o2 = order2_shapley([v[frozenset([k])] for k in range(3)],
                        {(k, l): v[frozenset([k, l])]
                         for k, l in itertools.combinations(range(3), 2)},
                        v[frozenset(range(3))])
    assert sum(o2["phi"]) == pytest.approx(0.85, abs=1e-12)   # anchored to v(N)
    assert o2["phi"] == pytest.approx(exact, abs=1e-12)       # exact at m <= 3


def test_spearman_monotone_and_short_input():
    assert spearman([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1.0], [2.0]))
    # ranks come from argsort-of-argsort, so ties are broken by position rather than averaged
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
