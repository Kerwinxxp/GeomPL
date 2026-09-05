"""Attribution on the raw cue list: exact Shapley + formal SII on every maskable image.

The BEFORE side of the de-duplication comparison. Same game and same code path as
`belief_elicit.shapley_v3`, only without merging duplicate-mask cues: v(S) comes from the
sweep (singles + all-masked) plus the lattice (intermediate subsets), v(empty) = 0, and
m <= 5 is enumerated directly. All 95 maskable images are included (at m = 1, phi = v({1}));
single-cue mPL and Shapley are paired on the *same* image / cue set, and nothing is
conditioned on country-hit — the run prints the whole set plus a hit/miss stratification.

Run: python -m belief_elicit.shapley_v2
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from belief_elicit.attribution import shapley, sii            # noqa: F401  (re-exported)
from belief_elicit.results import (LATTICE, SHAPLEY_V2 as OUT, SWEEP,  # noqa: F401
                                   build_v)
from belief_elicit.shapley_v3 import run


def main():
    run(dedup=False, out=OUT, sweep_path=SWEEP, lattice_path=LATTICE)


if __name__ == "__main__":
    main()
