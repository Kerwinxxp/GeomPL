"""Complete the full subset lattice for the 3-5 cue images (needed for exact Shapley).

The sweep covers: clean image, every single cue masked, every cue masked (with the full
distributions). This script fills in the intermediate subsets (2 <= |S| <= m-1), after which
every multi-cue image has all 2^m - 1 values of v(S).
It matches the sweep exactly: same gallery / geometry (2 km), same masking order (the cue
order is asserted identical), and it reuses the posterior stored by the sweep (the meter is
deterministic). Incremental saves + resume.

Run: belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_lattice
"""
import itertools
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from belief_elicit.cues import cue_masks_of
from belief_elicit.geometry import build_geometry, mpl
from belief_elicit.georanker_belief import score_labels
from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.results import (LATTICE as OUT, SAM3_DIR, SWEEP, by_image, image_path,
                                   load_gallery, load_subsets, load_sweep)

MERGE_KM = 2.0


def main():
    gv = load_gallery()
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)
    subset = load_subsets()
    sweep = by_image(load_sweep(SWEEP))

    done = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["image_id"]] = r
        print(f"resume: 已有 {len(done)} 张(可能部分完成)", flush=True)

    targets = [r for r in sweep.values() if r["n_cues"] >= 3]
    total_scorings = sum(2 ** r["n_cues"] - 2 - r["n_cues"] for r in targets)
    print(f"待补 {len(targets)} 张(m>=3),共 {total_scorings} 次打分", flush=True)

    t0 = time.time()
    n_done_scorings = 0
    for r in sorted(targets, key=lambda x: x["n_cues"]):
        iid = r["image_id"]
        m = r["n_cues"]
        cues, _cats, masks, (W, H) = cue_masks_of(SAM3_DIR, iid)
        assert [c for c in cues] == [pc["cue"] for pc in r["per_cue"]], \
            f"cue 顺序不一致: {iid}"
        img = Image.open(image_path(subset[iid])).resize((W, H)).convert("RGB")
        tl = r["true_label"]
        post = r["posterior"]                       # reuse the sweep posterior (deterministic)

        rec_out = done.get(iid, {"image_id": iid, "true_label": tl, "n_cues": m,
                                 "cue_names": cues, "merge_km": MERGE_KM, "combos": []})
        have = {tuple(c["subset"]) for c in rec_out["combos"]}
        need = [S for size in range(2, m)
                for S in itertools.combinations(range(m), size)
                if S not in have]
        if not need:
            done[iid] = rec_out; continue

        for S in need:
            u = np.zeros((H, W), bool)
            for k in S:
                u |= masks[k]
            pri, _ = score_labels(mask_solid_from_masks(img, [u]), gv, variant="B",
                                  batch_size=4)
            rec_out["combos"].append({"subset": list(S),
                                      "cov": float(u.sum() / (W * H)),
                                      "p_true": pri.get(tl, 0.0),
                                      "mpl": mpl(pri, post, rep, clusters, dist),
                                      "prior": pri})
            done[iid] = rec_out
            json.dump(list(done.values()), open(OUT, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            n_done_scorings += 1
            el = time.time() - t0
            eta = el / n_done_scorings * (total_scorings - n_done_scorings) / 60
            print(f"[{n_done_scorings}/{total_scorings}] {tl.split(',')[0][:14]:14s} "
                  f"S={list(S)} mPL={rec_out['combos'][-1]['mpl']:.3f} "
                  f"({el/60:.0f}min, 剩~{eta:.0f}min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
