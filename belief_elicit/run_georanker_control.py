"""Equal-area control (GeoRanker): measure the masking-artifact floor.

For every real cue, translate its SAM 3 mask to a random non-cue location (same shape and
area, avoiding the union of all cues) and score it through the same pipeline (variant B /
gallery_v2 / 2 km geometry) -> control mPL. If control ~ real cue, then mPL is measuring the
artifact of "something was masked" rather than that cue's information — the failure mode
first observed on the GPT-4o meter, closed-loop-verified here on GeoRanker.

Image selection: 10 images spread evenly over the v(N) quantiles + the NY / Bled / Cuba case
images; nctrl = 3 placements per cue. Incremental saves + resume. The real-cue mPL is reused
from the sweep (same posterior, no extra scoring).

Run: belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_control
"""
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

from belief_elicit.cues import cue_masks_of, sample_control, translate_mask  # noqa: F401
from belief_elicit.geometry import build_geometry, mpl
from belief_elicit.georanker_belief import score_labels
from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.results import (CONTROLS as OUT, SAM3_DIR, SWEEP, by_image, image_path,
                                   load_gallery, load_subsets, load_sweep)

MERGE_KM, SEED, NPICK = 2.0, 42, 10
FORCE = ["158307292", "754780171", "370717727"]      # the NY / Bled / Cuba case images


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="run every maskable image (otherwise sample by quantile)")
    ap.add_argument("--nctrl", type=int, default=3, help="random placements per cue")
    args = ap.parse_args()
    NCTRL = args.nctrl
    gv = load_gallery()
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)
    subset = load_subsets()
    sweep = by_image(load_sweep(SWEEP))

    # ---- image selection: quantiles of v(N) + the forced case images ----
    cand = sorted((r for r in sweep.values() if r["n_cues"] >= 1),
                  key=lambda r: r["mpl_all"])
    if args.all:
        picks = [r["image_id"] for r in cand]
        print(f"全量模式:{len(picks)} 张", flush=True)
    else:
        idx = np.linspace(0, len(cand) - 1, NPICK).round().astype(int)
        picks = [cand[i]["image_id"] for i in idx]
        for pref in FORCE:
            iid = next((k for k in sweep if k.startswith(pref)), None)
            if iid and iid not in picks:
                picks.append(iid)
        print(f"选图 {len(picks)} 张(按 v(N) 分位 + 案例图)", flush=True)

    done = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT, encoding="utf-8")):
            done[r["image_id"]] = r
        print(f"resume: 已有 {len(done)} 张", flush=True)

    rng = np.random.default_rng(SEED)
    t0, nsc = time.time(), 0
    total = sum(sweep[i]["n_cues"] * NCTRL for i in picks if i not in done)
    for iid in picks:
        if iid in done:
            continue
        r = sweep[iid]
        cues, _cats, masks, (W, H) = cue_masks_of(SAM3_DIR, iid)
        assert cues == [pc["cue"] for pc in r["per_cue"]], f"cue 顺序不一致 {iid}"
        union = np.zeros((H, W), bool)
        for m_ in masks:
            union |= m_
        img = Image.open(image_path(subset[iid])).resize((W, H)).convert("RGB")
        post = r["posterior"]

        rec_out = {"image_id": iid, "true_label": r["true_label"],
                   "country_hit": r["country_hit"], "n_cues": r["n_cues"],
                   "vN": r["mpl_all"], "cues": []}
        for k, (name, m_) in enumerate(zip(cues, masks)):
            ctrls = []
            for j in range(NCTRL):
                t, ovf = sample_control(m_, union, rng)
                if t is None:
                    continue
                pri, _ = score_labels(mask_solid_from_masks(img, [t]), gv, variant="B",
                                      batch_size=4)
                ctrls.append({"mpl": mpl(pri, post, rep, clusters, dist),
                              "overlap_frac": ovf,
                              "cov": float(t.sum() / (W * H))})
                nsc += 1
                el = time.time() - t0
                print(f"[{nsc}/{total}] {r['true_label'].split(',')[0][:12]:12s} "
                      f"cue{k+1} ctrl{j+1} mPL={ctrls[-1]['mpl']:.3f} "
                      f"(real={r['per_cue'][k]['mpl']:.3f}) "
                      f"({el/60:.0f}min 剩~{el/nsc*(total-nsc)/60:.0f}min)", flush=True)
            rec_out["cues"].append({"cue": name, "category": r["per_cue"][k]["category"],
                                    "area_frac": float(m_.sum() / (W * H)),
                                    "real_mpl": r["per_cue"][k]["mpl"],
                                    "controls": ctrls})
        done[iid] = rec_out
        json.dump(list(done.values()), open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
