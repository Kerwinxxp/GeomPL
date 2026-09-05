"""Precompute the LaMa-inpainted image variants (single / pair / all / equal-area control).

The scoring script lives in a different venv (GeoRanker); this one only generates pixels.
Output directory belief_elicit/inpaint_cache/<image_id>/:
    s<k>.png        cue k removed on its own (0-based, in the filtered source-JSON order)
    p<k>-<l>.png    cues k and l removed together (k < l, only when n_cues >= 2)
    all.png         every cue removed
    c<k>-<j>.png    equal-area control for cue k, placement j (same shape, translated) - inpainted
    cg<k>-<j>.png   the **gray-block** version of that same placement (gray vs inpaint, paired)
    manifest.json   cue table + per-file spec / cues / control mask RLE / overlap / coverage

Run (in the cue_extract venv: torch + CUDA + simple_lama_inpainting):
    cue_extract/.venv/Scripts/python.exe -m belief_elicit.precompute_inpaint
Different cue source:
    ... -m belief_elicit.precompute_inpaint --src cue_extract/results_vocab
        --out belief_elicit/inpaint_cache_vocab
"""
import argparse
import json
import os
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

from belief_elicit.cues import cue_masks_of, mask_to_rle, sample_control, translate_mask  # noqa: F401
from belief_elicit.inpaint_ops import DILATE_PX, inpaint_from_masks
from belief_elicit.masking import mask_solid_from_masks
from belief_elicit.results import SWEEP, load_subsets as load_subset_paths  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("cue_extract", "results_sam3"),
                    help="cue source directory (<image_id>.json, results_sam3 schema)")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="only these image_id prefixes (default: every sweep image with n_cues>=1)")
    ap.add_argument("--nctrl", type=int, default=2, help="equal-area control placements per cue")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--dilate", type=int, default=DILATE_PX, help="mask dilation before inpainting")
    ap.add_argument("--out", default=os.path.join("belief_elicit", "inpaint_cache"))
    args = ap.parse_args()

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    out_root = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(out_root, exist_ok=True)

    sweep = json.load(open(SWEEP, encoding="utf-8"))
    all_ids = [r["image_id"] for r in sweep if r["n_cues"] >= 1]
    if args.ids:
        ids = [i for i in all_ids if any(i.startswith(p) for p in args.ids)]
        missing = [p for p in args.ids if not any(i.startswith(p) for i in all_ids)]
        if missing:
            print(f"警告:这些前缀在 sweep 里没匹配到:{missing}", flush=True)
    else:
        ids = all_ids
    print(f"src={src}\nout={out_root}\n待处理 {len(ids)} 张,nctrl={args.nctrl} "
          f"seed={args.seed} dilate={args.dilate}px", flush=True)

    subset = load_subset_paths()
    t0, n_new, n_skip, n_img = time.time(), 0, 0, 0
    for ii, iid in enumerate(ids):
        got = cue_masks_of(src, iid)
        if got is None:
            print(f"[{ii+1}/{len(ids)}] {iid[:24]} 源 JSON 缺失,跳过", flush=True)
            continue
        cues, cats, masks, (W, H) = got
        m = len(cues)
        if m == 0:
            print(f"[{ii+1}/{len(ids)}] {iid[:24]} 无可遮蔽线索,跳过", flush=True)
            continue
        d = os.path.join(out_root, iid)
        os.makedirs(d, exist_ok=True)
        p = subset[iid]["path"]
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        img = Image.open(p).resize((W, H)).convert("RGB")
        union = np.zeros((H, W), bool)
        for m_ in masks:
            union |= m_
        npx = float(W * H)

        # per-image rng seeded by (seed, image_id), so resuming reproduces the placements
        rng = np.random.default_rng([args.seed, zlib.crc32(iid.encode("utf-8"))])

        files = []
        i_new = 0

        def emit(name, sel_masks, entry, gray=False):
            """Write one variant (skipped if it already exists) and record its manifest entry."""
            nonlocal n_new, n_skip, i_new
            fp = os.path.join(d, name)
            if os.path.exists(fp):
                n_skip += 1
            else:
                op = mask_solid_from_masks if gray else \
                    (lambda im, ms: inpaint_from_masks(im, ms, dilate_px=args.dilate))
                op(img, sel_masks).save(fp, format="PNG")
                n_new += 1; i_new += 1
            files.append(entry)

        # --- singles / pairs / all ---
        for k in range(m):
            emit(f"s{k}.png", [masks[k]],
                 {"file": f"s{k}.png", "spec": "single", "op": "inpaint", "cues": [k],
                  "cov": float(masks[k].sum() / npx)})
        for k in range(m):
            for l in range(k + 1, m):
                mm = masks[k] | masks[l]
                emit(f"p{k}-{l}.png", [masks[k], masks[l]],
                     {"file": f"p{k}-{l}.png", "spec": "pair", "op": "inpaint",
                      "cues": [k, l], "cov": float(mm.sum() / npx)})
        emit("all.png", masks,
             {"file": "all.png", "spec": "all", "op": "inpaint",
              "cues": list(range(m)), "cov": float(union.sum() / npx)})

        # --- equal-area controls: same shape translated off the cues; inpainted + gray ---
        for k in range(m):
            for j in range(args.nctrl):
                t, ovf = sample_control(masks[k], union, rng)
                if t is None:
                    print(f"    {iid[:20]} cue{k} ctrl{j}: 采样失败,跳过", flush=True)
                    continue
                emit(f"c{k}-{j}.png", [t],
                     {"file": f"c{k}-{j}.png", "spec": "control", "op": "inpaint",
                      "cues": [k], "placement": j, "overlap_frac": ovf,
                      "cov": float(t.sum() / npx), "mask_rle": mask_to_rle(t)})
                emit(f"cg{k}-{j}.png", [t],
                     {"file": f"cg{k}-{j}.png", "spec": "control", "op": "gray",
                      "cues": [k], "placement": j, "overlap_frac": ovf,
                      "cov": float(t.sum() / npx), "mask_from": f"c{k}-{j}.png"},
                     gray=True)

        json.dump({"image_id": iid, "src": os.path.relpath(src, ROOT).replace("\\", "/"),
                   "image_path": os.path.relpath(p, ROOT).replace("\\", "/"),
                   "image_size": [W, H], "n_cues": m, "nctrl": args.nctrl,
                   "seed": args.seed, "rng": "default_rng([seed, crc32(image_id)])",
                   "dilate_px": args.dilate,
                   "cues": [{"index": k, "cue": cues[k], "category": cats[k],
                             "area_frac": float(masks[k].sum() / npx)} for k in range(m)],
                   "files": files},
                  open(os.path.join(d, "manifest.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        n_img += 1
        el = time.time() - t0
        print(f"[{ii+1}/{len(ids)}] {iid[:28]:28s} m={m} 文件{len(files)} "
              f"(新{i_new}) 累计新{n_new}/跳{n_skip} {el/60:.1f}min", flush=True)

    el = time.time() - t0
    print(f"\n完成:{n_img} 张图,新写 {n_new} 个 PNG,跳过已存在 {n_skip} 个,"
          f"用时 {el/60:.2f} min ({el/max(n_new,1):.2f}s/新文件)", flush=True)


if __name__ == "__main__":
    main()
