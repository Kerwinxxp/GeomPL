"""GeoRanker scoring for the fixed-vocabulary inpaint cache - a thin wrapper over
run_georanker_inpaint.py.

**Why it is needed**: run_georanker_inpaint.py takes all of its cue metadata from the
sweep's `per_cue` (`r["per_cue"][k]["cue"]` / `["category"]` / `["mpl"]`, `r["n_cues"]`,
`r["mpl_all"]`). inpaint_cache_vocab comes from cue_extract/results_vocab, whose **cue list
is completely different** from the sweep's (checked image by image across the 10: sweep 3
cues vs vocabulary 7, 4 vs 6, 3 vs 1, 1 vs 2, ...; even the first cue name differs).
Running it directly would:

  1. take `cue_names` / `cue_categories` / `n_cues` from the sweep's cue list -> **silently
     mislabelled**;
  2. write `gray_mpl` as **another cue's** gray mPL (when the vocabulary has fewer cues);
  3. when the vocabulary has more cues, let `if 0 <= k < len(r["per_cue"])` silently empty
     or truncate cue_names without an error;
  4. still default `--out` to georanker_inpaint_results.json -> **collide with the running
     main job** (variants for the same image_id would be merged into one record, with
     identical spec names).

This wrapper reuses run_georanker_inpaint's cache parsing (parse_stem / list_specs /
PART_OF) plus its scoring and geometry, and changes only three things:
  * cue name / category / area always come from `<cache>/<image_id>/manifest.json` cues[];
  * no `gray_mpl` / `mpl_all_gray` (this arm has no gray baseline), and `n_cues` from the
    manifest;
  * output defaults to georanker_inpaint_vocab_results.json / ..._vocab_control_results.json.
The sweep is still read, but **only** for the posterior (the meter is deterministic, so the
clean image need not be re-scored) and the image-level true_label / country_hit / km_error,
none of which depend on the cue list.

Downstream analysis:
  python -m belief_elicit.inpaint_report --results belief_elicit/georanker_inpaint_vocab_results.json \
         --controls belief_elicit/georanker_inpaint_vocab_control_results.json \
         --cues-from manifest --cache belief_elicit/inpaint_cache_vocab

Run (GPU environment, note the offline env vars):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint_vocab --part main
"""
import argparse
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

from PIL import Image

from belief_elicit.geometry import build_geometry, mpl
from belief_elicit.georanker_belief import score_labels
from belief_elicit.results import (INPAINT, INPAINT_CACHE_VOCAB, INPAINT_CONTROLS, SWEEP,
                                   by_image, load_gallery, load_manifest, load_sweep)
from belief_elicit.run_georanker_inpaint import (BATCH_SIZE, MERGE_KM, PART_OF,
                                                 VARIANT, list_specs)

HERE = os.path.dirname(os.path.abspath(__file__))


def read_cue_table(dirpath):
    """manifest.json -> [{cue, category, area_frac}, ...]; None if missing or malformed."""
    p = os.path.join(dirpath, "manifest.json")
    d = load_manifest(dirpath)
    if not isinstance(d, dict):
        return None
    cs = d.get("cues")
    if not isinstance(cs, list) or not cs:
        print(f"  [warn] manifest 里没有 cues[]:{p}", flush=True)
        return None
    out = []
    for i, c in enumerate(cs):
        if not isinstance(c, dict) or not c.get("cue"):
            print(f"  [warn] manifest cues[{i}] 形态不对:{p}", flush=True)
            return None
        out.append({"cue": c["cue"], "category": c.get("category") or "unknown",
                    "area_frac": c.get("area_frac")})
    n = d.get("n_cues")
    if isinstance(n, int) and n != len(out):
        print(f"  [warn] manifest n_cues={n} 与 cues[] 长度 {len(out)} 不一致:{p}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=INPAINT_CACHE_VOCAB)
    ap.add_argument("--part", choices=["main", "control", "all"], default="main")
    ap.add_argument("--ops", choices=["inpaint", "gray", "both"], default="both",
                    help="applies to the **controls** only: inpaint = c* only; "
                         "gray = cg* only; both = both")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sweep", default=SWEEP,
                    help="only for the posterior and image-level metadata; the cue table "
                         "always comes from the manifest")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        HERE, "georanker_inpaint_vocab_control_results.json" if args.part == "control"
        else "georanker_inpaint_vocab_results.json")
    if os.path.abspath(out_path) in (os.path.abspath(INPAINT),
                                     os.path.abspath(INPAINT_CONTROLS)):
        print("[fatal] --out 指向了 SAM3 口径的主结果文件;词表口径必须写到单独的文件"
              "(两者 spec 名相同、线索表不同,混在一起就废了)")
        return 1

    sweep = by_image(load_sweep(args.sweep))
    try:
        order = sorted(d for d in os.listdir(args.cache)
                       if os.path.isdir(os.path.join(args.cache, d)))
    except OSError as e:
        print(f"[fatal] 读不了缓存目录 {args.cache}: {e}")
        return 1
    if args.ids:
        order = [i for i in order if any(i.startswith(p) for p in args.ids)]
    if not order:
        print(f"缓存目录里没有可用的图:{args.cache}")
        return 1

    gv = load_gallery()
    rep, clusters, dist = build_geometry(gv, merge_km=MERGE_KM)

    # ---- resume, at spec level ----
    done = {}
    if os.path.exists(out_path):
        try:
            for r in json.load(open(out_path, encoding="utf-8")):
                done[r["image_id"]] = r
        except Exception as e:
            print(f"[warn] 旧结果不可读({e}),从头开始", flush=True)
            done = {}
    done_specs = {i: {v["spec"] for v in r.get("variants", [])} for i, r in done.items()}
    if done:
        print(f"resume: 已有 {len(done)} 张 / "
              f"{sum(len(s) for s in done_specs.values())} 个 spec", flush=True)

    def save():
        recs = [done[i] for i in order if i in done]
        recs += [r for i, r in done.items() if i not in order]
        tmp = out_path + ".tmp"
        json.dump(recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)

    # ---- take stock of what is left ----
    todo, cue_tables, n_skip = [], {}, 0
    for iid in order:
        d = os.path.join(args.cache, iid)
        cues = read_cue_table(d)
        if cues is None:                       # no manifest = the cache may still be generating
            n_skip += 1
            continue
        if iid not in sweep:                   # without a posterior there is no mPL
            print(f"  [warn] {iid[:16]} 不在 sweep 里(没有 posterior),跳过", flush=True)
            n_skip += 1
            continue
        cue_tables[iid] = cues
        for it in list_specs(d, args.part, args.ops):
            if it["spec"] in done_specs.get(iid, set()):
                continue
            bad = [k for k in (it["cues"] or []) if not (0 <= k < len(cues))]
            if bad:
                print(f"  [warn] {iid[:16]} {it['spec']}: 线索索引 {bad} 超出 manifest "
                      f"的 {len(cues)} 条,跳过", flush=True)
                continue
            todo.append((iid, d, it))
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
    total = len(todo)
    if n_skip:
        print(f"[warn] {n_skip} 张图无 manifest / 无 posterior,本次跳过", flush=True)
    ops_note = "" if args.ops == "both" else f", ops={args.ops}"
    print(f"待打分 {total} 个变体(part={args.part}{ops_note}, cache={args.cache}) "
          f"→ {out_path}", flush=True)
    if total == 0:
        print("done (无待办)", flush=True)
        return 0

    t0, nsc, cur_iid = time.time(), 0, None
    for iid, d, it in todo:
        r = sweep[iid]
        cues_meta = cue_tables[iid]
        if iid not in done:
            done[iid] = {
                "image_id": iid, "true_label": r["true_label"],
                "country_hit": r["country_hit"], "km_error": r["km_error"],
                "n_cues": len(cues_meta), "merge_km": MERGE_KM, "variant": VARIANT,
                "cue_source": "manifest", "cache": os.path.abspath(args.cache),
                "cue_names": [c["cue"] for c in cues_meta],
                "cue_categories": [c["category"] for c in cues_meta],
                "cue_area_fracs": [c["area_frac"] for c in cues_meta],
                # this arm has no gray baseline: no mpl_all_gray / gray_mpl
                "variants": [],
            }
            done_specs.setdefault(iid, set())
        if iid != cur_iid:
            cur_iid = iid
            print(f"\n#### {r['true_label'].split(',')[0][:20]} ({iid[:12]}) "
                  f"{len(cues_meta)} vocab cues ####", flush=True)

        img = Image.open(os.path.join(d, it["file"])).convert("RGB")
        ts = time.time()
        prior, _ = score_labels(img, gv, variant=VARIANT, batch_size=BATCH_SIZE)
        secs = time.time() - ts
        val = mpl(prior, r["posterior"], rep, clusters, dist)

        ks = it["cues"] or []
        rec = {"spec": it["spec"], "file": it["file"], "kind": it["kind"],
               "op": it["op"], "cues": ks,
               "cue_names": [cues_meta[k]["cue"] for k in ks],
               "cue_categories": [cues_meta[k]["category"] for k in ks],
               "mpl": val, "seconds": secs, "prior": prior}
        if it["coverage"] is not None:
            rec["coverage"] = it["coverage"]
        if it["placement"] is not None:
            rec["placement"] = it["placement"]
        if PART_OF[it["kind"]] == "control":
            rec["overlap_frac"] = it["overlap_frac"]

        done[iid]["variants"].append(rec)
        done_specs[iid].add(it["spec"])
        save()

        nsc += 1
        el = time.time() - t0
        print(f"[{nsc}/{total}] {it['spec']:10s} {it['kind']:12s} mPL={val:.4f} "
              f"{secs:.0f}s ({el/60:.0f}min 剩~{el/nsc*(total-nsc)/60:.0f}min)", flush=True)

    save()
    print(f"\ndone → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
