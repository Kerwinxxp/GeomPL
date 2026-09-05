"""Per-cue mPL under the LaMa-inpaint operator: batch-score the precomputed image variants.

Gray-block masking introduces a tampering artifact (the equal-area control shows that
graying out an irrelevant region also moves the posterior a lot). This script reruns the
same measurement on **pre-inpainted image variants**: variant image -> GeoRanker variant B /
gallery_v2 -> prior; the posterior is reused from the sweep (the meter is deterministic, so
the clean image is never re-scored) -> 2 km geometry -> mPL.

Cache contract (belief_elicit/inpaint_cache/<image_id>/):
  s<k>.png       cue k removed
  p<k>-<l>.png   cue pair (k, l) removed
  all.png        every cue removed
  c<k>-<j>.png   equal-area control placement j for cue k (inpainted)
  cg<k>-<j>.png  the same placement filled with gray (a direct inpaint-vs-gray comparison)
  manifest.json  per-file metadata (spec / cue indices / control mask RLE / overlap / coverage)
A missing or partial manifest falls back to parsing the filename; files not on disk are
skipped (so the cache is allowed to still be generating).

Output: per variant, the **full prior distribution** ({label: prob}, 138 entries) - the
distribution must be persisted, the downstream Shapley and distribution-level analyses need
it, a scalar is not enough. Written after every scoring; restarts resume at **spec level**
(a spec already scored for an image is not scored again).

Run (note the offline env vars, so HF cannot hang on the network):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.run_georanker_inpaint --part main
"""
import argparse
import json
import os
import re
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
from belief_elicit.results import (INPAINT, INPAINT_CACHE, INPAINT_CONTROLS, SWEEP,
                                   load_gallery, load_sweep)
from belief_elicit.results import manifest_index as load_manifest      # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
MERGE_KM = 2.0
VARIANT = "B"
BATCH_SIZE = 4

RE_ALL = re.compile(r"^all$")
RE_S = re.compile(r"^s(\d+)$")
RE_P = re.compile(r"^p(\d+)-(\d+)$")
RE_C = re.compile(r"^c(\d+)-(\d+)$")
RE_CG = re.compile(r"^cg(\d+)-(\d+)$")


# ---------------- cache manifest parsing ----------------

def parse_stem(stem):
    """Filename stem -> (kind, cue indices, placement); None when unrecognised.

    kind in {all, single, pair, ctrl_inpaint, ctrl_gray}; the first three belong to `main`,
    the last two to `control`.
    """
    m = RE_ALL.match(stem)
    if m:
        return "all", None, None
    m = RE_S.match(stem)
    if m:
        return "single", [int(m.group(1))], None
    m = RE_P.match(stem)
    if m:
        return "pair", [int(m.group(1)), int(m.group(2))], None
    m = RE_CG.match(stem)                      # must precede RE_C ("cg" also matches "c")
    if m:
        return "ctrl_gray", [int(m.group(1))], int(m.group(2))
    m = RE_C.match(stem)
    if m:
        return "ctrl_inpaint", [int(m.group(1))], int(m.group(2))
    return None


PART_OF = {"all": "main", "single": "main", "pair": "main",
           "ctrl_inpaint": "control", "ctrl_gray": "control"}


def pick(entry, keys, default=None):
    for k in keys:
        if entry.get(k) is not None:
            return entry[k]
    return default


def list_specs(dirpath, part, ops="both"):
    """List this image's cached variants belonging to `part`, in a stable order.

    ops in {inpaint, gray, both} filters the **controls** (c*/cg*) only: a control's op comes
    from the manifest `op` field, falling back to the filename prefix (cg -> gray,
    c -> inpaint). The main part (s*/p*/all) is unaffected by ops.
    """
    manifest = load_manifest(dirpath)
    try:
        files = sorted(os.listdir(dirpath))
    except OSError:
        return []
    items = []
    for fn in files:
        if not fn.lower().endswith(".png"):
            continue
        stem = os.path.splitext(fn)[0]
        parsed = parse_stem(stem)
        if parsed is None:
            print(f"  [warn] 无法识别的缓存文件名,跳过: {fn}", flush=True)
            continue
        kind, cues, place = parsed
        if part != "all" and PART_OF[kind] != part:
            continue
        e = manifest.get(fn, {})
        cues_m = pick(e, ["cues", "cue_indices", "cue_idx", "cue_ids", "k"])
        if isinstance(cues_m, int):
            cues_m = [cues_m]
        if isinstance(cues_m, list) and cues_m:
            cues = [int(c) for c in cues_m]
        # the manifest "spec" is a *category* (single/pair/all/control) shared by several
        # files, so the unique id has to be the filename stem (resume dedupes on it).
        op = pick(e, ["op"])
        if PART_OF[kind] == "control" and op in ("gray", "inpaint"):
            kind = "ctrl_gray" if op == "gray" else "ctrl_inpaint"
        if (PART_OF[kind] == "control" and ops != "both"
                and ("gray" if kind == "ctrl_gray" else "inpaint") != ops):
            continue
        items.append({
            "spec": stem,
            "file": fn,
            "kind": kind,
            "spec_kind": pick(e, ["spec"]),
            "op": op or ("gray" if kind == "ctrl_gray" else "inpaint"),
            "cues": cues,
            "placement": pick(e, ["placement", "j", "ctrl_idx"], place),
            "coverage": pick(e, ["coverage", "cov", "area_frac"]),
            "overlap_frac": pick(e, ["overlap_frac", "overlap"]),
        })
    order = {"single": 0, "pair": 1, "all": 2, "ctrl_inpaint": 3, "ctrl_gray": 3}

    def key(it):
        c = it["cues"] or []
        return (order[it["kind"]], c, it["placement"] if it["placement"] is not None else -1,
                0 if it["kind"] != "ctrl_gray" else 1, it["spec"])

    return sorted(items, key=key)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=INPAINT_CACHE,
                    help="inpaint cache root (holds one <image_id>/ directory per image)")
    ap.add_argument("--part", choices=["main", "control", "all"], default="main",
                    help="main = s*/p*/all; control = c*/cg*; all = both")
    ap.add_argument("--ops", choices=["inpaint", "gray", "both"], default="both",
                    help="applies to the **controls** only: inpaint = score c* only; "
                         "gray = score cg* only; both (default) = score both. Use it to "
                         "halve the control scoring budget")
    ap.add_argument("--ids", nargs="*", default=None, help="filter by image_id prefix")
    ap.add_argument("--out", default=None, help="result JSON (default chosen by --part)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on scorings this run (0 = no cap; for smoke tests)")
    ap.add_argument("--sweep", default=SWEEP,
                    help="sweep results: supplies the posterior and per-cue metadata")
    ap.add_argument("--no-manifest-ok", action="store_true",
                    help="also run images without a manifest.json (skipped by default: such "
                         "an image may still be generating)")
    args = ap.parse_args()

    out_path = args.out or (INPAINT_CONTROLS if args.part == "control" else INPAINT)

    sweep = load_sweep(args.sweep)
    order = [r["image_id"] for r in sweep]
    sw = {r["image_id"]: r for r in sweep}
    if args.ids:
        order = [i for i in order if any(i.startswith(p) for p in args.ids)]
        if not order:
            print(f"没有匹配 --ids {args.ids} 的图", flush=True)
            return

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
        recs += [r for i, r in done.items() if i not in order]     # keep other images' results
        tmp = out_path + ".tmp"
        json.dump(recs, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, out_path)

    # ---- take stock of what is left ----
    todo = []
    n_nomani = 0
    for iid in order:
        d = os.path.join(args.cache, iid)
        if not os.path.isdir(d):
            continue
        # precompute writes manifest.json **last**: no manifest means the image is still
        # being generated and its PNGs may be half-written -> skip (rerun once the cache is
        # complete and it will be picked up).
        if not args.no_manifest_ok and not os.path.exists(os.path.join(d, "manifest.json")):
            n_nomani += 1
            continue
        for it in list_specs(d, args.part, args.ops):
            if it["spec"] in done_specs.get(iid, set()):
                continue
            todo.append((iid, d, it))
    if args.limit and args.limit > 0:
        todo = todo[:args.limit]
    total = len(todo)
    if n_nomani:
        print(f"[warn] {n_nomani} 张图无 manifest.json(缓存仍在生成?),本次跳过", flush=True)
    ops_note = "" if args.ops == "both" else f", ops={args.ops}"
    print(f"待打分 {total} 个变体(part={args.part}{ops_note}, cache={args.cache})",
          flush=True)
    if total == 0:
        print("done (无待办)", flush=True)
        return

    t0, nsc = time.time(), 0
    cur_iid = None
    for iid, d, it in todo:
        r = sw[iid]
        if iid not in done:
            done[iid] = {
                "image_id": iid, "true_label": r["true_label"],
                "country_hit": r["country_hit"], "km_error": r["km_error"],
                "n_cues": r["n_cues"], "merge_km": MERGE_KM, "variant": VARIANT,
                "cue_names": [pc["cue"] for pc in r["per_cue"]],
                "cue_categories": [pc["category"] for pc in r["per_cue"]],
                "mpl_all_gray": r["mpl_all"],
                "variants": [],
            }
            done_specs.setdefault(iid, set())
        if iid != cur_iid:
            cur_iid = iid
            print(f"\n#### {r['true_label'].split(',')[0][:20]} ({iid[:12]}) "
                  f"{r['n_cues']} cues ####", flush=True)

        img = Image.open(os.path.join(d, it["file"])).convert("RGB")
        ts = time.time()
        prior, _ = score_labels(img, gv, variant=VARIANT, batch_size=BATCH_SIZE)
        secs = time.time() - ts
        val = mpl(prior, r["posterior"], rep, clusters, dist)

        cues = it["cues"] or []
        rec = {
            "spec": it["spec"], "file": it["file"], "kind": it["kind"],
            "op": it["op"],
            "cues": cues,
            "cue_names": [r["per_cue"][k]["cue"] for k in cues
                          if 0 <= k < len(r["per_cue"])],
            "mpl": val,
            "seconds": secs,
            "prior": prior,                      # the full distribution, must be persisted
        }
        if it["coverage"] is not None:
            rec["coverage"] = it["coverage"]
        if it["placement"] is not None:
            rec["placement"] = it["placement"]
        if PART_OF[it["kind"]] == "control":
            rec["overlap_frac"] = it["overlap_frac"]
        # gray baseline: for singles, carry the sweep's gray mPL for a direct comparison
        if it["kind"] == "single" and 0 <= cues[0] < len(r["per_cue"]):
            rec["gray_mpl"] = r["per_cue"][cues[0]]["mpl"]

        done[iid]["variants"].append(rec)
        done_specs[iid].add(it["spec"])
        save()                                   # write after every scoring (spec-level resume)

        nsc += 1
        el = time.time() - t0
        ref = f" (gray={rec['gray_mpl']:.3f})" if "gray_mpl" in rec else ""
        print(f"[{nsc}/{total}] {it['spec']:10s} {it['kind']:12s} "
              f"mPL={val:.4f}{ref} {secs:.0f}s "
              f"({el/60:.0f}min 剩~{el/nsc*(total-nsc)/60:.0f}min)", flush=True)

    save()
    print(f"\ndone → {out_path}", flush=True)


if __name__ == "__main__":
    main()
