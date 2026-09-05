"""Coverage analysis: how much of the GPT-4o (top-down) cue inventory does the fixed
vocabulary (bottom-up) cover?

Per image, take the GPT-4o cue masks from results_sam3 and the vocabulary cue masks from
results_vocab (both via cue_extract.common.cue_masks: maskable cues, union of non-degenerate
instances), compute pairwise IoU and pixel recall = |gpt and vocab| / |gpt|, and keep the
best match.

An "important" GPT-4o cue is one whose phi in shapley_v2_results.json is above that image's
median. In the other direction, a vocabulary cue that matches no GPT-4o cue is a candidate
cue GPT-4o may have missed.

Run: cue_extract/.venv/Scripts/python.exe -m cue_extract.compare_vocab
     (or any python with numpy; no GPU needed)
"""
import argparse
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from cue_extract.common import cue_masks

SAM3DIR = os.path.join(os.path.dirname(__file__), "results_sam3")
VOCABDIR = os.path.join(os.path.dirname(__file__), "results_vocab")
SHAPLEY = os.path.join(ROOT, "belief_elicit", "shapley_v2_results.json")
SWEEP = os.path.join(ROOT, "belief_elicit", "georanker_sweep_results.json")

IOU_HIT = 0.5          # "匹配上"的严格阈值(几何基本重合)
REC_HIT = 0.7          # 像素召回阈值(词表盖住了 GPT-4o 线索的大部分像素)
IOU_MATCHED = 0.1      # 判定"词表线索有对应 GPT-4o 线索"的宽松阈值


def cue_rows(path):
    """cue_extract.common.cue_masks in the [(cue, category, mask)] shape this report uses."""
    cues, cats, masks, size = cue_masks(path)
    return list(zip(cues, [c if c is not None else "?" for c in cats], masks)), size


def iou_recall(g, v):
    inter = float(np.logical_and(g, v).sum())
    union = float(np.logical_or(g, v).sum())
    return (inter / union if union else 0.0), (inter / float(g.sum()) if g.sum() else 0.0)


def analyse(iid, phi_by_cue, label):
    gpt, (W, H) = cue_rows(os.path.join(SAM3DIR, iid + ".json"))
    voc, _ = cue_rows(os.path.join(VOCABDIR, iid + ".json"))
    total = float(W * H)
    phis = [phi_by_cue.get(c, float("nan")) for c, _, _ in gpt]
    known = [p for p in phis if p == p]
    med = statistics.median(known) if known else float("nan")

    # "重要" = phi 严格高于本图中位数;若因并列/单线索而无人入选,则退回取本图 phi 最大者
    # (否则单线索图和 phi 全相等的图会整张从"重要"统计里消失)。
    strict = [p for p in known if p > med]
    top = max(known) if known else float("nan")

    def is_important(p):
        if p != p:
            return False
        return p > med if strict else p == top

    rows = []
    for (name, cat, gm), phi in zip([(a, b, c) for a, b, c in gpt], phis):
        best = ("-", 0.0, 0.0)
        for vname, _, vm in voc:
            i, r = iou_recall(gm, vm)
            if i > best[1]:
                best = (vname, i, r)
        # 召回最好的词表线索(未必是 IoU 最好的那条)
        brec = max(((vn, iou_recall(gm, vm)[1]) for vn, _, vm in voc),
                   key=lambda t: t[1], default=("-", 0.0))
        rows.append({"cue": name, "category": cat, "phi": phi,
                     "important": is_important(phi),
                     "area_frac": float(gm.sum() / total),
                     "best_iou_cue": best[0], "iou": best[1],
                     "best_rec_cue": brec[0], "recall": max(best[2], brec[1])})

    unmatched = []
    for vname, vcat, vm in voc:
        bi = max((iou_recall(gm, vm)[0] for _, _, gm in gpt), default=0.0)
        # 词表线索被 GPT-4o 覆盖了多少(反向召回)
        br = max((float(np.logical_and(gm, vm).sum()) / float(vm.sum())
                  for _, _, gm in gpt), default=0.0) if vm.sum() else 0.0
        if bi < IOU_MATCHED:
            unmatched.append({"cue": vname, "category": vcat,
                              "area_frac": float(vm.sum() / total),
                              "best_iou": bi, "covered_frac": br})
    return {"image_id": iid, "label": label, "size": [W, H], "median_phi": med,
            "n_gpt": len(gpt), "n_vocab": len(voc), "rows": rows,
            "vocab_cues": [v[0] for v in voc], "unmatched_vocab": unmatched}


def _fmt(x, n=3):
    """Fixed-point float, or "n/a" for nan (this report never sees None or inf)."""
    return "n/a" if x != x else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(VOCABDIR, "coverage_report.md"))
    args = ap.parse_args()

    shap = {r["image_id"]: r for r in json.load(open(SHAPLEY, encoding="utf-8"))}
    sweep = {r["image_id"]: r for r in json.load(open(SWEEP, encoding="utf-8"))}
    ids = sorted(os.path.basename(p)[:-5]
                 for p in glob.glob(os.path.join(VOCABDIR, "*.jpg.json")))
    if not ids:
        raise SystemExit("results_vocab is empty — run extract_vocab.py first")

    reports = []
    for iid in ids:
        phi = {c["cue"]: c["phi"] for c in shap.get(iid, {}).get("cues", [])}
        reports.append(analyse(iid, phi, sweep.get(iid, {}).get("true_label", "?")))
    reports.sort(key=lambda r: r["label"])

    L = ["# Fixed-vocabulary (bottom-up) vs GPT-4o (top-down) cue coverage", "",
         f"Images: {len(reports)}. GPT-4o cues from `cue_extract/results_sam3/`, "
         f"vocabulary cues from `cue_extract/results_vocab/`, both masked with the "
         f"`cue_masks_of` convention (maskable cue, union of non-degenerate instances).",
         f"`phi` = Shapley value from `belief_elicit/shapley_v2_results.json`; "
         f"an *important* cue has phi above that image's median (if ties or a single cue "
         f"leave nobody above the median, the image's max-phi cue(s) count as important).",
         f"`IoU` and `recall = |gpt ∩ vocab| / |gpt|` are against the single best-matching "
         f"vocabulary cue. Hit thresholds: IoU >= {IOU_HIT}, recall >= {REC_HIT}.", ""]

    allrows, imp = [], []
    L.append("## Per image")
    for r in reports:
        L += ["", f"### {r['label']} — `{r['image_id']}`  ({r['size'][0]}x{r['size'][1]})",
              "", f"GPT-4o cues: {r['n_gpt']} | vocabulary cues: {r['n_vocab']} "
                  f"| median phi: {_fmt(r['median_phi'])}", "",
              "| GPT-4o cue | category | phi | imp | area | best vocab match | IoU | recall |",
              "|---|---|---|---|---|---|---|---|"]
        for row in r["rows"]:
            allrows.append(row)
            if row["important"]:
                imp.append(row)
            L.append(f"| {row['cue']} | {row['category']} | {_fmt(row['phi'])} | "
                     f"{'*' if row['important'] else ''} | {row['area_frac']*100:.1f}% | "
                     f"{row['best_iou_cue']} | {row['iou']:.3f} | {row['recall']:.3f} |")
        L += ["", f"Vocabulary cues present: {', '.join(r['vocab_cues']) or '(none)'}"]
        if r["unmatched_vocab"]:
            L += ["", f"Vocabulary cues matching NO GPT-4o cue (IoU < {IOU_MATCHED}) — "
                      f"candidates GPT-4o missed:", "",
                  "| vocab cue | category | area | best IoU | frac already covered by any GPT-4o mask |",
                  "|---|---|---|---|---|"]
            for u in r["unmatched_vocab"]:
                L.append(f"| {u['cue']} | {u['category']} | {u['area_frac']*100:.1f}% | "
                         f"{u['best_iou']:.3f} | {u['covered_frac']*100:.0f}% |")
        else:
            L += ["", "All vocabulary cues matched some GPT-4o cue."]

    def frac(rows, key, th):
        return (sum(1 for x in rows if x[key] >= th) / len(rows), len(rows)) if rows else (float("nan"), 0)

    L += ["", "## Summary", ""]
    for tag, rows in (("all GPT-4o cues", allrows), ("important only (phi > per-image median)", imp)):
        f1, n = frac(rows, "iou", IOU_HIT)
        f2, _ = frac(rows, "recall", REC_HIT)
        mi = statistics.mean([x["iou"] for x in rows]) if rows else float("nan")
        mr = statistics.mean([x["recall"] for x in rows]) if rows else float("nan")
        L += [f"- **{tag}** (n={n}): IoU >= {IOU_HIT} for {_fmt(f1)} "
              f"({sum(1 for x in rows if x['iou'] >= IOU_HIT)}/{n}); "
              f"recall >= {REC_HIT} for {_fmt(f2)} "
              f"({sum(1 for x in rows if x['recall'] >= REC_HIT)}/{n}); "
              f"mean IoU {_fmt(mi)}, mean recall {_fmt(mr)}."]
    nun = sum(len(r["unmatched_vocab"]) for r in reports)
    nv = sum(r["n_vocab"] for r in reports)
    L += ["", f"- Vocabulary cues with no GPT-4o counterpart: {nun}/{nv} "
              f"({nun/nv:.2f} of all vocabulary cues) across {len(reports)} images.",
          f"- Vocabulary cues per image: {nv/len(reports):.1f} "
          f"(GPT-4o: {sum(r['n_gpt'] for r in reports)/len(reports):.1f}).", ""]

    txt = "\n".join(L)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
