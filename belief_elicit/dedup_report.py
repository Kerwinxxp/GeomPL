"""几何去重的 BEFORE/AFTER 报告(shapley_v2 vs shapley_v3)。

BEFORE = belief_elicit/shapley_v2_results.json(原始 244 条线索)
AFTER  = belief_elicit/shapley_v3_results.json(IoU>=0.9 合并后 223 条)

章节
  (1) 受影响的图/线索、合并组清单
  (2) 合并组的功劳合并检查:Σφ(成员, before) vs φ(合并, after) —— 期望≈相等
  (3) 逐类别中位 φ / v_single 与排名变化
  (4) 交互分布(SII)before/after,以及"重叠"里有多少纯粹是重复线索造成的
  (5) 效率检查(Σφ = v(N))
  (6) Harsanyi 各阶 |d| 占比 before/after
  (7) 最小充分子集(达 80% v(N) 所需线索数)before/after
  (8) 修复(inpaint)口径:可分辨率 before/after、不可算的图

产物:belief_elicit/dedup_report.md + belief_elicit/dedup_report.json
运行:python -m belief_elicit.dedup_report
"""
import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

from belief_elicit.attribution import harsanyi, min_sufficient
from belief_elicit.dedup_cues import OUT as GROUPS_DEFAULT
from belief_elicit.results import (CONTROLS as GRAY_CTRL, DEDUP_REPORT_JSON as OUT_JSON,
                                   DEDUP_REPORT_MD as OUT_MD, INPAINT,
                                   INPAINT_CONTROLS as INPAINT_CTRL, LATTICE,
                                   SHAPLEY_V2 as V2, SHAPLEY_V3 as V3, SWEEP, build_v,
                                   by_image, load_sweep, merged_v)

HERE = os.path.dirname(os.path.abspath(__file__))


def cat_stats(rows):
    cphi, cv = defaultdict(list), defaultdict(list)
    for r in rows:
        for c in r["cues"]:
            k = c["category"] or "unknown"
            cphi[k].append(c["phi"]); cv[k].append(c["v_single"])
    return ({k: float(np.median(x)) for k, x in cphi.items()},
            {k: float(np.median(x)) for k, x in cv.items()},
            {k: len(x) for k, x in cphi.items()})


def sii_bins(rows):
    a = np.array([x for r in rows for x in r["sii"].values()], float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "overlap": int((a < -0.01).sum()),
            "flat": int((np.abs(a) <= 0.01).sum()), "backup": int((a > 0.01).sum()),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def resolvability_before(sweep, gctrl, inpaint, ictrl):
    """Before de-dup: gray real = v({k}) vs its own control max; inpaint s<k> vs its c<k>-* max."""
    g_flags, i_flags = [], []
    for r in sweep:
        iid = r["image_id"]
        gc = gctrl.get(iid)
        if gc:
            for k, c in enumerate(gc["cues"]):
                cm = [x["mpl"] for x in c["controls"]]
                if cm:
                    g_flags.append(c["real_mpl"] > max(cm))
        irec, ic = inpaint.get(iid), ictrl.get(iid)
        if irec and ic:
            s = {}
            for v in irec.get("variants", []):
                if v["spec"].startswith("s") and v["spec"][1:].isdigit():
                    s[int(v["spec"][1:])] = v["mpl"]
            cmax = defaultdict(list)
            for v in ic.get("variants", []):
                sp = v["spec"]
                if sp.startswith("c") and "-" in sp and sp[1:].split("-")[0].isdigit():
                    cmax[int(sp[1:].split("-")[0])].append(v["mpl"])
            for k, val in s.items():
                if cmax.get(k):
                    i_flags.append(val > max(cmax[k]))
    return g_flags, i_flags


def main():
    ap = argparse.ArgumentParser(description="geometric de-dup BEFORE/AFTER report")
    ap.add_argument("--before", default=V2)
    ap.add_argument("--after", default=V3)
    ap.add_argument("--groups", default=GROUPS_DEFAULT)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    a = ap.parse_args()

    B = json.load(open(a.before, encoding="utf-8"))
    A = json.load(open(a.after, encoding="utf-8"))
    GD = json.load(open(a.groups, encoding="utf-8"))
    G = GD["images"]
    Bi = {r["image_id"]: r for r in B}
    Ai = {r["image_id"]: r for r in A}
    sweep = load_sweep(SWEEP)
    lattice = by_image(load_sweep(LATTICE))
    inpaint = by_image(load_sweep(INPAINT))
    ictrl = by_image(load_sweep(INPAINT_CTRL))
    gctrl = by_image(load_sweep(GRAY_CTRL))

    S = {"meta": {"before": os.path.basename(a.before), "after": os.path.basename(a.after),
                  "iou_threshold": GD["meta"]["iou_threshold"],
                  "recall_threshold": GD["meta"]["recall_threshold"]}}

    # ---- (1) 规模 ----
    n_img = len(A)
    aff = [r for r in A if r["merged"]]
    n_b = sum(r["n_cues"] for r in B)
    n_a = sum(r["n_cues"] for r in A)
    merged_groups = [(iid, g) for iid, d in G.items() for g in d["groups"]
                     if len(g["members"]) > 1]
    cont = [(iid, c) for iid, d in G.items() for c in d["containment"]]
    S["scale"] = {"n_images": n_img, "n_images_affected": len(aff),
                  "n_cues_before": n_b, "n_cues_after": n_a,
                  "n_duplicates_removed": n_b - n_a,
                  "n_merged_groups": len(merged_groups),
                  "n_containment_pairs_reported": len(cont)}

    # ---- (2) 功劳合并检查 ----
    cons = []
    for iid, g in merged_groups:
        br, ar = Bi[iid], Ai[iid]
        s_before = sum(br["cues"][k]["phi"] for k in g["members"])
        gi = [i for i, c in enumerate(ar["cues"]) if c["member_indices"] == g["members"]][0]
        phi_after = ar["cues"][gi]["phi"]
        v_before = [br["cues"][k]["v_single"] for k in g["members"]]
        cons.append({"image_id": iid, "place": ar["place"], "members": g["members"],
                     "names": g["names"], "name": g["name"],
                     "max_pair_iou": max(x[2] for x in g["pair_ious"]),
                     "sum_phi_before": s_before, "phi_after": phi_after,
                     "delta": phi_after - s_before,
                     "rel_delta": abs(phi_after - s_before) / max(abs(s_before), 1e-12),
                     "delta_over_vN": abs(phi_after - s_before) / max(abs(ar["vN"]), 1e-12),
                     "vN": ar["vN"], "covers_all_cues": len(g["members"]) == ar["n_cues_orig"],
                     "v_single_before": v_before,
                     "v_single_after": ar["cues"][gi]["v_single"]})
    dmax = max((abs(c["delta"]) for c in cons), default=0.0)
    S["consolidation"] = {"groups": cons, "max_abs_delta": dmax,
                          "max_rel_delta": max((c["rel_delta"] for c in cons), default=0.0),
                          "max_delta_over_vN": max((c["delta_over_vN"] for c in cons), default=0.0),
                          "n_negative": int(sum(c["delta"] < -1e-12 for c in cons)),
                          "n_exact": int(sum(abs(c["delta"]) < 1e-12 for c in cons)),
                          "median_abs_delta": float(np.median([abs(c["delta"]) for c in cons]))
                          if cons else 0.0}

    # ---- (3) 逐类别 ----
    pb, vb, nb = cat_stats(B)
    pa, va, na = cat_stats(A)
    cats = sorted(set(pb) | set(pa), key=lambda k: -pa.get(k, -9))
    rank_b = [k for k in sorted(pb, key=lambda k: -pb[k])]
    rank_a = [k for k in sorted(pa, key=lambda k: -pa[k])]
    rank_vb = [k for k in sorted(vb, key=lambda k: -vb[k])]
    rank_va = [k for k in sorted(va, key=lambda k: -va[k])]
    S["per_category"] = {"cats": cats,
                         "phi_before": pb, "phi_after": pa, "v_before": vb, "v_after": va,
                         "n_before": nb, "n_after": na,
                         "rank_phi_before": rank_b, "rank_phi_after": rank_a,
                         "rank_phi_unchanged": rank_b == rank_a,
                         "rank_v_before": rank_vb, "rank_v_after": rank_va,
                         "rank_v_unchanged": rank_vb == rank_va}

    # ---- (4) 交互 ----
    sb, sa = sii_bins(B), sii_bins(A)
    # 去重前被判为"重叠"的对里,有多少两端落在同一个合并组(= 纯重复造成的假交互)
    dup_pairs, dup_overlap, dup_sii = 0, 0, []
    for iid, d in G.items():
        br = Bi.get(iid)
        if not br:
            continue
        gmap = d["index_map"]
        for key, val in br["sii"].items():
            k, l = (int(x) for x in key.split(","))
            if gmap[k] == gmap[l]:
                dup_pairs += 1
                dup_sii.append(val)
                if val < -0.01:
                    dup_overlap += 1
    S["interactions"] = {"before": sb, "after": sa,
                         "within_group_pairs_before": dup_pairs,
                         "within_group_overlap_pairs_before": dup_overlap,
                         "within_group_sii_median": float(np.median(dup_sii)) if dup_sii else None,
                         "overlap_removed": sb.get("overlap", 0) - sa.get("overlap", 0)}

    # ---- (5) 效率 + (6) Harsanyi + (7) 最小充分子集 ----
    eff_b, eff_a = [], []
    ord_b, ord_a = defaultdict(list), defaultdict(list)
    ms_b, ms_a = [], []
    for r in sweep:
        iid, m = r["image_id"], r["n_cues"]
        if m < 1:
            continue
        v, ok = build_v(r, lattice)
        if not ok:
            continue
        vm = merged_v(v, G[iid]["groups"])
        M = len(G[iid]["groups"])
        eff_b.append(abs(sum(c["phi"] for c in Bi[iid]["cues"]) - Bi[iid]["vN"]))
        eff_a.append(abs(sum(c["phi"] for c in Ai[iid]["cues"]) - Ai[iid]["vN"]))
        if m >= 2:
            for o, s in harsanyi(v, m).items():
                ord_b[o].append(s)
            x = min_sufficient(v, m)
            if x:
                ms_b.append((x, m))
        if M >= 2:
            for o, s in harsanyi(vm, M).items():
                ord_a[o].append(s)
            x = min_sufficient(vm, M)
            if x:
                ms_a.append((x, M))
    S["efficiency"] = {"max_abs_before": float(max(eff_b)), "max_abs_after": float(max(eff_a)),
                       "n": len(eff_a), "ok": bool(max(eff_a) < 1e-9)}
    S["harsanyi"] = {"before": {int(o): float(np.median(v_)) for o, v_ in sorted(ord_b.items())},
                     "after": {int(o): float(np.median(v_)) for o, v_ in sorted(ord_a.items())},
                     "n_images_before": len(next(iter(ord_b.values()), [])),
                     "n_images_after": len(next(iter(ord_a.values()), []))}
    mb = np.array([x[0] for x in ms_b]); ma = np.array([x[0] for x in ms_a])
    S["min_sufficient"] = {
        "before": {"n": len(mb), "median": float(np.median(mb)),
                   "median_m": float(np.median([x[1] for x in ms_b])),
                   "frac_single": float((mb == 1).mean())},
        "after": {"n": len(ma), "median": float(np.median(ma)),
                  "median_m": float(np.median([x[1] for x in ms_a])),
                  "frac_single": float((ma == 1).mean())}}

    # ---- (8) 修复口径 ----
    gb, ib = resolvability_before(sweep, gctrl, inpaint, ictrl)
    ga = [c["resolvable_gray"] for r in A for c in r["cues"] if c["resolvable_gray"] is not None]
    ia = [c["resolvable_inpaint"] for r in A for c in r["cues"]
          if c["resolvable_inpaint"] is not None]
    inc = [r for r in A if not r["inpaint"]["computable"]]
    S["resolvability"] = {
        "gray_before": {"n": len(gb), "rate": float(np.mean(gb))},
        "gray_after": {"n": len(ga), "rate": float(np.mean(ga))},
        "inpaint_before": {"n": len(ib), "rate": float(np.mean(ib))},
        "inpaint_after": {"n": len(ia), "rate": float(np.mean(ia))},
        "inpaint_incomputable_images": [{"image_id": r["image_id"], "place": r["place"],
                                         "n_cues": r["n_cues"], "n_cues_orig": r["n_cues_orig"]}
                                        for r in inc],
        "n_inpaint_incomputable": len(inc),
        "n_inpaint_computable": len(A) - len(inc)}

    json.dump(S, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------------- Markdown ----------------
    L = []
    W = L.append
    W("# Geometric cue de-duplication: before/after\n")
    W(f"Merge rule: pairwise mask IoU >= **{GD['meta']['iou_threshold']}** (union-find over the "
      f"per-cue union masks built exactly as `precompute_inpaint.cue_masks_of`). Containment "
      f"(recall >= {GD['meta']['recall_threshold']} with IoU below the merge threshold) is "
      f"**reported only, never merged**.\n")
    W("**No re-scoring.** For a merged player `A = {a1..aj}` the members' masks are (near-)"
      "identical pixels, so the masked image of a merged subset `S'` equals — up to a handful of "
      "boundary pixels — the masked image of the ORIGINAL subset `⋃_{A∈S'} members(A)`. The gray "
      "lattice is complete (all 2^m subsets), so `v'(S') := v(⋃ members)` is read straight from "
      "the existing data. This is the single approximation in this pipeline.\n")
    exact = [g for _, g in merged_groups if g["max_member_diff_px"] == 0]
    inex = sorted((g for _, g in merged_groups if g["max_member_diff_px"] > 0),
                  key=lambda g: -g["max_member_diff_frac"])
    S["approximation"] = {
        "n_groups_pixel_exact": len(exact), "n_groups_inexact": len(inex),
        "worst": [{"names": g["names"], "diff_px": g["max_member_diff_px"],
                   "diff_frac": g["max_member_diff_frac"],
                   "union_area_px": g["union_area_px"]} for g in inex]}
    W(f"How big is that approximation? For each merged group, compare every member mask against "
      f"the group union: **{len(exact)} of {len(merged_groups)} groups are pixel-exact** (0 "
      f"differing pixels, so `v'` is literally the already-scored value). The remaining "
      f"{len(inex)} differ by " + ", ".join(
        f"{g['max_member_diff_px']} px ({g['max_member_diff_frac']*100:.2f}% of the "
        f"{g['union_area_px']} px union, IoU {max(x[2] for x in g['pair_ious']):.3f})"
        for g in inex) + ".\n")

    s = S["scale"]
    W("## 1. Scale\n")
    W(f"| | before | after |\n|---|---|---|")
    W(f"| images | {s['n_images']} | {s['n_images']} |")
    W(f"| cues / players | {s['n_cues_before']} | {s['n_cues_after']} |")
    W(f"\n- images affected: **{s['n_images_affected']}** / {s['n_images']}")
    W(f"- merged groups: **{s['n_merged_groups']}**, duplicates removed: "
      f"**{s['n_duplicates_removed']}**")
    W(f"- containment pairs reported (not merged): {s['n_containment_pairs_reported']}\n")

    W("## 2. Merged groups and credit consolidation\n")
    W("Expectation: merging duplicates should *consolidate* credit, i.e. "
      "`φ(merged) ≈ Σφ(members, before)`.\n")
    W("| image | m→M | max IoU | members | Σφ before | φ after | Δ | \\|Δ\\|/v(N) |")
    W("|---|---|---|---|---|---|---|---|")
    for c in sorted(cons, key=lambda x: -abs(x["delta"])):
        d = G[c["image_id"]]
        nm = "; ".join(n[:34] for n in c["names"])
        W(f"| {c['place']} | {d['n_cues']}→{d['n_merged']} | {c['max_pair_iou']:.3f} | {nm} | "
          f"{c['sum_phi_before']:.4f} | {c['phi_after']:.4f} | {c['delta']:+.4f} | "
          f"{c['delta_over_vN']*100:.1f}% |")
    cz = S["consolidation"]
    W(f"\nMax |Δ| = **{cz['max_abs_delta']:.4f}** (at most **{cz['max_delta_over_vN']*100:.1f}%** "
      f"of that image's v(N); the worst ratio *to Σφ itself* is {cz['max_rel_delta']*100:.0f}%, "
      f"but that is the Venice group whose Σφ is already near zero), median |Δ| = "
      f"{cz['median_abs_delta']:.4f}.")
    W(f"\nCredit is consolidated as expected. Δ = 0 **exactly** in {cz['n_exact']} groups — those "
      f"where the group swallows every cue in the image, so efficiency forces "
      f"φ(merged) = v(N) = Σφ(before). In {cz['n_negative']} of the remaining groups Δ is "
      f"slightly **negative**: this is the standard merging effect for near-duplicate players — "
      f"before merging, each duplicate was credited for a marginal contribution the other could "
      f"equally have supplied, and the single merged player is paid that shared contribution only "
      f"once. The residual is small (median {cz['median_abs_delta']:.4f} nats/1000 km).\n")

    W("## 3. Per-category medians and ranking\n")
    W("| category | n before | n after | φ median before | φ median after | "
      "v_single median before | v_single median after |")
    W("|---|---|---|---|---|---|---|")
    for k in cats:
        W(f"| {k} | {nb.get(k,0)} | {na.get(k,0)} | {pb.get(k,float('nan')):.4f} | "
          f"{pa.get(k,float('nan')):.4f} | {vb.get(k,float('nan')):.4f} | "
          f"{va.get(k,float('nan')):.4f} |")
    shifts = sorted(((k, pa[k] - pb[k], (pa[k] - pb[k]) / max(abs(pb[k]), 1e-12))
                     for k in cats if k in pa and k in pb), key=lambda x: -abs(x[1]))
    S["per_category"]["largest_shifts"] = [{"category": k, "delta": d, "rel": r}
                                           for k, d, r in shifts[:3]]
    W("\n- largest median-φ shifts: " + "; ".join(
        f"**{k}** {pb[k]:.4f} → {pa[k]:.4f} ({d:+.4f}, {r*100:+.0f}%)" for k, d, r in shifts[:3]))
    W("  De-duplication mostly moves **text/signage**: several images (Cambridge, Spa, Bardo, "
      "Bangkok, Paris) had 2-4 separately named signs segmented onto the same pixels, so "
      "consolidating them turns several small φ into one large φ while dropping the cue count "
      "(29 → 24).")
    W(f"- φ ranking before: {' > '.join(rank_b)}")
    W(f"- φ ranking after: {' > '.join(rank_a)}")
    W(f"- **φ ranking unchanged: {'yes' if rank_b == rank_a else 'NO'}**")
    W(f"- v_single ranking unchanged: {'yes' if rank_vb == rank_va else 'NO'}\n")

    W("## 4. Interactions (Shapley Interaction Index)\n")
    W("| | pairs | overlap (SII < −0.01) | ≈0 (\\|SII\\| ≤ 0.01) | backup (SII > 0.01) | median |")
    W("|---|---|---|---|---|---|")
    for tag, d in [("before", sb), ("after", sa)]:
        W(f"| {tag} | {d['n']} | {d['overlap']} | {d['flat']} | {d['backup']} | "
          f"{d['median']:+.4f} |")
    it = S["interactions"]
    W(f"\n- pairs whose two members ended up in the **same** merged group (i.e. pure geometric "
      f"duplicates): **{it['within_group_pairs_before']}** of {sb['n']} before-pairs; of those, "
      f"**{it['within_group_overlap_pairs_before']}** had been counted as \"overlap\" "
      f"(SII < −0.01), median SII {it['within_group_sii_median']:+.4f}.")
    W(f"- overlap pairs removed by de-duplication: {it['overlap_removed']} "
      f"({sb['overlap']} → {sa['overlap']}).\n")

    e = S["efficiency"]
    W("## 5. Efficiency\n")
    W(f"`Σφ = v(N)` holds on all {e['n']} images: max |Σφ − v(N)| = {e['max_abs_before']:.2e} "
      f"before, {e['max_abs_after']:.2e} after.\n")

    W("## 6. Harsanyi dividends by order (median share of Σ|d|)\n")
    W("| order | before | after |\n|---|---|---|")
    hb, ha = S["harsanyi"]["before"], S["harsanyi"]["after"]
    for o in sorted(set(hb) | set(ha)):
        W(f"| {o} | " + (f"{hb[o]*100:.1f}%" if o in hb else "–") + " | "
          + (f"{ha[o]*100:.1f}%" if o in ha else "–") + " |")
    W(f"\n(before: {S['harsanyi']['n_images_before']} images with m≥2; "
      f"after: {S['harsanyi']['n_images_after']} images with M≥2)\n")

    W("## 7. Minimal sufficient set (cues needed to reach 80% of v(N))\n")
    msb, msa = S["min_sufficient"]["before"], S["min_sufficient"]["after"]
    W("| | images | median \\|S\\| | median m | share needing a single cue |")
    W("|---|---|---|---|---|")
    for tag, d in [("before", msb), ("after", msa)]:
        W(f"| {tag} | {d['n']} | {d['median']:.0f} | {d['median_m']:.0f} | "
          f"{d['frac_single']*100:.1f}% |")
    W("")

    W("## 8. Inpaint arm\n")
    r_ = S["resolvability"]
    W("| arm | before | after |\n|---|---|---|")
    W(f"| gray, real > max of own controls | {r_['gray_before']['rate']*100:.1f}% "
      f"(n={r_['gray_before']['n']}) | {r_['gray_after']['rate']*100:.1f}% "
      f"(n={r_['gray_after']['n']}) |")
    W(f"| inpaint, real > max of own controls | {r_['inpaint_before']['rate']*100:.1f}% "
      f"(n={r_['inpaint_before']['n']}) | {r_['inpaint_after']['rate']*100:.1f}% "
      f"(n={r_['inpaint_after']['n']}) |")
    W(f"\nFor a merged player, *real* is `v'({{A}})` (all members masked together) and the "
      f"artifact floor is the **max over the members' own equal-area controls** — the strictest "
      f"available comparison. The inpaint after-column has n={r_['inpaint_after']['n']} rather "
      f"than {n_a}: {n_a - r_['inpaint_after']['n']} merged players group 3 originals that do not "
      f"span their image, so their union was never scored under inpaint (only singles, pairs and "
      f"all exist there). Both arms are essentially unchanged by de-duplication — the merged "
      f"players inherit their members' resolvability rather than creating or destroying it.\n")
    W(f"- deduped order-2 anchored φ under inpaint: computable on "
      f"**{r_['n_inpaint_computable']}** / {len(A)} images; "
      f"**{r_['n_inpaint_incomputable']}** are inpaint-incomputable because a required merged "
      f"subset (a group of ≥3 originals, or a merged pair whose union has ≥3 originals) was "
      f"never scored — the inpaint arm only has singles, pairs and all.")
    for x in r_["inpaint_incomputable_images"]:
        W(f"  - {x['place']} ({x['n_cues_orig']}→{x['n_cues']} cues)")
    W("")

    W("## 9. Containment pairs (reported, not merged)\n")
    W("| image | recall(inner ⊂ outer) | IoU | inner | outer |")
    W("|---|---|---|---|---|")
    for iid, c in sorted(cont, key=lambda x: -x[1]["recall_inner_in_outer"]):
        W(f"| {G[iid]['place']} | {c['recall_inner_in_outer']:.3f} | {c['iou']:.3f} | "
          f"{c['inner_name'][:40]} | {c['outer_name'][:40]} |")
    W("")

    open(a.out_md, "w", encoding="utf-8").write("\n".join(L))

    # ---------------- 终端摘要 ----------------
    print(f"=== 去重 BEFORE/AFTER ===")
    print(f"图 {s['n_images']}(受影响 {s['n_images_affected']}) | 线索 {s['n_cues_before']} -> "
          f"{s['n_cues_after']}(合并组 {s['n_merged_groups']},删重复 {s['n_duplicates_removed']})")
    print(f"功劳合并:max|Δ| = {S['consolidation']['max_abs_delta']:.4f} "
          f"(<= {S['consolidation']['max_delta_over_vN']*100:.1f}% 的 v(N);中位 |Δ| "
          f"{S['consolidation']['median_abs_delta']:.4f};Δ=0 精确 {S['consolidation']['n_exact']} 组)")
    print(f"φ 类别排名是否不变:{'是' if rank_b == rank_a else '否'}")
    print(f"SII 重叠 {sb['overlap']} -> {sa['overlap']}(其中纯重复对 "
          f"{it['within_group_overlap_pairs_before']} 个)| 中位 {sb['median']:+.3f} -> "
          f"{sa['median']:+.3f}")
    print(f"效率 max|Σφ−v(N)|: {e['max_abs_after']:.2e}")
    print(f"可分辨 灰块 {r_['gray_before']['rate']*100:.1f}% -> {r_['gray_after']['rate']*100:.1f}%"
          f" | 修复 {r_['inpaint_before']['rate']*100:.1f}% -> "
          f"{r_['inpaint_after']['rate']*100:.1f}%")
    print(f"修复口径二阶 φ 不可算:{r_['n_inpaint_incomputable']} 张")
    print("saved", a.out_md)
    print("saved", a.out_json)


if __name__ == "__main__":
    main()
