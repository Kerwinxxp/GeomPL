"""Exact Shapley + SII attribution, with or without geometric de-duplication of the cue list.

`run(dedup=False)` is the raw-cue-list attribution (the shapley_v2 arm, exposed as
`python -m belief_elicit.shapley_v2`); `run(dedup=True)` first merges cues whose masks are
the same pixels — `belief_elicit/dedup_cues.py` unions cues at mask IoU >= 0.9 into
"merged players" — and then plays the exact game on the merged players.

No re-scoring is needed for the merged game (the key point): the member masks of a merged
player A = {a1..aj} are near-identical pixel-wise, so the image with "every merged player
in S' removed" is the image with "the union of members(A) for A in S' removed", differing
only in a handful of border pixels (IoU = 1.000 groups are identical; the IoU 0.903 / 0.998
groups differ by a few pixels). The gray-block 2^m lattice already exists, so

    v'(S') := v(union of members(A) for A in S')

is read straight out of it. **That is the only approximation here** and it must be stated
in the report.

The inpaint arm only has singles s<k> / pairs p<k>-<l> / all, so the "union of members"
subsets a merge needs are not always present:
  * per-player v'_inp({A}) is readable whenever union members(A) is there (1 member -> s,
    2 -> p, the whole image -> all);
  * the whole-image second-order anchored phi additionally needs every merged pair; if any
    is missing the image is counted as inpaint-incomputable.

Output: belief_elicit/shapley_v3_results.json (same schema as shapley_v2_results.json, with
extra members / member_indices / inpaint fields on cues[]), so downstream scripts can point
straight at it.

Run: python -m belief_elicit.shapley_v3        # de-duplicated (primary attribution)
     python -m belief_elicit.shapley_v2        # raw 244-cue list
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

from belief_elicit.attribution import empty_interaction, order2_shapley, shapley, sii
from belief_elicit.dedup_cues import OUT as GROUPS_DEFAULT
from belief_elicit.results import (CONTROLS as GRAY_CTRL, INPAINT, INPAINT_CONTROLS,
                                   LATTICE, SHAPLEY_V2, SHAPLEY_V3 as OUT, SWEEP,
                                   build_v, load_inpaint, load_lattice, load_sweep,
                                   merged_v)

HERE = os.path.dirname(os.path.abspath(__file__))
INPAINT_CTRL = INPAINT_CONTROLS


# ---------------- reading the inpaint arm ----------------

def inpaint_lookup(rec):
    """One inpaint record -> {frozenset(original cue indices): mpl}; singles / pairs / all only."""
    out = {frozenset(): 0.0}
    m = rec["n_cues"]
    for v in rec.get("variants", []):
        sp = v["spec"]
        if sp.startswith("s") and sp[1:].isdigit():
            out[frozenset([int(sp[1:])])] = v["mpl"]
        elif sp.startswith("p") and "-" in sp:
            a, b = sp[1:].split("-")
            out[frozenset([int(a), int(b)])] = v["mpl"]
        elif sp == "all":
            out[frozenset(range(m))] = v["mpl"]
    return out


def merged_inpaint(rec, groups):
    """The v' the merged second-order phi needs. Returns (None, missing subsets) if incomplete."""
    lut = inpaint_lookup(rec)
    M = len(groups)
    need, miss = {}, []
    for size in (1, 2):
        for S in itertools.combinations(range(M), size):
            orig = frozenset(k for gi in S for k in groups[gi]["members"])
            if orig in lut:
                need[frozenset(S)] = lut[orig]
            else:
                miss.append(sorted(orig))
    allset = frozenset(range(M))
    orig_all = frozenset(k for g in groups for k in g["members"])
    if orig_all in lut:
        need[allset] = lut[orig_all]
    else:
        miss.append(sorted(orig_all))
    return (None, miss) if miss else (need, [])


# ---------------- the shared game ----------------

def attribute(v, m, iid):
    """Exact phi + SII + empty-context interaction on one image's game."""
    phis = shapley(v, m)
    vN = v[frozenset(range(m))]
    if m >= 2:
        assert abs(sum(phis) - vN) < 1e-9, f"效率失败 {iid}"
    inter_sii = {f"{k},{l}": sii(v, m, k, l) for k, l in itertools.combinations(range(m), 2)}
    return phis, vN, inter_sii, empty_interaction(v, m)


def control_maxima(gc, ic):
    """Per original cue index: the gray control max, and the list of inpaint control mPLs."""
    gmax_by_k = {}
    if gc:
        for k, c in enumerate(gc["cues"]):
            cm = [x["mpl"] for x in c["controls"]]
            gmax_by_k[k] = max(cm) if cm else None
    imax_by_k = defaultdict(list)
    if ic:
        for vv in ic.get("variants", []):
            sp = vv["spec"]
            if sp.startswith("c") and "-" in sp and sp[1:].split("-")[0].isdigit():
                imax_by_k[int(sp[1:].split("-")[0])].append(vv["mpl"])
    return gmax_by_k, imax_by_k


# ---------------- the run ----------------

def run(dedup, out, groups_path=GROUPS_DEFAULT, sweep_path=SWEEP, lattice_path=LATTICE):
    """Attribute every maskable image and write `out`. dedup=True merges duplicate-mask cues."""
    sweep = load_sweep(sweep_path)
    lattice = load_lattice(lattice_path)
    G = json.load(open(groups_path, encoding="utf-8"))["images"] if dedup else {}
    inpaint = load_inpaint(INPAINT) if dedup else {}
    ictrl = load_inpaint(INPAINT_CTRL) if dedup else {}
    gctrl = load_inpaint(GRAY_CTRL) if dedup else {}

    rows, skipped, n_inpaint_ok, n_inpaint_bad, bad_ids = [], 0, 0, 0, []
    for r in sweep:
        m = r["n_cues"]
        if m < 1:
            continue
        iid = r["image_id"]
        groups = None
        if dedup:
            d = G.get(iid)
            if d is None:
                skipped += 1; continue
            assert d["n_cues"] == m, f"n_cues 不一致 {iid}"
            groups = d["groups"]
        v, ok = build_v(r, lattice)
        if not ok:
            skipped += 1; continue

        if dedup:
            vm = merged_v(v, groups)
            assert vm is not None, f"灰块格不完整,合并子集缺失 {iid}"
            M = len(groups)
        else:
            vm, M = v, m
        phis, vN, inter_sii, inter_empty = attribute(vm, M, iid)

        if not dedup:
            rows.append({"image_id": iid, "place": r["true_label"].split(",")[0],
                         "country_hit": r["country_hit"], "km_error": r["km_error"],
                         "n_cues": M, "vN": vN,
                         "cues": [{"cue": pc["cue"], "category": pc["category"],
                                   "v_single": pc["mpl"], "phi": phis[k]}
                                  for k, pc in enumerate(r["per_cue"])],
                         "sii": inter_sii, "empty_interaction": inter_empty})
            continue

        # ---- inpaint arm: merged second-order anchored phi, where computable ----
        inp_phi, inp_single = None, None
        inp_info = {"computable": False, "reason": "no inpaint data"}
        irec = inpaint.get(iid)
        if irec is not None:
            # the per-player v'_inp({A}) is readable regardless of whether the whole-image
            # second-order phi is
            lut = inpaint_lookup(irec)
            inp_single = [lut.get(frozenset(g["members"])) for g in groups]
            need, miss = merged_inpaint(irec, groups)
            if need is None:
                n_inpaint_bad += 1; bad_ids.append(iid)
                inp_info = {"computable": False, "reason": "merged subset not scored under inpaint",
                            "missing_subsets": miss}
            else:
                singles = [need[frozenset([k])] for k in range(M)]
                pairs = {(k, l): need[frozenset([k, l])]
                         for k, l in itertools.combinations(range(M), 2)}
                o2 = order2_shapley(singles, pairs, need[frozenset(range(M))])
                inp_phi = o2["phi"]
                inp_info = {"computable": True,
                            "residual_share": o2["residual_share"],
                            "vN": need[frozenset(range(M))]}
                n_inpaint_ok += 1

        # ---- resolvability: real = v'({A}); the null is the max over the members' controls
        gmax_by_k, imax_by_k = control_maxima(gctrl.get(iid), ictrl.get(iid))

        cues_out = []
        for gi, g in enumerate(groups):
            mem = g["members"]
            gm = [gmax_by_k.get(k) for k in mem]
            gm = [x for x in gm if x is not None]
            im = [x for k in mem for x in imax_by_k.get(k, [])]
            v_single = vm[frozenset([gi])]
            cues_out.append({
                "cue": g["name"], "category": g["category"],
                "v_single": v_single, "phi": phis[gi],
                "members": g["names"], "member_indices": mem,
                "n_members": len(mem),
                "categories": g["categories"],
                "ctrl_gray_max": (max(gm) if gm else None),
                "resolvable_gray": (v_single > max(gm)) if gm else None,
                "v_single_inpaint": (inp_single[gi] if inp_single else None),
                "phi_inpaint": (inp_phi[gi] if inp_phi else None),
                "ctrl_inpaint_max": (max(im) if im else None),
                "resolvable_inpaint": ((inp_single[gi] > max(im))
                                       if (inp_single and inp_single[gi] is not None and im)
                                       else None),
            })

        rows.append({"image_id": iid, "place": r["true_label"].split(",")[0],
                     "country_hit": r["country_hit"], "km_error": r["km_error"],
                     "n_cues": M, "n_cues_orig": m, "merged": d["affected"],
                     "vN": vN,
                     "cues": cues_out,
                     "sii": inter_sii, "empty_interaction": inter_empty,
                     "inpaint": inp_info})

    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if dedup:
        report_dedup(rows, out, skipped, n_inpaint_ok, n_inpaint_bad, bad_ids)
    else:
        report_plain(rows, out, skipped)
    return rows


# ---------------- reporting ----------------

def report_plain(rows, out, skipped):
    def cat_table(subset, tag):
        cphi, cv = defaultdict(list), defaultdict(list)
        for r in subset:
            for c in r["cues"]:
                cphi[c["category"] or "unknown"].append(c["phi"])
                cv[c["category"] or "unknown"].append(c["v_single"])
        print(f"\n[{tag}] 逐类别(φ=Shapley, v=single-cue;同一图/线索集配对):")
        print(f"{'类别':24s} {'φ中位':>8s} {'v中位':>8s} {'n':>4s}")
        rank = sorted(cphi, key=lambda x: -np.median(cphi[x]))
        for k in rank:
            print(f"{k:24s} {np.median(cphi[k]):8.4f} {np.median(cv[k]):8.4f} {len(cphi[k]):4d}")
        vr = sorted(cv, key=lambda x: -np.median(cv[x]))
        print(f"  φ 排名: {[k[:14] for k in rank]}")
        print(f"  v 排名: {[k[:14] for k in vr]}")

    hit = [r for r in rows if r["country_hit"]]
    miss = [r for r in rows if not r["country_hit"]]
    print(f"=== 归因:{len(rows)} 张 maskable 图(m>=1;跳过未补全 {skipped})===")
    print(f"效率(m>=2):全部通过 | hit {len(hit)} / miss {len(miss)}")
    cat_table(rows, "全体 95 张 [主结果]")
    cat_table(hit, "仅 country-hit [分层]")

    sii_all = np.array([x for r in rows for x in r["sii"].values()])
    emp_all = np.array([x for r in rows for x in r["empty_interaction"].values()])
    print(f"\n交互对照({len(sii_all)} 对):")
    for name, arr in [("SII(正式)", sii_all), ("empty-context(旧)", emp_all)]:
        print(f"  {name:18s} 重叠<-.01 {int((arr<-0.01).sum()):3d} | ~0 "
              f"{int((np.abs(arr)<=0.01).sum()):3d} | 备份>.01 {int((arr>0.01).sum()):3d} "
              f"| 中位 {np.median(arr):+.3f}")
    print("saved", out)


def report_dedup(rows, out, skipped, n_inpaint_ok, n_inpaint_bad, bad_ids):
    n_before = sum(x["n_cues_orig"] for x in rows)
    n_after = sum(x["n_cues"] for x in rows)
    aff = [x for x in rows if x["merged"]]
    print(f"=== 去重后归因:{len(rows)} 张图(跳过 {skipped})===")
    print(f"线索 {n_before} -> {n_after}(合并掉 {n_before-n_after});受影响的图 {len(aff)}")
    print(f"效率(m>=2):全部通过")
    print(f"修复口径二阶 φ:可算 {n_inpaint_ok} 张 | 不可算(合并子集未打分) {n_inpaint_bad} 张")
    if bad_ids:
        by_place = [x["place"] for x in rows if x["image_id"] in set(bad_ids)]
        print(f"  不可算的图: {by_place}")

    cphi, cv = defaultdict(list), defaultdict(list)
    for x in rows:
        for c in x["cues"]:
            cphi[c["category"] or "unknown"].append(c["phi"])
            cv[c["category"] or "unknown"].append(c["v_single"])
    print("\n逐类别(去重后;φ=Shapley, v=single-cue):")
    print(f"{'类别':24s} {'φ中位':>8s} {'v中位':>8s} {'n':>4s}")
    rank = sorted(cphi, key=lambda x: -np.median(cphi[x]))
    for k in rank:
        print(f"{k:24s} {np.median(cphi[k]):8.4f} {np.median(cv[k]):8.4f} {len(cphi[k]):4d}")
    print(f"  φ 排名: {[k[:14] for k in rank]}")

    sii_all = np.array([x for r_ in rows for x in r_["sii"].values()])
    if len(sii_all):
        print(f"\nSII({len(sii_all)} 对): 重叠<-.01 {int((sii_all<-0.01).sum())} | "
              f"~0 {int((np.abs(sii_all)<=0.01).sum())} | 备份>.01 {int((sii_all>0.01).sum())} "
              f"| 中位 {np.median(sii_all):+.3f}")
    rg = [c["resolvable_gray"] for x in rows for c in x["cues"] if c["resolvable_gray"] is not None]
    ri = [c["resolvable_inpaint"] for x in rows for c in x["cues"]
          if c["resolvable_inpaint"] is not None]
    if rg:
        print(f"可分辨(灰块,real>成员对照最大): {np.mean(rg)*100:.1f}% (n={len(rg)})")
    if ri:
        print(f"可分辨(修复,real>成员对照最大): {np.mean(ri)*100:.1f}% (n={len(ri)})")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description="Shapley/SII attribution after geometric de-dup")
    ap.add_argument("--groups", default=GROUPS_DEFAULT)
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--lattice", default=LATTICE)
    ap.add_argument("--out", default=None,
                    help="default: shapley_v3_results.json, or shapley_v2_results.json "
                         "with --no-dedup")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip the merge and attribute the raw cue list (= shapley_v2)")
    a = ap.parse_args()
    dedup = not a.no_dedup
    run(dedup, a.out or (OUT if dedup else SHAPLEY_V2), a.groups, a.sweep, a.lattice)


if __name__ == "__main__":
    main()
