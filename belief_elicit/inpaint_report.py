"""修复(inpaint)口径逐线索泄漏分析:与灰块口径全面对照。

输入
  --results   run_georanker_inpaint.py 的主结果(s*/p*/all),默认
              belief_elicit/georanker_inpaint_results.json
  --controls  对照结果(c* 修复对照 / cg* 同位置灰块对照),默认
              belief_elicit/georanker_inpaint_control_results.json(不存在则跳过 (d) 节)
  --cues-from sweep(默认)= 线索名/类别取自 sweep 的 per_cue;
              manifest = 取自 <cache>/<image_id>/manifest.json 的 cues[](词表口径专用,
              此时没有灰块基线,(a)(b)(c) 中所有 gray 对照自动跳过)

章节
  (a) 单条移除:灰块 v({k}) vs 修复 v({k})(散点数据 / Spearman / inpaint<gray 比例 / 逐类别中位)
  (b) 修复口径二阶锚定 Shapley φ_inpaint,与灰块精确 φ 对照(全局 & 图内 Spearman、top-1、逐类别中位)
  (c) 修复口径非可加性:Σ_k v({k}) vs v(N)、次可加比例、成对交互 d_kl 符号分布 vs 灰块 SII
  (d) 伪影地板:修复对照 c* 与灰块对照的比较、可分辨率(real > 自身对照最大)。
      优先用同一放置的 cg* 做配对比较;若结果里没有 cg*(当前情况),回退到更早的独立
      灰块对照 --gray-controls(belief_elicit/georanker_control_results.json,放置位置不同)
      做**非配对**分布对照,并在报告里明确标注。
  (e) 落盘 belief_elicit/<prefix>inpaint_report.md + belief_elicit/<prefix>inpaint_summary.json
      (--out-prefix,默认空;--out-md / --out-json 仍可单独覆盖)

**部分数据安全**:每节只纳入所需 spec 齐全的图,并打印可用图数。

运行:python -m belief_elicit.inpaint_report
     python -m belief_elicit.inpaint_report --results ... --cues-from manifest \
            --cache belief_elicit/inpaint_cache_vocab --out-prefix vocab_
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

from belief_elicit.attribution import order2_shapley, spearman
from belief_elicit.plotstyle import fmt
from belief_elicit.results import (CONTROLS as GRAY_CONTROLS, INPAINT as RESULTS,
                                   INPAINT_CACHE as CACHE,
                                   INPAINT_CONTROLS as CONTROLS, SHAPLEY_V2 as SHAP, SWEEP,
                                   index_variants, load_json, load_manifest)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "inpaint_report.md")
OUT_JSON = os.path.join(HERE, "inpaint_summary.json")


def default_outputs(prefix=""):
    """--out-prefix -> (report md, summary json); prefix="" reproduces the historical defaults."""
    return (os.path.join(HERE, f"{prefix}inpaint_report.md"),
            os.path.join(HERE, f"{prefix}inpaint_summary.json"))


# ---------------- loading ----------------

def manifest_cues(cache, iid):
    """<cache>/<iid>/manifest.json -> [{cue, category, area_frac}, ...]; None on failure."""
    d = load_manifest(os.path.join(cache, iid))
    if not isinstance(d, dict):
        return None
    cs = d.get("cues")
    if not isinstance(cs, list) or not cs:
        return None
    out = []
    for i, c in enumerate(cs):
        if not isinstance(c, dict):
            return None
        out.append({"cue": c.get("cue") or f"cue{i}",
                    "category": c.get("category") or "unknown",
                    "area_frac": c.get("area_frac")})
    return out


def cue_meta(rec, mode, cache, sweep_by_id):
    """Per-image cue metadata [{cue, category, area_frac}]; None when unavailable."""
    iid = rec["image_id"]
    if mode == "manifest":
        return manifest_cues(cache, iid)
    sr = sweep_by_id.get(iid)
    if sr:
        return [{"cue": pc["cue"], "category": pc.get("category") or "unknown",
                 "area_frac": pc.get("cov")} for pc in sr["per_cue"]]
    names = rec.get("cue_names") or []
    cats = rec.get("cue_categories") or []
    if not names:
        return None
    return [{"cue": n, "category": (cats[i] if i < len(cats) else None) or "unknown",
             "area_frac": None} for i, n in enumerate(names)]


def med(a):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return float(np.median(a)) if len(a) else float("nan")


# ---------------- (a) single-cue removal ----------------

def section_a(records, meta_by_id, use_gray):
    rows, n_img = [], 0
    for r in records:
        iid = r["image_id"]
        meta = meta_by_id.get(iid)
        if not meta:
            continue
        V = index_variants(r)
        singles = {}
        for k in range(len(meta)):
            v = V.get(f"s{k}")
            if v is not None:
                singles[k] = v
        if not singles:
            continue
        n_img += 1
        for k, v in singles.items():
            g = v.get("gray_mpl") if use_gray else None
            rows.append({"image_id": iid, "k": k, "cue": meta[k]["cue"],
                         "category": meta[k]["category"],
                         "area_frac": meta[k].get("area_frac"),
                         "coverage": v.get("coverage"),
                         "inpaint": v["mpl"],
                         "gray": g})
    out = {"n_images": n_img, "n_cues": len(rows), "scatter": rows}
    inp = np.array([x["inpaint"] for x in rows], float) if rows else np.array([])
    out["inpaint_median"] = med(inp)
    paired = [x for x in rows if x["gray"] is not None]
    out["n_paired"] = len(paired)
    if paired:
        g = np.array([x["gray"] for x in paired], float)
        i = np.array([x["inpaint"] for x in paired], float)
        out["spearman_gray_inpaint"] = spearman(g, i)
        out["pearson_gray_inpaint"] = (float(np.corrcoef(g, i)[0, 1])
                                       if len(g) > 1 and g.std() > 0 and i.std() > 0
                                       else float("nan"))
        out["frac_inpaint_lt_gray"] = float(np.mean(i < g))
        out["gray_median"] = float(np.median(g))
        out["inpaint_median_paired"] = float(np.median(i))
        out["ratio_median"] = float(np.median(i / np.maximum(g, 1e-9)))
        out["delta_median"] = float(np.median(i - g))
    bycat = defaultdict(lambda: {"gray": [], "inpaint": []})
    for x in rows:
        bycat[x["category"]]["inpaint"].append(x["inpaint"])
        if x["gray"] is not None:
            bycat[x["category"]]["gray"].append(x["gray"])
    out["per_category"] = {c: {"n": len(d["inpaint"]),
                               "inpaint_median": med(d["inpaint"]),
                               "gray_median": med(d["gray"]) if d["gray"] else None}
                           for c, d in bycat.items()}
    return out


# ---------------- (b) 二阶锚定 Shapley ----------------

def image_order2(rec, m):
    """该图的 s*/p*/all 是否齐全;齐全则返回 order2_shapley 结果,否则 None。"""
    V = index_variants(rec)
    try:
        singles = [V[f"s{k}"]["mpl"] for k in range(m)]
    except KeyError:
        return None
    if m == 1:
        return {"phi": [singles[0]], "phi2": [singles[0]], "d": singles,
                "interactions": {}, "anchor": 0.0, "residual": 0.0,
                "residual_share": 0.0, "sum_phi2": singles[0]}
    pairs = {}
    for k, l in itertools.combinations(range(m), 2):
        v = V.get(f"p{k}-{l}") or V.get(f"p{l}-{k}")
        if v is None:
            return None
        pairs[(k, l)] = v["mpl"]
    va = V.get("all")
    if va is None:
        if m == 2:                                  # m=2 时 all == p0-1
            va = V.get("p0-1")
        if va is None:
            return None
    return order2_shapley(singles, pairs, va["mpl"])


def section_b(records, meta_by_id, gray_shap):
    rows, per_img, n_img = [], [], 0
    for r in records:
        iid = r["image_id"]
        meta = meta_by_id.get(iid)
        if not meta:
            continue
        m = len(meta)
        o2 = image_order2(r, m)
        if o2 is None:
            continue
        n_img += 1
        gs = gray_shap.get(iid)
        gphi = None
        if gs and gs.get("n_cues") == m:
            gphi = [c["phi"] for c in gs["cues"]]
        for k in range(m):
            rows.append({"image_id": iid, "k": k, "cue": meta[k]["cue"],
                         "category": meta[k]["category"],
                         "phi_inpaint": o2["phi"][k],
                         "phi_gray": (gphi[k] if gphi else None)})
        pi = {"image_id": iid, "m": m, "residual_share": o2["residual_share"],
              "sum_phi": float(sum(o2["phi"]))}
        if gphi and m >= 2:
            pi["rho_within"] = spearman(o2["phi"], gphi)
            pi["top1_agree"] = int(np.argmax(o2["phi"])) == int(np.argmax(gphi))
        per_img.append(pi)

    out = {"n_images": n_img, "n_cues": len(rows), "per_cue": rows,
           "per_image": per_img}
    out["residual_share_median"] = med([x["residual_share"] for x in per_img])
    paired = [x for x in rows if x["phi_gray"] is not None]
    out["n_paired_cues"] = len(paired)
    out["n_paired_images"] = sum(1 for x in per_img if "rho_within" in x)
    if paired:
        a = np.array([x["phi_inpaint"] for x in paired], float)
        b = np.array([x["phi_gray"] for x in paired], float)
        out["spearman_global"] = spearman(a, b)
        out["rho_within_median"] = med([x.get("rho_within") for x in per_img])
        t1 = [x["top1_agree"] for x in per_img if "top1_agree" in x]
        out["top1_agreement"] = float(np.mean(t1)) if t1 else float("nan")
        out["n_top1"] = len(t1)
        out["phi_inpaint_median"] = float(np.median(a))
        out["phi_gray_median"] = float(np.median(b))
    bycat = defaultdict(lambda: {"i": [], "g": []})
    for x in rows:
        bycat[x["category"]]["i"].append(x["phi_inpaint"])
        if x["phi_gray"] is not None:
            bycat[x["category"]]["g"].append(x["phi_gray"])
    out["per_category"] = {c: {"n": len(d["i"]), "phi_inpaint_median": med(d["i"]),
                               "phi_gray_median": med(d["g"]) if d["g"] else None}
                           for c, d in bycat.items()}
    return out


# ---------------- (c) 非可加性 ----------------

def section_c(records, meta_by_id, gray_shap):
    per_img, inter, n_img = [], [], 0
    for r in records:
        iid = r["image_id"]
        meta = meta_by_id.get(iid)
        if not meta:
            continue
        m = len(meta)
        if m < 2:
            continue
        V = index_variants(r)
        try:
            singles = [V[f"s{k}"]["mpl"] for k in range(m)]
        except KeyError:
            continue
        va = V.get("all") or (V.get("p0-1") if m == 2 else None)
        if va is None:
            continue
        n_img += 1
        vN = va["mpl"]
        ssum = float(sum(singles))
        per_img.append({"image_id": iid, "m": m, "sum_singles": ssum, "vN": vN,
                        "ratio": vN / ssum if abs(ssum) > 1e-12 else float("nan"),
                        "sub_additive": vN < ssum})
        for k, l in itertools.combinations(range(m), 2):
            pv = V.get(f"p{k}-{l}") or V.get(f"p{l}-{k}")
            if pv is None:
                continue
            inter.append({"image_id": iid, "k": k, "l": l,
                          "cat_k": meta[k]["category"], "cat_l": meta[l]["category"],
                          "d_kl": pv["mpl"] - singles[k] - singles[l]})

    def signs(arr, thr=0.01):
        a = np.asarray(arr, float)
        if not len(a):
            return {"n": 0, "neg": 0, "zero": 0, "pos": 0, "median": float("nan")}
        return {"n": int(len(a)), "neg": int((a < -thr).sum()),
                "zero": int((np.abs(a) <= thr).sum()), "pos": int((a > thr).sum()),
                "median": float(np.median(a))}

    out = {"n_images": n_img, "per_image": per_img, "n_pairs": len(inter),
           "interactions": inter}
    if per_img:
        out["frac_sub_additive"] = float(np.mean([x["sub_additive"] for x in per_img]))
        out["ratio_median"] = med([x["ratio"] for x in per_img])
    out["inpaint_interaction_signs"] = signs([x["d_kl"] for x in inter])

    seen = {x["image_id"] for x in per_img}
    gs_sii = [val for iid in seen for val in (gray_shap.get(iid, {}).get("sii") or {}).values()]
    gs_emp = [val for iid in seen
              for val in (gray_shap.get(iid, {}).get("empty_interaction") or {}).values()]
    out["gray_sii_signs_same_images"] = signs(gs_sii)
    out["gray_empty_interaction_signs_same_images"] = signs(gs_emp)
    all_sii = [val for g in gray_shap.values() for val in (g.get("sii") or {}).values()]
    out["gray_sii_signs_all"] = signs(all_sii)
    return out


# ---------------- (d) 对照 / 伪影地板 ----------------

def gray_control_fallback(gray_controls):
    """更早的独立灰块对照跑(georanker_control_results.json)→ 非配对灰块地板。

    结构:[{image_id, cues:[{cue, category, area_frac, real_mpl,
                            controls:[{mpl, overlap_frac, cov}]}]}]
    放置位置与修复对照不同,因此只能做分布层面的非配对对照。
    """
    if not gray_controls:
        return None
    vals, flags, bycat = [], [], defaultdict(list)
    n_img = n_cue = 0
    for r in gray_controls:
        if not isinstance(r, dict):
            continue
        cues = r.get("cues") or []
        used = False
        for cu in cues:
            ctrls = [c.get("mpl") for c in (cu.get("controls") or [])
                     if isinstance(c, dict) and c.get("mpl") is not None]
            if not ctrls:
                continue
            used = True
            n_cue += 1
            vals += [float(x) for x in ctrls]
            rm = cu.get("real_mpl")
            if rm is not None:
                ok = bool(float(rm) > max(ctrls))
                flags.append(ok)
                bycat[cu.get("category") or "unknown"].append(ok)
        if used:
            n_img += 1
    if not vals:
        return None
    a = np.array(vals, float)
    out = {"n_images": n_img, "n_cues": n_cue, "n_placements": len(vals),
           "values": vals,
           "floor": {"median": float(np.median(a)),
                     "p90": float(np.percentile(a, 90)),
                     "max": float(a.max())},
           "per_category_flags": {c: list(v) for c, v in bycat.items()}}
    if flags:
        out["resolvable"] = {"n": len(flags), "rate": float(np.mean(flags))}
    return out


def section_d(controls, records, meta_by_id, gray_controls=None):
    """c<k>-<j> = 修复对照,cg<k>-<j> = 同一放置的灰块对照。

    没有 cg* 时(0 个配对放置)回退到 gray_controls:更早的独立灰块对照跑,
    放置位置不同 → **非配对**分布对照,所有产出都带 gray_source 标注。
    """
    if controls is None:
        return {"available": False,
                "note": "control file not yet available — section skipped"}
    main_by_id = {r["image_id"]: index_variants(r) for r in records}
    cvals, gvals, placements, per_cue = [], [], [], []
    n_img = 0
    for r in controls:
        iid = r["image_id"]
        meta = meta_by_id.get(iid)
        if not meta:
            continue
        V = index_variants(r)
        byk = defaultdict(lambda: {"c": [], "cg": []})
        for spec, v in V.items():
            if not v.get("cues"):
                continue
            k = int(v["cues"][0])
            j = v.get("placement")
            if spec.startswith("cg"):
                byk[k]["cg"].append((j, v))
            elif spec.startswith("c"):
                byk[k]["c"].append((j, v))
        if not byk:
            continue
        n_img += 1
        M = main_by_id.get(iid, {})
        for k, d in sorted(byk.items()):
            ci = {j: v for j, v in d["c"]}
            cg = {j: v for j, v in d["cg"]}
            cvals += [v["mpl"] for v in ci.values()]
            gvals += [v["mpl"] for v in cg.values()]
            for j in sorted(set(ci) & set(cg)):
                placements.append({"image_id": iid, "k": k, "j": j,
                                   "inpaint": ci[j]["mpl"], "gray": cg[j]["mpl"],
                                   "overlap_frac": ci[j].get("overlap_frac"),
                                   "coverage": ci[j].get("coverage"),
                                   "category": meta[k]["category"] if k < len(meta) else "unknown"})
            real_i = M.get(f"s{k}", {}).get("mpl")
            real_g = M.get(f"s{k}", {}).get("gray_mpl")
            per_cue.append({
                "image_id": iid, "k": k,
                "cue": meta[k]["cue"] if k < len(meta) else f"cue{k}",
                "category": meta[k]["category"] if k < len(meta) else "unknown",
                "real_inpaint": real_i, "real_gray": real_g,
                "ctrl_inpaint_max": (max(v["mpl"] for v in ci.values()) if ci else None),
                "ctrl_gray_max": (max(v["mpl"] for v in cg.values()) if cg else None),
                "n_ctrl_inpaint": len(ci), "n_ctrl_gray": len(cg)})

    out = {"available": True, "n_images": n_img, "n_cues": len(per_cue),
           "n_ctrl_inpaint": len(cvals), "n_ctrl_gray": len(gvals),
           "placements": placements, "per_cue": per_cue,
           "control_values_inpaint": [float(x) for x in cvals],
           "control_values_gray": [float(x) for x in gvals]}
    if cvals:
        a = np.array(cvals, float)
        out["floor_inpaint"] = {"median": float(np.median(a)),
                                "p90": float(np.percentile(a, 90)),
                                "max": float(a.max())}
    if gvals:
        b = np.array(gvals, float)
        out["floor_gray"] = {"median": float(np.median(b)),
                             "p90": float(np.percentile(b, 90)),
                             "max": float(b.max())}
    if placements:
        pi = np.array([x["inpaint"] for x in placements], float)
        pg = np.array([x["gray"] for x in placements], float)
        out["paired_placements"] = {
            "n": len(placements),
            "inpaint_median": float(np.median(pi)),
            "gray_median": float(np.median(pg)),
            "frac_inpaint_lt_gray": float(np.mean(pi < pg)),
            "delta_median": float(np.median(pi - pg)),
            "ratio_median": float(np.median(pi / np.maximum(pg, 1e-9)))}
    ri = [x for x in per_cue if x["real_inpaint"] is not None and x["ctrl_inpaint_max"] is not None]
    rg = [x for x in per_cue if x["real_gray"] is not None and x["ctrl_gray_max"] is not None]
    if ri:
        out["resolvable_inpaint"] = {
            "n": len(ri),
            "rate": float(np.mean([x["real_inpaint"] > x["ctrl_inpaint_max"] for x in ri]))}
    if rg:
        out["resolvable_gray"] = {
            "n": len(rg),
            "rate": float(np.mean([x["real_gray"] > x["ctrl_gray_max"] for x in rg]))}

    cat_i = defaultdict(list)
    cat_g = defaultdict(list)
    for x in ri:
        cat_i[x["category"]].append(bool(x["real_inpaint"] > x["ctrl_inpaint_max"]))
    for x in rg:
        cat_g[x["category"]].append(bool(x["real_gray"] > x["ctrl_gray_max"]))

    # 灰块一侧:优先同位置配对 cg*;没有就回退到更早的独立灰块跑(非配对)
    out["gray_source"] = "paired" if gvals else None
    if not gvals:
        fb = gray_control_fallback(gray_controls)
        if fb:
            out["gray_source"] = "fallback_unpaired"
            out["gray_fallback"] = {k: v for k, v in fb.items()
                                    if k not in ("values", "per_category_flags")}
            out["gray_fallback_note"] = (
                "no cg* placements in the inpaint control file; gray floor and gray "
                "resolvability come from the earlier independent gray control run "
                "(different random placements) — UNPAIRED distribution comparison")
            out["control_values_gray"] = fb["values"]
            out["n_ctrl_gray"] = fb["n_placements"]
            out["floor_gray"] = fb["floor"]
            if fb.get("resolvable"):
                out["resolvable_gray"] = dict(fb["resolvable"])
            cat_g = defaultdict(list, {c: list(v)
                                       for c, v in fb["per_category_flags"].items()})

    out["per_category_resolvable"] = {
        c: {"n_inpaint": len(cat_i.get(c, [])),
            "rate_inpaint": (float(np.mean(cat_i[c])) if cat_i.get(c) else None),
            "n_gray": len(cat_g.get(c, [])),
            "rate_gray": (float(np.mean(cat_g[c])) if cat_g.get(c) else None)}
        for c in sorted(set(cat_i) | set(cat_g))}
    return out


# ---------------- 报告 ----------------

def render_md(S, meta):
    L = []
    A = L.append
    A("# Inpainting vs gray masking: per-cue location leakage\n")
    A(f"- results file: `{meta['results']}`")
    A(f"- cue metadata: `{meta['cues_from']}`"
      + (f" (cache `{meta['cache']}`)" if meta["cues_from"] == "manifest" else ""))
    A(f"- gray baselines: {'available' if meta['use_gray'] else 'NOT used (vocabulary run)'}")
    A(f"- images in results file: **{meta['n_records']}** "
      f"(scored variants: {meta['n_variants']})")
    A(f"- generated from partial data: sections report their own usable-image counts\n")

    a = S["a"]
    A("## (a) Single-cue removal: gray vs inpaint\n")
    A(f"Usable images: **{a['n_images']}**, cues: **{a['n_cues']}** "
      f"(paired with a gray baseline: {a['n_paired']}).\n")
    if a["n_paired"]:
        A(f"- Spearman rho(gray, inpaint) = **{fmt(a['spearman_gray_inpaint'],3)}** "
          f"(Pearson r = {fmt(a.get('pearson_gray_inpaint'),3)})")
        A(f"- fraction with inpaint < gray = **{a['frac_inpaint_lt_gray']*100:.1f}%**")
        A(f"- median mPL: gray {fmt(a['gray_median'])} → inpaint "
          f"{fmt(a['inpaint_median_paired'])} "
          f"(median delta {fmt(a['delta_median'])}, median ratio "
          f"{fmt(a['ratio_median'],3)}x)")
    else:
        A(f"- no gray baseline available; median inpaint mPL = {fmt(a['inpaint_median'])}")
    A("")
    A("| category | n | inpaint median | gray median |")
    A("|---|---:|---:|---:|")
    for c, d in sorted(a["per_category"].items(),
                       key=lambda kv: -(kv[1]["inpaint_median"] if np.isfinite(kv[1]["inpaint_median"]) else -9)):
        A(f"| {c} | {d['n']} | {fmt(d['inpaint_median'])} | "
          f"{fmt(d['gray_median']) if d['gray_median'] is not None else 'n/a'} |")
    A("")

    b = S["b"]
    A("## (b) Order-2 anchored Shapley under inpainting\n")
    A(f"phi_k = d_k + 1/2 * sum_l d_kl, then anchored so that sum_k phi_k = v(N).\n")
    A(f"Usable images (all singles + all pairs + all): **{b['n_images']}**, "
      f"cues: **{b['n_cues']}**; paired with gray exact phi: "
      f"{b['n_paired_images']} images / {b['n_paired_cues']} cues.\n")
    A(f"- median truncation residual share |v(N)-sum phi^(2)|/|v(N)| = "
      f"**{fmt(b['residual_share_median'],3)}**")
    if b.get("n_paired_cues"):
        A(f"- global Spearman(phi_inpaint, phi_gray) = **{fmt(b.get('spearman_global'),3)}**")
        A(f"- within-image Spearman median = **{fmt(b.get('rho_within_median'),3)}** "
          f"(n={b['n_paired_images']} images)")
        A(f"- top-1 cue agreement = **{fmt(b.get('top1_agreement')*100 if np.isfinite(b.get('top1_agreement', np.nan)) else float('nan'),1)}%** "
          f"(n={b.get('n_top1',0)})")
        A(f"- median phi: gray {fmt(b.get('phi_gray_median'))} → inpaint "
          f"{fmt(b.get('phi_inpaint_median'))}")
    A("")
    A("| category | n | phi_inpaint median | phi_gray median |")
    A("|---|---:|---:|---:|")
    for c, d in sorted(b["per_category"].items(),
                       key=lambda kv: -(kv[1]["phi_inpaint_median"] if np.isfinite(kv[1]["phi_inpaint_median"]) else -9)):
        A(f"| {c} | {d['n']} | {fmt(d['phi_inpaint_median'])} | "
          f"{fmt(d['phi_gray_median']) if d['phi_gray_median'] is not None else 'n/a'} |")
    A("")

    c = S["c"]
    A("## (c) Non-additivity under inpainting\n")
    A(f"Usable images (singles + all, m>=2): **{c['n_images']}**; "
      f"pairs with an interaction value: **{c['n_pairs']}**.\n")
    if c["n_images"]:
        A(f"- fraction sub-additive (v(N) < sum_k v({{k}})) = "
          f"**{c.get('frac_sub_additive', float('nan'))*100:.1f}%**")
        A(f"- median v(N) / sum_k v({{k}}) = **{fmt(c.get('ratio_median'),3)}**")
    A("")
    A("| interaction estimator | n | d < -0.01 | ~0 | d > 0.01 | median |")
    A("|---|---:|---:|---:|---:|---:|")
    for tag, key in [("inpaint d_kl (empty context)", "inpaint_interaction_signs"),
                     ("gray SII (same images)", "gray_sii_signs_same_images"),
                     ("gray d_kl empty context (same images)",
                      "gray_empty_interaction_signs_same_images"),
                     ("gray SII (all 95 images)", "gray_sii_signs_all")]:
        s = c.get(key) or {}
        if not s.get("n"):
            continue
        A(f"| {tag} | {s['n']} | {s['neg']} | {s['zero']} | {s['pos']} | "
          f"{fmt(s['median'],3)} |")
    A("")

    d = S["d"]
    A("## (d) Artifact floor and resolvability (equal-area controls)\n")
    if not d.get("available"):
        A(f"_{d.get('note')}_\n")
    else:
        fb = d.get("gray_fallback")
        unpaired = d.get("gray_source") == "fallback_unpaired"
        A(f"Usable images: **{d['n_images']}**, cues: **{d['n_cues']}**; "
          f"placements scored: {d['n_ctrl_inpaint']} inpaint / {d['n_ctrl_gray']} gray.\n")
        if unpaired:
            A(f"> **Unpaired gray comparison.** The inpaint control run contains no "
              f"same-placement gray twins (`cg*`), so the gray side below comes from the "
              f"earlier, independent gray control run "
              f"(`{os.path.basename(meta.get('gray_controls') or '')}`: "
              f"{fb['n_images']} images / {fb['n_cues']} cues / {fb['n_placements']} "
              f"placements at **different random positions**). Gray vs inpaint is therefore "
              f"a *distribution-level* comparison, not a per-placement one.\n")
        fi, fg = d.get("floor_inpaint"), d.get("floor_gray")
        for tag, f in [("inpaint control floor", fi),
                       ("gray control floor" + (" (earlier run, different placements)"
                                                if unpaired else ""), fg)]:
            if f:
                A(f"- {tag}: median {fmt(f['median'])}, P90 {fmt(f['p90'])}, "
                  f"max {fmt(f['max'])}")
        if fi and fg:
            A(f"- **floor gray vs inpaint**: median {fmt(fg['median'])} vs "
              f"{fmt(fi['median'])} ({fmt(fi['median'] - fg['median'])}), "
              f"P90 {fmt(fg['p90'])} vs {fmt(fi['p90'])} "
              f"({fmt(fi['p90'] - fg['p90'])})"
              + ("  _[unpaired]_" if unpaired else ""))
        rgo, rio = d.get("resolvable_gray"), d.get("resolvable_inpaint")
        if rgo and rio:
            A(f"- **overall resolvability gray vs inpaint**: "
              f"{rgo['rate']*100:.1f}% (n={rgo['n']}) vs "
              f"{rio['rate']*100:.1f}% (n={rio['n']})"
              + ("  _[unpaired]_" if unpaired else ""))
        p = d.get("paired_placements")
        if p:
            A(f"- paired on identical placements (n={p['n']}): inpaint median "
              f"{fmt(p['inpaint_median'])} vs gray median {fmt(p['gray_median'])}; "
              f"inpaint < gray in {p['frac_inpaint_lt_gray']*100:.1f}% "
              f"(median delta {fmt(p['delta_median'])}, ratio {fmt(p['ratio_median'],3)}x)")
        for tag, key in [("inpaint", "resolvable_inpaint"), ("gray", "resolvable_gray")]:
            rr = d.get(key)
            if rr:
                A(f"- resolvability ({tag}, real > max of own controls): "
                  f"**{rr['rate']*100:.1f}%** (n={rr['n']})")
        A("")
        A("| category | n (inpaint) | resolvable inpaint | n (gray"
          + (", earlier run" if unpaired else "") + ") | resolvable gray |")
        A("|---|---:|---:|---:|---:|")
        for cat, dd in sorted(d.get("per_category_resolvable", {}).items(),
                              key=lambda kv: -(kv[1]["rate_inpaint"] or 0)):
            A(f"| {cat} | {dd['n_inpaint']} | "
              f"{'%.1f%%' % (dd['rate_inpaint']*100) if dd['rate_inpaint'] is not None else 'n/a'} | "
              f"{dd['n_gray']} | "
              f"{'%.1f%%' % (dd['rate_gray']*100) if dd['rate_gray'] is not None else 'n/a'} |")
        A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="inpaint vs gray 逐线索泄漏报告")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--controls", default=CONTROLS)
    ap.add_argument("--gray-controls", default=GRAY_CONTROLS,
                    help="没有同位置 cg* 时用来回退的更早灰块对照跑(非配对分布对照)")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--cues-from", choices=["sweep", "manifest"], default="sweep")
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--shapley", default=SHAP)
    ap.add_argument("--out-prefix", default="",
                    help="落盘文件名前缀(默认空 = inpaint_report.md / inpaint_summary.json;"
                         "词表口径用 vocab_)")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    d_md, d_json = default_outputs(a.out_prefix)
    if a.out_md is None:
        a.out_md = d_md
    if a.out_json is None:
        a.out_json = d_json

    records = load_json(a.results)
    if not records:
        print(f"[fatal] 主结果不可用或为空: {a.results}\n"
              f"        (GeoRanker 修复打分作业可能还没写出第一条记录)")
        return 1
    controls = load_json(a.controls)
    if controls is None:
        print(f"[info] 对照结果尚不可用({a.controls}) → 跳过 (d) 节")
    gray_controls = load_json(a.gray_controls)
    if gray_controls is None and controls is not None:
        print(f"[info] 没有更早的灰块对照({a.gray_controls}) → (d) 节灰块一侧可能为空")

    use_gray = (a.cues_from == "sweep")
    sweep_by_id = {}
    gray_shap = {}
    if use_gray:
        sw = load_json(a.sweep, [])
        sweep_by_id = {r["image_id"]: r for r in sw}
        gray_shap = {r["image_id"]: r for r in load_json(a.shapley, [])}
        if not sweep_by_id:
            print(f"[warn] 没读到 sweep({a.sweep}),灰块基线只能靠记录里的 gray_mpl")
    else:
        print("[info] --cues-from manifest:线索表来自 manifest,灰块对照全部跳过")

    meta_by_id, n_nometa = {}, 0
    for r in records:
        m = cue_meta(r, a.cues_from, a.cache, sweep_by_id)
        if m:
            meta_by_id[r["image_id"]] = m
        else:
            n_nometa += 1
    if n_nometa:
        print(f"[warn] {n_nometa} 张图拿不到线索元数据,已排除")

    n_var = sum(len(r.get("variants", [])) for r in records)
    print(f"载入 {len(records)} 张图 / {n_var} 个已打分变体"
          f"(cues-from={a.cues_from})")

    S = {"a": section_a(records, meta_by_id, use_gray),
         "b": section_b(records, meta_by_id, gray_shap if use_gray else {}),
         "c": section_c(records, meta_by_id, gray_shap if use_gray else {}),
         "d": section_d(controls, records, meta_by_id,
                        gray_controls if use_gray else None)}
    meta = {"results": os.path.abspath(a.results), "cues_from": a.cues_from,
            "cache": os.path.abspath(a.cache), "use_gray": use_gray,
            "n_records": len(records), "n_variants": n_var,
            "controls": (os.path.abspath(a.controls) if controls is not None else None),
            "gray_controls": (os.path.abspath(a.gray_controls)
                              if gray_controls is not None else None)}
    S["meta"] = meta

    print(f"\n(a) 单条移除:{S['a']['n_images']} 张可用 / {S['a']['n_cues']} 条线索"
          f"(配对灰块 {S['a']['n_paired']})")
    if S["a"]["n_paired"]:
        print(f"    rho={fmt(S['a']['spearman_gray_inpaint'],3)} | "
              f"inpaint<gray {S['a']['frac_inpaint_lt_gray']*100:.0f}% | "
              f"中位 {fmt(S['a']['gray_median'])} → {fmt(S['a']['inpaint_median_paired'])}")
    print(f"(b) 二阶 Shapley:{S['b']['n_images']} 张可用(s*+p*+all 齐全)/ "
          f"{S['b']['n_cues']} 条线索;配对灰块 φ 的 {S['b']['n_paired_images']} 张")
    if S["b"].get("n_paired_cues"):
        print(f"    全局 rho={fmt(S['b'].get('spearman_global'),3)} | 图内中位 "
              f"{fmt(S['b'].get('rho_within_median'),3)} | top-1 "
              f"{fmt(S['b'].get('top1_agreement',float('nan'))*100,1)}%")
    print(f"(c) 非可加性:{S['c']['n_images']} 张可用 / {S['c']['n_pairs']} 对")
    if S["c"]["n_images"]:
        print(f"    次可加 {S['c'].get('frac_sub_additive',float('nan'))*100:.0f}% | "
              f"v(N)/Σv 中位 {fmt(S['c'].get('ratio_median'),3)}")
    print(f"(d) 对照:{'不可用(尚未生成)' if not S['d'].get('available') else str(S['d']['n_images']) + ' 张可用'}")
    if S["d"].get("gray_source") == "fallback_unpaired":
        gfb = S["d"]["gray_fallback"]
        print(f"    灰块一侧回退到更早的独立对照跑(非配对):{gfb['n_images']} 图 / "
              f"{gfb['n_cues']} 线索 / {gfb['n_placements']} 个放置")
        fi, fg = S["d"].get("floor_inpaint"), S["d"].get("floor_gray")
        if fi and fg:
            print(f"    地板中位 gray {fmt(fg['median'])} vs inpaint {fmt(fi['median'])} | "
                  f"P90 {fmt(fg['p90'])} vs {fmt(fi['p90'])}")
        rg_, ri_ = S["d"].get("resolvable_gray"), S["d"].get("resolvable_inpaint")
        if rg_ and ri_:
            print(f"    可分辨率 gray {rg_['rate']*100:.1f}% (n={rg_['n']}) vs "
                  f"inpaint {ri_['rate']*100:.1f}% (n={ri_['n']})")

    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write(render_md(S, meta))
    json.dump(S, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {a.out_md}\nsaved {a.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
