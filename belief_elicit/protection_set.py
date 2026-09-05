"""Which cues should actually be removed? — selection strategies scored by residual risk.

Attribution (`shapley_v3`) answers *"which cue explains the measured belief shift"*.
Protection asks *"which set S should I treat, so that the released image `I (-) S` carries
as little location evidence as possible"* — a forward-looking, **signed** question.  This
module runs that comparison on the full gray lattice.

For every de-duplicated image with at least two players and a complete `2^P` lattice, and
for every budget k = 1 .. P-1, six strategies each propose a set S of size k:

===============  =============================================================
`exhaustive`     argmin of the primary risk over all C(P, k) subsets — the
                 lower envelope every other strategy is measured against
`greedy`         forward greedy on the primary risk (equals `exhaustive` at k=1)
`shapley_topk`   the k largest Shapley `phi` — *the attribution answer*
`signed_topk`    the k largest **signed** single contributions
                 R1(I) - R1(I (-) {j}), which unlike `phi` can be negative
`largest_area`   the k largest cue masks — the naive pixel-budget baseline
`random`         mean over 20 uniform draws (seed 0)
===============  =============================================================

Risks (all from `risk.py`, all against the content-free prior pi from
`location_prior.json`):

  R1  signed evidence  logit q(x*) - logit pi(x*)   **primary**
  R2  local epsilon at r = 200 km, worst pair
  R3  p(x*)
  R4  km error of the attacker's argmax
  R5  residual mPL against the all-masked release

Outputs `protection_set_results.json` + `protection_set_report.md`.

Run: python -m belief_elicit.protection_set
"""
import argparse
import itertools
import json
import math
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

from belief_elicit import risk
from belief_elicit.attribution import spearman
from belief_elicit.cues import cue_masks_of
from belief_elicit.results import (INPAINT, LOCATION_PRIOR, PROTECTION_SET,
                                   PROTECTION_SET_MD, SAM3_DIR, SHAPLEY_V3, build_q,
                                   by_image, index_variants, load_gallery, load_inpaint,
                                   load_lattice, load_shapley, load_sweep)

MERGE_KM = 2.0
ETAS = (0.0, 0.5, 1.0)
N_RANDOM = 20
SEED = 0
TOL = 1e-9
#: order used in every table
STRATEGIES = ["exhaustive", "greedy", "shapley_topk", "signed_topk", "largest_area", "random"]


# ---------------- inputs ----------------

def load_prior(path=LOCATION_PRIOR, labels=None):
    """-> (pi dict, meta dict). Falls back to uniform, loudly, if the GPU job never ran."""
    d = None
    if os.path.exists(path):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] cannot read {path}: {e}", flush=True)
    if d and d.get("pi"):
        H, Hu = d["entropy"]["pi"], d["entropy"]["uniform"]
        return d["pi"], {"source": os.path.basename(path), "top_label": d["top_label"],
                         "top_p": d["top_p"], "entropy": H, "uniform_entropy": Hu,
                         "entropy_ratio": H / Hu, "fallback": False}
    print("!" * 78)
    print("!! location_prior.json missing or unusable — FALLING BACK TO THE UNIFORM PRIOR.")
    print("!! Every R1 / R2 number below is then measured against uniform, not against the")
    print("!! adversary's real content-free belief. Re-run belief_elicit.location_prior.")
    print("!" * 78, flush=True)
    pi = {lab: 1.0 / len(labels) for lab in labels}
    Hu = math.log(len(labels))
    return pi, {"source": "uniform-fallback", "top_label": None, "top_p": 1.0 / len(labels),
                "entropy": Hu, "uniform_entropy": Hu, "entropy_ratio": 1.0, "fallback": True}


def player_masks(image_id, groups, n_cues):
    """-> (list of per-player boolean masks, (W, H)); None when the cue record is unusable."""
    got = cue_masks_of(SAM3_DIR, image_id)
    if got is None:
        return None
    _, _, masks, size = got
    if len(masks) != n_cues:
        return None
    out = []
    for g in groups:
        u = np.zeros_like(masks[0])
        for k in g["member_indices"]:
            u |= masks[k]
        out.append(u)
    return out, size


def area_frac(masks, size, S):
    """Fraction of the frame covered by the union of the players in S."""
    W, H = size
    if not S:
        return 0.0
    u = np.zeros_like(masks[0])
    for j in S:
        u |= masks[j]
    return float(u.sum()) / (W * H)


# ---------------- per-image evaluation ----------------

def evaluate_image(r, lattice, s, rg, pi, pi_vec):
    """One sweep record + its shapley_v3 record -> the per-image case, or (None, reason)."""
    m = r["n_cues"]
    q = build_q(r, lattice)
    if len(q) != 2 ** m:
        return None, "incomplete lattice"
    groups = s["cues"]
    P = len(groups)
    if P < 2:
        return None, f"only {P} de-duplicated player(s)"

    # merged player -> original cue indices; a merged subset's belief is the belief of
    # the union of its members' original indices (must already exist in the lattice).
    members = [frozenset(g["member_indices"]) for g in groups]
    qm = {}
    for size_ in range(P + 1):
        for S in itertools.combinations(range(P), size_):
            orig = frozenset().union(*[members[j] for j in S]) if S else frozenset()
            if orig not in q:
                return None, f"merged subset {sorted(S)} missing from the lattice"
            qm[frozenset(S)] = q[orig]

    got = player_masks(r["image_id"], groups, m)
    if got is None:
        return None, "cue masks unreadable / count mismatch"
    masks, size = got

    x_star = r["true_label"]
    q_all_vec = rg.vec(q[frozenset(range(m))])

    # full risk vector for every subset, once
    cache = {}
    for S, qs in qm.items():
        qv = rg.vec(qs)
        vs_pi = rg.metrics(qv, pi_vec)
        vs_all = rg.metrics(qv, q_all_vec)
        cache[S] = {
            "R1": risk.signed_evidence(qs, x_star, pi),
            "R2": vs_pi["eps"][200.0]["max"],
            "R3": risk.p_true(qs, x_star),
            "R4": risk.err_km(qs, x_star, rg.coords),
            "R5": vs_all["mpl"],
            "rank_true": risk.rank_true(qs, x_star),
            "exp_err_km": risk.exp_err_km(qs, x_star, rg.coords),
            "mpl_vs_pi": vs_pi["mpl"],
            "mpl_max_vs_pi": vs_pi["mpl_max"],
            "eps_local": {str(int(rad)): vs_pi["eps"][rad] for rad in rg.radii},
            "area": area_frac(masks, size, S),
        }

    empty = frozenset()
    signed = [cache[empty]["R1"] - cache[frozenset([j])]["R1"] for j in range(P)]
    per_player = [{"cue": g["cue"], "category": g["category"] or "unknown",
                   "phi": g["phi"], "v_single": g["v_single"],
                   "signed_contrib": signed[j],
                   "dp_true": cache[frozenset([j])]["R3"] - cache[empty]["R3"],
                   "area_frac": area_frac(masks, size, (j,)),
                   "n_members": g["n_members"]}
                  for j, g in enumerate(groups)]

    return {"image_id": r["image_id"], "true_label": x_star, "n_cues_orig": m,
            "P": P, "merged": s["merged"], "country_hit": r["country_hit"],
            "per_player": per_player, "cache": cache}, None


# ---------------- strategies ----------------

def select(case, k, rng):
    """-> {strategy: S or [S, ...] for `random`} at budget k."""
    P, cache, pp = case["P"], case["cache"], case["per_player"]
    subsets = [frozenset(S) for S in itertools.combinations(range(P), k)]

    out = {}
    out["exhaustive"] = min(subsets, key=lambda S: (cache[S]["R1"], sorted(S)))
    out["exhaustive_r2"] = min(subsets, key=lambda S: (cache[S]["R2"], sorted(S)))

    S = frozenset()
    for _ in range(k):                                        # forward greedy on R1
        S = min((S | {j} for j in range(P) if j not in S),
                key=lambda T: (cache[T]["R1"], sorted(T)))
    out["greedy"] = S

    def topk(key):
        return frozenset(sorted(range(P), key=lambda j: (-key(j), j))[:k])

    out["shapley_topk"] = topk(lambda j: pp[j]["phi"])
    out["signed_topk"] = topk(lambda j: pp[j]["signed_contrib"])
    out["largest_area"] = topk(lambda j: pp[j]["area_frac"])
    out["random"] = [frozenset(rng.choice(P, size=k, replace=False).tolist())
                     for _ in range(N_RANDOM)]
    return out


def metrics_of(cache, sel):
    """One selection (a set, or a list of sets for `random`) -> averaged metric dict."""
    keys = ("R1", "R2", "R3", "R4", "R5", "rank_true", "exp_err_km", "area")
    if isinstance(sel, list):
        vals = {k: float(np.nanmean([cache[S][k] for S in sel])) for k in keys}
        vals["S"] = None
        vals["n_draws"] = len(sel)
        return vals
    vals = {k: cache[sel][k] for k in keys}
    vals["S"] = sorted(sel)
    return vals


# ---------------- aggregation ----------------

def agg(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------- the inpaint replicate ----------------

def inpaint_cases(sweep, shap, inp, rg, pi, pi_vec):
    """k <= 2 replicate on the LaMa arm (un-merged images only, where s{j}/p{a}-{b} exist)."""
    cases, skipped = [], []
    for r in sweep:
        iid = r["image_id"]
        s, ir = shap.get(iid), inp.get(iid)
        m = r["n_cues"]
        if s is None or ir is None or m < 2:
            continue
        if s["merged"]:
            skipped.append((iid, "merged players have no inpaint pair variant"))
            continue
        var = index_variants(ir)
        need = ([f"s{j}" for j in range(m)] +
                [f"p{a}-{b}" for a, b in itertools.combinations(range(m), 2)] + ["all"])
        if any(sp not in var or "prior" not in var[sp] for sp in need):
            skipped.append((iid, "missing inpaint variant / prior"))
            continue
        got = player_masks(iid, s["cues"], m)
        if got is None:
            skipped.append((iid, "cue masks unreadable"))
            continue
        masks, size = got

        qm = {frozenset(): r["posterior"], frozenset(range(m)): var["all"]["prior"]}
        for j in range(m):
            qm[frozenset([j])] = var[f"s{j}"]["prior"]
        for a, b in itertools.combinations(range(m), 2):
            qm[frozenset([a, b])] = var[f"p{a}-{b}"]["prior"]

        x_star = r["true_label"]
        q_all_vec = rg.vec(var["all"]["prior"])
        cache = {}
        for S, qs in qm.items():
            qv = rg.vec(qs)
            vs_pi, vs_all = rg.metrics(qv, pi_vec), rg.metrics(qv, q_all_vec)
            cache[S] = {"R1": risk.signed_evidence(qs, x_star, pi),
                        "R2": vs_pi["eps"][200.0]["max"],
                        "R3": risk.p_true(qs, x_star),
                        "R4": risk.err_km(qs, x_star, rg.coords),
                        "R5": vs_all["mpl"],
                        "rank_true": risk.rank_true(qs, x_star),
                        "exp_err_km": risk.exp_err_km(qs, x_star, rg.coords),
                        "area": area_frac(masks, size, S)}
        empty = frozenset()
        pp = [{"cue": g["cue"], "category": g["category"] or "unknown",
               "phi": g.get("phi_inpaint", g["phi"]),
               "signed_contrib": cache[empty]["R1"] - cache[frozenset([j])]["R1"],
               "dp_true": cache[frozenset([j])]["R3"] - cache[empty]["R3"],
               "area_frac": area_frac(masks, size, (j,)), "n_members": 1}
              for j, g in enumerate(s["cues"])]
        cases.append({"image_id": iid, "true_label": x_star, "P": m, "n_cues_orig": m,
                      "merged": False, "per_player": pp, "cache": cache})
    return cases, skipped


# ---------------- driver ----------------

def run_strategies(cases, k_max=None):
    """-> (per-image selections, aggregated table rows, the raw regret lists)."""
    rows = defaultdict(lambda: defaultdict(list))       # (strategy, k) -> metric -> [values]
    per_image = []
    violations = []
    for case in cases:
        rng = np.random.default_rng(SEED)
        P = case["P"]
        ks = range(1, P) if k_max is None else range(1, min(P, k_max + 1))
        sels = {}
        for k in ks:
            chosen = select(case, k, rng)
            best1 = min(case["cache"][frozenset(S)]["R1"]
                        for S in itertools.combinations(range(P), k))
            best2 = min(case["cache"][frozenset(S)]["R2"]
                        for S in itertools.combinations(range(P), k))
            entry = {}
            for name in STRATEGIES + ["exhaustive_r2"]:
                mv = metrics_of(case["cache"], chosen[name])
                mv["regret_R1"] = mv["R1"] - best1
                mv["regret_R2"] = mv["R2"] - best2
                mv["optimal_R1"] = bool(mv["regret_R1"] <= TOL)
                mv["optimal_R2"] = bool(mv["regret_R2"] <= TOL)
                if name == "random":                    # fraction of draws that are optimal
                    mv["optimal_R1"] = float(np.mean(
                        [case["cache"][S]["R1"] - best1 <= TOL for S in chosen[name]]))
                    mv["optimal_R2"] = float(np.mean(
                        [case["cache"][S]["R2"] - best2 <= TOL for S in chosen[name]]))
                if name != "random" and mv["regret_R1"] < -TOL:
                    violations.append((case["image_id"], k, name, mv["regret_R1"]))
                entry[name] = mv
                for met in ("R1", "R2", "R3", "R4", "R5", "area", "regret_R1", "regret_R2"):
                    rows[(name, k)][met].append(mv[met])
                rows[(name, k)]["optimal"].append(float(mv["optimal_R1"]))
                rows[(name, k)]["optimal_R2"].append(float(mv["optimal_R2"]))
            entry["best_R1"], entry["best_R2"] = best1, best2
            sels[k] = entry
        per_image.append({"image_id": case["image_id"], "true_label": case["true_label"],
                          "P": P, "n_cues_orig": case["n_cues_orig"],
                          "merged": case["merged"],
                          "R1_full": case["cache"][frozenset()]["R1"],
                          "R1_allmask": case["cache"][frozenset(range(P))]["R1"],
                          "R2_full": case["cache"][frozenset()]["R2"],
                          "R3_full": case["cache"][frozenset()]["R3"],
                          "per_player": case["per_player"],
                          "selections": {str(k): {n: v for n, v in e.items()
                                                  if n not in ("best_R1", "best_R2")}
                                         | {"best_R1": e["best_R1"], "best_R2": e["best_R2"]}
                                         for k, e in sels.items()}})
    table = []
    for (name, k), mets in sorted(rows.items(), key=lambda kv: (STRATEGIES.index(kv[0][0])
                                                               if kv[0][0] in STRATEGIES
                                                               else 99, kv[0][1])):
        row = {"strategy": name, "k": k, "n_images": len(mets["R1"])}
        for met in ("R1", "R2", "R3", "R4", "R5", "area", "regret_R1", "regret_R2"):
            row[met] = agg(mets[met])
        row["pct_optimal"] = 100.0 * float(np.mean(mets["optimal"]))
        row["pct_optimal_R2"] = 100.0 * float(np.mean(mets["optimal_R2"]))
        table.append(row)
    return per_image, table, violations


def min_k_analysis(cases, per_image):
    """Smallest k at which each strategy pushes R1 below eta, plus the infeasible share."""
    out = {}
    by_id = {p["image_id"]: p for p in per_image}
    for eta in ETAS:
        rec = {"eta": eta, "n_images": len(cases), "infeasible": 0, "per_strategy": {}}
        feas = {}
        for case in cases:
            P = case["cache"]
            n = case["P"]
            best_any = min(P[frozenset(S)]["R1"]
                           for k in range(1, n)
                           for S in itertools.combinations(range(n), k))
            feas[case["image_id"]] = best_any <= eta
            if not feas[case["image_id"]]:
                rec["infeasible"] += 1
        rec["infeasible_frac"] = rec["infeasible"] / max(len(cases), 1)
        for name in STRATEGIES:
            ks, never = [], 0
            for case in cases:
                iid = case["image_id"]
                if not feas[iid]:
                    continue
                sel = by_id[iid]["selections"]
                hit = None
                for k in sorted(int(x) for x in sel):
                    if sel[str(k)][name]["R1"] <= eta:
                        hit = k
                        break
                if hit is None:
                    never += 1
                else:
                    ks.append(hit)
            nfeas = sum(feas.values())
            rec["per_strategy"][name] = {
                "n_feasible": nfeas, "n_reached": len(ks),
                "pct_never_reached": 100.0 * never / max(nfeas, 1),
                "mean_k": float(np.mean(ks)) if ks else float("nan"),
                "median_k": float(np.median(ks)) if ks else float("nan")}
        out[str(eta)] = rec
    return out


def misleading_analysis(cases):
    """Signed single contributions: how often is a cue's removal a gift to the attacker?"""
    rows = [dict(p, image_id=c["image_id"]) for c in cases for p in c["per_player"]]
    sc = [p["signed_contrib"] for p in rows]
    ph = [p["phi"] for p in rows]
    by_cat = defaultdict(list)
    for p in rows:
        by_cat[p["category"]].append(p["signed_contrib"])
    top1_neg = top1 = 0
    top1_rows = []
    for c in cases:
        j = max(range(c["P"]), key=lambda i: (c["per_player"][i]["phi"], -i))
        top1 += 1
        p = c["per_player"][j]
        top1_rows.append({"image_id": c["image_id"], "cue": p["cue"],
                          "category": p["category"], "phi": p["phi"],
                          "signed_contrib": p["signed_contrib"]})
        if p["signed_contrib"] < 0:
            top1_neg += 1
    return {
        "n_players": len(rows),
        "frac_negative": float(np.mean([x < 0 for x in sc])),
        "frac_p_true_up": float(np.mean([p["dp_true"] > 0 for p in rows
                                         if "dp_true" in p])) if rows else float("nan"),
        "mean_signed": float(np.mean(sc)), "median_signed": float(np.median(sc)),
        "by_category": {k: {"n": len(v), "frac_negative": float(np.mean([x < 0 for x in v])),
                            "median_signed": float(np.median(v))}
                        for k, v in sorted(by_cat.items())},
        "shapley_top1": {"n": top1, "n_negative": top1_neg,
                         "frac_negative": top1_neg / max(top1, 1), "rows": top1_rows},
        "corr_phi_signed_pearson": pearson(ph, sc),
        "corr_phi_signed_spearman": spearman(ph, sc),
    }


# ---------------- report ----------------

def _f(x, n=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def write_report(res, path):
    L = []
    W = L.append
    pr = res["prior"]
    W("# Protection-set selection — which cues to remove, and what is left behind\n")
    W(f"Generated by `belief_elicit.protection_set` over {res['meta']['n_images']} "
      f"de-duplicated images with `P >= 2` players and a complete `2^P` gray lattice "
      f"(of {res['meta']['n_candidates']} candidates).\n")
    W("## 0. The reference prior pi\n")
    W(f"Source `{pr['source']}` — top label **{pr['top_label']}** at p = {_f(pr['top_p'], 4)} "
      f"(uniform = {_f(1.0 / res['meta']['n_labels'], 4)}); "
      f"H(pi) = {_f(pr['entropy'])} nats vs H(uniform) = {_f(pr['uniform_entropy'])} "
      f"({pr['entropy_ratio'] * 100:.1f} %). "
      + ("**Uniform fallback in use — the GPU elicitation did not run.**\n"
         if pr["fallback"] else "The prior is diffuse, as it should be.\n"))

    for met, title, unit in (("R1", "R1 — signed evidence vs pi (nats, lower = safer)", "nats"),
                             ("R2", "R2 — local epsilon at 200 km, worst pair "
                                    "(nats/1000 km, lower = safer)", "nats/1000km")):
        W(f"\n## {'1' if met == 'R1' else '2'}. Headline: {title}\n")
        W(f"| strategy | k | n | mean {met} | median {met} | mean regret {met} "
          f"| % optimal {met} | mean area |")
        W("|---|---|---|---|---|---|---|---|")
        table = res["summary"]["table"]
        if met == "R1":
            table = [r for r in table if r["strategy"] != "exhaustive_r2"]
        else:                                      # put the R2 envelope on top
            table = ([r for r in table if r["strategy"] == "exhaustive_r2"] +
                     [r for r in table if r["strategy"] != "exhaustive_r2"])
        opt = "pct_optimal" if met == "R1" else "pct_optimal_R2"
        for row in table:
            W(f"| {row['strategy']} | {row['k']} | {row['n_images']} | "
              f"{_f(row[met]['mean'])} | {_f(row[met]['median'])} | "
              f"{_f(row['regret_' + met]['mean'])} | {row[opt]:.0f} | "
              f"{_f(row['area']['mean'], 4)} |")
        W(f"\n`regret` and `% optimal` are measured against the argmin of {met} over all "
          f"C(P, k) subsets: `exhaustive` for R1, `exhaustive_r2` for R2. Every strategy "
          f"except `exhaustive_r2` selects on the **primary** risk R1, so the non-zero R2 "
          f"regret on the `exhaustive` row is the price of optimising R1 instead of R2. "
          f"`mean area` is the masked fraction of the frame. Unit: {unit}.")

    W("\n## 3. Secondary risks at the selected sets\n")
    W("| strategy | k | mean R3 p(x*) | mean R4 km error | mean R5 residual vs all-masked |")
    W("|---|---|---|---|---|")
    for row in res["summary"]["table"]:
        if row["strategy"] == "exhaustive_r2":
            continue
        W(f"| {row['strategy']} | {row['k']} | {_f(row['R3']['mean'], 4)} | "
          f"{_f(row['R4']['mean'], 1)} | {_f(row['R5']['mean'], 4)} |")

    W("\n## 4. Minimal budget k to bring R1 below eta\n")
    for eta, rec in res["summary"]["min_k"].items():
        W(f"\n**eta = {eta} nats** — infeasible for {rec['infeasible']}/{rec['n_images']} "
          f"images ({rec['infeasible_frac'] * 100:.1f} %: no proper subset S of N reaches it).\n")
        W("| strategy | n feasible | reached | % never reached | mean k (of reached) "
          "| median k |")
        W("|---|---|---|---|---|---|")
        for name in STRATEGIES:
            s = rec["per_strategy"][name]
            W(f"| {name} | {s['n_feasible']} | {s['n_reached']} | "
              f"{s['pct_never_reached']:.0f} | {_f(s['mean_k'], 2)} | {_f(s['median_k'], 2)} |")

    ml = res["summary"]["misleading"]
    W("\n## 5. Misleading cues — the sign attribution cannot see\n")
    W(f"Of {ml['n_players']} de-duplicated players, **{ml['frac_negative'] * 100:.1f} %** have a "
      f"*negative* signed single contribution: removing that cue alone **raises** the "
      f"attacker's log-odds on the truth. Median signed contribution "
      f"{_f(ml['median_signed'])} nats, mean {_f(ml['mean_signed'])}.\n")
    W(f"Correlation of Shapley phi with the signed contribution: "
      f"Pearson r = {_f(ml['corr_phi_signed_pearson'], 3)}, "
      f"Spearman rho = {_f(ml['corr_phi_signed_spearman'], 3)}.\n")
    W(f"Among the {ml['shapley_top1']['n']} Shapley top-1 picks, "
      f"**{ml['shapley_top1']['n_negative']}** "
      f"({ml['shapley_top1']['frac_negative'] * 100:.1f} %) are misleading cues — "
      f"attribution would have told the user to remove exactly the cue that was helping them.\n")
    W("| category | n | % negative | median signed contribution |")
    W("|---|---|---|---|")
    for k, v in sorted(ml["by_category"].items(), key=lambda kv: -kv[1]["frac_negative"]):
        W(f"| {k} | {v['n']} | {v['frac_negative'] * 100:.0f} | {_f(v['median_signed'])} |")

    ip = res.get("inpaint") or {}
    W("\n## 6. Robustness: the LaMa inpainting arm (k <= 2)\n")
    if not ip.get("table"):
        W("No usable inpaint cases.")
    else:
        W(f"{ip['n_images']} un-merged images with a complete singles+pairs inpaint set "
          f"({ip['n_skipped']} skipped).\n")
        W("| strategy | k | n | mean R1 | mean regret R1 | % optimal | mean R2 | mean area |")
        W("|---|---|---|---|---|---|---|---|")
        for row in ip["table"]:
            if row["strategy"] == "exhaustive_r2":
                continue
            W(f"| {row['strategy']} | {row['k']} | {row['n_images']} | "
              f"{_f(row['R1']['mean'])} | {_f(row['regret_R1']['mean'])} | "
              f"{row['pct_optimal']:.0f} | {_f(row['R2']['mean'])} | "
              f"{_f(row['area']['mean'], 4)} |")
        mi = ip["misleading"]
        W(f"\nMisleading fraction on the inpaint arm: {mi['frac_negative'] * 100:.1f} % of "
          f"{mi['n_players']} cues (gray arm: {ml['frac_negative'] * 100:.1f} %); "
          f"phi/signed Spearman rho = {_f(mi['corr_phi_signed_spearman'], 3)}.")

    sa = res["summary"]["sanity"]
    W("\n## 7. Sanity checks\n")
    W(f"- R1(full image) >= R1(all cues masked) on "
      f"{sa['frac_full_ge_allmask'] * 100:.1f} % of images "
      f"(masking everything should not *help* the attacker).")
    W(f"- exhaustive <= every other strategy on R1: "
      f"{'OK' if not sa['violations'] else 'VIOLATED ' + str(sa['violations'][:5])}.")
    W(f"- H(pi) = {_f(pr['entropy'])} nats "
      f"({pr['entropy_ratio'] * 100:.1f} % of uniform).")
    if res["meta"]["skipped"]:
        W(f"- skipped images ({len(res['meta']['skipped'])}): "
          + "; ".join(f"`{i}` ({w})" for i, w in res["meta"]["skipped"]))
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("saved", path)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=PROTECTION_SET)
    ap.add_argument("--report", default=PROTECTION_SET_MD)
    ap.add_argument("--prior", default=LOCATION_PRIOR)
    ap.add_argument("--no-inpaint", action="store_true")
    args = ap.parse_args()

    gv = load_gallery()
    rg = risk.RiskGeometry(gv, merge_km=MERGE_KM)
    pi, pmeta = load_prior(args.prior, labels=[g["label"] for g in gv])
    pi_vec = rg.vec(pi)
    print(f"prior: {pmeta['source']} top={pmeta['top_label']!r} "
          f"H={pmeta['entropy']:.4f} ({pmeta['entropy_ratio'] * 100:.1f}% of uniform)")

    sweep = load_sweep()
    lattice = load_lattice()
    shap = by_image(load_shapley(SHAPLEY_V3))

    cases, skipped = [], []
    for r in sweep:
        s = shap.get(r["image_id"])
        if s is None:
            skipped.append((r["image_id"], "no shapley_v3 record"))
            continue
        case, why = evaluate_image(r, lattice, s, rg, pi, pi_vec)
        (cases.append(case) if case else skipped.append((r["image_id"], why)))
    print(f"cases: {len(cases)} usable / {len(sweep)} sweep images "
          f"({len(skipped)} skipped)")

    per_image, table, violations = run_strategies(cases)
    mink = min_k_analysis(cases, per_image)
    misl = misleading_analysis(cases)
    full_ge = float(np.mean([p["R1_full"] >= p["R1_allmask"] for p in per_image]))

    res = {"meta": {"n_images": len(cases), "n_candidates": len(sweep),
                    "n_labels": len(gv), "merge_km": MERGE_KM, "etas": list(ETAS),
                    "n_random": N_RANDOM, "seed": SEED, "strategies": STRATEGIES,
                    "radii": list(rg.radii), "skipped": skipped},
           "prior": pmeta,
           "images": per_image,
           "summary": {"table": table, "min_k": mink, "misleading": misl,
                       "sanity": {"frac_full_ge_allmask": full_ge,
                                  "violations": violations}}}

    if not args.no_inpaint:
        icases, iskip = inpaint_cases(sweep, shap, load_inpaint(INPAINT), rg, pi, pi_vec)
        icases2 = [c for c in icases if c["P"] >= 2]
        i_per_image, i_table, i_viol = run_strategies(icases2, k_max=2)
        res["inpaint"] = {"n_images": len(icases2), "n_skipped": len(iskip),
                          "skipped": iskip, "table": i_table,
                          "misleading": misleading_analysis(icases2),
                          "violations": i_viol, "images": i_per_image}
        print(f"inpaint replicate: {len(icases2)} images ({len(iskip)} skipped)")

    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=float)
    print("saved", args.out)
    write_report(res, args.report)

    # ---- sanity prints ----
    print(f"\n[sanity] R1(full) >= R1(all-masked) on {full_ge * 100:.1f}% of images")
    assert not violations, f"exhaustive was beaten: {violations[:5]}"
    print(f"[sanity] exhaustive <= every other strategy on R1: OK "
          f"({sum(r['n_images'] for r in table if r['strategy'] == 'exhaustive')} (image,k) cells)")
    print(f"[sanity] H(pi) = {pmeta['entropy']:.4f} nats "
          f"({pmeta['entropy_ratio'] * 100:.1f}% of uniform)")
    print(f"[sanity] misleading players: {misl['frac_negative'] * 100:.1f}% "
          f"({misl['shapley_top1']['n_negative']}/{misl['shapley_top1']['n']} Shapley top-1 picks)")


if __name__ == "__main__":
    main()
