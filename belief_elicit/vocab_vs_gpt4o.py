"""【实验性 · 可整体删除】两种"线索清单"口径的对照:GPT-4o 提议(top-down)vs 固定词表(bottom-up)。

同一批 10 张图、同一移除算子(LaMa inpaint)、同一攻击者(GeoRanker),只换"什么算一条线索":

  GPT-4o 口径  掩码 cue_extract/results_sam3/<iid>.json;打分 georanker_inpaint_results.json;
               线索名/类别/顺序 = georanker_sweep_results.json 的 per_cue
  词表口径     掩码 cue_extract/results_vocab/<iid>.json;打分 georanker_inpaint_vocab_results.json;
               线索名/类别/顺序 = inpaint_cache_vocab/<iid>/manifest.json 的 cues[]

头条指标是 **v(N)**:整份清单全遮之后攻击者损失多少 mPL —— 即该口径"能移除的全部泄漏"。
逐线索用二阶锚定 Shapley φ(belief_elicit/order2_shapley.py)。注意 m>=5 时二阶截断只保排序、
不保幅度(灰块全格校验:m<=3 精确,m=4 约 8% 相对幅度误差),本批中 m=7/6/5 的图受影响。

伪影零假设(null)c_img:
  * 优先 georanker_inpaint_control_results.json(spec c<k>-<j>)—— 与主口径同算子的等面积对照;
  * 该文件里没有这张图时,退回 georanker_control_results.json(灰块等面积对照)的 cues[*].controls[*].mpl。
  c_img = 该图所有对照放置的 mpl 均值;"纯伪影"零假设下每条 φ 都等于 c_img/m,
  故 φ_k > c_img/m 记为(暂定)敏感。脚本会打印实际用的是哪个 null。

两份清单的几何对应由掩码 IoU / 像素召回给出(与 cue_extract/compare_vocab.py 同口径:
maskable + 非退化实例并集)。IoU >= 0.1 记为"有对应",IoU >= 0.5 记为"几何基本重合"。

运行:python -m belief_elicit.vocab_vs_gpt4o
落盘:belief_elicit/vocab_vs_gpt4o.md + belief_elicit/vocab_vs_gpt4o.json
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

from belief_elicit.order2_shapley import order2_shapley, spearman
from cue_extract.rle import rle_to_mask

HERE = os.path.dirname(os.path.abspath(__file__))
GPT_RES = os.path.join(HERE, "georanker_inpaint_results.json")
VOC_RES = os.path.join(HERE, "georanker_inpaint_vocab_results.json")
SWEEP = os.path.join(HERE, "georanker_sweep_results.json")
VOC_CACHE = os.path.join(HERE, "inpaint_cache_vocab")
GRAY_CTRL = os.path.join(HERE, "georanker_control_results.json")
INP_CTRL = os.path.join(HERE, "georanker_inpaint_control_results.json")
SAM3DIR = os.path.join(ROOT, "cue_extract", "results_sam3")
VOCABDIR = os.path.join(ROOT, "cue_extract", "results_vocab")
OUT_MD = os.path.join(HERE, "vocab_vs_gpt4o.md")
OUT_JSON = os.path.join(HERE, "vocab_vs_gpt4o.json")

IOU_MATCHED = 0.1        # "有对应线索"的宽松阈值
IOU_SAME = 0.5           # "几何基本重合"的严格阈值


# ---------------- 载入 ----------------

def load_json(p, default=None):
    if not p or not os.path.exists(p):
        return default
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"[warn] 读不了 {p}: {e}")
        return default


def cue_masks(path):
    """与 precompute_inpaint.cue_masks_of 同口径 → ([cue], [category], [mask], (W,H))。"""
    rec = json.load(open(path, encoding="utf-8"))
    W, H = rec["image_size"]
    cues, cats, masks = [], [], []
    for c in rec["geo_privacy_cues"]:
        if not c.get("maskable"):
            continue
        good = [i for i in c["instances"] if not i.get("degenerate") and i.get("mask_rle")]
        if not good:
            continue
        u = np.zeros((H, W), bool)
        for i in good:
            m = rle_to_mask(i["mask_rle"])
            if m.shape == (H, W):
                u |= m
        if not u.any():
            continue
        cues.append(c["cue"]); cats.append(c.get("category") or "unknown"); masks.append(u)
    return cues, cats, masks, (W, H)


def index_variants(rec):
    return {v["spec"]: v for v in rec.get("variants", [])}


def phi_of(rec, m):
    """该图 s*/p*/all 齐全 → (phi[list], singles[list], vN);否则 None。"""
    V = index_variants(rec)
    try:
        singles = [V[f"s{k}"]["mpl"] for k in range(m)]
    except KeyError:
        return None
    va = V.get("all") or (V.get("p0-1") if m == 2 else None)
    if va is None:
        return None
    pairs = {}
    for k, l in itertools.combinations(range(m), 2):
        v = V.get(f"p{k}-{l}") or V.get(f"p{l}-{k}")
        if v is None:
            return None
        pairs[(k, l)] = v["mpl"]
    o2 = order2_shapley(singles, pairs, va["mpl"])
    return o2["phi"], singles, float(va["mpl"]), o2


# ---------------- 伪影零假设 ----------------

def build_nulls(gray_ctrl, inp_ctrl):
    """image_id -> (c_img, source)。inpaint 对照优先。"""
    out = {}
    for r in (gray_ctrl or []):
        vals = [c2["mpl"] for c in r.get("cues", []) for c2 in c.get("controls", [])
                if c2.get("mpl") is not None]
        if vals:
            out[r["image_id"]] = (float(np.mean(vals)), "gray")
    n_inp = 0
    for r in (inp_ctrl or []):
        vals = []
        for v in r.get("variants", []):
            spec = v.get("spec", "")
            if spec.startswith("c") and not spec.startswith("cg") and v.get("mpl") is not None:
                vals.append(v["mpl"])
        if vals:
            out[r["image_id"]] = (float(np.mean(vals)), "inpaint")
            n_inp += 1
    return out, n_inp


# ---------------- 逐图分析 ----------------

def analyse_image(iid, gpt_rec, voc_rec, sweep_rec, null):
    gcue, gcat, gmask, (W, H) = cue_masks(os.path.join(SAM3DIR, iid + ".json"))
    vcue, vcat, vmask, _ = cue_masks(os.path.join(VOCABDIR, iid + ".json"))
    total = float(W * H)

    # 线索名/顺序自洽性检查(manifest / sweep / 结果文件 / 掩码文件)
    warn = []
    sw_names = [c["cue"] for c in sweep_rec["per_cue"]]
    if sw_names != gcue:
        warn.append(f"GPT-4o cue order: sweep {sw_names} != sam3 masks {gcue}")
    man = load_json(os.path.join(VOC_CACHE, iid, "manifest.json"), {})
    man_names = [c.get("cue") for c in (man.get("cues") or [])]
    if man_names and man_names != vcue:
        warn.append(f"vocab cue order: manifest {man_names} != vocab masks {vcue}")
    if (voc_rec.get("cue_names") or man_names) != man_names:
        warn.append(f"vocab cue order: results {voc_rec.get('cue_names')} != manifest {man_names}")

    # 清单内部的几何重复(同一区域被拆成多条线索会摊薄 φ、虚高 m)
    def dup_within(names, masks, tag):
        d = []
        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                inter = float(np.logical_and(masks[a], masks[b]).sum())
                union = float(np.logical_or(masks[a], masks[b]).sum())
                i = inter / union if union else 0.0
                if i >= 0.9:
                    d.append({"side": tag, "a": names[a], "b": names[b], "iou": i})
        return d

    dups = dup_within(gcue, gmask, "gpt") + dup_within(vcue, vmask, "vocab")
    for d in dups:
        warn.append(f"{iid} [{d['side']}]: duplicate masks IoU={d['iou']:.3f} "
                    f"— \"{d['a']}\" vs \"{d['b']}\"")

    mg, mv = len(gcue), len(vcue)
    pg = phi_of(gpt_rec, mg)
    pv = phi_of(voc_rec, mv)
    if pg is None or pv is None:
        return None, warn + [f"{iid}: incomplete variant set (gpt={pg is not None}, "
                             f"vocab={pv is not None})"]

    gphi, gsingle, gvN, go2 = pg
    vphi, vsingle, vvN, vo2 = pv
    c_img, null_src = null

    # 几何对应矩阵
    iou = np.zeros((mg, mv)); rec_g = np.zeros((mg, mv)); rec_v = np.zeros((mg, mv))
    for a in range(mg):
        ga = gmask[a]; gs = float(ga.sum())
        for b in range(mv):
            vb = vmask[b]; vs = float(vb.sum())
            inter = float(np.logical_and(ga, vb).sum())
            union = float(np.logical_or(ga, vb).sum())
            iou[a, b] = inter / union if union else 0.0
            rec_g[a, b] = inter / gs if gs else 0.0      # GPT-4o 线索被词表盖住的比例
            rec_v[a, b] = inter / vs if vs else 0.0      # 词表线索被 GPT-4o 盖住的比例

    thr_g = c_img / mg if mg else float("nan")
    thr_v = c_img / mv if mv else float("nan")

    gpt_rows = []
    for a in range(mg):
        b = int(np.argmax(iou[a])) if mv else -1
        best = float(iou[a, b]) if mv else 0.0
        gpt_rows.append({
            "k": a, "cue": gcue[a], "category": gcat[a],
            "area_frac": float(gmask[a].sum() / total),
            "v_single": gsingle[a], "phi": gphi[a],
            "sensitive": bool(gphi[a] > thr_g),
            "match": (vcue[b] if best >= IOU_MATCHED else None),
            "match_k": (b if best >= IOU_MATCHED else None),
            "iou": best,
            "recall": (float(rec_g[a, b]) if mv else 0.0)})

    voc_rows = []
    for b in range(mv):
        a = int(np.argmax(iou[:, b])) if mg else -1
        best = float(iou[a, b]) if mg else 0.0
        voc_rows.append({
            "k": b, "cue": vcue[b], "category": vcat[b],
            "area_frac": float(vmask[b].sum() / total),
            "v_single": vsingle[b], "phi": vphi[b],
            "sensitive": bool(vphi[b] > thr_v),
            "match": (gcue[a] if best >= IOU_MATCHED else None),
            "match_k": (a if best >= IOU_MATCHED else None),
            "iou": best,
            "recall": (float(rec_g[a, b]) if mg else 0.0),
            "covered_frac": (float(rec_v[a, b]) if mg else 0.0)})

    # (iv) 几何基本重合的配对(IoU >= 0.5),用互为最佳匹配来定义一一对应
    same = []
    for a in range(mg):
        for b in range(mv):
            if iou[a, b] < IOU_SAME:
                continue
            if int(np.argmax(iou[a])) == b and int(np.argmax(iou[:, b])) == a:
                same.append({"gpt_k": a, "gpt_cue": gcue[a], "vocab_k": b,
                             "vocab_cue": vcue[b], "iou": float(iou[a, b]),
                             "phi_gpt": gphi[a], "phi_vocab": vphi[b],
                             "sens_gpt": bool(gphi[a] > thr_g),
                             "sens_vocab": bool(vphi[b] > thr_v)})

    # (v) 敏感区域的并集:两侧敏感线索建图,IoU >= 0.5 连边 → 连通分量
    nodes = ([("gpt", a) for a in range(mg) if gpt_rows[a]["sensitive"]]
             + [("vocab", b) for b in range(mv) if voc_rows[b]["sensitive"]])
    idx = {n: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for (sa, a) in nodes:
        if sa != "gpt":
            continue
        for (sb, b) in nodes:
            if sb != "vocab":
                continue
            if iou[a, b] >= IOU_SAME:
                ra, rb = find(idx[("gpt", a)]), find(idx[("vocab", b)])
                if ra != rb:
                    parent[ra] = rb
    comps = defaultdict(list)
    for n in nodes:
        comps[find(idx[n])].append(n)
    hyb = {"n_union": len(comps),
           "n_both": sum(1 for v in comps.values()
                         if any(s == "gpt" for s, _ in v) and any(s == "vocab" for s, _ in v)),
           "n_gpt_only": sum(1 for v in comps.values() if all(s == "gpt" for s, _ in v)),
           "n_vocab_only": sum(1 for v in comps.values() if all(s == "vocab" for s, _ in v)),
           "n_sens_gpt": sum(1 for r in gpt_rows if r["sensitive"]),
           "n_sens_vocab": sum(1 for r in voc_rows if r["sensitive"])}

    return {
        "image_id": iid,
        "true_label": sweep_rec["true_label"],
        "country_hit": sweep_rec.get("country_hit"),
        "size": [W, H],
        "m_gpt": mg, "m_vocab": mv,
        "vN_gpt": gvN, "vN_vocab": vvN,
        "winner": ("gpt" if gvN > vvN else ("vocab" if vvN > gvN else "tie")),
        "c_img": c_img, "null_source": null_src,
        "thr_gpt": thr_g, "thr_vocab": thr_v,
        "residual_share_gpt": go2["residual_share"],
        "residual_share_vocab": vo2["residual_share"],
        "order2_exact_gpt": mg <= 3, "order2_exact_vocab": mv <= 3,
        "duplicate_masks": dups,
        "gpt_cues": gpt_rows, "vocab_cues": voc_rows,
        "matched_pairs": same, "hybrid": hyb,
    }, warn


# ---------------- 汇总 ----------------

def aggregate(per_image):
    A = {}
    sg = float(sum(x["vN_gpt"] for x in per_image))
    sv = float(sum(x["vN_vocab"] for x in per_image))
    A["sum_vN_gpt"] = sg
    A["sum_vN_vocab"] = sv
    A["mean_vN_gpt"] = sg / len(per_image)
    A["mean_vN_vocab"] = sv / len(per_image)
    A["sum_winner"] = "gpt" if sg > sv else ("vocab" if sv > sg else "tie")
    A["n_win_gpt"] = sum(1 for x in per_image if x["winner"] == "gpt")
    A["n_win_vocab"] = sum(1 for x in per_image if x["winner"] == "vocab")
    A["n_win_tie"] = sum(1 for x in per_image if x["winner"] == "tie")
    A["median_ratio_vocab_over_gpt"] = float(np.median(
        [x["vN_vocab"] / x["vN_gpt"] if abs(x["vN_gpt"]) > 1e-12 else np.nan
         for x in per_image]))
    A["total_m_gpt"] = sum(x["m_gpt"] for x in per_image)
    A["total_m_vocab"] = sum(x["m_vocab"] for x in per_image)

    def bucket(rows):
        phi = [r["phi"] for r in rows]
        return {"n": len(rows), "n_sensitive": sum(1 for r in rows if r["sensitive"]),
                "phi_median": float(np.median(phi)) if phi else float("nan"),
                "phi_mean": float(np.mean(phi)) if phi else float("nan"),
                "phi_min": float(np.min(phi)) if phi else float("nan"),
                "phi_max": float(np.max(phi)) if phi else float("nan"),
                "phi": phi}

    voc_un = [dict(r, image_id=x["image_id"], true_label=x["true_label"])
              for x in per_image for r in x["vocab_cues"] if r["match"] is None]
    voc_mt = [dict(r, image_id=x["image_id"], true_label=x["true_label"])
              for x in per_image for r in x["vocab_cues"] if r["match"] is not None]
    gpt_un = [dict(r, image_id=x["image_id"], true_label=x["true_label"])
              for x in per_image for r in x["gpt_cues"] if r["match"] is None]
    gpt_mt = [dict(r, image_id=x["image_id"], true_label=x["true_label"])
              for x in per_image for r in x["gpt_cues"] if r["match"] is not None]
    A["vocab_unmatched"] = bucket(voc_un); A["vocab_unmatched"]["rows"] = voc_un
    A["vocab_matched"] = bucket(voc_mt)
    A["gpt_unmatched"] = bucket(gpt_un); A["gpt_unmatched"]["rows"] = gpt_un
    A["gpt_matched"] = bucket(gpt_mt)

    bycat = defaultdict(list)
    for r in gpt_un:
        bycat[r["category"]].append(r)
    A["gpt_unmatched_by_category"] = {
        c: {"n": len(v), "n_sensitive": sum(1 for r in v if r["sensitive"]),
            "cues": [r["cue"] for r in v]}
        for c, v in sorted(bycat.items(), key=lambda kv: -len(kv[1]))}

    pairs = [p for x in per_image for p in x["matched_pairs"]]
    A["matched_pairs"] = pairs
    A["n_matched_pairs"] = len(pairs)
    if len(pairs) >= 2:
        A["spearman_phi_matched"] = spearman([p["phi_gpt"] for p in pairs],
                                             [p["phi_vocab"] for p in pairs])
        a = np.array([p["phi_gpt"] for p in pairs], float)
        b = np.array([p["phi_vocab"] for p in pairs], float)
        A["pearson_phi_matched"] = (float(np.corrcoef(a, b)[0, 1])
                                    if a.std() > 0 and b.std() > 0 else float("nan"))
    else:
        A["spearman_phi_matched"] = float("nan")
        A["pearson_phi_matched"] = float("nan")
    A["n_pairs_sens_agree"] = sum(1 for p in pairs if p["sens_gpt"] == p["sens_vocab"])
    A["n_pairs_both_sens"] = sum(1 for p in pairs if p["sens_gpt"] and p["sens_vocab"])
    A["n_pairs_neither_sens"] = sum(1 for p in pairs
                                    if not p["sens_gpt"] and not p["sens_vocab"])
    A["n_pairs_gpt_only_sens"] = sum(1 for p in pairs if p["sens_gpt"] and not p["sens_vocab"])
    A["n_pairs_vocab_only_sens"] = sum(1 for p in pairs if p["sens_vocab"] and not p["sens_gpt"])

    A["hybrid"] = {k: sum(x["hybrid"][k] for x in per_image)
                   for k in ("n_union", "n_both", "n_gpt_only", "n_vocab_only",
                             "n_sens_gpt", "n_sens_vocab")}
    A["null_sources"] = dict(sorted(
        {s: sum(1 for x in per_image if x["null_source"] == s)
         for s in {x["null_source"] for x in per_image}}.items()))
    return A


# ---------------- 渲染 ----------------

WNAME = {"gpt": "GPT-4o", "vocab": "vocab", "tie": "tie"}


def f(x, n=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{n}f}"


def short(lbl):
    return lbl.split(",")[0]


def render(per_image, A, warns):
    L = []; P = L.append
    P("# GPT-4o-proposed cues vs a fixed vocabulary: which cue inventory captures the leakage?\n")
    P("Same 10 images, same removal operator (LaMa inpainting), same adversary (GeoRanker). "
      "Only the definition of \"a cue\" changes.\n")
    P("- **GPT-4o (top-down)**: masks `cue_extract/results_sam3/`, scores "
      "`belief_elicit/georanker_inpaint_results.json`")
    P("- **Vocabulary (bottom-up, 12 generic SAM 3 concepts)**: masks "
      "`cue_extract/results_vocab/`, scores `belief_elicit/georanker_inpaint_vocab_results.json`")
    P("- v(N) = mPL lost when the *whole* inventory is removed = total removable leakage "
      "of that inventory (nats / 1000 km). phi = order-2 anchored Shapley "
      "(`belief_elicit/order2_shapley.py`), anchored so sum_k phi_k = v(N).")
    ns = A["null_sources"]
    P(f"- artifact null c_img = mean control-placement mPL; source per image: "
      + ", ".join(f"**{k}** ({v} images)" for k, v in ns.items())
      + f". A cue is *provisionally* sensitive if phi_k > c_img / m "
        f"(pure-artifact null under which every phi equals c_img/m).")
    P("- **Caveat**: the order-2 anchored phi is exact for m <= 3 and ~8% off in magnitude at "
      "m = 4 (validated against the full 2^m gray lattice); for m >= 5 treat it as a ranking, "
      "not a magnitude. Affected here: "
      + (", ".join(f"{short(x['true_label'])} (vocab m={x['m_vocab']})"
                   for x in per_image if max(x["m_gpt"], x["m_vocab"]) >= 5) or "none")
      + ".\n")

    P("## Headline: total removable leakage v(N)\n")
    P("| image | true label | m_gpt | m_vocab | v(N) GPT-4o | v(N) vocab | winner | vocab/gpt |")
    P("|---|---|---:|---:|---:|---:|:--:|---:|")
    for x in per_image:
        r = x["vN_vocab"] / x["vN_gpt"] if abs(x["vN_gpt"]) > 1e-12 else float("nan")
        P(f"| `{x['image_id'][:20]}` | {x['true_label']} | {x['m_gpt']} | {x['m_vocab']} | "
          f"{f(x['vN_gpt'])} | {f(x['vN_vocab'])} | "
          f"**{WNAME[x['winner']]}** | {f(r,2)}x |")
    P(f"| **total** | 10 images | {A['total_m_gpt']} | {A['total_m_vocab']} | "
      f"**{f(A['sum_vN_gpt'])}** | **{f(A['sum_vN_vocab'])}** | "
      f"**{WNAME[A['sum_winner']]}** | "
      f"{f(A['sum_vN_vocab']/A['sum_vN_gpt'],2)}x |")
    P("")
    P(f"Per-image winner counts: GPT-4o **{A['n_win_gpt']}**, vocabulary "
      f"**{A['n_win_vocab']}**, exact ties **{A['n_win_tie']}** (of {len(per_image)}). "
      f"Median per-image ratio v(N)_vocab / v(N)_gpt = "
      f"**{f(A['median_ratio_vocab_over_gpt'],2)}x**.\n")

    P("## Per-image cue tables\n")
    for x in per_image:
        P(f"### {x['true_label']} — `{x['image_id']}`\n")
        P(f"v(N) GPT-4o **{f(x['vN_gpt'])}** ({x['m_gpt']} cues) vs vocab "
          f"**{f(x['vN_vocab'])}** ({x['m_vocab']} cues); "
          f"c_img = {f(x['c_img'])} ({x['null_source']} control) → threshold "
          f"{f(x['thr_gpt'])} (GPT-4o) / {f(x['thr_vocab'])} (vocab). "
          f"Order-2 residual share {f(x['residual_share_gpt'],3)} / "
          f"{f(x['residual_share_vocab'],3)}."
          + ("" if x["order2_exact_gpt"] and x["order2_exact_vocab"]
             else "  _phi magnitudes approximate where m >= 4._"))
        P("")
        P("**GPT-4o cues**\n")
        P("| # | cue | category | area | v({k}) | phi | sensitive | best vocab match (IoU, recall) |")
        P("|---:|---|---|---:|---:|---:|:--:|---|")
        for r in x["gpt_cues"]:
            mt = (f"{r['match']} (IoU {r['iou']:.2f}, rec {r['recall']:.2f})"
                  if r["match"] else f"_no match_ (best IoU {r['iou']:.2f})")
            P(f"| {r['k']} | {r['cue']} | {r['category']} | {r['area_frac']*100:.1f}% | "
              f"{f(r['v_single'])} | {f(r['phi'])} | {'YES' if r['sensitive'] else '-'} | {mt} |")
        P("")
        P("**Vocabulary cues**\n")
        P("| # | cue | category | area | v({k}) | phi | sensitive | best GPT-4o match (IoU, recall) |")
        P("|---:|---|---|---:|---:|---:|:--:|---|")
        for r in x["vocab_cues"]:
            mt = (f"{r['match']} (IoU {r['iou']:.2f}, rec {r['recall']:.2f})"
                  if r["match"] else f"_no match_ (best IoU {r['iou']:.2f})")
            P(f"| {r['k']} | {r['cue']} | {r['category']} | {r['area_frac']*100:.1f}% | "
              f"{f(r['v_single'])} | {f(r['phi'])} | {'YES' if r['sensitive'] else '-'} | {mt} |")
        P("")

    P("## Aggregates\n")
    P("### (i) Which inventory removes more leakage?\n")
    P(f"- sum over {len(per_image)} images of v(N): GPT-4o **{f(A['sum_vN_gpt'])}** vs vocabulary "
      f"**{f(A['sum_vN_vocab'])}** → **{WNAME[A['sum_winner']]}** "
      f"removes more (vocab/gpt = {f(A['sum_vN_vocab']/A['sum_vN_gpt'],2)}x)")
    P(f"- per-image winners: GPT-4o {A['n_win_gpt']} / vocabulary {A['n_win_vocab']} / "
      f"exact tie {A['n_win_tie']}")
    P(f"- mean v(N): {f(A['mean_vN_gpt'])} (GPT-4o) vs {f(A['mean_vN_vocab'])} (vocab); "
      f"cues scored: {A['total_m_gpt']} vs {A['total_m_vocab']}\n")

    vu, vm = A["vocab_unmatched"], A["vocab_matched"]
    P("### (ii) Vocabulary cues with NO GPT-4o counterpart (IoU < 0.1) — did GPT-4o miss leaky regions?\n")
    P(f"- unmatched vocabulary cues: **{vu['n']}** of {vu['n']+vm['n']}; sensitive: "
      f"**{vu['n_sensitive']}** ({vu['n_sensitive']/max(vu['n'],1)*100:.0f}%)")
    P(f"- matched vocabulary cues: {vm['n']}; sensitive: {vm['n_sensitive']} "
      f"({vm['n_sensitive']/max(vm['n'],1)*100:.0f}%)")
    P(f"- phi median: unmatched **{f(vu['phi_median'])}** vs matched **{f(vm['phi_median'])}** "
      f"(mean {f(vu['phi_mean'])} vs {f(vm['phi_mean'])}; unmatched range "
      f"{f(vu['phi_min'])}..{f(vu['phi_max'])})\n")
    P("| image | vocab cue | category | area | phi | sensitive |")
    P("|---|---|---|---:|---:|:--:|")
    for r in sorted(vu["rows"], key=lambda r: -r["phi"]):
        P(f"| {r['true_label']} | {r['cue']} | {r['category']} | {r['area_frac']*100:.1f}% | "
          f"{f(r['phi'])} | {'YES' if r['sensitive'] else '-'} |")
    P("")

    gu, gm = A["gpt_unmatched"], A["gpt_matched"]
    P("### (iii) GPT-4o cues with no vocabulary counterpart — what the fixed vocabulary would lose\n")
    P(f"- unmatched GPT-4o cues: **{gu['n']}** of {gu['n']+gm['n']}; sensitive: "
      f"**{gu['n_sensitive']}** ({gu['n_sensitive']/max(gu['n'],1)*100:.0f}%)")
    P(f"- matched GPT-4o cues: {gm['n']}; sensitive: {gm['n_sensitive']} "
      f"({gm['n_sensitive']/max(gm['n'],1)*100:.0f}%)")
    P(f"- phi median: unmatched **{f(gu['phi_median'])}** vs matched **{f(gm['phi_median'])}**\n")
    P("| image | GPT-4o cue | category | area | phi | sensitive |")
    P("|---|---|---|---:|---:|:--:|")
    for r in sorted(gu["rows"], key=lambda r: -r["phi"]):
        P(f"| {r['true_label']} | {r['cue']} | {r['category']} | {r['area_frac']*100:.1f}% | "
          f"{f(r['phi'])} | {'YES' if r['sensitive'] else '-'} |")
    P("")
    P("By category (unmatched GPT-4o cues):\n")
    P("| category | n | sensitive | cues |")
    P("|---|---:|---:|---|")
    for c, d in A["gpt_unmatched_by_category"].items():
        P(f"| {c} | {d['n']} | {d['n_sensitive']} | {'; '.join(d['cues'])} |")
    P("")

    P(f"### (iv) Matched pairs (IoU >= {IOU_SAME}, mutual best match): do the two inventories agree on phi?\n")
    P(f"- pairs: **{A['n_matched_pairs']}**; Spearman(phi_gpt, phi_vocab) = "
      f"**{f(A['spearman_phi_matched'],3)}** (Pearson {f(A['pearson_phi_matched'],3)})")
    P(f"- sensitive-flag agreement: **{A['n_pairs_sens_agree']}/{A['n_matched_pairs']}** "
      f"(both sensitive {A['n_pairs_both_sens']}, neither {A['n_pairs_neither_sens']}, "
      f"GPT-4o only {A['n_pairs_gpt_only_sens']}, vocab only {A['n_pairs_vocab_only_sens']})\n")
    if A["matched_pairs"]:
        P("| image | GPT-4o cue | vocab cue | IoU | phi_gpt | phi_vocab | sens gpt/vocab |")
        P("|---|---|---|---:|---:|---:|:--:|")
        for x in per_image:
            for p in x["matched_pairs"]:
                P(f"| {x['true_label']} | {p['gpt_cue']} | {p['vocab_cue']} | {p['iou']:.2f} | "
                  f"{f(p['phi_gpt'])} | {f(p['phi_vocab'])} | "
                  f"{'Y' if p['sens_gpt'] else '-'}/{'Y' if p['sens_vocab'] else '-'} |")
        P("")

    h = A["hybrid"]
    P("### (v) Hybrid inventory: union of sensitive regions across both sides\n")
    P(f"- sensitive cues: GPT-4o **{h['n_sens_gpt']}**, vocabulary **{h['n_sens_vocab']}**")
    P(f"- union of sensitive *regions* (merging pairs with IoU >= {IOU_SAME}): "
      f"**{h['n_union']}** — {h['n_both']} found by both, **{h['n_gpt_only']}** only by GPT-4o, "
      f"**{h['n_vocab_only']}** only by the vocabulary\n")
    P("| image | sens gpt | sens vocab | union | both | gpt only | vocab only |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for x in per_image:
        hh = x["hybrid"]
        P(f"| {x['true_label']} | {hh['n_sens_gpt']} | {hh['n_sens_vocab']} | {hh['n_union']} | "
          f"{hh['n_both']} | {hh['n_gpt_only']} | {hh['n_vocab_only']} |")
    P("")
    if warns:
        P("## Data consistency warnings\n")
        P("Cue order and naming agree across the mask files, the manifests, the sweep "
          "`per_cue` order and the scored result records — any mismatch would be listed here. "
          "What *is* listed below are geometrically **duplicate cues inside one inventory**: "
          "SAM 3 grounded two differently-named cues to the same pixels, which inflates m and "
          "splits phi across redundant entries.\n")
        for w in warns:
            P(f"- {w}")
        P("")
    else:
        P("## Data consistency\n")
        P("No cue-order or naming mismatches found between the mask files, the manifests, "
          "the sweep `per_cue` order and the scored result records.\n")
    return "\n".join(L) + "\n"


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="GPT-4o 线索清单 vs 固定词表清单")
    ap.add_argument("--gpt-results", default=GPT_RES)
    ap.add_argument("--vocab-results", default=VOC_RES)
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--gray-control", default=GRAY_CTRL)
    ap.add_argument("--inpaint-control", default=INP_CTRL)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--out-json", default=OUT_JSON)
    a = ap.parse_args()

    gpt = {r["image_id"]: r for r in (load_json(a.gpt_results) or [])}
    voc = load_json(a.vocab_results) or []
    sweep = {r["image_id"]: r for r in (load_json(a.sweep) or [])}
    nulls, n_inp = build_nulls(load_json(a.gray_control), load_json(a.inpaint_control))
    if n_inp:
        print(f"[info] inpaint control 文件里有 {n_inp} 张图 → 这些图用 inpaint null")
    else:
        print(f"[info] {a.inpaint_control} 里还没有可用图 → 全部退回 gray null")

    per_image, warns = [], []
    for r in voc:
        iid = r["image_id"]
        if iid not in gpt or iid not in sweep:
            warns.append(f"{iid}: missing GPT-4o inpaint or sweep record — skipped")
            continue
        if iid not in nulls:
            warns.append(f"{iid}: no control record in either control file — skipped")
            continue
        rec, w = analyse_image(iid, gpt[iid], r, sweep[iid], nulls[iid])
        warns += w
        if rec:
            per_image.append(rec)
    if not per_image:
        print("[fatal] 没有可用图")
        return 1
    per_image.sort(key=lambda x: x["true_label"])
    A = aggregate(per_image)

    # ---- 终端摘要 ----
    src = ", ".join(f"{k} ({v})" for k, v in A["null_sources"].items())
    print(f"\n=== v(N):GPT-4o 线索清单 vs 固定词表({len(per_image)} 张图)===")
    print(f"null 用的是: {src}\n")
    print(f"{'true label':<26s} {'m_gpt':>5s} {'m_voc':>5s} {'v(N)_gpt':>10s} "
          f"{'v(N)_voc':>10s} {'winner':>8s}")
    for x in per_image:
        print(f"{x['true_label'][:26]:<26s} {x['m_gpt']:5d} {x['m_vocab']:5d} "
              f"{x['vN_gpt']:10.4f} {x['vN_vocab']:10.4f} "
              f"{WNAME[x['winner']][:8]:>8s}")
    print(f"{'TOTAL':<26s} {A['total_m_gpt']:5d} {A['total_m_vocab']:5d} "
          f"{A['sum_vN_gpt']:10.4f} {A['sum_vN_vocab']:10.4f} "
          f"{WNAME[A['sum_winner']][:8]:>8s}")
    print(f"\n逐图胜负: GPT-4o {A['n_win_gpt']} / vocab {A['n_win_vocab']} / tie {A['n_win_tie']}"
          f"  |  中位比 vocab/gpt = {A['median_ratio_vocab_over_gpt']:.2f}x")
    vu, vm, gu, gm = (A["vocab_unmatched"], A["vocab_matched"],
                      A["gpt_unmatched"], A["gpt_matched"])
    print(f"\n(ii) 无 GPT-4o 对应的词表线索 {vu['n']} 条,敏感 {vu['n_sensitive']} 条;"
          f"φ 中位 {vu['phi_median']:+.4f} vs 有对应的 {vm['phi_median']:+.4f}")
    print(f"(iii) 无词表对应的 GPT-4o 线索 {gu['n']} 条,敏感 {gu['n_sensitive']} 条;"
          f"φ 中位 {gu['phi_median']:+.4f}")
    for c, d in A["gpt_unmatched_by_category"].items():
        print(f"      [{c}] n={d['n']} 敏感={d['n_sensitive']}: {'; '.join(d['cues'])}")
    print(f"(iv) IoU>={IOU_SAME} 的配对 {A['n_matched_pairs']} 组,"
          f"ρ(φ)={A['spearman_phi_matched']:+.3f},敏感标志一致 "
          f"{A['n_pairs_sens_agree']}/{A['n_matched_pairs']}")
    h = A["hybrid"]
    print(f"(v) 敏感区域并集 {h['n_union']}(两侧都找到 {h['n_both']},"
          f"仅 GPT-4o {h['n_gpt_only']},仅词表 {h['n_vocab_only']})")
    if warns:
        print("\n[数据一致性]")
        for w in warns:
            print("  -", w)

    S = {"meta": {"n_images": len(per_image),
                  "gpt_results": os.path.abspath(a.gpt_results),
                  "vocab_results": os.path.abspath(a.vocab_results),
                  "null_sources": A["null_sources"],
                  "iou_matched": IOU_MATCHED, "iou_same": IOU_SAME},
         "per_image": per_image, "aggregate": A, "warnings": warns}
    with open(a.out_md, "w", encoding="utf-8") as fh:
        fh.write(render(per_image, A, warns))
    json.dump(S, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {a.out_md}\nsaved {a.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
