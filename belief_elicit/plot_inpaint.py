"""【实验性 · 可整体删除】修复(inpaint)vs 灰块 四联图。

(a) 逐线索 v({k}):灰块 vs 修复(y=x 参考线)
(b) 逐类别配对条形:灰块精确 φ vs 修复二阶锚定 φ
(c) 成对交互 d_kl 直方图:修复 vs 灰块
(d) 伪影地板直方图:优先同一放置的灰块 cg* vs 修复 c*(配对);没有 cg* 时回退到更早的
    独立灰块对照跑(放置位置不同 → 非配对分布对照,图上明确标注)

数据来自 inpaint_report.py 落盘的 belief_elicit/<prefix>inpaint_summary.json(先跑它)。
运行:python -m belief_elicit.plot_inpaint [--out-prefix vocab_] [--summary ...] [--out ...]
"""
import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "inpaint_summary.json")
OUT = os.path.join(HERE, "figures", "inpaint_vs_gray.png")


def default_paths(prefix=""):
    """--out-prefix -> (summary json, figure png)。prefix="" 时与历史默认完全一致。"""
    return (os.path.join(HERE, f"{prefix}inpaint_summary.json"),
            os.path.join(HERE, "figures", f"{prefix}inpaint_vs_gray.png"))
BLUE, RED, GRAY, GREEN = "#1E88E5", "#E53935", "#B0BEC5", "#43A047"


def empty(ax, title, msg):
    """数据还不够时的占位格:虚线框 + 说明,保持四联版式不塌。"""
    ax.set_title(title, fontsize=10.5)
    ax.text(0.5, 0.5, msg + "\n\n(partial run — rerun when scoring finishes)",
            ha="center", va="center", fontsize=9.5, color="#777",
            transform=ax.transAxes, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.9", facecolor="#FAFAFA",
                      edgecolor=GRAY, linestyle="--", linewidth=1))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------- (a) 单条散点 ----------------

def panel_a(ax, S):
    a = S["a"]
    rows = [x for x in a.get("scatter", []) if x.get("gray") is not None]
    if len(rows) < 2:
        n = len(a.get("scatter", []))
        empty(ax, "(a) Single-cue removal: gray vs inpaint",
              f"no gray-paired cues yet\n({n} inpainted single-cue scores so far)")
        return
    g = np.array([x["gray"] for x in rows], float)
    i = np.array([x["inpaint"] for x in rows], float)
    area = np.array([(x.get("coverage") or x.get("area_frac") or 0.05) for x in rows], float)
    below = i < g
    lim = float(max(g.max(), i.max())) * 1.06
    ax.plot([0, lim], [0, lim], color="#999", ls="--", lw=1, zorder=2,
            label="inpaint = gray")
    for msk, c, lab in [(below, BLUE, "inpaint < gray"), (~below, RED, "inpaint ≥ gray")]:
        if msk.sum():
            ax.scatter(g[msk], i[msk], s=18 + area[msk] * 240, c=c, alpha=0.6,
                       edgecolors="white", linewidths=0.5, zorder=3,
                       label=f"{lab}  ({int(msk.sum())})")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("gray-fill  v({k})   (nats / 1000 km)")
    ax.set_ylabel("inpaint  v({k})")
    ax.set_title(f"(a) Single-cue removal, {len(rows)} cues / {a['n_images']} images\n"
                 f"ρ = {a.get('spearman_gray_inpaint', float('nan')):+.3f}, "
                 f"inpaint < gray in {a.get('frac_inpaint_lt_gray', float('nan'))*100:.0f}%"
                 "   (point size ∝ mask area)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(color="#EEE", zorder=0); ax.set_axisbelow(True)


# ---------------- (b) 逐类别 φ 配对条形 ----------------

def panel_b(ax, S):
    b = S["b"]
    cats = b.get("per_category") or {}
    have_gray = any(d.get("phi_gray_median") is not None for d in cats.values())
    rows = [(c, d) for c, d in cats.items() if d["n"] >= 1]
    if not rows:
        empty(ax, "(b) Per-category Shapley φ: gray vs inpaint",
              "no image has the full singles + pairs + all set yet")
        return
    rows.sort(key=lambda kv: -(kv[1]["phi_inpaint_median"]
                               if np.isfinite(kv[1]["phi_inpaint_median"]) else -9))
    names = [c for c, _ in rows]
    y = np.arange(len(names))[::-1]
    h = 0.38
    pin = [d["phi_inpaint_median"] for _, d in rows]
    pgr = [(d["phi_gray_median"] if d["phi_gray_median"] is not None else np.nan)
           for _, d in rows]
    if have_gray:
        ax.barh(y + h / 2, pgr, h, color=GRAY, zorder=3, label="gray exact φ (median)")
    ax.barh(y - h / 2 if have_gray else y, pin, h if have_gray else h * 1.6,
            color=BLUE, zorder=3, label="inpaint order-2 φ (median)")
    xmax = float(np.nanmax([np.nanmax(pin), (np.nanmax(pgr) if have_gray else 0)]))
    xmin = float(min(0.0, np.nanmin(pin)))
    pad = max(xmax, abs(xmin)) * 0.02

    def label(v, yi, color):
        if v is None or not np.isfinite(v):
            return
        if v < 0:                       # 负值:标签放进条内,避免撞到 y 轴刻度文字
            ax.text(v + pad * 0.5, yi, f"{v:.3f}", va="center", ha="left",
                    fontsize=8, color="white", zorder=5)
        else:
            ax.text(v + pad, yi, f"{v:.3f}", va="center", ha="left",
                    fontsize=8, color=color)

    for yi, (_, d) in zip(y, rows):
        label(d["phi_inpaint_median"], yi - (h / 2 if have_gray else 0), "#1565C0")
        if have_gray:
            label(d["phi_gray_median"], yi + h / 2, "#555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={d['n']})" for c, d in rows], fontsize=8.5)
    ax.set_ylim(-0.65, len(rows) - 0.35)      # 只有 1-2 个类别时防止条形被拉成巨块
    ax.set_xlabel("median φ (nats / 1000 km)")
    ax.set_xlim(xmin * 1.12 if xmin < 0 else 0, xmax * 1.32)
    ax.set_title(f"(b) Per-category attribution, {b['n_images']} usable images\n"
                 + (f"global ρ(inpaint, gray) = {b.get('spearman_global', float('nan')):+.3f}, "
                    f"top-1 agreement {b.get('top1_agreement', float('nan'))*100:.0f}%"
                    if have_gray else "inpaint only (no gray baseline)"), fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", color="#EEE", zorder=0); ax.set_axisbelow(True)


# ---------------- (c) 交互直方图 ----------------

def panel_c(ax, S):
    c = S["c"]
    inp = np.array([x["d_kl"] for x in c.get("interactions", [])], float)
    if len(inp) < 2:
        empty(ax, "(c) Pairwise interaction d(k,l)",
              f"only {len(inp)} inpainted cue pair(s) scored so far")
        return
    gs = c.get("gray_sii_signs_same_images") or {}
    gray_vals = S.get("_gray_sii_values")
    lo = float(inp.min()); hi = float(inp.max())
    if gray_vals is not None and len(gray_vals):
        lo = min(lo, float(np.min(gray_vals))); hi = max(hi, float(np.max(gray_vals)))
    pad = max(1e-3, (hi - lo) * 0.05)
    bins = np.linspace(lo - pad, hi + pad, 26)
    if gray_vals is not None and len(gray_vals):
        ax.hist(gray_vals, bins=bins, color=GRAY, alpha=0.85, zorder=2,
                label=f"gray SII  (n={len(gray_vals)}, med {np.median(gray_vals):+.3f})")
    ax.hist(inp, bins=bins, color=BLUE, alpha=0.6, zorder=3,
            label=f"inpaint d(k,l)  (n={len(inp)}, med {np.median(inp):+.3f})")
    ax.axvline(0, color="#666", lw=1, ls="--", zorder=4)
    sg = c.get("inpaint_interaction_signs", {})
    ax.set_xlabel("interaction value (nats / 1000 km)")
    ax.set_ylabel("cue pairs")
    ax.set_title("(c) Pairwise interaction: redundancy (<0) vs backup (>0)\n"
                 f"inpaint: {sg.get('neg',0)} redundant / {sg.get('zero',0)} ≈0 / "
                 f"{sg.get('pos',0)} synergistic" +
                 (f"  |  gray: {gs.get('neg',0)}/{gs.get('zero',0)}/{gs.get('pos',0)}"
                  if gs.get("n") else ""), fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#EEE", zorder=0); ax.set_axisbelow(True)


# ---------------- (d) 伪影地板 ----------------

def panel_d(ax, S):
    """伪影地板。优先同位置配对 cg*;没有则回退到更早独立灰块跑的非配对分布对照。"""
    d = S["d"]
    if not d.get("available"):
        empty(ax, "(d) Artifact floor on control placements",
              "control run not available yet\n(georanker_inpaint_control_results.json)")
        return
    pl = d.get("placements") or []
    paired = len(pl) >= 2
    if paired:                                   # 同位置灰块孪生存在 → 配对口径
        gi = np.array([x["gray"] for x in pl], float)
        ii = np.array([x["inpaint"] for x in pl], float)
        icol, ilab = GREEN, "inpaint control c*"
        glab = "gray control cg*  (same placements)"
        head = f"(d) Artifact floor on identical placements, {d['n_images']} images"
    else:                                        # 回退:非配对分布对照
        gi = np.array(d.get("control_values_gray") or [], float)
        ii = np.array(d.get("control_values_inpaint") or [], float)
        icol, ilab = BLUE, "inpaint control c*"
        glab = "gray: earlier run, different placements"
        head = (f"(d) Artifact floor — UNPAIRED, {d['n_images']} images / "
                f"{d['n_cues']} cues")
    if len(ii) < 2 or len(gi) < 2:
        empty(ax, "(d) Artifact floor on control placements",
              f"not enough control placements yet\n"
              f"({len(ii)} inpaint / {len(gi)} gray)")
        return

    mg, mi = float(np.median(gi)), float(np.median(ii))
    pg, pi_ = float(np.percentile(gi, 90)), float(np.percentile(ii, 90))
    bins = np.linspace(0.0, float(max(gi.max(), ii.max())) * 1.05, 31)
    ax.hist(gi, bins=bins, color=GRAY, alpha=0.85, zorder=2,
            label=f"{glab}\n     n={len(gi)}, med {mg:.3f}, P90 {pg:.3f}")
    ax.hist(ii, bins=bins, color=icol, alpha=0.6, zorder=3,
            label=f"{ilab}  (n={len(ii)}, med {mi:.3f}, P90 {pi_:.3f})")
    ax.axvline(mg, color="#607D8B", lw=1.4, ls="--", zorder=5)
    ax.axvline(mi, color=icol, lw=1.4, ls="--", zorder=5)
    ax.axvline(pg, color="#607D8B", lw=1.1, ls=":", zorder=5)
    ax.axvline(pi_, color=icol, lw=1.1, ls=":", zorder=5)
    ymax = ax.get_ylim()[1]
    for x, c, t in [(mg, "#546E7A", "med"), (pg, "#546E7A", "P90"),
                    (mi, icol, "med"), (pi_, icol, "P90")]:
        ax.text(x, ymax * (0.985 if t == "med" else 0.90), t, fontsize=7.5,
                color=c, ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.75))

    ri = d.get("resolvable_inpaint") or {}
    rg = d.get("resolvable_gray") or {}
    sub = f"floor median: gray {mg:.3f} vs inpaint {mi:.3f}   (P90 {pg:.3f} vs {pi_:.3f})"
    if ri or rg:
        sub += ("\nresolvable (real > max own controls): gray "
                f"{rg.get('rate', float('nan'))*100:.0f}% (n={rg.get('n',0)})"
                f"  vs  inpaint {ri.get('rate', float('nan'))*100:.0f}% "
                f"(n={ri.get('n',0)})")
    if paired:
        p = d.get("paired_placements", {})
        sub += (f"\ninpaint < gray on {p.get('frac_inpaint_lt_gray', float('nan'))*100:.0f}% "
                f"of {len(pl)} identical placements")
    else:
        sub += "\ndistribution-level comparison only — placements are not matched"
    ax.set_xlabel("control-placement mPL (artifact floor)")
    ax.set_ylabel("placements")
    ax.set_title(f"{head}\n{sub}", fontsize=10.5)
    ax.legend(fontsize=7.8, loc="upper right")
    ax.grid(axis="y", color="#EEE", zorder=0); ax.set_axisbelow(True)


def gray_sii_values(S, shap_path):
    """从 shapley_v2 结果里取出与 (c) 节同一批图的 SII 原始值(直方图用)。"""
    if not os.path.exists(shap_path) or not S["meta"].get("use_gray"):
        return None
    try:
        shap = json.load(open(shap_path, encoding="utf-8"))
    except Exception:
        return None
    seen = {x["image_id"] for x in S["c"].get("per_image", [])}
    vals = [v for r in shap if r["image_id"] in seen
            for v in (r.get("sii") or {}).values()]
    return np.array(vals, float) if vals else None


def main():
    ap = argparse.ArgumentParser(description="inpaint vs gray 四联图")
    ap.add_argument("--out-prefix", default="",
                    help="输入/输出文件名前缀(默认空 = inpaint_summary.json / "
                         "figures/inpaint_vs_gray.png;词表口径用 vocab_)")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--shapley", default=os.path.join(HERE, "shapley_v2_results.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d_sum, d_out = default_paths(a.out_prefix)
    if a.summary is None:
        a.summary = d_sum
    if a.out is None:
        a.out = d_out

    if not os.path.exists(a.summary):
        print(f"[fatal] 找不到 {a.summary};先跑 python -m belief_elicit.inpaint_report")
        return 1
    S = json.load(open(a.summary, encoding="utf-8"))
    S["_gray_sii_values"] = gray_sii_values(S, a.shapley)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11))
    panel_a(axes[0, 0], S)
    panel_b(axes[0, 1], S)
    panel_c(axes[1, 0], S)
    panel_d(axes[1, 1], S)

    m = S["meta"]
    fig.suptitle("Inpainting vs gray-fill masking: per-cue location leakage "
                 f"({m['n_records']} images, {m['n_variants']} scored variants"
                 + ("" if m["use_gray"] else ", vocabulary run — no gray baseline") + ")",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", dpi=130)
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
