"""Equal-area control figure.

(a) per-cue paired scatter, real vs control (same mask shape and area);
(b) the control distribution against the real-cue distribution.

Run: python -m belief_elicit.plot_control_v2
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from belief_elicit.plotstyle import BLUE, GRAY, GREEN, RED, apply_style, save
from belief_elicit.results import CONTROLS as DATA, FIGDIR, load_controls

apply_style()
OUT = os.path.join(FIGDIR, "georanker_control.png")


def main():
    R = load_controls(DATA)
    rows = []
    for r in R:
        for c in r["cues"]:
            if not c["controls"]:
                continue
            cm = [x["mpl"] for x in c["controls"]]
            rows.append(dict(place=r["true_label"].split(",")[0], cue=c["cue"],
                             real=c["real_mpl"], cmean=float(np.mean(cm)),
                             cmax=float(np.max(cm)), ctrls=cm))
    real = np.array([x["real"] for x in rows])
    cmean = np.array([x["cmean"] for x in rows])
    cmax = np.array([x["cmax"] for x in rows])
    resolv = real > cmax
    ctrl_all = np.array([v for x in rows for v in x["ctrls"]])

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # (a) paired scatter
    lim = max(real.max(), cmax.max()) * 1.12
    ax[0].plot([0, lim], [0, lim], color="#888", lw=1.2, ls="--", label="y = x")
    ax[0].plot([0, lim / 2], [0, lim], color="#BBB", lw=1.0, ls=":", label="y = 2x")
    ax[0].errorbar(cmean[resolv], real[resolv], xerr=(cmax - cmean)[resolv],
                   fmt="o", color=GREEN, ms=7, capsize=2.5, lw=1,
                   label=f"resolvable: real > max(ctrl)  ({resolv.sum()})")
    ax[0].errorbar(cmean[~resolv], real[~resolv], xerr=(cmax - cmean)[~resolv],
                   fmt="o", color=RED, ms=7, capsize=2.5, lw=1,
                   label=f"NOT resolvable  ({(~resolv).sum()})")
    for x in rows:
        if x["real"] / max(x["cmean"], 1e-9) > 3.4:
            ax[0].annotate(f"{x['place']}: {x['cue'][:16]}",
                           (x["cmean"], x["real"]), textcoords="offset points",
                           xytext=(6, 3), fontsize=7.5, color="#333")
    ax[0].set_xlabel("equal-area control mPL (mean over placements; bar → max)")
    ax[0].set_ylabel("real-cue mPL")
    ax[0].set_title("(a) Paired: real cue vs equal-area control\n"
                    "(same mask shape/size, translated to non-cue region)", fontsize=11)
    ax[0].legend(fontsize=8.5, loc="lower right")
    ax[0].set_xlim(0, lim * 0.62); ax[0].set_ylim(0, lim)
    ax[0].grid(color="#EEE", zorder=0); ax[0].set_axisbelow(True)

    # (b) distribution comparison
    bins = np.linspace(0, max(ctrl_all.max(), real.max()) * 1.05, 25)
    ax[1].hist(ctrl_all, bins=bins, color=GRAY, alpha=0.85, zorder=3,
               label=f"control placements (n={len(ctrl_all)})")
    ax[1].hist(real, bins=bins, color=BLUE, alpha=0.65, zorder=4,
               label=f"real cues (n={len(real)})")
    ax[1].axvline(np.median(ctrl_all), color="#555", lw=1.4, ls="--",
                  label=f"control median {np.median(ctrl_all):.3f}")
    ax[1].axvline(np.percentile(ctrl_all, 90), color=RED, lw=1.4, ls="--",
                  label=f"control P90 {np.percentile(ctrl_all,90):.3f}")
    ax[1].set_xlabel("mPL (nats/1000 km)")
    ax[1].set_ylabel("count")
    ax[1].set_title("(b) Masking-artifact floor vs real-cue distribution", fontsize=11)
    ax[1].legend(fontsize=8.5)
    ax[1].grid(axis="y", color="#EEE", zorder=0); ax[1].set_axisbelow(True)

    fig.suptitle(f"Equal-area control under the GeoRanker belief meter "
                 f"({len(R)} images, {len(rows)} cues, {len(ctrl_all)} control placements)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT, dpi=130)


if __name__ == "__main__":
    main()
