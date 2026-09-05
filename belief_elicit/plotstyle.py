"""Shared figure style: the palette, the rcParams every figure sets, and a save helper.

`apply_style()` reproduces the rcParams block each plot script used to carry; pass
`hide_spines=False` for image-panel figures that keep their frames, and `extra=` for a
script that layers more on top.  `save()` centralises makedirs + savefig + the "saved"
line; the per-figure dpi / bbox conventions stay with the caller.

Library module — no CLI.
"""
import os

import numpy as np

# ---- palette ----
BLUE = "#1E88E5"
RED = "#E53935"
GREEN = "#43A047"
ORANGE = "#FB8C00"
PURPLE = "#8E24AA"
CYAN = "#00ACC1"
GRAY = "#B0BEC5"
GRAY_LIGHT = "#C4CDD5"
INK = "#212121"

#: colour cycle for per-cue overlays and bars
CUE_COLORS = [RED, BLUE, GREEN, ORANGE, PURPLE, CYAN]

RC_FONT = {"font.family": "DejaVu Sans"}
RC_SPINES = {"axes.spines.top": False, "axes.spines.right": False}


def apply_style(hide_spines=True, extra=None):
    """Set the shared rcParams. Import matplotlib.pyplot yourself; this only updates rcParams."""
    import matplotlib.pyplot as plt
    rc = dict(RC_FONT)
    if hide_spines:
        rc.update(RC_SPINES)
    if extra:
        rc.update(extra)
    plt.rcParams.update(rc)


def save(fig, path, dpi=130, bbox_inches="tight", close=False, **kw):
    """makedirs + savefig + print("saved", path); returns the path."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kw)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    print("saved", path)
    return path


# ---- text helpers shared by the reports and the figures ----

def short(lbl, maxlen=None):
    """First comma-separated field of a gallery label, ellipsised at `maxlen` if given."""
    a = lbl.split(",")[0]
    if maxlen is not None and len(a) > maxlen:
        return a[:maxlen - 1] + "…"
    return a


def fmt(x, n=4):
    """Fixed-point float for report tables; "n/a" for None / nan / inf."""
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"
