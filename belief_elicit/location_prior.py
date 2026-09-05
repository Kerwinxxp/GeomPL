"""The attacker's content-free location prior pi — what GeoRanker believes with no evidence.

The signed risk metrics in `risk.py` need a *reference* distribution: the belief the
adversary holds before it sees any location evidence at all.  A uniform distribution is
the wrong reference, because the frozen adversary is not uniform on a blank input — the
prompt itself (candidate GPS + candidate label text) already induces a preference over
the 138 gallery labels.  We therefore elicit that preference empirically by scoring
three *content-free* images at the sweep's modal resolution:

  ``gray``   mid-gray (128) fill;
  ``noise``  the same mid-gray plus mild Gaussian noise (sigma = 8 gray levels), so the
             score is not an artefact of a perfectly constant image;
  ``blur``   one real sweep image Gaussian-blurred to featurelessness (sigma >= 60 px),
             which keeps the natural low-frequency colour statistics but destroys every
             recognisable cue.

`pi` is the per-label arithmetic mean of the three, renormalised.  The uniform
distribution is stored alongside so downstream code can switch reference with one flag.

GPU runner (GeoRanker venv):

    belief_elicit/.venv_gr/Scripts/python.exe -m belief_elicit.location_prior
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageFilter

from belief_elicit.results import LOCATION_PRIOR as OUT, image_path, load_gallery, load_subsets

#: modal (W, H) of the sweep's masked inputs (SAM 3 mask frame)
SIZE = (1036, 756)
#: real image used for the blurred-to-featureless probe
BLUR_IMAGE = "483717602_2445d0d4ae_206_40425693@N00.jpg"
BLUR_SIGMA = 60.0
NOISE_SIGMA = 8.0


def entropy(dist):
    """Shannon entropy of a {label: p} distribution, in nats."""
    return float(-sum(p * math.log(p) for p in dist.values() if p > 0))


def probe_images(size=SIZE, blur_image=BLUR_IMAGE, seed=0):
    """-> {name: PIL.Image}: the three content-free probes at `size`."""
    W, H = size
    gray = Image.new("RGB", (W, H), (128, 128, 128))

    rng = np.random.default_rng(seed)
    arr = np.full((H, W, 3), 128.0)
    arr += rng.normal(0.0, NOISE_SIGMA, arr.shape)
    noise = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    subset = load_subsets()
    blur = (Image.open(image_path(subset[blur_image])).convert("RGB").resize((W, H))
            .filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA)))
    return {"gray": gray, "noise": noise, "blur": blur}


def average(dists):
    """Per-label arithmetic mean of several {label: p} distributions, renormalised."""
    labels = sorted(dists[0])
    avg = {lab: float(np.mean([d.get(lab, 0.0) for d in dists])) for lab in labels}
    z = sum(avg.values())
    return {lab: p / z for lab, p in avg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from belief_elicit.georanker_belief import score_labels

    gv = load_gallery()
    probes = probe_images()
    dists = {}
    for name, img in probes.items():
        d, _ = score_labels(img, gv, variant=args.variant, batch_size=args.batch_size)
        dists[name] = d
        top = max(d, key=d.get)
        print(f"[{name:5s}] top={top!r} p={d[top]:.4f}  H={entropy(d):.4f} nats", flush=True)

    pi = average([dists[k] for k in ("gray", "noise", "blur")])
    uniform = {lab: 1.0 / len(gv) for lab in sorted(pi)}
    top = max(pi, key=pi.get)
    H_pi, H_u = entropy(pi), entropy(uniform)

    out = {"size": list(SIZE), "variant": args.variant,
           "blur_image": BLUR_IMAGE, "blur_sigma": BLUR_SIGMA, "noise_sigma": NOISE_SIGMA,
           "n_labels": len(gv),
           "distributions": dists,
           "pi": pi, "uniform": uniform,
           "entropy": {k: entropy(v) for k, v in dists.items()} |
                      {"pi": H_pi, "uniform": H_u},
           "top_label": top, "top_p": pi[top],
           "entropy_ratio": H_pi / H_u}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\npi: top={top!r} p={pi[top]:.4f}  H(pi)={H_pi:.4f} vs H(uniform)={H_u:.4f} "
          f"({H_pi / H_u * 100:.1f}% of uniform)")
    print("saved", args.out)


if __name__ == "__main__":
    main()
