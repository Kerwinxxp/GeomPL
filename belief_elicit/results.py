"""Locations of, and loaders for, every result / data file the analysis layer reads.

One place that knows where `georanker_*_results.json`, the Shapley outputs, the dedup
groups, the inpaint caches, the gallery and the image manifests live, plus the small
readers (`load_sweep`, `load_lattice`, `build_v`, ...) that every report and plot script
used to re-implement.

Library module — no CLI.
"""
import itertools
import json
import os

from cue_extract.common import load_subsets      # noqa: F401  (re-exported loader)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figures")

# ---- GPU scoring runs ----
SWEEP = os.path.join(HERE, "georanker_sweep_results.json")
LATTICE = os.path.join(HERE, "georanker_lattice_results.json")
CONTROLS = os.path.join(HERE, "georanker_control_results.json")
GRAY_CONTROLS = CONTROLS                      # alias: the gray-block equal-area control
INPAINT = os.path.join(HERE, "georanker_inpaint_results.json")
INPAINT_CONTROLS = os.path.join(HERE, "georanker_inpaint_control_results.json")
INPAINT_VOCAB = os.path.join(HERE, "georanker_inpaint_vocab_results.json")

# ---- CPU analyses ----
SHAPLEY_V2 = os.path.join(HERE, "shapley_v2_results.json")
SHAPLEY_V3 = os.path.join(HERE, "shapley_v3_results.json")
DEDUP_GROUPS = os.path.join(HERE, "cue_dedup_groups.json")
ALT_ATTRIBUTION = os.path.join(HERE, "alt_attribution_results.json")
CALIBRATE_TAU = os.path.join(HERE, "calibrate_tau_results.json")
ORDER2_VALIDATION = os.path.join(HERE, "order2_shapley_validation.json")
LOCATION_PRIOR = os.path.join(HERE, "location_prior.json")
PROTECTION_SET = os.path.join(HERE, "protection_set_results.json")

# ---- generated reports ----
DEDUP_REPORT_MD = os.path.join(HERE, "dedup_report.md")
DEDUP_REPORT_JSON = os.path.join(HERE, "dedup_report.json")
INPAINT_REPORT_MD = os.path.join(HERE, "inpaint_report.md")
INPAINT_SUMMARY = os.path.join(HERE, "inpaint_summary.json")
VOCAB_REPORT_MD = os.path.join(HERE, "vocab_vs_gpt4o.md")
VOCAB_REPORT_JSON = os.path.join(HERE, "vocab_vs_gpt4o.json")
PROTECTION_SET_MD = os.path.join(HERE, "protection_set_report.md")

# ---- pixel caches and inputs ----
INPAINT_CACHE = os.path.join(HERE, "inpaint_cache")
INPAINT_CACHE_VOCAB = os.path.join(HERE, "inpaint_cache_vocab")
SAM3_DIR = os.path.join(ROOT, "cue_extract", "results_sam3")
VOCAB_DIR = os.path.join(ROOT, "cue_extract", "results_vocab")
GALLERY = os.path.join(ROOT, "data", "gallery_v2.json")


# ---------------- generic loaders ----------------

def load_json(p, default=None):
    """Read a JSON file; return `default` when it is missing or unreadable."""
    if not p or not os.path.exists(p):
        return default
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"[warn] cannot read {p}: {e}", flush=True)
        return default


def load_sweep(path=SWEEP):
    """The main sweep, in file order (posterior + per-cue priors + all-masked)."""
    return json.load(open(path, encoding="utf-8"))


def by_image(records):
    """[record, ...] -> {image_id: record}."""
    return {r["image_id"]: r for r in records}


def load_lattice(path=LATTICE):
    """Intermediate subset scores keyed by image_id; {} when the file is absent."""
    return by_image(json.load(open(path, encoding="utf-8"))) if os.path.exists(path) else {}


def load_controls(path=CONTROLS):
    """Equal-area gray-block control placements, in file order."""
    return json.load(open(path, encoding="utf-8"))


def load_inpaint(path):
    """Inpaint scoring results keyed by image_id; {} when the file is absent."""
    return by_image(json.load(open(path, encoding="utf-8"))) if os.path.exists(path) else {}


def load_shapley(path):
    """A shapley_v2 / shapley_v3 result file, in file order."""
    return json.load(open(path, encoding="utf-8"))


def load_gallery(path=GALLERY):
    """The candidate gallery, restricted to entries that carry GPS."""
    return [g for g in json.load(open(path, encoding="utf-8")) if g["gps"]]


# ---------------- image manifests ----------------

def image_path(record):
    """A `data/subset*.jsonl` record -> absolute path of its source image."""
    p = record["path"]
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_manifest(image_dir):
    """Read `<image_dir>/manifest.json` (written last by precompute_inpaint); None if absent."""
    return load_json(os.path.join(image_dir, "manifest.json"))


def manifest_index(image_dir):
    """Read a cache manifest and normalise it to {filename: entry dict}; {} on failure.

    Tolerates the several shapes a manifest has had: a bare list, a dict with a
    `files`/`variants`/`entries`/`items` list or mapping, or a top-level filename->entry map.
    """
    data = load_manifest(image_dir)
    if data is None:
        return {}
    entries = None
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        for key in ("files", "variants", "entries", "items"):
            if isinstance(data.get(key), list):
                entries = data[key]
                break
            if isinstance(data.get(key), dict):       # {"s0.png": {...}} shape
                entries = [dict(v, file=k) for k, v in data[key].items()]
                break
        if entries is None:                            # top-level filename keys
            cand = [(k, v) for k, v in data.items() if isinstance(v, dict) and "." in k]
            entries = [dict(v, file=k) for k, v in cand]
    out = {}
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        fn = None
        for key in ("file", "filename", "name", "path", "png"):
            if isinstance(e.get(key), str):
                fn = os.path.basename(e[key])
                break
        if fn is None and isinstance(e.get("spec"), str):
            fn = e["spec"] + ".png"
        if fn:
            out[fn] = e
    return out


def index_variants(rec):
    """One scoring record -> {spec: variant record}."""
    return {v["spec"]: v for v in rec.get("variants", [])}


# ---------------- the set function v(S) ----------------

def build_v(r, lattice):
    """One sweep record (+ the lattice) -> (v, complete?) over cue-index frozensets.

    v(empty) = 0, v({k}) from the per-cue single-cue mPL, v(N) from `mpl_all`, and for
    m >= 3 the intermediate subsets from the lattice run.  `complete` is False when the
    lattice for that image is not filled in yet.
    """
    m = r["n_cues"]
    v = {frozenset(): 0.0}
    for k, pc in enumerate(r["per_cue"]):
        v[frozenset([k])] = pc["mpl"]
    if m == 1:
        return v, True
    v[frozenset(range(m))] = r["mpl_all"]
    if m >= 3:
        lat = lattice.get(r["image_id"])
        if not lat or len(lat["combos"]) < 2 ** m - 2 - m:
            return v, False
        for c in lat["combos"]:
            v[frozenset(c["subset"])] = c["mpl"]
    return v, True


def build_q(r, lattice):
    """One sweep record (+ the lattice) -> {frozenset of cue indices: belief dict}.

    The *belief* counterpart of `build_v`: instead of the scalar mPL of masking S, this
    returns the full posterior over gallery labels that the adversary holds when S is
    gray-filled.  q(empty) is the clean-image posterior, q({k}) the per-cue prior, q(N)
    the all-masked prior, and the intermediate subsets come from the lattice run.

    The dict is complete iff ``len(q) == 2 ** r["n_cues"]``; callers that need the whole
    lattice must check that, because the lattice run only covers m >= 3 images and may be
    unfinished for some of them.
    """
    m = r["n_cues"]
    q = {frozenset(): r["posterior"]}
    for k, pc in enumerate(r["per_cue"]):
        q[frozenset([k])] = pc["prior"]
    q[frozenset(range(m))] = r["prior_allmask"]
    if m >= 3:
        lat = lattice.get(r["image_id"])
        for c in (lat or {}).get("combos", []):
            q[frozenset(c["subset"])] = c["prior"]
    return q


def merged_v(v, groups):
    """Raw v (keyed by original cue indices) -> merged game v' (keyed by group indices).

    A merged player stands for the union of its members' masks, so v'(S') is read off the
    existing lattice at v(union of members).  Returns None if any needed subset is absent.
    """
    M = len(groups)
    vm = {}
    for size in range(M + 1):
        for S in itertools.combinations(range(M), size):
            orig = frozenset(k for gi in S for k in groups[gi]["members"])
            if orig not in v:
                return None
            vm[frozenset(S)] = v[orig]
    return vm
