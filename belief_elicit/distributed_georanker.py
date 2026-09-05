"""Deterministic sharding and strict validation for distributed GeoRanker jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


SCHEMA_VERSION = 1
PROB_TOLERANCE = 1e-6
ACTIVE_OUTPUT_NAMES = {
    "georanker_inpaint_results.json",
    "georanker_inpaint_control_results.json",
    "georanker_inpaint_vocab_results.json",
    "georanker_sweep_results.json",
    "georanker_lattice_results.json",
}
CACHE_PARAMETER_KEYS = ("src", "nctrl", "seed", "rng", "dilate_px")


class MergeError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_shard(image_id, num_shards):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    digest = hashlib.sha256(str(image_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def slug(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not value:
        raise ValueError("worker_name must contain a letter or digit")
    return value.lower()


def build_output_path(out_dir, part, worker_name, shard_index, num_shards):
    return Path(out_dir) / (
        f"georanker_inpaint.{part}.{slug(worker_name)}."
        f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )


def validate_output_path(path, root, safe_dir):
    path, root, safe_dir = Path(path).resolve(), Path(root).resolve(), Path(safe_dir).resolve()
    active_dir = root / "belief_elicit"
    if path.parent == active_dir and path.name in ACTIVE_OUTPUT_NAMES:
        raise ValueError(f"refusing canonical active output path: {path}")
    try:
        path.relative_to(safe_dir)
    except ValueError as e:
        raise ValueError(f"shard output must stay under the selected safe output directory: {safe_dir}") from e
    if not re.fullmatch(r"georanker_inpaint\.(main|control|all)\.[a-z0-9._-]+\.shard-\d{3}-of-\d{3}\.json", path.name):
        raise ValueError("shard output must use the deterministic worker/shard filename")
    return path


def _is_active_path(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    active_dir = root / "belief_elicit"
    logs_dir = active_dir / "logs"
    if path.parent == active_dir and path.name in ACTIVE_OUTPUT_NAMES:
        return True
    try:
        path.relative_to(logs_dir)
        return True
    except ValueError:
        return False


def validate_merge_destination(path, root, safe_dir):
    """Protect active artifacts and confine merge products to a selected directory."""
    path, safe_dir = Path(path).resolve(), Path(safe_dir).resolve()
    if _is_active_path(path, root):
        raise ValueError(f"refusing active result/log destination: {path}")
    try:
        path.relative_to(safe_dir)
    except ValueError as e:
        raise ValueError(f"merge destination must stay under the selected safe output directory: {safe_dir}") from e
    return path


def _windows_path_key(path):
    """Normalize separators, dot segments, absoluteness, and Windows case."""
    return os.path.normcase(os.path.abspath(os.path.normpath(os.fspath(path))))


def validate_merge_destinations(inputs, out, report, root, safe_dir):
    """Validate merge paths before any shard or sidecar is opened."""
    out = validate_merge_destination(out, root, safe_dir)
    report = validate_merge_destination(report, root, safe_dir)
    out_key, report_key = _windows_path_key(out), _windows_path_key(report)
    if out_key == report_key:
        raise ValueError("--out and --report must be different files")
    shard_keys = {_windows_path_key(p): os.fspath(p) for p in inputs}
    sidecar_keys = {_windows_path_key(_meta_path(p)): os.fspath(_meta_path(p)) for p in inputs}
    for label, key in (("--out", out_key), ("--report", report_key)):
        if key in shard_keys:
            raise ValueError(f"{label} resolves to an input shard path: {shard_keys[key]}")
        if key in sidecar_keys:
            raise ValueError(f"{label} resolves to an input metadata sidecar path: {sidecar_keys[key]}")
    return out, report


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _meta_path(result_path):
    return Path(str(result_path) + ".meta.json")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_expected_specs(cache_dir, part, ops):
    """Read the cache contract without importing the GPU/model runner."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    entries = manifest.get("files", []) if isinstance(manifest, dict) else []
    op_by_file = {e.get("file"): e.get("op") for e in entries if isinstance(e, dict)}
    specs = []
    for path in cache_dir.glob("*.png"):
        stem = path.stem
        if stem == "all" or re.fullmatch(r"s\d+", stem) or re.fullmatch(r"p\d+-\d+", stem):
            kind, op = "main", "inpaint"
        elif re.fullmatch(r"cg\d+-\d+", stem):
            kind, op = "control", op_by_file.get(path.name) or "gray"
        elif re.fullmatch(r"c\d+-\d+", stem):
            kind, op = "control", op_by_file.get(path.name) or "inpaint"
        else:
            continue
        if part != "all" and kind != part:
            continue
        if kind == "control" and ops != "both" and op != ops:
            continue
        specs.append(stem)
    return sorted(specs)


def validate_cache_manifests(cache_root, image_ids, no_manifest_ok=False):
    """Require and compare cache-generation parameters for every assigned image."""
    del no_manifest_ok  # distributed runs intentionally do not allow partial manifests
    cache_root = Path(cache_root)
    common = None
    missing = []
    for iid in image_ids:
        path = cache_root / iid / "manifest.json"
        if not path.exists():
            missing.append(iid)
            continue
        manifest = _read_json(path)
        params = {k: manifest.get(k) for k in CACHE_PARAMETER_KEYS}
        if any(v is None for v in params.values()):
            absent = [k for k, v in params.items() if v is None]
            raise ValueError(f"cache manifest {iid} lacks parameters: {absent}")
        if common is None:
            common = params
        elif _canonical(params) != _canonical(common):
            raise ValueError(f"inconsistent cache parameters: {iid} has {params}, expected {common}")
    if missing:
        raise ValueError(f"missing cache manifests for assigned images: {missing}")
    if common is None:
        raise ValueError("shard has no assigned cache manifests")
    return common


def _unknown(reason):
    return {"status": "unknown", "deliberate": True, "reason": reason}


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return _unknown("package metadata unavailable in the launching environment")


def _base_revision(model_id):
    model_dir = "models--" + model_id.replace("/", "--")
    roots = []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        ref = root / model_dir / "refs" / "main"
        if ref.exists():
            value = ref.read_text(encoding="utf-8").strip()
            if value:
                return value
    return _unknown("Hugging Face cached revision ref was not found")


def build_model_signature(root, tau=1.0, quantization="nf4"):
    root = Path(root)
    model_impl = root / "belief_elicit" / "georanker_belief.py"
    adapter = root / "belief_elicit" / "georanker_ckpt" / "adapter_model.safetensors"
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    return {
        "base_model_id": model_id,
        "base_model_revision": _base_revision(model_id),
        "lora_sha256": sha256_file(adapter) if adapter.exists() else _unknown("LoRA checkpoint is absent"),
        "implementation_sha256": sha256_file(model_impl),
        "tau": float(tau),
        "quantization": {
            "mode": quantization, "load_in_4bit": quantization == "nf4",
            "bnb_4bit_quant_type": "nf4" if quantization == "nf4" else None,
            "bnb_4bit_compute_dtype": "bfloat16" if quantization == "nf4" else None,
            "attn_implementation": "sdpa", "device_map": "cuda",
            "skip_modules": ["value_head", "lm_head"] if quantization == "nf4" else [],
        },
        "packages": {name: _package_version(name) for name in
                     ("torch", "transformers", "peft", "bitsandbytes", "qwen-vl-utils", "safetensors")},
    }


def _valid_unknown(value):
    return (isinstance(value, dict) and value.get("status") == "unknown"
            and value.get("deliberate") is True and isinstance(value.get("reason"), str)
            and bool(value["reason"].strip()))


def _validate_signature(signature, source, errors):
    required = ("base_model_id", "base_model_revision", "lora_sha256",
                "implementation_sha256", "tau", "quantization", "packages")
    for key in required:
        if key not in signature or signature[key] is None:
            errors.append(f"model_signature.{key} missing in {source}")
    required_quant = ("mode", "load_in_4bit", "bnb_4bit_quant_type",
                      "bnb_4bit_compute_dtype", "attn_implementation", "device_map", "skip_modules")
    quant = signature.get("quantization", {})
    if isinstance(quant, dict) and quant.get("status") != "unknown":
        for key in required_quant:
            if key not in quant:
                errors.append(f"model_signature.quantization.{key} missing in {source}")
    required_packages = ("torch", "transformers", "peft", "bitsandbytes", "qwen-vl-utils", "safetensors")
    packages = signature.get("packages", {})
    if isinstance(packages, dict) and packages.get("status") != "unknown":
        for key in required_packages:
            if key not in packages:
                errors.append(f"model_signature.packages.{key} missing in {source}")
    def walk(value, path):
        if isinstance(value, dict):
            if value.get("status") == "unknown":
                if not _valid_unknown(value):
                    errors.append(f"{path} has an invalid unknown marker in {source}")
                return
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif value is None and path not in ("model_signature.quantization.bnb_4bit_quant_type",
                                             "model_signature.quantization.bnb_4bit_compute_dtype"):
            errors.append(f"{path} is implicitly unknown in {source}")
    walk(signature, "model_signature")


def _validate_prior(prior, gallery_labels, image_id, spec, errors):
    if not isinstance(prior, dict):
        errors.append({"image_id": image_id, "spec": spec, "error": "prior is not an object"})
        return
    keys = set(prior)
    expected = set(gallery_labels)
    if len(prior) != 138 or keys != expected:
        errors.append({
            "image_id": image_id, "spec": spec,
            "error": "distribution labels do not match the 138-label gallery",
            "count": len(prior), "missing_labels": sorted(expected - keys),
            "extra_labels": sorted(keys - expected),
        })
        return
    try:
        values = [float(prior[k]) for k in gallery_labels]
    except (TypeError, ValueError):
        errors.append({"image_id": image_id, "spec": spec, "error": "non-numeric probability"})
        return
    if any(not math.isfinite(v) or v < 0.0 or v > 1.0 for v in values):
        errors.append({"image_id": image_id, "spec": spec, "error": "probability outside [0,1] or non-finite"})
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROB_TOLERANCE):
        errors.append({"image_id": image_id, "spec": spec, "error": "probabilities do not sum to 1", "sum": total})


def merge_shards(paths, allow_incomplete=False, require_all_shards=True):
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("at least one shard is required")
    metas = []
    records_by_image = {}
    ownership = {}
    duplicate_images = []
    conflicts = []
    duplicate_specs = []
    distribution_errors = []
    compatibility_errors = []
    schema_errors = []
    ownership_errors = []
    expected = {}
    assigned_images = set()

    for path in paths:
        meta_path = _meta_path(path)
        if not meta_path.exists():
            raise MergeError(f"missing metadata sidecar: {meta_path}", {"valid": False, "missing_metadata": [str(meta_path)]})
        meta = _read_json(meta_path)
        records = _read_json(path)
        if not isinstance(records, list):
            raise MergeError(f"shard is not a record list: {path}", {"valid": False, "invalid_shards": [str(path)]})
        metas.append((path, meta))
        if meta.get("schema_version") != SCHEMA_VERSION:
            schema_errors.append({"file": str(path), "found": meta.get("schema_version"),
                                  "expected": SCHEMA_VERSION})
        num_shards = meta.get("num_shards")
        shard_index = meta.get("shard_index")
        assigned_here = set(meta.get("assigned_image_ids", []))
        expected_here = set(meta.get("expected", {}))
        assigned_images.update(assigned_here)
        if expected_here != assigned_here:
            ownership_errors.append({"file": str(path), "error": "expected image set differs from assigned image set",
                                     "assigned_only": sorted(assigned_here - expected_here),
                                     "expected_only": sorted(expected_here - assigned_here)})
        if isinstance(num_shards, int) and isinstance(shard_index, int):
            for iid in sorted(assigned_here | expected_here):
                actual_shard = assign_shard(iid, num_shards)
                if actual_shard != shard_index:
                    ownership_errors.append({"file": str(path), "image_id": iid,
                                             "declared_shard": shard_index, "computed_shard": actual_shard})
        else:
            ownership_errors.append({"file": str(path), "error": "invalid num_shards or shard_index"})
        signature = meta.get("compatibility", {}).get("model_signature")
        if not isinstance(signature, dict):
            compatibility_errors.append(f"model_signature missing in {path.name}")
        else:
            _validate_signature(signature, path.name, compatibility_errors)
        for iid, specs in meta.get("expected", {}).items():
            expected.setdefault(iid, set()).update(specs)
        gallery_labels = meta.get("gallery_labels", [f"place-{i}" for i in range(138)])
        for rec in records:
            iid = rec.get("image_id")
            if not isinstance(iid, str):
                conflicts.append({"file": str(path), "error": "record missing image_id"})
                continue
            if iid not in assigned_here:
                ownership_errors.append({"file": str(path), "image_id": iid,
                                         "error": "actual image is not assigned to this shard"})
            elif isinstance(num_shards, int) and assign_shard(iid, num_shards) != shard_index:
                ownership_errors.append({"file": str(path), "image_id": iid,
                                         "declared_shard": shard_index,
                                         "computed_shard": assign_shard(iid, num_shards)})
            owner = ownership.setdefault(iid, str(path))
            if owner != str(path) and not any(x["image_id"] == iid for x in duplicate_images):
                duplicate_images.append({"image_id": iid, "files": [owner, str(path)]})
            target = records_by_image.setdefault(iid, {k: v for k, v in rec.items() if k != "variants"})
            variants = target.setdefault("variants", [])
            known = {v.get("spec"): v for v in variants}
            for variant in rec.get("variants", []):
                spec = variant.get("spec")
                _validate_prior(variant.get("prior"), gallery_labels, iid, spec, distribution_errors)
                if spec in known:
                    if _canonical(known[spec]) != _canonical(variant):
                        conflicts.append({"image_id": iid, "spec": spec, "files": [owner, str(path)]})
                    else:
                        duplicate_specs.append({"image_id": iid, "spec": spec, "files": [owner, str(path)]})
                else:
                    variants.append(variant)
                    known[spec] = variant

    baseline_path, baseline = metas[0]
    baseline_compat = baseline.get("compatibility", {})
    for path, meta in metas[1:]:
        compat = meta.get("compatibility", {})
        for key in sorted(set(baseline_compat) | set(compat)):
            if baseline_compat.get(key) != compat.get(key):
                compatibility_errors.append(
                    f"{key}: {baseline_path.name}={baseline_compat.get(key)!r}, {path.name}={compat.get(key)!r}"
                )

    shard_indices = [m.get("shard_index") for _, m in metas]
    num_shards_values = {m.get("num_shards") for _, m in metas}
    missing_shards = []
    if len(num_shards_values) != 1:
        compatibility_errors.append(f"num_shards differs across sidecars: {sorted(num_shards_values, key=str)}")
    elif require_all_shards:
        total_shards = next(iter(num_shards_values))
        if isinstance(total_shards, int) and total_shards > 0:
            missing_shards = sorted(set(range(total_shards)) - set(shard_indices))
    if len(shard_indices) != len(set(shard_indices)):
        compatibility_errors.append(f"duplicate shard_index values: {shard_indices}")

    missing_specs = []
    extra_specs = []
    for iid, specs in sorted(expected.items()):
        actual = {v.get("spec") for v in records_by_image.get(iid, {}).get("variants", [])}
        for spec in sorted(specs - actual):
            missing_specs.append({"image_id": iid, "spec": spec})
        for spec in sorted(actual - specs):
            extra_specs.append({"image_id": iid, "spec": spec})
    unexpected_images = sorted(set(records_by_image) - set(expected))
    missing_images = sorted(assigned_images - set(records_by_image))

    fatal = bool(conflicts or duplicate_images or compatibility_errors or schema_errors or ownership_errors
                 or distribution_errors or unexpected_images or extra_specs)
    incomplete = bool(missing_specs or missing_images or missing_shards)
    report = {
        "valid": not fatal and (allow_incomplete or not incomplete),
        "complete": not incomplete,
        "inputs": [str(p) for p in paths],
        "counts": {
            "images": len(records_by_image),
            "variants": sum(len(r.get("variants", [])) for r in records_by_image.values()),
            "expected_images": len(expected),
            "expected_variants": sum(len(s) for s in expected.values()),
        },
        "compatibility_errors": compatibility_errors,
        "schema_errors": schema_errors,
        "ownership_errors": ownership_errors,
        "conflicts": conflicts,
        "duplicate_images": duplicate_images,
        "duplicate_specs": duplicate_specs,
        "distribution_errors": distribution_errors,
        "missing_specs": missing_specs,
        "extra_specs": extra_specs,
        "missing_images": missing_images,
        "missing_shards": missing_shards,
        "unexpected_images": unexpected_images,
    }
    if fatal or (incomplete and not allow_incomplete):
        raise MergeError("shard validation failed", report)
    merged = []
    for iid in sorted(records_by_image):
        rec = records_by_image[iid]
        rec["variants"] = sorted(rec.get("variants", []), key=lambda v: v.get("spec", ""))
        merged.append(rec)
    return merged, report


def _git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def prepare_shard(args):
    root = Path(__file__).resolve().parents[1]
    sweep = _read_json(args.sweep)
    image_ids = [r["image_id"] for r in sweep]
    assigned = [iid for iid in image_ids if assign_shard(iid, args.num_shards) == args.shard_index]
    cache = Path(args.cache)
    expected = {}
    cache_parameters = validate_cache_manifests(cache, assigned, args.no_manifest_ok)
    for iid in assigned:
        d = cache / iid
        expected[iid] = list_expected_specs(d, args.part, args.ops)
    gallery_path = root / "data" / "gallery_v2.json"
    gallery = [g for g in _read_json(gallery_path) if g.get("gps")]
    labels = [g["label"] for g in gallery]
    runner = root / "belief_elicit" / "run_georanker_inpaint.py"
    out = Path(args.out) if args.out else build_output_path(args.out_dir, args.part, args.worker_name, args.shard_index, args.num_shards)
    out = validate_output_path(out, root, args.out_dir)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "worker_name": args.worker_name,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "gallery_labels": labels,
        "assigned_image_ids": assigned,
        "expected": expected,
        "compatibility": {
            "git_commit": _git_commit(root), "gallery_sha256": sha256_file(gallery_path),
            "gallery_size": len(labels), "sweep_sha256": sha256_file(args.sweep),
            "runner_sha256": sha256_file(runner), "cache_contract": "inpaint-v1",
            "orchestrator_sha256": sha256_file(Path(__file__)),
            "cache_parameters": cache_parameters,
            "model_signature": build_model_signature(root, tau=1.0, quantization="nf4"),
            "part": args.part, "ops": args.ops, "variant": "B", "merge_km": 2.0,
        },
    }
    old_meta_path = _meta_path(out)
    if old_meta_path.exists() and _canonical(_read_json(old_meta_path)) != _canonical(meta):
        raise ValueError(f"metadata mismatch for resumed shard: {old_meta_path}")
    atomic_json(old_meta_path, meta)
    return out, meta


def run_cli(argv=None):
    p = argparse.ArgumentParser(description="Run one deterministic GeoRanker image shard.")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--shard-index", type=int, required=True)
    p.add_argument("--worker-name", required=True)
    p.add_argument("--part", choices=["main", "control", "all"], default="main")
    p.add_argument("--ops", choices=["inpaint", "gray", "both"], default="both")
    p.add_argument("--cache", default=str(Path(__file__).with_name("inpaint_cache")))
    p.add_argument("--sweep", default=str(Path(__file__).with_name("georanker_sweep_results.json")))
    p.add_argument("--out-dir", default=str(Path(__file__).with_name("distributed_results")))
    p.add_argument("--out")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-manifest-ok", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        p.error("--shard-index must be in [0, --num-shards)")
    out, meta = prepare_shard(args)
    print(f"worker={args.worker_name} shard={args.shard_index}/{args.num_shards} images={len(meta['assigned_image_ids'])} output={out}")
    if args.prepare_only:
        return 0
    command = [sys.executable, "-m", "belief_elicit.run_georanker_inpaint", "--cache", args.cache,
               "--sweep", args.sweep, "--part", args.part, "--ops", args.ops,
               "--out", str(out), "--ids", *meta["assigned_image_ids"]]
    if args.limit:
        command += ["--limit", str(args.limit)]
    return subprocess.call(command, cwd=Path(__file__).resolve().parents[1])


def merge_cli(argv=None):
    p = argparse.ArgumentParser(description="Strictly validate and merge GeoRanker shard files.")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--out-dir", default=str(Path(__file__).with_name("distributed_results")),
                   help="safe directory for both merged output and validation report")
    p.add_argument("--allow-incomplete", action="store_true")
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        out, report = validate_merge_destinations(
            args.inputs, args.out, args.report, root, args.out_dir
        )
        args.out, args.report = str(out), str(report)
    except ValueError as e:
        p.error(str(e))
    try:
        merged, report = merge_shards(args.inputs, args.allow_incomplete)
    except MergeError as e:
        atomic_json(args.report, e.report)
        print(f"validation failed; report: {args.report}", file=sys.stderr)
        return 2
    atomic_json(args.out, merged)
    atomic_json(args.report, report)
    print(f"merged {report['counts']['images']} images / {report['counts']['variants']} variants -> {args.out}")
    return 0
