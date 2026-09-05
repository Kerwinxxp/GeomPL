import json
from pathlib import Path

import pytest

from belief_elicit.distributed_georanker import (
    MergeError,
    assign_shard,
    build_output_path,
    list_expected_specs,
    merge_shards,
    validate_cache_manifests,
    validate_merge_destination,
    validate_merge_destinations,
    validate_output_path,
)


def distribution(n=138):
    return {f"place-{i}": 1.0 / n for i in range(n)}


def record(image_id, specs):
    return {
        "image_id": image_id,
        "variants": [
            {"spec": spec, "prior": distribution(), "mpl": 0.1}
            for spec in specs
        ],
    }


def write_shard(tmp_path, name, records, *, expected, commit="abc", gallery="g",
                num_shards=2, shard_index=None, assigned=None, signature=None,
                schema_version=1):
    result = tmp_path / f"{name}.json"
    result.write_text(json.dumps(records), encoding="utf-8")
    meta = {
        "schema_version": schema_version,
        "worker_name": name,
        "num_shards": num_shards,
        "shard_index": (assign_shard(next(iter(expected)), num_shards) if expected
                          else 0) if shard_index is None else shard_index,
        "compatibility": {
            "git_commit": commit,
            "gallery_sha256": gallery,
            "gallery_size": 138,
            "sweep_sha256": "s",
            "runner_sha256": "r",
            "cache_contract": "inpaint-v1",
            "part": "main",
            "ops": "both",
            "variant": "B",
            "merge_km": 2.0,
            "model_signature": signature or {
                "base_model_id": "Qwen/Qwen2-VL-7B-Instruct",
                "base_model_revision": "rev",
                "lora_sha256": "lora",
                "implementation_sha256": "impl",
                "tau": 1.0,
                "quantization": {"mode": "nf4", "load_in_4bit": True,
                                 "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
                                 "attn_implementation": "sdpa", "device_map": "cuda",
                                 "skip_modules": ["value_head", "lm_head"]},
                "packages": {"torch": "2.0", "transformers": "4.0", "peft": "1",
                             "bitsandbytes": "1", "qwen-vl-utils": "1", "safetensors": "1"},
            },
        },
        "expected": expected,
        "assigned_image_ids": list(expected) if assigned is None else assigned,
    }
    Path(str(result) + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return result


def test_assignment_is_stable_and_partitioned():
    ids = [f"image-{i}" for i in range(200)]
    first = [assign_shard(i, 7) for i in ids]
    assert first == [assign_shard(i, 7) for i in ids]
    assert all(0 <= i < 7 for i in first)
    assert len(set(first)) == 7


def test_output_name_is_worker_and_shard_specific(tmp_path):
    p = build_output_path(tmp_path, "main", "local gpu", 0, 2)
    assert p.name == "georanker_inpaint.main.local-gpu.shard-000-of-002.json"


def test_expected_specs_are_read_without_importing_gpu_runner(tmp_path):
    for name in ("s0.png", "p0-1.png", "all.png", "c0-0.png", "cg0-0.png", "noise.png"):
        (tmp_path / name).touch()
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [
        {"file": "c0-0.png", "op": "inpaint"},
        {"file": "cg0-0.png", "op": "gray"},
    ]}), encoding="utf-8")
    assert list_expected_specs(tmp_path, "main", "both") == ["all", "p0-1", "s0"]
    assert list_expected_specs(tmp_path, "control", "inpaint") == ["c0-0"]


def test_merge_validates_and_combines_complete_shards(tmp_path):
    ids = {assign_shard(x, 2): x for x in (f"image-{i}" for i in range(100))}
    a_id, b_id = ids[0], ids[1]
    a = write_shard(tmp_path, "local", [record(a_id, ["s0", "all"])],
                    expected={a_id: ["s0", "all"]}, shard_index=0)
    b = write_shard(tmp_path, "remote", [record(b_id, ["s0", "all"])],
                    expected={b_id: ["s0", "all"]}, shard_index=1)
    merged, report = merge_shards([a, b])
    assert {r["image_id"] for r in merged} == {a_id, b_id}
    assert report["valid"] is True
    assert report["counts"]["variants"] == 4


def test_merge_reports_missing_specs_without_writing_success(tmp_path):
    a = write_shard(tmp_path, "local", [record("a", ["s0"])],
                    expected={"a": ["s0", "all"]})
    with pytest.raises(MergeError) as exc:
        merge_shards([a], require_all_shards=False)
    assert exc.value.report["missing_specs"] == [{"image_id": "a", "spec": "all"}]


def test_merge_rejects_compatibility_mismatch(tmp_path):
    a = write_shard(tmp_path, "local", [record("a", ["s0"])], expected={"a": ["s0"]})
    b = write_shard(tmp_path, "remote", [record("b", ["s0"])],
                    expected={"b": ["s0"]}, commit="different")
    with pytest.raises(MergeError) as exc:
        merge_shards([a, b])
    assert "git_commit" in exc.value.report["compatibility_errors"][0]


def test_merge_rejects_swapped_shard_ownership(tmp_path):
    iid = next(x for x in (f"image-{i}" for i in range(100)) if assign_shard(x, 2) == 0)
    a = write_shard(tmp_path, "local", [record(iid, ["s0"])], expected={iid: ["s0"]},
                    shard_index=1)
    with pytest.raises(MergeError) as exc:
        merge_shards([a], require_all_shards=False)
    assert exc.value.report["ownership_errors"]


def test_merge_rejects_actual_extra_spec(tmp_path):
    iid = "a"
    a = write_shard(tmp_path, "local", [record(iid, ["s0", "bogus"])],
                    expected={iid: ["s0"]}, num_shards=1, shard_index=0)
    with pytest.raises(MergeError) as exc:
        merge_shards([a])
    assert exc.value.report["extra_specs"] == [{"image_id": iid, "spec": "bogus"}]


def test_merge_rejects_incompatible_model_signature(tmp_path):
    a = write_shard(tmp_path, "local", [record("a", ["s0"])], expected={"a": ["s0"]})
    sig = {
        "base_model_id": "other", "base_model_revision": "rev", "lora_sha256": "lora",
        "implementation_sha256": "impl", "tau": 1.0,
        "quantization": {"mode": "nf4", "load_in_4bit": True,
                         "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
                         "attn_implementation": "sdpa", "device_map": "cuda",
                         "skip_modules": ["value_head", "lm_head"]},
        "packages": {"torch": "2.0", "transformers": "4.0", "peft": "1",
                     "bitsandbytes": "1", "qwen-vl-utils": "1", "safetensors": "1"},
    }
    b = write_shard(tmp_path, "remote", [record("b", ["s0"])], expected={"b": ["s0"]},
                    signature=sig)
    with pytest.raises(MergeError) as exc:
        merge_shards([a, b])
    assert any("model_signature" in e for e in exc.value.report["compatibility_errors"])


def test_merge_rejects_implicit_unknown_model_value(tmp_path):
    sig = {
        "base_model_id": "Qwen/Qwen2-VL-7B-Instruct", "base_model_revision": None,
        "lora_sha256": "lora", "implementation_sha256": "impl", "tau": 1.0,
        "quantization": {"mode": "nf4", "load_in_4bit": True,
                         "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16",
                         "attn_implementation": "sdpa", "device_map": "cuda",
                         "skip_modules": ["value_head", "lm_head"]},
        "packages": {"torch": "2.0", "transformers": "4.0", "peft": "1",
                     "bitsandbytes": "1", "qwen-vl-utils": "1", "safetensors": "1"},
    }
    a = write_shard(tmp_path, "local", [record("a", ["s0"])], expected={"a": ["s0"]},
                    num_shards=1, shard_index=0, signature=sig)
    with pytest.raises(MergeError) as exc:
        merge_shards([a])
    assert any("base_model_revision" in e for e in exc.value.report["compatibility_errors"])


def test_merge_rejects_conflicting_duplicate_spec(tmp_path):
    a = write_shard(tmp_path, "local", [record("same", ["s0"])],
                    expected={"same": ["s0"]})
    bad = record("same", ["s0"])
    bad["variants"][0]["mpl"] = 0.2
    b = write_shard(tmp_path, "remote", [bad], expected={"same": ["s0"]})
    with pytest.raises(MergeError) as exc:
        merge_shards([a, b])
    assert exc.value.report["conflicts"][0]["spec"] == "s0"


def test_merge_rejects_duplicate_image_ownership_even_for_distinct_specs(tmp_path):
    a = write_shard(tmp_path, "local", [record("same", ["s0"])], expected={"same": ["s0"]})
    b = write_shard(tmp_path, "remote", [record("same", ["all"])], expected={"same": ["all"]})
    with pytest.raises(MergeError) as exc:
        merge_shards([a, b])
    assert exc.value.report["duplicate_images"][0]["image_id"] == "same"


@pytest.mark.parametrize("prior", [
    {f"place-{i}": 1.0 / 137 for i in range(137)},
    {**distribution(), "place-0": 0.5},
])
def test_merge_rejects_invalid_distribution(tmp_path, prior):
    rec = record("a", ["s0"])
    rec["variants"][0]["prior"] = prior
    a = write_shard(tmp_path, "local", [rec], expected={"a": ["s0"]})
    with pytest.raises(MergeError) as exc:
        merge_shards([a], require_all_shards=False)
    assert exc.value.report["distribution_errors"]


def test_merge_rejects_missing_shard_by_default(tmp_path):
    a = write_shard(tmp_path, "local", [record("a", ["s0"])], expected={"a": ["s0"]})
    with pytest.raises(MergeError) as exc:
        merge_shards([a])
    assert exc.value.report["missing_shards"] == [1]


def test_merge_reports_assigned_image_with_no_result(tmp_path):
    a = write_shard(tmp_path, "local", [], expected={}, assigned=["missing"], num_shards=1,
                    shard_index=0)
    with pytest.raises(MergeError) as exc:
        merge_shards([a])
    assert exc.value.report["missing_images"] == ["missing"]


def test_unsafe_output_rejects_active_result_path(tmp_path):
    active = tmp_path / "belief_elicit" / "georanker_inpaint_results.json"
    with pytest.raises(ValueError, match="active output"):
        validate_output_path(active, tmp_path, tmp_path / "belief_elicit" / "distributed_results")


@pytest.mark.parametrize("relative", [
    "belief_elicit/georanker_inpaint_results.json",
    "belief_elicit/logs/chain.log",
])
def test_merge_destination_rejects_active_result_and_log_paths(tmp_path, relative):
    destination = tmp_path / relative
    with pytest.raises(ValueError, match="active"):
        validate_merge_destination(
            destination, tmp_path, tmp_path / "belief_elicit" / "distributed_results"
        )


def test_merge_destination_must_be_under_selected_safe_directory(tmp_path):
    with pytest.raises(ValueError, match="safe output directory"):
        validate_merge_destination(
            tmp_path / "elsewhere" / "merged.json",
            tmp_path,
            tmp_path / "belief_elicit" / "distributed_results",
        )


def test_merge_output_cannot_overwrite_input_shard_after_normalization(tmp_path):
    safe = tmp_path / "belief_elicit" / "distributed_results"
    safe.mkdir(parents=True)
    shard = safe / "worker.json"
    with pytest.raises(ValueError, match="input shard"):
        validate_merge_destinations(
            [shard], safe / "subdir" / ".." / "worker.json", safe / "report.json",
            tmp_path, safe,
        )


def test_merge_report_cannot_overwrite_input_sidecar_case_insensitively(tmp_path):
    safe = tmp_path / "belief_elicit" / "distributed_results"
    safe.mkdir(parents=True)
    shard = safe / "Worker.JSON"
    sidecar_case_variant = Path(str(shard).upper() + ".META.JSON")
    with pytest.raises(ValueError, match="metadata sidecar"):
        validate_merge_destinations(
            [shard], safe / "merged.json", sidecar_case_variant,
            tmp_path, safe,
        )


def test_merge_rejects_schema_version_mismatch(tmp_path):
    a = write_shard(tmp_path, "local", [record("a", ["s0"])], expected={"a": ["s0"]},
                    num_shards=1, shard_index=0, schema_version=2)
    with pytest.raises(MergeError) as exc:
        merge_shards([a])
    assert exc.value.report["schema_errors"]


def test_cache_manifests_must_have_identical_parameters(tmp_path):
    for iid, seed in (("a", 42), ("b", 43)):
        d = tmp_path / iid
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "src": "cues", "nctrl": 2, "seed": seed, "rng": "stable", "dilate_px": 5,
        }), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent cache parameters"):
        validate_cache_manifests(tmp_path, ["a", "b"], no_manifest_ok=False)
