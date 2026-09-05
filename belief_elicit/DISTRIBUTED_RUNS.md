# Dual-machine GeoRanker runs

Images are assigned with `SHA-256(image_id) mod num_shards`. Assignment is stable across machines and Python versions. Keep every subset of one image on the same worker.

Prepare or run the local half:

```powershell
belief_elicit\.venv_gr\Scripts\python.exe -m belief_elicit.run_distributed_georanker --num-shards 2 --shard-index 0 --worker-name local --part control --ops inpaint --prepare-only
```

Remove `--prepare-only` to score. On the remote machine use the same Git commit, gallery, sweep, cache contract, and arguments, changing only `--shard-index 1 --worker-name remote4090`. Each worker writes a distinct JSON under `belief_elicit/distributed_results/` and a `.meta.json` sidecar. Resume by running the identical command again. A safe alternate directory may be selected with `--out-dir`; canonical active result paths and arbitrary `--out` paths are rejected.

The sidecar locks the Qwen model ID and cached revision, GeoRanker LoRA hash, model implementation hash, temperature, NF4 settings, relevant package versions, gallery/sweep/runner hashes, and cache-generation parameters. Every image manifest in a shard must carry identical cache parameters. Unknown model metadata must use an explicit deliberate marker with a reason and must match on every worker.

After copying both shard files and sidecars into one directory:

```powershell
belief_elicit\.venv_gr\Scripts\python.exe -m belief_elicit.merge_distributed_georanker `
  belief_elicit/distributed_results/georanker_inpaint.control.local.shard-000-of-002.json `
  belief_elicit/distributed_results/georanker_inpaint.control.remote4090.shard-001-of-002.json `
  --out belief_elicit/distributed_results/georanker_inpaint.control.merged.json `
  --report belief_elicit/distributed_results/georanker_inpaint.control.merge-report.json
```

The merge recomputes `SHA-256(image_id) mod num_shards` for assigned, expected, and actual records. It rejects incompatible model/configuration signatures, swapped ownership, duplicate image/spec records, missing or extra specs, malformed 138-label support, non-finite probabilities, and distributions that do not sum to one. It always writes a report on validation failure and does not write the merged output. Use `--allow-incomplete` only for an explicitly partial analysis; ownership, compatibility, extra-spec, and distribution failures remain fatal.
