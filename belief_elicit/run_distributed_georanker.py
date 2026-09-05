"""CLI entry point for a deterministic GeoRanker worker shard."""
from belief_elicit.distributed_georanker import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
