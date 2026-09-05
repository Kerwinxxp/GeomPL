"""CLI entry point for strict GeoRanker shard merging."""
from belief_elicit.distributed_georanker import merge_cli

if __name__ == "__main__":
    raise SystemExit(merge_cli())
