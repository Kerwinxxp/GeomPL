"""Download an immutable HF revision and make offline model-ID loads resolve to it."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: Prefetch-Pinned-Model.py MODEL_ID REVISION")
    model_id, revision = sys.argv[1:]
    snapshot = Path(snapshot_download(model_id, revision=revision))
    assert snapshot.name == revision, f"resolved {snapshot.name}, expected {revision}"

    model_cache = snapshot.parent.parent
    refs = model_cache / "refs"
    refs.mkdir(exist_ok=True)
    temp_ref = refs / f"main.tmp-{os.getpid()}"
    temp_ref.write_text(revision, encoding="utf-8")
    os.replace(temp_ref, refs / "main")

    resolved = Path(snapshot_download(model_id, revision="main", local_files_only=True))
    assert resolved.name == revision, f"offline main resolved {resolved.name}, expected {revision}"
    print(f"PASS pinned {model_id} at {revision}: {resolved}")


if __name__ == "__main__":
    main()
