"""Verify/import the actual manual return; never run the VM or resume training."""
import argparse
import json
from pathlib import Path

from research.codeformer_adaptation_v1.return_archive import inspect_and_import


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    review = root / "outputs/codeformer_adaptation_v1_return_review"
    review.mkdir(exist_ok=True)
    print(json.dumps(inspect_and_import(
        root / "outputs/cctv-codeformer-adaptation-v1-results.tar.gz",
        root / "outputs/cctv-codeformer-adaptation-v1-results.tar.gz.sha256",
        root / "outputs/cctv-codeformer-adaptation-v1-export.json",
        root / "outputs/cctv_codeformer_adaptation_v1_vm",
        root / "outputs/codeformer_adaptation_v1_return", review / "archive_import_v1.json")))
