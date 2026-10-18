"""Strict local FP32 comparison return import, never a training/model runner."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.codeformer_fp32_comparison_v1.return_archive import inspect_and_import


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.absolute(); outputs = root / 'outputs'
    review = outputs / 'codeformer_fp32_comparison_v1_return_review'
    review.mkdir(exist_ok=True)
    prefix = 'cctv-codeformer-fp32-comparison-v1'
    result = inspect_and_import(
        outputs / (prefix + '-results.tar.gz'), outputs / (prefix + '-results.tar.gz.sha256'),
        outputs / (prefix + '-export.json'), outputs / 'cctv_codeformer_fp32_comparison_v1_vm',
        outputs / 'codeformer_fp32_comparison_v1_return', review / 'transport_v1.json',
        dependency_root=outputs / 'cctv_codeformer_adaptation_v2_vm')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
