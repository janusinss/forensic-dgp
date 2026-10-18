"""Strict local diagnostic return verification/import; never run packet code."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from research.codeformer_bounded_fusion_feasibility_v1.return_archive import inspect_and_import


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-return',action='store_true',required=True)
    parser.parse_args()
    review=ROOT/'outputs/codeformer_bounded_fusion_feasibility_v1_return_review'
    review.mkdir(parents=True,exist_ok=True)
    prefix=ROOT/'outputs/cctv-codeformer-bounded-fusion-feasibility-v1'
    result=inspect_and_import(Path(str(prefix)+'-results.tar.gz'),Path(str(prefix)+'-results.tar.gz.sha256'),
        Path(str(prefix)+'-export.json'),ROOT/'outputs/cctv_codeformer_bounded_fusion_feasibility_v1_vm',
        ROOT/'outputs/codeformer_bounded_fusion_feasibility_v1_return',review/'transport_v1.json',
        parent_packet=ROOT/'outputs/cctv_codeformer_fp32_comparison_v1_vm',
        original_asset=ROOT/'outputs/cctv_codeformer_adaptation_v2_vm/weights/codeformer.pth',
        stopped_asset=ROOT/'outputs/codeformer_fp32_comparison_v1_return/outputs/fp32_comparison_v1/stopped_checkpoint.pt')
    print(json.dumps({key:result[key] for key in ('complete','members','payload_bytes','import_directory','classification')}))


if __name__=='__main__':main()
