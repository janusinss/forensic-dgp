"""Retain an implementing-assistant observation after a displayed sheet view.

This records the human-readable observation, not an independent final review.
It cannot verify a view occurred; view_image tool calls supply that evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = [ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/visual_review',
           ROOT/'outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis/visual_review']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--sheet',required=True)
    parser.add_argument('--notes',required=True)
    args = parser.parse_args()
    dest=args.dest.resolve()
    assert dest in [path.resolve() for path in ALLOWED]
    manifest_path=dest/'sheet_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    sheets={row['name']:row['sha256'] for row in manifest['sheets']}
    assert args.sheet in sheets and Path(args.sheet).name==args.sheet
    assert hashlib.sha256((dest/args.sheet).read_bytes()).hexdigest()==sheets[args.sheet]
    assert len(args.notes.strip())>=30
    observations=dest.parent/'visual_observations'
    observations.mkdir(exist_ok=True)
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        sheet=args.sheet,sheet_sha256=sheets[args.sheet],
        sheet_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        displayed_through='view_image at original resolution',
        reviewer='implementing assistant',independent_final_review=False,
        image_generation_alone_not_view_evidence=True,notes=args.notes,
        model_qualified=False,goal_complete=False)
    with (observations/(args.sheet+'.json')).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(recorded_sheet=args.sheet,independent_final_review=False))


if __name__=='__main__':main()
