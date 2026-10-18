"""Local returned-evidence audits; no training, neural inference or VM actions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=['availability', 'transport', 'metrics', 'trajectory'],
                        default='availability')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    outputs = root / 'outputs'
    archive = outputs / 'cctv-codeformer-adaptation-v2-results.tar.gz'
    sidecar = outputs / 'cctv-codeformer-adaptation-v2-results.tar.gz.sha256'
    export = outputs / 'cctv-codeformer-adaptation-v2-export.json'
    review = outputs / 'codeformer_adaptation_v2_return_review'
    imported = outputs / 'codeformer_adaptation_v2_return'
    packet = outputs / 'cctv_codeformer_adaptation_v2_vm'
    if args.stage == 'availability':
        print(json.dumps({'files': {p.name: {'present': p.is_file(),
            'bytes': p.stat().st_size if p.is_file() else None} for p in [archive, sidecar, export]},
            'presence_is_not_verified_return': True, 'neural_calls': 0, 'VM_calls': 0}))
        return
    review.mkdir(exist_ok=True)
    if args.stage == 'transport':
        from research.codeformer_adaptation_v2.return_archive import inspect_and_import
        receipt = inspect_and_import(archive, sidecar, export, packet, imported,
                                     review / 'archive_import_v1.json')
    else:
        transport = json.loads((review / 'archive_import_v1.json').read_text())
        assert transport['complete'] and transport['exact_namespace']
        assert transport['protocol_sha256'] == '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
        assert Path(transport['import_directory']).resolve() == imported.resolve()
        if args.stage == 'metrics':
            from research.codeformer_adaptation_v2.return_metrics import audit_return
            receipt = audit_return(imported, packet, review / 'metrics_v1')
        else:
            from research.codeformer_adaptation_v2.return_trajectory import audit_return
            receipt = audit_return(imported, packet, review / 'trajectory_v1')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
