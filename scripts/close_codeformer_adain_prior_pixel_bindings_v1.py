"""Close all compared prior pixels against the already audited FP32 manifest.

CPU bytes only; separate prospective-strengthening receipt, no audit/source or
historical artifact rewritten and no model interpretation added.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_pixels import require, sha, read, write, PARENT_SHA

MANIFEST = 'outputs/fp32_comparison_v1/export_manifest_20261016T132557Z-e00becd6.json'
MANIFEST_SHA = '4562d4961c7862828d2164b96dd6df675f7b09171298109aba29cce44fb8395b'


def close(prior, metrics_root, output):
    prior, metrics_root = Path(prior).resolve(strict=True), Path(metrics_root).resolve(strict=True)
    source = prior / MANIFEST
    require(sha(source) == MANIFEST_SHA, 'Retained independently verified FP32 manifest differs')
    manifest = read(source)
    require(manifest['complete'] and manifest['protocol_sha256'] == PARENT_SHA,
            'Prior manifest/protocol incomplete')
    require(sha(prior / 'protocol.json') == PARENT_SHA, 'Prior frozen protocol differs')
    audit = read(metrics_root / 'metrics_audit.json')
    visual = read(metrics_root / 'visual_manifest.json')
    require(audit['complete'] and visual['complete'] and len(visual['cases']) == 406,
            'Closed full-cohort pixel audit required')
    files = {}
    by_id = {row['id']: row for row in visual['cases']}
    for update in (0, 50):
        relative_dir = 'outputs/fp32_comparison_v1/snapshot_%04d' % update
        relative = relative_dir + '/metrics.json'
        require(sha(prior / relative) == manifest['files_sha256'][relative], 'Prior metrics manifest hash differs')
        files[relative] = manifest['files_sha256'][relative]
        records = read(prior / relative)
        require(records['complete'] and records['update'] == update and
                len(records['rows']) == 406 and {row['id'] for row in records['rows']} == set(by_id),
                'Prior compared namespace differs')
        for record in records['rows']:
            for name, key in [('raw.npy', 'raw_sha256'), ('native.png', 'native_png_sha256')]:
                relative = relative_dir + '/' + record['id'] + '/' + name
                require(record[key] == manifest['files_sha256'][relative] == sha(prior / relative),
                        'Prior saved pixel/record is not the retained manifest artifact')
                files[relative] = record[key]
                if update == 50:
                    value = by_id[record['id']]['states']['retained_FP32_update50']
                    require(value[key] == record[key] and Path(value['raw' if name == 'raw.npy' else 'native_png']).resolve() ==
                            (prior / relative).resolve(), 'Pixel audit compared a different prior artifact')
    require(len(files) == 1626, 'Exact two metrics +1624 prior RAW/PNG artifact census required')
    result = {'complete': True, 'prior_protocol_sha256': PARENT_SHA, 'prior_manifest_path': str(source),
        'prior_manifest_sha256': MANIFEST_SHA, 'compared_prior_files_freshly_hashed': len(files),
        'all_prior_outputs_metrics_match_historical_manifest': True,
        'all406_new_baseline_RAW_PNG_exact_independently_checked_by_pixel_audit': audit['all406_baseline_RAW_PNG_exact'],
        'bound_metrics_audit_sha256': sha(metrics_root / 'metrics_audit.json'),
        'bound_visual_manifest_sha256': sha(metrics_root / 'visual_manifest.json'),
        'files_sha256': files, 'source_sha256': sha(__file__),
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'model_qualified': False}
    write(output, result)
    return {'complete': True, 'files': len(files), 'receipt_sha256': sha(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('prior', 'metrics-root', 'output'):
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args(); print(json.dumps(close(a.prior, a.metrics_root, a.output), indent=2))
