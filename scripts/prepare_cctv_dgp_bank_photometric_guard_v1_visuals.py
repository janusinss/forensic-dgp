"""Bind matched input/preview evidence while the full quality audit runs.

This is a preview-only preparation receipt, never a full metric/state audit or
model qualification. It permits reviewing already verified saved previews.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
OLD = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_analysis'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = ANALYSIS/'visual_input_match.json'
    assert not receipt.exists()
    bindings = {}
    for root, analysis in [(CURRENT, ANALYSIS), (OLD, ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis')]:
        for name in ['import.json', 'execution_audit.json', 'preview_integrity_audit.json']:
            path = analysis/name
            row = read(path)
            assert row['complete'] and row['protocol_sha256'] == sha(root/'protocol.json')
            bindings[str(path.resolve())] = sha(path)
        preview = read(analysis/'preview_integrity_audit.json')
        assert preview['saved_paired_preview_compositions_checked'] == 300
        assert preview['native_compositions_checked'] == 72
    before, p = read(OLD/'protocol.json'), read(CURRENT/'protocol.json')
    common = ['cases', 'references', 'schedule', 'native_development', 'preview_case_ids',
              'canonical_target_RGB_sha256', 'original_checkpoint_sha256', 'original_state',
              'recognizer_state', 'standalone_prior_sha256', 'mean_style_sha256',
              'scientific_thresholds', 'normalizers', 'optimizer', 'budgets', 'visual_review_plan']
    for name in common:
        assert p[name] == before[name], name
    for name, value in p['assets_sha256'].items():
        if name.startswith(('data/', 'weights/', 'initializers/')):
            assert before['assets_sha256'][name] == value
    old_out, out = [root/'outputs/bank_comparison_v1' for root in [OLD, CURRENT]]
    old_rows, rows = [read(folder/'baseline/metrics.json')['rows'] for folder in [old_out, out]]
    assert len(rows) == len(old_rows) == 3905
    assert [(r['id'], r['raw_RGB_float32_sha256'], r['PNG_RGB_sha256']) for r in rows] == [
        (r['id'], r['raw_RGB_float32_sha256'], r['PNG_RGB_sha256']) for r in old_rows]
    paired, native = 0, 0
    for label, ids, relative in [('paired', p['preview_case_ids'], 'previews'),
                                 ('native', [c['id'] for c in p['native_development']], 'native')]:
        for id_ in ids:
            paths = [folder/'baseline'/relative/(id_+'.png') for folder in [old_out, out]]
            assert sha(paths[0]) == sha(paths[1])
            for path in paths:
                bindings[str(path.resolve())] = sha(path)
            if label == 'paired':
                paired += 1
            else:
                native += 1
                arrays = [np.load(path.with_suffix('.npy'), allow_pickle=False) for path in paths]
                assert np.array_equal(arrays[0], arrays[1])
    assert paired == 100 and native == 24
    for name, value in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == value
    result = dict(complete=True, checker_sha256=sha(Path(__file__)), compared_receipt_bindings=bindings,
        protocol_A_sha256=sha(OLD/'protocol.json'), protocol_B_sha256=sha(CURRENT/'protocol.json'),
        scope='matched inputs and independently checked planned previews only', matched_protocol_fields=common,
        matching_baseline_metadata_case_hashes=3905, matching_baseline_preview_PNGs=100,
        unchanged_native_baseline_cases=24, full_781_baseline_pack_comparison_pending=True,
        full_metric_and_state_audit_pending=True, local_neural_gradient_or_optimizer_calls=0,
        model_qualified=False, app_promotion=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, paired_previews=paired, native_baselines=native, full_quality_audit_pending=True), flush=True)


if __name__ == '__main__':
    main()
