"""Saved-output bank ablation and user feedback; no model or gradient calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_analysis'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    receipt = ANALYSIS/'saved_bank_attribution.json'
    assert not receipt.exists()
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)

    for name in ['independent_audit.json', 'matched_guard_comparison.json', 'visual_review.json']:
        path = ANALYSIS/name
        assert read(path)['complete']
        bind(path)
    audit = read(ANALYSIS/'independent_audit.json')
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7910
    assert audit['native_compositions_checked'] == 72
    assert audit['local_optimizer_updates'] == audit['local_gradient_queries'] == 0
    p = read(CURRENT/'protocol.json')
    bind(CURRENT/'protocol.json')
    out = CURRENT/'outputs/bank_comparison_v1'
    metrics = {}
    for label in ['baseline', 'B50', 'B50_bank_disabled']:
        path = out/label/'metrics.json'
        bind(path)
        metrics[label] = {r['id']: r for r in read(path)['rows']}
    ids = set(metrics['B50_bank_disabled'])
    assert ids == set(p['preview_case_ids']) and len(ids) == 100
    cases = [c for c in p['cases'] if c['id'] in ids]
    assert len(cases) == 100
    exposed = {p['cases'][i]['source_person_or_reference'] for batch in p['schedule'] for i in batch}
    sys.path.insert(0, str(CURRENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    shifts = []
    for begin in range(0, len(p['cases']), 5):
        block = p['cases'][begin:begin+5]
        if block[0]['id'] not in ids:
            continue
        assert all(c['id'] in ids for c in block)
        block_ids = [c['id'] for c in block]
        base, _ = unpack(out/'baseline/packs'/f'b{begin//5:03d}.npz', block_ids)
        enabled, _ = unpack(out/'B50/packs'/f'b{begin//5:03d}.npz', block_ids, baseline=base)
        disabled, _ = unpack(out/'B50_bank_disabled/packs'/f'b{begin//5:03d}.npz', block_ids, baseline=base)
        for i, case in enumerate(block):
            assert np.isfinite(enabled[i]).all() and np.isfinite(disabled[i]).all()
            delta = enabled[i].astype(np.float64)-disabled[i].astype(np.float64)
            shifts.append(dict(id=case['id'], mean_RGB_bank_delta=delta.mean(axis=(0, 1)).tolist(),
                               bank_delta_RMS=float(np.sqrt(np.square(delta).mean()))))
    assert len(shifts) == 100
    groups = {}
    sources = sorted({c['source'] for c in cases})
    for source in ['all']+sources:
        for profile in ['degraded']+[c['profile'] for c in p['cases'][:5]]:
            chosen = [c['id'] for c in cases if (source == 'all' or c['source'] == source)
                      and (c['profile'] != 'clear' if profile == 'degraded' else c['profile'] == profile)]
            assert chosen
            stages = {}
            for stage in ['raw', 'png']:
                values = {label: {key: float(np.mean([rows[i][stage][key] for i in chosen])) for key in
                    ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE']}
                    for label, rows in metrics.items()}
                before = values['baseline']
                stages[stage] = dict(values=values, relative_detail_gain_from_original={label:
                    1-values[label]['landmark_high_frequency_MSE']/before['landmark_high_frequency_MSE']
                    for label in ['B50', 'B50_bank_disabled']}, enabled_minus_disabled={key:
                    values['B50'][key]-values['B50_bank_disabled'][key] for key in before})
            groups[source+'|'+profile] = dict(cases=len(chosen), stages=stages)
    native = []
    assert len(p['native_development']) == 24
    for case in p['native_development']:
        arrays = []
        for label in ['B50', 'B50_bank_disabled']:
            path = out/label/'native'/(case['id']+'.npy')
            bind(path)
            a = np.load(path, allow_pickle=False)
            assert a.shape == (256, 256, 3) and a.dtype == np.float32 and np.isfinite(a).all()
            arrays.append(a)
        delta = arrays[0].astype(np.float64)-arrays[1].astype(np.float64)
        native.append(dict(id=case['id'], mean_RGB_bank_delta=delta.mean(axis=(0, 1)).tolist(),
                           bank_delta_RMS=float(np.sqrt(np.square(delta).mean())), unpaired=True))
    sheet = ANALYSIS/'visual_review/native_00.png'
    bind(sheet)
    feedback = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        sheet=sheet.relative_to(ROOT).as_posix(), sheet_sha256=sha(sheet),
        question_comparison='brightness guard fifth column versus original DGP second column',
        user_verbatim='the fifth column is much better, but id like you to do the best approach here',
        scope='User development feedback on the fixed native comparison; not independent final review',
        comparison_against_B_r4_not_claimed=True, numerical_gates_unchanged=True,
        no_hidden_identity_or_Zamboanga_performance_claim=True)
    write(ANALYSIS/'user_development_feedback.json', feedback)
    bind(ANALYSIS/'user_development_feedback.json')
    for name, expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == expected
    for name, expected in bindings.items():
        assert sha(ROOT/name) == expected
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        evidence_files_sha256=bindings, paired_TRAIN_ablation_cases=100,
        preview_reference_count=20, source_case_counts=dict(Counter(c['source'] for c in cases)),
        exposed_TRAIN_reference_count=len({c['source_person_or_reference'] for c in cases}&exposed),
        native_DEV_unpaired_cases=24, groups=groups, paired_bank_deltas=shifts, native_bank_deltas=native,
        ablation='Disable bank only at the rejected jointly trained B50 state; decoder weights unchanged',
        not_an_independently_trained_decoder_control=True, no_full_3905_case_disabled_audit_claim=True,
        not_a_population_or_final_identity_claim=True, useful_DEV_and_final_review_pending=True,
        local_model_forwards=0, local_gradients=0, local_backwards=0, local_optimizer_updates=0,
        failed_stop_retained=True, automatic_training_or_promotion=False, model_qualified=False, goal_complete=False))
    print(dict(complete=True, paired_ablation_cases=100, native_ablation_cases=24, local_training_calls=0))


if __name__ == '__main__':
    main()
