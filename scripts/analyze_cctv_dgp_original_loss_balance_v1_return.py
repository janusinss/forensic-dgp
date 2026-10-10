"""Saved-array comparison and failure attribution, without neural or gradient calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm'
RETURN = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm_return'
PARENT = ROOT / 'outputs/cctv_dgp_original_feature_probe_v1_vm_return'
OUT = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1'
PIN = '8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    destination = OUT / 'failure_attribution.json'
    assert not destination.exists(), 'Retain any prior analysis'
    assert sha(BUNDLE / 'protocol.json') == PIN
    p = read(BUNDLE / 'protocol.json')
    audit_path = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['protocol_sha256'] == PIN
    assert audit['all36_stored_row_comparisons_and_failure_decisions_exact']
    parent_audit_path = ROOT / 'outputs/cctv_dgp_original_feature_probe_v1_independent_audit_r2.json'
    parent_audit = read(parent_audit_path)
    assert parent_audit['complete']
    oldp = read(PARENT / 'protocol.json')
    assert p['cases'] == oldp['cases'] and p['cohorts'] == oldp['cohorts']
    assert p['references'] == oldp['references']
    result = read(RETURN / 'outputs/results.json')
    parent_result = read(PARENT / 'outputs/results.json')
    assert result['protocol_sha256'] == PIN
    decoder = np.zeros(1996035, bool)
    for desc in p['parameter_layout']:
        if desc['partition'] == 'decoder_control':
            decoder[desc['start']:desc['end']] = True
    baseline = []
    for case in p['cases']:
        cid = case['id']
        current = RETURN / 'outputs/baseline' / (cid + '.npy')
        old = PARENT / 'outputs/baseline' / (cid + '.npy')
        a, b = np.load(current, allow_pickle=False), np.load(old, allow_pickle=False)
        baseline.append({'id': cid, 'raw_array_exact': bool(np.array_equal(a, b)),
            'raw_maximum_error': float(np.abs(a - b).max()),
            'PNG_encoded_bytes_exact': (RETURN / 'outputs/baseline' / (cid + '.png')).read_bytes()
                == (PARENT / 'outputs/baseline' / (cid + '.png')).read_bytes()})
    rows = []
    for trial in result['trial_summaries']:
        comparisons = {}
        for cohort, stages in trial['comparisons'].items():
            comparisons[cohort] = {}
            for stage, gate in stages.items():
                comparisons[cohort][stage] = {
                    'structure_gain_percent': gate['relative_feature_gain'] * 100,
                    'preservation_failure_records': len(gate['preservation_failures']),
                    'failure_metric_counts': dict(Counter(item['metric'] for item in gate['preservation_failures'])),
                    'failure_membership_and_values': gate['preservation_failures'],
                    'source_structure_gains_percent': {key: value * 100 for key, value in gate['source_feature_gains'].items()},
                    'brightness_gain_fraction': gate['brightness_gain_fraction'],
                    'pass': gate['pass']}
        row = {'variant': trial['variant'], 'comparisons': comparisons}
        if trial['scope'] in ['balanced_1', 'balanced_2']:
            old_label = 'decoder_control_' + format(trial['relative_fraction'], '.0e').replace('-', 'm')
            old_trial = next(item for item in parent_result['trial_summaries'] if item['variant'] == old_label)
            vector = np.load(RETURN / 'outputs' / (trial['variant'] + '_parameters.npy'), allow_pickle=False)
            oldvector = np.load(PARENT / 'outputs' / (old_label + '_parameters.npy'), allow_pickle=False)
            row['matched_decoder_control'] = {
                'variant': old_label,
                'decoder_tensors_exact': bool(np.array_equal(vector[decoder], oldvector[decoder])),
                'decoder_maximum_error': float(np.abs(vector[decoder] - oldvector[decoder]).max()),
                'old_structure_and_failures': {cohort: {stage: {
                    'structure_gain_percent': gate['relative_feature_gain'] * 100,
                    'preservation_failure_records': len(gate['preservation_failures']),
                    'pass': gate['pass']} for stage, gate in stages.items()}
                    for cohort, stages in old_trial['comparisons'].items()}}
        rows.append(row)
    assert len(rows) == 9
    assert all(not gate['pass'] for row in rows for stages in row['comparisons'].values() for gate in stages.values())
    selected = next(row for row in rows if row['variant'] == 'balanced_2_1em03')
    receipt = {'complete': True, 'protocol_sha256': PIN,
        'results_sha256': sha(RETURN / 'outputs/results.json'),
        'parent_results_sha256': sha(PARENT / 'outputs/results.json'),
        'independent_audit_sha256': sha(audit_path),
        'parent_independent_audit_sha256': sha(parent_audit_path),
        'cases_references_and_cohorts_identical': True,
        'all100_baseline_raw_arrays_exact': all(row['raw_array_exact'] for row in baseline),
        'all100_baseline_encoded_PNGs_exact': all(row['PNG_encoded_bytes_exact'] for row in baseline),
        'maximum_baseline_raw_error': max(row['raw_maximum_error'] for row in baseline),
        'baseline_comparisons': baseline, 'all_nine_trials': rows,
        'detailed_balance2_1e3': selected,
        'failure_record_count_meaning': 'Metric/group records, not identification-error counts',
        'interpretation': 'Mean-gradient coordination reduces some failures but does not preserve every source/profile or clear face. All nine directions remain unqualified.',
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'model_qualification': False, 'goal_complete': False,
        'analysis_script_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - start}
    OUT.mkdir(exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'complete': True, 'all_baselines_exact': receipt['all100_baseline_raw_arrays_exact'],
        'decoder_controls_exact': all(row['matched_decoder_control']['decoder_tensors_exact']
            for row in rows if 'matched_decoder_control' in row),
        'all_nine_trials_unqualified': True, 'seconds': receipt['seconds']}, flush=True)


if __name__ == '__main__':
    main()
