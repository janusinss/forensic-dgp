"""Analyze independently verified saved derivatives and raw pixels; no differentiation."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_analysis'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    import numpy as np
    from PIL import Image
    start = time.monotonic(); assert not OUT.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['VM_failure_retained']
    assert audit['members_verified'] == 756 and audit['CPU_replay']['cases_at_both_states'] == 200
    assert audit['optimizer_updates'] == audit['local_gradient_calls'] == 0
    bundle = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_vm'
    returned = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_return'
    p = read(bundle / 'protocol.json'); assert sha(bundle / 'protocol.json') == audit['protocol_sha256']
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    rows, hashes, matrices = [], {}, {}
    for state in [0, 50]:
        for cohort in p['cohorts']:
            folder = returned / f'outputs/state{state}_{cohort["name"]}'
            path = folder / 'gradient_components.npy'; hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            a = np.load(path, allow_pickle=False); assert a.shape == (7, 498627) and a.dtype == np.float64 and np.isfinite(a).all()
            matrices[(state, cohort['name'])] = a
    pixel_rows = []
    for cohort in p['cohorts']:
        for c in cohort['cases']:
            paths = [returned / f'outputs/state{s}_{cohort["name"]}/{c["id"]}.npy' for s in [0, 50]]
            for path in paths + [mixed / c['target'], mixed / c['observed']]: hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            baseline, stopped = [np.load(path, allow_pickle=False) for path in paths]
            with Image.open(mixed / c['target']) as im: target = np.asarray(im.convert('RGB'), dtype=np.float32) / np.float32(255)
            with Image.open(mixed / c['observed']) as im: mask = np.asarray(im) > 0
            b = float(np.square(baseline - target)[mask].mean()); s = float(np.square(stopped - target)[mask].mean())
            v = max((s - b) / max(b, 1e-5), 0)
            pixel_rows.append({'cohort': cohort['name'], 'id': c['id'], 'source': c['source'], 'profile': c['profile'],
                               'raw_baseline_MSE': b, 'raw_stopped50_MSE': s, 'relative_pixel_regression': v})
    for row in audit['cohort_gradient_analysis']:
        norm = row['preservation_gradient_norm'] / row['improvement_gradient_norm']
        rows.append({**row, 'preservation_to_improvement_norm_ratio': norm,
                     'whole_observed_detail_derivative_positive': row['directional_derivatives_along_negative_original_objective'][1] > 0})
    positive = [r for r in pixel_rows if r['relative_pixel_regression'] > 0]
    assert sum(r['cohort'] == 'exposed' for r in positive) == 0
    assert sum(r['cohort'] == 'unexposed' and r['profile'] == 'clear' for r in positive) == 4
    assert sum(r['cohort'] == 'unexposed' and r['profile'] != 'clear' for r in positive) == 1
    cosines = {}
    for state in [0, 50]:
        a, b = [matrices[(state, label)][:3].sum(0) for label in ['exposed', 'unexposed']]
        cosines[str(state)] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    report = {'complete': True, 'date': '2026-10-07', 'checker_sha256': sha(Path(__file__)),
              'independent_audit_sha256': sha(audit_path), 'protocol_sha256': audit['protocol_sha256'],
              'gradient_analysis': rows, 'exposed_unexposed_improvement_cosines': cosines,
              'all_selected12_improvement_gradients_nonzero': all(v > 0 for r in rows for v in r['selected12_improvement_norms'].values()),
              'raw_pixel_rows': pixel_rows, 'pixel_regressions': positive,
              'maximum_clear_relative_pixel_regression': max(r['relative_pixel_regression'] for r in positive if r['profile'] == 'clear'),
              'source_bindings_sha256': hashes,
              'interpretation': 'Initial improvement gradients are active and reasonably coherent. At stopped50 the full-objective plain-gradient direction raises whole-observed detail error on both cohorts, while preservation terms respond to measured clear-control drift. This does not reconstruct AdamW or establish a unique historical cause.',
              'next_hypothesis': 'Keep each reference clear control and its four degraded views in the same batch, retaining all781 TRAIN references and all original decoder/loss/optimizer/gate settings. This tests paired anchoring and profile balance rather than weakening preservation penalties.',
              'raw_pixel_MSE_is_not_the_delivered_PNG_gate': True, 'unexposed_TRAIN_is_not_held_out': True,
              'native_or_reserved_used': False, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
              'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    OUT.mkdir()
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'active_selected_tensors': 12, 'unexposed_pixel_regressions': len(positive),
                      'clear_pixel_regressions': 4, 'maximum_clear_relative_pixel_regression': report['maximum_clear_relative_pixel_regression'],
                      'exposed_unexposed_improvement_cosines': cosines, 'neural_or_gradient_calls': 0}, indent=2))


if __name__ == '__main__': main()
