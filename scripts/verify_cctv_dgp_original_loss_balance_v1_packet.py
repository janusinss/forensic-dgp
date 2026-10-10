"""Independent transfer/readback and frozen initial replay; no local trial or derivative."""
import ast
import hashlib
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_vm'
PREP = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_preparation'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_original_loss_balance_v1_contract import NAME, STEM, sha, read, write, verified_assets


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def main():
    start = time.monotonic(); prepared = read(PREP/'prepared.json'); pin = prepared['protocol_sha256']
    p = verified_assets(BUNDLE, pin); archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().split() == [sha(archive), archive.name]
    for n, d in p['local_sources_sha256'].items(): assert sha(ROOT/n) == d, n
    for n, origin in p['copied_source_mapping'].items(): assert sha(BUNDLE/n) == sha(ROOT/origin)
    members = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for m in tar.getmembers():
            assert m.isfile() and not m.issym() and not m.islnk()
            parts = m.name.split('/'); assert parts[0] == NAME and all(q not in ['', '.', '..'] for q in parts)
            assert '\\' not in m.name and ':' not in m.name
            n = '/'.join(parts[1:]); assert n not in members
            h = hashlib.sha256()
            with tar.extractfile(m) as f:
                for block in iter(lambda: f.read(1024**2), b''): h.update(block)
            members[n] = h.hexdigest(); assert sha(BUNDLE/n) == members[n]
    assert set(members) == set(p['assets_sha256']) | {'protocol.json'}
    parent = read(BUNDLE/'evidence/parent_protocol.json'); previous = read(BUNDLE/'evidence/parent_results.json')
    audit = read(BUNDLE/'evidence/parent_independent_audit_r2.json'); visual = read(BUNDLE/'evidence/parent_visual_review.json')
    assert sha(BUNDLE/'evidence/parent_protocol.json') == p['parent_protocol_sha256']
    assert audit['complete'] and previous['complete'] and visual['complete']
    assert audit['archive_sha256'] == p['parent_archive_sha256'] and visual['unique_model_outputs_reviewed'] == 1000
    assert audit['all36_stored_row_comparisons_and_failure_decisions_exact'] and not audit['scientific_thresholds_changed']
    for key in ['cases', 'references', 'parameter_layout', 'cohorts', 'terms_and_overlap', 'scientific_thresholds', 'CPU_replay_tolerances']:
        assert p[key] == parent[key]
    trees = {}
    for file in BUNDLE.rglob('*.py'): trees[file.relative_to(BUNDLE).as_posix()] = ast.parse(file.read_text(encoding='utf-8'), feature_version=(3, 10))
    worker_name = 'scripts/cctv_dgp_original_loss_balance_v1_vm.py'; source = (BUNDLE/worker_name).read_text()
    assert 'torch.optim' not in source and 'torch.autograd' not in source and 'requires_grad_(True)' not in source
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ['backward', 'grad'] for n in ast.walk(trees[worker_name]))
    assert 'initial.astype(np.float64)+planned' in source and 'candidate.assign_trial(initial); frozen(True)' in source
    candidate_source = (BUNDLE/'cctv_dgp_original_loss_balance_v1_candidate.py').read_text()
    assert 'requires_grad_(True)' not in candidate_source and NAME in candidate_source
    shell = (BUNDLE/'scripts/run_loss_balance.sh').read_text()
    assert '\r' not in shell and NAME in shell and '1230s' in shell and '330s' in shell and 'PIPESTATUS[0]' in shell
    # Rebuild proposed vectors independently of the worker direction helper, without assigning a model.
    with np.load(BUNDLE/'evidence/aggregate_gradients.npz', allow_pickle=False) as arrays: gradients = {k: arrays[k].copy() for k in p['terms']}
    initial = np.load(BUNDLE/'evidence/initial_parameters.npy', allow_pickle=False)
    dec = np.zeros(len(initial), bool); feature = np.zeros(len(initial), bool)
    for r in p['parameter_layout']:
        (dec if r['partition'] == 'decoder_control' else feature)[r['start']:r['end']] = True
    d = np.zeros(len(initial), np.float64); f = d.copy()
    d[dec] = -gradients[p['terms'][0]][dec]; f[feature] = -gradients[p['terms'][2]][feature]
    d /= np.linalg.norm(d); f /= np.linalg.norm(f)
    ratio = float(gradients[p['terms'][2]]@d)/(-float(gradients[p['terms'][2]]@f))
    assert ratio == p['measured_balance_ratio']
    from cctv_dgp_original_loss_balance_v1_contract import directions
    reference_directions, reference_ratio, _ = directions(p['parameter_layout'], gradients)
    independent = {'feature_identity': f, 'balanced_1': d+ratio*f, 'balanced_2': d+2*ratio*f}
    assert reference_ratio == ratio
    with np.load(BUNDLE/'evidence/canonical_directions.npz', allow_pickle=False) as saved:
        for name, direction in independent.items(): assert np.array_equal(saved[name], direction)
    formulas = []
    dec_weight = float(np.linalg.norm(initial.astype(np.float64)[dec]))
    assert dec_weight == p['initial_decoder_weight_L2']
    for mode, direction in independent.items():
        assert np.array_equal(direction, reference_directions[mode])
        for fraction in p['relative_displacement_fractions']:
            trial = (initial.astype(np.float64)+fraction*dec_weight*direction).astype(np.float32)
            assert np.isfinite(trial).all() and (mode != 'feature_identity' or np.array_equal(trial[dec], initial[dec]))
            delta = trial.astype(np.float64)-initial.astype(np.float64)
            formulas.append({'mode': mode, 'fraction': fraction, 'actual_displacement_L2': float(np.linalg.norm(delta)),
                'gradient_dot_displacement': {k: float(v@delta) for k, v in gradients.items()}, 'model_assignment_local': False})
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_original_loss_balance_v1_candidate import OriginalLossBalanceProbe
    from cctv_dgp_pilot import state_hash, buffer_hash
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    candidate = OriginalLossBalanceProbe(original.net, p['parameter_layout'])
    assert np.array_equal(candidate.vector(), initial)
    before = (state_hash(original.net), state_hash(candidate.net), buffer_hash(candidate.net))
    denied = []
    for name, call in [('finite_trial_enable', lambda: candidate.enable_finite_trials(BUNDLE)),
                       ('trial_assignment', lambda: candidate.assign_trial(initial))]:
        try: call()
        except AssertionError: denied.append(name)
        else: raise AssertionError('A local VM-only operation was allowed')
    worst = 0.; count = 0
    previous_out = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm_return/outputs/baseline'
    with torch.no_grad():
        for begin in range(0, 100, 5):
            assert time.monotonic()-start <= 600
            cases = p['cases'][begin:begin+5]
            x = torch.from_numpy(np.stack([pixels(BUNDLE/c['input']) for c in cases]).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2)
            mask = torch.from_numpy(np.stack([pixels(BUNDLE/c['observed'], 'L') > 0 for c in cases]).astype(np.float32))[:, None]
            base = original(x).detach().clone(); result = candidate(x, mask, base)
            assert torch.equal(result, torch.where(mask.bool(), base, x)); count += len(cases)
            for slot, case in enumerate(cases):
                old = np.load(previous_out/(case['id']+'.npy'), allow_pickle=False)
                worst = max(worst, float(np.abs(result[slot].permute(1, 2, 0).numpy()-old).max()))
    assert worst <= 3e-6 and count == 100
    after = (state_hash(original.net), state_hash(candidate.net), buffer_hash(candidate.net)); assert before == after
    assert before[0] == before[1] == p['original_state']
    verified_assets(BUNDLE, pin)
    write(PREP/'independent_packet_audit.json', {'complete': True, 'protocol_sha256': pin,
        'archive_sha256': sha(archive), 'archive_members_verified': len(members), 'source_bindings_verified': len(p['local_sources_sha256']),
        'initial_CPU_parity_cases': count, 'maximum_parent_raw_error': worst, 'local_guards_denied': denied,
        'all_nine_direction_formulas': formulas, 'states_before': before, 'states_after': after,
        'Python310_AST_files': len(trees), 'local_neural_forward_batches': 40,
        'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'local_trial_assignments': 0,
        'VM_connections': 0, 'VM_diagnostic_launched': False, 'model_qualification': False,
        'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic()-start, 'goal_complete': False})
    print({'complete': True, 'initial_parity_cases': count, 'parent_raw_error': worst, 'VM_diagnostic_launched': False})


if __name__ == '__main__': main()
