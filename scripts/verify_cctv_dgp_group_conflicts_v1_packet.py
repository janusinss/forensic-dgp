"""Independent archive, source/guard, loss-forward and initial CPU replay checks."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm'
PREP = ROOT/'outputs/cctv_dgp_group_conflicts_v1_preparation'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_group_conflicts_v1_contract import NAME, STEM, sha, read, write, verified_assets


def pixels(path, mode='RGB'):
    with Image.open(path) as image: return np.asarray(image.convert(mode)).copy()


def main():
    started = time.monotonic(); prepared = read(PREP/'prepared.json'); pin = prepared['protocol_sha256']
    p = verified_assets(BUNDLE, pin); archive = ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().split() == [sha(archive), archive.name]
    for name, digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest, name
    for name, origin in p['copied_source_mapping'].items(): assert sha(BUNDLE/name) == sha(ROOT/origin), name
    members = {}
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            assert member.isfile() and not member.issym() and not member.islnk()
            parts = member.name.split('/'); assert parts[0] == NAME and all(q not in ['', '.', '..'] for q in parts)
            assert '\\' not in member.name and ':' not in member.name
            name = '/'.join(parts[1:]); assert name not in members
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
            members[name] = digest.hexdigest(); assert sha(BUNDLE/name) == members[name]
    assert set(members) == set(p['assets_sha256']) | {'protocol.json'}
    parent = read(BUNDLE/'evidence/parent_protocol.json')
    assert sha(BUNDLE/'evidence/parent_protocol.json') == p['parent_protocol_sha256']
    for key in ['cases', 'references', 'parameter_layout', 'cohorts', 'terms_and_overlap', 'scientific_thresholds', 'CPU_replay_tolerances']:
        assert p[key] == parent[key], key
    audit = read(BUNDLE/'evidence/parent_independent_audit.json'); visual = read(BUNDLE/'evidence/parent_visual_review.json')
    assert audit['complete'] and visual['complete'] and visual['unique_model_outputs_reviewed'] == 1000
    assert audit['all36_stored_row_comparisons_and_failure_decisions_exact'] and audit['archive_sha256'] == p['parent_archive_sha256']
    parity = read(BUNDLE/'evidence/forward_parity_r2.json'); failure = read(BUNDLE/'evidence/forward_parity_failure.json')
    assert parity['complete'] and parity['passed'] and parity['cases'] == 100 and not failure['passed']
    for key in ['neural_calls', 'gradient_queries', 'optimizer_updates']: assert parity[key] == 0
    assert parity['maximum_errors']['MSE'] <= 1e-12 and parity['maximum_errors']['SSIM'] <= 3e-5
    assert parity['maximum_errors']['structure'] <= 1e-10
    binding = read(BUNDLE/'evidence/forward_parity_source_binding.json')
    assert binding['current_loss_source_sha256'] == sha(BUNDLE/'cctv_dgp_group_conflicts_v1_losses.py')
    assert binding['corrected_forward_receipt_sha256'] == sha(BUNDLE/'evidence/forward_parity_r2.json')
    assert failure['failed_loss_source_sha256'] == sha(BUNDLE/'evidence/failed_losses.py.txt')
    assert failure['forward_parity_receipt_sha256'] == sha(BUNDLE/'evidence/forward_parity_failed_receipt.json')
    trees = {file.relative_to(BUNDLE).as_posix(): ast.parse(file.read_text(encoding='utf-8'), feature_version=(3, 10))
             for file in BUNDLE.rglob('*.py')}
    worker = BUNDLE/'scripts/cctv_dgp_group_conflicts_v1_vm.py'; source = worker.read_text()
    assert 'torch.optim' not in source and '.backward(' not in source
    assert source.count('torch.autograd.grad(') == 1 and 'retain_graph=slot < 15' in source
    assert 'if direction is not None:' in source and 'candidate.assign_trial(initial); frozen(True)' in source
    assert 'np.zeros((32, 1996035), np.float64)' in source and "progress['gradient_queries'] == 160" in source
    candidate_source = (BUNDLE/'cctv_dgp_group_conflicts_v1_candidate.py').read_text()
    original_source = (ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm/cctv_dgp_original_feature_probe_v1_candidate.py').read_text()
    assert candidate_source == original_source.replace('original_feature_probe_v1', 'group_conflicts_v1')
    shell = (BUNDLE/'scripts/run_group_conflicts.sh').read_text()
    assert '\r' not in shell and NAME in shell and '1530s' in shell and '330s' in shell and 'PIPESTATUS[0]' in shell
    denied_worker = []
    for mode in ['verify-transfer', 'run', 'export', 'record-supervision']:
        child = subprocess.run([sys.executable, '-B', str(worker), '--root', str(BUNDLE), '--protocol-sha', pin, '--'+mode],
            capture_output=True, text=True, timeout=15)
        assert child.returncode != 0 and 'AssertionError' in child.stderr
        denied_worker.append(mode)
    from cctv_dgp_group_conflicts_v1_math import common_direction
    fixtures = [(np.array([[1., 0.], [0., 1.]], np.float64), True),
        (np.array([[1., 0.], [-1., 0.]], np.float64), False),
        (np.array([[1., 1.], [1., 0.], [0., 1.]], np.float64), True),
        (np.array([[0., 0.], [1., 0.]], np.float64), False)]
    for matrix, expected in fixtures:
        direction, certificate = common_direction(matrix)
        assert certificate['common_direction_found'] == expected and (direction is not None) == expected
        if expected: assert np.all(matrix@direction < 0)
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_group_conflicts_v1_candidate import OriginalFeatureProbe
    from cctv_dgp_pilot import state_hash, buffer_hash
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    candidate = OriginalFeatureProbe(original.net, p['parameter_layout'])
    initial = np.load(BUNDLE/'evidence/initial_parameters.npy', allow_pickle=False)
    assert np.array_equal(candidate.vector(), initial)
    before = (state_hash(original.net), state_hash(candidate.net), buffer_hash(candidate.net))
    denied_candidate = []
    for name, call in [('gradient_enable', lambda: candidate.enable_diagnostic_gradients(BUNDLE)),
                       ('trial_assignment', lambda: candidate.assign_trial(initial))]:
        try: call()
        except AssertionError: denied_candidate.append(name)
        else: raise AssertionError('A local VM-only operation was allowed')
    chosen = []
    for source_name in sorted({c['source'] for c in p['cases']}):
        chosen.extend([c for c in p['cases'] if c['source'] == source_name][:2])
    worst = 0.
    for case in chosen:
        assert time.monotonic()-started <= 180
        x = torch.from_numpy(pixels(BUNDLE/case['input']).astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
        mask = torch.from_numpy((pixels(BUNDLE/case['observed'], 'L') > 0).astype(np.float32))[None, None]
        with torch.no_grad():
            base = original(x).detach().clone(); result = candidate(x, mask, base)
            assert torch.equal(result, torch.where(mask.bool(), base, x))
        prior = np.load(ROOT/'outputs/cctv_dgp_original_loss_balance_v1_vm_return/outputs/baseline'/(case['id']+'.npy'), allow_pickle=False)
        worst = max(worst, float(np.abs(result[0].permute(1, 2, 0).numpy()-prior).max()))
    try:
        with torch.enable_grad(): candidate(x, mask, base)
    except AssertionError: denied_candidate.append('gradient_forward')
    else: raise AssertionError('A local differentiable model forward was allowed')
    assert worst <= 3e-6 and len(chosen) == 4
    after = (state_hash(original.net), state_hash(candidate.net), buffer_hash(candidate.net)); assert before == after
    assert before[0] == before[1] == p['original_state']
    assert all(v.grad is None and not v.requires_grad for v in candidate.parameters())
    verified_assets(BUNDLE, pin)
    write(PREP/'independent_packet_audit.json', {'complete': True, 'protocol_sha256': pin,
        'archive_sha256': sha(archive), 'archive_members_verified': len(members), 'source_bindings_verified': len(p['local_sources_sha256']),
        'initial_CPU_parity_cases': 4, 'maximum_parent_raw_error': worst, 'local_worker_guards_denied': denied_worker,
        'local_candidate_guards_denied': denied_candidate, 'pure_math_fixtures': 4, 'cached_loss_parity_cases': 100,
        'loss_parity_failure_preserved': True, 'states_before': before, 'states_after': after,
        'Python310_AST_files': len(trees), 'local_neural_forward_calls': 8,
        'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'local_trial_assignments': 0,
        'VM_connections': 0, 'VM_diagnostic_launched': False, 'model_qualification': False,
        'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic()-started, 'goal_complete': False})
    print({'complete': True, 'initial_parity_cases': 4, 'parent_raw_error': worst, 'VM_diagnostic_launched': False})


if __name__ == '__main__': main()
