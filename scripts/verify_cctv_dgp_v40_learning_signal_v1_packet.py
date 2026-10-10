"""Independent packet, metadata selection, source equivalence and rejection checks."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm'
PREP = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_preparation'
PARENT = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def main():
    start = time.monotonic(); import numpy as np
    assert not (PREP/'independent_packet_audit.json').exists()
    p = read(BUNDLE/'protocol.json'); prepared = read(PREP/'prepared.json'); pin = sha(BUNDLE/'protocol.json')
    assert pin == prepared['protocol_sha256']
    archive = ROOT/'outputs/cctv-dgp-v40-learning-signal-v1-execution.tar.gz'
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [prepared['archive_sha256'], archive.name]
    expected = {'protocol.json': pin, **p['assets_sha256']}; seen = set(); total = 0
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == len(expected) == prepared['files']
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(BUNDLE.name+'/')
            name = member.name[len(BUNDLE.name)+1:]; assert name in expected and name not in seen and '\\' not in name and ':' not in name
            with tar.extractfile(member) as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == expected[name]; seen.add(name); total += member.size
    assert seen == set(expected) and total < 512*1024**2
    for name, value in p['assets_sha256'].items(): assert sha(BUNDLE/name) == value, name
    for name, value in p['local_basis_sha256'].items(): assert sha(ROOT/name) == value, name
    for name, source in p['copied_source_mapping'].items(): assert sha(BUNDLE/name) == sha(ROOT/source), name
    old = read(PARENT/'protocol.json'); schedule = read(PARENT/'schedule.json')['batches'][:50]
    old_cases = {c['id']: c for c in old['cases']}; old_refs = {r['id']: r for r in old['references']}
    fixed = old['preview_case_ids']; exposed = {old['cases'][i]['id'] for batch in schedule for i in batch}
    assert p['cohorts']['not_yet_optimized'] == fixed and not set(fixed) & exposed
    # Independent selection by source lists, then original reference order.
    order = [old['cases'][batch[0]]['source_person_or_reference'] for batch in schedule]
    per_source = {s: [rid for rid in order if old_refs[rid]['source'] == s][:5] for s in sorted({c['source'] for c in old['cases']})}
    chosen = [rid for rid in order if any(rid in ids for ids in per_source.values())]
    assert p['first50_reference_order'] == order and p['selected_optimized_references'] == chosen and len(chosen) == 10
    optimized = [old['cases'][i]['id'] for batch in schedule if old['cases'][batch[0]]['source_person_or_reference'] in chosen for i in batch]
    assert p['cohorts']['optimized'] == optimized and len(optimized) == len(set(optimized)) == 50
    assert [c['id'] for c in p['cases']] == fixed+optimized and len(p['cases']) == 100
    for c in p['cases']:
        actual = {k: v for k, v in c.items() if k not in ['historical_initial_raw', 'historical_stopped_raw']}
        assert actual == old_cases[c['id']] and c['role'] == 'train'
    assert p['references'] == [old_refs[rid] for rid in dict.fromkeys(c['source_person_or_reference'] for c in p['cases'])]
    for begin in range(0, 100, 5):
        batch = p['cases'][begin:begin+5]
        assert len({c['source_person_or_reference'] for c in batch}) == 1
        assert [c['profile'] for c in batch] == ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
    assert p['parameter_layout'] == old['parameter_layout'] and p['terms'] == old['terms'] and p['retained_capacity_gates'] == old['retained_capacity_gates']
    assert p['expected_initial_states'] == old['initial_states']
    analysis = read(ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis/results.json')
    assert p['decoder_states'] == {'initial': analysis['initial_state'], 'stopped50': analysis['stopped_state']}
    assert sha(BUNDLE/'saved_parameter_delta.npy') == analysis['delta_sha256']
    assert read(BUNDLE/'historical_cohort_loss_setup.json') == read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_return/outputs/cohort_loss_setup.json')
    candidate = (BUNDLE/'cctv_dgp_v40_learning_signal_v1_decoder.py').read_text(encoding='utf-8')
    inverse = candidate.replace(BUNDLE.name, PARENT.name).replace('Distinct endpoint diagnostic VM root', 'Distinct V40 VM root').replace('SpatialDGPCandidateV40LearningSignalV1', 'SpatialDGPCandidateV40')
    original = (PARENT/'cctv_dgp_spatial_decoder_v40.py').read_text(encoding='utf-8')
    assert inverse == original and ast.dump(ast.parse(inverse), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False)
    for name in ['frozen_definitions.py', 'cctv_dgp_degraded_objective_v24.py', 'cctv_dgp_batchmatched_identity_v26.py']:
        assert (BUNDLE/name).read_bytes() == (PARENT/name).read_bytes()
    assert p['optimizer_updates'] == p['parameter_updates'] == p['epochs'] == p['backwards'] == 0 and p['component_gradient_calls'] == 280
    assert not p['automatic_follow_on'] and not p['new_learning_recipe_or_checkpoint'] and not p['native_or_reserved_used']
    worker_path = BUNDLE/'scripts/cctv_dgp_v40_learning_signal_v1_vm.py'; worker = module('endpoint_metadata_only', worker_path)
    assert worker.verify(BUNDLE, pin) == p
    scope_rejections = 0
    for platform_name, host, path in [('win32', 'forensic-dgp-thesis', BUNDLE), ('linux', 'wrong-vm', Path.home()/'forensic-dgp'/BUNDLE.name),
                                    ('linux', 'forensic-dgp-thesis', Path.home()/'forensic-dgp'/'wrong-root')]:
        with patch.object(worker.sys, 'platform', platform_name), patch.object(worker.platform, 'node', return_value=host):
            try: worker.scope(path.resolve())
            except AssertionError: scope_rejections += 1
            else: raise AssertionError('Incorrect platform/host/root accepted')
    with patch.object(worker.sys, 'platform', 'linux'), patch.object(worker.platform, 'node', return_value='forensic-dgp-thesis'):
        worker.scope((Path.home()/'forensic-dgp'/BUNDLE.name).resolve())
    before = set(q.relative_to(BUNDLE).as_posix() for q in BUNDLE.rglob('*') if q.is_file())
    result = subprocess.run([sys.executable, '-B', str(worker_path), '--root', str(BUNDLE), '--protocol-sha', pin, '--verify-transfer'], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'Existing Linux VM only' in result.stderr
    assert before == set(q.relative_to(BUNDLE).as_posix() for q in BUNDLE.rglob('*') if q.is_file())
    with (PREP/'Windows_pre_neural_rejection.txt').open('x', encoding='utf-8') as stream: stream.write(result.stdout+result.stderr)
    auditor_path = ROOT/'scripts/audit_cctv_dgp_v40_learning_signal_v1_return.py'; auditor = module('endpoint_prospective_auditor', auditor_path)
    assert worker.RETURNED_ASSETS == auditor.RETURNED_ASSETS and all(name in p['assets_sha256'] for name in auditor.RETURNED_ASSETS)
    def item(name, size=0, kind=tarfile.REGTYPE):
        value = tarfile.TarInfo(name); value.size = size; value.type = kind; return value
    prefix = auditor.PREFIX; auditor.safe_members([item(prefix+'protocol.json')], p)
    bad = [[item(prefix+'../protocol.json')], [item('/'+prefix+'protocol.json')], [item(prefix+'protocol.json', kind=tarfile.SYMTYPE)],
           [item(prefix+'protocol.json', kind=tarfile.LNKTYPE)], [item(prefix+'protocol.json'), item(prefix+'protocol.json')],
           [item(prefix+'C:protocol.json')], [item(prefix+'unknown.npy')], [item(prefix+'protocol.json', 8*1024**2+1)],
           [item(prefix+'outputs/baseline/'+c['id']+'.npy', 8*1024**2) for c in p['cases'][:65]]]
    unsafe = 0
    for members in bad:
        try: auditor.safe_members(members, p)
        except AssertionError: unsafe += 1
        else: raise AssertionError('Unsafe/oversized archive accepted')
    fixtures = []
    # Known arithmetic fixtures test corruption rejection, never actual gradients.
    with tempfile.TemporaryDirectory(prefix='endpoint-array-fixtures-', dir=PREP) as directory:
        folder = Path(directory).resolve(); assert folder.is_relative_to(PREP.resolve())
        summary = {'complete': True, 'terms': p['terms'], 'parameter_layout': p['parameter_layout'], 'states': {}, 'case_term_values': {}}
        delta = np.load(BUNDLE/'saved_parameter_delta.npy', allow_pickle=False)
        for state in ['initial', 'stopped50']:
            summary['states'][state] = {}; (folder/('outputs/'+state+'/gradients')).mkdir(parents=True)
            for cohort, ids in p['cohorts'].items():
                batches = []; total_matrix = np.zeros((7, 17952), np.float64); values = np.zeros(7, np.float64)
                for i in range(10):
                    matrix = np.repeat((np.arange(1, 8, dtype=np.float64)*.00001*(i+1))[:, None], 17952, 1)
                    per_case = np.ones((5, 7), np.float32)*np.asarray([.1, .025, .005, .001, .002, .003, .004], np.float32)
                    if state == 'initial': matrix[3:] = 0; per_case[:, 3:] = 0
                    v = (per_case.mean(0)/np.float32(10)).astype(np.float64)
                    path = folder/('outputs/'+state+'/gradients/'+cohort+'_batch'+str(i)+'.npy'); np.save(path, matrix, allow_pickle=False)
                    batches.append({'batch': i, 'ids': ids[i*5:i*5+5], 'values': v.tolist(), 'norms': np.linalg.norm(matrix, axis=1).tolist(), 'gradient_sha256': sha(path)})
                    for cid, vals in zip(ids[i*5:i*5+5], per_case): summary['case_term_values'][state+'/'+cid] = vals.tolist()
                    total_matrix += matrix; values += v
                path = folder/('outputs/'+state+'/'+cohort+'_gradient_components.npy'); np.save(path, total_matrix, allow_pickle=False)
                norms = np.linalg.norm(total_matrix, axis=1); direction = -total_matrix.sum(0); dn = np.linalg.norm(direction); products = total_matrix @ direction
                summary['states'][state][cohort] = {'batches': batches, 'values': values.tolist(), 'component_norms': norms.tolist(),
                    'component_gram': (total_matrix @ total_matrix.T).tolist(), 'gradient_sha256': sha(path),
                    'gradient_dot_actual50_weight_change': (total_matrix @ delta).tolist(), 'negative_total_direction_component_derivatives': products.tolist(),
                    'negative_total_direction_cosines': [None if n == 0 else float(d/(n*dn)) for n, d in zip(norms, products)],
                    'partitions': {r['name']: np.linalg.norm(total_matrix[:, r['start']:r['end']], axis=1).tolist() for r in p['parameter_layout']}}
        path = folder/'outputs/gradient_summary.json'; path.write_text(json.dumps(summary, allow_nan=False))
        with patch.object(auditor, 'OUT', folder):
            assert auditor.gradient_readback(p)['component_queries_readback'] == 280
            batch_path = folder/'outputs/initial/gradients/not_yet_optimized_batch0.npy'; original_array = np.load(batch_path, allow_pickle=False)
            for mode in ['nonfinite', 'wrong_dtype', 'wrong_shape', 'initial_preservation_nonzero', 'wrong_case_order', 'wrong_sum', 'wrong_dot', 'wrong_direction', 'wrong_partition']:
                changed = copy.deepcopy(summary); matrix = original_array.copy(); row = changed['states']['initial']['not_yet_optimized']
                if mode == 'nonfinite': matrix[0, 0] = np.nan
                elif mode == 'wrong_dtype': matrix = matrix.astype(np.float32)
                elif mode == 'wrong_shape': matrix = matrix[:, :-1]
                elif mode == 'initial_preservation_nonzero': matrix[3, 0] = .1
                elif mode == 'wrong_case_order': row['batches'][0]['ids'].reverse()
                elif mode == 'wrong_sum': row['values'][0] += .1
                elif mode == 'wrong_dot': row['gradient_dot_actual50_weight_change'][0] += .1
                elif mode == 'wrong_direction': row['negative_total_direction_component_derivatives'][0] += .1
                elif mode == 'wrong_partition': row['partitions'][p['parameter_layout'][0]['name']][0] += .1
                np.save(batch_path, matrix, allow_pickle=False); row['batches'][0]['gradient_sha256'] = sha(batch_path)
                path.write_text(json.dumps(changed, allow_nan=False))
                try: auditor.gradient_readback(p)
                except AssertionError: fixtures.append(mode)
                else: raise AssertionError('Corrupt saved gradient accepted: '+mode)
            np.save(batch_path, original_array, allow_pickle=False)
    shell_path = BUNDLE/'scripts/run_learning_signal.sh'; shell = shell_path.read_text(encoding='utf-8')
    assert shell.count('630s') == shell.count('330s') == 1 and shell.count('--kill-after=30s') == 2 and 'CODE=${PIPESTATUS[0]}' in shell
    syntax = read(PREP/'Bash_readonly_syntax.json'); assert syntax['exit_code'] == 0 and syntax['script_sha256'] == sha(shell_path) and not syntax['script_executed']
    sources = [*BUNDLE.rglob('*.py'), auditor_path, Path(__file__), ROOT/'scripts/prepare_cctv_dgp_v40_learning_signal_v1.py']
    for source in sources: ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10))
    worker_source = worker_path.read_text(); audit_source = auditor_path.read_text()
    for source in [worker_source, audit_source]: assert not any(s in source for s in ['torch.optim', '.backward(', '.step(', 'torch.save('])
    assert worker_source.count('torch.autograd.grad(') == 1 and 'autograd.grad(' not in audit_source
    guide = (ROOT/'CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md').read_text(encoding='utf-8')
    commands = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert pin in guide and prepared['archive_sha256'] in guide and 'tmux new-session -A -s dgp_v40_learning_signal_v1' in guide
    assert '280' in guide and 'zero optimizer updates' in guide
    app = read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256']; assert len(app) == 14
    for name, value in app.items(): assert sha(ROOT/name) == value
    assert not any(name in sys.modules for name in ['torch', 'dgp_frozen_inference_v2', 'cctv_dgp_pilot'])
    assert time.monotonic()-start <= 300
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin, 'archive_sha256': prepared['archive_sha256'],
        'archive_files': len(expected), 'archive_bytes': prepared['archive_bytes'], 'uncompressed_packet_bytes': total,
        'assets_verified': len(p['assets_sha256']), 'local_basis_verified': len(p['local_basis_sha256']),
        '100_TRAIN_cases20_references_metadata_selection_exact': True, 'all57_tensor_layouts_and_states_bound': True,
        'V40_decoder_inverse_source_and_AST_exact': True, 'V40_losses_filters_normalizers_and_gates_exact': True,
        'Windows_pre_neural_rejection_passed': True, 'scope_rejections': scope_rejections, 'unsafe_archive_rejections': unsafe,
        'saved_gradient_rejections': fixtures, 'synthetic_fixtures_are_not_VM_or_training_evidence': True,
        'Python310_sources_checked': len(sources), 'Bash_readonly_syntax_verified': True, 'single_remote_source_gcloud_commands_verified': 5,
        'all14_app_bindings_unchanged': True, 'neural_calls': 0, 'local_gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
        'diagnostic_VM_execution_started': False, 'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 300}
    with (PREP/'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print({'complete': True, 'files': len(expected), 'scope_rejections': scope_rejections, 'unsafe_archive_rejections': unsafe, 'seconds': receipt['seconds']}, flush=True)


if __name__ == '__main__': main()
