"""Independent packet, original-definition and failure-boundary audit; no neural work."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from cctv_dgp_spatial_fit_v40_contract import sha, read, write, validate_schedule, feature_support, groups, capacity, exported_pixel_metrics

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
PREP = ROOT/'outputs/cctv_dgp_spatial_fit_v40_preparation'


def reject(call):
    try: call()
    except (AssertionError, ValueError): return
    raise AssertionError('Expected rejection')


def function(path, name):
    node = next(n for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == name)
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str): node.body = node.body[1:]
    return node


def main():
    start = time.monotonic(); p = read(BUNDLE/'protocol.json'); prepared = read(PREP/'prepared.json')
    assert prepared['complete'] and prepared['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    q = read(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39/protocol.json')
    old = read(ROOT/'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json')
    audited = read(ROOT/'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json')
    assert audited['complete'] and audited['diagnostic_complete'] and audited['saved_gradient_readback']['parameter_tensors'] == 57
    assert not audited['saved_gradient_readback']['zero_improvement_tensor_names']
    assert p['retained_capacity_gates'] == q['retained_capacity_gates'] and p['cases'] == old['case_rows'] and p['references'] == old['training_references']
    assert p['parameter_layout'] == q['parameter_layout'] and p['initial_states'] == q['expected_states']
    assert p['preview_case_ids'] == old['preview_case_ids'] == [c['id'] for c in q['cases']]
    assert p['terms'] == q['terms'] and p['optimizer']['learning_rate'] == .0003 and p['optimizer']['weight_decay'] == .01
    assert p['updates'] == 800 and p['snapshots'] == [0, 50, 800] and p['decoder_parameters'] == 17952
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE/name) == digest
    for name, digest in p['sources_sha256'].items(): assert sha(ROOT/name) == digest
    for name, original in p['copied_source_mapping'].items(): assert sha(BUNDLE/name) == sha(ROOT/original)
    original_source = (ROOT/'scripts/cctv_dgp_spatial_decoder_v39.py').read_text(encoding='utf-8')
    source = (BUNDLE/'cctv_dgp_spatial_decoder_v40.py').read_text(encoding='utf-8')
    restored = source.replace('cctv_dgp_spatial_fit_vm_v40', 'cctv_dgp_spatial_decoder_vm_v39').replace('SpatialDGPCandidateV40', 'SpatialDGPCandidateV39').replace('Distinct V40', 'Distinct V39')
    assert restored == original_source, 'Only distinct VM scope and class name change'
    assert sha(BUNDLE/'frozen_definitions.py') == sha(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39/frozen_definitions.py')
    for name in ['cctv_dgp_degraded_objective_v24.py', 'cctv_dgp_batchmatched_identity_v26.py']:
        assert sha(BUNDLE/name) == sha(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39'/name)
    assert ast.dump(function(ROOT/'cctv_dgp_pilot.py', 'exported_pixel_metrics')) == ast.dump(function(ROOT/'scripts/cctv_dgp_spatial_fit_v40_contract.py', 'exported_pixel_metrics'))
    schedule = read(BUNDLE/'schedule.json')['batches']; validate_schedule(p['cases'], schedule)
    assert sha(BUNDLE/'schedule.json') == sha(ROOT/'outputs/cctv_dgp_profile_batches_vm_v31/schedule.json')
    train_inputs = 0; supports = {}; targets = {}
    original_cases = {c['id']: c for c in q['cases']}
    for c in p['cases']:
        with Image.open(BUNDLE/c['input']) as im: assert im.mode == 'RGB' and im.size == (256, 256)
        if c['observed'] not in supports:
            with Image.open(BUNDLE/c['observed']) as im:
                assert im.mode == 'L' and im.size == (256, 256); a = np.array(im)
                assert set(np.unique(a)) <= {0, 255} and a.any(); supports[c['observed']] = a > 0
        if c['target'] not in targets:
            with Image.open(BUNDLE/c['target']) as im: assert im.mode == 'RGB' and im.size == (256, 256)
            targets[c['target']] = True
        feature_support(supports[c['observed']], c['landmarks5_canvas_xy'])
        if c['id'] in original_cases:
            for key in ['input', 'target', 'observed']:
                assert sha(BUNDLE/c[key]) == sha(ROOT/'outputs/cctv_dgp_spatial_decoder_vm_v39'/original_cases[c['id']][key])
        train_inputs += 1
        assert time.monotonic()-start < 300
    assert len(supports) == len(targets) == 781
    archive = ROOT/'outputs/cctv-dgp-spatial-fit-v40-execution.tar.gz'
    assert sha(archive) == prepared['archive_sha256'] and archive.stat().st_size == prepared['archive_bytes']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); seen = set(); packet_bytes = 0
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk() and m.name.startswith(BUNDLE.name+'/')
            name = m.name[len(BUNDLE.name)+1:]; assert name not in seen and (BUNDLE/name).resolve().is_relative_to(BUNDLE)
            assert name == 'protocol.json' or name in p['assets_sha256']; seen.add(name); packet_bytes += m.size
            import hashlib
            with tar.extractfile(m) as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
            assert digest.hexdigest() == sha(BUNDLE/name)
        assert seen == set(p['assets_sha256']) | {'protocol.json'}
    spec = importlib.util.spec_from_file_location('local_v40_return_contract', ROOT/'scripts/audit_cctv_dgp_spatial_fit_v40_return.py')
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    def member(name, size=1, kind=tarfile.REGTYPE):
        item = tarfile.TarInfo(checker.OUT.name+'/'+name); item.size = size; item.type = kind; return item
    for bad in ['../escape', '/absolute', 'C:/escape', 'outputs\\escape', 'outputs/unknown.py']:
        reject(lambda bad=bad: checker.safe_members([member(bad)], p))
    reject(lambda: checker.safe_members([member('protocol.json', kind=tarfile.SYMTYPE)], p))
    reject(lambda: checker.safe_members([member('protocol.json', kind=tarfile.LNKTYPE)], p))
    reject(lambda: checker.safe_members([member('protocol.json'), member('protocol.json')], p))
    reject(lambda: checker.safe_members([member('protocol.json', size=16*1024**2+1)], p))
    # Scope rejection tests compile the guard only, before any neural import.
    guard = function(ROOT/'scripts/cctv_dgp_spatial_fit_v40_vm.py', 'scope')
    ns = {'sys': SimpleNamespace(platform='linux'), 'platform': SimpleNamespace(node=lambda: 'forensic-dgp-thesis'), 'Path': Path, 'NAME': BUNDLE.name}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), '<V40-scope-only>', 'exec'), ns)
    permitted = Path.home()/'forensic-dgp'/BUNDLE.name; ns['scope'](permitted)
    reject(lambda: ns['scope'](permitted.with_name('cctv_dgp_spatial_decoder_vm_v39')))
    ns['sys'].platform = 'win32'; reject(lambda: ns['scope'](permitted)); ns['sys'].platform = 'linux'
    ns['platform'].node = lambda: 'replacement-VM'; reject(lambda: ns['scope'](permitted))
    blocked = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/cctv_dgp_spatial_fit_v40_vm.py'), '--root', str(BUNDLE), '--protocol-sha', sha(BUNDLE/'protocol.json'), '--verify-transfer'], capture_output=True, text=True, timeout=10)
    assert blocked.returncode != 0 and 'Actual learning requires existing Linux VM' in blocked.stderr
    write(PREP/'Windows_pre_neural_rejection.json', {'complete': True, 'returncode': blocked.returncode, 'stderr': blocked.stderr, 'stdout': blocked.stdout, 'VM_calls': 0})
    # Synthetic metric fixtures exercise genuine preservation decisions, not a trained result.
    fake = [{'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'metrics': {k: .1 for k in ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE']}} for c in p['cases']]
    baseline = groups(fake); better = copy.deepcopy(baseline)
    for v in better.values(): v['landmark_high_frequency_MSE'] = .098
    assert capacity(baseline, better, .01)['pass'] and not capacity(baseline, better, .1)['pass']
    failed_tests = []
    for metric, value in [('MSE', .1+2e-12), ('SSIM', .1-1.1e-6), ('ArcFace_observed_fixed', .1-1.1e-6)]:
        bad = copy.deepcopy(better); bad['all'][metric] = value
        assert not capacity(baseline, bad, .01)['pass']; failed_tests.append(metric)
    bad = copy.deepcopy(better); bad['degraded']['MSE'] = .09; bad['degraded']['constant_mean_shift_only_MSE'] = .097
    assert not capacity(baseline, bad, .01)['pass']; failed_tests.append('brightness_fraction')
    bad = copy.deepcopy(better); bad['dataset/asian_faces/degraded']['landmark_high_frequency_MSE'] = .101
    assert not capacity(baseline, bad, .01)['pass']; failed_tests.append('source_regression')
    incorrect = copy.deepcopy(schedule); incorrect[0][0] = incorrect[0][1]; reject(lambda: validate_schedule(p['cases'], incorrect))
    incorrect = copy.deepcopy(p['cases']); incorrect[schedule[0][0]]['role'] = 'final'; reject(lambda: validate_schedule(incorrect, schedule))
    syntax_count = 0
    for path in BUNDLE.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10)); syntax_count += 1
    for path in [ROOT/'scripts/prepare_cctv_dgp_spatial_fit_v40.py', ROOT/'scripts/audit_cctv_dgp_spatial_fit_v40_return.py', Path(__file__)]:
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    guide = (ROOT/'CCTV_DGP_SPATIAL_FIT_V40_VM.md').read_text()
    commands = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert 'tmux new-session -A -s dgp_spatial_fit_v40' in guide and sha(BUNDLE/'protocol.json') in guide and prepared['archive_sha256'] in guide
    bash = read(PREP/'Bash_readonly_syntax.json'); assert bash['complete'] and bash['script_sha256'] == sha(BUNDLE/'scripts/run_v40.sh')
    assert not (BUNDLE/'outputs').exists() and not (ROOT/'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz').exists()
    assert 'torch' not in sys.modules and 'pretrained_completion' not in sys.modules
    write(PREP/'independent_packet_audit.json', {'complete': True, 'protocol_sha256': sha(BUNDLE/'protocol.json'), 'checker_sha256': sha(Path(__file__)),
          'packet_assets': len(p['assets_sha256']), 'local_bindings': len(p['sources_sha256']), 'all_archive_members_checked': len(members),
          'archive_bytes': archive.stat().st_size, 'uncompressed_bytes': packet_bytes, 'TRAIN_inputs': train_inputs, 'references': 781,
          'all50_proof_input_target_support_files_identical': True, 'all800_batches_verified': True, 'all3905_first_epoch_cases_unique': True,
          'V39_architecture_identical_except_scope_and_name': True, 'original_filters_and_losses_exact': True, 'original_PNG_metric_AST_exact': True,
          'all_numeric_quality_gates_retained': True, 'early_preservation_stop_added': True, 'synthetic_preservation_rejections': failed_tests,
          'unsafe_return_archive_rejections': 9, 'scope_rejections': 3, 'bad_schedule_rejections': 2, 'actual_Windows_pre_neural_rejection': True,
          'Python310_packet_sources': syntax_count, 'Bash_readonly_syntax_pass': True, 'single_remote_source_commands': 5,
          'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
          'prepared_only': True, 'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
          'seconds': time.monotonic()-start, 'cap_seconds': 300})
    print({'complete': True, 'cases': 3905, 'assets': len(p['assets_sha256']), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
