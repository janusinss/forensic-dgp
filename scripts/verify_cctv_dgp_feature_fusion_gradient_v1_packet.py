"""Independent archive, data-boundary and manual-command verification; no models."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation'
V31 = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
CLOSED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
STEM = 'cctv-dgp-feature-fusion-gradient-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)


def main():
    assert sys.platform == 'win32'
    start = time.monotonic()
    prep, p = read(PREP / 'preparation.json'), read(BUNDLE / 'protocol.json')
    pin = sha(BUNDLE / 'protocol.json')
    assert prep['complete'] and pin == prep['protocol_sha256']
    assert p['format'] == 'own-DGP-feature-fusion-zero-update-gradient-diagnostic-v1'
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == 0 and p['gradient_queries'] == 280
    assert p['selected_parameters'] == 978243 and p['selected_tensors'] == 23
    assert p['fusion_parameters'] == 479616 and p['decoder_parameters'] == 498627
    names = {'fpn.lateral' + str(i) + '.weight' for i in range(5)} | {
        'fpn.td' + str(i) + '.0.' + suffix for i in range(1, 4) for suffix in ['weight', 'bias']}
    assert set(p['fusion_parameter_names']) == names and len(p['fusion_parameter_names']) == 11
    assert len(set(p['decoder_parameter_names'])) == 12 and not names.intersection(p['decoder_parameter_names'])
    old = read(V31 / 'protocol.json')
    assert p['decoder_parameter_names'] == [r['name'] for r in old['parameter_layout']]
    assert p['terms'] == old['terms'] and p['retained_gates'] == old['retained_capacity_gates']
    assert {r['name'] for r in p['parameter_layout']} == names | set(p['decoder_parameter_names'])
    offset = 0
    for row in p['parameter_layout']:
        assert row['start'] == offset
        count = 1
        for dim in row['shape']:
            assert isinstance(dim, int) and dim > 0
            count *= dim
        offset += count
        assert row['end'] == offset
    assert offset == 978243
    assert sum(r['end'] - r['start'] for r in p['parameter_layout'] if r['name'] in names) == 479616
    assert p['selection_independent_of_output_or_gradient'] and p['states'] == [0, 50]
    assert [c['name'] for c in p['cohorts']] == ['exposed', 'unexposed']
    schedule = read(V31 / 'schedule.json')['batches']
    first = [old['case_rows'][i] for batch in schedule[:10] for i in batch]
    touched = {old['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    assert len(touched) == 50 and p['cohorts'][0]['cases'] == first
    selected, used, pairs = [], set(), {}
    for case in first:
        reference = case['source_person_or_reference']
        match = next(c for c in old['case_rows'] if c['source'] == case['source'] and c['profile'] == case['profile']
                     and c['id'] not in used and c['source_person_or_reference'] not in touched
                     and (c['source_person_or_reference'] == pairs[reference] if reference in pairs
                          else c['source_person_or_reference'] not in pairs.values()))
        pairs[reference] = match['source_person_or_reference']; used.add(match['id']); selected.append(match)
    assert selected == p['cohorts'][1]['cases'] and p['matched_reference_pairs'] == pairs
    assert len(set(pairs.values())) == len(pairs) == 10
    assert Counter((c['source'], c['profile']) for c in first) == Counter((c['source'], c['profile']) for c in selected)
    for group in p['cohorts']:
        assert len(group['cases']) == len({c['id'] for c in group['cases']}) == 50
        for begin in range(0, 50, 5):
            batch = group['cases'][begin:begin+5]
            assert len({c['source_person_or_reference'] for c in batch}) == 1
            assert {c['profile'] for c in batch} == {'clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'}
            assert all(c['role'] == 'train' for c in batch)
    assert not set(c['source_person_or_reference'] for c in selected).intersection(touched)
    assert sha(V31 / 'protocol.json') == p['V31_protocol_sha256'] == 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
    for name, digest in p['V31_dependencies_sha256'].items():
        assert sha(V31 / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name) == digest
    for base, key in [(ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27', 'parent_dependencies_sha256'),
                      (ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28', 'active_decoder_dependencies_sha256'),
                      (ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2', 'TRAIN_assets_sha256')]:
        for name, digest in p[key].items():
            assert sha(base / name) == digest, name
    for name, digest in p['local_basis_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in p['assets_sha256'].items():
        assert sha(BUNDLE / name) == digest, name
    assert read(CLOSED / 'outputs/failure.json')['resume_permitted'] is False
    assert not read(CLOSED / 'outputs/early_structure_stop.json')['pass']
    setup = read(CLOSED / 'outputs/cohort_loss_setup.json')
    assert p['normalizers'] == [setup['feature_normalizer'], setup['interior_normalizer']]
    assert p['budgets']['minimum_free_disk_bytes'] == 6 * 1024**3
    assert p['budgets']['worker_seconds'] == 600 and p['budgets']['export_uncompressed_bytes'] == 1536 * 1024**2
    files = {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()}
    assert files == set(p['assets_sha256']) | {'protocol.json'}
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == prep['packet_files'] == len(files) == 3
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(BUNDLE.name + '/')
            name = member.name[len(BUNDLE.name) + 1:]
            assert name in files and hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(BUNDLE / name)
    worker = BUNDLE / 'scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py'
    for path in [worker, ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_return_audit_template.py']:
        tree = ast.parse(path.read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ['backward', 'step', 'AdamW', 'Adam', 'SGD']
                assert not (node.func.attr == 'save' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch')
                if 'audit_template' in path.name:
                    assert node.func.attr != 'grad'
    template = (ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_return_audit_template.py').read_text()
    auditor = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py'
    assert auditor.read_text() == template.replace('PROTOCOL_PIN_TO_FILL', pin)
    assert sha(auditor) == prep['prospective_return_auditor_sha256']
    guard = subprocess.run([sys.executable, '-B', str(worker), '--root', str(BUNDLE), '--protocol-sha', pin, '--run'],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local differentiation' in guard.stderr
    assert not (BUNDLE / 'outputs').exists()
    with (PREP / 'Windows_gradient_guard.txt').open('x', encoding='utf-8') as stream:
        stream.write(guard.stderr)
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p',
                            'test_cctv_dgp_feature_fusion_gradient_v1.py', '-v'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    with (PREP / 'regressions.txt').open('x', encoding='utf-8') as stream:
        stream.write(tests.stdout + tests.stderr)
    assert tests.returncode == 0 and 'Ran 7 tests' in tests.stderr and tests.stderr.strip().endswith('OK')
    shell = BUNDLE / 'scripts/run_fusion.sh'
    bash = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(shell).replace('\\', '/')],
                          cwd=ROOT, capture_output=True, text=True, timeout=30)
    write(PREP / 'bash_syntax.json', {'complete': bash.returncode == 0, 'exit_code': bash.returncode,
                                     'script_sha256': sha(shell), 'read_only_local_syntax_check': True,
                                     'stdout': bash.stdout, 'stderr': bash.stderr})
    assert bash.returncode == 0
    guide = (ROOT / 'CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_VM.md').read_text()
    assert all(f'{i}. ' in guide for i in range(1, 6)) and pin in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 5
    assert guide.count('"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + '-results.tar.gz"') == 1
    assert 'tmux new-session -A -s dgp_feature_fusion' in guide
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin,
               'archive_sha256': prep['archive_sha256'], 'packet_files_verified': 3,
               'TRAIN_cases': 100, 'TRAIN_files_verified': len(p['TRAIN_assets_sha256']),
               'two_paired_ten_reference_cohorts_verified': True, 'unexposed_excludes_all50_touched': True,
               'selected23_layout_and_fusion11_verified': True, 'initial_and_stopped50_states_fixed': True,
               'gradient_queries_bound': 280, 'optimizer_updates': 0,
               'Windows_pre_neural_gradient_rejection': True, 'regressions_passed': 7,
               'read_only_local_Bash_syntax_pass': True, 'five_manual_steps_verified': True,
               'PuTTY_downloads_separate': True, 'local_neural_or_gradient_calls': 0,
               'VM_execution_started': False, 'native_or_reserved_used': False, 'app_promotion': False,
               'goal_complete': False, 'seconds': time.monotonic() - start}
    write(PREP / 'independent_packet_audit.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
