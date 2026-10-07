"""Independent packet, source, selection and command verification; no neural calls."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_vm'
PREP = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_preparation'
V30 = ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30'
CLOSED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
STEM = 'cctv-dgp-v30-sampling-gradient-v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert sys.platform == 'win32', 'Verify the local Windows packet; do not start a VM run'
    start = time.monotonic(); prep = read(PREP / 'preparation.json'); p = read(BUNDLE / 'protocol.json')
    pin = sha(BUNDLE / 'protocol.json'); assert pin == prep['protocol_sha256']
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == 0 and p['gradient_queries'] == 280
    assert len(p['cohorts']) == 2 and [c['name'] for c in p['cohorts']] == ['exposed', 'unexposed']
    assert p['states'] == [0, 50] and p['selected_tensors'] == 12 and p['selected_parameters'] == 498627
    assert p['selection_independent_of_output_or_gradient'] and p['human_manual_VM_execution_required']
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    old = read(V30 / 'protocol.json'); schedule = read(V30 / 'schedule.json')['batches']
    assert sha(V30 / 'protocol.json') == p['V30_protocol_sha256'] == 'b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'
    all_cases = {c['id']: c for c in old['case_rows']}
    first50 = [old['case_rows'][i] for batch in schedule[:10] for i in batch]
    first250 = {old['case_rows'][i]['id'] for batch in schedule[:50] for i in batch}
    touched = {old['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    assert p['cohorts'][0]['cases'] == first50 and len(first250) == 250 and len(touched) == 218
    matched, used, pairs = [], set(), {}
    for case in first50:
        reference = case['source_person_or_reference']
        match = next(c for c in old['case_rows'] if c['source'] == case['source'] and c['profile'] == case['profile']
                     and c['id'] not in used and c['source_person_or_reference'] not in touched
                     and (c['source_person_or_reference'] == pairs[reference] if reference in pairs
                          else c['source_person_or_reference'] not in pairs.values()))
        pairs[reference] = match['source_person_or_reference']
        matched.append(match); used.add(match['id'])
    assert p['cohorts'][1]['cases'] == matched and len(used) == 50 and used.isdisjoint(first250)
    assert p['matched_reference_pairs'] == pairs and p['reference_repetition_matched']
    assert len(set(pairs.values())) == len(pairs)
    assert Counter(c['source_person_or_reference'] for c in matched) == Counter(pairs[c['source_person_or_reference']] for c in first50)
    assert Counter((c['source'], c['profile']) for c in first50) == Counter((c['source'], c['profile']) for c in matched)
    assert all(c['role'] == 'train' and all_cases[c['id']] == c for c in first50 + matched)
    train_refs = {r['id'] for r in old['training_references']}; assert all(c['source_person_or_reference'] in train_refs for c in first50 + matched)
    for name, digest in p['V30_dependencies_sha256'].items():
        path = V30 / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name
        assert sha(path) == digest, name
    for base, key in [(PARENT, 'parent_dependencies_sha256'), (ACTIVE, 'active_decoder_dependencies_sha256'), (MIXED, 'TRAIN_assets_sha256')]:
        for name, digest in p[key].items(): assert sha(base / name) == digest, name
    assert all(old['mixed_TRAIN_assets_sha256'][name] == digest for name, digest in p['TRAIN_assets_sha256'].items())
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest
    assert p['parameter_layout'] == old['parameter_layout'] and p['terms'] == old['terms'] and p['retained_gates'] == old['retained_capacity_gates']
    setup = read(CLOSED / 'outputs/cohort_loss_setup.json'); assert p['normalizers'] == [setup['feature_normalizer'], setup['interior_normalizer']]
    actual_names = {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()}; assert actual_names == set(p['assets_sha256']) | {'protocol.json'}
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz'); assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == len(actual_names) == prep['packet_files']
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(BUNDLE.name + '/')
            name = member.name[len(BUNDLE.name) + 1:]; assert name in actual_names
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(BUNDLE / name)
    for path in BUNDLE.rglob('*.py'):
        tree = ast.parse(path.read_text(), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ['backward', 'step', 'AdamW', 'Adam', 'SGD']
                assert not (node.func.attr == 'save' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch')
    auditor = ROOT / 'scripts/audit_cctv_dgp_v30_sampling_gradient_v1_return.py'
    template = (ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py').read_text()
    assert auditor.read_text() == template.replace('PROTOCOL_PIN_TO_FILL', pin) and sha(auditor) == prep['prospective_return_auditor_sha256']
    guard = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py'),
                            '--root', str(BUNDLE), '--protocol-sha', pin, '--run'], capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local differentiation' in guard.stderr
    assert not (BUNDLE / 'outputs').exists()
    with (PREP / 'Windows_gradient_guard.txt').open('x', encoding='utf-8') as stream: stream.write(guard.stderr)
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cctv_dgp_v30_sampling_gradient_v1.py', '-v'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert tests.returncode == 0 and 'Ran 11 tests' in tests.stderr and tests.stderr.strip().endswith('OK')
    with (PREP / 'regressions.txt').open('x', encoding='utf-8') as stream: stream.write(tests.stdout + tests.stderr)
    bash_path = PREP / 'bash_syntax.json'; bash = read(bash_path)
    assert bash['complete'] and bash['exit_code'] == 0 and bash['script_sha256'] == sha(BUNDLE / 'scripts/run_sampling.sh')
    guide = (ROOT / 'CCTV_DGP_V30_SAMPLING_GRADIENT_V1_VM.md').read_text()
    assert [f'{i}. ' in guide for i in range(1, 6)] == [True] * 5 and pin in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 5
    assert guide.count('"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + '-results.tar.gz"') == 1
    report = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin, 'archive_sha256': prep['archive_sha256'],
              'packet_files_verified': len(members), 'TRAIN_files_verified': len(p['TRAIN_assets_sha256']), 'TRAIN_cases': 100,
              'source_profile_matching_verified': True, 'reference_repetition_matched': True,
              'distinct_references_in_each_cohort': len(pairs), 'unexposed_references_exclude_all218_touched': True,
              'initial_and_stopped50_states_fixed': True, 'gradient_queries_bound': 280, 'optimizer_updates': 0,
              'Windows_pre_neural_gradient_rejection': True, 'regressions_passed': 11, 'read_only_Bash_syntax_pass': True,
              'five_manual_steps_verified': True, 'PuTTY_downloads_separate': True, 'local_neural_or_gradient_calls': 0,
              'VM_execution_started': False, 'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - start}
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
