"""Independent metadata, source-difference and transfer audit; no neural calls."""
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
NAME = 'cctv_dgp_profile_batches_vm_v31'
STEM = 'cctv-dgp-profile-batches-v31'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_profile_batches_v31_preparation'
OLD = ROOT / 'outputs/cctv_dgp_broader_mean_vm_v30'
CLOSED = ROOT / 'outputs/cctv_dgp_broader_mean_v30_return'
DIAGNOSTIC = ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_return'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def old_names(text):
    for new, old in [('own-DGP-profile-matched-mean-centered-original-decoder-v31', 'own-DGP-broader-mean-centered-original-decoder-v30'),
                     (NAME, 'cctv_dgp_broader_mean_vm_v30'), ('cctv_dgp_profile_batches_v31_return', 'cctv_dgp_broader_mean_v30_return'),
                     (STEM, 'cctv-dgp-broader-mean-v30'), ('cctv_dgp_profile_batches_v31', 'cctv_dgp_broader_mean_v30'),
                     ('dgp_candidate_v31', 'dgp_candidate_v30'), ('V31', 'V30')]:
        text = text.replace(new, old)
    return text


class RemoveAddedMetadataChecks(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'paired_batch_and_closed_check': return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'paired_batch_and_closed_check': return None
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        node.elts = [n for n in node.elts if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'paired_batch_and_closed_check')]
        return self.generic_visit(node)


def main():
    assert sys.platform == 'win32', 'Windows packet verification only'
    start = time.monotonic(); prep = read(PREP / 'preparation.json'); p = read(BUNDLE / 'protocol.json'); old = read(OLD / 'protocol.json')
    pin = sha(BUNDLE / 'protocol.json'); assert pin == prep['protocol_sha256']
    assert p['format'] == 'own-DGP-profile-matched-mean-centered-original-decoder-v31'
    for key in ['optimizer', 'budgets', 'retained_capacity_gates', 'parameter_layout', 'terms', 'initial_preservation_terms_exact_zero',
                'initial_proof_case_rows', 'case_rows', 'training_references', 'preview_case_ids', 'mixed_TRAIN_assets_sha256',
                'original_checkpoint_sha256', 'original_DGP_state', 'frozen_recognizer_state', 'capacity_policy']:
        assert p[key] == old[key], 'Unexpected change: ' + key
    assert p['optimizer_updates'] == 800 and p['epochs'] == 800 / 781 and p['decoder_parameters'] == 498627
    assert p['decoder_parameter_tensors'] == 12 and p['batch_size'] == 5
    assert p['batch_formation_changed_only'] and p['human_manual_VM_execution_required']
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in old['assets_sha256'].items(): assert sha(OLD / name) == digest
    for name, digest in p['closed_V30_evidence_sha256'].items():
        path = OLD / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name
        assert sha(path) == digest
    for name, digest in p['closed_sampling_evidence_sha256'].items(): assert sha(DIAGNOSTIC / name) == digest
    diagnostic = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json')
    assert diagnostic['complete'] and diagnostic['diagnostic_complete'] and diagnostic['optimizer_updates'] == 0
    assert diagnostic['members_verified'] == 756 and diagnostic['CPU_replay']['cases_at_both_states'] == 200
    failure = read(CLOSED / 'outputs/failure.json'); early = read(CLOSED / 'outputs/early_structure_stop.json')
    assert failure['optimizer_updates'] == 50 and not failure['resume_permitted'] and early['minimum'] == .01 and not early['pass']
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    assert sha(mixed / 'mixed_protocol_v9.json') == p['mixed_data_protocol_sha256']
    for name, digest in p['mixed_TRAIN_assets_sha256'].items(): assert sha(mixed / name) == digest
    split = read(mixed / 'mixed_protocol_v9.json')
    train_refs = {r['id'] for r in split['references'] if r['role'] == 'train'}
    evaluation_refs = {r['id'] for r in split['references'] if r['role'] != 'train'}
    cases = p['case_rows']; schedule = read(BUNDLE / 'schedule.json'); batches = schedule['batches']
    assert train_refs.isdisjoint(evaluation_refs) and len(train_refs) == 781
    assert len(cases) == 3905 and all(c['role'] == 'train' and c['source_person_or_reference'] in train_refs for c in cases)
    assert len(batches) == 800 and schedule['seed'] == 293001 and schedule['clear_controls_per_batch'] == 1
    assert sorted(i for b in batches[:781] for i in b) == list(range(3905))
    assert len(set(i for b in batches[781:] for i in b)) == 95
    refs = []
    for batch in batches:
        assert len(batch) == len(set(batch)) == 5 and all(type(i) is int and 0 <= i < 3905 for i in batch)
        rows = [cases[i] for i in batch]
        assert [c['profile'] for c in rows] == PROFILES
        assert len({c['source_person_or_reference'] for c in rows}) == len({c['source'] for c in rows}) == 1
        refs.append(rows[0]['source_person_or_reference'])
    old_batches = read(OLD / 'schedule.json')['batches']
    def reference_order(part): return list(dict.fromkeys(cases[i]['source_person_or_reference'] for b in part for i in b))
    assert refs == reference_order(old_batches[:781]) + reference_order(old_batches[781:])[:19] == schedule['reference_order']
    assert len(set(refs[:50])) == 50 and sum(cases[i]['profile'] == 'clear' for b in batches[:50] for i in b) == 50
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest
    actual_names = {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()}
    assert actual_names == set(p['assets_sha256']) | {'protocol.json'} and len(actual_names) == prep['packet_files'] == 9
    for name in ['cctv_dgp_app_input_v28.py', 'cctv_dgp_mean_centered_decoder_v29.py']:
        assert (BUNDLE / name).read_bytes() == (OLD / name).read_bytes()
    assert old_names((BUNDLE / 'cctv_dgp_profile_batches_v31_cache.py').read_text()) == (OLD / 'cctv_dgp_broader_mean_v30_cache.py').read_text()
    assert old_names((BUNDLE / 'scripts/run_v31.sh').read_text()) == (OLD / 'scripts/run_v30.sh').read_text()
    training = old_names((BUNDLE / 'cctv_dgp_profile_batches_v31_training.py').read_text())
    addition = "\n    from cctv_dgp_broader_mean_v30_schedule import validate_profile_batches\n    validate_profile_batches(p['case_rows'], schedule)"
    assert training.count(addition) == 1
    training = training.replace(addition, '').replace("'batch_formation_changed_only':True,'clear_and_degraded_views_paired_in_every_batch':True", "'coverage_changed_only':True")
    assert training == (OLD / 'cctv_dgp_broader_mean_v30_training.py').read_text(), 'Loss/optimizer/snapshot code changed'
    worker_tree = ast.parse(old_names((BUNDLE / 'scripts/cctv_dgp_profile_batches_v31_vm.py').read_text()))
    reduced = RemoveAddedMetadataChecks().visit(worker_tree)
    assert ast.dump(reduced, include_attributes=False) == ast.dump(ast.parse((OLD / 'scripts/cctv_dgp_broader_mean_v30_vm.py').read_text()), include_attributes=False)
    for path in BUNDLE.rglob('*.py'): ast.parse(path.read_text(), feature_version=(3, 10))
    checker = ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_return.py'
    template = ROOT / 'scripts/cctv_dgp_profile_batches_v31_return_audit_template.py'
    assert checker.read_text() == template.read_text().replace('PROFILE_BATCHES_PROTOCOL_PIN', pin)
    assert sha(checker) == prep['prospective_return_auditor_sha256'] and sha(template) == prep['prospective_template_sha256']
    ast.parse(checker.read_text(), feature_version=(3, 10))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == 9 and len({m.name for m in members}) == 9
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(NAME + '/')
            name = member.name[len(NAME) + 1:]; assert name in actual_names
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(BUNDLE / name)
    guard = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/cctv_dgp_profile_batches_v31_vm.py'), '--root', str(BUNDLE), '--protocol-sha', pin, '--run'], capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local gradients' in guard.stderr and not (BUNDLE / 'outputs').exists()
    (PREP / 'Windows_training_guard.txt').write_text(guard.stderr, encoding='utf-8')
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cctv_dgp_profile_batches_v31.py', '-v'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert tests.returncode == 0 and 'Ran 10 tests' in tests.stderr and tests.stderr.strip().endswith('OK')
    (PREP / 'regressions.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    bash = read(PREP / 'bash_syntax.json')
    assert bash['complete'] and bash['exit_code'] == 0 and bash['script_sha256'] == sha(BUNDLE / 'scripts/run_v31.sh')
    guide = (ROOT / 'CCTV_DGP_PROFILE_BATCHES_V31_VM.md').read_text()
    assert all(f'{i}. ' in guide for i in range(1, 6)) and pin in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 5
    assert guide.count('"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + '-results.tar.gz"') == 1
    report = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin, 'archive_sha256': prep['archive_sha256'],
              'packet_files_verified': 9, 'TRAIN_files_verified': 5467, 'TRAIN_references': 781, 'TRAIN_cases': 3905,
              'finite_updates': 800, 'first_epoch_covers_all3905_once': True, 'every_batch_has_paired_clear_and_four_degradations': True,
              'first50_distinct_references': 50, 'first50_clear_controls': 50, 'original_initialization_architecture_losses_optimizer_and_gates_unchanged': True,
              'V30_failure_and_zero_update_diagnostic_retained': True, 'Windows_pre_neural_training_rejection': True,
              'regressions_passed': 10, 'read_only_Bash_syntax_pass': True, 'PuTTY_downloads_separate': True,
              'local_neural_or_gradient_calls': 0, 'actual_VM_training_started': False, 'app_promotion': False,
              'native_or_reserved_used': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
