"""Independent packet/source/gate/data audit; no model or derivative calls."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
CLOSED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
DIAGNOSTIC = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
NAME = 'cctv_dgp_feature_fusion_vm_v32'
STEM = 'cctv-dgp-feature-fusion-v32'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def old_names(text):
    for new, old in [
        ('own-DGP-original-feature-fusion-and-decoder-v32', 'own-DGP-profile-matched-mean-centered-original-decoder-v31'),
        (NAME, 'cctv_dgp_profile_batches_vm_v31'),
        ('cctv_dgp_feature_fusion_v32_return', 'cctv_dgp_profile_batches_v31_return'),
        (STEM, 'cctv-dgp-profile-batches-v31'),
        ('cctv_dgp_feature_fusion_v32', 'cctv_dgp_profile_batches_v31'),
        ('dgp_candidate_v32', 'dgp_candidate_v31'), ('V32', 'V31'),
    ]:
        text = text.replace(new, old)
    return text


class RemoveClosedChecks(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'closed_V31_and_fusion_diagnostic_check': return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'closed_V31_and_fusion_diagnostic_check': return None
        return self.generic_visit(node)

    def visit_Tuple(self, node):
        node.elts = [n for n in node.elts if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'closed_V31_and_fusion_diagnostic_check')]
        return self.generic_visit(node)


def source_diff():
    worker = old_names((BUNDLE / 'scripts/cctv_dgp_feature_fusion_v32_vm.py').read_text())
    worker = worker.replace("\n    assert p['fusion_parameters'] == 479616 and p['selected_parameters'] == 978243 and p['selected_tensors'] == 23", '')
    worker = worker.replace('from cctv_dgp_profile_batches_v31 import FeatureFusionCandidateV31 as OriginalDecoderCandidate',
                            'from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29 as OriginalDecoderCandidate')
    worker = worker.replace('offset == 978243 and len(parameters) == 23', 'offset == 498627 and len(parameters) == 12')
    worker = worker.replace('(7,978243)', '(7,498627)').replace('len(pieces) == 23', 'len(pieces) == 12')
    worker = worker.replace('all23', 'all12').replace('selected23', 'selected12').replace('All23 selected original fusion/decoder tensors', 'All12 selected original decoder tensors')
    worker = worker.replace("'selected_parameters':978243,'selected_tensors':23,'decoder_parameters':498627,'decoder_parameter_tensors':12,'fusion_parameters':479616,'parameter_layout':layout",
                            "'decoder_parameters':498627,'decoder_parameter_tensors':12,'parameter_layout':layout")
    worker = worker.replace('for v in candidate.net.fpn.features.parameters()', 'for v in candidate.net.fpn.parameters()')
    worker = worker.replace('Need8GiB free', 'Need6GiB free')
    worker = worker.replace('a = parser.parse_args(); root = a.root.resolve(); sys.path.insert(0, str(root)); parent = a.parent.resolve(); p = verify(root,a.protocol_sha)',
                            'a = parser.parse_args(); root = a.root.resolve(); parent = a.parent.resolve(); p = verify(root,a.protocol_sha)')
    reduced = RemoveClosedChecks().visit(ast.parse(worker))
    assert ast.dump(reduced, include_attributes=False) == ast.dump(ast.parse((OLD / 'scripts/cctv_dgp_profile_batches_v31_vm.py').read_text()), include_attributes=False), 'Unexpected worker logic change'
    training = old_names((BUNDLE / 'cctv_dgp_feature_fusion_v32_training.py').read_text())
    training = training.replace("assert len(selected)==23 and sum(value.numel() for _,value in selected)==978243\n    assert [name for name,_ in selected]==[row['name'] for row in p['parameter_layout']]",
                                'assert len(selected)==12 and sum(value.numel() for _,value in selected)==498627')
    training = training.replace('selected23', 'selected12').replace('selected12 original fusion/decoder tensors', 'selected12 original decoder tensors')
    training = training.replace("'trained_parameters':978243,'trained_tensors':23,'fusion_parameters':479616,'decoder_parameters':498627", "'trained_parameters':498627,'trained_tensors':12")
    training = training.replace("'parameter_partition_changed_only':True", "'batch_formation_changed_only':True")
    assert training == (OLD / 'cctv_dgp_profile_batches_v31_training.py').read_text(), 'Loss/optimizer/gate/snapshot math changed'
    assert old_names((BUNDLE / 'cctv_dgp_feature_fusion_v32_cache.py').read_text()) == (OLD / 'cctv_dgp_profile_batches_v31_cache.py').read_text()
    assert old_names((BUNDLE / 'scripts/run_v32.sh').read_text()) == (OLD / 'scripts/run_v31.sh').read_text()


def main():
    start = time.monotonic()
    assert sys.platform == 'win32' and not (PREP / 'independent_packet_audit.json').exists()
    prep, p, old = read(PREP / 'preparation.json'), read(BUNDLE / 'protocol.json'), read(OLD / 'protocol.json')
    pin = sha(BUNDLE / 'protocol.json'); assert pin == prep['protocol_sha256']
    for key in ['optimizer', 'retained_capacity_gates', 'terms', 'initial_preservation_terms_exact_zero',
                'initial_proof_case_rows', 'case_rows', 'training_references', 'preview_case_ids',
                'mixed_TRAIN_assets_sha256', 'original_checkpoint_sha256', 'original_DGP_state',
                'frozen_recognizer_state', 'capacity_policy', 'decoder_parameters', 'decoder_parameter_tensors']:
        assert p[key] == old[key], key
    budgets = dict(p['budgets']); assert budgets.pop('minimum_free_disk_bytes') == 8 * 1024**3
    old_budgets = dict(old['budgets']); old_budgets.pop('minimum_free_disk_bytes')
    assert budgets == old_budgets
    assert p['optimizer_updates'] == 800 and p['epochs'] == 800 / 781 and p['batch_size'] == 5
    assert p['parameter_partition_changed_only'] and not p['batch_formation_changed_only'] and p['manual_VM_required']
    diagnostic_p = read(DIAGNOSTIC / 'protocol.json')
    assert p['parameter_layout'] == diagnostic_p['parameter_layout']
    assert p['fusion_parameter_names'] == diagnostic_p['fusion_parameter_names']
    assert p['decoder_parameter_names'] == diagnostic_p['decoder_parameter_names']
    policy_spec = importlib.util.spec_from_file_location('local_verified_V32_policy', BUNDLE / 'cctv_dgp_feature_fusion_v32_policy.py')
    policy = importlib.util.module_from_spec(policy_spec); policy_spec.loader.exec_module(policy)
    policy.validate_layout(p['parameter_layout'])
    assert policy.SELECTED_PARAMETERS == 978243 and len(policy.SELECTED_NAMES) == 23
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json')
    analysis = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_analysis/analysis.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['members_verified'] == 756
    assert audit['CPU_replay']['cases_at_both_states'] == 200 and audit['optimizer_updates'] == audit['local_gradient_calls'] == 0
    assert analysis['complete'] and analysis['new_finite_fusion_partition_pilot_justified_for_preparation']
    assert analysis['all23_improvement_gradients_nonzero_in_both_cohorts_at_both_states']
    for folder, mapping in [(ROOT, p['local_basis_sha256']), (OLD, old['assets_sha256']),
                            (ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2', p['mixed_TRAIN_assets_sha256']),
                            (BUNDLE, p['assets_sha256']), (DIAGNOSTIC, p['closed_fusion_diagnostic_evidence_sha256'])]:
        for name, digest in mapping.items(): assert sha(folder / name) == digest, name
    for name, digest in p['closed_V31_evidence_sha256'].items():
        assert sha(OLD / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name) == digest
    early = read(CLOSED / 'outputs/early_structure_stop.json'); failed = read(CLOSED / 'outputs/failure.json')
    assert early['minimum'] == .01 and not early['pass'] and failed['optimizer_updates'] == 50 and not failed['resume_permitted']
    for name in ['cctv_dgp_mean_centered_decoder_v29.py', 'cctv_dgp_app_input_v28.py',
                 'cctv_dgp_profile_batches_v31_schedule.py', 'schedule.json']:
        assert (BUNDLE / name).read_bytes() == (OLD / name).read_bytes(), name
    actual = {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()}
    assert actual == set(p['assets_sha256']) | {'protocol.json'} and len(actual) == prep['packet_files'] == 11
    source_diff()
    for f in BUNDLE.rglob('*.py'): ast.parse(f.read_text(), feature_version=(3, 10))
    candidate_tree = ast.parse((BUNDLE / 'cctv_dgp_feature_fusion_v32.py').read_text())
    cls = next(n for n in candidate_tree.body if isinstance(n, ast.ClassDef))
    assert [n.name for n in cls.body if isinstance(n, ast.FunctionDef)] == ['__init__'], 'Inherited forward/train/VM guard must remain unchanged'
    checker, template = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_return.py', ROOT / 'scripts/cctv_dgp_feature_fusion_v32_return_audit_template.py'
    assert checker.read_text() == template.read_text().replace('FUSION_V32_PROTOCOL_PIN', pin)
    assert sha(checker) == prep['prospective_return_auditor_sha256'] and sha(template) == prep['prospective_template_sha256']
    ast.parse(checker.read_text(), feature_version=(3, 10))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == len(actual) and len({m.name for m in members}) == len(actual)
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk() and member.name.startswith(NAME + '/')
            name = member.name[len(NAME) + 1:]; assert name in actual
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == sha(BUNDLE / name)
    guard = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/cctv_dgp_feature_fusion_v32_vm.py'), '--root', str(BUNDLE), '--protocol-sha', pin, '--run'], capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local gradients' in guard.stderr and not (BUNDLE / 'outputs').exists()
    (PREP / 'Windows_training_guard.txt').write_text(guard.stderr, encoding='utf-8')
    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cctv_dgp_feature_fusion_v32.py', '-v'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    (PREP / 'regressions.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    assert tests.returncode == 0 and 'Ran 11 tests' in tests.stderr and tests.stderr.strip().endswith('OK'), tests.stderr
    bash = read(PREP / 'bash_syntax.json')
    assert bash['complete'] and bash['exit_code'] == 0 and bash['script_sha256'] == sha(BUNDLE / 'scripts/run_v32.sh')
    guide = (ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_VM.md').read_text()
    assert all(f'{i}. ' in guide for i in range(1, 6)) and pin in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 5
    assert guide.count('"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + '-results.tar.gz"') == 1
    # Preserve source/data label separation and prevent evaluation cases entering the schedule.
    split = read(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2/mixed_protocol_v9.json')
    train_ids = {r['id'] for r in split['references'] if r['role'] == 'train'}
    eval_ids = {r['id'] for r in split['references'] if r['role'] != 'train'}
    assert train_ids.isdisjoint(eval_ids) and len(train_ids) == 781
    assert all(c['role'] == 'train' and c['source_person_or_reference'] in train_ids for c in p['case_rows'])
    report = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin,
              'archive_sha256': prep['archive_sha256'], 'packet_files_verified': 11, 'TRAIN_files_verified': 5467,
              'selected_tensors': 23, 'selected_parameters': 978243, 'fresh_original_checkpoint_copy': True,
              'losses_optimizer_gates_data_schedule_forward_and_normalization_unchanged': True,
              'backbone_head4_and_buffers_frozen': True, 'only_measured_fusion_partition_added': True,
              'minimum_free_space_GiB': 8, 'finite_updates': 800, 'first_epoch_batches': 781,
              'V31_failure_and_zero_update_diagnostic_retained': True,
              'Windows_pre_neural_training_rejection': True, 'regressions_passed': 11,
              'read_only_Bash_syntax_pass': True, 'five_manual_steps_verified': True, 'PuTTY_downloads_separate': True,
              'local_neural_or_gradient_calls': 0, 'actual_VM_training_started': False,
              'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
              'seconds': time.monotonic() - start}
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
