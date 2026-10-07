"""Independently check transfer routing, fixed cohorts and no local differentiation."""
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_v32_loss_gradient_v1_vm'
STEM = 'cctv-dgp-v32-loss-gradient-v1'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_preparation'
V32 = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
CLOSED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def verify(mapping, folder):
    for name, digest in mapping.items():
        path = (folder / name).resolve()
        assert path.is_relative_to(folder) and path.is_file() and sha(path) == digest, name


def expect_assertion(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError('Expected adversarial boundary rejection')


def main():
    started = time.monotonic()
    prep, p, old = map(read, [PREP / 'preparation.json', BUNDLE / 'protocol.json', V32 / 'protocol.json'])
    pin = sha(BUNDLE / 'protocol.json')
    assert prep['complete'] and pin == prep['protocol_sha256']
    assert p['format'] == 'own-DGP-V32-stopped-loss-gradient-diagnostic-v1'
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == 0 and p['gradient_queries'] == 280
    assert p['parameter_layout'] == old['parameter_layout'] and p['selected_parameters'] == 978243 and p['selected_tensors'] == 23
    assert p['fusion_parameter_names'] == old['fusion_parameter_names'] and p['decoder_parameter_names'] == old['decoder_parameter_names']
    assert p['terms'] == old['terms'] and p['retained_gates'] == old['retained_capacity_gates']
    assert sha(V32 / 'protocol.json') == p['V32_protocol_sha256'] == '87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c'
    assert [c['name'] for c in p['cohorts']] == ['exposed', 'unexposed']
    assert p['states'] == [0, 50] and p['selection_independent_of_output_or_gradient']
    schedule = read(V32 / 'schedule.json')['batches']
    seen, counts, selected_refs = [], Counter(), set()
    for indices in schedule[:50]:
        c = old['case_rows'][indices[0]]
        ref, source = c['source_person_or_reference'], c['source']
        if counts[source] < 5 and ref not in selected_refs:
            seen.extend(old['case_rows'][i] for i in indices)
            selected_refs.add(ref); counts[source] += 1
    assert p['cohorts'][0]['cases'] == seen and len(seen) == 50 and sorted(counts.values()) == [5, 5]
    touched = {old['case_rows'][i]['source_person_or_reference'] for indices in schedule[:50] for i in indices}
    unexposed = p['cohorts'][1]['cases']
    assert {c['id'] for c in unexposed} == set(old['preview_case_ids'])
    assert not touched.intersection(c['source_person_or_reference'] for c in unexposed)
    assert Counter((c['source'], c['profile']) for c in seen) == Counter((c['source'], c['profile']) for c in unexposed)
    case_lookup = {c['id']: c for c in old['case_rows']}
    for cohort in p['cohorts']:
        cases = cohort['cases']
        assert len(cases) == len({c['id'] for c in cases}) == 50
        for begin in range(0, 50, 5):
            group = cases[begin:begin + 5]
            assert len({c['source_person_or_reference'] for c in group}) == 1
            assert {c['profile'] for c in group} == {'clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'}
            assert all(c['role'] == 'train' and case_lookup[c['id']] == c for c in group)
    pairs = p['matched_reference_pairs']
    assert len(pairs) == len(set(pairs.values())) == 10
    for a, b in zip(seen, unexposed, strict=True):
        assert pairs[a['source_person_or_reference']] == b['source_person_or_reference']
        assert a['source'] == b['source'] and a['profile'] == b['profile']
    for name, digest in p['V32_dependencies_sha256'].items():
        assert sha(V32 / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name) == digest, name
    for folder, key in [(ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27', 'parent_dependencies_sha256'),
                        (ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28', 'active_decoder_dependencies_sha256'),
                        (ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2', 'TRAIN_assets_sha256'), (ROOT, 'local_basis_sha256'), (BUNDLE, 'assets_sha256')]:
        verify(p[key], folder)
    failure, stop = map(read, [CLOSED / 'outputs/failure.json', CLOSED / 'outputs/early_structure_stop.json'])
    assert failure['optimizer_updates'] == 50 and not failure['resume_permitted'] and not stop['pass'] and stop['minimum'] == .01
    assert not (CLOSED / 'outputs/results.json').exists()
    assert read(CLOSED / 'outputs/update50/metrics.json')['candidate_DGP_state'] == p['stopped50_DGP_state']
    assert p['stopped50_DGP_state'] == 'd3a7d5c6118313c2a123bdda56719732c3413b2e53670ca91a4713e9294d4b7d'
    setup = read(CLOSED / 'outputs/cohort_loss_setup.json')
    assert p['normalizers'] == [setup['feature_normalizer'], setup['interior_normalizer']]
    expected_budgets = {'worker_seconds': 600, 'external_seconds': 900, 'kill_grace_seconds': 30, 'export_seconds': 300,
                        'external_export_seconds': 330, 'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 6 * 1024**3,
                        'export_uncompressed_bytes': 1536 * 1024**2, 'local_forward_audit_seconds': 1800}
    assert p['budgets'] == expected_budgets
    files = {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()}
    assert files == set(p['assets_sha256']) | {'protocol.json'} and len(files) == 3
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == 3
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk() and m.name.startswith(NAME + '/')
            relative = m.name[len(NAME) + 1:]
            assert relative in files and hashlib.sha256(tar.extractfile(m).read()).hexdigest() == sha(BUNDLE / relative)
    worker = BUNDLE / ('scripts/' + NAME + '.py')
    template = ROOT / 'scripts/cctv_dgp_v32_loss_gradient_v1_return_audit_template.py'
    auditor = ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py'
    assert auditor.read_text(encoding='utf-8') == template.read_text(encoding='utf-8').replace('PROTOCOL_PIN_TO_FILL', pin)
    assert sha(auditor) == prep['prospective_return_auditor_sha256']
    for path in [worker, template, auditor]:
        tree = ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ['backward', 'step', 'AdamW', 'Adam', 'SGD']
                if path != worker: assert node.func.attr != 'grad'
                assert not (node.func.attr == 'save' and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch')
    guard = subprocess.run([sys.executable, '-B', str(worker), '--root', str(BUNDLE), '--protocol-sha', pin, '--run'],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local differentiation' in guard.stderr
    assert not (BUNDLE / 'outputs').exists()
    with (PREP / 'Windows_gradient_guard.txt').open('x', encoding='utf-8') as stream: stream.write(guard.stderr)
    helper = module('packet_metadata_regressions', worker)
    poisoned = copy.deepcopy(old['case_rows'])
    poisoned[schedule[0][0]]['role'] = 'dev'
    expect_assertion(lambda: helper.balanced_exposed_cases(schedule, poisoned))
    duplicate = copy.deepcopy(schedule); duplicate[0][1] = duplicate[0][0]
    expect_assertion(lambda: helper.balanced_exposed_cases(duplicate, old['case_rows']))
    expect_assertion(lambda: helper.matched_unexposed_cases(seen, old['case_rows'], touched, old['preview_case_ids'][:-1]))
    boundary = module('new_packet_return_boundaries', template)
    for name, kind in [('../escape', tarfile.REGTYPE), ('outputs/new.pth', tarfile.REGTYPE), ('outputs/results.json', tarfile.SYMTYPE)]:
        member = tarfile.TarInfo(boundary.PREFIX + name); member.size = 10; member.type = kind
        expect_assertion(lambda: boundary.safe_members([member], {'outputs/results.json'}))
    shell = BUNDLE / 'scripts/run_loss_gradient.sh'
    bash = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(shell).replace('\\', '/')], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert bash.returncode == 0
    guide = (ROOT / 'CCTV_DGP_V32_LOSS_GRADIENT_V1_VM.md').read_text(encoding='utf-8')
    assert all(f'{i}. ' in guide for i in range(1, 6)) and pin in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 5
    assert guide.count('"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + '-results.tar.gz"') == 1
    assert 'tmux new-session -A -s dgp_v32_loss' in guide and 'bash scripts/run_loss_gradient.sh ' + pin in guide
    assert '--verify-transfer &&' in guide and 'test ! -e ~/forensic-dgp/' + NAME in guide
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin, 'archive_sha256': prep['archive_sha256'],
               'packet_files_verified': 3, 'TRAIN_files_verified': len(p['TRAIN_assets_sha256']), 'source_balanced100_cases_verified': True,
               'fixed50_unoptimized_previews_used_without_outcome_selection': True, 'V32_changed_fusion_and_decoder_state_bound': True,
               'gradient_queries_bound': 280, 'optimizer_updates': 0, 'Windows_pre_neural_gradient_rejection': True,
               'adversarial_boundary_regressions_passed': 6, 'Python3_10_source_syntax_pass': True,
               'read_only_local_Bash_syntax_pass': True, 'five_manual_steps_verified': True, 'PuTTY_downloads_separate': True,
               'local_neural_or_gradient_calls': 0, 'VM_execution_started': False, 'native_or_reserved_used': False,
               'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    with (PREP / 'independent_packet_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
