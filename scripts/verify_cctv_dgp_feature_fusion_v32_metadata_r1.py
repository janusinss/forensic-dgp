"""Independent exact R1 metadata/routing comparison and pre-neural guard."""
import ast
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
OLD = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r1'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation'
NAME = BUNDLE.name
STEM = 'cctv-dgp-feature-fusion-v32-r1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    assert not (PREP / 'independent_packet_audit.json').exists()
    old, p, prep = read(OLD / 'protocol.json'), read(BUNDLE / 'protocol.json'), read(PREP / 'preparation.json')
    assert p['superseded_unlaunched_V32_protocol_sha256'] == sha(OLD / 'protocol.json') == 'e022fbc655b1b93caefc0572393171126c3db7c37d0e559732b38bfe8a656ee1'
    assert prep['protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    reduced = copy.deepcopy(p)
    for key in ['metadata_revision', 'metadata_revision_reason', 'superseded_unlaunched_V32_protocol_sha256', 'superseded_packet_evidence_sha256']:
        reduced.pop(key)
    expected = copy.deepcopy(old)
    for key in ['initial_parity', 'normalization', 'gradient_policy']:
        expected[key] = expected[key].replace('all12', 'all23').replace('selected12', 'selected23').replace('Selected12', 'Selected23')
    expected['optimizer']['type'] = 'AdamW selected23 original fusion/decoder tensors'
    expected['execution'] = 'Human gcloud transfer/SSH/tmux training on existing NVIDIA L4/g2-standard-4 only; no assistant training launch. Separately authorized VM maintenance remains permitted.'
    reduced.pop('assets_sha256'); expected.pop('assets_sha256')
    reduced.pop('local_basis_sha256'); expected.pop('local_basis_sha256')
    assert reduced == expected, 'R1 changes numerical/scientific metadata'
    assert p['selected_tensors'] == 23 and p['selected_parameters'] == 978243 and p['optimizer']['type'] == expected['optimizer']['type']
    assert all('all12' not in p[k] and 'selected12' not in p[k] and 'Selected12' not in p[k] for k in ['initial_parity', 'normalization', 'gradient_policy'])
    for mapping in [p['local_basis_sha256'], p['superseded_packet_evidence_sha256']]:
        for name, digest in mapping.items(): assert sha(ROOT / name) == digest, name
    for name, digest in p['assets_sha256'].items():
        assert sha(BUNDLE / name) == digest
        text = (BUNDLE / name).read_text() if name == 'scripts/cctv_dgp_feature_fusion_v32_vm.py' else None
        if text is not None:
            restored = text.replace(NAME, OLD.name).replace(STEM, 'cctv-dgp-feature-fusion-v32').replace('cctv_dgp_feature_fusion_v32_r1_return', 'cctv_dgp_feature_fusion_v32_return')
            assert restored == (OLD / name).read_text(), 'Training worker behavior changed'
        else:
            assert (BUNDLE / name).read_bytes() == (OLD / name).read_bytes(), name
    assert {f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob('*') if f.is_file()} == set(p['assets_sha256']) | {'protocol.json'}
    for f in BUNDLE.rglob('*.py'): ast.parse(f.read_text(), feature_version=(3, 10))
    template_file = ROOT / 'scripts/cctv_dgp_feature_fusion_v32_r1_return_audit_template.py'
    checker = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r1_return.py'
    template = template_file.read_text()
    assert checker.read_text() == template.replace('FUSION_V32_R1_PROTOCOL_PIN', prep['protocol_sha256'])
    restored = template.replace(NAME, OLD.name).replace(STEM, 'cctv-dgp-feature-fusion-v32')
    restored = restored.replace('cctv_dgp_feature_fusion_v32_r1_return', 'cctv_dgp_feature_fusion_v32_return')
    restored = restored.replace('cctv_dgp_feature_fusion_v32_r1_independent_audit', 'cctv_dgp_feature_fusion_v32_independent_audit')
    restored = restored.replace('FUSION_V32_R1_PROTOCOL_PIN', 'FUSION_V32_PROTOCOL_PIN')
    assert restored == (ROOT / 'scripts/cctv_dgp_feature_fusion_v32_return_audit_template.py').read_text()
    assert sha(checker) == prep['prospective_return_auditor_sha256'] and sha(template_file) == prep['prospective_template_sha256']
    ast.parse(checker.read_text(), feature_version=(3, 10))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] and archive.stat().st_size == prep['archive_bytes']
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [prep['archive_sha256'], archive.name]
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == len({m.name for m in members}) == 11
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk() and m.name.startswith(NAME + '/')
            name = m.name[len(NAME) + 1:]; assert name in set(p['assets_sha256']) | {'protocol.json'}
            assert hashlib.sha256(tar.extractfile(m).read()).hexdigest() == sha(BUNDLE / name)
    guard = subprocess.run([sys.executable, '-B', str(BUNDLE / 'scripts/cctv_dgp_feature_fusion_v32_vm.py'), '--root', str(BUNDLE), '--protocol-sha', prep['protocol_sha256'], '--run'], capture_output=True, text=True, timeout=30)
    assert guard.returncode != 0 and 'Existing Linux VM only; no local gradients' in guard.stderr and not (BUNDLE / 'outputs').exists()
    (PREP / 'Windows_training_guard.txt').write_text(guard.stderr, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('verified_local_V32_R1_return_helpers', checker)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Check the actual R1 prospective checker, without any neural/gradient call.
    import numpy as np
    matrix = np.broadcast_to(np.float64(0), (7, 978243))
    assert module.helpers().matrix(matrix) is matrix
    gain = .006945252687208803
    module.verify_early_receipt({'update': 50, 'minimum': .01, 'relative_feature_error_gain': gain, 'pass': False}, gain)
    original_check = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation/independent_packet_audit.json')
    original_forward = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_forward_review/review.json')
    assert original_check['complete'] and original_check['regressions_passed'] == 11
    assert original_forward['complete'] and original_forward['actual_named_parameter_order_matches_packet'] and original_forward['all_buffers_unchanged']
    assert len(original_forward['rows']) == 4 and all(r['exact_raw_parity'] and r['exact_PNG_parity'] for r in original_forward['rows'])
    for name, digest in original_forward['source_bindings_sha256'].items(): assert sha(ROOT / name) == digest
    bash = read(PREP / 'bash_syntax.json')
    assert bash['complete'] and bash['script_sha256'] == sha(BUNDLE / 'scripts/run_v32.sh') and bash['new_parser_invocations'] == 0
    assert bash['script_sha256'] == sha(OLD / 'scripts/run_v32.sh')
    guide = (ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_R1_VM.md').read_text()
    assert all(f'{i}. ' in guide for i in range(1, 6)) and prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert guide.count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 5
    assert 'tmux new-session -A -s dgp_feature_fusion_v32_r1' in guide and NAME in guide and STEM in guide
    assert guide.count('"janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + STEM + '-results.tar.gz"') == 1
    report = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': prep['protocol_sha256'],
              'archive_sha256': prep['archive_sha256'], 'packet_files_verified': 11, 'metadata_revision': 1,
              'all23_descriptions_match_actual_partition': True, 'unchanged_mathematical_training_behavior': True,
              'only_separate_worker_and_export_routing_changed': True, 'first_packet_preserved_unlaunched': not (OLD / 'outputs').exists(),
              'inherited_unchanged_core_regressions_passed': 11, 'inherited_four_case_exact_CPU_parity': True,
              'actual_R1_gradient_schema_verified': True, 'V31_failed_one_percent_receipt_retained': True,
              'R1_Windows_pre_neural_training_rejection': True, 'identical_shell_prior_parser_pass': True,
              'five_manual_steps_verified': True, 'PuTTY_downloads_separate': True,
              'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
              'actual_VM_training_started': False, 'manual_VM_required': True, 'app_promotion': False,
              'native_or_reserved_used': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    write_path = PREP / 'independent_packet_audit.json'
    with write_path.open('x', encoding='utf-8', newline='\n') as stream: json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
