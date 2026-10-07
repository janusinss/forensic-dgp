"""Read back the corrected R1 milestone while preserving all earlier evidence."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r1'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    return len(mapping)


def main():
    start = time.monotonic(); m = read(OUT / 'milestone.json'); new = verify(m['new_evidence_sha256'])
    assert m['complete'] and m['metadata_revision'] == 1 and m['goal_status'] == 'active' and not m['goal_complete']
    assert m['manual_VM_required'] and not m['actual_VM_training_started_by_agent']
    assert m['local_neural_calls'] == m['local_gradient_calls'] == m['local_optimizer_updates'] == 0
    assert not any(m[k] for k in ['app_changes', 'native_or_reserved_used', 'quality_qualification', 'app_promotion'])
    doc = m['document']; before = (ROOT / doc['before_path']).read_bytes(); after = (ROOT / doc['name']).read_bytes()
    split = before.index(b'\n') + 1
    assert after[:split] + after[split + doc['addition_bytes']:] == before
    assert hashlib.sha256(before).hexdigest() == doc['before_sha256'] and sha(ROOT / doc['name']) == doc['after_sha256']
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == m['previous_milestone_sha256'] == '6c18e0032af25c2eb8615cd173367b25509dd92e40c36d5745c10fca6397b631'
    prior = verify(previous['new_evidence_sha256'], m['previous_document_locations'])
    old_closed = read(PREVIOUS / 'independent_closure_audit.json')
    assert old_closed['complete'] and old_closed['milestone_sha256'] == m['previous_milestone_sha256']
    assert old_closed['returned_files_preserved'] == 756 and old_closed['previous_actual_Windows_backup_receipt_preserved'] == 4431
    checked, prep, p = read(PREP / 'independent_packet_audit.json'), read(PREP / 'preparation.json'), read(PACKET / 'protocol.json')
    assert checked['complete'] and checked['all23_descriptions_match_actual_partition']
    assert checked['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_metadata_r1.py')
    assert checked['protocol_sha256'] == prep['protocol_sha256'] == sha(PACKET / 'protocol.json') == m['protocol_sha256']
    assert checked['archive_sha256'] == prep['archive_sha256'] == m['archive_sha256']
    assert checked['unchanged_mathematical_training_behavior'] and checked['only_separate_worker_and_export_routing_changed']
    assert checked['inherited_unchanged_core_regressions_passed'] == 11 and checked['inherited_four_case_exact_CPU_parity']
    assert checked['R1_Windows_pre_neural_training_rejection'] and checked['actual_R1_gradient_schema_verified']
    assert checked['identical_shell_prior_parser_pass'] and checked['five_manual_steps_verified'] and checked['PuTTY_downloads_separate']
    prototype_bindings = verify(p['superseded_packet_evidence_sha256']); basis = verify(p['local_basis_sha256'])
    assert p['selected_tensors'] == 23 and p['selected_parameters'] == 978243
    assert p['optimizer']['type'] == 'AdamW selected23 original fusion/decoder tensors'
    assert p['optimizer_updates'] == 800 and p['budgets']['minimum_free_disk_bytes'] == 8 * 1024**3
    assert not (PACKET / 'outputs').exists() and not (ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32/outputs').exists()
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['CPU_replay']['cases_at_both_states'] == 200
    assert sum(r['values_checked'] for r in audit['cohort_gradient_analysis']) == 301298844
    early = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/early_structure_stop.json')
    failed = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_return/outputs/failure.json')
    assert early['minimum'] == .01 and not early['pass'] and failed['optimizer_updates'] == 50 and not failed['resume_permitted']
    app = read(ROOT / 'outputs/dgp_app_v3_integration_record.json')
    app_count = verify({n: d for n, d in app['sources_sha256'].items() if n != 'static/face_workflow.js'})
    assert sha(ROOT / 'static/face_workflow.js') == '9be6cfab57e0041f87571cd38ba6d0d1e0850742b8fafd498f1b1d126137a060'
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == app['checkpoint_sha256']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)), 'milestone_sha256': sha(OUT / 'milestone.json'),
               'new_bindings_verified': new, 'prior45_bindings_preserved': prior, 'prototype_bindings_verified': prototype_bindings,
               'R1_local_basis_verified': basis, 'all23_descriptions_correct': True, 'scientific_behavior_unchanged': True,
               'prior756_returned_files_audit_preserved': True, 'prior301298844_saved_values_audit_preserved': True,
               'prior200_CPU_outputs_audit_preserved': True, 'previous_actual_Windows_backup_receipt_preserved': 4431,
               'first_packet_and_documents_preserved': True, 'full_previous_handoff_preserved': True,
               'V31_failed_one_percent_gate_retained': True, 'unchanged_app_sources': app_count, 'original_app_checkpoint_preserved': True,
               'actual_VM_training_started_by_agent': False, 'local_neural_calls': 0, 'local_gradient_calls': 0,
               'local_optimizer_updates': 0, 'manual_VM_required': True, 'quality_qualification': False,
               'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
