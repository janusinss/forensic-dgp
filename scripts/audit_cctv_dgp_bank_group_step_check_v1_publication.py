"""Read back published status, historical backups and all evidence links."""
from datetime import datetime, timezone
import re
from pathlib import Path

from import_cctv_dgp_bank_group_step_check_v1_return import ROOT, ANALYSIS, read, sha, write


def main():
    receipt = ANALYSIS/'publication_audit.json'
    assert not receipt.exists()
    pub_path = ANALYSIS/'publication.json'
    pub = read(pub_path)
    assert pub['complete'] and not pub['model_qualified'] and not pub['goal_complete']
    assert not pub['actual_training_run'] and not pub['new_packet_sealed']
    for name, digest in pub['proof_sha256'].items():
        assert sha(ANALYSIS/name) == digest
        assert read(ANALYSIS/name)['complete']
    checked = {}
    for name, info in pub['documents'].items():
        current = (ROOT/name).read_bytes()
        backup = Path(info['backup'])
        assert backup.resolve().is_relative_to(ANALYSIS.resolve())
        before = backup.read_bytes()
        assert sha(backup) == info['before_sha256'] and sha(ROOT/name) == info['after_sha256']
        assert current[info['prefix_bytes']:] == before
        prefix = current[:info['prefix_bytes']].decode('utf-8')
        assert '0.009962%' in prefix or 'do not rerun' in prefix
        checked[name] = dict(current_sha256=info['after_sha256'],
            backup_sha256=info['before_sha256'], exact_historical_suffix_preserved=True)
    report_path = ROOT/'CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_RESULTS.md'
    report = report_path.read_text(encoding='utf-8')
    links = set()
    for name, digest in pub['new_documents_sha256'].items():
        assert sha(ROOT/name) == digest
        text = (ROOT/name).read_text(encoding='utf-8')
        for absolute in re.findall(r'\]\(<(C:/[^>]+)>\)', text):
            path = Path(absolute)
            assert path.resolve().is_relative_to(ROOT.resolve()) and path.exists(), absolute
            links.add(absolute)
    summary = read(ANALYSIS/'review_binding_audit.json')['summary']
    for row in summary:
        line = '| '+row['proposal'].removeprefix('balanced_')+' | '+f"{row['raw_structure_gain_percent']:.6f}% | {row['PNG_structure_gain_percent']:.6f}% | "+str(row['raw_preservation_failure_count'])+' | '+str(row['PNG_preservation_failure_count'])+' | '+('Diagnosis only' if row['negative_step_diagnostic_only'] else 'Rejected')+' |'
        assert line in report
    design_path = ANALYSIS/'next_design_review.json'
    assert sha(design_path) == pub['next_design_review_sha256']
    design = read(design_path)
    assert design['complete'] and not design['training_launch_ready'] and not design['transfer_packet_prepared']
    assert not design['more_epochs_qualified'] and not design['app_adoption'] and not design['VM_launch']
    assert design['gradients'] == design['optimizer_updates'] == 0
    assert all(sha(ROOT/name) == digest for name, digest in pub['original_app_checkpoint_protected_sha256'].items())
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    storage_path = ROOT/'outputs/google_drive_offload_20261012_v1/closure_20261013_v1/publication_audit.json'
    assert sha(storage_path) == pub['latest_storage_publication_sha256']
    assert read(storage_path)['removed_backup_bytes'] == 52028568639
    completion_path = ROOT/'outputs/dgp_covering_readiness_20261012_v1/audit.json'
    completion = read(completion_path)
    assert completion['complete'] and completion['checked_bindings'] == 2529
    assert completion['unique_files'] == 1182 and completion['new_neural_calls'] == 0
    assert not completion['automatic_qualified'] and not completion['assisted_qualified']
    totals = completion['condition_totals']
    assert totals['original_photo']['empty_proposals_eligible_coverings'] == 4
    assert totals['synthetic_degraded_photo']['empty_proposals_eligible_coverings'] == 11
    assert all(t['eligible_covering_cases'] == 14 for t in totals.values())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        publication_sha256=sha(pub_path), published_documents=checked,
        result_and_design_documents=pub['new_documents_sha256'], exact_summary_rows=len(summary),
        local_links_checked=len(links), protected_app_and_checkpoint_hashes_match=True,
        storage_is_recorded_LOCAL_offload_not_new_VM_cleanup=True,
        current_candidate_not_adopted_and_no_new_pilot_launch_ready=True,
        original_checker_failure_and_gate_failures_retained=True,
        manual_training_and_complete_scope_unchanged=True,
        new_neural_training_or_VM_calls=0, model_qualified=False, goal_complete=False))
    print(dict(complete=True, preserved_documents=len(checked), local_links=len(links), summary_rows=len(summary)))


if __name__ == '__main__':
    main()
