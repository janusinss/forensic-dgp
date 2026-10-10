"""Preserve a documentation stop caused by a newly arriving human return."""
from datetime import datetime, timezone
from pathlib import Path
from completion_feature_fusion_off_v1_common import ROOT, OUT, sha, read, write

source = ROOT/'scripts/record_completion_feature_fusion_off_v1_milestone.py'
prior = ROOT/'outputs/cctv_dgp_v39_return_v40_prepared_milestone'
archive = ROOT/'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz'
export = ROOT/'outputs/cctv-dgp-spatial-fit-v40-export.json'
assert not (ROOT/'outputs/completion_feature_fusion_off_v1_milestone').exists()
assert sha(ROOT/'PROJECT_HANDOFF.md') == read(prior/'milestone.json')['new_evidence_sha256']['PROJECT_HANDOFF.md']
receipt = read(export)
write(OUT/'milestone_record_stop.json', {
    'complete': False, 'UTC': datetime.now(timezone.utc).isoformat(), 'recorder_sha256': sha(source),
    'exit_code': 1, 'tool_chunks': ['ba8416', 'e72487'],
    'location': 'scripts/record_completion_feature_fusion_off_v1_milestone.py:23',
    'cause': 'V40 archive became present between inventory and documentation closure; absence guard stopped before handoff changes',
    'traceback': 'Traceback (most recent call last):\n  File "scripts/record_completion_feature_fusion_off_v1_milestone.py", line 136, in <module>\n    if __name__ == "__main__": main()\n  File "scripts/record_completion_feature_fusion_off_v1_milestone.py", line 23, in main\n    assert not (ROOT/"outputs/cctv-dgp-spatial-fit-v40-results.tar.gz").exists()\nAssertionError',
    'handoff_unchanged_verified': True, 'milestone_folder_not_created': True,
    'archive_size_at_stop_record': archive.stat().st_size, 'expected_archive_bytes': receipt['bytes'],
    'expected_archive_sha256': receipt['archive_sha256'], 'export_receipt_sha256': sha(export),
    'returned_export_reports_optimizer_updates': receipt['optimizer_updates'],
    'export_is_not_training_success': receipt['training_success_not_implied'],
    'archive_hash_verification_pending': True, 'independent_V40_audit_pending': True,
    'no_original_recorder_modification': True, 'no_model_or_quality_rule_change': True,
    'VM_calls': 0, 'gradient_calls': 0, 'optimizer_updates_here': 0})
print({'retained_stop': True, 'received_bytes': archive.stat().st_size, 'expected_bytes': receipt['bytes']}, flush=True)
