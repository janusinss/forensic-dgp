"""Record audited stopped V31 and completion source review, preserving history."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_milestone'
PREVIOUS = ROOT / 'outputs/completion_pixel_support_v1_r1'
REVIEW = ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1'
PIN = 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists(), 'Preserve a previous milestone'
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == '9d0e0a9393fe09897d10dc57b1a86ec9d9c517984cfbfdea0d591ff273659de8'
    for name in ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']:
        assert sha(ROOT / name) == previous['new_evidence_sha256'][name]
    audit = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json')
    prepared, visual = read(REVIEW / 'preparation.json'), read(REVIEW / 'visual_review.json')
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['necessary_capacity_pass']
    assert not audit['training_completed800'] and not audit['native_or_reserved_used']
    assert audit['local_gradient_calls'] == audit['local_optimizer_updates'] == 0
    assert visual['complete'] and visual['cases_reviewed'] == 50 and visual['comparison_cells_reviewed'] == 250
    assert not visual['useful_whole_face_gain_established'] and not visual['independent_final_review']
    assert visual['independent_return_audit_sha256'] == sha(ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json')
    assert read(REVIEW / 'independent_preparation_audit.json')['complete']
    feature = read(ROOT / 'outputs/cctv_dgp_post_v31_feature_path_review_v1/review.json')
    assert feature['complete'] and feature['all_FPN_state_entries_unchanged']
    assert feature['feature_fusion_parameters'] == 479616 and len(feature['feature_fusion_tensors']) == 11
    assert feature['neural_calls'] == feature['gradient_calls'] == feature['optimizer_updates'] == 0
    source_audit = read(ROOT / 'outputs/completion_mat_source_review_v1_r1/independent_source_audit.json')
    assert source_audit['complete'] and source_audit['official_source_files_verified'] == 5
    assert not source_audit['checkpoint_downloaded'] and not source_audit['source_executed']
    status = read(ROOT / 'outputs/cctv_dgp_v31_progress_status_v1_r1/launch_status.log')
    assert status['complete'] and status['read_only'] and not status['V31_processes']
    assert not status['GPU']['stdout'] and status['export_receipt']['optimizer_updates'] == 50
    assert status['export_receipt']['failure_present'] and not status['export_receipt']['run_results_present']
    gain = prepared['early_stop']['relative_feature_error_gain']
    prior_gain = prepared['groups']['degraded']['V30_feature_gain']
    assert not prepared['early_stop']['pass'] and gain < .01
    addition = f'''
**Latest research milestone - 7 October 2026: V31's 50-update structure stop independently audited.**

The returned {audit['archive_bytes']:,}-byte archive matches its original VM export
hash cbb89bb5fb85ca7e8b77203188164ab2472342f8d7f3b3ecd7a07f6b53c6f49b.
All {audit['members_verified']:,} regular allowlisted files and 5,467 TRAIN assets are verified.
The original protocol/source, gradient records, stopped checkpoint and failure
remain retained. The independent local checker uses inference only, never returned
code, gradients, optimizer updates or reserved-final pixels.

V31 improves whole-TRAIN degraded landmark detail by {gain * 100:.6f}% at update 50,
below the unchanged 1% requirement; V30's same-baseline gain was {prior_gain * 100:.6f}%.
No continuation or automatic promotion is permitted. All 17 delivered preservation
groups pass at this stopped snapshot. That does not waive the structure stop or
establish native usefulness. The paired-batch change did not qualify this recipe.
All 50 fixed TRAIN previews were viewed in ten original-resolution sheets;
eyes, nose, mouth, outline and visible appearance remain in the review together.
The stopped result remains soft without useful whole-face improvement established.

A static source and checkpoint review confirms that all FPN state entries stayed
fixed while all 12 active decoder tensors changed. The next distinct hypothesis
measures the original five lateral and three top-down convolutions: 11 fusion
tensors with 479,616 parameters. A finite zero-update existing-L4 diagnostic must
test connectivity and preservation tradeoffs before selecting new trainable
weights. This source finding does not prove a unique cause or improved capacity.
Keep the backbone, evaluation statistics, inactive-head finding and all gates.

At {status['snapshot_utc']}, the read-only maintenance observation confirms no
V31 worker and an idle GPU; export is complete with the failure preserved.
The agent starts, stops or modifies no VM workload. Two original small export
sidecars were read exactly to verify the user-downloaded archive. Actual new
training remains the manual finite existing-L4 workflow. No unchanged failed
recipe or immutable historical pilot may be launched automatically.

A separate completion source review verifies five official MAT files at commit
d273d891ecdad2e1df106516423a75bc45b2d800, including mask polarity, loading and
runtime prerequisites. No MAT checkpoint is acquired or run. Its usefulness,
dependency/checkpoint provenance and local parity remain unverified. The previous
copied-fragment versus generated-anatomy diagnosis remains binding; a different
prior cannot fix pixels retained outside the reviewed removal area. Automatic
and assisted completion still require separate whole-family review.

The existing app, original own-DGP primary model, checkpoints, splits, research
caches, local backup and failed gates remain preserved. No new training packet
or app candidate is released by this milestone. Useful native restoration, all
seven automatic/assisted covering families and independent final review remain
outstanding. The full goal is active/incomplete.

[V31 audited results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_RESULTS.md>)
[Completion source review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_SOURCE_REVIEW_V1.md>)

The complete previous handoff body follows. Its live V31 observation is historical
and superseded by the terminal snapshot and independently audited return above.

'''.encode('ascii')
    OUT.mkdir()
    (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    backup = OUT / 'before_docs/PROJECT_HANDOFF.md'
    with backup.open('xb') as stream:
        stream.write(before)
    split = before.index(b'\n') + 1
    handoff.write_bytes(before[:split] + addition + before[split:])
    files = [backup]
    for folder in [REVIEW, ROOT / 'outputs/cctv_dgp_v31_return_metadata_v1',
                   ROOT / 'outputs/cctv_dgp_v31_progress_status_v1_r1',
                   ROOT / 'outputs/cctv_dgp_post_v31_feature_path_review_v1',
                   ROOT / 'outputs/completion_mat_source_review_v1',
                   ROOT / 'outputs/completion_mat_source_review_v1_r1']:
        files += [path for path in folder.rglob('*') if path.is_file()]
    names = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
             'CCTV_DGP_PROFILE_BATCHES_V31_RESULTS.md', 'CCTV_DGP_COMPLETION_MAT_SOURCE_REVIEW_V1.md',
             'outputs/cctv-dgp-profile-batches-v31-results.tar.gz',
             'outputs/cctv-dgp-profile-batches-v31-results.tar.gz.sha256',
             'outputs/cctv-dgp-profile-batches-v31-export.json',
             'outputs/cctv_dgp_profile_batches_v31_return_import.json',
             'outputs/cctv_dgp_profile_batches_v31_independent_audit.json',
             'scripts/audit_cctv_dgp_profile_batches_v31_return.py',
             'scripts/read_cctv_dgp_v31_return_metadata_v1.py',
             'scripts/prepare_cctv_dgp_profile_batches_v31_failure_review.py',
             'scripts/audit_cctv_dgp_profile_batches_v31_failure_review.py',
             'scripts/record_cctv_dgp_profile_batches_v31_visual_review.py',
             'scripts/review_cctv_dgp_post_v31_feature_path_v1.py',
             'scripts/acquire_completion_mat_source_review_v1.py',
             'scripts/audit_completion_mat_source_review_v1.py',
             'scripts/record_cctv_dgp_profile_batches_v31_return_milestone.py',
             'scripts/verify_cctv_dgp_profile_batches_v31_return_milestone.py']
    files += [ROOT / name for name in names]
    milestone = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'scope': 'Independent V31 stopped return and all50 TRAIN review; separate completion source audit',
                 'new_evidence_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(files))},
                 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
                 'previous_document_locations': {'PROJECT_HANDOFF.md': backup.relative_to(ROOT).as_posix()},
                 'document': {'name': 'PROJECT_HANDOFF.md', 'before_path': backup.relative_to(ROOT).as_posix(),
                              'before_sha256': sha(backup), 'after_sha256': sha(handoff), 'addition_bytes': len(addition)},
                 'scope_documents_unchanged': ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'],
                 'V31_protocol_sha256': PIN, 'V31_return_sha256': audit['archive_sha256'],
                 'V31_return_members': audit['members_verified'], 'V31_optimizer_updates': 50,
                 'V31_early_gain': gain, 'V31_early_requirement': .01, 'V31_resume_permitted': False,
                 'V31_necessary_capacity_pass': False, 'V31_all17_preservation_groups_pass_at50': True,
                 'fixed_TRAIN_previews_viewed': 50, 'visual_review_not_independent_final': True,
                 'VM_snapshot_utc': status['snapshot_utc'], 'V31_live_at_terminal_snapshot': False,
                 'VM_workloads_started_stopped_or_modified_by_agent': False,
                 'local_model_forwards_for_independent_audit': audit['CPU_original_DGP_forwards'] +
                     audit['CPU_candidate_DGP_forwards'] + audit['CPU_recognizer_forwards'],
                 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
                 'MAT_source_files_audited': 5, 'MAT_checkpoint_acquired_or_run': False,
                 'original_FPN_fusion_tensors_reviewed': 11, 'original_FPN_fusion_parameters': 479616,
                 'new_training_packet_released': False, 'app_or_model_changes': False,
                 'reserved_final_used': False, 'quality_qualification': False, 'app_promotion': False,
                 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(milestone, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(milestone['new_evidence_sha256']),
                      'milestone_sha256': sha(OUT / 'milestone.json'), 'goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
