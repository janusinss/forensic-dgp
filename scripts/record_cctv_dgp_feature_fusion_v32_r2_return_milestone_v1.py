"""Close the verified R2 stop, whole-face review and the user's selected design discussion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1'
PREVIOUS = ROOT / 'outputs/completion_mat_mirror_comparison_v1_milestone'
REVIEW = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists()
    prior = read(PREVIOUS / 'milestone.json')
    for name, digest in prior['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
    prior_audit = read(PREVIOUS / 'independent_closure_audit.json')
    assert prior_audit['complete'] and prior_audit['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    audit = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json')
    prepared, visual = map(read, [REVIEW / 'preparation.json', REVIEW / 'visual_review.json'])
    early = read(RETURNED / 'outputs/early_structure_stop.json')
    failure = read(RETURNED / 'outputs/failure.json')
    decision = read(ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json')
    assert decision['complete'] and decision['answer'] == 'apply the best approach and do research also if needed'
    assert decision['selected_direction'] == 'Diagnose the current DGP learning design first'
    assert audit['complete'] and audit['VM_failure_retained'] and audit['members_verified'] == 27625
    assert audit['completed_snapshot_PNG_metrics_vectors_and_mean_controls_audited'] == 7810
    assert audit['saved_gradient_values_checked'] == 75324711
    assert [s['update'] for s in audit['snapshots_audited']] == [0, 50]
    assert not audit['necessary_capacity_pass'] and not audit['training_completed800']
    assert not early['pass'] and early['minimum'] == .01 and failure['optimizer_updates'] == 50
    assert visual['complete'] and visual['cases_reviewed'] == 50 and visual['comparison_cells_reviewed'] == 250
    assert not prepared['preservation_regressions_at50'] and not visual['useful_incremental_whole_face_gain_established']
    addition = f'''
**Latest restoration milestone - 7 October 2026: V32 r2 independently audited; early structure requirement failed.**

The human ran R2 on the existing L4. The directory repair worked; training reached
50/800 updates and then stopped deliberately. Degraded delivered-PNG structure
error fell from0.001876217941608562 to0.001858006016206468:0.9706721697% against
the unchanged1% early requirement. Export complete:true means packaging only.
The stopped checkpoint, original failure/traceback and all prior gates remain
retained. No800-update result exists. Do not resume, repeat unchanged or promote.

The user authorized direct download after only the R1 files were found locally.
The1,324,635,165-byte R2 archive and two sidecars match the reported SHA256
576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9.
Separate pinned-host gcloud downloads finish in302.07s with zero VM writes or
agent training launches. The independent return audit verifies27,625 files,
5,467 TRAIN assets,75,324,711 saved gradient values and both complete3,905-case
snapshots:7,810 delivered PNG metric/vector/mean-control rows. Source/protocol,
all23 connectivity and frozen parameters are checked. CPU inference replay covers
50 fixed previews at each snapshot plus initial50, not all3,905 neural outputs.
Maximum raw discrepancy0.0000025034 and one-byte PNG differences stay within
the unchanged audit tolerances. No local gradients/backwards/optimizer updates.

All17 delivered preservation groups pass at50. Photographic-source structure
reductions are0.952033% for dataset/asian_faces and0.975071% for thumbnails128x128;
both are positive but below1%. Source labels are not ethnicity. These are paired
synthetic photographic TRAIN observations, not native CCTV or Zamboanga evidence.
The50 paired batches optimize50 references/250 cases and no full epoch. Worker
time1033.71s and allocated VRAM8,942,142,976 bytes remain within their finite caps.

All50 fixed previews and10 original-detail sheets were actually reviewed across
eyes, nose, mouth, face outline and overall visible appearance. All250 image cells
are verified exact256px copies, with no display processing. Broad expression and
appearance generally remain, including clear glasses, hair, facial hair, cap and
adjacent hands. Fine eyes/nostrils/lips stay soft. R2 shows little added clarity
over V31; a useful incremental whole-face improvement is not established. Severe
compound crops need clearer input. The fixed previews were not optimized by50,
but are preflight/normalization TRAIN, not held-out or independent final evidence.

Saved parameter analysis confirms actual fusion/decoder movements of about
0.339% relative L2, with frozen state unchanged. Initial improvement-gradient dot
products with the actual displacement are negative, but do not reconstruct AdamW
history or isolate a cause. V30-V32 have missed the same structure requirement.
The invalid assumption is that favourable initial gradients and more original
fusion weights were enough under the unchanged finite learning design. The
recommended discussion now examines raw/delivered losses, normalization,
preservation activation and effective updates before another model change.
The user answered the required diagnostic question: "apply the best approach and
do research also if needed". The selected direction diagnoses current learning
first using saved evidence and primary research. The discussion is satisfied;
the cause remains under investigation. No new trainer is prepared here.

The separate converted-MAT comparison is closed:32 assisted cases,28 CPU forwards,
four exact clear-control bypasses,11 adapter/format checks and5,225,232 exact
visible source bytes. All eight pages were reviewed. Hand/glare/hair/scarf defects
prevent consistent usefulness; the converted candidate is retained, not promoted.
Automatic proposals and independent final quality remain unqualified.

The original own-DGP app checkpoint,256 processing, selector, design and current
frontend stay unchanged. R1's separate zero-update failure is fully audited.
Original checkpoints, frozen splits, research caches, all failures, prior
documents and the actual Windows backup receipt remain preserved. No new native
or reserved pixels are viewed. Actual training remains the human's manual
transfer/SSH/tmux workflow on the existing L4; this agent starts no training.
Useful native structure, all seven completion families with separate automatic/
assisted quality and independent final review remain required. Goal active/incomplete.

[Verified R2 results and full50 review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R2_RESULTS.md>)
[Post-V32 design discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V32_ARCHITECTURE_REVIEW.md>)
[Converted MAT findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_MIRROR_V1_RESULTS.md>)

The complete prior handoff is preserved below. Its R2-audit-pending wording is
historical and superseded by this completed independent return audit.

'''
    OUT.mkdir()
    before_dir = OUT / 'before_docs'
    before_dir.mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    with (before_dir / handoff.name).open('xb') as stream:
        stream.write(before)
    split = before.index(b'\n') + 1
    handoff.write_bytes(before[:split] + addition.encode('utf-8') + before[split:])
    files = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_r2_return_milestone_v1.py',
             ROOT / 'scripts/download_cctv_dgp_feature_fusion_v32_r2_return_v1.py',
             ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r2_return.py',
             ROOT / 'scripts/prepare_cctv_dgp_feature_fusion_v32_r2_failure_review_v1.py',
             ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r2_failure_review_v1.py',
             ROOT / 'scripts/record_cctv_dgp_feature_fusion_v32_r2_visual_review_v1.py',
             ROOT / 'scripts/analyze_cctv_dgp_feature_fusion_v32_r2_displacement_v1.py',
             ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_R2_RESULTS.md', ROOT / 'CCTV_DGP_POST_V32_ARCHITECTURE_REVIEW.md',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_import.json',
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r2-results.tar.gz',
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r2-results.tar.gz.sha256',
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r2-export.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_download_v1/download_receipt.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_displacement_v1/analysis.json',
             ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2/protocol.json',
             handoff, before_dir / handoff.name,
             PREVIOUS / 'milestone.json', PREVIOUS / 'independent_closure_audit.json',
             PREVIOUS / 'closure_audit_r1_correction.json', PREVIOUS / 'final_readback.json',
             ROOT / 'scripts/verify_completion_mat_mirror_comparison_v1_milestone_audit_r1.py']
    files.extend(path for path in sorted(REVIEW.rglob('*')) if path.is_file())
    for name in ['outputs/failure.json', 'outputs/early_structure_stop.json', 'outputs/execution_receipt.json',
                 'outputs/update0/metrics.json', 'outputs/update50/metrics.json', 'outputs/update0/dgp_candidate_v32.pth',
                 'outputs/update50/dgp_candidate_v32.pth', 'outputs/stopped_dgp_candidate_v32.pth', 'supervisor_receipt.json']:
        files.append(RETURNED / name)
    m = {'complete': True, 'datetime_UTC': datetime.now(timezone.utc).isoformat(),
         'recorder_sha256': sha(Path(__file__)), 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'new_evidence_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
         'previous_document_locations': {'PROJECT_HANDOFF.md': (before_dir / handoff.name).relative_to(ROOT).as_posix()},
         'document': {'name': handoff.name, 'before_path': (before_dir / handoff.name).relative_to(ROOT).as_posix(),
                      'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff),
                      'addition_bytes': len(addition.encode('utf-8')), 'full_previous_body_preserved': True},
         'return_protocol_sha256': audit['protocol_sha256'], 'return_archive_sha256': audit['archive_sha256'],
         'returned_files': 27625, 'snapshots_rows': 7810, 'VM_optimizer_updates': 50,
         'structure_gain': early['relative_feature_error_gain'], 'minimum_gain': .01, 'early_gate_pass': False,
         'preservation_groups_passed': 17, 'all50_fixed_previews_actually_reviewed': True,
         'useful_incremental_whole_face_gain_established': False, 'architecture_discussion_pending': False,
         'user_design_answer': decision['answer'], 'selected_direction': decision['selected_direction'],
         'primary_research_requested_if_needed': True,
         'R1_full_gradient_audit_complete': True, 'MAT_comparison_complete_but_unqualified': True,
         'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'new_model_code_or_trainer_released': False,
         'VM_writes': 0, 'training_started_by_agent': False, 'manual_VM_training_required': True,
         'app_changes': False, 'app_promotion': False, 'native_or_reserved_used': False,
         'independent_final_review': False, 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(m, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(m['new_evidence_sha256']),
                      'milestone_sha256': sha(OUT / 'milestone.json'), 'early_gate_pass': False,
                      'architecture_discussion_pending': False}, indent=2))


if __name__ == '__main__':
    main()
