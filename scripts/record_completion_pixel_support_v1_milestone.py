"""Append a bounded diagnostic milestone; retain complete preceding handoff bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_pixel_support_v1_r1'
PREVIOUS = ROOT / 'outputs/dgp_mask_review_race_fix_v1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not (OUT / 'milestone.json').exists()
    previous = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == 'e26d5f973722184a612c86db7a1e2d2987db4eee0476c78c0baf7a147d2c60df'
    for doc in previous['documents']: assert sha(ROOT / doc['name']) == doc['after_sha256']
    audit = read(OUT / 'independent_audit.json'); visual = read(OUT / 'visual_review.json')
    assert audit['complete'] and visual['complete'] and audit['eligible_Off_compositions_independently_verified'] == 32
    assert audit['raw_completion_compositions'] == 28 and audit['empty_mask_exact_bypasses'] == 4
    assert audit['support_ROIs_independently_recounted'] == 8
    status = read(ROOT / 'outputs/cctv_dgp_v31_progress_status_v1/launch_status.log')
    assert status['complete'] and status['read_only'] and not status['training_started_by_agent']
    assert status['protocol_sha256'] == 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
    assert any(p['pid'] == 4942 and '--run' in p['args'] for p in status['V31_processes'])
    addition = '''
**Latest diagnostic milestone - 7 October 2026: completion pixel origin audited; manual V31 run observed live.**

A separate saved-pixel R1 diagnostic verifies 300 source bindings, all 32 eligible
Off photographic outputs, 28 raw completion compositions and four exact empty-mask
bypasses. Four earlier exclusions retain their later operator input-review
decisions. Eight post-hoc diagnostic rectangles distinguish exactly retained
uploaded pixels from generated pixels. They are not covering truth, facial-region
annotations or independent evaluation. No neural/gradient/optimizer calls, mask
edits, app changes or new pilot are involved; the two failed diagnostic attempts
and their source/receipts remain preserved.

The lower-right hand-join rectangle retains 874/1,656 source pixels (52.8%);
the central scarf/chin rectangle estimates 5,138/5,687 pixels (90.3%). Retained
support includes ordinary visible face, so these are not covering-miss rates.
The earlier context6 variant also retains all source support exactly and cannot
remove a copied fragment outside the reviewed output footprint. Generated
texture/anatomy and the earlier gaze issue need a separate completion-prior
comparison on fixed reviewed support. Global dilation or another unchanged
context recipe is not justified. Both original-detail support sheets are reviewed.
These exposed photographs provide no hidden accuracy, ethnicity, native CCTV or
Zamboanga-performance claim; no hidden PSNR/SSIM is computed.

The read-only VM snapshot at 09:20:06 UTC confirms the human-launched V31 Python
PID4942 is live on the GPU and evaluating update50 over all 3,905 TRAIN cases.
The log reaches case2,750 at that snapshot; no terminal result/export receipt is
observed. This is timestamped progress, not an early-gate or training pass. The
agent starts, stops or modifies no workload. The frozen packet/protocol, original
checkpoints, splits, failed gates and local backup remain retained. New actual
training remains the user's manual existing-L4 workflow. Independently audit the
completed user download before choosing any next DGP experiment or app promotion.

The full goal remains active/incomplete: useful native DGP restoration, all seven
automatic/assisted covering families and independent final review are outstanding.
Earlier app function, mask-review race fix and historical source preservation
remain verified; this diagnostic adds no model-quality qualification.

[Completion pixel-origin findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_PIXEL_SUPPORT_V1_RESULTS.md>)
[Corrected V31 tmux and download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md>)

The complete previous handoff body is preserved below. Its launch-unstarted
statements are historical and superseded by the timestamped manual-run observation.

'''.encode('ascii')
    addition = addition.replace(b'PID4942', b'PID 4942').replace(b'update50', b'update 50').replace(b'case2,750', b'case 2,750')
    handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    (OUT / 'before_docs').mkdir()
    backup = OUT / 'before_docs/PROJECT_HANDOFF.md'
    with backup.open('xb') as stream: stream.write(before)
    split = before.index(b'\n') + 1; handoff.write_bytes(before[:split] + addition + before[split:])
    files = [f for f in OUT.rglob('*') if f.is_file()]
    files += [ROOT / name for name in ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
              'CCTV_DGP_COMPLETION_PIXEL_SUPPORT_V1_RESULTS.md', 'scripts/diagnose_completion_pixel_support_v1_r1.py',
              'scripts/audit_completion_pixel_support_v1_r1.py', 'scripts/record_completion_pixel_support_v1_milestone.py',
              'scripts/verify_completion_pixel_support_v1_milestone.py']]
    files += [f for f in (ROOT / 'outputs/cctv_dgp_v31_progress_status_v1').iterdir() if f.is_file()]
    milestone = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved-pixel origin diagnosis and read-only observation of the manually launched V31 process',
        'new_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(set(files))},
        'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
        'previous_document_locations': {'PROJECT_HANDOFF.md': backup.relative_to(ROOT).as_posix()},
        'document': {'name': 'PROJECT_HANDOFF.md', 'before_path': backup.relative_to(ROOT).as_posix(),
                     'before_sha256': sha(backup), 'after_sha256': sha(handoff), 'addition_bytes': len(addition)},
        'prior_root_scopes_unchanged': ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'],
        'diagnostic_eligible_cases': 32, 'raw_completion_compositions': 28, 'empty_mask_bypasses': 4,
        'post_hoc_support_ROIs': 8, 'new_model_forwards': 0, 'local_gradient_calls': 0, 'optimizer_updates': 0,
        'mask_edits': False, 'app_or_model_changes': False, 'original_diagnostic_failures_retained': 2,
        'VM_actions': ['read-only observation of human-launched V31'], 'live_V31_PID_at_snapshot': 4942,
        'V31_snapshot_utc': status['snapshot_utc'], 'V31_early_gate_pass_observed': False,
        'V31_terminal_receipt_observed': False, 'V31_training_started_by_agent': False,
        'V31_protocol_sha256': status['protocol_sha256'],
        'V31_archive_sha256': 'bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8',
        'V31_return_present_locally_at_recording': (ROOT / 'outputs/cctv-dgp-profile-batches-v31-results.tar.gz').exists(),
        'reserved_final_used': False, 'quality_qualification': False, 'app_promotion': False,
        'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(milestone, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(milestone['new_evidence_sha256']),
                      'milestone_sha256': sha(OUT / 'milestone.json'), 'goal_complete': False}, indent=2))


if __name__ == '__main__': main()
