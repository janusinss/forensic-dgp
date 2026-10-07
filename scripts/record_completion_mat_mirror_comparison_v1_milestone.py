"""Close the completed converted-MAT comparison without promoting it to the app."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_mirror_comparison_v1_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_milestone'
COMPARISON = ROOT / 'outputs/completion_mat_mirror_comparison_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists(), 'Retain earlier milestone records'
    previous = read(PREVIOUS / 'milestone.json')
    for name, digest in previous['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
    closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert closure['complete'] and closure['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    results, audit, visual = map(read, [COMPARISON / 'results.json', COMPARISON / 'independent_saved_output_audit.json', COMPARISON / 'visual_review.json'])
    assert results['complete'] and results['requests'] == 32 and results['pretrained_forwards'] == 28
    assert audit['complete'] and audit['all32_saved_outputs_verified'] and audit['all28_raw512_to256_compositions_verified']
    assert visual['complete'] and visual['all32_cases_reviewed'] and visual['all8_sheets_viewed_at_original_detail']
    assert audit['results_sha256'] == visual['results_sha256'] == sha(COMPARISON / 'results.json')
    assert visual['independent_saved_output_audit_sha256'] == sha(COMPARISON / 'independent_saved_output_audit.json')
    assert not visual['assisted_quality_qualification'] and not visual['automatic_quality_qualification']
    r1 = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_independent_audit.json')
    assert r1['complete'] and r1['VM_failure_retained'] and r1['snapshots_audited'] == []
    download = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_download_v1/download_receipt.json')
    assert download['complete'] and download['archive_bytes'] == 1324635165 and download['VM_writes'] == 0
    addition = '''
**Completion comparison milestone - 7 October 2026: converted MAT tested and retained as an unsuccessful comparison.**

The pinned third-party FP16 EMA MAT FFHQ512 conversion completed32 eligible
assisted development requests:28 actual CPU forwards and four exact empty-mask
bypasses, with no gradients or training. Its source and published125,280,246-byte
tensor archive are independently checked;465 tensors/all62,612,683 finite values
are verified. This is the converted release and pinned CPU implementation, with
original-author checkpoint/CUDA parity unverified. Original MAT CC-BY-NC4.0 and
ChaiNNer MIT notices remain retained. No general MAT failure or deployment claim.

The frozen comparison keeps the same reviewed masks and current CodeFormer Off
outputs. All seven covering families and uncovered/clear-glasses controls are
included. The32 cases are exposed photographs and synthetic degradations, not
native CCTV. Four prior input-only exclusions stay excluded. Pretraining overlap
is unknown. No aligned clean hidden-face reference, hidden PSNR/SSIM, exact hidden
identity, ethnicity or Zamboanga-performance claim is introduced.

All32 PNG/stage pairs and28 raw512-to256 compositions pass independent readback.
All5,225,232 visible source bytes outside the reviewed masks remain exact;
all four clear controls bypass exactly. One fixed CPU replay matches raw values
with maximum error0.0 and exact PNG. Eleven adapter/tensor-format regressions pass.
The bounded inference completed in201.10s; external observation204.89s is within
1230s. No timeout occurred. Raw512 outputs remain separate from delivered256 PNGs.

All32 cases/eight original-detail comparison pages were actually reviewed by the
implementing assistant as development evidence. Eye/mouth estimates can be
plausible, but hand joins, bright lens glare, hair still obscuring an eye and
scarf/nose artifacts prevent a consistent useful improvement. Retained fingers
outside a reviewed footprint are separate input-mask misses. Zero extra removal
margin is used; no global dilation or favorable-seed selection. This assisted-only
comparison qualifies neither automatic proposals nor independent final quality.
The candidate is retained as a negative comparison and is not promoted.

The original own-trained DGP remains the main local restorer. App source/design,
original checkpoints, frozen splits, all failed gates and the actual Windows
research-cache backup remain preserved. Functional app verification already
passes; useful native DGP structure and all-family quality remain incomplete.

The separate R1 archive is now fully downloaded and independently audited:
180 files,75,324,711 saved gradient values and50 original CPU replay cases, zero
optimizer updates and no new checkpoint. Its pre-optimizer routing failure stays
retained. The user separately launched V32 r2 and reported a50-update early
structure stop. Its1,324,635,165-byte archive and two sidecars are now downloaded
with the reported SHA256576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9.
Full independent R2 snapshot/metric audit is in progress; no R2 quality claim or
promotion follows from its export. Actual new training remains the human's manual
existing-L4 workflow. No training was launched by the agent. Goal active/incomplete.

[Converted MAT full-family findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MAT_MIRROR_V1_RESULTS.md>)
[Independent saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_mat_mirror_comparison_v1/independent_saved_output_audit.json>)

The complete prior handoff is preserved below. Its R1 full-download-pending
wording is historical and superseded by the completed independent return audit.

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
    files = [Path(__file__), ROOT / 'scripts/verify_completion_mat_mirror_comparison_v1_milestone.py',
             ROOT / 'mat_mirror_completion_v1.py', ROOT / 'tests/test_mat_mirror_completion_v1.py',
             ROOT / 'CCTV_DGP_COMPLETION_MAT_MIRROR_V1_RESULTS.md', handoff, before_dir / handoff.name,
             PREVIOUS / 'milestone.json', PREVIOUS / 'independent_closure_audit.json', PREVIOUS / 'final_readback.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_return_import.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_download_v1/download_receipt.json']
    files.extend(sorted((ROOT / 'scripts').glob('*completion_mat*py')))
    for folder in [COMPARISON, ROOT / 'outputs/completion_mat_mirror_review_v1',
                   ROOT / 'outputs/completion_mat_mirror_review_v1_r1', ROOT / 'outputs/completion_mat_mirror_assets_v1']:
        files.extend(path for path in sorted(folder.rglob('*')) if path.is_file())
    evidence = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    milestone = {'complete': True, 'datetime_UTC': datetime.now(timezone.utc).isoformat(),
                 'recorder_sha256': sha(Path(__file__)), 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
                 'new_evidence_sha256': evidence,
                 'previous_document_locations': {'PROJECT_HANDOFF.md': (before_dir / handoff.name).relative_to(ROOT).as_posix()},
                 'document': {'name': handoff.name, 'before_path': (before_dir / handoff.name).relative_to(ROOT).as_posix(),
                              'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff),
                              'addition_bytes': len(addition.encode('utf-8')), 'full_previous_body_preserved': True},
                 'MAT_requests': 32, 'MAT_forwards': 28, 'CPU_independent_replay_forwards': 1,
                 'clear_bypasses': 4, 'exact_visible_source_bytes': 5225232,
                 'all32_development_outputs_actually_reviewed': True, 'assisted_only': True,
                 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
                 'original_author_equivalence_verified': False, 'app_changes': False, 'app_promotion': False,
                 'MAT_native_CCTV_or_reserved_final_used': False, 'independent_final_review': False,
                 'R1_full_gradient_audit_complete': True, 'R1_optimizer_updates': 0,
                 'R2_full_audit_pending': True, 'VM_writes': 0, 'training_started_by_agent': False,
                 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'manual_VM_training_required': True,
                 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(milestone, stream, indent=2)
    print(json.dumps({'complete': True, 'bindings': len(evidence), 'milestone_sha256': sha(OUT / 'milestone.json'),
                      'quality_qualification': False, 'goal_complete': False}, indent=2))


if __name__ == '__main__':
    main()
