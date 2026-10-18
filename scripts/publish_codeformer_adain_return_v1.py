"""Publish one audited return and requested input-usability requirement, preserving exact history."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'
BEFORE = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_root_pre_audit'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'PROJECT_HANDOFF.md')
GUIDE = 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_VM.md'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_exact_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())

def main():
    chainpath = REVIEW / 'closed_chain_v1.json'
    chain = read(chainpath)
    assert chain['complete'] and chain['completed_updates'] == 50
    assert chain['imported_regular_files_freshly_rehashed'] == 6558
    assert chain['root_personally_reviewed_sheets'] == 55
    assert chain['model_qualified'] is chain['overall_goal_complete'] is False
    assert chain['new_training_or_resume_authorized'] is False
    assert chain['independent_blur_guard_also_failed'] is True
    report = ROOT / 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md'
    assert sha(report) == chain['report_sha256']
    assert sha(REVIEW / 'visual_review_v1.json') == chain['visual_review_sha256']
    # The closed chain just rehashed all 6,558 payload files and frozen assets.
    # Publication rechecks its metadata, reports and review sources; it does
    # not repeat the multi-gigabyte scientific payload scan.
    for name, digest in chain['evidence_sha256'].items():
        if name.startswith('outputs/codeformer_adain_gradient_isolation_v1_return_review/') or name.startswith('scripts/') or name.startswith('research/') or name.startswith('tests/'):
            assert sha(ROOT / name) == digest, name
    prior = read(BEFORE / 'snapshot.json')
    assert prior['complete'] and set(prior['documents']) == set(DOCS) | {GUIDE}
    for name, pin in prior['documents'].items():
        assert sha(ROOT / name) == sha(BEFORE / name) == pin['sha256']
        assert (ROOT / name).stat().st_size == pin['bytes']
    current = '''# CodeFormer AdaIN return audited; usable-input gate required — 17 October 2026

**CURRENT: 50 REAL ADAM UPDATES / PRESERVATION FAILURE / NO MODEL PROMOTION.**
[Actual return review](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md).
[All 55 image sheets](outputs/codeformer_adain_gradient_isolation_v1_return_review/visual_sheets/INDEX.md).
[Next design review](outputs/codeformer_adain_gradient_isolation_v1_return_review/next_design_review_v1.md).
[Input-usability requirement](outputs/codeformer_adain_gradient_isolation_v1_return_review/input_usability_requirement_v1.md).

All 6,558 returned files passed hash-bound import and independent audit. All
350 trainable tensors changed; all 165 frozen tensors stayed exact. Root
personally reviewed all 55 sheets and the separate three-case full256 figure.
The mechanism witness passed, but isolating the AdaIN image-gradient route
did not reduce the prior FP32 failure: the same three of 61 clear controls
exceed the per-case RAW/PNG limit. Worst delivered PNG MAE increase is
0.055917 against the unchanged 0.03 limit. The normal-blur photometric guards
also fail. Six HQ profile mean PNG MAE/SSIM pairs are slightly worse than
retained FP32. Useful compound/lowlight improvements over original CodeFormer
remain; training/export/audit completion does not establish acceptance.

**Latest user requirement: apply an input-only usable-face threshold before
generation for block-like or otherwise insufficient face crops.** Assess
visible facial relationships, native/effective resolution, blur/blocking,
exposure/clipping, coverage and supported frontal/mild-turn pose jointly.
Request a clearer, more frontal or less-covered image when information is
insufficient. Enlargement to256 and plausible output do not prove usability.
Darkness or blur alone must not reject a usable crop. Assess features actually
visible; a covered mouth/nose alone does not reject sufficient upper-face
context for later separate completion. Too little context requires a new crop.

The current app requires operator review and basic flat/empty/coverage checks;
it has no calibrated automatic facial-structure gate. Its Auto blur/noise
signals only suggest restoration. A future confirmed hard-insufficiency
decision must not be bypassed by restoration On/Off or mask confirmation.
Uncertain inputs can receive input-only review. Calibrate any numeric cutoff
prospectively on DEV-only input labels with independent review. Keep existing
labels, all 406 historical results and unchanged 61-clear/53-blur cohorts.
Report accepted/rejected/uncertain/out-of-scope coverage and refusal rates;
do not discard failed examples or tune the policy on reserved FINAL identities.
This is a recorded requirement, not an implemented or qualified new gate.

The next model work is design review of which retained objectives cause
normal-image colour/exposure changes despite existing clear training and
preservation anchors, while supplying useful dark restoration. No sole cause,
component-freeze solution or new recipe is established. No next packet,
unchanged resume, extra epochs or new launch is prepared here. Original
CodeFormer remains the initializer/benchmark and retained DGP models remain
comparison baselines. Preserve all original and failed checkpoints, inputs,
splits, provenance, caches, gate failures and migration backups.

The full restoration-first CodeFormer thesis goal stays active: a useful
fine-tuned256×256 restorer for one cropped frontal/mild-turn face, accurately
credited, preserving visible appearance and existing app design. Useful
moderate brightening is accepted with structure/appearance preserved. Keep
paired synthetic measurements separate from native/unpaired CCTV by source;
24 ChokePoint and 24 QMUL-SurvFace crops provide no aligned clean reference.
No ethnicity, exact recovered identity or Zamboanga performance claim follows.
FINAL pixels remain untouched. User acceptance, independent final review,
useful native outputs, calibrated refusal handling and CodeFormer app
Auto/override/PNG verification through bundled inline Playwright remain open.

Separate detector/completion model improvement remains deferred until the
user accepts restoration. The later scope retains masks, sunglasses, strong
lens glare, hands, obstructing hair, scarves and objects; automatic removal
area review/correction; clear-glasses/non-obstructing-hair/visible-pixel
preservation; one plausible estimate; original/mask/result and PNG/optional
bundle downloads; separate automatic versus assisted evidence and no exact
hidden-identity claim. Too little visible face requires a less-covered crop.

Actual training remains the user's manual finite NVIDIA L4/tmux workflow with
verified transfers and independently audited returns. This audit made no VM
connection, neural/gradient/optimizer call, training launch, app change,
cleanup, new Drive transfer or deletion. Prior storage offload remains
complete and separate from model qualification; no reclaimed space is counted
again. All earlier instruction bytes are retained exactly below and backed up.
Earlier PREPARED/UNRUN and storage-readiness headers are superseded history.
The overall goal remains incomplete.

---

'''
    history = '''# Historical AdaIN V1 commands — returned and audited 17 October 2026

**50 UPDATES VERIFIED; QUALITY STOP RETAINED. DO NOT RELAUNCH OR RESUME UNCHANGED.**
[Actual result](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md)
supersedes PREPARED/UNRUN text below. All original guide bytes remain history.
The next design review has no new executable packet or launch command.
The requested usable-input gate is documented prospectively and unimplemented.

---

'''
    out = REVIEW / 'publication_v1'
    out.mkdir(exist_ok=False)
    (out / 'prior').mkdir()
    rows = []
    prepared = []
    for name in (*DOCS, GUIDE):
        path = ROOT / name
        prefix = (history if name == GUIDE else current).encode('utf-8')
        before = path.read_bytes()
        assert hashlib.sha256(before).hexdigest() == prior['documents'][name]['sha256']
        backup = out / 'prior' / name
        write_exact_new(backup, before)
        after = prefix + before
        staged = path.with_name(name + '.adain-return-publication-stage')
        write_exact_new(staged, after)
        assert staged.read_bytes() == after
        prepared.append((path, staged, before, prefix, backup))
    for path, staged, before, prefix, backup in prepared:
        assert path.read_bytes() == before
        os.replace(staged, path)
        assert path.read_bytes()[len(prefix):] == backup.read_bytes() == before
        rows.append({'path': path.name, 'before_sha256': sha(backup), 'after_sha256': sha(path),
            'prior_bytes': len(before), 'retained_suffix_offset': len(prefix),
            'backup': str(backup.relative_to(ROOT)), 'prior_bytes_retained_exactly': True})
    receipt = {'complete': True, 'date': '2026-10-17', 'documents': rows,
        'closed_chain_sha256': sha(chainpath), 'report_sha256': sha(report),
        'visual_review_sha256': chain['visual_review_sha256'], 'source_sha256': sha(__file__),
        'pre_return_snapshot_sha256': sha(BEFORE / 'snapshot.json'),
        'input_gate_requirement_recorded': True, 'automatic_input_gate_implemented': False,
        'numerical_threshold_calibrated': False, 'actual_updates': 50,
        'quality_stop_retained': True, 'new_training_packet_prepared': False,
        'goal_complete': False, 'model_qualified': False,
        'local_neural_gradient_optimizer_VM_calls': 0, 'app_modified': False}
    write_exact_new(out / 'receipt.json', (json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'complete':True,'documents_updated':len(rows),'prior_bytes_retained':True,
        'quality_stop_retained':True,'automatic_input_gate_implemented':False}))

if __name__ == '__main__':
    main()
