"""Publish completed returned-evidence review and preserve previous document bytes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit'


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)


def main():
    pixels_path = OUT / 'saved_pixel_audit_v1.json'
    peer_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_return_peer/independent_return_review_v1.json'
    state_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_state_audit/saved_state_audit_v1.json'
    cross_path = OUT / 'peer_state_and_metadata_crossread_v1.json'
    pixels, peer, state, cross = [read(p) for p in [pixels_path, peer_path, state_path, cross_path]]
    assert pixels['complete'] and pixels['pixel_phase_complete'] and pixels['independent_saved_pixel_integrity_pass']
    assert pixels['pilot_quality_gates_pass'] is False and pixels['scientific_trial_pass'] is False
    assert peer['complete'] and peer['independent_return_review_pass'] and peer['scientific_trial_pass'] is False
    assert state['complete'] and state['audited_complete_checkpoint_prefix_pass']
    assert state['durable_committed_trajectory_fully_covered']
    assert state['artifact_outcome']['confirmed_optimizer_updates'] == 64
    assert cross['complete'] and cross['peer_state_and_metadata_crossread_pass']
    manifest_path = OUT / 'visual_manifest_v1.json'
    manifest = read(manifest_path)
    assert manifest['planned_sheets'] == 30 and manifest['generated_PNG_cells'] == 712
    assert manifest['all_generated_PNGs_exactly_once']
    notes = [
        'Red cap: slight tonal/edge changes, but eyes/mouth remain soft and no clear structural jump over U96. Existing cap-text mismatch remains.',
        'Fur hat/smile: subtle skin and edge changes through64, without clear new eye/teeth readability; paired target differences remain.',
        'Microphone portrait: small tonal changes; eye/teeth texture remains incumbent-like with no conspicuous consistent improvement.',
        'Blond-costume portrait: small contour/tonal changes; lips and eyes remain soft relative to the clean target.',
        'Suit portrait: subtle nose/mouth edge changes, without a clear consistent gain in fine visible structure.',
        'Mild-turn portrait: small shading changes; no clear improvement in eye, nose, mouth or jaw readability.',
        'Glasses portrait: frames remain readable and similar; subtle shading/edge changes with incumbent softness still present.',
        'Cap portrait: warm appearance and facial outlines persist; fine target texture remains reduced.',
        'Historical paired stronger Gaussian blur: small tonal changes; existing smooth skin and target eye/mouth differences remain.',
        'Historical paired stronger filters: subtle rendering changes; fine eyes/teeth/skin are not consistently clearer.',
        'Historical clear: second column is original CodeFormer behavior teacher, not clean truth. Some small tonal/edge changes; no obvious new destructive artifact.',
        'Historical clear: original behavior teacher only. Glasses/eyes/nose/mouth remain visually stable overall, with small local rendering changes.',
        'Historical dark: R2 behavior teacher is not aligned clean truth. Existing useful brightening persists with small tonal changes, without a clear structural jump.',
        'Historical dark: teacher is R2, not paired truth. Small illumination/detail changes; no obvious new catastrophic artifact or qualified restoration gain.',
    ]
    opt_features = [
        'Smile and teeth; fine eye/skin texture remains smoother than target.',
        'Glasses and mouth; no clear new facial texture recovery.',
        'Eyes, mouth and contour; incumbent smoothing persists.',
        'Hat/glasses smile; target creases/teeth detail remain different.',
        'Glasses/beard; moustache and skin detail remain incumbent-like.',
        'Mild-turn eyes/lips; fine target texture remains reduced.',
        'Eyes and lips; no conspicuous detail jump over U96.',
        'Glasses on forehead/contours; existing softness remains.',
        'Beard/smile; eyes, teeth and hair remain similar in readability.',
        'Beard portrait; target fine skin texture remains reduced.',
        'Cap portrait; eye/lip structures stay soft relative to target.',
        'Eyes and smile; target eyebrow/skin detail remains different.',
        'Mild-turn portrait; no clear gain in fine visible facial structure.',
        'Front portrait; skin/eye smoothing remains.',
        'Smile portrait; fine skin/teeth detail is not clearly newly recovered.',
        'Glasses portrait; frames, eyes and mouth stay similar in readability.',
    ]
    observations = []
    names = ['readout_%02d.png' % n for n in range(1, 15)] + ['optimization_%02d.png' % n for n in range(1, 17)]
    assert [Path(r['file']['path']).name for r in manifest['sheets']] == names
    for i, row in enumerate(manifest['sheets']):
        assert pin(ROOT / row['file']['path']) == row['file']
        count = sum(c['generated_return_PNG'] for c in row['cells'])
        assert count == (28 if i < 14 else 20)
        note = notes[i] if i < 14 else 'Subtle tonal/edge changes across four optimization exposures; no obvious new destructive artifact. ' + opt_features[i - 14]
        observations.append({'file': row['file'], 'personally_viewed_by_root': True,
                             'method': 'view_image original detail; unchanged saved256 cells',
                             'generated_cells_reviewed': count, 'observation': note})
    visual = {'complete': True, 'format': 'smoothanchors-B64-root-personal-visual-review-v1',
              'UTC': datetime.now(timezone.utc).isoformat(), 'reviewer': 'root assistant, unblinded TRAIN review',
              'visual_manifest': pin(manifest_path), 'all_planned_sheets_reviewed': True,
              'sheets_reviewed': 30, 'generated_PNG_cells_reviewed': 712, 'exact_saved256_cells': 952,
              'observations': observations, 'consistent_conspicuous_face_detail_gain_over_U96': False,
              'subtle_tonal_edge_changes_seen': True, 'obvious_new_catastrophic_artifacts_seen': False,
              'scientific_trial_pass': False, 'model_qualified': False,
              'decision': 'Retain U96. Preserve measurable learning progress and stopped candidate; no promotion or appended epochs.',
              'not_independent_human_FINAL_or_user_acceptance': True,
              'closes_original_pending_flags_without_overwriting_reports': True,
              'audits': [pin(p) for p in [pixels_path, peer_path, state_path, cross_path]],
              'new_model_optimizer_training_VM_browser_calls': 0}
    visual_path = OUT / 'root_visual_review_v1.json'
    write(visual_path, (json.dumps(visual, indent=2) + '\n').encode())
    report_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_RETURN_REVIEW.md'
    report = '''# Smoother-anchor U96 trial: measured learning progress, retained quality stop

The new loss helped the finite training process. Global structure gain rose from0.01313% to0.91927% in delivered PNGs; localized center gain rose from0.00436% to0.66466%. All preservation and dark checks passed. All four structure endpoints still missed the unchanged1% screen. The authoritative result remains QUALITY_STOP at64; U96 stays current.

## Verified return

Archive `outputs/cctv-codeformer-u96-postfusion-smoothanchors-v1-results.tar.gz` is1,166,119,912bytes, SHA256 `003674db37fb0bb317067591fcc324b32ea72bb34fa9735e8c3ae5653ac37caf`. Checksum and export match. ProtocolSHA `7bf251dd257d79dfcd8e4bd07fe7674badf4b2f30c15fac10cc048b9a36e778a`. All3,462 regular files/1,308,832,513payload bytes and243 returned nonweight static files plus protocol match; expanded gzip bytes1,315,143,680 and CRC/footer are verified. All244 local sealed static files remain exact.

Worker416.2247seconds, supervisor419.0522seconds/exit2/no timeout, export66.4009seconds.64 confirmed Adam commits/64 attempts, no unknown optimizer outcome. Failure phase `snapshot_u64_quality_gates`; no success result/final invariant receipt is invented. All committed checkpoints, per-commit frozen-state associations, partial/failure records and original weights remain retained. Resource limits passed; initial storage/workload on a future VM must still be checked afresh.

| TRAIN-monitor structure endpoint | Prior L1-anchor RAW | New RAW | Prior PNG | New PNG | Required |
|---|---:|---:|---:|---:|---:|
| Global blur, whole image |0.016153%|0.921550%|0.013132%|0.919274%|1% each|
| Local blur, central128 |0.010889%|0.674250%|0.004359%|0.664661%|1% each|

These are reductions in paired adjacent-residual error on reused TRAIN-monitor photographs, not percentages of recovered identity or native CCTV quality. Both recipes started from exactly the same U96, inputs, head seed, sampler, Adam/LR and64-update cap. The common anchor criterion changed. Scalar gate sums now explicitly match VM3.10 arithmetic; no threshold/tolerance changed. A larger numerical gain alone does not qualify the model.

## Independent audit coverage

[Pixel audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/saved_pixel_audit_v1.json>): all712 saved RAW/PNG pairs, floor conversion, teacher/parity associations and regional metrics verified. All12 entire gate dictionaries match exactly.256 objectives were independently reconstructed using both new SmoothL1 terms and retained L1 diagnostics; maximum GPU-vs-CPUfloat64 objective discrepancy5.05e-8, CPUfloat32 discrepancy5.96e-8. These reported reduction differences do not relax any quality gate. No Torch/model/neural/gradient/optimizer/VM call occurred in the pixel audit.

[State audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_state_audit/saved_state_audit_v1.json>): all65 head/resume checkpoints,64 saved gradient sets and4,644 arrays, all64 Adam commits/moments/steps, sampler exposures and RNG verified. All36 trainable tensors changed; final head movementL2=0.5714068613 versus0.3471820095 before. Norms0.02581–0.14920 never triggered clipping. Independent NumPy recurrence agrees within explicitly bounded FP32 arithmetic differences. Torch only decoded bounded CPU weights-only saved checkpoints; no model/optimizer construction, step, forward/backward or CUDA initialization occurred locally. Frozen539 payloads are omitted, so unchanged claims remain exact original-file and source/runtime-receipt associations, not independently decoded frozen return values.

[Metadata peer](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_peer/independent_return_review_v1.json>) passed29,128 semantic checks. [271-check crossread](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/peer_state_and_metadata_crossread_v1.json>) joins actual pixel/state/metadata evidence. Auditors were independently reviewed with20 saved-pixel and11 saved-state parser/numerical fixtures before reading this return. Their repaired preparation sources and findings remain retained; those parser repairs changed no VM source, model, result or quality requirement. Partial/incomplete returns have explicitly narrower audit coverage and must never manufacture a full64 success.

## All planned images reviewed

[Root visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_audit/root_visual_review_v1.json>) records personal review of all30 sheets,952 total exact256 cells and all712 generated PNG cells exactly once. There are subtle tonal/edge changes and no obvious new catastrophic artifact. Existing softness and clean-target differences in eyes, teeth, skin and other facial details remain. No clear, consistent jump in useful facial readability over U96 is established. This is an unblinded assistant TRAIN review, not user acceptance or independent human FINAL review. Teacher comparisons do not establish true identity.

[Saved trajectory plot](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison/structure_trajectory_v1.png>) compares both fixed recipes on identical inputs. Lines connect the six measured snapshots; they do not forecast a later gain. New PNG global gains at16/32/64 are0.14027%,0.33208%,0.91927%; local gains0.07168%,0.22840%,0.66466%. The late upward course and consistent saved gradients support further investigation of this learning route, while preserving the stopped state.

## Next design decision, with safeguards

[Saved-course method review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison/NEXT_METHOD_REVIEW_V1.md>) finds actual paired objectives decreased across the four fixed data passes and same-slot gradient directions became more persistent than in the prior L1 trial. Clear L1 behavior deviation also increased. The cause is not proved to be a slow LR, and doubling LR cannot be assumed safe. The smallest historical clear mean preservation headroom across snapshots was only0.00239893; the smallest case headroom was0.00389103. These are observed scalar margins, not an authorized additional pixel or identity allowance.

Prepare one distinct fresh64-update comparison changing only maxAdamLR5e-5 to1e-4. Keep complete U96/head seed, SmoothL1 beta0.01, all data/order/coefficients/optimizer settings,64-update budget, snapshot/preservation/1% gates and resource caps fixed. Start fresh with Adam; do not resume B2, append epochs, alter thresholds or choose a lucky intermediate checkpoint retrospectively. The factor is justified as a limited learning-dynamics test, not because0.92 is close to1 or as a guaranteed quality improvement. A separately verified packet/manual guide is required before execution. No new training or VM call was launched here.

If a later finite TRAIN screen passes, broader fixed DEV, separate native/unpaired CCTV, user acceptance and independent FINAL review still decide usefulness. Stable faces or a passed1% screen cannot replace them. Preserve every original model, label, split, source term and failed proposal. These96 synthetic cases remain16 optimization/8 adaptive TRAIN-monitor photographs, not verified separate people or unseen validation. There are no new native/wild/FINAL results, no ethnicity or Zamboanga-performance claim.

Optional protected-surroundings preparation remains separate, with exact protected input pixels, original/derived previews, correction, confirmation and bypass. Prior paired studies did not justify always-on blur. Restoration stays first; completion/removal-mask development remains deferred, with its later full covering-family scope intact. The existing app/defaults are unchanged. The full active goal remains incomplete.
'''
    write(report_path, report.encode())
    note = '''# 22 October2026: smoother-anchor return audited; structure improved, all four endpoints still fail

[Returned review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_RETURN_REVIEW.md>):64 Adam commits independently verified. PNG structure gains improved from prior0.01313%/0.00436% to global0.91927%/localized-center0.66466%, but all four RAW/PNG1% endpoints still failed. All historical/new TRAIN preservation and dark checks passed. Worker QUALITY_STOP/exit2 at64 is authoritative. Preserve all states and failure records; U96 stays current, with no promotion or appended epochs.

All712 RAW/PNG pairs/256 changed objectives,65 head/resume checkpoints/64 gradients and all30 planned sheets/all712 generated cells were independently audited/root personally reviewed. Subtle tonal/edge changes coexist with existing softness; no clear consistent facial-detail jump or user/FINAL acceptance. Same complete U96/input/headseed/LR budget comparison supports measured learning progress; no native/identity/Zamboanga claim. The monitor is adaptive TRAIN after two experiments. Full broader DEV/native/FINAL/app qualification remains outstanding.

[Next-method review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison/NEXT_METHOD_REVIEW_V1.md>) supports preparing one fresh64-update LR-only comparison5e-5→1e-4, retaining SmoothL1/data/order/headseed/allgates/caps. Upward measured course and persistent gradients motivate the test; LR safety/usefulness is unproven and legacy-clear headroom is narrow. Do not continue failedB2, append epochs or relax1%. No new pilot is launched by this review; a distinct verified manual-transfer/tmux guide must precede execution.

Preserve original models/checkpoints/splits/terms/failures and app design. Optional protected-surroundings preparation stays separately evaluated with exact protected input pixels, immutable originals, correction/confirmation/bypass; always-on blur remains unjustified. Restoration first, later completion/covering-family scope unchanged. The full goal remains ACTIVE and incomplete. Earlier entries remain exact history; this entry supersedes their next actions only.

---

'''.encode()
    documents = []
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md']:
        path = ROOT / name
        old = path.read_bytes()
        before = pin(path)
        pending = ROOT / (name + '.smooth-return-v1.pending')
        write(pending, note + old)
        assert path.read_bytes() == old
        os.replace(pending, path)
        assert path.read_bytes()[len(note):] == old
        documents.append({'before': before, 'after': pin(path),
                          'preserved_suffix_offset_bytes': len(note), 'old_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'root_visual': pin(visual_path), 'report': pin(report_path),
               'audits': visual['audits'], 'documents': documents, 'scientific_trial_pass': False,
               'model_qualified': False, 'next_recipe_status': 'DESIGN_NOT_YET_SEALED',
               'new_model_optimizer_training_VM_calls': 0}
    write(OUT / 'return_publication_v1.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'report': str(report_path), 'all30_sheets_reviewed': True,
                      'scientific_trial_pass': False, 'model_qualified': False}))


if __name__ == '__main__':
    main()
