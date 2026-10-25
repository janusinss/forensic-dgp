"""Publish root's completed saved-image review; no model or training calls."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_return_audit'


def doc(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)


def main():
    pixel_path = OUT / 'saved_pixel_arithmetic_companion_v1.json'
    state_path = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_state_audit/saved_state_audit_v1.json'
    peer_path = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_return_peer/independent_return_review_v1.json'
    cross_path = OUT / 'peer_state_and_metadata_crossread_v1.json'
    assert doc(pixel_path)['independent_saved_pixel_integrity_pass'] is True
    assert doc(pixel_path)['pilot_quality_gates_pass'] is False
    assert doc(state_path)['saved_state_trajectory_audit_pass'] is True
    assert doc(peer_path)['independent_return_review_pass'] is True
    manifest_path = OUT / 'visual_manifest_v1.json'
    manifest = doc(manifest_path)
    assert manifest['planned_sheets'] == 30 and manifest['generated_PNG_cells'] == 712
    assert manifest['all_generated_PNGs_exactly_once'] is True
    readout_notes = [
        'Cap-face and expression: eyes/mouth remain close to U96; no visible added structure. Existing hat-letter reconstruction differs from target.',
        'Hat/smile face: baseline teeth and skin rendering differ from clean target; no visible improvement across checkpoints.',
        'Microphone portrait: eye/teeth detail and skin rendering remain at incumbent quality; no visible structural gain.',
        'Blond-costume portrait: eyes/lips/skin softness remains; all candidate checkpoints look essentially identical to U96.',
        'Suit portrait: incumbent smoothing and jaw/teeth differences from target persist; no visible B64 improvement.',
        'Mild-turn portrait: no significant visible eye/nose/mouth/jaw improvement over U96.',
        'Glasses portrait: frames stay readable, with incumbent eye/skin softness; no visible checkpoint advantage.',
        'Cap portrait: warm colour and reconstructed texture remain essentially unchanged from U96.',
        'Historical paired Gaussian filters: existing pale/smooth appearance and eye/mouth target differences persist; no candidate advantage.',
        'Historical paired filters: eyebrow/teeth/skin deficits remain, including the glasses example; no visible checkpoint advantage.',
        'Historical clear controls: original CodeFormer is behavior teacher, not clean truth. Candidate outputs remain visually stable against U96.',
        'Historical clear controls: nose/teeth/glasses rendering remains stable against U96; teacher agreement is not identity verification.',
        'Historical dark inputs: second column is R2 behavior teacher, with no aligned clean target. Useful incumbent brightening persists; no new visible structure.',
        'Historical dark inputs: R2 teacher is not truth. Similar illumination and eye/mouth rendering across U96 and all snapshots; no visible challenger benefit.',
    ]
    opt_details = [
        'Smile and teeth remain U96-like; target fine eye/skin texture is not recovered.',
        'Glasses and mouth remain stable; no visible improvement in eye or facial texture.',
        'Eyes, mouth and facial contour remain stable; incumbent smoothing persists.',
        'Hat/glasses smile remains stable; target creases and teeth detail remain different.',
        'Glasses/beard portrait has no new visible structure; moustache/skin detail remains incumbent-like.',
        'Mild-turn portrait retains incumbent smoothing and lip/eye rendering.',
        'Eyes and lips remain visually stable across all four exposures.',
        'Glasses on forehead and face contours remain stable; no visible detail gain.',
        'Beard/smile portrait shows no visible gain in eyes, teeth or hair detail.',
        'Beard portrait remains stable; target fine skin texture is still reduced.',
        'Cap portrait has no new visible eye/lip structure; incumbent smoothing persists.',
        'Eyes and smile remain stable; target skin/eyebrow detail remains different.',
        'Mild-turn portrait remains stable, with no visible structural improvement.',
        'Front portrait retains incumbent smoothing of skin/eyes; no visible gain.',
        'Smile portrait remains stable; fine target skin/teeth detail is not newly recovered.',
        'Glasses portrait remains stable; no visible improvement in frames, eyes or mouth.',
    ]
    observations = []
    expected = ['readout_%02d.png' % n for n in range(1, 15)] + ['optimization_%02d.png' % n for n in range(1, 17)]
    assert [Path(row['file']['path']).name for row in manifest['sheets']] == expected
    for index, row in enumerate(manifest['sheets']):
        actual = pin(ROOT / row['file']['path'])
        assert actual == row['file']
        cells = sum(cell['generated_return_PNG'] for cell in row['cells'])
        assert cells == (28 if index < 14 else 20)
        note = readout_notes[index] if index < 14 else opt_details[index - 14]
        observations.append({'file': actual, 'personally_viewed_by_root': True,
                             'view_method': 'view_image original detail; exact saved256 cells',
                             'generated_cells_reviewed': cells, 'observation': note})
    visual = {'format': 'B64-root-personal-visual-review-v1', 'complete': True,
              'reviewer': 'root assistant; not independent human FINAL reviewer',
              'UTC': datetime.now(timezone.utc).isoformat(), 'visual_manifest': pin(manifest_path),
              'all_planned_sheets_reviewed': True, 'sheets_reviewed': 30,
              'generated_PNG_cells_reviewed': sum(row['generated_cells_reviewed'] for row in observations),
              'exact_saved256_cells': 952, 'observations': observations,
              'decision': 'No visible useful structure improvement over U96; retain incumbent and quality stop.',
              'new_obvious_catastrophic_artifacts_seen': False,
              'existing_incumbent_target_mismatches_fixed': False,
              'blinded_review': False, 'TRAIN_only': True, 'model_qualified': False,
              'closes_pending_flags_without_rewriting_original_audits': True,
              'qualification': 'No native/FINAL/user acceptance or verified identity claim. Stable images do not establish exact identity preservation.',
              'audits': [pin(path) for path in [pixel_path, state_path, peer_path, cross_path]],
              'new_model_optimizer_training_VM_browser_calls': 0}
    assert visual['generated_PNG_cells_reviewed'] == 712
    visual_path = OUT / 'root_visual_review_v1.json'
    write(visual_path, (json.dumps(visual, indent=2) + '\n').encode('utf-8'))
    report_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_RETURN_REVIEW.md'
    report = '''# U96 post-fusion learning V1 / B64: real learning, failed quality endpoint

The real downloaded archive passed independent integrity, saved-pixel and saved-state audits. All64 Adam updates committed. All preservation and dark-retention checks passed, but all four terminal blur-structure checks missed1%. The worker correctly returned QUALITY_STOP/exit2. U96 remains current; the failed B64 state is retained for research and migration, without app promotion or appended epochs.

## Exact evidence and result

Archive `outputs/cctv-codeformer-u96-postfusion-learning-v1-results.tar.gz`:1,166,039,684bytes, SHA256 `5f99df00c2eff645d3bc957606766753d3308aaa4e0e22da57ac28edeb8d858a`. Protocol SHA256 `11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903`. The three returned archive/checksum/export files agree. All3,455 regular members/1,308,001,394payload bytes,236 nonweight static files and protocol match. Gzip completion is verified. Successful export is evidence transfer, not model qualification.

| Terminal TRAIN-monitor structure screen | RAW gain | Delivered PNG gain | Requirement |
|---|---:|---:|---:|
| Global blur, whole image |0.0161526573%|0.0131317376%|at least1% each|
| Local blur, central128 region |0.0108894708%|0.0043591067%|at least1% each|

These percentages measure reduction in adjacent-residual error against aligned synthetic clean targets. They are not percentages of recovered identity or the older DGP feature metric. The authoritative failure is `snapshot_u64_quality_gates`; all24 unchanged historical guard cases and the new RAW/PNG preservation/clear-anchor/dark-illumination checks passed. All resource caps passed. Runtime was416.668seconds; export63.892seconds is not inferred here--use the exact returned export sidecar for timing. The final success result/invariant file is absent because the terminal quality stop occurs before that success-only section; all committed checkpoints, intermediate invariants and failure/status receipts remain preserved.

## What was independently checked

[Saved-pixel arithmetic companion](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_learning_v1_return_audit/saved_pixel_arithmetic_companion_v1.json>) closes the immutable first audit: all712 RAW/PNG pairs, flooring, teacher/parity associations, regional metrics, all six entire legacy/monitor gate dictionaries and256 objective reconstructions. Python3.13's changed builtin sum initially caused seven dictionary differences around1e-18; explicit left-to-right sums matching VM Python3.10 reproduce every gate exactly. No epsilon, threshold change or model recipe change resolves those audit-only differences. Original report and discrepancy list remain retained.

[Saved-state audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_learning_v1_state_audit/saved_state_audit_v1.json>) checked65 portable head checkpoints,64 postclip gradient sets/all4,644 arrays and65 safe CPU weights-only resume decodes. All64 updates, sampler exposures, LR/Adam steps and moments are consistent. Every one of36 trainable tensors differs at64; all eight output-projection gradient tensors are nonzero at every update. Independent NumPy Adam recurrence agrees within explicitly reported FP32 arithmetic bounds. These audit bounds do not modify scientific gates. No model/optimizer construction, forward, backward or optimizer step ran locally. Torch was used only for safe CPU saved-state decoding.

[Independent metadata review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_learning_v1_return_peer/independent_return_review_v1.json>) and [265-check crossread](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_learning_v1_return_audit/peer_state_and_metadata_crossread_v1.json>) agree with the archive, commit chain, counters and endpoint stop. Frozen539 tensor payloads are omitted from return, so unchanged frozen-weight claims remain source/runtime-receipt-bound; the exact originalR2/U96 files stay preserved locally and on the source stage.

## Actual images and learning diagnosis

Root personally viewed all30 planned sheets/all712 generated PNG cells, including every optimization exposure and all monitor/legacy snapshots. [Root visual companion](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_learning_v1_return_audit/root_visual_review_v1.json>) records every sheet/hash/observation. It closes pending flags without rewriting original manifests. The new faces remain visually essentially identical to U96. Existing softness and eye/teeth/skin/hat-detail target differences persist. No obvious new catastrophic artifact was seen. This is an unblinded assistant TRAIN review, not independent human FINAL or user acceptance.

At64, only1.06%–2.05% of monitor PNG RGB components differ from U96, depending on role. RAW mean absolute changes are around4.1e-5–8.1e-5 on the0–1 scale. These small differences align with the negligible visible gain. All36 head tensors moved, so the optimizer was not broken or unused. Norm clipping never activated: recorded gradient norms0.0303–0.5017 stayed below1.0. The LR warmed up over four updates, then stayed5e-5. Increasing clipping allowance cannot fix an inactive clip; this result does not justify simply appending more epochs or increasing LR.

Clear anchoring became smaller across the four fixed role-table passes while local/global paired objectives barely improved. A testable hypothesis is that sign-based L1 anchors oppose tiny corrective changes once exact U96 parity is left. The derivative of L1 stays sign-sized for any nonzero residual, while SmoothL1 decreases its derivative near zero. This mathematical fact motivates a distinct objective comparison; actual opposing head-gradient causation is not established. See [PyTorch2.9 L1](https://docs.pytorch.org/docs/2.9/generated/torch.nn.L1Loss.html) and [SmoothL1](https://docs.pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html).

The next proposed trial starts fresh from the same complete U96 and same initial new heads, retaining all data/order/seed/Adam/LR/64-update caps and preservation gates. It changes only the common clear/dark anchor criterion to SmoothL1 beta0.01; paired reconstruction/adjacent losses and coefficients stay unchanged. This is a prospective hypothesis, not a promised fix. Beta is a loss transition scale, not permission for0.01 mean error or a flat dead zone. Preserve old L1 diagnostics for comparison. No training launch, B64 continuation or new packet is established by this report.

## Research boundaries and continuation

New96 cases are paired synthetic TRAIN photographs, split16 optimization/8 monitor photographs with no verified subject separation. Monitor has now informed design and must be labeled reused/adaptive TRAIN evidence in any follow-up. Legacy dark uses an R2 behavior teacher, not an aligned clean target. No native CCTV/wild/FINAL output is newly evaluated here; no ethnicity, exact identity or Zamboanga-performance claim follows. Preserve all original labels, exclusions, source terms, models, states and gate failures.

The user manually runs any separately reviewed finite successor on the L4 in tmux. Returned outputs must again be independently audited and all planned sheets reviewed. Useful broader DEV/native evidence, user acceptance, independent FINAL and full app qualification remain outstanding. Optional protected-surroundings preparation stays separately evaluated with original/protected pixels/confirmation/bypass; paired evidence does not support always-on blur. Restoration remains first and later completion/covering-family scope remains required. The full active goal is incomplete.
'''
    # Avoid substituting unverified export timing into the report.
    report = report.replace('Runtime was416.668seconds; export63.892seconds is not inferred here--use the exact returned export sidecar for timing.',
                            'Worker runtime was416.668seconds, within the sealed stop limits.')
    write(report_path, report.encode('utf-8'))
    receipt = {'complete': True, 'UTC': visual['UTC'], 'report': pin(report_path),
               'root_visual': pin(visual_path), 'audits': visual['audits'],
               'scientific_trial_pass': False, 'model_qualified': False,
               'next_trial_status': 'DESIGN_ONLY_NOT_YET_SEALED',
               'no_new_model_training_VM_calls': True}
    write(OUT / 'return_publication_v1.json', (json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'complete': True, 'review': str(report_path), 'all30_sheets_reviewed': True,
                      'scientific_trial_pass': False, 'model_qualified': False}))


if __name__ == '__main__':
    main()
