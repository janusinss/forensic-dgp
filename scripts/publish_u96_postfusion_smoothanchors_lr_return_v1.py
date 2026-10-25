"""Publish actual LR return review; preserve source records and document history."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit'


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def write(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def main():
    pixel_path = OUT / 'saved_pixel_audit_v1.json'
    peer_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_peer/independent_return_review_v1.json'
    state_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_state_audit/saved_state_audit_v1.json'
    cross_path = OUT / 'peer_state_and_metadata_crossread_v1.json'
    trajectory_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_comparison/independent_trajectory_review_v1.json'
    pixel, peer, state, cross = [read(p) for p in (pixel_path, peer_path, state_path, cross_path)]
    assert pixel['complete'] and pixel['pixel_phase_complete'] and pixel['independent_saved_pixel_integrity_pass']
    assert pixel['pilot_quality_gates_pass'] is True and pixel['scientific_trial_pass'] is True
    assert peer['complete'] and peer['independent_return_metadata_audit_pass']
    assert peer['authoritative_status'] == 'FINITE_B64_LEARNING_SCREEN_PASSED_NOT_QUALIFIED'
    assert peer['optimizer_updates'] == peer['optimizer_update_attempts'] == 64
    assert peer['worker_exit'] == 0 and not peer['timeout'] and not peer['optimizer_outcome_unknown']
    assert state['complete'] and state['audited_complete_checkpoint_prefix_pass']
    assert state['durable_committed_trajectory_fully_covered'] and state['head_checkpoint_sets_verified'] == 65
    assert state['postclip_gradient_sets_verified'] == 64 and state['verified_fresh_Adam_state_at0']
    assert state['same_fresh_U96_seeded_head_initializer_as_B2']
    assert state['artifact_outcome']['confirmed_optimizer_updates'] == 64
    assert state['artifact_outcome']['scientific_status'] == 'FINITE_SCREEN_PASSED_NOT_QUALIFIED'
    assert cross['complete'] and cross['independent_crossread_pass']
    assert len(cross['checks']) == 399 and not cross['failures']
    trajectory = read(trajectory_path)
    assert trajectory['complete'] and trajectory['independent_trajectory_review_pass']
    assert len(trajectory['checks']) == 175 and not trajectory['failures']
    assert trajectory['all48_published_gain_points_match_actual_saved_pixels']
    assert trajectory['all120_reference_RAW_PNG_hashes_exact']
    document_names = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                      'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
    for path in (OUT / 'root_visual_review_v1.json', OUT / 'return_publication_v1.json',
                 ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_RETURN_REVIEW.md'):
        assert not path.exists() and not path.is_symlink(), path
    for name in document_names:
        assert (ROOT / name).is_file() and not (ROOT / name).is_symlink()
        assert not (ROOT / (name + '.lr-return-v1.pending')).exists()
    manifest_path = OUT / 'visual_manifest_v1.json'
    manifest = read(manifest_path)
    assert manifest['complete'] and manifest['planned_sheets'] == 30
    assert manifest['generated_PNG_cells'] == 712 and manifest['exact_saved256_cells'] == 952
    assert manifest['all_generated_PNGs_exactly_once']
    readout_notes = [
        'Red-cap portrait: small shading and facial-edge changes at64, while eyes/mouth remain soft. Cap lettering still differs from paired truth; no conspicuous consistent face-detail jump.',
        'Fur-hat smile: small tonal changes, similar eye/teeth readability; smoothing and differences from clean skin/teeth remain.',
        'Microphone portrait: subtle complexion and mouth-edge changes; fine skin and eye detail remain reduced relative to target.',
        'Blond-costume portrait: lip and nose shading change slightly; fine eyes/skin remain smooth. No obvious new catastrophic artifact.',
        'Suit portrait: slight contour/shading changes through64; eye/nose/mouth readability remains close to U96.',
        'Mild-turn portrait: small nose, lip and skin changes; no clear large improvement in visible structure.',
        'Glasses portrait: frames and eyes remain readable and broadly stable; minor local shading changes, existing softness persists.',
        'Cap portrait: warm appearance and facial outlines persist, with slight eye/lip changes and reduced fine target texture.',
        'Historical stronger Gaussian cases: target eye, mouth and skin differences persist; small tone changes, no consistent clear sharpening jump.',
        'Historical stronger Gaussian cases: some edge and skin changes; existing target differences remain in eyes/teeth/skin. Glasses remain present.',
        'Historical clear bank: second column is original CodeFormer behavior teacher, not clean truth. Small facial/tonal changes; no obvious new destructive change.',
        'Historical clear bank: original behavior teacher only; glasses and expression remain broadly stable with small skin/eye/mouth changes.',
        'Historical dark bank: R2 behavior teacher is not aligned clean truth. Existing brightening persists; small tone changes, fine structure remains similar.',
        'Historical dark bank: R2 teacher only. Illumination and readable outlines remain, with subtle local changes rather than a large detail gain.',
    ]
    optimization_features = [
        'Smile and teeth; eye/skin detail remains smoother than paired target.',
        'Glasses and mouth; small changes, fine texture remains reduced.',
        'Eyes, lips and face contour; incumbent softness remains.',
        'Hat/glasses smile; target creases and teeth detail remain different.',
        'Glasses and beard; fine skin/moustache detail remains similar.',
        'Mild-turn portrait; lip/eye shading changes subtly.',
        'Eyes and lips; small changes without conspicuous structural gain.',
        'Glasses above forehead; facial creases remain softer than target.',
        'Beard/smile portrait; eye and tooth texture remains similar.',
        'Beard portrait; subtle skin/edge changes and reduced target texture.',
        'Cap portrait; fine eyes/lips remain soft.',
        'Front portrait; skin/eye smoothing remains.',
        'Mild-turn portrait; small eye/lip/shading changes.',
        'Front portrait; skin texture remains reduced versus clean target.',
        'Smile portrait; subtle lip and skin changes, no large detail jump.',
        'Glasses portrait; frames, eyes and mouth remain broadly stable.',
    ]
    names = ['readout_%02d.png' % n for n in range(1, 15)] + ['optimization_%02d.png' % n for n in range(1, 17)]
    assert [Path(row['file']['path']).name for row in manifest['sheets']] == names
    observations = []
    for index, row in enumerate(manifest['sheets']):
        assert pin(ROOT / row['file']['path']) == row['file']
        count = sum(cell['generated_return_PNG'] for cell in row['cells'])
        assert count == (28 if index < 14 else 20)
        note = readout_notes[index] if index < 14 else (
            'Four optimization exposures shown before their declared commits; small tonal/edge changes, no obvious new catastrophic artifact. '
            + optimization_features[index - 14])
        observations.append({'file': row['file'], 'personally_viewed_by_root': True,
                             'method': 'view_image original detail; unchanged saved256 RGB cells',
                             'generated_cells_reviewed': count, 'observation': note})
    visual = {'complete': True, 'format': 'smoothanchors-lr-B64-root-personal-visual-review-v1',
              'UTC': datetime.now(timezone.utc).isoformat(), 'reviewer': 'root assistant, unblinded TRAIN review',
              'visual_manifest': pin(manifest_path), 'all_planned_sheets_reviewed': True,
              'sheets_reviewed': 30, 'generated_PNG_cells_reviewed': 712, 'exact_saved256_cells': 952,
              'observations': observations, 'consistent_conspicuous_face_detail_gain_over_U96': False,
              'subtle_tonal_edge_changes_seen': True, 'obvious_new_catastrophic_artifacts_seen': False,
              'finite_TRAIN_screen_pass': True, 'model_qualified': False,
              'decision': 'Retain current U96. Advance exact terminal u64 to frozen local DEV comparison and user testing after source/export/runtime checks; no promotion or new training.',
              'not_independent_human_FINAL_or_user_acceptance': True,
              'closes_original_pending_flags_without_overwriting_reports': True,
              'audits': [pin(p) for p in (pixel_path, peer_path, state_path, cross_path, trajectory_path)],
              'new_model_optimizer_training_VM_browser_calls': 0}
    visual_path = OUT / 'root_visual_review_v1.json'
    write(visual_path, (json.dumps(visual, indent=2) + '\n').encode())
    report_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_RETURN_REVIEW.md'
    report = '''# U96 learning-rate comparison: finite TRAIN screen passed

The exact64-update comparison passed all four unchanged1% TRAIN structure endpoints and every historical/new preservation and dark check. Global RAW/PNG gains are1.628624%/1.627216%; localized-center gains1.239396%/1.210135%. Previous maximumLR5e-5 produced PNG0.919274%/0.664661%; this fresh1e-4 trial used the same completeU96 initializer, head seed, SmoothL1criterion, ordered data,64-update cap and gates. This establishes progress in the finite paired screen. It does not establish useful native CCTV restoration, identity recovery or a qualified successor.

## Verified returned evidence

Archive `outputs/cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-results.tar.gz` is1,166,628,159bytes, SHA256`7e5208d2ff69713c834c21fb466020ae54d7ee68b84795e2c044ffc11847041e`. Checksum/export/protocol agree. ProtocolSHA`8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9`. All3,469 regular members/1,311,565,858payloadbytes and gzip/footer/static namespace were checked;250 local sealed files were rehashed. Worker424.916seconds, supervisor427.727seconds, exit0/no timeout.64confirmed Adam commits/64attempts/no unknown outcome, result present/failure absent. Authoritative status is `FINITE_B64_LEARNING_SCREEN_PASSED_NOT_QUALIFIED`.

[Independent pixel audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/saved_pixel_audit_v1.json>) verifies all712 RAW-to-floorPNG pairs, all regional metrics, all12 entire gate dictionaries and256 objective reconstructions. Maximum GPU-to-CPUfloat64 objective reduction difference5.15735e-8, CPUfloat32 difference4.47035e-8. These disclosed arithmetic differences do not change thresholds or quality gates. [State audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_state_audit/saved_state_audit_v1.json>) verifies65 head/resume checkpoints,64 gradients and all64 Adam recurrences, fresh initialization, sampler/RNG and exactLRwarmup. All36 trainable tensors changed; totalheadL2movement0.84122960 versus prior0.57140686; clipping never activated. TerminalheadcanonicalSHA`6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be`. Frozen539 values were omitted from return; their original files/source/runtime associations are verified, not independently redecoded frozen return values.

[Metadata review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_peer/independent_return_review_v1.json>) passed76,609 checks. [399-check crossread](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/peer_state_and_metadata_crossread_v1.json>) joins actual pixel/state/metadata chains. The separate CPU weights-only resume decoding constructed no model or optimizer. Other audit paths used NumPy/Pillow saved evidence only; no new training or VM call occurred.

## Visual interpretation and decision

[Root review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/root_visual_review_v1.json>) covers all30 planned sheets,952 unchanged256cells/all712 generatedPNGcells exactly once. Small tonal, shading and facial-edge changes are visible. Existing softness and clean-target differences remain in eyes, teeth and skin. No obvious new catastrophic artifact was observed; no clear consistent conspicuous facial-detail jump over U96 is established. This is unblinded assistant TRAIN review, not user acceptance or independent human FINAL. Legacy clear teachers are original CodeFormer behavior and legacy dark teachers are R2 behavior, not clean identity truth.

[Saved trajectory plot](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_comparison/structure_trajectory_v1.png>) and its independent175-check review bind all48 gain points and120 identical frozenU96 reference RAW/PNG hashes. Lines connect measured snapshots only, with no prediction of continued improvement. Monitors have been reused after multiple designs and are adaptive TRAIN evidence, not unseen validation or verified independent people.

Advance the exactterminalu64 state to an inference-only fixedDEV comparison against U96, after frozen adapter/export review and a quantified CPU/L4 witness. Retain U96 as current. Do not append training, retune gates, choose a retrospective earlier checkpoint or promote from this screen. The same fixed68DEV inputs/order/terms/labels and insufficient zero-call controls can be reused under their new data-only admission; no new selection or FINAL inspection is permitted. Report paired synthetic and native unpaired sources separately. A visually subtle gain can still fail practical usefulness on DEV.

## Scope remaining

User testing, broader DEV/native preservation/usefulness, independent FINAL and fullapp qualification remain required. Native/wild images without aligned targets have no paired accuracy claim; no ethnicity, Zamboanga performance or exact hidden identity is inferred. Optional protected-surroundings preparation remains separate: exact protected input pixels, immutable original previews, correction/confirmation/bypass. Always-on blur and arbitrary noise/mosaic copying are not justified. Restoration comes first; later completion/removal-mask covering-family work remains in the full ACTIVE goal.
'''
    write(report_path, report.encode())
    note = '''# 22 October2026: LR-only U96 return passed finite TRAIN screen; frozen DEV comparison next

[Returned review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_RETURN_REVIEW.md>):64actual Adam commits independently audited. All four1%TRAIN structure endpoints passed: globalRAW/PNG1.628624%/1.627216%,localcenter1.239396%/1.210135%. All historical/new preservation/dark checks passed. Authoritative finite-screen status is passed/not qualified, workerexit0/resultpresent/no failure or unknown optimizer outcome. Old failedB1/B2states and all originalmodels/splits/terms/failure records remain exact and retained.

All712RAW/PNG pairs,256objectives,65head/resume checkpoints/64gradients were independently audited and399crosschecks passed. Root personally reviewed all30 planned sheets/all712 generatedcells: subtle tonal/edge changes, no obvious new catastrophicartifact, existing softness remains, no clear consistent conspicuous detail jump. U96 stays current; this adaptiveTRAIN pass is not practical DEV/native/user/FINAL acceptance.

Next: source-reviewed frozenCPU inference export of exactterminalu64 and quantifiedCPU/L4witness, then original fixed68DEV comparison againstU96 with sameinputs/order/terms/labels and zero-call insufficientcontrols. No new training, failedstatecontinuation, automatic promotion, data relabeling or FINALinspection. Separate RAW/PNG and nativeunpaired/pairedsynthetic evidence. Optional reviewed protected-surroundings preparation and later separatecompletion scope remain unchanged. The fullgoal remains ACTIVE/incomplete. Earlier dated records are preserved below; this entry supersedes their nextaction only.

---

'''.encode()
    documents = []
    for name in document_names:
        path = ROOT / name
        before = pin(path)
        original = path.read_bytes()
        pending = ROOT / (name + '.lr-return-v1.pending')
        write(pending, note + original)
        assert path.read_bytes() == original
        os.replace(pending, path)
        assert path.read_bytes()[len(note):] == original
        documents.append({'before': before, 'after': pin(path),
                          'preserved_suffix_offset_bytes': len(note), 'old_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
               'report': pin(report_path), 'visual_review': pin(visual_path), 'documents': documents,
               'finite_TRAIN_screen_pass': True, 'model_qualified': False,
               'app_default_changed': False, 'new_model_optimizer_training_VM_calls': 0,
               'audits': [pin(p) for p in (pixel_path, peer_path, state_path, cross_path, trajectory_path)]}
    write(OUT / 'return_publication_v1.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'report': str(report_path), 'all30_root_visuals_reviewed': True,
                      'finite_TRAIN_screen_pass': True, 'model_qualified': False}))


if __name__ == '__main__':
    main()
