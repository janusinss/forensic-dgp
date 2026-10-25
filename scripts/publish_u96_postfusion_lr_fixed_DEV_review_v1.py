"""Publish the completed fixed-DEV review; retain all earlier evidence verbatim.

This records the root's personal view_image review of the 17 exact-cell sheets.
No inference, training, model promotion, or quality threshold is performed here.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1'
OUT = ROOT / (PREFIX + '_DEV_aggregate_audit_v1')
REPORT = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_LR_U64_DEV_REVIEW.md'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def exact(path, sha):
    need(pin(path)['sha256'] == sha, 'Expected evidence SHA: ' + str(path))
    return read(path)


def put_json(path, data):
    with path.open('xb') as stream:
        stream.write((json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))


def main():
    need(not REPORT.exists() and not (OUT / 'root_visual_review_v1.json').exists()
         and not (OUT / 'DEV_publication_v1.json').exists(), 'New publication namespace required')
    pixel_path = OUT / 'saved_pixel_audit_v1.json'
    pixel = exact(pixel_path, '0cda9111b7b8b97933669dd532aece49f085695859b3ea0253cc1b35e3579f97')
    numeric_path = OUT / 'numeric_summary_v1.json'
    numeric = exact(numeric_path, '4a08c9a04fe6e1f5f9585d13096cc2348213c3f724b7aae0a68c73149922b369')
    metadata_path = ROOT / (PREFIX + '_DEV_tail_metadata_audit/independent_terminal_metadata_v1.json')
    metadata = exact(metadata_path, '424bd8109c6b4636f108c0e2e7c300737a69e3535744427bc610ca42773a2c76')
    feedback_path = ROOT / (PREFIX + '_page_saved_audit/user_development_feedback_v1.json')
    feedback = exact(feedback_path, 'fa15b53757ad6d4dae65637f4b7a0df3a9d50371ba11e6153fe28993182990e7')
    ui_path = ROOT / (PREFIX + '_page_saved_audit/saved_UI_evidence_audit_v1.json')
    ui = exact(ui_path, 'a3983e8156d6506fee472f9a19ae8efa732200454d91d6c46b04c136418d7016')
    browser_path = ROOT / (PREFIX + '_page_saved_audit/root_browser_visual_review_v1.json')
    browser = exact(browser_path, 'b0948d438003645e9a0bd9257aa6b3cdf1322f2cd623862c7276d9bd0555c7ba')
    need(pixel['complete'] and pixel['pixel_phase_complete'] and pixel['independent_saved_pixel_integrity_pass'], 'Saved-pixel audit must pass')
    need(len(pixel['checks']) == 3630 and all(c['pass'] for c in pixel['checks']) and not pixel['failures'], 'All actual integrity checks')
    need(pixel['outputs_verified'] == 136 and pixel['aggregate_reservations_including_failed_pre_call'] == 137
         and pixel['all136_native_to_unclipped_RAW_floorPNG_exact'] and pixel['all136_outside_support_exact'], 'Exact aggregate output scope')
    need(metadata['complete'] and metadata['independent_terminal_metadata_audit_pass'], 'Independent terminal metadata audit')
    need(numeric['integrity_pass'] and numeric['source_saved_pixel_audit'] == pin(pixel_path), 'Derived numerical source binding')
    need(ui['complete'] and ui['saved_UI_evidence_audit_pass'] and browser['root_personal_screenshot_review_complete'], 'Separate functional page evidence')
    need(feedback['verbatim_answer'] == 'The new u64 looks clearer and preserves appearance', 'Exact human feedback')
    manifest_path = OUT / 'visual_manifest_v1.json'
    need(pin(manifest_path) == pixel['visual_manifest'], 'Exact complete visual manifest')
    manifest = read(manifest_path)
    need(manifest['planned_sheets'] == 17 and manifest['case_rows'] == 68
         and manifest['generated_PNG_cells'] == 136 and manifest['total_exact256_cells'] == 240, 'Full planned visual scope')
    notes = {
        'filter_DEV_01.png': 'Clear and global/anisotropic Gaussian rows: both arms retain similar overall appearance. Small tonal/edge differences do not establish a consistent visible detail jump. Both differ from clean fine texture and eye shape; strong blur also changes the bow detail.',
        'filter_DEV_02.png': 'Repeated-row artifacts, globally mosaiced bow/background and locally blurred face remain visible in both arms. u64 makes small tonal changes; localized face blur is not solved.',
        'filter_DEV_03.png': 'Local mosaic remains central and obvious. The second photograph has similar clear and Gaussian results across arms; generated eyes/mouth/texture differ from the paired target in both.',
        'filter_DEV_04.png': 'Anisotropic rows look closely similar across arms. Repeated-row and global-mosaic defects remain, including simplified texture/geometry. No conspicuous new u64 artifact is apparent.',
        'filter_DEV_05.png': 'Local Gaussian softness and local mosaic remain. Makeup/eye detail in the clear and mildly blurred third photograph is changed by both models compared with the clean reference; u64 does not resolve this inherited appearance limitation.',
        'filter_DEV_06.png': 'The third photograph retains broadly similar face shape across arms, with small tonal/texture changes. Hat, makeup, mouth and repeated-row distortions remain relative to the paired target.',
        'filter_DEV_07.png': 'Global/local mosaic and localized face blur remain prominent. The clear glasses photograph retains frames across arms but both generated faces differ in fine eyes/texture from the paired reference.',
        'filter_DEV_08.png': 'Gaussian and anisotropic glasses rows are closely similar across arms. No consistent conspicuous new facial-detail improvement is visible; both retain smoothness and some frame/eye differences from clean.',
        'filter_DEV_09.png': 'Repeated rows, mosaic over glasses/nose/mouth and local Gaussian softness persist. The small u64 changes do not remove central degradation.',
        'native_QMUL-SurvFace_01.png': 'Four unpaired rows show subtle u64 tonal/texture differences. Severe lighting and soft/generated eye detail remain. No aligned clean truth exists; the sheets do not establish native identity fidelity or a consistent detail improvement.',
        'native_QMUL-SurvFace_02.png': 'Glasses and turned-face rows look close across arms; both render details unsupported by an aligned reference, including weak/closed-looking eye regions. No consistent conspicuous u64 advantage or exact identity claim follows.',
        'native_ChokePoint_01.png': 'Two observed frames per each of the first two development identities: arms appear closely similar, with small tone/edge changes. Existing softness and appearance deviations from the observed input remain.',
        'native_ChokePoint_02.png': 'Development identities three/four: overall appearance and clear glasses are similar across arms. u64 differences are small; no new large artifact is apparent at exact256, and no clean-reference accuracy is claimed.',
        'native_ChokePoint_03.png': 'Development identities five/six: arms remain visually similar with subtle contrast/texture changes. Dark facial regions remain uncertain; generated clarity cannot be equated with recovered detail.',
        'native_ChokePoint_04.png': 'Development identities seven/eight: glasses, hair and broad outline remain similar across arms. Softness, glare and weak visible eye detail remain. No consistent conspicuous native improvement is established.',
        'native_ChokePoint_05.png': 'Development identities nine/ten: u64 has small tonal/edge differences. Existing soft eyes and glare remain; no additional severe defect is apparent relative to U96 in these rows.',
        'native_ChokePoint_06.png': 'Development identities eleven/twelve: closely similar generated facial structure across arms with small texture changes. Glasses/glare and input uncertainty remain; native unpaired evidence cannot establish true identity recovery.',
    }
    reviews = []
    for sheet in manifest['sheets']:
        path = ROOT / sheet['file']['path']
        need(pin(path) == sheet['file'], 'Exact personally viewed sheet')
        need(path.name in notes and len(sheet['case_ids']) == 4, 'All17 explicit observations')
        reviews.append({'file': pin(path), 'cohort': sheet['cohort'], 'case_ids': sheet['case_ids'],
                        'generated_cells': sum(c['generated_output'] for c in sheet['cells']),
                        'personally_viewed_by_root': True, 'observation': notes[path.name]})
    need(len(reviews) == 17 and sum(r['generated_cells'] for r in reviews) == 136, 'All planned generated cells reviewed')
    review = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
              'root_personal_visual_review_complete': True, 'visual_manifest': pin(manifest_path),
              'sheets_reviewed': 17, 'case_rows_reviewed': 68, 'generated_PNG_cells_reviewed': 136,
              'total_exact256_cells_reviewed': 240, 'reviews': reviews,
              'method': 'Root personally viewed each full sheet through view_image at original detail; fixed order, unblinded descriptive review, not independent FINAL.',
              'conclusion': 'Modest numerical development progress and positive human feedback coexist with mixed paired metrics and visually small changes. Local blur/mosaic and inherited appearance limitations remain. No consistent conspicuous native detail improvement is established.',
              'user_feedback': pin(feedback_path), 'pixel_audit': pin(pixel_path), 'metadata_audit': pin(metadata_path),
              'historical_pending_flags_preserved': True, 'this_companion_closes_root_DEV_sheet_review_only': True,
              'model_qualified': False, 'app_default_changed': False, 'U96_incumbent_retained': True,
              'new_training_optimizer_or_neural_calls': 0}
    put_json(OUT / 'root_visual_review_v1.json', review)
    profiles = numeric['paired36_profiles']
    table = '\n'.join('| {profile} | {PNG_MAE_delta:+.6f} | {PNG_whole_adjacent_gain_percent:+.4f}% | {PNG_central_adjacent_gain_percent:+.4f}% | {PNG_PSNR_delta_dB:+.4f} | {PNG_SSIM_delta:+.6f} |'.format(**r) for r in profiles)
    report = f'''# U96 versus post-fusion LR u64: fixed development review

22 October 2026. **Experimental candidate; U96 remains the incumbent.** The new u64 shows modest numerical progress and received positive user feedback, but this review does not establish consistent conspicuous native facial-detail improvement or broad model qualification.

## What was completed

The fixed 68-case cohort contains 36 paired synthetic cases (four photographs, nine profiles), eight native QMUL-SurvFace cases and 24 native ChokePoint cases. Terms, input-only labels, original order, appearance flags and 40 excluded zero-call controls remain unchanged. The original 47 saved outputs were retained and the distinct recovery generated only the remaining 89: 136 known saved returns, 137 reservations including the earlier failed pre-call reservation. All 454 original files remain unchanged. The original Windows status-write failure and historical unknown reservation are retained; the recovery exited zero without timeout or unknown tail outcome. No optimization or new training occurred locally.

Independent metadata checks passed. The saved-output audit passed all 3,630 integrity checks, verifying every native512-to-unclipped/clamped/bilinear RAW output and exact floor-quantized PNG. Padding outside observed support remains exact for every output. Root personally reviewed all 17 planned sheets, 68 rows and 136 generated cells at exact256. These are integrity and review facts, not a new quality gate or a preservation qualification for all possible inputs.

## Paired synthetic findings

The table uses u64 minus U96 for MAE/PSNR/SSIM; negative MAE and positive PSNR/SSIM favor u64. Adjacent-error reduction is relative to U96 against the aligned target. Whole-image and central-region values are separate; the central local-blur improvement is much smaller than the whole-image value. Each row averages only four DEV photographs. These metrics are separate from the small adaptive TRAIN feature-error gates.

| Profile | PNG MAE difference | Whole adjacent-error reduction | Central reduction | PSNR difference dB | SSIM difference |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

MAE was lower in seven of nine profile means, whole adjacent error in nine, central adjacent error in seven, PSNR higher in seven and SSIM higher in six. Strong Gaussian blur and global mosaic worsened MAE, PSNR and SSIM. Global/local mosaic worsened central adjacent error; local mosaic also worsened SSIM. No thresholds were changed or new aggregate pass invented. Clear-image paired error improved on average, but inherited fine-feature and makeup deviations remain visible in both models. Localized blur and mosaic remain evident; u64 does not solve strong anonymizing degradation.

## Native unpaired findings

QMUL-SurvFace and ChokePoint remain separate sources without aligned clean targets. Candidate-versus-U96 PNG observed-region MAE averaged {numeric['native32_source_separated_drift'][0]['candidate_vs_U96_PNG_observed_MAE_mean']:.8f} for QMUL and {numeric['native32_source_separated_drift'][1]['candidate_vs_U96_PNG_observed_MAE_mean']:.8f} for ChokePoint. These measure output change only. Native PSNR/SSIM, recovered identity, ethnicity and Zamboanga performance are not claimed.

Root found small tonal/edge changes and broadly similar generated structure across arms, with existing softness, glare, lighting, eye and turned-face limitations. No consistent conspicuous native detail improvement was established. Existing input-only labels were not changed after seeing results. Independent FINAL and broader preservation remain outstanding.

## Human feedback and local test

The user reported: “The new u64 looks clearer and preserves appearance.” This is positive unblinded development feedback. The exact uploaded case was not identified, so no measured case binding or independent FINAL acceptance is inferred.

[Experimental comparison page](http://127.0.0.1:59807/restoration-postfusion-test) and [test guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LR_U64_TEST.md>) compare identical prepared inputs with U96 and the complete R2+U96+terminal-u64 candidate. The source-reviewed frozen CPU model has a separate four-case L4 correspondence witness. The experimental browser flow passed 375/768/1280 layouts, actual PNG/ZIP download readback, input-review refusal and stale-result clearing. Its saved outputs matched the corresponding fixed-DEV outputs. A concurrent409 timing assertion was not established and its earlier test stop remains retained. This is separate-page functionality, not the complete application or model qualification.

## Decision and next work

Retain u64 as the preferred experimental challenger for further review, with U96 as the unchanged incumbent/control. Do not switch the main app default, append epochs or auto-launch another pilot based on this result. Retain all original models, failed states, optimizer/RNG/sampler evidence, source terms and splits. The small TRAIN screen passed; fixed DEV demonstrates mixed modest progress, not a failed export and not a fully qualified successor.

Next finish broader clear/degraded preservation and source-separated development review, then freeze an independent FINAL evaluation protocol before held-out outputs are examined. The protected-surroundings workflow remains optional and separate: immutable original upload, every protected input pixel unchanged, explicit separate Protect/Apply painting with erase, preview/correction/confirmation/bypass, blur only where permitted with an outside transition. Separate Apply painting is still an implementation gap; always-on blur and arbitrary noise/mosaic copying are not justified. Completion/removal-mask model development remains deferred until restoration is satisfactory. The active goal remains incomplete.

## Evidence

[Saved-pixel audit](<C:/xampp/htdocs/YEAR 4/Testing/{pixel_path.relative_to(ROOT).as_posix()}>), [independent terminal metadata](<C:/xampp/htdocs/YEAR 4/Testing/{metadata_path.relative_to(ROOT).as_posix()}>), [source-separated numerical summary](<C:/xampp/htdocs/YEAR 4/Testing/{numeric_path.relative_to(ROOT).as_posix()}>), [all17 exact-cell sheets](<C:/xampp/htdocs/YEAR 4/Testing/{manifest_path.relative_to(ROOT).as_posix()}>), and [root visual review](<C:/xampp/htdocs/YEAR 4/Testing/{(OUT/'root_visual_review_v1.json').relative_to(ROOT).as_posix()}>). Earlier pending flags remain immutable; the companion root review closes only the documented DEV-sheet review.
'''
    with REPORT.open('xb') as stream:
        stream.write(report.encode('utf-8'))
    prefix = '''# 22 October 2026: u64 experimental comparison and fixed DEV review completed

[Fixed development review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LR_U64_DEV_REVIEW.md>): all136 outputs audited and all17 planned sheets personally reviewed. The original47 outputs and454 files remain exact; recovery generated only89 remaining outputs, exited0/no timeout.136 known returns/137 reservations preserve the original failed pre-call/unknown record. No local optimization occurred. Paired36 and native32 evidence remain separate; original labels/terms/40 zero-call controls remain unchanged.

The small TRAIN screen passed. Fixed DEV shows modest mixed progress: PNG MAE lower7/9 profile means, adjacent whole error lower9/9, central lower7/9; PSNR higher7/9 andSSIM6/9. Strong Gaussian/global mosaic worsened MAE/PSNR/SSIM; global/local mosaic worsened central adjacent error. Local blur/mosaic and inherited fine-feature/makeup limitations remain. Root found small tonal/edge changes without consistent conspicuous native detail improvement; no native paired/identity/Zamboanga claim follows.

User feedback: “The new u64 looks clearer and preserves appearance.” This is positive subjective DEV preference with no exact case binding or independentFINAL acceptance. Frozen u64 is available on the separate experimental comparison page; bundled inline Playwright layouts375/768/1280, PNG/ZIP readback, refusals and stale-result behavior passed. Full-app and concurrent409 behavior are not established by that test. Keep u64 as the preferred experimental challenger and U96 as the unchanged incumbent/control; no main-default promotion or new training follows.

Next: broader preservation/source-separated DEV review and an independentFINAL protocol before held-out outputs. Optional protected-surroundings preparation still needs separate Protect/Apply painting and erase; preserve every protected input pixel and all outside user-permitted Apply, rebuild from original, invalidate confirmation after edits, retain bypass. Always-on blur is unjustified; restoration remains first and completion later. All original models/splits/provenance/failures remain retained. The goal stays ACTIVE/incomplete. Earlier dated records remain exact history below; this entry supersedes next actions only.

---

'''.encode('utf-8')
    documents = []
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md']:
        path = ROOT / name
        old = path.read_bytes()
        before = pin(path)
        path.write_bytes(prefix + old)
        after = path.read_bytes()
        need(after[len(prefix):] == old, 'Exact preserved document suffix')
        documents.append({'before': before, 'after': pin(path), 'preserved_suffix_offset_bytes': len(prefix), 'old_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'publisher': pin(Path(__file__)),
               'report': pin(REPORT), 'root_visual_review': pin(OUT / 'root_visual_review_v1.json'),
               'saved_pixel_audit': pin(pixel_path), 'numeric_summary': pin(numeric_path), 'metadata_audit': pin(metadata_path),
               'user_feedback': pin(feedback_path), 'UI_saved_evidence': pin(ui_path), 'browser_root_review': pin(browser_path),
               'documents': documents, 'model_qualified': False, 'app_default_changed': False,
               'preferred_experimental_challenger': 'postfusion_LR_u64', 'unchanged_incumbent': 'U96',
               'goal_complete': False, 'neural_optimizer_training_VM_calls': 0}
    put_json(OUT / 'DEV_publication_v1.json', receipt)
    print(json.dumps({'complete': True, 'report': pin(REPORT), 'publication': pin(OUT / 'DEV_publication_v1.json'), 'documents': len(documents)}))


if __name__ == '__main__':
    main()
