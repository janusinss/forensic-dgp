"""Record root's completed original-resolution inspection; no image inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_normal_tone_training_v1_return_review'
NOTES = [
    'ChokePoint native: facial layout remains recognizable across the three restorers; historical AdaIN is usually brighter, new50 is intermediate. Downward gaze remains difficult. No aligned target; correctness cannot be inferred.',
    'ChokePoint native: glasses, hair and broad expression remain visible, while soft texture and illumination vary. New50 gives no clear general structural advantage over historical AdaIN. Unpaired only.',
    'FFHQ blur: both adaptations suppress the original brittle texture; new50 resembles AdaIN50. Hair, hats and glasses remain reconstructed rather than exact, with smoother skin and changed tone versus clean targets.',
    'FFHQ blur: clean targets expose missing or changed fine hair, makeup and accessory detail. New50 is often pleasant and similar to AdaIN50, but facial shading and texture differ from targets.',
    'FFHQ blur: adapted faces are smoother than untouched outputs. New50 retains broad expression but does not restore all target skin and hair detail; some light faces look flatter.',
    'FFHQ blur: visible eyes and mouth become clearer than the degraded crop, while hairstyles, cheek texture and tone differ from targets. New50 offers little distinct structural change over AdaIN50.',
    'Transition blur to clear: severe blur produces convincing yet changed details. On clear controls, new50 changes skin shading and smoothness despite adequate original information.',
    'FFHQ clear: broad feature positions remain stable in many rows, but new50 changes contrast, facial colour and fine texture. A visually small change is not exact clear-image preservation.',
    'FFHQ clear: clean targets and originals are close; both adaptations alter skin rendering. New50 changes warm/cool balance and lip or cheek contrast without a restoration need.',
    'FFHQ clear: new50 keeps recognizable structure but adds tone and smoothing changes. Bright faces and dark backgrounds make those changes visible; no broad clear-control advantage.',
    'Clear to compound: clear inputs show unnecessary photometric changes. Compound outputs improve plausibility over brittle original reconstructions, but can remain dark or low contrast versus targets.',
    'FFHQ compound: new50 generally removes distracting original reconstruction texture, while eyes, expression, accessories and exposure still differ from clean targets. New50 remains close to historical AdaIN.',
    'FFHQ compound: many new50 faces are easier to view than original0. Some skin and background remain low contrast; hats, glasses and makeup are not exact target reconstructions.',
    'FFHQ compound: smoother facial surfaces and clearer broad features are visible; target skin detail, fine expression and lighting remain imperfect. New50 is often slightly darker than historical AdaIN.',
    'FFHQ compound: severe inputs still lead to changed facial details and accessories. New50 reduces brittle texture but leaves dark, smooth or generic-looking regions relative to clean targets.',
    'Compound to JPEG: compound gains do not extend to unchanged targets. JPEG controls have clear visible features already; new50 alters their skin tone, contrast and texture.',
    'FFHQ JPEG: broad appearance is retained in many rows, but cheek colour, brightness and smoothing change unnecessarily. Historical AdaIN and new50 are similar; neither is exact original behaviour.',
    'FFHQ JPEG: new50 renders faces smoothly and often attractively, but clean-target texture and shading differ. This sheet does not justify replacing the clear-preserving benchmark.',
    'FFHQ JPEG: texture and complexion changes persist across diverse lighting. Clear glasses and hair are still visible; broad structure is not evidence of unchanged appearance.',
    'JPEG to light: small degradations need less alteration. New50 changes brightness and facial colour; light-degraded examples retain good structure but show no universal target-faithful improvement.',
    'FFHQ light: restored details are clear, but both adaptations smooth skin and alter shading compared with clean targets. New50 often stays close to AdaIN50 rather than distinctly improving structure.',
    'FFHQ light: expressions and glasses remain legible; tone and fine texture shift. The normally lit examples expose unnecessary alteration despite pleasing output quality.',
    'FFHQ light: new50 produces smooth faces and clear broad features, but target colour and texture are not preserved exactly. Some lip and cheek rendering changes are visible.',
    'FFHQ light: background and skin tone changes accompany denoising. New50 often resembles historical AdaIN; improvement on severe crops cannot establish benefit on these mild cases.',
    'Light to lowlight: new50 makes several dark faces easier to read than original0, reducing brittle reconstruction texture. Shading, hair and accessory details still differ from clean targets.',
    'FFHQ lowlight: useful brightening and smoother structure are visible in many rows. Some dark appearance and fine facial detail remain uncertain; comparison with aligned targets is necessary.',
    'FFHQ lowlight: adapted outputs look cleaner than original0, with visible eyes and mouths. New50 retains residual errors in expression, skin texture and illumination; darkest rows remain difficult.',
    'FFHQ lowlight: broad appearance improves over many original reconstructions, but smooth texture and tone mismatch remain. New50 is close to AdaIN50, with variable target agreement.',
    'Lowlight to QMUL native: paired lowlight rows show useful but incomplete restoration. Native glare and block-like crops yield misplaced, absent or invented facial features; request clearer input rather than treating texture as recovered structure.',
    'QMUL native: severe blocks, side views, occlusion and glare remain unusable in multiple rows. Brightening does not solve tiny eyes, misplaced mouths or fabricated glasses. No clean reference exists.',
    'QMUL to older paired: the native block-like rows remain severely malformed despite fine texture. Older paired rows include added glasses and changed smile structure; new50 does not solve insufficient visible information.',
    'Older paired comparison: very severe compound inputs still create blank or malformed facial regions. Easier lowlight cases improve, but clear and motion cases show changed appearance. Dataset labels do not establish ethnicity.',
    'Older paired comparison: mild/clear outputs keep much of the visible face but change tone; severe blur or compound cases can alter hair, expression and hand/face boundaries. New50 closely resembles AdaIN50.',
    'Older paired comparison: compound and lowlight outputs reduce brittle texture yet still differ from targets in expression and features. Motion examples are clearer but can lose texture and alter details; no universal structural win.'
]


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    pixel_path = REVIEW / 'pixels_v1/audit.json'
    audit = json.loads(pixel_path.read_text(encoding='utf-8'))
    imported = json.loads((REVIEW / 'return_import_v1.json').read_text(encoding='utf-8'))
    sheets = audit['visual_sheets']
    assert audit['complete'] and len(sheets) == len(NOTES) == 34
    cases = []
    records = []
    for number, (sheet, note) in enumerate(zip(sheets, NOTES), 1):
        assert sheet['path'] == f'visual_sheets/sheet_{number:02}.png'
        path = REVIEW / 'pixels_v1' / sheet['path']
        assert path.stat().st_size == sheet['bytes'] and sha(path) == sheet['sha256']
        cases.extend(sheet['case_ids'])
        records.append({**sheet, 'viewed_by_root': True,
                        'inspection_resolution': 'original; exact256 saved cells',
                        'observation': note})
    assert len(cases) == len(set(cases)) == 406 and audit['exact256_sheet_cells'] == 2030
    result = {
        'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': audit['protocol_sha256'], 'archive_sha256': audit['archive_sha256'],
        'import_receipt_sha256': sha(REVIEW / 'return_import_v1.json'),
        'pixel_audit_sha256': sha(pixel_path), 'reviewer': 'root assistant',
        'personal_visual_review': True, 'sheets_reviewed': records,
        'sheet_count': 34, 'case_rows_reviewed': 406, 'exact256_cells_reviewed': 2030,
        'case_ids': cases, 'native_unpaired_cases': 48, 'paired_cases': 358,
        'columns': ['saved input', 'clean target or native blank', 'original0', 'historical AdaIN50', 'new normal-tone50'],
        'new0_exact_original0_alias_from_pixel_audit': True,
        'conclusion': 'Real trained50 candidate; useful dark-profile changes coexist with clear/normal preservation failures and severe-input structural defects. Reject centered recipe and app promotion.',
        'all_planned_visual_sheets_reviewed': True, 'model_qualified': False,
        'user_acceptance': False, 'independent_final_review_completed': False,
        'final_identities_reviewed': False, 'app_promoted': False,
        'limitations': ['Assistant visual inspection is not human user acceptance or an independent final evaluation.',
                        'Native CCTV has no aligned clean target; visual plausibility does not establish recovered identity.',
                        'No new inference, training, gradient, optimizer or VM calls occurred during this review.'],
        'recorder_source_sha256': sha(__file__)
    }
    assert result['protocol_sha256'] == imported['protocol_sha256']
    assert result['archive_sha256'] == imported['archive_sha256']
    with (REVIEW / 'root_visual_review_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print({'complete': True, 'sheets': 34, 'rows': 406, 'cells': 2030, 'model_qualified': False})


if __name__ == '__main__':
    main()
