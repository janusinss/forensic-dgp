"""Close an explicitly completed visual ledger; never infer model qualification."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest_path = OUT / 'gallery/gallery_manifest.json'
    manifest = read(manifest_path)
    gallery = read(OUT / 'gallery_independent_audit.json')
    audit_path = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_path)
    assert gallery['complete'] and audit['complete']
    assert sha(manifest_path) == gallery['gallery_manifest_sha256']
    assert sha(audit_path) == gallery['source_bound_audit_sha256']
    pages = {page['name']: page for page in manifest['pages']}
    batches = sorted((OUT / 'visual_review_batches').glob('batch*.json'))
    rows = []
    for path in batches:
        batch = read(path)
        assert batch['complete'] and batch['gallery_manifest_sha256'] == sha(manifest_path)
        assert batch['local_model_updates'] == 0 and not batch['model_qualification']
        for row in batch['observations']:
            page = pages[row['page']]
            assert row['page_sha256'] == page['sha256'] == sha(ROOT / page['file'])
            assert row['case_ids'] == page['case_ids'] and row['kind'] == page['kind']
            assert row['explicitly_viewed'] and row['all_visible_features_considered']
            assert row['observation'].strip() and not row['model_qualification']
            assert not row['native_hidden_identity_or_clean_reference_claim']
            rows.append(row)
    assert len(rows) == len(pages) == 64
    assert len({row['page'] for row in rows}) == 64
    assert {row['page'] for row in rows} == set(pages)
    paired = [row for row in rows if row['kind'] == 'paired_TRAIN']
    native = [row for row in rows if row['kind'] == 'unpaired_native_DEV']
    # Classification names are frozen in the source gallery manifest.
    if not native:
        native = [row for row in rows if row['kind'] != 'paired_TRAIN']
    assert len(paired) == 40 and len(native) == 24
    paired_ids = sorted({cid for row in paired for cid in row['case_ids']})
    native_ids = sorted({cid for row in native for cid in row['case_ids']})
    assert len(paired_ids) == 100 and len(native_ids) == 24
    quality = read(OUT / 'quality_summary.json')
    assert len(quality['arms']) == 12 and quality['arms_passing_sampled_requirements'] == 0
    result = dict(
        complete=True, visual_review_complete=True, planned_sheets=64, viewed_sheets=64,
        paired_TRAIN_sheets=40, native_unpaired_DEV_sheets=24,
        paired_TRAIN_case_ids=paired_ids, native_unpaired_DEV_case_ids=native_ids,
        exact_saved_PNG_cells=gallery['exact_cells_verified'],
        gallery_manifest_sha256=sha(manifest_path), independent_gallery_audit_sha256=sha(OUT/'gallery_independent_audit.json'),
        returned_audit_sha256=sha(audit_path),
        batch_bindings={str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in batches},
        observations=rows,
        reviewer_method='primary reviewer explicitly inspected every planned saved-PNG sheet through view_image',
        independent_pixel_integrity_is_separate_from_human_visual_judgment=True,
        findings={
            'deep3': 'Little visible change or useful whole-face clarity beyond retained DGP; severe profiles remain soft.',
            'decoder15_low_rates': 'Little added useful detail; unchanged scientific preservation gates still fail.',
            'decoder15_highest_rate': 'Cool blue-purple or gray-green appearance, lifted or washed-out skin and backgrounds; degraded eyes, noses and mouths remain soft.',
            'clear_glasses_and_visible_hair': 'Generally remain recognizable; this does not override altered skin color or scientific preservation failures.',
            'insufficient_information': 'Do not treat brightness or smooth facial estimates as recovered detail; retain frozen input-only review and clearer-crop requests.',
        },
        arms_passing_unchanged_scientific_gates=0, arms_selected_for_promotion=0,
        paired_role='historically exposed photographic TRAIN with synthetic degradation; not held-out validation',
        native_role='24 ChokePoint C1 development crops; unpaired, without aligned clean reference',
        final_identity_pixels_reviewed=0, native_reference_PSNR_SSIM_claim=False,
        ethnicity_or_Zamboanga_performance_claim=False, hidden_identity_recovery_claim=False,
        full_goal_independent_final_review_complete=False,
        current_checkpoint_app_selection_changed=False, local_model_updates=0,
        no_failed_arm_resume=True, longer_training_recipe_qualified=False,
        earlier_planning_receipts_with_visual_pending_are_superseded_only_for_this_gallery=True,
        checker_sha256=sha(Path(__file__)), model_qualification=False, goal_complete=False,
    )
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, viewed_sheets=64, exact_saved_PNG_cells=result['exact_saved_PNG_cells'],
               qualifying_arms=0, local_model_updates=0, goal_complete=False))


if __name__ == '__main__':
    main()
