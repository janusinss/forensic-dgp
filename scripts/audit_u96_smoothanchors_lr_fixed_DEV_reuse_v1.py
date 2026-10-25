"""Saved-data-only audit of the unchanged fixed 68-case development cohort.

No network, model, Torch, gradients, optimizer, VM, browser or FINAL pixel read.
"""
import collections
import hashlib
import io
import json
from pathlib import Path
import stat

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'outputs/codeformer_spatial_dev_review_v1/cohort_manifest_v1.json'
STAGE = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_preparation'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    checks, verified, historical_suffixes = 0, {}, []
    def need(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)
    def filepin(path):
        path = Path(path)
        need(path.resolve().is_relative_to(ROOT), 'Workspace file only')
        info = path.lstat()
        need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024, 'Regular evidence file')
        data = path.read_bytes()
        row = {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': digest(data)}
        verified[row['path']] = row
        return row
    def bound(row):
        actual = filepin(ROOT / row['path'])
        need(all(actual[k] == row[k] for k in ('path', 'bytes', 'sha256')), 'Exact pinned file ' + row['path'])
        return ROOT / row['path']
    def pixels(row, mode='RGB'):
        path = bound(row)
        with Image.open(path) as image:
            return np.asarray(image.convert(mode)).copy()
    m = load(COHORT)
    cohort_pin = filepin(COHORT)
    need(cohort_pin['sha256'] == 'e42afd973305e42422bcb273dc419ab608bbc1437ab332e62d6eb006e0fe9307' and len(m['cases']) == 68, 'Unchanged original fixed68 cohort')
    inventory = load(bound(m['inventory_evidence']))
    need(len(inventory['cases']) == len(m['cases']) and all(
        {k: value for k, value in current.items() if k not in ('upload_source', 'frozen_evaluation_row')} == old
        for old, current in zip(inventory['cases'], m['cases'])) and
        inventory['native_input_only_controls'] == m['native_input_only_controls'] and
        inventory['filter_insufficient_controls'] == m['filter_insufficient_controls'],
        'Original inventory/case/control joins; only two explicit manifest binding enrichments')
    for old in m['source_documents']:
        current = filepin(ROOT / old['path'])
        if current['bytes'] == old['bytes'] and current['sha256'] == old['sha256']:
            continue
        need(old['path'] in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md') and current['bytes'] > old['bytes'], 'Only prepend-evolving instruction documents')
        body = (ROOT / old['path']).read_bytes()
        need(digest(body[-old['bytes']:]) == old['sha256'], 'Historical instruction exact suffix ' + old['path'])
        historical_suffixes.append({'path': old['path'], 'historical': {k: old[k] for k in ('bytes', 'sha256')}, 'current': current, 'historical_suffix_offset': len(body) - old['bytes'], 'exact_suffix': True})
    fbase = ROOT / 'outputs/codeformer_filter_pairs_v1_data'
    f = load(fbase / 'manifest.json')
    root_review = load(ROOT / 'outputs/codeformer_filter_pairs_v1_preparation/root_input_only_review_v1.json')
    frows = {r['id']: r for r in f['cases']}
    freview = {r['reference_id'] + '__' + r['profile']: r for r in root_review['cases']}
    v2refs = load(ROOT / 'outputs/cctv_codeformer_adaptation_v2_vm/references.json')['references']
    v2by = {r['id']: r for r in v2refs}
    train_ids = {r['id'] for r in v2refs if r['historical_role'] == 'train'}
    train_hashes = {r['source_sha256'] for r in v2refs if r['historical_role'] == 'train'}
    need(len(train_ids) == 391, 'Retained original391TRAIN role census')
    qreview = {r['id']: r for r in load(ROOT / 'outputs/cctv_native_development_v2/input_review.json')['rows']}
    creview_doc = load(ROOT / 'outputs/cctv_chokepoint_native_development_v1/input_review.json')
    creview = {r['id']: r for r in creview_doc['rows']}
    p = load(STAGE / 'protocol.json')
    lr_data = load(bound({'path': (STAGE / p['data']['provenance']['path']).relative_to(ROOT).as_posix(), 'bytes': p['data']['provenance']['bytes'], 'sha256': p['data']['provenance']['sha256']}))
    lr_sources = lr_data['sources']
    need(len(lr_sources) == 24 and len({r['reference_id'] for r in lr_sources}) == 24, 'Exactly16optimizer8monitor TRAIN photos')
    lr_ids = {r['reference_id'] for r in lr_sources}
    lr_source_hashes = {r['source']['sha256'] for r in lr_sources}
    lr_source_pixels = {r['source_RGB8_sha256'] for r in lr_sources}
    lr_photo_urls = {r['attribution']['photo_url'] for r in lr_sources}
    lr_input_hashes = {r['input']['sha256'] for r in p['cases']}
    lr_target_hashes = {r['target']['sha256'] for r in p['cases']}
    lr_input_pixels = {r['input_RGB8_sha256'] for r in p['cases']}
    lr_target_pixels = {r['target_RGB8_sha256'] for r in p['cases']}
    source1024, term_pins, case_audits = {}, {}, []

    def prepare(rgb):
        h, w = rgb.shape[:2]
        side = max(h, w); top, left = (side - h) // 2, (side - w) // 2
        canvas = Image.new('RGB', (side, side), (128, 128, 128)); canvas.paste(Image.fromarray(rgb), (left, top))
        support = Image.new('L', (side, side), 0); support.paste(255, (left, top, left + w, top + h))
        prepared = np.asarray(canvas.resize((256, 256), Image.Resampling.BILINEAR)).copy()
        observed = np.asarray(support.resize((256, 256), Image.Resampling.NEAREST)).copy() > 0
        yy, xx = np.nonzero(observed)
        geometry = {'native_size': [w, h], 'native_center_offset': [left, top], 'observed_bounds_256': [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)], 'output_size': [256, 256], 'policy': 'dgp256-observed-canvas-v1', 'input_rgb_sha256': digest(rgb.tobytes()), 'canvas_rgb_sha256': digest(prepared.tobytes()), 'observed_sha256': digest(observed.tobytes()), 'removal_mask_sha256': digest(np.zeros((256, 256), dtype=bool).tobytes())}
        return prepared, observed, geometry

    def corruption(base, profile):
        op, params = profile['operator'], profile['parameters']
        rounding = lambda value: np.floor(np.clip(value, 0, 255) + .5).astype(np.uint8)
        def mosaic(value, cell):
            h, w = value.shape[:2]
            means = value.astype(np.float64).reshape(h // cell, cell, w // cell, cell, 3).mean(axis=(1, 3))
            return rounding(means).repeat(cell, axis=0).repeat(cell, axis=1)
        if op == 'identity':
            return base.copy()
        if op in ('gaussian', 'local_gaussian'):
            result = rounding(gaussian_filter(base.astype(np.float64), sigma=(params['sigma_y'], params['sigma_x'], 0), mode='reflect', truncate=3.0))
        elif op == 'repeated_rows':
            rows = np.arange(256) // params['block_rows'] * params['block_rows']
            weight = params['repeat_weight']
            result = rounding(base.astype(np.float64) * (1 - weight) + base[rows].astype(np.float64) * weight)
        elif op == 'mosaic':
            result = mosaic(base, params['cell'])
        elif op == 'local_mosaic':
            result = base.copy(); result[64:192, 64:192] = mosaic(base[64:192, 64:192], params['cell'])
        else:
            raise AssertionError('Unrecognized frozen operator')
        if op.startswith('local_'):
            mask = np.zeros((256, 256), dtype=bool); mask[64:192, 64:192] = True
            result[~mask] = base[~mask]
        return result

    selected_ids = []
    for row in m['cases']:
        selected_ids.append(row['id'])
        need(row['split'] in ('DEV', 'native_DEV'), 'No optimization/FINAL selected case')
        image, support = pixels(row['input']), pixels(row['support'], 'L') > 0
        need(image.shape == (256, 256, 3) and digest(image.tobytes()) == row['input_pixels']['RGB8_sha256'], 'Exact input RGB256 ' + row['id'])
        need(row['reference_id'] not in lr_ids and row['input']['sha256'] not in lr_input_hashes | lr_target_hashes and digest(image.tobytes()) not in lr_input_pixels | lr_target_pixels, 'No exact selected input TRAIN overlap ' + row['id'])
        raw_upload = pixels(row['upload_source'])
        rederived, observed, geometry = prepare(raw_upload)
        need(np.array_equal(image, rederived) and np.array_equal(support, observed) and geometry == row['geometry'], 'Exact original-upload geometry/support ' + row['id'])
        need(row['upload_source']['sha256'] not in lr_source_hashes | lr_input_hashes | lr_target_hashes, 'No exact upload TRAIN overlap')
        if row['cohort'] == 'filter_DEV':
            need(row['frozen_evaluation_row'] == frows[row['id']] and row['input_review'] == freview[row['id']] and row['input_review']['decision'] in ('readable', 'borderline'), 'Exact filter membership/readability labels')
            need(v2by[row['reference_id']]['historical_role'] == 'validation' and row['reference_id'] not in train_ids and row['source1024']['sha256'] not in train_hashes | lr_source_hashes, 'Original validation role/source exact file separation')
            pair = load(bound(row['pair_record']))
            source = pixels(row['source1024'])
            need(source.shape == (1024, 1024, 3) and digest(source.tobytes()) not in lr_source_pixels, 'DEV original1024 pixels separated')
            need(row['terms'] == pair['attribution'] == v2by[row['reference_id']]['attribution'] and row['terms']['photo_url'] not in lr_photo_urls, 'Unchanged original photo attribution and no LR-photo URL overlap')
            target = pixels(row['target'])
            clean = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.LANCZOS))
            base = np.asarray(Image.fromarray(source).resize((256, 256), Image.Resampling.BILINEAR))
            need(np.array_equal(clean, target) and np.array_equal(corruption(base, pair['profile']), image), 'Independent source-to-target/input corruption exact')
            need(row['target']['sha256'] not in lr_target_hashes | lr_input_hashes and digest(target.tobytes()) not in lr_target_pixels | lr_input_pixels, 'No exact clean target TRAIN overlap')
            need(row['appearance_flag'] == (row['reference_id'] == 'va_ffhq_08025'), 'Retain all nine08025 appearance flags')
            source1024[row['reference_id']] = {'source': row['source1024'], 'target': row['target'], 'attribution': row['terms'], 'original_role': v2by[row['reference_id']]['historical_role'], 'source_RGB8_sha256': digest(source.tobytes())}
        else:
            need(row['target'] is None and row['evidence'] == 'native_unpaired', 'Native evidence remains unpaired')
            review = qreview[row['id']] if row['source'] == 'QMUL-SurvFace' else creview[row['id']]
            need(row['input_review'] == review and row['frozen_evaluation_row']['input_review'] == review, 'Native original input-only labels exact')
            need(row['frozen_evaluation_row']['prepared_sha256'] == row['input']['sha256'] and row['frozen_evaluation_row']['observed_sha256'] == row['support']['sha256'], 'Original native evaluation membership')
            for name in row['terms_records']:
                term_pins[name] = filepin(ROOT / name)
        case_audits.append({'id': row['id'], 'cohort': row['cohort'], 'split': row['split'], 'input_sha256': row['input']['sha256'], 'upload_source_sha256': row['upload_source']['sha256'], 'target_sha256': row['target']['sha256'] if row['target'] else None, 'support_sha256': row['support']['sha256'], 'original_input_review_unchanged': True, 'source_geometry_and_pixels_exact': True, 'optimization_permitted': False})
    need(len(set(selected_ids)) == 68 and collections.Counter(r['cohort'] for r in m['cases']) == {'filter_DEV': 36, 'native_QMUL-SurvFace': 8, 'native_ChokePoint': 24}, 'Frozen case order/counts')
    controls = []
    for control in m['native_input_only_controls'] + m['filter_insufficient_controls']:
        image = pixels(control['input'])
        need(image.shape == (256, 256, 3) and control['neural_calls_allowed'] == 0 and control['id'] not in selected_ids, 'Exact zero-call control')
        if 'source' in control:
            need(control['decision'] == qreview[control['id']], 'Native16 excluded input-only labels unchanged')
        else:
            need(control['decision'] == 'insufficient' and freview[control['id']]['decision'] == 'insufficient', 'Filter24 insufficient labels unchanged')
        controls.append(control)
    need(len(m['native_input_only_controls']) == 16 and len(m['filter_insufficient_controls']) == 24, 'All40zero-call controls retained')
    need(collections.Counter(r['input_review']['decision'] for r in m['cases'] if r['cohort'] == 'filter_DEV') == {'readable': 12, 'borderline': 24}, 'Original readable12/borderline24 unchanged')
    need(not set(lr_ids) & set(source1024), 'No LR optimization/monitor reference-ID overlap with pairedDEV4')
    for key in ('r2', 'u96'):
        bound(dict(p['model'][key], path=(STAGE / p['model'][key]['path']).relative_to(ROOT).as_posix()))
    old_exports = load(ROOT / 'outputs/codeformer_spatial_continuation_v1_models/model_exports_v1.json')
    old_u96 = old_exports['arms']['spatial_u']['export']
    bound(old_u96)
    need(old_u96['sha256'] == p['model']['u96']['sha256'], 'Historical U96 adapter exact incumbent used in LR')
    peer = filepin(ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_peer/independent_return_review_v1.json')
    pixel = filepin(ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/saved_pixel_audit_v1.json')
    state = filepin(ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_state_audit/saved_state_audit_v1.json')
    review = {'format': 'conditional-fixed68-DEV-reuse-for-U96-versus-fresh-smoothanchors-lr-u64-v1', 'complete': True,
        'independent_data_reuse_audit_pass': True, 'data_checks': checks, 'source': filepin(Path(__file__).resolve()),
        'original_cohort_manifest': cohort_pin, 'historical_evidence_documents_preserved_exact_or_suffix': historical_suffixes,
        'case_order': selected_ids, 'cases': case_audits, 'cohort_counts': {'paired_filter_DEV': 36, 'native_QMUL-SurvFace': 8, 'native_ChokePoint': 24},
        'filter_input_only_decisions': {'readable': 12, 'borderline': 24}, 'appearance_flagged_08025_cases': 9,
        'zero_call_controls': controls, 'zero_call_control_counts': {'native': 16, 'filter_insufficient': 24},
        'conditional_local_inference_permitted': True, 'maximum_new_local_neural_forwards': 136,
        'models': ['incumbent_complete_U96', 'same_complete_U96_plus_trained_LR_u64_postfusion_heads'],
        'incumbent_R2': p['model']['r2'], 'incumbent_U96': old_u96,
        'candidate_source_archive_sha256': '7e5208d2ff69713c834c21fb466020ae54d7ee68b84795e2c044ffc11847041e',
        'candidate_source_checkpoint_member': 'outputs/u96_postfusion_smoothanchors_lr_v1/checkpoints/u0064/checkpoint.json',
        'prior_TRAIN_screen_reviews': {'metadata': peer, 'pixels': pixel, 'state': state},
        'required_before_inference': ['Root personally completes all30TRAIN visual sheets and records that review.', 'Independent frozen-inference adapter source and exact36head export/state bindings pass.', 'Frozen comparison plan preserves all68case IDs/order, same observed canvas, support and zero-call40controls.'],
        'paired_DEV_sources': source1024, 'native_terms_records': list(term_pins.values()),
        'LR_TRAIN_source_reference_order': [r['reference_id'] for r in lr_sources],
        'overlap': {'paired_DEV_reference_ID_vs_LR24': [], 'paired_DEV1024_file_SHA_vs_LR24': [], 'paired_DEV1024_RGB8_SHA_vs_LR24': [], 'paired_DEV_Flickr_photo_URL_vs_LR24': [], 'all68input_and_upload_file_SHA_vs_LR_inputs_targets_sources': [], 'all68input_RGB8_SHA_vs_LR_inputs_targets': [], 'paired36target_file_and_RGB8_SHA_vs_LR_inputs_targets': [], 'paired_DEV4_vs_original391TRAIN_exact_reference_and_source_SHA': [], 'subject_identity_independence': 'Unknown beyond exact reference/file/pixel/photo URL checks; no person-disjoint claim.', 'pretraining_overlap': 'Possible/unknown for FFHQ and CodeFormer; no unseen identity guarantee.'},
        'terms_scope': 'Cached original public research terms and photo attribution are retained. ChokePoint derivatives are noncommercial research with original license/notice. FFHQ DEV photos retain two CC BY, one CC BY-NC and one Public Domain Mark attribution. No new licensing conclusion or redistribution permission is inferred.',
        'qualification_limits': ['Repeated adaptive development exposure; this fixed68subset is not independent FINAL evaluation.', 'The8QMULcoarse and24borderlinefilter cases remain diagnostic challenges, not newly admitted usable user inputs.', 'Native32have no aligned clean target; no native PSNR/SSIM or recovered-identity accuracy.', 'Only local frozen inference is conditionally admitted; no data optimization, new training, automatic model promotion or app-default change.', 'Full406/478development, separate reservedFINAL, user acceptance and app qualification remain outstanding.'],
        'verified_files': list(verified.values()), 'optimization_permitted': False, 'new_training_authorized': False,
        'source_selection_or_input_review_changed': False, 'input_based_quality_threshold_recalibrated': False,
        'native_data_used_for_optimization': False, 'FINAL_identity_or_pixels_inspected': False,
        'Torch_model_neural_gradient_optimizer_training_VM_HTTP_browser_calls': 0, 'model_qualified': False}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'conditional_fixed_DEV_reuse_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(review, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'data_checks': checks, 'verified_files': len(verified), 'receipt': filepin(target)}))


if __name__ == '__main__':
    main()
