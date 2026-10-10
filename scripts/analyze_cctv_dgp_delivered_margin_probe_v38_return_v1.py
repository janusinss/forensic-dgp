"""Recount audited finite trials and show every unchanged-size TRAIN image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return'
OUT = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1'
V36 = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    audit_path = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json'
    audit = read(audit_path)
    external = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_audit_run_v1/external_receipt.json')
    assert audit['complete'] and audit['finite_probe_complete'] and external['complete']
    assert audit['CPU_replay']['outputs'] == 100 and audit['CPU_replay']['all_states_restored']
    assert audit['local_gradient_calls'] == audit['local_optimizer_updates'] == 0
    p = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == audit['protocol_sha256']
    imported = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return_import.json')
    assert imported['members'] == audit['members_verified'] == 2135
    for name, digest in imported['files_sha256'].items():
        assert sha(RETURN / name) == digest, name
    metric_path = ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py'
    spec = importlib.util.spec_from_file_location('V38_audited_delivered_metrics', metric_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    prior = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json')
    prior_p = read(V36 / 'protocol.json')
    assert p['cohorts'] == prior_p['cohorts']
    constraint_labels = read(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/analysis.json')['constraint_labels']
    assert not OUT.exists()
    OUT.mkdir()
    (OUT / 'pages').mkdir()
    bindings = {q.relative_to(ROOT).as_posix(): sha(q) for q in [audit_path, metric_path,
        ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json',
        ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/analysis.json']}
    rows, pages, parity = [], [], []
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else font
    columns = [('Input', None), ('Paired target', None), ('Original DGP', 'before')]
    columns += [('PNG margin scale ' + str(v['scale']), v['name']) for v in p['variants']]
    assert len(columns) == 7

    def pixels(path):
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB')).copy()

    def group_rows(group, key):
        return [r for r in group if key in {'all', 'clear' if r['profile'] == 'clear' else 'degraded',
            r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'),
            r['source'] + '/' + r['profile']}]

    for cohort in p['cohorts']:
        label = cohort['name']
        base_folder = RETURN / f'outputs/state0_{label}'
        before = read(base_folder / 'before/receipt.json')
        for case in cohort['cases']:
            cid = case['id']
            old_png, old_raw = V36 / f'outputs/state0_{label}/before/{cid}.png', V36 / f'outputs/state0_{label}/before/{cid}.npy'
            fresh_png, fresh_raw = base_folder / f'before/{cid}.png', base_folder / f'before/{cid}.npy'
            raw_difference = float(np.abs(np.load(old_raw, allow_pickle=False) - np.load(fresh_raw, allow_pickle=False)).max())
            png_difference = int(np.abs(pixels(old_png).astype(int) - pixels(fresh_png).astype(int)).max())
            assert raw_difference <= p['CPU_raw_absolute_tolerance'] and png_difference <= p['CPU_PNG_byte_tolerance']
            parity.append({'cohort': label, 'id': cid, 'raw_maximum_error': raw_difference,
                'PNG_maximum_byte_error': png_difference,
                'raw_file_exact': sha(old_raw) == sha(fresh_raw), 'PNG_file_exact': sha(old_png) == sha(fresh_png)})
            for q in [old_png, old_raw]: bindings[q.relative_to(ROOT).as_posix()] = sha(q)
        for variant in p['variants']:
            folder = base_folder / variant['name']
            current, comparison = read(folder / 'receipt.json'), read(folder / 'comparison.json')
            decision = metrics.compare_groups(before['groups'], current['groups'])
            assert decision == comparison['preservation_against_original']
            raw_failures, linear_checks = [], []
            for index, constraint in enumerate(constraint_labels[:102]):
                if constraint['cohort'] != label or constraint['metric'] == 'one_minus_raw_SSIM': continue
                key = 'raw_MSE' if constraint['metric'] == 'raw_MSE' else 'raw_ArcFace'
                old_value = float(np.mean([r['metrics'][key] for r in group_rows(before['rows'], constraint['group'])]))
                new_value = float(np.mean([r['metrics'][key] for r in group_rows(current['rows'], constraint['group'])]))
                change = new_value - old_value
                bad = change > 1e-12 if key == 'raw_MSE' else change < -1e-6
                finite_loss_change = change if key == 'raw_MSE' else -change
                predicted = comparison['all210_linear_function_changes'][index]
                linear_checks.append({**constraint, 'finite_loss_change': finite_loss_change,
                    'linear_loss_change': predicted, 'finite_minus_linear': finite_loss_change - predicted})
                if bad: raw_failures.append({'group': constraint['group'], 'metric': key,
                                            'baseline': old_value, 'candidate': new_value})
            changes = []
            for case, old, new in zip(cohort['cases'], before['rows'], current['rows'], strict=True):
                cid = case['id']
                difference = pixels(folder / (cid + '.png')).astype(np.int16) - pixels(base_folder / 'before' / (cid + '.png')).astype(np.int16)
                changes.append({'id': cid, 'source': case['source'], 'profile': case['profile'],
                    'changed_RGB_bytes': int(np.count_nonzero(difference)),
                    'maximum_PNG_byte_change': int(np.abs(difference).max()),
                    'raw_metric_changes': {key: new['metrics'][key] - old['metrics'][key] for key in ['raw_MSE', 'raw_SSIM', 'raw_ArcFace']}})
            earlier = next(r for r in prior['variants'] if r['cohort'] == label and r['scale'] == variant['scale'])
            gain = 100 * comparison['incremental_degraded_PNG_structure_gain']
            rows.append({'cohort': label, 'variant': variant['name'], 'scale': variant['scale'],
                'degraded_PNG_structure_gain_percent': gain,
                'V36_same_scale_structure_gain_percent': earlier['degraded_PNG_structure_gain_percent'],
                'change_from_V36_structure_gain_percentage_points': gain - earlier['degraded_PNG_structure_gain_percent'],
                'V36_same_scale_PNG_failure_count': len(earlier['preservation_against_original']['failures']),
                'preservation_against_original': decision, 'case_changes': changes,
                'all_case_maximum_PNG_byte_change': max(r['maximum_PNG_byte_change'] for r in changes),
                'finite_raw_component_change': comparison['finite_raw_component_change'],
                'all210_linear_function_changes': comparison['all210_linear_function_changes'],
                'descriptive_raw_MSE_ArcFace_failures': raw_failures,
                'first_order_to_finite_raw_checks': linear_checks,
                'subset_preservation_source_brightness_pass': decision['all17_preservation_groups_pass']
                    and decision['brightness_gate_pass'] and all(v >= 0 for v in decision['source_structure_gains'].values()),
                'training_capacity_pass': False})
        for begin in range(0, len(cohort['cases']), 5):
            subset = cohort['cases'][begin:begin + 5]
            assert len(subset) == 5 and len({c['reference_id'] for c in subset}) == 1
            canvas = Image.new('RGB', (1792, 1480), '#f5f5f5')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f'V38 | {label} TRAIN | {subset[0]["source"]} | source is not ethnicity', fill='black', font=font)
            for column, (title, _) in enumerate(columns): draw.text((column * 256 + 5, 29), title, fill='black', font=font)
            cells = []
            for row, case in enumerate(subset):
                y = 57 + row * 284
                draw.text((8, y), case['id'] + ' | ' + case['profile'], fill='black', font=small)
                for column, (_, route) in enumerate(columns):
                    source = MIXED / case['input' if column == 0 else 'target'] if route is None else base_folder / route / (case['id'] + '.png')
                    image = pixels(source)
                    assert image.shape == (256, 256, 3) and image.dtype == np.uint8
                    xy = [column * 256, y + 22]
                    canvas.paste(Image.fromarray(image), tuple(xy))
                    name = source.relative_to(ROOT).as_posix()
                    bindings[name] = sha(source)
                    cells.append({'id': case['id'], 'column': column, 'xy': xy, 'path': name, 'sha256': bindings[name]})
            destination = OUT / f'pages/{label}_{begin // 5 + 1:02d}.png'
            canvas.save(destination)
            pages.append({'path': destination.relative_to(ROOT).as_posix(), 'sha256': sha(destination),
                'case_ids': [c['id'] for c in subset], 'cells': cells})
    assert len(rows) == 8 and len(pages) == 20 and sum(len(v['cells']) for v in pages) == 700
    eligible = [v['name'] for v in p['variants'] if all(r['subset_preservation_source_brightness_pass'] for r in rows if r['variant'] == v['name'])]
    result = {'complete': True, 'protocol_sha256': audit['protocol_sha256'], 'variants': rows,
        'jointly_eligible_subset_variants': eligible, 'V36_original_output_parity': parity,
        'all100_V36_original_PNG_and_raw_files_exact': all(r['raw_file_exact'] and r['PNG_file_exact'] for r in parity),
        'visual_pages': pages, 'visual_selection': 'All cases in frozen order: both TRAIN cohorts, all references and profiles',
        'visual_cases': 100, 'model_output_cells': 500, 'input_target_output_cells': 700,
        'all500_metrics_audited': True, 'actual_visual_review_pending': True,
        'raw_SSIM_note': 'Saved raw SSIM uses skimage; it is not the differentiable VM guard definition',
        'paired_photographic_TRAIN_only': True, 'unexposed_not_DEV_or_final': True,
        'native_or_reserved_used': False, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
        'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
        'bindings_sha256': bindings, 'script_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started}
    assert result['seconds'] <= 300
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'comparisons': 8, 'visual_pages': 20,
        'jointly_eligible_subset_variants': eligible, 'V36_baselines_exact': result['all100_V36_original_PNG_and_raw_files_exact']}))


if __name__ == '__main__':
    main()
