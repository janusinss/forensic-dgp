"""Summarize audited finite trials and render every TRAIN case without image processing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return'
OUT = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1'
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
    audit_path = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2.json'
    audit = read(audit_path)
    external = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_audit_run_r2/external_receipt.json')
    assert audit['complete'] and audit['finite_probe_complete'] and external['complete']
    assert audit['CPU_replay']['outputs'] == 100 and audit['CPU_replay']['all_states_restored']
    assert audit['local_gradient_calls'] == audit['local_optimizer_updates'] == 0
    p = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == audit['protocol_sha256']
    imported = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return_import.json')
    assert imported['members'] == audit['members_verified'] == 2135
    for name, digest in imported['files_sha256'].items():
        assert sha(RETURN / name) == digest, name
    path = ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py'
    spec = importlib.util.spec_from_file_location('V35_audited_PNG_group_metrics', path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    assert not OUT.exists()
    OUT.mkdir()
    (OUT / 'pages').mkdir()
    bindings = {audit_path.relative_to(ROOT).as_posix(): sha(audit_path)}
    rows, pages = [], []
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 15) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else font
    columns = [('Input', None), ('Paired target', None), ('Original DGP', 'before'),
               ('Guarded scale 1', 'joint_1'), ('Scale 1/2', 'joint_half'),
               ('Scale 1/4', 'joint_quarter'), ('Scale 1/8', 'joint_eighth')]
    constraint_labels = read(ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1/analysis.json')['constraint_labels']

    def subset(rows, key):
        return [r for r in rows if key in {'all', 'clear' if r['profile'] == 'clear' else 'degraded',
            r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'),
            r['source'] + '/' + r['profile']}]
    for cohort in p['cohorts']:
        label = cohort['name']
        base_folder = RETURN / f'outputs/state0_{label}'
        before = read(base_folder / 'before/receipt.json')
        for variant in p['variants']:
            folder = base_folder / variant['name']
            current, comparison = read(folder / 'receipt.json'), read(folder / 'comparison.json')
            decision = metrics.compare_groups(before['groups'], current['groups'])
            assert decision == comparison['preservation_against_original']
            raw_failures, linear_checks = [], []
            for index, constraint in enumerate(constraint_labels[:102]):
                if constraint['cohort'] != label or constraint['metric'] == 'one_minus_raw_SSIM':
                    continue
                key = 'raw_MSE' if constraint['metric'] == 'raw_MSE' else 'raw_ArcFace'
                old_value = float(np.mean([r['metrics'][key] for r in subset(before['rows'], constraint['group'])]))
                new_value = float(np.mean([r['metrics'][key] for r in subset(current['rows'], constraint['group'])]))
                change = new_value - old_value
                bad = change > 1e-12 if key == 'raw_MSE' else change < -1e-6
                finite_loss_change = change if key == 'raw_MSE' else -change
                predicted = comparison['all108_linear_function_changes'][index]
                linear_checks.append({**constraint, 'finite_loss_change': finite_loss_change,
                    'linear_loss_change': predicted, 'finite_minus_linear': finite_loss_change - predicted})
                if bad:
                    raw_failures.append({'group': constraint['group'], 'metric': key,
                                         'baseline': old_value, 'candidate': new_value})
            changes = []
            for case, old, new in zip(cohort['cases'], before['rows'], current['rows'], strict=True):
                cid = case['id']
                with Image.open(base_folder / 'before' / (cid + '.png')) as image:
                    old_png = np.asarray(image.convert('RGB')).copy()
                with Image.open(folder / (cid + '.png')) as image:
                    new_png = np.asarray(image.convert('RGB')).copy()
                difference = new_png.astype(np.int16) - old_png.astype(np.int16)
                changes.append({'id': cid, 'source': case['source'], 'profile': case['profile'],
                    'changed_RGB_bytes': int(np.count_nonzero(difference)),
                    'maximum_PNG_byte_change': int(np.abs(difference).max()),
                    'raw_metric_changes': {key: new['metrics'][key] - old['metrics'][key]
                                           for key in ['raw_MSE', 'raw_SSIM', 'raw_ArcFace']}})
            rows.append({'cohort': label, 'variant': variant['name'], 'scale': variant['scale'],
                'degraded_PNG_structure_gain_percent': 100 * comparison['incremental_degraded_PNG_structure_gain'],
                'preservation_against_original': decision, 'case_changes': changes,
                'all_case_maximum_PNG_byte_change': max(row['maximum_PNG_byte_change'] for row in changes),
                'finite_raw_component_change': comparison['finite_raw_component_change'],
                'all108_linear_function_changes': comparison['all108_linear_function_changes'],
                'descriptive_raw_MSE_ArcFace_failures': raw_failures,
                'first_order_to_finite_raw_checks': linear_checks,
                'subset_preservation_source_brightness_pass': decision['all17_preservation_groups_pass']
                    and decision['brightness_gate_pass'] and all(v >= 0 for v in decision['source_structure_gains'].values()),
                'training_capacity_pass': False})
        # Every case, in frozen protocol order, regardless of metrics or appearance.
        for begin in range(0, len(cohort['cases']), 5):
            subset = cohort['cases'][begin:begin + 5]
            assert len(subset) == 5 and len({c['reference_id'] for c in subset}) == 1
            canvas = Image.new('RGB', (1792, 1480), '#f5f5f5')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f'V35 R1 | {label} TRAIN | {subset[0]["source"]} | source is not ethnicity', fill='black', font=font)
            for column, (title, _) in enumerate(columns):
                draw.text((column * 256 + 5, 29), title, fill='black', font=font)
            cells = []
            for row, case in enumerate(subset):
                y = 57 + row * 284
                draw.text((8, y), case['id'] + ' | ' + case['profile'], fill='black', font=small)
                for column, (_, route) in enumerate(columns):
                    source = MIXED / case['input' if column == 0 else 'target'] if route is None else base_folder / route / (case['id'] + '.png')
                    with Image.open(source) as image:
                        pixels = np.asarray(image.convert('RGB')).copy()
                    assert pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8
                    xy = [column * 256, y + 22]
                    canvas.paste(Image.fromarray(pixels), tuple(xy))
                    name = source.relative_to(ROOT).as_posix()
                    bindings[name] = sha(source)
                    cells.append({'id': case['id'], 'column': column, 'xy': xy, 'path': name, 'sha256': bindings[name]})
            destination = OUT / f'pages/{label}_{begin // 5 + 1:02d}.png'
            canvas.save(destination)
            pages.append({'path': destination.relative_to(ROOT).as_posix(), 'sha256': sha(destination),
                'case_ids': [c['id'] for c in subset], 'cells': cells})
    assert len(rows) == 8 and len(pages) == 20 and sum(len(v['cells']) for v in pages) == 700
    jointly_eligible = [variant['name'] for variant in p['variants'] if all(
        row['subset_preservation_source_brightness_pass'] for row in rows if row['variant'] == variant['name'])]
    result = {'complete': True, 'protocol_sha256': audit['protocol_sha256'], 'variants': rows,
        'jointly_eligible_subset_variants': jointly_eligible, 'visual_pages': pages,
        'visual_selection': 'Every case in frozen protocol order: both cohorts, all references and profiles',
        'visual_cases': 100, 'model_output_cells': 500, 'input_target_output_cells': 700,
        'all500_metrics_audited': True, 'actual_visual_review_pending': True,
        'raw_SSIM_note': 'Saved raw SSIM uses skimage; it is not the differentiable VM guard definition',
        'paired_photographic_TRAIN_only': True, 'unexposed_not_DEV_or_final': True,
        'native_or_reserved_used': False, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0,
        'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
        'bindings_sha256': bindings, 'script_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started}
    with (OUT / 'analysis.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'comparisons': 8, 'visual_pages': 20, 'jointly_eligible_subset_variants': jointly_eligible}))


if __name__ == '__main__':
    main()
