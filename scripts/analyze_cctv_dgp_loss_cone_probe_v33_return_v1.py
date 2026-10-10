"""Read audited V33 measurements and create fixed, metadata-selected review pages.

This script performs no neural inference, differentiation, or optimization.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RETURN = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return'
OUT = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    audit_path = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['finite_probe_complete'] and audit['CPU_replay']['outputs'] == 280
    assert audit['local_gradient_calls'] == audit['local_optimizer_updates'] == 0
    p = read(RETURN / 'protocol.json'); assert sha(RETURN / 'protocol.json') == audit['protocol_sha256']
    assert not OUT.exists(), 'Retain any preceding analysis'
    OUT.mkdir(); (OUT / 'pages').mkdir()
    metric_path = ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py'
    spec = importlib.util.spec_from_file_location('audited_delivered_group_arithmetic', metric_path)
    metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
    summary, before_rows, bindings = [], [], {audit_path.relative_to(ROOT).as_posix(): sha(audit_path),
        metric_path.relative_to(ROOT).as_posix(): sha(metric_path)}
    before = {}
    for cohort in p['cohorts']:
        label = cohort['name']
        for state in p['states']:
            folder = RETURN / f'outputs/state{state}_{label}'
            receipt = read(folder / 'before/receipt.json'); before[(state, label)] = receipt
            guards = metrics.compare_groups(before[(0, label)]['groups'], receipt['groups'])
            before_rows.append({'state': state, 'cohort': label,
                'preservation_against_original': guards,
                'degraded_structure_gain_against_original': 1 - receipt['groups']['degraded']['landmark_high_frequency_MSE'] /
                    before[(0, label)]['groups']['degraded']['landmark_high_frequency_MSE']})
            bindings[(folder / 'before/receipt.json').relative_to(ROOT).as_posix()] = sha(folder / 'before/receipt.json')
            old_bad = {(r['group'], r['metric']) for r in guards['failures']}
            for variant in p['variants']:
                sub = folder / variant['name']; current = read(sub / 'receipt.json'); cmp = read(sub / 'comparison.json')
                decision = metrics.compare_groups(before[(0, label)]['groups'], current['groups'])
                assert decision == cmp['preservation_against_original']
                bad = {(r['group'], r['metric']) for r in decision['failures']}
                summary.append({'state': state, 'cohort': label, 'variant': variant['name'],
                    'incremental_structure_gain_percent': 100 * cmp['incremental_degraded_PNG_structure_gain'],
                    'structure_gain_against_original_percent': 100 * (1 - current['groups']['degraded']['landmark_high_frequency_MSE'] /
                        before[(0, label)]['groups']['degraded']['landmark_high_frequency_MSE']),
                    'preservation_against_original': decision,
                    'new_failure_keys_relative_to_before': [list(v) for v in sorted(bad - old_bad)],
                    'resolved_failure_keys_relative_to_before': [list(v) for v in sorted(old_bad - bad)],
                    'actual_linear_component_derivatives': cmp['actual_linear_component_derivatives'],
                    'finite_raw_component_change': cmp['finite_raw_component_change'],
                    'training_capacity_qualified': False})
                for q in [sub / 'receipt.json', sub / 'comparison.json']:
                    bindings[q.relative_to(ROOT).as_posix()] = sha(q)
    assert len(summary) == 24 and len(before_rows) == 4
    # Same prospective metadata-selected references as the pinned CPU replay:
    # first reference of each source, all five profiles, each matched cohort.
    # No selection by successful outputs or metric ranking.
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 13) if font_path.is_file() else font
    columns = [('Input', None), ('Paired target', None), ('Original', (0, 'before')),
        ('Original + cone 1', (0, 'cone_1')), ('Stopped 50', (50, 'before')),
        ('Stopped + summed', (50, 'summed_adamw')), ('Stopped + cone 1', (50, 'cone_1')),
        ('Stopped + cone 1/8', (50, 'cone_eighth'))]
    pages = []
    for cohort in p['cohorts']:
        label = cohort['name']; ids = set(p['CPU_replay_case_ids'][label])
        cases = [c for c in cohort['cases'] if c['id'] in ids]; assert len(cases) == 10
        for begin in range(0, 10, 5):
            subset = cases[begin:begin + 5]
            canvas = Image.new('RGB', (2048, 1480), '#f5f5f5'); draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f'V33 | {label} TRAIN cohort | source: {subset[0]["source"]} | source labels are not ethnicity', fill='black', font=font)
            for slot, (title, _) in enumerate(columns): draw.text((slot * 256 + 5, 29), title, fill='black', font=font)
            page_bindings = {}
            for row, case in enumerate(subset):
                y = 57 + row * 284
                draw.text((8, y), f'{case["id"]} | {case["profile"]}', fill='black', font=small)
                for slot, (_, route) in enumerate(columns):
                    path = MIXED / case['input' if slot == 0 else 'target'] if route is None else RETURN / f'outputs/state{route[0]}_{label}/{route[1]}/{case["id"]}.png'
                    with Image.open(path) as source:
                        im = source.convert('RGB'); assert im.size == (256, 256)
                        canvas.paste(im, (slot * 256, y + 22))
                    page_bindings[path.relative_to(ROOT).as_posix()] = sha(path)
            destination = OUT / f'pages/{label}_{begin // 5 + 1}.png'; canvas.save(destination)
            pages.append({'path': destination.relative_to(ROOT).as_posix(), 'sha256': sha(destination),
                'case_ids': [c['id'] for c in subset], 'input_output_bindings_sha256': page_bindings})
            bindings.update(page_bindings)
    assert len(pages) == 4
    result = {'complete': True, 'independent_audit_sha256': sha(audit_path),
        'protocol_sha256': audit['protocol_sha256'], 'before_states': before_rows, 'variants': summary,
        'visual_pages': pages, 'visual_selection': 'Prospectively frozen CPU replay IDs: first reference per source, all five profiles, both TRAIN cohorts',
        'visual_case_count': 20, 'visual_cells': 160, 'all1400_metrics_audited': True,
        'all1400_outputs_visually_reviewed': False, 'actual_page_review_pending': True,
        'paired_photographic_TRAIN_only': True, 'unexposed_not_DEV_or_final': True,
        'native_or_reserved_used': False, 'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
        'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
        'bindings_sha256': bindings, 'script_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started}
    write(OUT / 'analysis.json', result)
    print(json.dumps({'complete': True, 'comparisons': 24, 'before_states': 4, 'visual_pages': 4, 'seconds': result['seconds']}))


if __name__ == '__main__': main()
