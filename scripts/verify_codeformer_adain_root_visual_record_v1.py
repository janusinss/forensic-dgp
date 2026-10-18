"""Independent CPU closure of root visual-record bindings, not visual judgment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_pixels import require, sha, read, write


def run_tests(review, module, count, output, sources):
    require(not (review / output).exists(), 'Preserve existing CPU test receipt')
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', module, '-v']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60)
    require(result.returncode == 0 and ('Ran %d tests' % count) in result.stderr and
            result.stderr.rstrip().endswith('OK'), 'Bounded CPU fixture suite failed')
    receipt = {'complete': True, 'command': command, 'returncode': result.returncode, 'CPU_tests': count,
        'stdout': result.stdout, 'stderr': result.stderr,
        'source_sha256': {name: sha(ROOT / name) for name in sources},
        'neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False}
    write(review / output, receipt)
    return {'path': str(review / output), 'sha256': sha(review / output), 'CPU_tests': count}


def verify(review):
    review = Path(review).resolve(strict=True)
    visual_path = review / 'visual_review_v1.json'
    root_review = read(visual_path)
    manifest_path = review / 'visual_sheets/manifest.json'
    manifest = read(manifest_path)
    require(root_review['complete'] and root_review['reviewer'] == 'root' and
        root_review['root_personally_viewed_all_planned_sheets'] is True and
        root_review['visual_acceptance'] is False and root_review['model_qualified'] is False and
        root_review['goal_complete'] is False, 'Root record flags differ')
    require(root_review['source_sha256'] == sha(ROOT / 'scripts/close_codeformer_adain_visual_review_v1.py'),
            'Root visual closure source differs')
    require(root_review['visual_manifest_sha256'] == sha(manifest_path) and
            root_review['guards_sha256'] == sha(review / 'metrics_v1/guards.json'), 'Root record source table differs')
    require(len(root_review['sheets']) == len(manifest['pages']) == 55, 'All55 planned sheets required')
    unique_cases, outputs = set(), []
    source_bindings = 0
    for index, (saved, planned) in enumerate(zip(root_review['sheets'], manifest['pages']), 1):
        path = ROOT / saved['path']
        require(path.resolve() == (review / 'visual_sheets' / planned['file']).resolve() and
            saved['sha256'] == planned['sha256'] == sha(path) and
            saved['case_ids'] == planned['case_ids'] and saved['root_personally_viewed'] is True,
            'Root sheet order/path/hash/cases/view flag differs')
        require(planned['tile_pixels'] == 256 and planned['resampling'] == 'none',
                'Exact source-pixel display rule differs')
        source_bindings += len(planned['bindings'])
        if index <= 54:
            unique_cases.update(planned['case_ids'])
        for row in planned['bindings']:
            if row['role'] in {'original_update0', 'isolated_update50', 'retained_FP32_update50'} or row['role'].startswith('witness_'):
                outputs.append((row['case_id'], row['role']))
    require(len(unique_cases) == 406 and len(outputs) == len(set(outputs)) == 1222 and source_bindings == 1988,
            'Full406/1222output/1988cell visual coverage differs')
    pages = []
    batch_receipts = []
    require(len(root_review['observation_batches']) == 3, 'Three complete root observation batches required')
    for batch_record in root_review['observation_batches']:
        path = ROOT / batch_record['path']
        require(sha(path) == batch_record['sha256'], 'Root observation batch hash differs')
        batch = read(path)
        require(batch['reviewer'] == 'root' and batch['visual_acceptance'] is False and
            batch['pages'] == batch_record['pages'] and batch['notes'] == batch_record['notes'] and batch['notes'],
            'Root batch notes/page declarations differ')
        pages.extend(batch['pages']); batch_receipts.append({'path': str(path), 'sha256': sha(path), 'pages': batch['pages']})
    require(sorted(pages) == list(range(1, 56)) and len(pages) == len(set(pages)),
            'Root observation batches do not cover each page1..55 exactly once')
    critical = root_review['critical_figure']
    figure_path = ROOT / critical['path']
    binding_path = review / 'critical_clear_256_bindings.json'
    require(critical['root_personally_viewed_original_detail'] is True and
            critical['exact256_cells_verified'] == 15 and
            critical['bindings_sha256'] == sha(binding_path) and critical['sha256'] == sha(figure_path),
            'Root critical figure binding differs')
    binding = read(binding_path)
    require(binding['complete'] and binding['figure_sha256'] == critical['sha256'] and
            binding['ids'] == critical['case_ids'] and len(binding['rows']) == 15 and
            binding['resampling'] == binding['enhancement'] == 'none', 'Critical figure source record differs')
    with Image.open(figure_path) as image:
        require(image.mode == 'RGB', 'Critical figure not RGB')
        figure = np.asarray(image).copy()
    critical_cells = []
    for row in binding['rows']:
        path = Path(row['path'])
        require(sha(path) == row['sha256'], 'Critical source file changed')
        with Image.open(path) as image:
            require(image.mode == 'RGB' and image.size == (256, 256), 'Critical source geometry differs')
            pixels = np.asarray(image).copy()
        x0, y0, x1, y1 = row['box']
        require(x1-x0 == y1-y0 == 256 and np.array_equal(figure[y0:y1, x0:x1], pixels),
                'Critical figure crop differs from exact source pixels')
        critical_cells.append({'id': row['id'], 'column': row['column'], 'source_sha256': row['sha256'],
            'pixel_box': row['box'], 'exact_source_pixels': True})
    require(Counter(row['id'] for row in critical_cells) == Counter({rid: 5 for rid in critical['case_ids']}),
            'Each critical case needs all5 source/output columns')
    source = 'scripts/verify_codeformer_adain_root_visual_record_v1.py'
    tests = []
    tests.append(run_tests(review, 'tests.test_codeformer_adain_gradient_isolation_return_pixels_v1', 5,
        'metrics_CPU_tests_v1.json', ['research/codeformer_adain_gradient_isolation_v1/return_pixels.py',
        'tests/test_codeformer_adain_gradient_isolation_return_pixels_v1.py', source]))
    tests.append(run_tests(review, 'tests.test_codeformer_adain_error_decomposition_v1', 4,
        'decomposition_CPU_tests_v1.json', ['research/codeformer_adain_gradient_isolation_v1/return_error_decomposition.py',
        'tests/test_codeformer_adain_error_decomposition_v1.py', source]))
    result = {'complete': True, 'audit_scope': 'CPU visual-record coverage/source/crop verification; no personal visual verdict',
        'root_visual_review_sha256': sha(visual_path), 'visual_manifest_sha256': sha(manifest_path),
        'root_closure_source_sha256': root_review['source_sha256'],
        'source_sha256': sha(__file__), 'all55_sheet_hashes_paths_case_ids_checked': True,
        'root_observation_batches': batch_receipts, 'root_pages_exactly_once': pages,
        'DEV_case_coverage': 406, 'output_cells': 1222, 'source_cell_bindings': 1988,
        'critical_figure_sha256': sha(figure_path), 'critical_binding_sha256': sha(binding_path),
        'critical15_cells': critical_cells, 'critical_cells_exact256_reopened': True,
        'CPU_test_receipts': tests, 'neural_gradient_optimizer_VM_calls': 0,
        'personal_visual_verdict_by_this_verifier': False, 'root_only_personal_review_claim_preserved': True,
        'visual_acceptance': False, 'model_qualified': False, 'goal_complete': False}
    write(review / 'independent_root_visual_record_v1.json', result)
    return {'complete': True, 'root_visual_record_sha256': sha(review / 'independent_root_visual_record_v1.json'),
        'CPU_tests': sum(row['CPU_tests'] for row in tests), 'critical_cells': len(critical_cells)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--review', required=True)
    a = parser.parse_args(); print(json.dumps(verify(a.review), indent=2))
