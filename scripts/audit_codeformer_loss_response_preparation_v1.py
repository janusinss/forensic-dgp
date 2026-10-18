"""Independent local byte/data closure, without executing a camera or neural model."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_loss_response_v1 import contract as c
import numpy as np
from PIL import Image, __version__ as pillow_version


def pin(path):
    path = c.no_links(path)
    before = path.stat()
    value = {'sha256': c.sha256(path), 'bytes': before.st_size}
    after = path.stat()
    c.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
              (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed during read')
    return value


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-receipt', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    c.require(not args.receipt.exists(), 'Preserve previous audit')
    stage_meta = c.read_json(args.stage_receipt)
    stage = c.no_links(Path(stage_meta['stage']))
    protocol = c.read_json(stage / 'protocol.json')
    c.verify_protocol(protocol)
    c.require(c.sha256(stage / 'protocol.json') == stage_meta['protocol_sha256'], 'Protocol stage identity differs')
    c.require(c.sha256(ROOT / 'scripts/prepare_codeformer_loss_response_v1.py') == stage_meta['source_sha256'],
              'Builder changed since staging')
    names = {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
    c.require(names == set(protocol['files_sha256']) | {'protocol.json'} and len(names) == 17,
              'Exact source-only17-file namespace required')
    own = {}
    for name in sorted(names):
        c.safe_relative(name)
        value = pin(stage / name)
        own[name] = value
        if name != 'protocol.json':
            c.require(value == {'sha256': protocol['files_sha256'][name], 'bytes': protocol['files_bytes'][name]},
                      'Own file binding differs')
            c.require(stage_meta['source_files'][name] == value['sha256'] and
                      stage_meta['source_bytes'][name] == value['bytes'], 'Stage source-map differs')
        if name.startswith(('research/', 'scripts/')):
            if name == 'research/__init__.py':
                c.require((stage / name).read_bytes() == b'', 'Intentional empty namespace init differs')
            else:
                c.require(pin(ROOT / name) == value, 'Current executable source differs')
            ast.parse((stage / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    prior = c.read_json(ROOT / 'outputs/codeformer_loss_response_design_v1/stage_v1.json')
    c.require(protocol['dependencies'] == prior['dependencies'], 'Revised stage must not change frozen parent/data')
    c.require(pin(stage / 'cases.json') == pin(Path(prior['stage']) / 'cases.json'), 'Revised source selection changed')
    resolution = stage_meta['local_dependency_resolution']
    deps = protocol['dependencies']
    c.require(set(deps) == set(resolution) and len(deps) == 2011, 'Dependency namespace differs')
    def verify(item):
        name, expected = item
        c.safe_relative(name)
        path = Path(resolution[name])
        value = pin(path)
        c.require(value == expected and expected == stage_meta['dependencies'][name], 'Fresh dependency differs: ' + name)
        return name, {'path': str(path), **value}
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked = dict(pool.map(verify, deps.items()))
    cases = c.read_json(stage / 'cases.json')
    c.verify_cases(cases)
    history = [c.read_json(Path(resolution[r['pair_record_path']])) for r in cases['cases']]
    rows = []
    for row, saved_record in zip(cases['cases'], history):
        record, parameters = row['record'], row['parameters']
        c.require(saved_record == record and record['role'] == 'train', 'Stored record or role differs')
        with Image.open(resolution[row['source_path']]) as im:
            c.require(im.mode == 'RGB' and im.size == (1024, 1024), 'Source support differs')
            clean = np.asarray(im.resize((512, 512), Image.Resampling.LANCZOS)).copy()
        clean_f32 = np.ascontiguousarray(clean.astype(np.float32) / np.float32(255))
        c.require(array_sha(clean) == record['clean512_rgb8_sha256'] and
                  array_sha(clean_f32) == record['clean512_float32_sha256'], 'Clean target pixels differ')
        with Image.open(resolution[row['input_path']]) as im:
            c.require(im.mode == 'RGB' and im.size == (256, 256), 'Input support differs')
            rgb = np.asarray(im).copy()
        c.require(array_sha(rgb) == record['input256_rgb8_sha256'] and
                  array_sha(np.ascontiguousarray(rgb.astype(np.float32) / np.float32(255))) ==
                  record['input256_float32_sha256'], 'Saved input normalization differs')
        for purpose in ('parameters', 'linear_noise', 'rgb_noise'):
            key = f"codeformer-adaptation-v2|20261015|{row['update']-1}|{row['reference_id']}|{record['pair_role']}|{purpose}"
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
            c.require(parameters['seeds'][purpose] == {'key': key, 'seed64': seed}, 'Keyed seed differs')
            if purpose != 'parameters':
                noise = np.ascontiguousarray(np.random.Generator(np.random.PCG64(seed)).standard_normal(
                    (parameters['long_edge'], parameters['long_edge'], 3)), dtype=np.float64)
                c.require(array_sha(noise) == parameters['noise_fields'][purpose + '_sha256'], 'Keyed noise differs')
        indices = np.load(resolution[row['hq_indices_path']], allow_pickle=False)
        features = np.load(resolution[row['hq_features_path']], allow_pickle=False)
        hq = c.read_json(resolution[row['hq_record_path']])
        c.require(indices.dtype == np.int64 and indices.shape == (1, 256) and indices.min() >= 0 and indices.max() < 1024,
                  'HQ index type/support differs')
        c.require(features.dtype == np.float32 and features.shape == (1, 256, 16, 16) and np.isfinite(features).all(),
                  'HQ feature type/support differs')
        c.require(hq['reference_id'] == row['reference_id'] and hq['role'] == 'train' and
                  hq['source_sha256'] == record['source_sha256'] and hq['clean512_float32_sha256'] ==
                  record['clean512_float32_sha256'], 'HQ source provenance differs')
        retained = hq['exact_retained_FP32_target_parity']
        c.require(retained['indices_array_sha256'] == array_sha(indices) and
                  retained['features_array_sha256'] == array_sha(features), 'HQ retained arrays differ')
        if row['slot'] == 3:
            behavior = c.read_json(resolution[row['behavior_record_path']])
            delivered = np.load(resolution[row['behavior_delivered_path']], allow_pickle=False)
            logits = np.load(resolution[row['behavior_logits_path']], allow_pickle=False)
            c.require(delivered.dtype == logits.dtype == np.float32 and delivered.shape == (1, 3, 256, 256) and
                      logits.shape == (1, 256, 1024) and np.isfinite(delivered).all() and np.isfinite(logits).all(),
                      'Behavior arrays differ')
            c.require(behavior['pair_record'] == record and behavior['input256_float32_sha256'] ==
                      record['input256_float32_sha256'], 'Behavior source join differs')
        rows.append({'case': row['id'], 'clean_and_input_exact': True, 'noise_fields_exact': True,
                     'HQ_join_exact': True, 'behavior_join_exact': row['slot'] == 3})
    test_modules = ['tests.test_codeformer_loss_response_' + kind + '_v1' for kind in
                    ('contract', 'worker', 'transport', 'data_resource')]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in test_modules)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    c.require(result.wasSuccessful() and result.testsRun == 46 and len(result.skipped) == 2,
              'Current46CPU test census or result differs')
    c.require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'Neural framework imported')
    for name, value in own.items():
        c.require(pin(stage / name) == value, 'Stage changed during audit')
    receipt = {'complete': True, 'CPU_only': True, 'manual_unrun': True, 'source_data_review_pass': True,
        'protocol_sha256': stage_meta['protocol_sha256'], 'stage_receipt_sha256': c.sha256(args.stage_receipt),
        'source_sha256': c.sha256(__file__), 'UTC': datetime.now(timezone.utc).isoformat(),
        'own_files': own, 'dependencies': checked, 'dependency_count': len(checked),
        'dependency_bytes': sum(value['bytes'] for value in checked.values()), 'cases': rows,
        'CPU_tests_run': result.testsRun, 'CPU_tests_passed': result.testsRun-len(result.skipped),
        'CPU_tests_skipped': [(str(test), reason) for test, reason in result.skipped], 'CPU_test_output': stream.getvalue(),
        'test_source_pins': {name: pin(ROOT / (name.replace('.', '/') + '.py')) for name in test_modules},
        'local_numpy': np.__version__, 'local_Pillow': pillow_version,
        'full_camera_regeneration_executed_by_this_audit': False,
        'full_camera_local_runtime_parity_pass': False,
        'full_camera_limit': 'Preserved independent local runtime diagnostic:11/16 exact,5 fail; manual compatible VM must regenerate all16 exactly before model construction; no numeric relaxation',
        'manual_VM_preflight_executed': False, 'VM_layout_simulated': False,
        'actual_neural_gradient_optimizer_VM_calls': 0, 'checkpoint_loaded': False,
        'model_qualified': False}
    c.write_new_json(args.receipt, receipt)
    print({'complete': True, 'protocol_sha256': receipt['protocol_sha256'], 'dependencies': len(checked),
           'cases': len(rows), 'CPU_tests_run': result.testsRun, 'CPU_tests_skipped': len(result.skipped),
           'actual_neural_gradient_optimizer_VM_calls': 0})


if __name__ == '__main__':
    main()
