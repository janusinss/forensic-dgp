"""Timing, scientific-scope, bootstrap-type and VM guard regressions; zero model calls."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest

from cctv_dgp_cache_timing_v16_r2 import cache_order, projection, verify_cache_timing

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'pilots/cctv_dgp_broader_codes_v16_r2'
BASE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def plan():
    value = json.loads((BASE / 'broader_codes_protocol_v16.json').read_text())
    value['design']['cache_timing_at_reference'] = 30
    return value


def samples(value):
    rows = []; seen = set()
    for ref in cache_order(value)[:30]:
        key = (ref['source'], ref['role'])
        # Deliberately different train/validation costs and expensive warmups.
        seconds = .1 if ref['role'] == 'train' else .5
        if key not in seen:
            seconds += 2
        rows.append({'id': ref['id'], 'source': ref['source'], 'role': ref['role'], 'cases': 5,
                     'seconds': seconds, 'warmup': key not in seen})
        seen.add(key)
    return rows


class TimingRevisionTests(unittest.TestCase):
    def test_source_role_sample_and_full_cohort_are_fixed_without_duplication(self):
        p = plan(); ordered = cache_order(p)
        self.assertEqual(len(ordered), 885)
        self.assertEqual({ref['id'] for ref in ordered}, {ref['id'] for ref in p['references']})
        for source in sorted({ref['source'] for ref in ordered}):
            self.assertEqual(sum(ref['source'] == source and ref['role'] == 'train' for ref in ordered[:30]), 10)
            self.assertEqual(sum(ref['source'] == source and ref['role'] == 'validation' for ref in ordered[:30]), 5)

    def test_startup_and_warmups_are_counted_once_instead_of_multiplied(self):
        p = plan(); rows = samples(p)
        #30 refs: train20*.1 + validation10*.5 + four2s warmups =15s work.
        elapsed, startup = 36., 20.
        record = projection(p, rows, elapsed, startup)
        # Remaining train761*.1 plus validation94*.5 =123.1s; safety1.25+30s reserve.
        self.assertAlmostEqual(record['projected_seconds'], 219.875)
        more_startup = projection(p, rows, elapsed + 100, startup + 100)
        self.assertAlmostEqual(more_startup['projected_seconds'] - record['projected_seconds'], 100.)
        rows[0]['seconds'] += 50
        more_warmup = projection(p, rows, elapsed + 50, startup)
        self.assertAlmostEqual(more_warmup['projected_seconds'] - record['projected_seconds'], 50.)
        self.assertFalse(record['startup_scaled']); self.assertFalse(record['warmups_scaled'])

    def test_slow_steady_cache_still_fails_unchanged_900_second_cap(self):
        p = plan(); rows = samples(p)
        for row in rows:
            row['seconds'] = 2
        record = projection(p, rows, 81., 20.)
        self.assertFalse(record['passed']); self.assertEqual(record['cap_seconds'], 900)

    def test_tampering_and_nonfinite_reference_rates_are_rejected(self):
        p = plan(); rows = samples(p); record = projection(p, rows, 36., 20.)
        self.assertEqual(verify_cache_timing(p, json.loads(json.dumps(record))), record)
        changed = copy.deepcopy(record); changed['projected_seconds'] -= 1
        with self.assertRaisesRegex(ValueError, 'independent arithmetic'):
            verify_cache_timing(p, changed)
        for field, value in [('role', 'train'), ('seconds', float('nan')), ('warmup', False)]:
            changed = copy.deepcopy(rows); index = 20 if field == 'role' else 0; changed[index][field] = value
            with self.assertRaises(ValueError):
                projection(p, changed, 36., 20.)

    def test_missing_duplicate_and_uncovered_samples_are_rejected(self):
        p = plan(); rows = samples(p)
        for changed in [rows[:-1], [rows[0]] * 30]:
            with self.assertRaises(ValueError):
                projection(p, changed, 36., 20.)
        changed = copy.deepcopy(rows); changed[0]['cases'] = 4
        with self.assertRaisesRegex(ValueError, 'profile coverage'):
            projection(p, changed, 36., 20.)

    def test_learning_settings_objective_and_snapshot_code_match_original(self):
        v = load(WORK / 'cctv_dgp_broader_codes_v16.py', 'r2_contract_for_test')
        original = json.loads((BASE / 'broader_codes_protocol_v16.json').read_text())['design']
        changed = {key for key in v.DESIGN if key not in original or v.DESIGN[key] != original[key]}
        self.assertEqual(changed, {'cache_timing_at_reference', 'cache_projection_method'})
        def inner_functions(path):
            tree = ast.parse(path.read_text())
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name in ['objective', 'snapshot', 'batch']}
        self.assertEqual(inner_functions(WORK / 'scripts/train_cctv_dgp_broader_codes_v16.py'),
                         inner_functions(BASE / 'scripts/train_cctv_dgp_broader_codes_v16.py'))

    def test_bootstrap_uses_path_values_and_visible_preflight_error_capture(self):
        module = load(WORK / 'scripts/launch_cctv_dgp_broader_codes_v16.py', 'r2_launch_for_test')
        code = module.preflight_code(Path('root'), Path('parent'), Path('mixed'), Path('baseline'), 'pin')
        tree = ast.parse(code)
        verify = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'verify')
        self.assertTrue(all(isinstance(arg, ast.Call) and arg.func.id == 'Path' for arg in verify.args[:4]))
        with self.assertRaisesRegex(RuntimeError, 'Linux VM'):
            module.launch('pin', 'archive')

    def test_all_new_sources_compile_for_python310_without_neural_audit_imports(self):
        v = load(WORK / 'cctv_dgp_broader_codes_v16.py', 'r2_contract_syntax_test')
        for name in v.SOURCES_FILES:
            ast.parse((WORK / name).read_text(), feature_version=(3, 10))
        for name in ['cctv_dgp_cache_timing_v16_r2.py', 'scripts/audit_cctv_dgp_broader_codes_v16.py']:
            tree = ast.parse((WORK / name).read_text())
            modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
            modules += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any(module and (module == 'torch' or module.startswith('models')) for module in modules))


if __name__ == '__main__':
    unittest.main()
