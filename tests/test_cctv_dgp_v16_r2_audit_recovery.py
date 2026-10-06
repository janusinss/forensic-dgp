"""Reproduce the sample-count migration bug without models or trained results."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('r2_recovery',
    ROOT / 'scripts/recover_cctv_dgp_broader_codes_v16_r2_audit.py')
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)
ORIGINAL = (RECOVERY.BUNDLE / RECOVERY.AUDITOR).read_bytes()
DESIGN = json.loads((RECOVERY.BUNDLE / RECOVERY.PLAN).read_text())['design']


def run_timing_loop(source, records, design=DESIGN):
    """Execute the real audit timing loop, not a mirror of its conditions."""
    module = ast.parse(source)
    audit = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'audit')
    loop = next(node for node in audit.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple)
                and [value.id for value in node.target.elts] == ['name', 'countkey', 'count', 'cap'])
    executable = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    stub = types.SimpleNamespace(require=RECOVERY.require, read=lambda path: records[path.name])
    exec(compile(executable, '<actual-audit-timing-loop>', 'exec'),
         {'v': stub, 'd': design, 'out': Path('returned')})


def records():
    return {'cache_timing.json': {'references': 30, 'cap_seconds': 900,
                                  'seconds': 40., 'projected_seconds': 300.},
            'fit_timing.json': {'update': 25, 'cap_seconds': 1200,
                                'seconds': 30., 'projected_seconds': 1000.}}


class AuditRecoveryTests(unittest.TestCase):
    def test_original_bug_reproduced_and_protocol_bound_correction_passes(self):
        with self.assertRaisesRegex(ValueError, 'Timing stop receipt differs'):
            run_timing_loop(ORIGINAL, records())
        run_timing_loop(RECOVERY.corrected_source(ORIGINAL), records())

    def test_exact_two_tuple_changes_preserve_every_other_check(self):
        corrected = RECOVERY.corrected_source(ORIGINAL)
        reverted = corrected
        for before, after in RECOVERY.REPLACEMENTS:
            self.assertEqual(corrected.count(after), 1)
            reverted = reverted.replace(after, before, 1)
        self.assertEqual(reverted, ORIGINAL)
        ast.parse(corrected, feature_version=(3, 10))
        self.assertEqual(RECOVERY.sha(RECOVERY.BUNDLE / RECOVERY.AUDITOR), RECOVERY.AUDITOR_SHA)

    def test_modified_original_cannot_be_silently_recovered(self):
        for mutation in [ORIGINAL + b'\n# changed\n', ORIGINAL.replace(b"'references', 20", b"'references', 21")]:
            with self.subTest(mutation=mutation[-30:]):
                with self.assertRaisesRegex(ValueError, 'unchanged frozen R2 auditor'):
                    RECOVERY.corrected_source(mutation)

    def test_wrong_counts_and_caps_are_still_rejected(self):
        corrected = RECOVERY.corrected_source(ORIGINAL)
        for name, key, wrong in [('cache_timing.json', 'references', 20),
                                 ('fit_timing.json', 'update', 26),
                                 ('cache_timing.json', 'cap_seconds', 901),
                                 ('fit_timing.json', 'cap_seconds', 1201)]:
            with self.subTest(name=name, key=key):
                value = records(); value[name][key] = wrong
                with self.assertRaisesRegex(ValueError, 'Timing stop receipt differs'):
                    run_timing_loop(corrected, value)

    def test_time_limits_and_nonfinite_measurements_are_still_rejected(self):
        corrected = RECOVERY.corrected_source(ORIGINAL)
        for name in records():
            for key, wrong in [('seconds', 0), ('seconds', float('nan')),
                               ('projected_seconds', float('inf')),
                               ('projected_seconds', 1300), ('projected_seconds', 1)]:
                with self.subTest(name=name, key=key, value=wrong):
                    value = records(); value[name][key] = wrong
                    with self.assertRaisesRegex(ValueError, 'Timing stop receipt differs'):
                        run_timing_loop(corrected, value)

    def test_counts_come_from_design_instead_of_replacement_hardcodes(self):
        design = copy.deepcopy(DESIGN); design['cache_timing_at_reference'] = 31; design['timing_update'] = 26
        value = records(); value['cache_timing.json']['references'] = 31; value['fit_timing.json']['update'] = 26
        run_timing_loop(RECOVERY.corrected_source(ORIGINAL), value, design)
        # The complete recovery also checksum-binds the actual immutable R2 design.

    def test_audit_recovery_imports_no_neural_library(self):
        for source in [RECOVERY.corrected_source(ORIGINAL), Path(SPEC.origin).read_bytes()]:
            imports = []
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append((node.module or '').split('.')[0])
            self.assertTrue({'torch', 'torchvision', 'tensorflow', 'jax'}.isdisjoint(imports))


if __name__ == '__main__':
    unittest.main()
