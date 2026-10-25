"""Pure source/durability guards; no model construction or inference."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_u96_smoothanchors_lr_fixed_DEV_v1.py'
spec = importlib.util.spec_from_file_location('held_fixed_DEV', SCRIPT)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class PureReadiness(unittest.TestCase):
    def test_inert_import_help_and_selfcheck_without_numerical_modules(self):
        code = """import importlib.util,sys
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','numpy','PIL','psutil','fastapi'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
spec=importlib.util.spec_from_file_location('held',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.self_check();sys.argv=[m.__file__,'--help']
try:m.main()
except SystemExit as e:assert e.code==0
assert 'torch'not in sys.modules
"""
        run = subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPT)], capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_JSON_duplicate_nonfinite_and_overflow_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError): M.strict_json(raw)
        self.assertEqual(M.strict_json('{"x":1.2}'), {'x': 1.2})

    def test_reserved_or_interrupted_forward_is_unknown(self):
        counts = {'full_forward_attempts': 1, 'full_forward_returns': 0, 'completed_cases': 0}
        self.assertTrue(M.outcome(counts, False)['forward_outcome_unknown'])
        counts['full_forward_returns'] = 1
        self.assertFalse(M.outcome(counts, False)['forward_outcome_unknown'])
        self.assertTrue(M.outcome(counts, True)['forward_outcome_unknown'])

    def test_source_has_exact136budget_and_no_training_entrypoint(self):
        self.assertEqual(M.LIMITS['full_forward_attempts'], 136)
        self.assertEqual(M.LIMITS['per_arm_forward_attempts'], 68)
        self.assertEqual(len(M.ARMS), 2)
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'), feature_version=(3, 10))
        prohibited = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('backward', 'step', 'zero_grad'):
                    prohibited.append(node.func.attr)
        self.assertEqual(prohibited, [])
        self.assertNotIn('torch', sys.modules)

    def test_atomic_pending_receipt_is_not_overwritten(self):
        tree = ast.parse(SCRIPT.read_text(encoding='utf-8'), feature_version=(3, 10))
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'worker')
        status = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == 'status')
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            env = dict(vars(M), OUT=Path(directory), counters={'full_forward_attempts': 1,
                'full_forward_returns': 0, 'completed_cases': 0}, pending=True, phase='fixture',
                active_case='synthetic', active_arm=M.ARMS[0], started=time.monotonic(), expected_sha='0' * 64)
            lifted = ast.Module(body=[status], type_ignores=[])
            exec(compile(ast.fix_missing_locations(lifted), '<extracted-pure-status>', 'exec'), env)
            env['status']('first_reserved')
            final_before = (Path(directory) / 'status.json').read_bytes()
            pending = Path(directory) / 'status.pending'; pending.write_bytes(b'original interrupted bytes')
            with self.assertRaisesRegex(ValueError, 'Interrupted pending status'):
                env['status']('later_failure')
            self.assertEqual(pending.read_bytes(), b'original interrupted bytes')
            self.assertEqual((Path(directory) / 'status.json').read_bytes(), final_before)
            self.assertTrue(json.loads(final_before)['forward_outcome_unknown'])

    def test_no_actual_prerequisites_means_plan_preparation_refuses_before_writes(self):
        # CLI argument validation precedes any data/model source read or plan.
        run = subprocess.run([sys.executable, '-B', str(SCRIPT), '--prepare-plan'], capture_output=True, text=True, timeout=15)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('Six actual prerequisite receipts required', run.stderr)


if __name__ == '__main__':
    unittest.main()
