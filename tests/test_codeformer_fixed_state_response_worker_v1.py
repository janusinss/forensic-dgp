"""CPU-only finite-measurement guards; never import a neural framework."""
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_fixed_state_response_v1 import contract as c, worker as w


class FrozenMeasurementWorker(unittest.TestCase):
    def test_local_cli_refuses_before_framework_or_vm_work(self):
        result = subprocess.run([sys.executable, '-B', '-m', c.MODULE, '--root', str(ROOT),
            '--protocol-sha', 'a' * 64, '--run'], cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Manual NVIDIA L4 VM only', result.stderr)
        self.assertFalse((ROOT / c.LOCK_NAME).exists())

    def test_census_requires_all656_forwards_and_explicit_zero_training(self):
        self.assertTrue(w.counts_valid(dict(c.COUNTS), terminal=True))
        self.assertEqual(sum(c.COUNTS.values()), 656)
        for key in ('gradient_queries', 'backward_calls', 'optimizer_constructions', 'optimizer_steps', 'committed_training_updates'):
            changed = dict(c.COUNTS); changed[key] = 1
            with self.assertRaises(ValueError): w.counts_valid(changed)
        changed = dict(c.COUNTS); changed['controller_fitted_final'] -= 1
        self.assertTrue(w.counts_valid(changed))
        with self.assertRaises(ValueError): w.counts_valid(changed, terminal=True)

    def test_count_boolean_unknown_and_overflow_are_rejected(self):
        for value in (True, -1, 201, .0, None):
            changed = dict(c.COUNTS); changed['controller_fitted_final'] = value
            with self.assertRaises(ValueError): w.counts_valid(changed)
        changed = dict(c.COUNTS); changed['unplanned'] = 0
        with self.assertRaises(ValueError): w.counts_valid(changed)

    def test_exact_live_ancestry_admits_only_this_supervisor(self):
        root = Path('/home/research/forensic-dgp') / c.VM_NAME; pin = 'a' * 64
        import shlex
        module = shlex.join([sys.executable, '-B', '-u', '-m', c.MODULE, '--root', str(root), '--protocol-sha', pin, '--run'])
        supervisor = shlex.join([sys.executable, '-B', '-u',
            str(root / 'scripts/supervise_codeformer_fixed_state_response_v1_vm.py'), '--root', str(root), '--protocol-sha', pin])
        rows = [(300, 200, module), (200, 100, 'timeout --signal=TERM --kill-after=30 900 ' + module),
                (100, 1, supervisor), (1, 0, 'systemd')]
        conflicts, admitted = w.idle_conflicts(rows, 300, root, pin)
        self.assertEqual(conflicts, [])
        self.assertEqual(admitted, [100, 200, 300])
        for mutation in ('--kill-after=60', '--protocol-sha ' + 'b' * 64):
            changed = deepcopy(rows)
            command = changed[1][2].replace('--kill-after=30', mutation) if mutation.startswith('--kill') else \
                changed[1][2].replace('--protocol-sha ' + pin, mutation)
            changed[1] = (200, 100, command)
            with self.assertRaises(ValueError): w.idle_conflicts(changed, 300, root, pin)

    def test_distinct_training_process_is_never_excluded(self):
        root = Path('/home/research/forensic-dgp') / c.VM_NAME
        rows = [(300, 1, 'python -m ' + c.MODULE + ' --run'), (1, 0, 'systemd'),
                (900, 1, 'python train.py')]
        conflicts, _ = w.idle_conflicts(rows, 300, root, 'a' * 64)
        self.assertTrue(conflicts)

    def test_fixed_state_summary_retains_dark_regression(self):
        common = {'id': 'same', 'cohort': 'fitted', 'role': 'dark', 'reference_id': 'tr_ffhq_00001',
                  'target_mean_y': .2}
        rows = [{**common, 'head': 'initial', 'delta': 0., 'mean_y': .2, 'loss': 0., 'signed_residual': 0.},
                {**common, 'head': 'final', 'delta': -.02, 'mean_y': .18, 'loss': .0004, 'signed_residual': -.02}]
        summary = w.summarize(rows)
        self.assertFalse(summary['model_qualified'])
        self.assertGreater(summary['paired_fixed_state_comparisons'][0]['loss_change'], 0)
        self.assertEqual(next(r for r in summary['groups'] if r['head'] == 'final')['negative_offsets'], 1)

    def test_source_routes_have_no_optimizer_or_autograd_calls(self):
        source = (ROOT / 'research/codeformer_fixed_state_response_v1/worker.py').read_text()
        tree = ast.parse(source, feature_version=(3, 10))
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & {'backward', 'grad', 'Adam', 'SGD', 'load_state_dict'})
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
            isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch' and n.func.attr == 'save'
            for n in ast.walk(tree)))
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__': unittest.main()
