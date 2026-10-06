import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeCorrectionTests(unittest.TestCase):
    def load_launcher(self):
        path = ROOT / 'scripts/launch_cctv_dgp_input_selection_v19_r2.py'
        spec = importlib.util.spec_from_file_location('v19_r2_bootstrap_test', path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_bootstrap_uses_real_new_module_script_paths_and_path_objects(self):
        m = self.load_launcher()
        paths = [Path('/home/u/forensic-dgp/' + k) for k in ['v19_r2', 'parent', 'r2', 'mixed', 'baseline']]
        code = m.preflight_code(*paths, 'pin')
        ast.parse(code, feature_version=(3, 10))
        self.assertIn('from cctv_dgp_input_selection_v19_r2 import verify', code)
        self.assertIn('require_vm(Path(', code)
        self.assertNotIn('_r2_r2', code)
        for stem in ['launch', 'supervise', 'import', 'run', 'audit']:
            text = (ROOT / ('scripts/' + stem + '_cctv_dgp_input_selection_v19_r2.py')).read_text()
            self.assertNotIn('_r2_r2', text)
            tree = ast.parse(text, feature_version=(3, 10))
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith('.py'):
                    if 'cctv_dgp_input_selection' in n.value:
                        path = ROOT / n.value if n.value.startswith('scripts/') else ROOT / 'scripts' / n.value
                        self.assertTrue(path.is_file(), n.value)

    def test_new_runner_saves_failed_outputs_before_exact_gate(self):
        text = (ROOT / 'scripts/run_cctv_dgp_input_selection_v19_r2.py').read_text()
        self.assertLess(text.index("save('raw/' + key"), text.index("'Canonical fresh DGP PNG differs"))
        self.assertLess(text.index("save('predictions/' + key"), text.index("'Canonical fresh DGP PNG differs"))
        self.assertIn('core.core.dgp(retained_input(camera,', text)
        self.assertIn('internal_base, raw, arrays = predict(camera)', text)
        self.assertIn("'internal_spatial_DGP_base_raw': internal_name", text)

    def test_no_optimizer_or_backward_and_process_stops_are_finite(self):
        forbidden = {'backward', 'step', 'enable_vm_training', 'Adam', 'AdamW', 'SGD', 'grad'}
        text = (ROOT / 'scripts/run_cctv_dgp_input_selection_v19_r2.py').read_text()
        tree = ast.parse(text)
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(f, ast.Attribute) and f.attr in forbidden or
                             isinstance(f, ast.Name) and f.id in forbidden for f in calls))
        self.assertLess(text.index('require_vm(root)'), text.index('load_frozen_dgp_restorer(parent /'))
        supervisor = (ROOT / 'scripts/supervise_cctv_dgp_input_selection_v19_r2.py').read_text()
        self.assertIn('start_new_session=True', supervisor)
        self.assertIn('os.killpg(child.pid, signal.SIGINT)', supervisor)
        self.assertIn('os.killpg(child.pid, signal.SIGKILL)', supervisor)
        self.assertIn("'inference.log', 1200", supervisor)
        self.assertIn("'audit.log', 300", supervisor)

    def test_idle_tmux_shell_allowed_busy_or_unknown_task_rejected(self):
        m = self.load_launcher()
        self.assertEqual(m.live_tmux_tasks('shell\tbash\t0\nlogs\ttail\t0'), [])
        self.assertEqual(m.live_tmux_tasks('training\tpython\t0'), [{'session': 'training', 'command': 'python'}])
        with self.assertRaises(ValueError):
            m.live_tmux_tasks('unknown')


if __name__ == '__main__':
    unittest.main()
