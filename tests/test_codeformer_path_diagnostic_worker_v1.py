"""CPU environment/export/process guard fixtures; no model imports or execution."""
import ast
from io import BytesIO
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

from research.codeformer_path_diagnostic_v1 import contract as c
from research.codeformer_path_diagnostic_v1 import worker as w
from scripts import supervise_codeformer_path_diagnostic_v1 as supervisor

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def owned_fixture_directory():
    path = ROOT / 'scratch' / ('path_diagnostic_worker_' + uuid.uuid4().hex)
    assert path.resolve().is_relative_to((ROOT / 'scratch').resolve())
    path.mkdir(parents=True)
    # Retain these tiny sandbox fixtures for audit; no filesystem cleanup action.
    yield path


class WorkerGuards(unittest.TestCase):
    def test_windows_refuses_before_transfer_or_neural(self):
        with patch.object(w.platform, 'system', return_value='Windows'):
            with patch.object(c, 'verify_transfer') as verify:
                with self.assertRaisesRegex(ValueError, 'Linux L4'):
                    w.run(ROOT, '1' * 64)
                verify.assert_not_called()

    def test_exact_manual_tmux_path_required(self):
        with owned_fixture_directory() as directory:
            home = Path(directory)
            root = home / 'forensic-dgp' / c.PACKAGE_DIR
            root.mkdir(parents=True)
            self.assertEqual(w.manual_root(root, system='Linux', home=home,
                                           environment={'TMUX': 'test'}), root.resolve())
            with self.assertRaisesRegex(ValueError, 'tmux'):
                w.manual_root(root, system='Linux', home=home, environment={})
            other = home / 'forensic-dgp' / 'old_historical_worker'
            other.mkdir()
            with self.assertRaisesRegex(ValueError, 'exact new'):
                w.manual_root(other, system='Linux', home=home, environment={'TMUX': 'test'})

    def test_compressed_cap_checks_before_write(self):
        sink = BytesIO()
        bounded = w.BoundedArchive(sink, 4, time.monotonic() + 5)
        self.assertEqual(bounded.write(b'abcd'), 4)
        with self.assertRaisesRegex(ValueError, 'byte cap'):
            bounded.write(b'e')
        self.assertEqual(sink.getvalue(), b'abcd')

    def test_export_deadline_checks_before_write(self):
        sink = BytesIO()
        with self.assertRaisesRegex(ValueError, 'time exceeded'):
            w.BoundedArchive(sink, 100, time.monotonic() - 1).write(b'x')
        self.assertEqual(sink.getvalue(), b'')

    def test_weights_cannot_enter_export(self):
        with owned_fixture_directory() as directory:
            path = Path(directory)
            (path / 'hybrid.pth').write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError, 'model/optimizer weights'):
                w.reject_unexpected_files(path)

    def test_process_guard_ignores_idle_tmux_and_own_supervisor(self):
        rows = [(10, 'tmux new-session -A -s diagnostic'),
                (11, 'python -m scripts.supervise_codeformer_path_diagnostic_v1 --root .'),
                (12, 'python serve_restoration_checkpoint_comparison_v1.py')]
        self.assertEqual(supervisor.competing_commands(rows, 11), [])

    def test_process_guard_sees_cpu_loading_workers_and_supervisors(self):
        for command in ['python -m research.codeformer_adaptation_v2.worker --root . --run',
                        'python train_cctv_dgp_spatial.py',
                        'python scripts/supervise_codeformer_adaptation_v2.py --root .',
                        'python -m research.codeformer_path_diagnostic_v1.worker --run']:
            self.assertEqual(len(supervisor.competing_commands([(5, command)], 1)), 1)

    def test_live_worker_cannot_export_despite_terminal_file(self):
        with owned_fixture_directory() as directory:
            root = Path(directory)
            out = root / 'outputs/path_diagnostic_v1'
            out.mkdir(parents=True)
            pin = '1' * 64
            (out / 'manual_launch.json').write_text(json.dumps({'pid': 1234, 'protocol_sha256': pin}))
            proc = root / 'fakeproc'
            current = proc / '1234'
            current.mkdir(parents=True)
            (current / 'cmdline').write_bytes(('python\0-m\0research.codeformer_path_diagnostic_v1.worker\0--run\0--protocol-sha\0' + pin).encode())
            with self.assertRaisesRegex(ValueError, 'still live'):
                w.require_worker_terminal(root, pin, proc_base=proc)
            (current / 'cmdline').write_bytes(b'python\0unrelated_server.py\0')
            w.require_worker_terminal(root, pin, proc_base=proc)
            w.require_worker_terminal(root, pin, proc_base=root / 'absentproc')

    def test_direct_supervisor_help_is_runnable(self):
        result = subprocess.run([sys.executable, '-I', '-B',
            str(ROOT / 'scripts/supervise_codeformer_path_diagnostic_v1.py'), '--help'],
            cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--protocol-sha', result.stdout)

    def test_wrappers_are_python310_and_no_neural_import_on_import(self):
        for relative in ['research/codeformer_path_diagnostic_v1/worker.py',
                         'scripts/supervise_codeformer_path_diagnostic_v1.py']:
            tree = ast.parse((ROOT / relative).read_text(), feature_version=(3, 10))
            top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
            self.assertFalse(any('torch' in ast.unparse(node) or '.neural' in ast.unparse(node) for node in top_imports))


if __name__ == '__main__':
    unittest.main()
