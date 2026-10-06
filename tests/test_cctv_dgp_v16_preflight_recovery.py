"""Regression checks for the actual bootstrap call and evidence guards; no models."""
import ast
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v16_recovery', ROOT / 'scripts/recover_cctv_dgp_broader_codes_v16_preflight_r1.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@contextmanager
def scratch():
    path = ROOT / 'scratch' / ('v16-preflight-' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.resolve().is_relative_to((ROOT / 'scratch').resolve()) and path.name.startswith('v16-preflight-'):
            shutil.rmtree(path)


class PreflightRecoveryTests(unittest.TestCase):
    def test_original_frozen_verify_reproduces_the_string_path_error(self):
        sys.path.insert(0, str(ROOT))
        import cctv_dgp_broader_codes_v16 as original
        with self.assertRaisesRegex(TypeError, "unsupported operand type"):
            original.verify('root', 'parent', 'mixed', 'baseline', recovery.PROTOCOL_SHA)

    def test_exact_child_code_reproduces_bug_and_corrects_all_four_paths(self):
        # These stubs check the call interface, without CUDA, training or assets.
        with scratch() as base:
            parent = base / 'parent'; parent.mkdir()
            root = base / 'root'; root.mkdir()
            mixed, baseline = base / 'mixed', base / 'baseline'
            (parent / 'cctv_dgp_targets_v6.py').write_text('def require_vm(root): pass\n')
            (root / 'cctv_dgp_broader_codes_v16.py').write_text(
                'from pathlib import Path\n'
                'def verify(root,parent,mixed,baseline,pin):\n'
                '    value = root / "plan.json"\n'
                '    assert all(isinstance(x, Path) for x in [root,parent,mixed,baseline])\n')
            original = recovery.probe(sys.executable,
                recovery.preflight_code(root, parent, mixed, baseline, corrected=False), base, 'original')
            self.assertEqual(original['returncode'], 1)
            self.assertIn(recovery.PATH_TYPE_ERROR, original['stderr'])
            corrected = recovery.probe(sys.executable,
                recovery.preflight_code(root, parent, mixed, baseline, corrected=True), base, 'corrected')
            self.assertEqual(corrected['returncode'], 0)
            self.assertIn('preflight passed', corrected['stdout'])
            self.assertFalse((root / '__pycache__').exists())
            self.assertEqual(json.loads((base / 'original.json').read_text())['stderr'], original['stderr'])

    @contextmanager
    def bundle(self):
        with scratch() as base:
            root = base / 'forensic-dgp/cctv_dgp_broader_codes_vm_v16'; root.mkdir(parents=True)
            scripts = root / 'scripts'; scripts.mkdir()
            bootstrap = scripts / 'launch_cctv_dgp_broader_codes_v16.py'
            bootstrap.write_text('original = True\n')
            plan = root / 'broader_codes_protocol_v16.json'
            plan.write_text(json.dumps({'assets_sha256': {bootstrap.relative_to(root).as_posix(): recovery.sha(bootstrap)}}))
            archive = base / 'cctv-dgp-broader-codes-v16-execution.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                for path in [bootstrap, plan]:
                    stream.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
            archive_sha = recovery.sha(archive)
            Path(str(archive) + '.sha256').write_text(archive_sha + '  ' + archive.name + '\n')
            with patch.object(recovery, 'ARCHIVE_SHA', archive_sha), patch.object(recovery, 'PROTOCOL_SHA', recovery.sha(plan)), \
                    patch.object(recovery, 'ORIGINAL_LAUNCHER_SHA', recovery.sha(bootstrap)):
                yield root, archive

    def test_unstarted_bundle_is_accepted_but_changed_assets_are_preserved(self):
        with self.bundle() as (root, archive):
            self.assertIn('assets_sha256', recovery.verify_extracted(root, archive))
            bootstrap = root / 'scripts/launch_cctv_dgp_broader_codes_v16.py'
            bootstrap.write_text('changed = True\n')
            with self.assertRaisesRegex(ValueError, 'changed/missing'):
                recovery.verify_extracted(root, archive)
            self.assertEqual(bootstrap.read_text(), 'changed = True\n')

    def test_existing_launch_marker_blocks_recovery_without_modification(self):
        with self.bundle() as (root, archive):
            marker = root / 'supervisor_launch.json'; marker.write_text('{"preserve": true}')
            with self.assertRaisesRegex(RuntimeError, 'existing training'):
                recovery.verify_extracted(root, archive)
            self.assertEqual(marker.read_text(), '{"preserve": true}')

    def test_even_empty_outputs_prevents_training_resume(self):
        with self.bundle() as (root, archive):
            (root / 'outputs').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'existing outputs'):
                recovery.verify_extracted(root, archive)

    def test_local_recovery_rejected_before_any_hash_child_or_write(self):
        with patch.object(recovery, 'sha') as digest, patch.object(recovery, 'probe') as child, \
                patch.object(recovery, 'write') as write:
            with self.assertRaisesRegex(RuntimeError, 'Linux VM'):
                recovery.recover('invalid', launch=True)
            digest.assert_not_called(); child.assert_not_called(); write.assert_not_called()

    @contextmanager
    def mock_vm(self, root):
        base = root.parent.parent
        runtime = root.parent / 'cctv_dgp_vm_bundle/.venv/bin/python'
        runtime.parent.mkdir(parents=True); runtime.write_text('not executed\n')
        shutil.copyfile(root / 'scripts/launch_cctv_dgp_broader_codes_v16.py',
                        base / 'launch_cctv_dgp_broader_codes_v16.py')
        with patch.object(recovery.sys, 'platform', 'linux'), \
                patch.object(recovery.os, 'uname', return_value=SimpleNamespace(nodename='forensic-dgp-thesis'), create=True), \
                patch.object(recovery.Path, 'home', return_value=base), \
                patch.object(recovery.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 * 1024**3)), \
                patch.object(recovery, 'idle'):
            yield

    def test_verify_only_never_launches_and_explicit_launch_keeps_supervisor_and_provenance(self):
        for launch in [False, True]:
            with self.subTest(launch=launch), self.bundle() as (root, archive), self.mock_vm(root):
                receipts = [{'returncode': 1, 'stderr': recovery.PATH_TYPE_ERROR, 'stdout': ''},
                            {'returncode': 0, 'stderr': '', 'stdout': 'passed'}]
                with patch.object(recovery, 'probe', side_effect=receipts), \
                        patch.object(recovery.subprocess, 'run') as run:
                    recovery.recover(recovery.sha(recovery.__file__), launch=launch)
                marker = root / 'supervisor_launch.json'
                if not launch:
                    self.assertFalse(marker.exists()); run.assert_not_called()
                    continue
                receipt = json.loads(marker.read_text())
                self.assertFalse(receipt['preflight_recovery']['training_recipe_changed'])
                self.assertFalse(receipt['automatic_resume'])
                source = receipt['preflight_recovery']['launcher_source']
                self.assertEqual(source, Path(recovery.__file__).read_text())
                command = run.call_args.args[0]
                self.assertEqual(command[:5], ['tmux', 'new-session', '-d', '-s', recovery.SESSION])
                self.assertIn(str(root / 'scripts/supervise_cctv_dgp_broader_codes_v16.py'), shlex.split(command[-1]))
                self.assertIn('PYTHONDONTWRITEBYTECODE=1', command[-1])
                self.assertEqual(recovery.sha(root / 'scripts/launch_cctv_dgp_broader_codes_v16.py'),
                                 recovery.ORIGINAL_LAUNCHER_SHA)

    def test_unexpected_preflight_failure_does_not_start_training(self):
        with self.bundle() as (root, archive), self.mock_vm(root):
            error = {'returncode': 1, 'stderr': 'RuntimeError: Existing CUDA NVIDIA L4 required', 'stdout': ''}
            with patch.object(recovery, 'probe', return_value=error) as probe, \
                    patch.object(recovery.subprocess, 'run') as run, \
                    self.assertRaisesRegex(RuntimeError, 'not the known'):
                recovery.recover(recovery.sha(recovery.__file__), launch=True)
            self.assertEqual(probe.call_count, 1); run.assert_not_called()
            self.assertFalse((root / 'supervisor_launch.json').exists())

    def test_child_failure_and_timeout_are_visible_and_retained(self):
        with scratch() as base:
            failed = subprocess.CompletedProcess([], 1, 'stdout detail\n', 'true error detail\n')
            with patch.object(recovery.subprocess, 'run', return_value=failed):
                receipt = recovery.probe('python', 'code', base, 'failed')
            self.assertEqual(receipt['stderr'], 'true error detail\n')
            self.assertEqual(json.loads((base / 'failed.json').read_text())['returncode'], 1)
            error = subprocess.TimeoutExpired([], 120, output=b'partial stdout', stderr=b'partial stderr')
            with patch.object(recovery.subprocess, 'run', side_effect=error), self.assertRaises(TimeoutError):
                recovery.probe('python', 'code', base, 'timeout')
            self.assertTrue(json.loads((base / 'timeout.json').read_text())['timed_out'])

    def test_recovery_is_python310_compatible(self):
        ast.parse(Path(recovery.__file__).read_text(), feature_version=(3, 10))


if __name__ == '__main__':
    unittest.main()
