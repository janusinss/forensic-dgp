"""Measured model download events never claim verified completion prematurely."""
import contextlib
import hashlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from implementation import setup_models as restoration
from implementation import completion_install as completion
from implementation import capture_install as capture


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.headers = {'Content-Length': str(len(body))}
        self.status = 200

    def geturl(self):
        return 'https://release-assets.githubusercontent.com/fixture'


class Opener:
    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        return Response(self.bodies.pop(0))


def asset(name, body):
    return {'filename': name, 'size': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
            'url': restoration.RELEASE_BASE + name}


class DownloadProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.body = b'fixed measured weights'
        self.events = []

    def record(self, stage, status, message, **values):
        self.events.append(dict(stage=stage, status=status, message=message, **values))
        if status == 'complete' and stage.startswith('download:'):
            # File publication happens only after full size/hash verification.
            self.assertTrue((self.root / message).is_file())

    def assert_measured(self, name):
        rows = [row for row in self.events if row['stage'] == 'download:' + name]
        self.assertGreater(len(rows), 3)
        self.assertEqual(rows[0]['status'], 'downloading')
        self.assertEqual(rows[0]['current'], 0)
        self.assertEqual(rows[-1]['status'], 'complete')
        self.assertEqual(rows[-1]['current'], len(self.body))
        currents = [row['current'] for row in rows]
        self.assertEqual(currents, sorted(currents))
        self.assertTrue(all(0 <= row['current'] <= row['total'] == len(self.body) for row in rows))
        self.assertTrue(all(row['message'] == name for row in rows))

    def restoration_download(self, body=None):
        item = asset('fixture.pth', self.body)
        opener = Opener([self.body if body is None else body])
        with patch.object(restoration, 'event', side_effect=self.record), \
             patch.object(restoration, '_stage_bundled_asset_if_needed'), \
             patch.object(restoration.urllib.request, 'build_opener', return_value=opener), \
             patch.object(restoration, 'CHUNK_SIZE', 3), contextlib.redirect_stdout(io.StringIO()):
            restoration.download_asset(item, self.root)
        return opener

    def completion_download(self, body=None):
        item = asset('completion.pth', self.body)
        manifest = {'assets': {'completion_r1': item}}
        opener = Opener([self.body if body is None else body])
        with patch.object(completion, 'event', side_effect=self.record), \
             patch.object(completion.assets, 'load_manifest', return_value=manifest), \
             patch.object(completion.assets, 'MANIFEST_PATH', self.root / 'manifest.json'), \
             patch.object(completion, 'CHUNK_SIZE', 3), contextlib.redirect_stdout(io.StringIO()):
            (self.root / 'manifest.json').write_text('fixture', encoding='utf8')
            completion.install_assets(model_dir=self.root, opener=opener)
        return opener

    def test_restoration_chunk_progress_is_monotonic_and_verified(self):
        self.restoration_download()
        self.assert_measured('fixture.pth')

    def test_restoration_hash_failure_never_reports_complete(self):
        with self.assertRaises(restoration.ModelSetupError):
            self.restoration_download(b'x' * len(self.body))
        self.assertTrue(self.events)
        self.assertFalse(any(row['status'] == 'complete' for row in self.events))
        self.assertFalse((self.root / 'fixture.pth').exists())

    def test_restoration_cache_hit_does_not_report_a_download(self):
        (self.root / 'fixture.pth').write_bytes(self.body)
        opener = self.restoration_download()
        self.assertEqual(opener.calls, 0)
        self.assertEqual([row['status'] for row in self.events], ['complete'])
        self.assertEqual(self.events[0]['stage'], 'download:fixture.pth')
        self.assertEqual(self.events[0]['current'], len(self.body))

    def test_completion_chunk_progress_is_monotonic_and_verified(self):
        self.completion_download()
        self.assert_measured('completion.pth')

    def test_completion_hash_failure_never_reports_complete(self):
        with self.assertRaises(completion.CompletionInstallError):
            self.completion_download(b'x' * len(self.body))
        self.assertTrue(self.events)
        self.assertFalse(any(row['status'] == 'complete' for row in self.events))
        self.assertFalse((self.root / 'completion.pth').exists())

    def test_completion_cache_hit_does_not_report_a_download(self):
        (self.root / 'completion.pth').write_bytes(self.body)
        opener = self.completion_download()
        self.assertEqual(opener.calls, 0)
        self.assertEqual([row['status'] for row in self.events], ['complete'])
        self.assertEqual(self.events[0]['stage'], 'download:completion.pth')
        self.assertEqual(self.events[0]['current'], len(self.body))

    def test_model_progress_stays_bounded_on_download_overrun(self):
        for installer, operation in ((restoration, self.restoration_download), (completion, self.completion_download)):
            self.events.clear()
            # Fake response declares the expected size but sends extra bytes.
            class OverrunResponse(Response):
                def __init__(self, body):
                    super().__init__(body)
                    self.headers['Content-Length'] = str(len(self_body))
            self_body = self.body
            with self.subTest(installer=installer.__name__), patch(__name__ + '.Response', OverrunResponse):
                with self.assertRaises(RuntimeError):
                    operation(self.body + b'extra')
            self.assertFalse(any(row['status'] == 'complete' for row in self.events))
            self.assertTrue(all(row['current'] <= row['total'] for row in self.events))

    def test_completion_failed_publication_never_reports_complete(self):
        with patch.object(completion, '_publish', side_effect=OSError('fixture publication failure')):
            with self.assertRaises(completion.CompletionInstallError):
                self.completion_download()
        self.assertFalse(any(row['status'] == 'complete' for row in self.events))

    def test_restoration_failed_publication_never_reports_complete(self):
        with patch.object(restoration, '_publish_without_overwrite', side_effect=OSError('fixture publication failure')):
            with self.assertRaises(restoration.ModelSetupError):
                self.restoration_download()
        self.assertFalse(any(row['status'] == 'complete' for row in self.events))

    def test_capture_custom_install_routes_pip_cache_to_owned_runtime(self):
        python = self.root / 'Scripts' / 'python.exe'
        python.parent.mkdir()
        python.write_bytes(b'fixture')
        cache = self.root / 'runtime' / 'cache' / 'pip'
        with patch.object(capture, 'capture_python', return_value=python), \
             patch.object(capture, 'verified', side_effect=[False, True]), \
             patch.object(capture, 'event', side_effect=self.record), \
             patch.object(capture, 'custom_install_root', return_value=self.root), \
             patch.object(capture, 'managed_cache', return_value=cache) as cache_path, \
             patch.object(capture.subprocess, 'run') as run:
            self.assertEqual(capture.main([]), 0)
        cache_path.assert_called_once_with('pip')
        args = run.call_args.args[0]
        self.assertEqual(args[args.index('--cache-dir') + 1], str(cache))

    def test_capture_cache_hit_reports_verification_without_package_install(self):
        with patch.object(capture, 'capture_python', return_value=self.root / 'python.exe'), \
             patch.object(capture, 'verified', return_value=True), \
             patch.object(capture, 'event', side_effect=self.record), \
             patch.object(capture.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capture.main([]), 0)
        run.assert_not_called()
        self.assertEqual([row['status'] for row in self.events], ['running', 'complete'])
        self.assertTrue(all(row['stage'] == 'capture-runtime' and 'current' not in row and 'total' not in row for row in self.events))

    def test_capture_package_install_is_indeterminate_until_verified(self):
        python = self.root / 'Scripts' / 'python.exe'
        python.parent.mkdir()
        python.write_bytes(b'fixture')
        with patch.object(capture, 'capture_python', return_value=python), \
             patch.object(capture, 'verified', side_effect=[False, True]), \
             patch.object(capture, 'event', side_effect=self.record), \
             patch.object(capture.subprocess, 'run'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capture.main([]), 0)
        self.assertEqual(self.events[-1]['status'], 'complete')
        self.assertTrue(all('current' not in row and 'total' not in row for row in self.events))

    def test_capture_package_error_never_reports_complete(self):
        python = self.root / 'Scripts' / 'python.exe'
        python.parent.mkdir()
        python.write_bytes(b'fixture')
        with patch.object(capture, 'capture_python', return_value=python), \
             patch.object(capture, 'verified', return_value=False), \
             patch.object(capture, 'event', side_effect=self.record), \
             patch.object(capture.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'pip')), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(capture.main([]), 1)
        self.assertEqual(self.events[-1]['status'], 'error')
        self.assertFalse(any(row['status'] == 'complete' for row in self.events))


if __name__ == '__main__':
    unittest.main()
