"""Tiny offline-model update integration fixtures; no real models or network."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from implementation import completion_assets, completion_install, managed_paths
from implementation import model_store, offline_payload, setup_models


class FixturePayload:
    def __init__(self, sources):
        self.sources = sources
        self.calls = []

    def model_source(self, group, key, asset):
        self.calls.append((group, key, asset['sha256']))
        return self.sources[(group, key)]


class ModelUpdateFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.local = Path(self.temporary.name)
        self.managed = self.local / 'DGP' / 'app'
        self.managed.mkdir(parents=True)
        (self.managed / 'owner.json').write_text(json.dumps(managed_paths.MARKER))
        self.old = self.managed / 'versions' / ('a' * 40)
        self.new = self.managed / 'versions' / ('b' * 40)
        for root in (self.old, self.new):
            (root / 'implementation').mkdir(parents=True)
        self.pointer = self.managed / 'current.json'
        self.pointer.write_bytes(b'{"schema":1,"commit":"' + b'a' * 40 + b'"}\n')
        self.pointer_before = self.pointer.read_bytes()
        self.sources = {}
        self.manifests = {}
        for family, group, names in (
            ('implementation', 'restoration', {'face': 'face.pth'}),
            ('completion', 'completion', {'completion_r2': 'completion.pth', 'covering_detector': 'detector.pth'}),
        ):
            assets = {}
            for key, filename in names.items():
                data = ('new verified ' + key).encode()
                source = self.local / ('incoming-' + filename)
                source.write_bytes(data)
                self.sources[(group, key)] = source
                assets[key] = {'filename': filename, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                legacy = self.managed / 'data' / 'models' / family / filename
                legacy.parent.mkdir(parents=True, exist_ok=True)
                legacy.write_bytes(('old active ' + key).encode())
            self.manifests[family] = {'schema_version': 1, 'assets': assets}
            name = 'models.json' if family == 'implementation' else 'completion_assets.json'
            (self.new / 'implementation' / name).write_text(json.dumps(self.manifests[family]))
        self.payload = FixturePayload(self.sources)
        self.environment = patch.dict(os.environ, {
            'LOCALAPPDATA': str(self.local), 'DGP_MANAGED_ROOT': str(self.managed),
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def restoration_context(self, folder):
        stack = ExitStack()
        stack.enter_context(patch.object(setup_models, 'ROOT', self.new))
        stack.enter_context(patch.object(setup_models, 'MODEL_DIR', folder))
        stack.enter_context(patch.object(setup_models, 'load_manifest', return_value=self.manifests['implementation']))
        stack.enter_context(patch.object(setup_models, 'load_payload', return_value=self.payload))
        stack.enter_context(patch.object(offline_payload, '__file__', str(self.new / 'implementation' / 'offline_payload.py')))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def completion_context(self, folder):
        stack = ExitStack()
        stack.enter_context(patch.object(completion_assets, 'ROOT', self.new))
        stack.enter_context(patch.object(completion_assets, 'MODEL_DIR', folder))
        stack.enter_context(patch.object(completion_assets, 'MANIFEST_PATH', self.new / 'implementation' / 'completion_assets.json'))
        stack.enter_context(patch.object(completion_assets, 'load_manifest', return_value=self.manifests['completion']))
        stack.enter_context(patch.object(completion_install, 'load_payload', return_value=self.payload))
        stack.enter_context(patch.object(offline_payload, '__file__', str(self.new / 'implementation' / 'offline_payload.py')))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def assert_legacy_and_pointer_unchanged(self):
        self.assertEqual(self.pointer.read_bytes(), self.pointer_before)
        for family, names in (('implementation', {'face': 'face.pth'}),
                              ('completion', {'completion_r2': 'completion.pth', 'covering_detector': 'detector.pth'})):
            for key, filename in names.items():
                self.assertEqual((self.managed / 'data' / 'models' / family / filename).read_bytes(),
                                 ('old active ' + key).encode())

    def test_restoration_setup_binds_new_set_before_publishing_updated_weight(self):
        folder = model_store.model_directory(self.new, 'implementation')
        asset = self.manifests['implementation']['assets']['face']
        with self.restoration_context(folder):
            installed = setup_models.download_asset(asset, model_dir=folder)
        self.assertEqual(installed.read_bytes(), self.sources[('restoration', 'face')].read_bytes())
        self.assertEqual(model_store.model_directory(self.new, 'implementation'), folder)
        self.assertTrue((folder / model_store.MARKER).is_file())
        self.assert_legacy_and_pointer_unchanged()

    def test_completion_setup_binds_new_set_before_publishing_updated_weights(self):
        folder = model_store.model_directory(self.new, 'completion')
        with self.completion_context(folder):
            result = completion_install.install_assets(model_dir=folder)
        self.assertTrue(result['complete'])
        self.assertEqual(result['component'], 'completion-r2-distribution-installation')
        self.assertEqual({key for group, key, _ in self.payload.calls if group == 'completion'},
                         {'completion_r2', 'covering_detector'})
        self.assertEqual(model_store.model_directory(self.new, 'completion'), folder)
        for key, asset in self.manifests['completion']['assets'].items():
            self.assertEqual((folder / asset['filename']).read_bytes(), self.sources[('completion', key)].read_bytes())
        self.assert_legacy_and_pointer_unchanged()

    def test_restoration_corrupt_incoming_bytes_do_not_activate_or_damage_legacy(self):
        folder = model_store.model_directory(self.new, 'implementation')
        self.sources[('restoration', 'face')].write_bytes(b'corrupt incoming bytes')
        asset = self.manifests['implementation']['assets']['face']
        with self.restoration_context(folder), self.assertRaises(setup_models.ModelSetupError):
            setup_models.download_asset(asset, model_dir=folder)
        self.assertFalse((folder / asset['filename']).exists())
        self.assertTrue((folder / model_store.MARKER).is_file(), 'The namespace must be bound before any copy begins.')
        self.assert_legacy_and_pointer_unchanged()

    def test_completion_corrupt_incoming_bytes_do_not_activate_or_damage_legacy(self):
        folder = model_store.model_directory(self.new, 'completion')
        self.sources[('completion', 'covering_detector')].write_bytes(b'corrupt incoming bytes')
        with self.completion_context(folder), self.assertRaises(completion_install.CompletionInstallError):
            completion_install.install_assets(model_dir=folder)
        self.assertFalse((folder / 'completion.pth').exists())
        self.assertFalse((folder / 'detector.pth').exists())
        self.assert_legacy_and_pointer_unchanged()

    def test_unchanged_completion_set_rerun_reuses_verified_files_without_new_namespace(self):
        folder = model_store.model_directory(self.new, 'completion')
        with self.completion_context(folder):
            completion_install.install_assets(model_dir=folder)
            original = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()}
            result = completion_install.install_assets(model_dir=folder)
        self.assertEqual(original, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()})
        self.assertTrue(all(a['acquisition'] == 'existing_verified' for a in result['assets']))
        self.assertEqual([p.name for p in folder.parent.iterdir() if p.is_dir()], [folder.name])
        self.assertEqual(model_store.model_directory(self.new, 'completion'), folder)
        self.assert_legacy_and_pointer_unchanged()

    def test_guard_allows_repair_only_inside_prepared_computed_namespace(self):
        folder = model_store.prepare_model_directory(self.new, 'implementation')
        asset = self.manifests['implementation']['assets']['face']
        path = folder / asset['filename']; path.write_bytes(b'broken candidate')
        with patch.object(offline_payload, '__file__', str(self.new / 'implementation' / 'offline_payload.py')):
            offline_payload.protect_active_model(path, 'restoration', asset)
        self.assertEqual(path.read_bytes(), b'broken candidate')
        self.assert_legacy_and_pointer_unchanged()

    def test_guard_refuses_other_namespace_and_tampered_binding(self):
        folder = model_store.prepare_model_directory(self.new, 'completion')
        asset = self.manifests['completion']['assets']['completion_r2']
        other = folder.parent / ('f' * 32); other.mkdir()
        outside = other / asset['filename']; outside.write_bytes(b'unproven other weights')
        with patch.object(offline_payload, '__file__', str(self.new / 'implementation' / 'offline_payload.py')):
            with self.assertRaises(offline_payload.OfflinePayloadError):
                offline_payload.protect_active_model(outside, 'completion', asset)
            marker = folder / model_store.MARKER
            body = json.loads(marker.read_text()); body['family'] = 'implementation'
            marker.write_text(json.dumps(body))
            with self.assertRaises((offline_payload.OfflinePayloadError, model_store.ModelStoreError)):
                offline_payload.protect_active_model(folder / asset['filename'], 'completion', asset)
        self.assertEqual(outside.read_bytes(), b'unproven other weights')
        self.assert_legacy_and_pointer_unchanged()


    def test_guard_refuses_replacement_fingerprint_outside_bound_inventory(self):
        folder = model_store.prepare_model_directory(self.new, 'implementation')
        admitted = self.manifests['implementation']['assets']['face']
        path = folder / admitted['filename']; path.write_bytes(b'damaged candidate')
        for field, value in (('sha256', '0' * 64), ('size', admitted['size'] + 1)):
            replacement = dict(admitted); replacement[field] = value
            with self.subTest(field=field), patch.object(offline_payload, '__file__', str(self.new / 'implementation' / 'offline_payload.py')):
                with self.assertRaises((offline_payload.OfflinePayloadError, model_store.ModelStoreError)):
                    offline_payload.protect_active_model(path, 'restoration', replacement)
            self.assertEqual(path.read_bytes(), b'damaged candidate')
        self.assert_legacy_and_pointer_unchanged()

    def test_completion_verify_only_refuses_binding_changed_after_module_load(self):
        folder = model_store.prepare_model_directory(self.new, 'completion')
        for key, asset in self.manifests['completion']['assets'].items():
            (folder / asset['filename']).write_bytes(self.sources[('completion', key)].read_bytes())
        marker = folder / model_store.MARKER
        original = marker.read_bytes()
        with self.completion_context(folder):
            for tamper in ('missing', 'different'):
                if tamper == 'missing':
                    marker.unlink()
                else:
                    body = json.loads(original); body['fingerprint'] = '0' * 64
                    marker.write_text(json.dumps(body))
                with self.subTest(tamper=tamper), self.assertRaises((completion_install.CompletionInstallError, model_store.ModelStoreError)):
                    completion_install.install_assets(model_dir=folder, verify_only=True)
                marker.write_bytes(original)
        self.assert_legacy_and_pointer_unchanged()

    def test_completion_runtime_asset_lookup_revalidates_binding_after_module_load(self):
        folder = model_store.prepare_model_directory(self.new, 'completion')
        for key, asset in self.manifests['completion']['assets'].items():
            (folder / asset['filename']).write_bytes(self.sources[('completion', key)].read_bytes())
        marker = folder / model_store.MARKER
        before = marker.read_bytes()
        with self.completion_context(folder):
            marker.unlink()
            with self.assertRaises((completion_assets.CompletionAssetError, model_store.ModelStoreError)):
                completion_assets.asset_path('completion_r2', model_dir=folder)
        self.assertFalse(marker.exists())
        self.assertEqual((folder / 'completion.pth').read_bytes(), self.sources[('completion', 'completion_r2')].read_bytes())
        self.assert_legacy_and_pointer_unchanged()

    @unittest.skipUnless(os.name == 'nt', 'Windows MAX_PATH model publication regression')
    def test_completion_stages_long_folder_when_final_model_path_fits_windows_limit(self):
        # The final names fit below 260 characters. A partial name containing
        # both the model filename and a UUID crosses that limit on Windows.
        prefix = self.local / 'long-path-fixture'
        length = 205 - len(str(prefix)) - 1
        self.assertGreater(length, 0)
        folder = prefix / ('x' * length)
        self.assertEqual(len(str(folder)), 205)
        folder.mkdir(parents=True)
        self.assertLess(len(str(folder / 'completion.pth')), 260)
        with patch.object(completion_assets, 'load_manifest', return_value=self.manifests['completion']), \
                patch.object(completion_assets, 'MANIFEST_PATH', self.new / 'implementation' / 'completion_assets.json'), \
                patch.object(completion_install, 'load_payload', return_value=self.payload), redirect_stdout(io.StringIO()):
            result = completion_install.install_assets(model_dir=folder)
        self.assertTrue(result['complete'])
        for key, asset in self.manifests['completion']['assets'].items():
            self.assertEqual((folder / asset['filename']).read_bytes(), self.sources[('completion', key)].read_bytes())
        self.assert_legacy_and_pointer_unchanged()

    def test_restoration_stale_import_binding_never_publishes_into_previous_set(self):
        captured = model_store.prepare_model_directory(self.new, 'implementation')
        old_asset = self.manifests['implementation']['assets']['face']
        (captured / old_asset['filename']).write_bytes(self.sources[('restoration', 'face')].read_bytes())
        snapshot = {p.name: p.read_bytes() for p in captured.iterdir()}
        changed = dict(old_asset, filename='face-updated.pth')
        self.manifests['implementation']['assets']['face'] = changed
        (self.new / 'implementation' / 'models.json').write_text(json.dumps(self.manifests['implementation']))
        self.assertNotEqual(model_store.model_directory(self.new, 'implementation'), captured)
        with self.restoration_context(captured):
            with self.assertRaises((model_store.ModelStoreError, setup_models.ModelSetupError)):
                setup_models.download_asset(changed, model_dir=captured)
        self.assertEqual({p.name: p.read_bytes() for p in captured.iterdir()}, snapshot)
        self.assert_legacy_and_pointer_unchanged()

    def test_completion_stale_import_binding_never_publishes_into_previous_set(self):
        captured = model_store.prepare_model_directory(self.new, 'completion')
        for key, asset in self.manifests['completion']['assets'].items():
            (captured / asset['filename']).write_bytes(self.sources[('completion', key)].read_bytes())
        snapshot = {p.name: p.read_bytes() for p in captured.iterdir()}
        old_asset = self.manifests['completion']['assets']['completion_r2']
        self.manifests['completion']['assets']['completion_r2'] = dict(old_asset, filename='completion-updated.pth')
        (self.new / 'implementation' / 'completion_assets.json').write_text(json.dumps(self.manifests['completion']))
        self.assertNotEqual(model_store.model_directory(self.new, 'completion'), captured)
        with self.completion_context(captured):
            with self.assertRaises((model_store.ModelStoreError, completion_install.CompletionInstallError)):
                completion_install.install_assets(model_dir=captured)
        self.assertEqual({p.name: p.read_bytes() for p in captured.iterdir()}, snapshot)
        self.assert_legacy_and_pointer_unchanged()

if __name__ == '__main__':
    unittest.main()
