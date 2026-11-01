"""Versioned model sets stay isolated until the application pointer activates them.

All weights are tiny byte fixtures. These tests perform no inference, package
installation, model download, registry access or live-application mutation.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from implementation import managed_paths


class ModelStoreTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('implementation.model_store')
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
            self.write_manifest(root, 'implementation', b'old restoration weights')
            self.write_manifest(root, 'completion', b'old completion weights')
        self.pointer = self.managed / 'current.json'
        self.pointer.write_bytes(b'{"schema":1,"commit":"' + b'a' * 40 + b'"}\n')
        self.environment = patch.dict(os.environ, {
            'LOCALAPPDATA': str(self.local),
            'DGP_MANAGED_ROOT': str(self.managed),
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def manifest_path(self, root, family):
        name = 'models.json' if family == 'implementation' else 'completion_assets.json'
        return root / 'implementation' / name

    def write_manifest(self, root, family, content):
        asset = {
            'filename': 'face.pth', 'size': len(content),
            'sha256': hashlib.sha256(content).hexdigest(),
        }
        self.manifest_path(root, family).write_text(json.dumps({
            'schema_version': 1, 'assets': {'face': asset},
            'note': 'Metadata is not part of model byte identity.',
        }))
        return asset

    def marker(self, folder):
        markers = [p for p in folder.iterdir() if p.name.startswith('.') and p.suffix == '.json']
        self.assertEqual(len(markers), 1, 'Preparation must publish one model-set binding marker.')
        return markers[0]

    def test_unmanaged_checkout_keeps_existing_model_locations_without_writes(self):
        with patch.dict(os.environ, {}, clear=True):
            for family in ('implementation', 'completion'):
                expected = self.new / 'models' / family
                self.assertEqual(self.module.model_directory(self.new, family), expected)
                self.assertEqual(self.module.prepare_model_directory(self.new, family), expected)
                self.assertFalse(expected.exists())

    def test_resolution_is_read_only_and_computes_bounded_managed_namespace(self):
        for family in ('implementation', 'completion'):
            folder = self.module.model_directory(self.new, family)
            self.assertEqual(folder.parent, self.managed / 'data' / 'models' / family)
            self.assertRegex(folder.name, r'^[0-9a-f]{32}$')
            self.assertFalse(folder.exists())
        self.assertFalse((self.managed / 'data').exists())

    def test_prepare_binds_full_fingerprint_and_exact_asset_inventory(self):
        folder = self.module.prepare_model_directory(self.new, 'implementation')
        marker = self.marker(folder)
        body = json.loads(marker.read_text())
        expected_assets = json.loads(self.manifest_path(self.new, 'implementation').read_text())['assets']
        self.assertEqual(set(body), {'schema', 'family', 'fingerprint', 'assets'})
        self.assertIs(type(body['schema']), int)
        self.assertEqual(body['schema'], 1)
        self.assertEqual(body['family'], 'implementation')
        self.assertRegex(body['fingerprint'], r'^[0-9a-f]{64}$')
        self.assertEqual(folder.name, body['fingerprint'][:32])
        self.assertEqual(body['assets'], expected_assets)
        self.assertEqual(self.module.model_directory(self.new, 'implementation'), folder)

    def test_changed_restoration_and_completion_weights_get_separate_namespaces(self):
        for family in ('implementation', 'completion'):
            old_folder = self.module.prepare_model_directory(self.old, family)
            (old_folder / 'face.pth').write_bytes(b'old verified weights')
            old_marker = self.marker(old_folder).read_bytes()
            self.write_manifest(self.new, family, b'new verified weights')
            new_folder = self.module.prepare_model_directory(self.new, family)
            self.assertNotEqual(new_folder, old_folder)
            self.assertEqual((old_folder / 'face.pth').read_bytes(), b'old verified weights')
            self.assertEqual(self.marker(old_folder).read_bytes(), old_marker)
            self.assertEqual(self.module.model_directory(self.old, family), old_folder)

    def test_unchanged_family_reuses_its_namespace_when_other_family_changes(self):
        old_restoration = self.module.prepare_model_directory(self.old, 'implementation')
        old_completion = self.module.prepare_model_directory(self.old, 'completion')
        self.write_manifest(self.new, 'completion', b'new completion weights')
        self.assertEqual(self.module.prepare_model_directory(self.new, 'implementation'), old_restoration)
        self.assertNotEqual(self.module.prepare_model_directory(self.new, 'completion'), old_completion)

    def test_source_commit_metadata_and_json_order_do_not_duplicate_unchanged_assets(self):
        old_folder = self.module.prepare_model_directory(self.old, 'implementation')
        body = json.loads(self.manifest_path(self.new, 'implementation').read_text())
        body['note'] = 'New release notes and formatting.'
        body['assets']['face'] = dict(reversed(list(body['assets']['face'].items())))
        self.manifest_path(self.new, 'implementation').write_text(json.dumps(body, indent=4))
        self.assertEqual(self.module.prepare_model_directory(self.new, 'implementation'), old_folder)

    def test_first_generation_never_overwrites_legacy_model(self):
        legacy = self.managed / 'data' / 'models' / 'completion' / 'face.pth'
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b'old active legacy weights')
        self.write_manifest(self.new, 'completion', b'new candidate weights')
        folder = self.module.prepare_model_directory(self.new, 'completion')
        self.assertNotEqual(folder / 'face.pth', legacy)
        self.assertEqual(legacy.read_bytes(), b'old active legacy weights')

    def test_preparation_and_partial_candidate_do_not_activate_release(self):
        pointer_before = self.pointer.read_bytes()
        self.write_manifest(self.new, 'completion', b'candidate that will fail later')
        folder = self.module.prepare_model_directory(self.new, 'completion')
        (folder / '.face.pth.failed.partial').write_bytes(b'incomplete model')
        self.assertEqual(self.pointer.read_bytes(), pointer_before)
        self.assertFalse((folder / 'face.pth').exists())
        self.assertEqual(self.module.model_directory(self.new, 'completion'), folder)

    def test_missing_marker_in_nonempty_namespace_is_refused_without_data_loss(self):
        folder = self.module.model_directory(self.new, 'completion')
        folder.mkdir(parents=True)
        sentinel = folder / 'face.pth'
        sentinel.write_bytes(b'unproven existing bytes')
        for resolve in (self.module.model_directory, self.module.prepare_model_directory):
            with self.subTest(resolve=resolve.__name__), self.assertRaises(self.module.ModelStoreError):
                resolve(self.new, 'completion')
            self.assertEqual(sentinel.read_bytes(), b'unproven existing bytes')

    def test_empty_interrupted_namespace_can_receive_its_binding(self):
        folder = self.module.model_directory(self.new, 'implementation')
        folder.mkdir(parents=True)
        self.assertEqual(self.module.prepare_model_directory(self.new, 'implementation'), folder)
        self.assertTrue(self.marker(folder).is_file())

    def test_full_fingerprint_mismatch_refuses_short_prefix_collision(self):
        folder = self.module.prepare_model_directory(self.new, 'implementation')
        marker = self.marker(folder)
        body = json.loads(marker.read_text())
        suffix = '0' * 32 if body['fingerprint'][32:] != '0' * 32 else 'f' * 32
        body['fingerprint'] = body['fingerprint'][:32] + suffix
        marker.write_text(json.dumps(body))
        before = marker.read_bytes()
        for resolve in (self.module.model_directory, self.module.prepare_model_directory):
            with self.subTest(resolve=resolve.__name__), self.assertRaises(self.module.ModelStoreError):
                resolve(self.new, 'implementation')
        self.assertEqual(marker.read_bytes(), before)

    def test_marker_schema_family_inventory_and_extra_fields_are_strict(self):
        folder = self.module.prepare_model_directory(self.new, 'completion')
        marker = self.marker(folder)
        original = json.loads(marker.read_text())
        for field, value in (('schema', True), ('family', 'implementation'), ('assets', {}), ('unexpected', 'value')):
            body = dict(original); body[field] = value
            marker.write_text(json.dumps(body))
            with self.subTest(field=field), self.assertRaises(self.module.ModelStoreError):
                self.module.prepare_model_directory(self.new, 'completion')
        marker.write_text(json.dumps(original))

    def test_duplicate_case_insensitive_filenames_and_invalid_fingerprints_are_refused(self):
        path = self.manifest_path(self.new, 'completion')
        original = json.loads(path.read_text())
        duplicate = dict(original['assets']['face']); duplicate['filename'] = 'FACE.PTH'
        body = dict(original); body['assets'] = dict(original['assets'], duplicate=duplicate)
        path.write_text(json.dumps(body))
        with self.assertRaises(self.module.ModelStoreError):
            self.module.model_directory(self.new, 'completion')
        for field, value in (('sha256', 'A' * 64), ('sha256', 'x' * 64), ('size', True), ('size', 0)):
            body = json.loads(json.dumps(original)); body['assets']['face'][field] = value
            path.write_text(json.dumps(body))
            with self.subTest(field=field, value=value), self.assertRaises(self.module.ModelStoreError):
                self.module.model_directory(self.new, 'completion')

    def test_unsafe_windows_filenames_cannot_escape_or_alias_namespace(self):
        path = self.manifest_path(self.new, 'implementation')
        original = json.loads(path.read_text())
        for name in ('../face.pth', 'nested/face.pth', r'nested\face.pth', 'C:face.pth',
                     'face.pth:stream', 'face.pth.', 'face.pth ', 'CON.pth', '', '..'):
            body = json.loads(json.dumps(original)); body['assets']['face']['filename'] = name
            path.write_text(json.dumps(body))
            with self.subTest(name=name), self.assertRaises(self.module.ModelStoreError):
                self.module.prepare_model_directory(self.new, 'implementation')
        self.assertFalse((self.managed / 'data').exists())

    def test_unknown_family_and_invalid_manifest_are_refused(self):
        with self.assertRaises(self.module.ModelStoreError):
            self.module.model_directory(self.new, '../completion')
        path = self.manifest_path(self.new, 'completion')
        for content in ('{', '[]', '{"schema_version":true,"assets":{}}'):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaises(self.module.ModelStoreError):
                self.module.prepare_model_directory(self.new, 'completion')

    def test_linked_namespace_is_refused_and_external_bytes_are_preserved(self):
        folder = self.module.model_directory(self.new, 'implementation')
        folder.parent.mkdir(parents=True)
        outside = self.local / 'outside'; outside.mkdir()
        sentinel = outside / 'face.pth'; sentinel.write_bytes(b'outside data')
        try:
            folder.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(str(exc))
        for resolve in (self.module.model_directory, self.module.prepare_model_directory):
            with self.subTest(resolve=resolve.__name__), self.assertRaises(self.module.ModelStoreError):
                resolve(self.new, 'implementation')
        self.assertEqual(sentinel.read_bytes(), b'outside data')


    def test_duplicate_json_fields_and_oversized_manifests_are_refused(self):
        path = self.manifest_path(self.new, 'completion')
        valid = path.read_text()
        duplicate = valid.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1)
        self.assertNotEqual(duplicate, valid)
        for content in (duplicate, valid + ' ' * (2 * 1024 ** 2)):
            path.write_text(content)
            with self.subTest(oversized=len(content) > 2 * 1024 ** 2), self.assertRaises(self.module.ModelStoreError):
                self.module.model_directory(self.new, 'completion')

    def test_duplicate_json_fields_in_marker_are_not_silently_accepted(self):
        folder = self.module.prepare_model_directory(self.new, 'completion')
        marker = self.marker(folder)
        body = json.loads(marker.read_text())
        duplicated = '{"schema":1,"schema":1,' + json.dumps({k:v for k,v in body.items() if k != 'schema'})[1:]
        marker.write_text(duplicated)
        with self.assertRaises(self.module.ModelStoreError):
            self.module.model_directory(self.new, 'completion')
        self.assertEqual(marker.read_text(), duplicated)

    def test_model_inventory_is_bounded_before_creating_storage(self):
        path = self.manifest_path(self.new, 'completion')
        asset = json.loads(path.read_text())['assets']['face']
        assets = {str(i): dict(asset, filename='face-' + str(i) + '.pth') for i in range(33)}
        path.write_text(json.dumps({'schema_version': 1, 'assets': assets}))
        with self.assertRaises(self.module.ModelStoreError):
            self.module.prepare_model_directory(self.new, 'completion')
        self.assertFalse((self.managed / 'data').exists())

    def test_prevalidated_logical_and_physical_version_directory_identity_is_accepted(self):
        # Model-store selection must preserve data_root's established custom
        # install contract when Windows resolves a logical LocalAppData spelling
        # to the same physical versions directory. The identity mock is bounded
        # to these two exact ordinary fixture directories, never globally True.
        logical_runtime = self.local / 'logical-install' / 'runtime'
        logical_versions = logical_runtime / 'versions'
        logical_versions.mkdir(parents=True)
        logical_data = logical_runtime / 'data'
        permitted = {os.path.normcase(os.path.abspath(self.new.parent)),
                     os.path.normcase(os.path.abspath(logical_versions))}
        def same_directory(left, right):
            observed = {os.path.normcase(os.path.abspath(left)),
                        os.path.normcase(os.path.abspath(right))}
            self.assertEqual(observed, permitted)
            return True
        def computed_data_path(root, *parts):
            self.assertEqual(root, self.new)
            return logical_data.joinpath(*parts)
        with patch.object(self.module, 'data_root', return_value=logical_data) as validated, \
                patch.object(self.module, 'data_path', side_effect=computed_data_path), \
                patch.object(self.module.os.path, 'samefile', side_effect=same_directory) as identity:
            folder = self.module.model_directory(self.new, 'implementation')
        validated.assert_called_once_with(self.new)
        identity.assert_called_once()
        self.assertEqual(folder.parent, logical_data / 'models' / 'implementation')
        self.assertRegex(folder.name, r'^[0-9a-f]{32}$')
        self.assertFalse(logical_data.exists())

    def test_different_physical_version_directory_is_refused_without_writes(self):
        logical_runtime = self.local / 'unrelated-install' / 'runtime'
        logical_versions = logical_runtime / 'versions'
        logical_versions.mkdir(parents=True)
        logical_data = logical_runtime / 'data'
        permitted = {os.path.normcase(os.path.abspath(self.new.parent)),
                     os.path.normcase(os.path.abspath(logical_versions))}
        def different_directory(left, right):
            observed = {os.path.normcase(os.path.abspath(left)),
                        os.path.normcase(os.path.abspath(right))}
            self.assertEqual(observed, permitted)
            return False
        before = self.pointer.read_bytes()
        with patch.object(self.module, 'data_root', return_value=logical_data), \
                patch.object(self.module.os.path, 'samefile', side_effect=different_directory), \
                patch.object(self.module, 'data_path') as create_path:
            with self.assertRaises(self.module.ModelStoreError):
                self.module.prepare_model_directory(self.new, 'implementation')
        create_path.assert_not_called()
        self.assertFalse(logical_data.exists())
        self.assertEqual(self.pointer.read_bytes(), before)

    def test_linked_versions_ancestor_is_refused_before_model_selection(self):
        # An ordinary source leaf does not prove that its versions ancestor is
        # ordinary. Model-path selection must reject a reparse point anywhere
        # between the source and the already validated managed root.
        versions = self.new.parent
        ordinary = self.module._ordinary_existing
        def inspect(path, *, directory):
            if Path(path) == versions:
                raise managed_paths.ManagedPathError('Linked versions ancestor fixture.')
            return ordinary(path, directory=directory)
        before = self.pointer.read_bytes()
        with patch.object(self.module, '_ordinary_existing', side_effect=inspect):
            with self.assertRaises(self.module.ModelStoreError):
                self.module.model_directory(self.new, 'implementation')
        self.assertFalse((self.managed / 'data').exists())
        self.assertEqual(self.pointer.read_bytes(), before)

if __name__ == '__main__':
    unittest.main()
