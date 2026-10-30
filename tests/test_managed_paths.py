"""Managed-path opt-in checks using ordinary temporary fixtures only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from implementation import managed_paths


class ManagedPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.root = self.local / 'DGP' / 'app'
        self.root.mkdir(parents=True)
        self.marker = self.root / 'owner.json'
        self.marker.write_text(json.dumps(managed_paths.MARKER), encoding='utf-8')
        self.environment = patch.dict(os.environ, {'LOCALAPPDATA': str(self.local), 'DGP_MANAGED_ROOT': str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_unmanaged_defaults_to_checkout(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(managed_paths.managed_root())
            self.assertEqual(managed_paths.data_path(Path('checkout'), 'models', 'implementation'),
                             Path('checkout/models/implementation'))

    def test_managed_root_requires_exact_owner_marker_and_fixed_location(self):
        self.assertEqual(managed_paths.managed_root(), self.root)
        self.assertEqual(managed_paths.data_root(Path('checkout')), self.root / 'data')
        self.assertEqual(managed_paths.data_path(Path('checkout'), 'models', 'completion'),
                         self.root / 'data' / 'models' / 'completion')
        for content in ({'schema_version': True, 'repository': 'janusinss/forensic-dgp'},
                        {'schema_version': 1, 'repository': 'wrong'},
                        {'schema_version': 1, 'repository': 'janusinss/forensic-dgp', 'extra': 1}):
            self.marker.write_text(json.dumps(content), encoding='utf-8')
            with self.subTest(content=content), self.assertRaises(managed_paths.ManagedPathError):
                managed_paths.data_root(Path('checkout'))
        self.marker.write_text(json.dumps(managed_paths.MARKER), encoding='utf-8')
        with patch.dict(os.environ, {'DGP_MANAGED_ROOT': str(self.local / 'other')}):
            with self.assertRaises(managed_paths.ManagedPathError):
                managed_paths.data_root(Path('checkout'))

    def test_missing_invalid_marker_or_relative_environment_fails_closed(self):
        self.marker.unlink()
        with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(Path('checkout'))
        self.marker.write_text('{', encoding='utf-8')
        with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(Path('checkout'))
        with patch.dict(os.environ, {'DGP_MANAGED_ROOT': 'relative'}):
            with self.assertRaises(managed_paths.ManagedPathError): managed_paths.data_root(Path('checkout'))

    def test_managed_components_reject_escape_and_linked_directory(self):
        for component in ('..', 'models/nested', 'models\\nested', ''):
            with self.subTest(component=component), self.assertRaises(managed_paths.ManagedPathError):
                managed_paths.data_path(Path('checkout'), component)
        data = self.root / 'data'
        data.mkdir()
        target = self.local / 'target'
        target.mkdir()
        try:
            (data / 'models').symlink_to(target, target_is_directory=True)
        except OSError as exc:
            self.skipTest(str(exc))
        with self.assertRaises(managed_paths.ManagedPathError):
            managed_paths.data_path(Path('checkout'), 'models', 'completion')


if __name__ == '__main__':
    unittest.main()
