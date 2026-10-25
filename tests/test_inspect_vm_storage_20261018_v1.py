"""Pure source/closure/TLS fixtures; no Google Cloud API/SSH execution."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fresh_vm_inventory_tested', ROOT / 'scripts/inspect_vm_storage_20261018_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Readonly(unittest.TestCase):
    def test_exact_pinned_read_only_guest_source(self):
        self.assertEqual(m.file_pin(m.SOURCE)['sha256'], m.SOURCE_SHA)
        tree = ast.parse(m.SOURCE.read_text(), feature_version=(3, 10))
        text = m.SOURCE.read_text()
        self.assertNotIn('unlink(', text)
        self.assertNotIn('write_text(', text)
        self.assertNotIn('torch', text)
        self.assertIn('signal.alarm(180)', text)
        self.assertTrue(all(n.module in ('datetime', 'pathlib')
                            for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)))

    def test_historical_host_key_pinned(self):
        self.assertEqual(m.file_pin(m.TRUST)['sha256'], m.TRUST_SHA)
        self.assertEqual(m.HOSTKEY, 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E')

    def test_local_closure_requires_independent_actual3bytes(self):
        good = {'complete': True, 'independently_verified': True, 'files_removed': 3,
                'logical_bytes_removed': 15463424100,
                'apply_receipt': {'path': 'outputs/apply.json', 'bytes': 100, 'sha256': 'a' * 64}}
        self.assertEqual(m.require_local_closure(good), good['apply_receipt'])
        for key, value in [('complete', False), ('independently_verified', False),
                           ('files_removed', 2), ('logical_bytes_removed', 0), ('apply_receipt', None)]:
            with self.assertRaises(ValueError):
                m.require_local_closure({**good, key: value})

    def test_tls_validation_kept_enabled_in_child_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            env = m.environment({'TLS_CA_bundle': {'path': 'verified.pem'}})
            self.assertEqual(env['CLOUDSDK_CORE_DISABLE_SSL_VALIDATION'], 'false')
            self.assertEqual(env['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'], 'verified.pem')

    def test_disabled_tls_environment_rejected(self):
        with patch.dict(os.environ, {'CLOUDSDK_CORE_DISABLE_SSL_VALIDATION': 'True'}, clear=True):
            with self.assertRaises(ValueError):
                m.environment({'TLS_CA_bundle': {'path': 'verified.pem'}})

    def test_source_inert_to_import(self):
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(hasattr(m, 'apply'))
        self.assertFalse(hasattr(m, 'upload'))

    def test_driver_exact_api_identity_before_ssh_literal_and_no_start(self):
        text = (ROOT / 'scripts/inspect_vm_storage_20261018_v1.py').read_text()
        self.assertIn("api.get('status') != 'RUNNING'", text)
        self.assertIn("acceleratorType', '').endswith('/nvidia-l4')", text)
        self.assertIn("'--ssh-flag=-batch', '--ssh-flag=-hostkey'", text)
        self.assertNotIn("'start'", text)
        self.assertNotIn("'stop'", text)


if __name__ == '__main__':
    unittest.main()
