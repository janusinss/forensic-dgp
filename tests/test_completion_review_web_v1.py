"""Focused security/binding tests for the saved-output-only completion viewer."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('completion_review_viewer', Path(__file__).resolve().parents[1] / 'completion_review_web_v1.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue, cls.assets = V.build_catalogue(V.DEFAULT_ASSETS)
        cls.server = V.ThreadingHTTPServer(('127.0.0.1', 0), V.make_handler(V.DEFAULT_ASSETS, cls.catalogue, cls.assets))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, route, host=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        try:
            if host:
                conn.putrequest('GET', route, skip_host=True)
                conn.putheader('Host', host)
                conn.endheaders()
            else:
                conn.request('GET', route)
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def test_all_exact_cases_and_non_generation_health(self):
        status, _, body = self.request('/health')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data['mode'], 'retained_outputs_only')
        self.assertFalse(data['generation_enabled'])
        self.assertFalse(data['model_qualified'])
        self.assertEqual(len(self.catalogue['cases']), 32)
        self.assertEqual(len(self.catalogue['sheets']), 8)
        self.assertEqual(len(self.assets), 136)

    def test_png_download_is_exact_and_has_attachment_header(self):
        status, headers, body = self.request('/download/1.png')
        asset = self.assets['1']
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'image/png')
        self.assertIn('attachment;', headers['Content-Disposition'])
        self.assertEqual(hashlib.sha256(body).hexdigest(), asset['sha256'])
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')

    def test_large_pngs_fully_transfer_on_same_HTTP11_connection(self):
        # The real browser failure affected large PNG responses while tiny masks
        # loaded. Exercise repeated >64KiB bodies and a full original sheet on
        # one socket, including response framing and exact content fingerprints.
        large = [(key, row) for key, row in self.assets.items() if row['bytes'] > 65536]
        self.assertGreaterEqual(len(large), 8)
        selected = large[:3] + [large[-1]]
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        first_socket = None
        try:
            for key, row in selected:
                conn.request('GET', '/asset/' + key + '.png')
                response = conn.getresponse()
                self.assertEqual(response.version, 11)
                self.assertEqual(response.status, 200)
                self.assertFalse(response.will_close)
                self.assertEqual(int(response.getheader('Content-Length')), row['bytes'])
                body = response.read()
                self.assertEqual(len(body), row['bytes'])
                self.assertEqual(hashlib.sha256(body).hexdigest(), row['sha256'])
                if first_socket is None:
                    first_socket = conn.sock
                self.assertIs(conn.sock, first_socket)
        finally:
            conn.close()

    def test_unknown_ids_and_path_escape_routes_refused(self):
        for route in ['/asset/0.png', '/asset/99999.png', '/asset/../../.env', '/asset/%2e%2e/%2e%2e/.env', '/download/protocol.json', '/api/files?path=.env']:
            with self.subTest(route=route):
                self.assertEqual(self.request(route)[0], 404)

    def test_non_loopback_host_refused(self):
        self.assertEqual(self.request('/api/catalogue', 'example.invalid')[0], 403)

    def test_retained_fingerprint_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'example.json'
            body = b'{}'
            path.write_bytes(body)
            expected = hashlib.sha256(body).hexdigest()
            self.assertEqual(V.checked(Path(temp), 'example.json', expected)[0], body)
            path.write_bytes(b'{"changed":true}')
            with self.assertRaises(ValueError):
                V.checked(Path(temp), 'example.json', expected)

    def test_relative_path_boundary_refuses_escaping_or_windows_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ['../outside.png', '/absolute.png', 'C:/outside.png', 'folder\\outside.png']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    V.safe_file(Path(temp), name)

    def test_catalogue_observations_and_all_case_panels_present(self):
        status, _, body = self.request('/api/catalogue')
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertFalse(data['assisted_quality_qualified'])
        self.assertFalse(data['automatic_quality_qualified'])
        self.assertEqual(data['neural_or_training_calls'], 0)
        self.assertIn('Do not promote', data['decision'])
        for row in data['cases']:
            self.assertEqual(len(row['panels']), 4)
            self.assertTrue(row['observation'])
            for panel in row['panels']:
                self.assertEqual(panel['size'], [256, 256])


class CandidateAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, cls.baseline_assets = V.build_catalogue(V.DEFAULT_ASSETS)
        first = cls.baseline_assets['1']
        cls.png = (V.DEFAULT_ASSETS / first['relative']).read_bytes()

    def fixture(self, root, change=None):
        (root / 'images').mkdir()
        png_hash = hashlib.sha256(self.png).hexdigest()
        rows, outputs = [], []
        for c in self.baseline['cases']:
            relative = 'images/' + c['id'] + '.png'
            (root / relative).write_bytes(self.png)
            rows.append({'id': c['id'], 'output': relative, 'output_sha256': png_hash,
                         'exact_outside_reviewed_mask': True, 'empty_mask_bypass': c['empty_mask_bypass'], 'seconds': 0})
            outputs.append({'id': c['id'], 'file': {'path': relative, 'bytes': len(self.png), 'sha256': png_hash},
                            'exact_outside_reviewed_mask': True, 'empty_mask_bypass': c['empty_mask_bypass']})
        results = {'complete': True, 'assisted_quality_qualification': False, 'rows': rows}
        body = json.dumps(results).encode()
        (root / 'results.json').write_bytes(body)
        audit = {'complete': True, 'cases': 32, 'protocol': {'sha256': V.PINS['protocol.json']},
                 'results': {'path': 'results.json', 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()},
                 'outputs': outputs, 'model_qualified': False, 'training_calls': 0}
        if change:
            change(audit)
        body = json.dumps(audit).encode()
        (root / 'independent_output_audit.json').write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    def test_valid_synthetic_admission_adds_only_fifth_saved_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            digest = self.fixture(root)
            with patch.object(V, 'CANDIDATE_ROOT', root):
                catalogue, assets = V.build_catalogue(V.DEFAULT_ASSETS, digest)
            self.assertEqual(catalogue['mode'], 'audited_saved_comparison_with_migan')
            self.assertEqual(len(assets), 168)
            self.assertTrue(all(len(r['panels']) == 5 for r in catalogue['cases']))
            self.assertFalse(catalogue['candidate']['model_qualified'])
            self.assertFalse(catalogue['candidate']['new_generation_enabled'])

    def test_changed_audit_hash_refuses_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with patch.object(V, 'CANDIDATE_ROOT', root), self.assertRaises(ValueError):
                V.build_catalogue(V.DEFAULT_ASSETS, '0' * 64)

    def test_failed_preservation_or_mismatched_bank_refuses(self):
        changes = [lambda a: a['outputs'][0].update(exact_outside_reviewed_mask=False),
                   lambda a: a['protocol'].update(sha256='0' * 64),
                   lambda a: a.update(model_qualified=True),
                   lambda a: a['outputs'][0]['file'].update(path='../outside.png')]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                digest = self.fixture(root, change)
                with patch.object(V, 'CANDIDATE_ROOT', root), self.assertRaises(ValueError):
                    V.build_catalogue(V.DEFAULT_ASSETS, digest)

    def test_result_changed_after_audit_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            digest = self.fixture(root)
            with (root / 'results.json').open('ab') as stream:
                stream.write(b' ')
            with patch.object(V, 'CANDIDATE_ROOT', root), self.assertRaises(ValueError):
                V.build_catalogue(V.DEFAULT_ASSETS, digest)


if __name__ == '__main__':
    unittest.main()
