"""Pure/file transport regressions, no model/framework/VM execution."""
import ast
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from research.codeformer_u96_decoder_readout_v1 import entry, packet, install_weights, worker


class TransportTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'scratch'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(scratch.resolve()))
        self.root = Path(self.temp.name) / 'stage'
        self.root.mkdir()
        self.addCleanup(self.temp.cleanup)
        self.output = self.root / packet.OUTPUT_RELATIVE
        self.output.mkdir(parents=True)
        self.sha = 'a' * 64

    def dump_status(self, **changes):
        value = {'protocol_sha256': self.sha, 'counters': {'optimizer_updates': 3, 'optimizer_update_attempts': 3},
                 'optimizer_outcome_unknown': False, 'last_committed_checkpoint': None}
        value.update(changes)
        (self.output / 'status.json').write_text(json.dumps(value), encoding='utf-8')

    def test_contract_constants_match_worker_and_finite_budget(self):
        self.assertEqual(packet.LIMITS, worker.LIMITS)
        self.assertEqual(packet.LEARNING_FIXED, worker.LEARNING)
        self.assertEqual(120 + 56 * 6 + 64 * 4, packet.LIMITS['full_forward_attempts'])
        self.assertEqual(entry.EXPORT_SECONDS, 300)
        self.assertEqual(entry.EXPORT_EXTERNAL_SECONDS, 330)

    def test_status_absent_never_reports_zero(self):
        row = entry.status(self.root, self.sha)
        self.assertIsNone(row['confirmed_optimizer_updates'])
        self.assertTrue(row['optimizer_outcome_unknown'])

    def test_status_valid_committed_count(self):
        self.dump_status()
        row = entry.status(self.root, self.sha)
        self.assertEqual(row['confirmed_optimizer_updates'], 3)
        self.assertFalse(row['optimizer_outcome_unknown'])

    def test_pending_write_makes_outcome_unknown(self):
        self.dump_status()
        (self.output / 'status.pending.json').write_bytes(b'{')
        self.assertTrue(entry.status(self.root, self.sha)['optimizer_outcome_unknown'])

    def test_unreturned_attempt_is_unknown(self):
        self.dump_status(counters={'optimizer_updates': 3, 'optimizer_update_attempts': 4})
        self.assertTrue(entry.status(self.root, self.sha)['optimizer_outcome_unknown'])

    def test_corrupt_status_bytes_preserved(self):
        path = self.output / 'status.json'
        path.write_bytes(b'{incomplete')
        row = entry.status(self.root, self.sha)
        self.assertIsNone(row['confirmed_optimizer_updates'])
        self.assertEqual(row['status_receipt']['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(path.read_bytes(), b'{incomplete')

    def test_protocol_mismatch_and_boolean_counters_refused(self):
        self.dump_status(protocol_sha256='b' * 64)
        self.assertTrue(entry.status(self.root, self.sha)['optimizer_outcome_unknown'])
        self.dump_status(counters={'optimizer_updates': True, 'optimizer_update_attempts': 3})
        self.assertIsNone(entry.status(self.root, self.sha)['confirmed_optimizer_updates'])

    def test_checkpoint_binding_must_exist(self):
        self.dump_status(last_committed_checkpoint={'path': packet.OUTPUT_RELATIVE + '/checkpoints/u0003/checkpoint.json',
                                                   'bytes': 1, 'sha256': self.sha})
        self.assertIsNone(entry.status(self.root, self.sha)['confirmed_optimizer_updates'])

    def test_duplicate_nonfinite_json_refused(self):
        for parser in (packet.strict_json, install_weights.strict_json):
            for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":[1e999]}'):
                with self.assertRaises(ValueError):
                    parser(payload)

    def test_paths_and_hardlinks_refused(self):
        for value in ('../x', 'a/../x', '/x', 'a\\x', 'C:/x', './x'):
            with self.assertRaises(ValueError):
                packet.relative(value)
        self.assertEqual(str(packet.relative('assets/face.png')), 'assets/face.png')

    def test_windows_install_refuses_before_files_or_framework(self):
        with patch.object(install_weights.platform, 'system', return_value='Windows'), \
             patch.object(install_weights, '_install', side_effect=AssertionError('must not reach')):
            with self.assertRaisesRegex(ValueError, 'manually'):
                install_weights.install(self.root, self.root, self.sha)

    def test_exact_parsed_protocol_bytes_are_sha_bound(self):
        path = self.root / 'protocol.json'
        path.write_bytes(b'{"format":"bad"}')
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(packet, 'strict_json', side_effect=RuntimeError('parsed exact raw')) as parse:
            with self.assertRaisesRegex(RuntimeError, 'parsed exact raw'):
                packet.verify_static(self.root, expected)
            self.assertEqual(parse.call_args.args[0], path.read_bytes())
        with patch.object(packet, 'strict_json', side_effect=AssertionError('must not parse')):
            with self.assertRaisesRegex(ValueError, 'SHA'):
                packet.verify_static(self.root, 'b' * 64)

    def test_actual_small_gzip_export_preserves_corrupt_status_omits_weights(self):
        (self.root / 'asset.txt').write_bytes(b'face-source')
        weights = self.root / 'weights'
        weights.mkdir()
        (weights / 'do-not-return.pth').write_bytes(b'no-model-loading')
        (self.output / 'status.json').write_bytes(b'{partial')
        (self.output / 'failure.json').write_bytes(b'{also-partial')
        p = {'stage_name': packet.STAGE_NAME, 'limits': packet.LIMITS}
        with patch.object(entry.platform, 'system', return_value='Linux'), \
             patch.object(entry.Path, 'home', return_value=Path(self.temp.name)), \
             patch.object(packet, 'verify_static', return_value=p):
            entry._export(self.root, self.sha, time.monotonic())
        archive = Path(self.temp.name) / (entry.PREFIX + '-results.tar.gz')
        with tarfile.open(archive, 'r:gz') as tar:
            names = tar.getnames()
            self.assertFalse(any('/weights/' in name for name in names))
            self.assertEqual(tar.extractfile(packet.STAGE_NAME + '/asset.txt').read(), b'face-source')
            self.assertEqual(tar.extractfile(packet.STAGE_NAME + '/' + packet.OUTPUT_RELATIVE + '/status.json').read(), b'{partial')
        result = json.loads((self.root / 'export.json').read_bytes())
        self.assertTrue(result['complete'])
        self.assertTrue(result['optimizer_outcome_unknown'])
        self.assertTrue(result['failure_read_error'])
        self.assertEqual(result['archive_sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
        checksum = Path(str(archive) + '.sha256').read_bytes()
        self.assertNotIn(b'\r', checksum)

    def test_export_detects_same_size_payload_mutation(self):
        path = self.root / 'asset.txt'
        path.write_bytes(b'original')
        original_pin = packet.pin
        def mutate(root, target):
            row = original_pin(root, target)
            if target == path:
                target.write_bytes(b'changed!')
            return row
        with patch.object(entry.platform, 'system', return_value='Linux'), \
             patch.object(entry.Path, 'home', return_value=Path(self.temp.name)), \
             patch.object(packet, 'verify_static', return_value={'stage_name': packet.STAGE_NAME, 'limits': packet.LIMITS}), \
             patch.object(packet, 'pin', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during read'):
                entry._export(self.root, self.sha, time.monotonic())
        self.assertTrue((Path(self.temp.name) / (entry.PREFIX + '-results.tar.gz')).exists())
        self.assertFalse((self.root / 'export.json').exists())

    def test_valid_nonobject_failure_receipt_does_not_break_export(self):
        (self.output / 'failure.json').write_bytes(b'[]')
        with patch.object(entry.platform, 'system', return_value='Linux'), \
             patch.object(entry.Path, 'home', return_value=Path(self.temp.name)), \
             patch.object(packet, 'verify_static', return_value={'stage_name': packet.STAGE_NAME, 'limits': packet.LIMITS}):
            entry._export(self.root, self.sha, time.monotonic())
        result = json.loads((self.root / 'export.json').read_bytes())
        self.assertTrue(result['complete'])
        self.assertTrue(result['failure_present'])
        self.assertEqual(result['failure_read_error'], 'Failure receipt object required')
        self.assertFalse(result['quality_stop_recorded'])

    def test_source_imports_are_inert_and_no_automatic_launch(self):
        for name in ('packet', 'entry', 'install_weights'):
            module = importlib.import_module('research.codeformer_u96_decoder_readout_v1.' + name)
            tree = ast.parse(Path(module.__file__).read_bytes())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.split('.')[0] in ('torch', 'numpy', 'PIL') for alias in node.names))
            self.assertNotIn('torch', module.__dict__)


if __name__ == '__main__':
    unittest.main()
