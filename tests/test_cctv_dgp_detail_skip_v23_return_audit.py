"""Transfer/receipt tamper and exact zero-initializer checks for V23; no fitting."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import import_cctv_dgp_detail_skip_v23 as transfer
import audit_cctv_dgp_detail_skip_v23 as audit
from audit_cctv_dgp_detail_skip_v23_execution import check_execution


class V23ReturnChecks(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'scratch' / ('v23_return_fixture_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.actual = ROOT / 'outputs/cctv_dgp_detail_skip_v23_return'

    def make_archive(self, extra=None, bad_hash=False):
        names = ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_detail_skip_v23.py', 'scripts/run_v23.sh')
        files = {n: (audit.BUNDLE / n).read_bytes() for n in names}
        declared = {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}
        if bad_hash:
            declared['schedule.json'] = '0' * 64
        files['export_manifest.json'] = json.dumps({'complete': True, 'protocol_sha256': transfer.PIN, 'files_sha256': declared}).encode()
        archive = self.root / transfer.ARCHIVE_NAME
        with tarfile.open(archive, 'w:gz') as stream:
            for name, value in files.items():
                info = tarfile.TarInfo(transfer.PREFIX + '/' + name); info.size = len(value)
                stream.addfile(info, io.BytesIO(value))
            if extra is not None:
                info = tarfile.TarInfo(extra); info.size = 1
                stream.addfile(info, io.BytesIO(b'x'))
        digest = transfer.sha(archive)
        sidecar = Path(str(archive) + '.sha256')
        sidecar.write_text(digest + '  ' + transfer.ARCHIVE_NAME + '\n', encoding='ascii')
        receipt = self.root / 'export.json'
        transfer.write(receipt, {'complete': True, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
            'training_success_not_implied': True, 'run_results_present': False, 'failure_present': False})
        return archive, sidecar, receipt

    def execution_fixture(self, mutation=None):
        folder = self.root / 'return'; (folder / 'outputs').mkdir(parents=True)
        execution = transfer.read(self.actual / 'outputs/execution_receipt.json')
        gradient = transfer.read(self.actual / 'outputs/one_batch_gradient_preflight.json')
        timing = transfer.read(self.actual / 'outputs/timing_update20.json')
        supervisor = transfer.read(self.actual / 'supervisor_receipt.json')
        failure = transfer.read(self.actual / 'outputs/failure.json')
        if mutation:
            mutation(execution, gradient, timing, supervisor)
        for name, value in [('outputs/execution_receipt.json', execution), ('outputs/one_batch_gradient_preflight.json', gradient),
                            ('outputs/timing_update20.json', timing), ('supervisor_receipt.json', supervisor)]:
            transfer.write(folder / name, value)
        (folder / 'trainer_exit_code.txt').write_text('1\n', encoding='ascii')
        tensors = {k: None for k in gradient['per_tensor_gradient_sum_squares']}
        tensors.update({'kernel': None, 'reflect_indices': None})
        preflights = [{'recognizer_state': execution['initial_recognizer_state']}]
        return folder, preflights, {0: {}, 50: {}}, None, failure, tensors, execution['head_state']

    def test_real_packet_keeps_same_finite_exposures_and_gates(self):
        p = audit.verify_bundle(audit.BUNDLE)
        self.assertEqual(len(p['assets_sha256']), 209)
        self.assertEqual(p['design']['updates'], 800)
        self.assertEqual(p['prospective_gates']['early_minimum_degraded_feature_MSE_gain'], .01)

    def test_valid_transfer_without_training_does_not_claim_quality(self):
        archive, sidecar, receipt = self.make_archive()
        result = transfer.import_return(archive, sidecar, receipt, self.root / 'imported', self.root)
        self.assertTrue(result['complete'])
        self.assertEqual(result['neural_calls'], 0)
        self.assertFalse(result['app_promotion'])
        self.assertFalse(result['goal_complete'])

    def test_bad_download_hash_rejects_before_extraction(self):
        archive, sidecar, receipt = self.make_archive()
        sidecar.write_text('0' * 64 + '  ' + transfer.ARCHIVE_NAME + '\n', encoding='ascii')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            transfer.import_return(archive, sidecar, receipt, self.root / 'imported', self.root)
        self.assertFalse((self.root / 'imported').exists())

    def test_traversal_and_manifest_hash_tampering_rejected(self):
        archive, _, _ = self.make_archive(extra=transfer.PREFIX + '/../escaped')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
            transfer.inspect_archive(archive)
        # A second isolated fixture retains the first failed archive.
        self.root = self.root / 'second'; self.root.mkdir()
        archive, _, _ = self.make_archive(bad_hash=True)
        with self.assertRaisesRegex(ValueError, 'member hash differs'):
            transfer.inspect_archive(archive)

    def test_saved_execution_projection_counts_and_gradient_pass(self):
        result = check_execution(*self.execution_fixture())
        self.assertEqual(result['updates'], 50)
        self.assertTrue(result['update20_projection_arithmetic_verified'])
        self.assertTrue(result['neural_counts_verified'])

    def test_projection_sample_tampering_rejected(self):
        def mutation(e, g, t, s):
            t['steady_sample_seconds'][0] += .001
        with self.assertRaisesRegex(ValueError, 'samples differ'):
            check_execution(*self.execution_fixture(mutation))

    def test_projection_arithmetic_tampering_rejected(self):
        def mutation(e, g, t, s):
            t['projected_seconds'] += .1
        with self.assertRaisesRegex(ValueError, 'arithmetic differs'):
            check_execution(*self.execution_fixture(mutation))

    def test_zero_bypass_gradient_rejected(self):
        def mutation(e, g, t, s):
            g['direct_gradient_sum_squares'] = g['per_tensor_gradient_sum_squares']['direct.weight'] = 0
        with self.assertRaisesRegex(ValueError, 'direct gradient differs'):
            check_execution(*self.execution_fixture(mutation))

    def test_changed_counts_recognizer_and_budget_rejected(self):
        mutations = [
            (lambda e, g, t, s: e['neural_forward_counts'].__setitem__('DGP', 5), 'neural counts differ'),
            (lambda e, g, t, s: e.__setitem__('recognizer_state', '0' * 64), 'recognizer state changed'),
            (lambda e, g, t, s: e.__setitem__('peak_allocated_VRAM_bytes', 21 * 1024**3), 'VRAM cap'),
        ]
        for index, (mutation, message) in enumerate(mutations):
            self.root = self.root / str(index); self.root.mkdir()
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check_execution(*self.execution_fixture(mutation))

    def test_saved_initial_buffers_and_both_zero_RGB_tails_are_exact(self):
        import torch
        head = audit.make_head(audit.BUNDLE)
        saved = torch.load(self.actual / 'outputs/update0/head.pth', map_location='cpu', weights_only=True)
        reported = '98d24e4b6c9e7b8d960391c63fb859bf19d6d8979573a16938442f76c3f7885c'
        self.assertLessEqual(audit.verify_initial_state(head.state_dict(), saved, reported), 1e-8)
        changed = {k: v.clone() for k, v in saved.items()}
        changed['direct.weight'][0, 0, 0, 0] = 1e-9
        with self.assertRaisesRegex(ValueError, 'zero initial tail'):
            audit.verify_initial_state(head.state_dict(), changed, audit.state_hash(changed))
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in head.parameters()))


if __name__ == '__main__':
    unittest.main()
