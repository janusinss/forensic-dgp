"""Meaningful transfer, arithmetic, stopped-run and inference-only audit checks."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import uuid
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import import_cctv_dgp_detail_prior_v22_r1 as transfer
import audit_cctv_dgp_detail_prior_v22_r1 as audit


class ReturnAuditChecks(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        # Python3.13 mkdir(mode0700) gives TemporaryDirectory an ACL inaccessible
        # to the Windows restricted token. Ordinary workspace mkdir is readable.
        self.root = (scratch / ('v22_audit_fixture_' + uuid.uuid4().hex)).resolve()
        self.assertTrue(self.root.is_relative_to(scratch.resolve()))
        self.root.mkdir()

    def tearDown(self):
        # Keep small, explicitly named synthetic fixtures as failure evidence.
        pass

    def archive(self, extra=None, omit_binding=False, corrupt_binding=False, link=False):
        names = ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_detail_prior_v22_r1.py', 'scripts/run_v22.sh')
        payloads = {n: (audit.BUNDLE / n).read_bytes() for n in names}
        manifest = {'complete': True, 'protocol_sha256': transfer.PIN,
                    'files_sha256': {n: hashlib.sha256(v).hexdigest() for n, v in payloads.items()},
                    'scope': 'Synthetic transfer test; no VM or trained result'}
        if omit_binding:
            manifest['files_sha256'].pop('schedule.json')
        if corrupt_binding:
            manifest['files_sha256']['schedule.json'] = '0' * 64
        payloads['export_manifest.json'] = json.dumps(manifest).encode()
        archive = self.root / transfer.ARCHIVE_NAME
        with tarfile.open(archive, 'w:gz') as stream:
            for name, payload in payloads.items():
                info = tarfile.TarInfo(transfer.PREFIX + '/' + name); info.size = len(payload)
                stream.addfile(info, io.BytesIO(payload))
            if extra:
                info = tarfile.TarInfo(extra)
                if link:
                    info.type = tarfile.SYMTYPE; info.linkname = '../escaped'
                    stream.addfile(info)
                else:
                    info.size = 1; stream.addfile(info, io.BytesIO(b'x'))
        digest = transfer.sha(archive)
        sidecar = Path(str(archive) + '.sha256')
        sidecar.write_text(digest + '  ' + transfer.ARCHIVE_NAME + '\n', encoding='ascii')
        receipt = self.root / 'receipt.json'
        transfer.write(receipt, {'complete': True, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
            'training_success_not_implied': True, 'run_results_present': False, 'failure_present': False})
        return archive, sidecar, receipt

    def test_original_packet_schedule_stays_exact(self):
        p = audit.verify_bundle(audit.BUNDLE)
        self.assertEqual(len(p['assets_sha256']), 196)
        self.assertEqual(len(p['cases']), 50)

    def test_imported_export_without_training_is_never_a_capacity_pass(self):
        archive, sidecar, receipt = self.archive()
        destination = self.root / 'returned'
        imported = transfer.import_return(archive, sidecar, receipt, destination, self.root)
        self.assertTrue(imported['complete'])
        self.assertFalse(imported['goal_complete'])
        result = audit.audit_return(audit.BUNDLE, destination, self.root / 'audit.json')
        self.assertTrue(result['complete'])
        self.assertFalse(result['training_receipt_verified'])
        self.assertFalse(result['necessary_capacity_pass'])
        self.assertEqual(result['counts']['head_forwards'], 0)
        self.assertEqual(result['counts']['recognizer_forwards'], 0)
        self.assertEqual(result['complete_snapshot_updates'], [])

    def test_bad_download_hash_is_rejected_before_extraction(self):
        archive, sidecar, receipt = self.archive()
        sidecar.write_text('0' * 64 + '  ' + transfer.ARCHIVE_NAME + '\n', encoding='ascii')
        destination = self.root / 'returned'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            transfer.import_return(archive, sidecar, receipt, destination, self.root)
        self.assertFalse(destination.exists())

    def test_traversal_is_rejected(self):
        archive, _, _ = self.archive(extra=transfer.PREFIX + '/../escaped')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
            transfer.inspect_archive(archive)

    def test_symlink_and_duplicate_member_are_rejected(self):
        archive, _, _ = self.archive(extra=transfer.PREFIX + '/link', link=True)
        with self.assertRaisesRegex(ValueError, 'regular files'):
            transfer.inspect_archive(archive)
        archive.unlink(); Path(str(archive) + '.sha256').unlink(); (self.root / 'receipt.json').unlink()
        archive, _, _ = self.archive(extra=transfer.PREFIX + '/protocol.json')
        with self.assertRaisesRegex(ValueError, 'Duplicate Windows'):
            transfer.inspect_archive(archive)

    def test_unbound_file_and_changed_member_are_rejected(self):
        archive, _, _ = self.archive(omit_binding=True)
        with self.assertRaisesRegex(ValueError, 'exact returned file set'):
            transfer.inspect_archive(archive)
        archive.unlink(); Path(str(archive) + '.sha256').unlink(); (self.root / 'receipt.json').unlink()
        archive, _, _ = self.archive(corrupt_binding=True)
        with self.assertRaisesRegex(ValueError, 'member hash differs'):
            transfer.inspect_archive(archive)

    def test_windows_case_and_reserved_path_rules(self):
        for name in ('../outside', '/absolute', 'C:/absolute', 'a\\b', 'NUL.txt', 'a/COM1.npy', 'a/trailing.', 'a//b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                transfer.relative_name(name)

    def test_independent_historical_metrics_and_feature_kernel(self):
        import cv2
        import numpy as np
        from scipy.ndimage import convolve1d
        from skimage.metrics import structural_similarity
        yy, xx = np.indices((256, 256))
        target = np.stack((xx, yy, (xx + yy) // 2), axis=-1).astype(np.uint8)
        actual = target.copy(); actual[70:170, 90:160, 1] = np.clip(actual[70:170, 90:160, 1].astype(int) + 12, 0, 255)
        mask = np.zeros((256, 256), bool); mask[17:239, 29:221] = True
        got = audit.pixel_metrics(actual, target, mask)
        a, b = actual.astype(np.float32) / 255, target.astype(np.float32) / 255
        _, smap = structural_similarity(b, a, win_size=7, channel_axis=-1, data_range=1, full=True)
        expected_interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.assertTrue(np.array_equal(audit.interior(mask, 3), expected_interior))
        self.assertEqual(got['MSE'], float(np.square((a - b)[mask]).astype(np.float64).mean()))
        self.assertAlmostEqual(got['SSIM'], float(smap[expected_interior].astype(np.float64).mean()), places=12)
        feature = mask & audit.interior(mask, 6)
        kernel = np.exp(-.5 * (np.arange(-6, 7, dtype=np.float64) / 2)**2); kernel /= kernel.sum()
        luma = lambda value: (value.astype(np.float64) / 255 * [.299, .587, .114]).sum(2)
        hp = lambda value: value - convolve1d(convolve1d(value, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
        expected = float(np.square(hp(luma(actual)) - hp(luma(target)))[feature].mean())
        self.assertAlmostEqual(audit.detail_metric(actual, target, feature), expected, places=15)

    def test_unique_clear_counts_and_preservation_brightness_gates(self):
        p = transfer.read(audit.BUNDLE / 'protocol.json')
        metric = {k: v for k, v in zip(audit.GROUP_METRICS, (.05, .7, .4, .002, .05))}
        rows = [{'source': c['source'], 'profile': c['profile'], 'metrics': metric} for c in p['cases']]
        baseline = audit.aggregate(rows)
        self.assertEqual(len(baseline), 17)
        for source in {c['source'] for c in p['cases']}:
            self.assertEqual(baseline[source + '/clear']['cases'], 5)
        candidate = copy.deepcopy(baseline)
        for values in candidate.values():
            values['MSE'] = .045; values['landmark_high_frequency_MSE'] = .0016
        self.assertTrue(audit.capacity(baseline, candidate)['necessary_capacity_pass'])
        source = sorted({c['source'] for c in p['cases']})[0]
        candidate[source + '/clear']['MSE'] = baseline[source + '/clear']['MSE'] + 2e-12
        self.assertFalse(audit.capacity(baseline, candidate)['necessary_capacity_pass'])
        candidate[source + '/clear']['MSE'] = .045
        candidate['degraded']['constant_mean_shift_only_MSE'] = .03
        self.assertFalse(audit.capacity(baseline, candidate)['necessary_capacity_pass'])

    def test_initial_cpu_head_is_pinned_and_no_gradient(self):
        import numpy as np
        import torch
        head = audit.make_head(audit.BUNDLE)
        vm_initial = ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/update0/head.pth'
        saved = torch.load(vm_initial, map_location='cpu', weights_only=True)
        self.assertLessEqual(audit.verify_initial_state(head.state_dict(), saved,
            '2a733cade013bc2f06bf4b44e5a0d708695184a526df5feab212307d6699f326'), 1e-8)
        changed = {k: v.clone() for k, v in saved.items()}
        changed['tail.bias'][0] = 1e-7
        with self.assertRaisesRegex(ValueError, 'zero initial tail'):
            audit.verify_initial_state(head.state_dict(), changed, audit.state_hash(changed))
        p = transfer.read(audit.BUNDLE / 'protocol.json')
        selected = [next(c for c in p['cases'] if c['id'] == cid) for cid in
                    ('v9_tr_asian_00048_clear', 'v9_tr_ffhq_00084_clear')]
        before = audit.state_hash(head.state_dict())
        for c in selected:
            camera, base = audit.rgb(audit.BUNDLE / c['input']), audit.raw_rgb(audit.BUNDLE / c['raw_dgp'])
            mask = audit.observed(audit.BUNDLE / c['observed'])
            x = torch.from_numpy(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
            b = torch.from_numpy(base).permute(2, 0, 1)[None]
            m = torch.from_numpy(mask.astype(np.float32))[None, None]
            with torch.inference_mode():
                result = head(x, b, m)[0].permute(1, 2, 0).numpy().copy()
            self.assertTrue(np.array_equal(result, base))
        self.assertEqual(audit.state_hash(head.state_dict()), before)
        self.assertFalse(any(v.requires_grad or v.grad is not None for v in head.parameters()))


if __name__ == '__main__':
    unittest.main()
