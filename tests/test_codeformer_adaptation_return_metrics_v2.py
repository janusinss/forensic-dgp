"""Synthetic CPU metric fixtures; no model/weights/gradients/VM/real face pixels."""
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
import uuid

import numpy as np
from PIL import Image

from research.codeformer_adaptation_v2 import return_metrics as metrics


EVALUATION = {'mean_clear_mae_delta_stop': .01, 'worst_per_case_clear_mae_delta_stop': .03,
              'mean_blur_photometric_delta_stop': .01, 'worst_per_case_blur_photometric_delta_stop': .03}


class PixelAuditV2Tests(unittest.TestCase):
    def setUp(self):
        self.parent = (Path(__file__).resolve().parents[1] / 'scratch').resolve()
        self.parent.mkdir(exist_ok=True)
        self.root = self.parent / ('codeformer_return_metrics_v2_' + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        target = self.root.resolve()
        self.assertEqual(target.parent, self.parent)
        self.assertTrue(target.name.startswith('codeformer_return_metrics_v2_'))
        self.assertFalse(target.is_symlink())
        shutil.rmtree(target)

    def test_raw_hash_dtype_range_and_trailing_bytes_refused(self):
        path = self.root / 'raw.npy'
        for raw in (np.zeros((256, 256, 3), dtype=np.float64),
                    np.full((256, 256, 3), np.nan, dtype=np.float32),
                    np.full((256, 256, 3), 1.1, dtype=np.float32)):
            np.save(path, raw)
            with self.assertRaises(ValueError):
                metrics.load_raw(path, metrics.sha(path))
        np.save(path, np.full((256, 256, 3), .5, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            metrics.load_raw(path, '0' * 64)
        with path.open('ab') as f:
            f.write(b'not an array')
        with self.assertRaisesRegex(ValueError, 'Unexpected bytes'):
            metrics.load_raw(path, metrics.sha(path))

    def test_png_floor_and_exact_protected_padding(self):
        raw = np.full((256, 256, 3), .5, dtype=np.float32)
        observed = np.zeros((256, 256), dtype=bool); observed[30:220, 40:210] = True
        original = np.full((256, 256, 3), 197, dtype=np.uint8)
        png = metrics.expected_native_png(raw, original, observed)
        self.assertTrue(np.all(png[observed] == 127))
        self.assertTrue(np.all(png[~observed] == 197))
        self.assertFalse(np.array_equal(png, np.rint(raw * 255).astype(np.uint8)))

    def test_padding_excluded_from_metrics_and_native_target_forbidden(self):
        observed = np.zeros((256, 256), dtype=bool); observed[100:120, 100:120] = True
        target = np.full((256, 256, 3), .4, dtype=np.float32)
        prediction = target.copy(); prediction[~observed] = 1
        result = metrics.paired_metrics(prediction, target, observed, evidence='paired_synthetic')
        self.assertEqual(result, {'mae': 0., 'mse': 0., 'psnr': None})
        null = metrics.paired_metrics(prediction, None, observed, evidence='native_unpaired')
        self.assertTrue(all(v is None for v in null.values()))
        with self.assertRaisesRegex(ValueError, 'invented clean target'):
            metrics.paired_metrics(prediction, target, observed, evidence='native_unpaired')

    def test_photometry_independent_constant_and_spatial_bias(self):
        target = np.full((256, 256, 3), .25, dtype=np.float32)
        prediction = target + .125
        result = metrics.photometry(prediction, target)
        self.assertAlmostEqual(result['luminance_bias'], .125, places=12)
        self.assertAlmostEqual(result['coarse_rgb_mae'], .125, places=12)
        prediction[:128] = .375; prediction[128:] = .125
        result = metrics.photometry(prediction, target)
        self.assertAlmostEqual(result['luminance_bias'], 0., places=12)
        self.assertAlmostEqual(result['coarse_rgb_mae'], .125, places=12)

    def _blur(self):
        return {str(i): {v: {'luminance_bias': .2, 'coarse_rgb_mae': .2}
                         for v in metrics.VIEWS} for i in range(53)}

    def test_blur_guard_uses_regression_percase_png_without_mean_cancellation(self):
        baseline = self._blur(); candidate = self._blur()
        for row in candidate.values():
            for view in metrics.VIEWS:
                row[view]['luminance_bias'] = .19
        self.assertTrue(metrics.independent_blur_gate(candidate, baseline, EVALUATION)['pass'])
        candidate['17']['native_png']['coarse_rgb_mae'] = .231
        gate = metrics.independent_blur_gate(candidate, baseline, EVALUATION)
        self.assertFalse(gate['pass'])
        self.assertTrue(gate['views']['raw']['coarse_rgb_mae']['pass'])
        self.assertLess(gate['views']['native_png']['coarse_rgb_mae']['mean_delta'], .01)
        self.assertGreater(gate['views']['native_png']['coarse_rgb_mae']['worst_per_case_delta'], .03)
        self.assertEqual(metrics.check_gate(gate, copy.deepcopy(gate), EVALUATION, kind='blur')['status'], 'MATCH')

    def test_saved_gate_scalar_or_decision_mutation_refused(self):
        baseline = self._blur()
        gate = metrics.independent_blur_gate(baseline, baseline, EVALUATION)
        bad = copy.deepcopy(gate); bad['views']['raw']['luminance_bias']['mean_delta'] = .02
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            metrics.check_gate(gate, bad, EVALUATION, kind='blur')
        bad = copy.deepcopy(gate); bad['pass'] = False
        with self.assertRaisesRegex(ValueError, 'aggregate'):
            metrics.check_gate(gate, bad, EVALUATION, kind='blur')

    def test_unchanged_clear61_worst_case_png_guard(self):
        baseline = {str(i): {v: .1 for v in metrics.VIEWS} for i in range(61)}
        candidate = copy.deepcopy(baseline)
        candidate['4']['native_png'] = .131
        gate = metrics.base.independently_clear_gate(candidate, baseline, EVALUATION)
        self.assertTrue(gate['views']['raw']['pass'])
        self.assertLess(gate['views']['native_png']['mean_mae_delta'], .01)
        self.assertFalse(gate['pass'])
        self.assertEqual(metrics.check_gate(gate, copy.deepcopy(gate), EVALUATION, kind='clear')['status'], 'MATCH')

    def test_partial_snapshot_audits_orphan_pixels_reports_missing_and_never_qualifies(self):
        packet, returned, output = self.root / 'packet', self.root / 'return', self.root / 'review'
        packet.mkdir(); returned.mkdir()
        protocol = {'outputs': 'outputs/adaptation_v2', 'training': {'snapshots': [0, 50, 200, 600]},
                    'evaluation': EVALUATION}
        for root in (packet, returned):
            (root / 'protocol.json').write_text(json.dumps(protocol))
        run = returned / protocol['outputs']; run.mkdir(parents=True)
        (run / 'failure.json').write_text('{"complete":false,"completed_updates":0}')
        snapshot = run / 'snapshot_0000'; snapshot.mkdir()
        observed = np.zeros((256, 256), dtype=bool); observed[20:230, 10:245] = True
        input_pixels = np.full((256, 256, 3), 183, dtype=np.uint8)
        specs = {}
        rows = []
        for rid, kind in [('native_fixture', 'native_unpaired'), ('paired_orphan_fixture', 'paired_synthetic')]:
            folder = snapshot / rid; folder.mkdir()
            raw = np.full((256, 256, 3), .5, dtype=np.float32)
            np.save(folder / 'raw.npy', raw)
            png = metrics.expected_native_png(raw, input_pixels, observed)
            Image.fromarray(png).save(folder / 'native.png')
            ip = self.root / ('input_' + rid + '.png'); Image.fromarray(input_pixels).save(ip)
            specs[rid] = {'id': rid, 'evidence': kind, 'source': 'owned fixture', 'profile': 'blur',
                'observed': observed, 'input_pixels': input_pixels, 'input_path': str(ip),
                'input_relative': ip.name, 'input_sha256': metrics.sha(ip), 'target_path': None,
                'target_sha256': None, 'target': None if kind == 'native_unpaired' else raw.copy()}
            if kind == 'native_unpaired':
                rows.append({'id': rid, 'evidence': kind, 'source': 'owned fixture', 'profile': 'blur',
                    'observed_pixels': int(observed.sum()), 'raw_sha256': metrics.sha(folder / 'raw.npy'),
                    'native_png_sha256': metrics.sha(folder / 'native.png'),
                    'raw': {'mae': None, 'mse': None, 'psnr': None},
                    'native_png': {'mae': None, 'mse': None, 'psnr': None}})
        (snapshot / 'partial_metrics.json').write_text(json.dumps({'complete': False, 'update': 0,
            'rows': rows, 'clear': {}, 'error': 'owned resource-stop fixture'}))
        with (mock.patch.object(metrics, 'PROTOCOL_SHA', metrics.sha(packet / 'protocol.json')),
              mock.patch.object(metrics, 'case_specs', return_value=(specs, [], {}))):
            result = metrics.audit_return(returned, packet, output)
        self.assertFalse(result['model_qualified'])
        self.assertEqual(result['metric_rows'], 4)
        self.assertEqual(result['missing_planned_snapshots'], [50, 200, 600])
        summary = result['snapshot_summaries'][0]
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['partial_unrecorded_outputs'][0]['id'], 'paired_orphan_fixture')
        self.assertTrue(json.loads((output / 'visual_manifest.json').read_text())['visual_review_pending'])
        bad_png = snapshot / 'native_fixture/native.png'
        damaged = np.asarray(Image.open(bad_png)).copy(); damaged[40, 40, 0] += 1
        Image.fromarray(damaged).save(bad_png)
        rows[0]['native_png_sha256'] = metrics.sha(bad_png)
        (snapshot / 'partial_metrics.json').write_text(json.dumps({'complete': False, 'update': 0,
            'rows': rows, 'clear': {}, 'error': 'owned resource-stop fixture'}))
        # Even a replaced matching file checksum cannot hide incorrect floor
        # conversion when its pixels disagree with the retained RAW values.
        with (mock.patch.object(metrics, 'PROTOCOL_SHA', metrics.sha(packet / 'protocol.json')),
              mock.patch.object(metrics, 'case_specs', return_value=(specs, [], {}))):
            with self.assertRaisesRegex(ValueError, 'floor/protected padding differs'):
                metrics.audit_return(returned, packet, self.root / 'bad_pixel_review')
        with self.assertRaisesRegex(ValueError, 'previous pixel review'):
            with mock.patch.object(metrics, 'PROTOCOL_SHA', metrics.sha(packet / 'protocol.json')):
                metrics.audit_return(returned, packet, output)


if __name__ == '__main__':
    unittest.main()
