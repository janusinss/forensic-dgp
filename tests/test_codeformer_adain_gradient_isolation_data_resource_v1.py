"""Four independent CPU-only data/resource fixtures under an unshared name.

The earlier four fixture bytes at the shared preparation filename were replaced
by transport fixtures. These are equivalent new checks, not a byte restoration.
"""
from copy import deepcopy
import io
import unittest

import numpy as np

from research.codeformer_adain_gradient_isolation_v1 import preparation_audit as p
from research.codeformer_adain_gradient_isolation_v1 import data_resource_review as r


class DataResourceTests(unittest.TestCase):
    def test_first50_eligibility_and_reservations_are_independently_derived(self):
        counts = r.call_census()
        self.assertEqual(counts['perceptual_training_forwards'], 148)
        self.assertEqual(sum(counts.values()), 2073)
        self.assertEqual(sum(v for k, v in counts.items() if k.endswith('_forwards')), 1614)
        self.assertEqual(counts['gradient_queries'], 9)

    def test_NPY_geometry_and_strict_witness_caps_include_headers(self):
        for shape, dtype in [((512, 512, 3), np.float32), ((256, 256, 3), np.float32), ((256,), np.int64)]:
            value = np.zeros(shape, dtype=dtype); stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            self.assertEqual(p.npy_file_bytes(shape, dtype), len(stream.getvalue()))
            self.assertLessEqual(len(stream.getvalue()), value.nbytes + 256)
        self.assertEqual(sum(r.witness_file_caps().values()), 20728832)

    def test_payload_arithmetic_counts_four_RNG_files_and_27_artifacts(self):
        parent = p.read(p.ROOT / 'outputs/codeformer_fp32_comparison_v1_preparation/final_resource_ledger.json')['ledger']
        limits = {'result_payload_bytes': 3758096384, 'archive_bytes': 3865470566,
                  'checkpoint_temporary_bytes': r.GIB, 'reserve_free_bytes': 4*r.GIB,
                  'initial_free_bytes': 16*r.GIB}
        ledger = r.derive_ledger(parent, limits)
        self.assertEqual(ledger['conservative_payload_bytes'], 3442100480)
        self.assertEqual(ledger['planned_file_count_upper'], 6787)
        self.assertGreater(ledger['payload_margin_bytes'], 0)
        altered = deepcopy(limits); altered['archive_bytes'] = 1
        with self.assertRaises(ValueError): r.derive_ledger(parent, altered)

    def test_saved_RGB8_tensor_hash_is_exact_and_quantization_change_detected(self):
        original = np.arange(3*4*4, dtype=np.uint8).reshape(4, 4, 3)
        tensor = original.astype(np.float32)/255.0
        self.assertEqual(p.pixel_sha(tensor), p.pixel_sha(original.astype(np.float32)/255.0))
        changed = tensor.copy(); changed[0, 0, 0] += np.float32(1/255)
        self.assertNotEqual(p.pixel_sha(tensor), p.pixel_sha(changed))


if __name__ == '__main__':
    unittest.main()
