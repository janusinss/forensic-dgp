"""Prospective saved-array corruption fixtures; no network or derivative calls."""
from contextlib import contextmanager
import json
from pathlib import Path
import unittest
import uuid

import numpy as np

from audit_cctv_dgp_decoder_activation_v1_return import (
    EXPECTED_COUNTS, NORM_METADATA_BOUND, partial_evidence, verify_counts,
    verify_gradient_matrix,
)
from cctv_dgp_decoder_activation_v1_contract import array_sha


@contextmanager
def fixture_directory():
    # Ordinary inherited workspace ACLs; Windows TemporaryDirectory's private
    # mode can produce inaccessible children in this harness. Retain small JSON
    # fixtures rather than deleting partially inspected failure evidence.
    base = Path(__file__).resolve().parents[1]/'outputs/cctv_dgp_decoder_activation_v1_preparation/audit_fixtures'
    base.mkdir(parents=True,exist_ok=True)
    root = base/('case_'+uuid.uuid4().hex)
    root.mkdir()
    yield root


class SavedGradientFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One bounded shared 81.34 MiB matrix, with distinct sparse scalar and
        # full-objective gradients. These arrays are not a learned model/state.
        cls.matrix = np.zeros((35,609219),np.float32)
        cls.matrix[0,5] = .5
        cls.matrix[-1,3:5] = [3.,4.]
        cls.matrix.flags.writeable = False
        cls.direction = np.zeros(609219,np.float64)
        cls.direction[3:6] = [.001,-.0005,.0001]
        cls.labels = ['scalar_'+str(i) for i in range(35)]
        cls.slopes = np.array([row.astype(np.float64)@cls.direction for row in cls.matrix])
        cls.factor = np.float32(1.)/np.float32(np.float32(5.)+np.float32(1e-6))
        cls.matrix_sha = array_sha(cls.matrix)

    def fixture(self, ULPs=0):
        factor = self.factor
        for _ in range(ULPs):
            factor = np.nextafter(factor,np.float32(np.inf))
        clipped = (self.matrix[-1]*factor).astype(np.float32)
        record = dict(scalar_labels=self.labels,gradients_shape=[35,609219],gradients_dtype='float32',
            gradients_sha256=self.matrix_sha,clipped_gradient_sha256=array_sha(clipped),
            retained_direction_sha256=array_sha(self.direction),directional_slopes=self.slopes.tolist(),
            full_gradient_norm_before_clip=5.,clip_factor=float(factor))
        return clipped,record

    def check(self, matrix=None, clipped=None, record=None, direction=None):
        if clipped is None or record is None:
            clipped,record = self.fixture()
        return verify_gradient_matrix(self.matrix if matrix is None else matrix,clipped,
            self.direction if direction is None else direction,record,self.labels,NORM_METADATA_BOUND)

    def test_exact_to_two_ULP_factor_metadata_accepts_exact_clipped_bytes(self):
        for ULPs in [0,1,2]:
            with self.subTest(ULPs=ULPs):
                clipped,record = self.fixture(ULPs)
                slopes,proof = self.check(clipped=clipped,record=record)
                np.testing.assert_array_equal(slopes,self.slopes)
                self.assertEqual(proof['clipping_factor_metadata_ULPs'],ULPs)
                self.assertTrue(proof['clipped_float32_bytes_exact'])

    def test_three_ULP_factor_metadata_rejected_even_when_clipped_bytes_match(self):
        clipped,record = self.fixture(3)
        with self.assertRaisesRegex(AssertionError,'ULP bound'):
            self.check(clipped=clipped,record=record)

    def test_modified_clipped_pixel_rejected_despite_updated_array_hash(self):
        clipped,record = self.fixture(2)
        clipped[3] = np.nextafter(clipped[3],np.float32(np.inf))
        record['clipped_gradient_sha256'] = array_sha(clipped)
        with self.assertRaises(AssertionError):
            self.check(clipped=clipped,record=record)

    def test_nonfinite_matrix_rejected(self):
        matrix = self.matrix.copy();matrix[2,3] = np.nan
        clipped,record = self.fixture();record['gradients_sha256'] = array_sha(matrix)
        with self.assertRaises(AssertionError):
            self.check(matrix=matrix,clipped=clipped,record=record)

    def test_matrix_shape_and_dtype_corruptions_rejected(self):
        for matrix in [self.matrix[:-1],self.matrix[:,:-1],self.matrix.astype(np.float64)]:
            with self.subTest(shape=matrix.shape,dtype=matrix.dtype), self.assertRaises(AssertionError):
                self.check(matrix=matrix)

    def test_dot_product_metadata_corruption_rejected(self):
        clipped,record = self.fixture();record['directional_slopes'][0] += .001
        with self.assertRaises(AssertionError):
            self.check(clipped=clipped,record=record)

    def test_gradient_direction_hash_and_layout_corruptions_rejected(self):
        for field,value in [('gradients_sha256','0'*64),('retained_direction_sha256','f'*64),
                            ('scalar_labels',list(reversed(self.labels)))]:
            clipped,record = self.fixture();record[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.check(clipped=clipped,record=record)

    def test_norm_reduction_metadata_bound_is_frozen(self):
        clipped,record = self.fixture();record['full_gradient_norm_before_clip'] = 5.00002
        with self.assertRaises(AssertionError):
            self.check(clipped=clipped,record=record)
        clipped,record = self.fixture()
        with self.assertRaises(AssertionError):
            verify_gradient_matrix(self.matrix,clipped,self.direction,record,self.labels,dict(rtol=.01,atol=1e-12))


class CountsAndPartialFixtures(unittest.TestCase):
    def test_completed_counts_are_exact_and_no_new_updates_allowed(self):
        verify_counts(dict(EXPECTED_COUNTS))
        for field,value in [('optimizer_updates',1),('backwards',1),('gradient_queries',71),
                            ('recognizer_batch_calls',81),('original_batch_calls',True)]:
            counts = dict(EXPECTED_COUNTS);counts[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_counts(counts)

    def test_progress_alone_does_not_establish_a_terminal_worker(self):
        with fixture_directory() as root:
            out = root/'outputs';out.mkdir()
            (root/'protocol.json').write_text('{}\n')
            (out/'progress.jsonl').write_text(json.dumps(dict(counts=EXPECTED_COUNTS))+'\n')
            with self.assertRaisesRegex(AssertionError,'stopped-worker'):
                partial_evidence(root,out,{},root/'receipt.json')

    def test_SIGKILL_missing_failure_and_malformed_tail_keep_counts_unknown(self):
        with fixture_directory() as root:
            out = root/'outputs';out.mkdir()
            (root/'protocol.json').write_text('{}\n')
            (root/'run_exit.json').write_text(json.dumps(dict(flag='--run',exit_code=-9,timeout=True,kill_sent=True)))
            counts = dict(EXPECTED_COUNTS,gradient_queries=36)
            (out/'progress.jsonl').write_text(json.dumps(dict(counts=counts))+'\n{"counts":')
            (out/'run_results.json').write_text('{"complete":')
            receipt = root/'receipt.json';partial_evidence(root,out,{},receipt)
            result = json.loads(receipt.read_text())
            self.assertTrue(result['complete'] and result['partial_failure_evidence_only'])
            self.assertIsNone(result['run_gradient_queries'])
            self.assertFalse(result['exact_run_counts_known'] or result['model_qualified'])
            self.assertEqual(result['last_known_progress']['counts']['gradient_queries'],36)
            self.assertEqual(result['malformed_progress_lines_retained'],1)
            self.assertIn('outputs/run_results.json',result['saved_files'])

    def test_zero_exit_missing_result_cannot_be_hidden_as_partial_success(self):
        with fixture_directory() as root:
            out = root/'outputs';out.mkdir()
            (root/'protocol.json').write_text('{}\n')
            (root/'run_exit.json').write_text(json.dumps(dict(flag='--run',exit_code=0)))
            (out/'failure.json').write_text(json.dumps(dict(complete=False)))
            with self.assertRaisesRegex(AssertionError,'Successful worker'):
                partial_evidence(root,out,{},root/'receipt.json')


if __name__ == '__main__':
    unittest.main()
