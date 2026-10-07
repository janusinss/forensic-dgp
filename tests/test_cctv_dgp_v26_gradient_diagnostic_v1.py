"""Source/transfer tamper guards; arithmetic fixtures, no model or derivatives."""
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
import verify_cctv_dgp_v26_gradient_diagnostic_v1 as verify
import import_cctv_dgp_v26_gradient_diagnostic_v1 as transfer
import audit_cctv_dgp_v26_gradient_diagnostic_v1_return as audit


class Guards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (verify.BUNDLE / 'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py').read_text(encoding='utf-8')
        cls.folder = ROOT / 'scratch' / ('v26_gradient_contract_' + uuid.uuid4().hex)
        cls.folder.mkdir(parents=True)

    def test_fixed_source(self):
        verify.source_contract(self.source)

    def test_full_original_return_audit_preserved(self):
        verify.return_audit_contract()

    def test_no_epsilon_in_initial_gradient_proof(self):
        changed = self.source.replace('torch.count_nonzero(g)==0', 'torch.abs(g).max()<1e-30')
        self.assertNotEqual(changed, self.source)
        with self.assertRaises(AssertionError):
            verify.source_contract(changed)

    def test_corrected_objective_is_required(self):
        changed = self.source.replace("['batchmatched_scores','objective_terms']", "['objective_terms']")
        with self.assertRaises(AssertionError):
            verify.source_contract(changed)

    def test_optimizer_is_forbidden(self):
        changed = self.source.replace("        named=list(head.named_parameters())", "        optimizer=torch.optim.AdamW(head.parameters())\n        named=list(head.named_parameters())")
        with self.assertRaises(AssertionError):
            verify.source_contract(changed)

    def test_no_additional_saved_state(self):
        changed = self.source.replace('for update in [0,50]', 'for update in [0,50,800]')
        with self.assertRaises(AssertionError):
            verify.source_contract(changed)

    def fixture(self, extra=None, altered=False, unsafe=False):
        files = {name: (verify.BUNDLE / name).read_bytes() for name in [
            'protocol.json', 'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py', 'scripts/run_gradient.sh']}
        if altered:
            files['scripts/run_gradient.sh'] += b'\n# modified fixture\n'
        if extra:
            files[extra] = b'explicit synthetic fixture, not a trained checkpoint'
        manifest = {'complete': True, 'protocol_sha256': transfer.PIN,
                    'files_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
        files['export_manifest.json'] = json.dumps(manifest).encode('utf-8')
        path = self.folder / (uuid.uuid4().hex + '.tar.gz')
        with tarfile.open(path, 'x:gz') as tar:
            for name, value in files.items():
                member = tarfile.TarInfo(transfer.PREFIX + '/' + name)
                member.size = len(value)
                tar.addfile(member, io.BytesIO(value))
            if unsafe:
                member = tarfile.TarInfo(transfer.PREFIX + '/../escape.txt')
                member.size = 1
                tar.addfile(member, io.BytesIO(b'x'))
        return path

    def test_safe_partial_source_fixture(self):
        _, files, _ = transfer.inspect_archive(self.fixture())
        self.assertEqual(len(files), 4)

    def test_no_returned_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected diagnostic'):
            transfer.inspect_archive(self.fixture(extra='outputs/head.pth'))

    def test_frozen_source_required_even_with_matching_manifest(self):
        with self.assertRaisesRegex(ValueError, 'Frozen diagnostic source'):
            transfer.inspect_archive(self.fixture(altered=True))

    def test_no_path_traversal(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
            transfer.inspect_archive(self.fixture(unsafe=True))

    def test_matrix_nan_and_wrong_layout_rejected(self):
        import numpy as np
        matrix = np.zeros((7, 53781), np.float64)
        matrix[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Finite float64 gradient'):
            audit.matrix_statistics(matrix, verify.expected_layout())
        layout = copy.deepcopy(verify.expected_layout())
        layout[-1]['name'] = 'changed.bias'
        with self.assertRaisesRegex(ValueError, 'Original26'):
            audit.matrix_statistics(np.zeros((7, 53781), np.float64), layout)


if __name__ == '__main__':
    unittest.main()
