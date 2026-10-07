"""Source/archive/receipt regressions. Synthetic fixtures are not pilot results."""
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_cctv_dgp_feature_skips_v27 import NEW, OLD, PIN, head_contract, worker_contract
from cctv_dgp_feature_skips_v27_return_rules import protocol, source_hashes, allowed_return_name, check_identity_preflights, check_installation, REQUIRED_SOURCES
from audit_cctv_dgp_feature_skips_v27_execution import check_execution
from import_cctv_dgp_feature_skips_v27 import inspect_archive

FIXTURES = ROOT / 'scratch' / ('dgp_v27_contract_' + uuid.uuid4().hex)
FIXTURES.mkdir(parents=True)
SAVED = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def new_fixture(name):
    path = FIXTURES / name
    path.mkdir()
    return path


def identity_fixture(name):
    root = new_fixture(name)
    proof = read(next(SAVED.glob('batchmatched_identity_preflight_*.json')))
    for row in proof['rows']:
        row['all36_matched_gradient_tensors_exactly_zero'] = row.pop('all26_matched_gradient_tensors_exactly_zero')
    preflight = read(next(SAVED.glob('preflight_*.json')))
    preflight.update(protocol_sha256=PIN, trainable_parameters=55524,
        shared_V26_initial_tensors_exact=28, new_feature_skip_tensors_exact_zero=10,
        batchmatched_identity_proof=proof)
    write(root / 'batchmatched_identity_preflight_1.json', proof)
    write(root / 'preflight_1.json', preflight)
    return root


def execution_fixture(name):
    root = new_fixture(name)
    for file in ['supervisor_receipt.json', 'trainer_exit_code.txt', 'outputs/execution_receipt.json',
                 'outputs/one_batch_gradient_preflight.json', 'outputs/feature_path_gradient_update2.json',
                 'outputs/timing_update20.json', 'outputs/failure.json']:
        target = root / file; target.parent.mkdir(parents=True, exist_ok=True)
        if file.endswith('.json'):
            value = read(SAVED / file)
            if 'protocol_sha256' in value:
                value['protocol_sha256'] = PIN
            if file.endswith('one_batch_gradient_preflight.json'):
                value['per_feature_skip_weight_gradient_sum_squares'] = {str(index): .0001 for index in range(5)}
                for index in range(5):
                    value['per_tensor_gradient_sum_squares']['feature_skips.' + str(index) + '.weight'] = .0001
                    value['per_tensor_gradient_sum_squares']['feature_skips.' + str(index) + '.bias'] = 0.
            write(target, value)
        else:
            shutil.copy2(SAVED / file, target)
    preflight = read(next(SAVED.glob('preflight_*.json')))
    tensors = {name: None for name in read(root / 'outputs/one_batch_gradient_preflight.json')['per_tensor_gradient_sum_squares']}
    tensors.update(kernel=None, reflect_indices=None)
    failure = read(root / 'outputs/failure.json')
    state = read(root / 'outputs/execution_receipt.json')['head_state']
    return root, [preflight], {0: {}, 50: {}}, None, failure, tensors, state


class V27Contracts(unittest.TestCase):
    def test_declared_head_change_only(self):
        self.assertTrue(head_contract((NEW / 'cctv_dgp_feature_skips_v27.py').read_text()))

    def test_target_conditioning_is_rejected(self):
        text = (NEW / 'cctv_dgp_feature_skips_v27.py').read_text().replace('support = F.interpolate(mask,', 'support = F.interpolate(target,')
        with self.assertRaises(AssertionError):
            head_contract(text)

    def test_gate_relaxation_is_rejected(self):
        text = (NEW / 'scripts/cctv_dgp_feature_skips_v27_vm.py').read_text().replace("assert gain>=.01", "assert gain>=.001")
        with self.assertRaises(AssertionError):
            worker_contract(text)

    def test_skip_gradient_guard_removal_is_rejected(self):
        text = (NEW / 'scripts/cctv_dgp_feature_skips_v27_vm.py').read_text().replace('value > 0 for value in skip_gradients.values()', 'value >= 0 for value in skip_gradients.values()')
        with self.assertRaises(AssertionError):
            worker_contract(text)

    def test_waived_identity_source_is_rejected(self):
        root = new_fixture('source_identity_waiver')
        for name in REQUIRED_SOURCES:
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(NEW / name, path)
        path = root / 'cctv_dgp_feature_skips_v27_preflight.py'
        text = path.read_text().replace('torch.count_nonzero(v)==0', 'torch.count_nonzero(v)<=1')
        path.write_text(text, encoding='utf-8')
        with self.assertRaises(ValueError):
            source_hashes(root)

    def test_exact_identity_receipt_validity(self):
        root = identity_fixture('identity_valid')
        result = check_identity_preflights(root, protocol())
        self.assertTrue(result['all36_exact_zero_gradient_assertions_source_bound'])

    def test_identity_epsilon_is_rejected(self):
        root = identity_fixture('identity_epsilon')
        proof = read(root / 'batchmatched_identity_preflight_1.json')
        proof['rows'][0]['batchmatched_component_gradient_norm'] = 1e-15
        write(root / 'batchmatched_identity_preflight_1.json', proof)
        with self.assertRaises(ValueError):
            check_identity_preflights(root, protocol())

    def test_shared_original_initializer_proof_required(self):
        root = identity_fixture('identity_missing_shared_state')
        receipt = read(root / 'preflight_1.json'); receipt.pop('shared_V26_initial_tensors_exact')
        write(root / 'preflight_1.json', receipt)
        with self.assertRaises(ValueError):
            check_identity_preflights(root, protocol())

    def test_five_skip_gradient_receipt_validity(self):
        result = check_execution(*execution_fixture('execution_valid'))
        self.assertTrue(result['five_direct_feature_skip_preoptimizer_gradients_verified'])

    def test_zero_skip_gradient_is_rejected(self):
        values = execution_fixture('execution_zero_skip')
        path = values[0] / 'outputs/one_batch_gradient_preflight.json'; gradient = read(path)
        gradient['per_feature_skip_weight_gradient_sum_squares']['2'] = 0
        gradient['per_tensor_gradient_sum_squares']['feature_skips.2.weight'] = 0
        write(path, gradient)
        with self.assertRaises(ValueError):
            check_execution(*values)

    def test_hardlink_receipt_cannot_claim_data_copy(self):
        root = new_fixture('installation_copy_claim'); p = protocol()
        receipt = {'complete': True, 'protocol_sha256': PIN, 'inherited_assets_verified_and_hardlinked': 240,
            'assets_verified': 246, 'inherited_data_bytes_copied': 1, 'old_V26_protocol_sha256': p['closed_V26_protocol_sha256'],
            'old_V26_failure_preserved': True, 'old_V26_initial_head_preserved': True,
            'cap_seconds': 60, 'model_or_gradient_calls': 0, 'optimizer_updates': 0,
            'original_files_changed': False, 'copy_fallback_permitted': False, 'quality_acceptance_not_implied': True,
            'seconds': .5, 'inherited_logical_bytes': sum((NEW / name).stat().st_size for name in p['inherited_assets'])}
        write(root / 'installation_receipt.json', receipt)
        with self.assertRaises(ValueError):
            check_installation(root, p)

    def test_optimizer_and_reserved_file_roles_are_rejected(self):
        p = protocol()
        self.assertFalse(allowed_return_name('outputs/update800/optimizer.pth', p))
        self.assertFalse(allowed_return_name('outputs/update50/reserved_native_test.png', p))

    def test_archive_traversal_is_rejected(self):
        path = FIXTURES / 'traversal.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            member = tarfile.TarInfo('cctv_dgp_feature_skips_v27_return/../outside.py'); member.size = 1
            tar.addfile(member, io.BytesIO(b'x'))
        with self.assertRaises(ValueError):
            inspect_archive(path)

    def test_archive_link_is_rejected(self):
        path = FIXTURES / 'link.tar.gz'
        with tarfile.open(path, 'w:gz') as tar:
            member = tarfile.TarInfo('cctv_dgp_feature_skips_v27_return/protocol.json')
            member.type = tarfile.SYMTYPE; member.linkname = '../outside.json'
            tar.addfile(member)
        with self.assertRaises(ValueError):
            inspect_archive(path)


if __name__ == '__main__':
    unittest.main()
