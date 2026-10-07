"""Tamper regressions using artificial V26 receipts; no model or derivative calls."""
import ast
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
import import_cctv_dgp_batchmatched_identity_v26 as incoming
import audit_cctv_dgp_batchmatched_identity_v26 as audit
from audit_cctv_dgp_batchmatched_identity_v26_execution import check_execution
from cctv_dgp_batchmatched_identity_v26_return_rules import (
    BUNDLE, PIN, REQUIRED_SOURCES, REQUIRED_RECEIPTS, check_identity_preflights, check_installation,
)
from import_cctv_dgp_spatial_features_v25 import read, write

OLD = ROOT / 'outputs/cctv_dgp_spatial_features_v25_return'


class ReturnEvidence(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'scratch' / ('v26_artificial_return_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.p = read(BUNDLE / 'protocol.json')

    def identity_fixture(self):
        rows = []
        for begin in range(0, 50, 5):
            rows.append({'ids': [case['id'] for case in self.p['cases'][begin:begin + 5]],
                         'exact_cached_baseline_cases': 5, 'legacy_component_value': 1e-8,
                         'legacy_component_gradient_norm': .001,
                         'batchmatched_component_value': 0., 'batchmatched_component_gradient_norm': 0.,
                         'exact_reference_prediction_cosines': True,
                         'all26_matched_gradient_tensors_exactly_zero': True})
        initial = read(next(OLD.glob('preflight_*.json')))
        proof = {'complete': True, 'cases': 50, 'batches': 10, 'gradient_calls': 20,
                 'head_forwards': 10, 'recognizer_forwards': 20, 'optimizer_constructed': False,
                 'optimizer_updates': 0, 'legacy_component_value': sum(row['legacy_component_value'] for row in rows),
                 'legacy_component_gradient_norm': .005, 'batchmatched_component_value': 0.,
                 'batchmatched_component_gradient_norm': 0., 'baseline_and_prediction_identical_cases': 50,
                 'head_state_before_after': initial['initial_head_state'],
                 'recognizer_state_before_after': initial['recognizer_state'], 'head_grad_buffers_empty': True,
                 'frozen_features_and_recognizer_have_no_gradients': True, 'rows': rows,
                 'same_original_penalty_weight': 5, 'identity_margin': 0, 'quality_gate_relaxed': False, 'seconds': 10.}
        main = copy.deepcopy(initial)
        main['protocol_sha256'] = PIN
        main['seconds'] = 30.
        main['batchmatched_identity_proof'] = copy.deepcopy(proof)
        main['neural_forward_counts'] = {'detail_head': 60, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 120}
        return proof, main

    def installation_fixture(self):
        return {'complete': True, 'protocol_sha256': PIN, 'inherited_assets_verified_and_copied': 235,
                'assets_verified': 240, 'inherited_copy_bytes': sum((BUNDLE / name).stat().st_size for name in self.p['inherited_assets']),
                'old_V25_protocol_sha256': self.p['closed_V25_protocol_sha256'], 'old_V25_failure_preserved': True,
                'seconds': 2., 'cap_seconds': 60, 'model_or_gradient_calls': 0, 'optimizer_updates': 0,
                'original_files_changed': False, 'quality_acceptance_not_implied': True}

    def save_proof(self, change=None, include_main=True):
        folder = self.root / 'identity'; folder.mkdir()
        proof, main = self.identity_fixture()
        if change:
            change(proof, main)
        write(folder / 'batchmatched_identity_preflight_123.json', proof)
        if include_main:
            write(folder / 'preflight_456.json', main)
        return folder

    def execution_fixture(self, change=None):
        folder = self.root / 'execution'; (folder / 'outputs').mkdir(parents=True)
        e = read(OLD / 'outputs/execution_receipt.json')
        g = read(OLD / 'outputs/one_batch_gradient_preflight.json')
        t = read(OLD / 'outputs/timing_update20.json')
        s = read(OLD / 'supervisor_receipt.json')
        q = read(OLD / 'outputs/feature_path_gradient_update2.json')
        failure = read(OLD / 'outputs/failure.json')
        for value in [e, s, failure]:
            value['protocol_sha256'] = PIN
        e['neural_forward_counts']['detail_head'] += 10
        e['neural_forward_counts']['fixed_recognizer'] += 20
        g['neural_forward_counts']['detail_head'] += 10
        g['neural_forward_counts']['fixed_recognizer'] += 20
        if change:
            change(e, g, t, s, q)
        for name, value in [('outputs/execution_receipt.json', e), ('outputs/one_batch_gradient_preflight.json', g),
                            ('outputs/timing_update20.json', t), ('supervisor_receipt.json', s),
                            ('outputs/feature_path_gradient_update2.json', q)]:
            write(folder / name, value)
        (folder / 'trainer_exit_code.txt').write_text('1\n', encoding='ascii')
        proof, main = self.identity_fixture()
        tensors = {**{key: None for key in g['per_tensor_gradient_sum_squares']}, 'kernel': None, 'reflect_indices': None}
        return folder, [main], {0: {}, 50: {}}, None, failure, tensors, e['head_state']

    def archive_fixture(self, *, changed_source=None, extra=None, duplicate=None, omit=None):
        files = {name: (BUNDLE / name).read_bytes() for name in REQUIRED_SOURCES}
        files['installation_receipt.json'] = json.dumps(self.installation_fixture()).encode()
        supervisor = read(OLD / 'supervisor_receipt.json'); supervisor['protocol_sha256'] = PIN
        files['supervisor_receipt.json'] = json.dumps(supervisor).encode()
        files['trainer_exit_code.txt'] = b'1\n'
        files['trainer.log'] = b'Artificial partial-stop fixture, not actual V26 VM evidence.\n'
        if changed_source:
            files[changed_source] += b'\n# changed source\n'
        if omit:
            files.pop(omit)
        if extra:
            files[extra] = b'Unexpected or unsafe file\n'
        declared = {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}
        manifest = {'complete': True, 'protocol_sha256': PIN, 'files_sha256': declared,
                    'scope': 'Training capacity results only; failed gates retained; independent local audit required'}
        files['export_manifest.json'] = json.dumps(manifest).encode()
        path = self.root / incoming.ARCHIVE_NAME
        with tarfile.open(path, 'x:gz') as archive:
            for name, payload in files.items():
                info = tarfile.TarInfo(incoming.PREFIX + '/' + name); info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            if duplicate:
                payload = files[duplicate]
                info = tarfile.TarInfo(incoming.PREFIX + '/' + duplicate); info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return path

    def test_valid_artificial_receipts_and_sources_import_without_training_claim(self):
        archive = self.archive_fixture()
        digest = incoming.sha(archive)
        sidecar = self.root / (incoming.ARCHIVE_NAME + '.sha256')
        sidecar.write_text(digest + '  ' + incoming.ARCHIVE_NAME + '\n', encoding='ascii')
        receipt = self.root / 'export.json'
        write(receipt, {'complete': True, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
                        'seconds': 1., 'training_success_not_implied': True,
                        'run_results_present': False, 'failure_present': False})
        result = incoming.import_return(archive, sidecar, receipt, self.root / 'returned', allowed_root=self.root)
        self.assertEqual(result['members'], 15)
        self.assertEqual(result['neural_calls'], 0)
        self.assertFalse(result['app_promotion'])
        self.assertFalse(result['goal_complete'])

    def test_changed_helper_with_valid_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'changed returned frozen source'):
            incoming.inspect_archive(self.archive_fixture(changed_source='cctv_dgp_batchmatched_identity_v26.py'))

    def test_undeclared_optimizer_state_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected returned file role'):
            incoming.inspect_archive(self.archive_fixture(extra='outputs/optimizer.pth'))

    def test_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
            incoming.inspect_archive(self.archive_fixture(extra='../outside.json'))

    def test_duplicate_windows_member_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate Windows archive member'):
            incoming.inspect_archive(self.archive_fixture(duplicate='protocol.json'))

    def test_installer_receipt_cannot_be_omitted(self):
        with self.assertRaisesRegex(ValueError, 'execution/install receipt'):
            incoming.inspect_archive(self.archive_fixture(omit='installation_receipt.json'))

    def test_original_asset_mutation_receipt_is_rejected(self):
        folder = self.root / 'install'; folder.mkdir()
        receipt = self.installation_fixture()
        receipt['original_files_changed'] = True
        write(folder / 'installation_receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'Installation policy'):
            check_installation(folder, self.p)

    def test_installation_copy_byte_count_is_verified(self):
        folder = self.root / 'copy_bytes'; folder.mkdir()
        receipt = self.installation_fixture(); receipt['inherited_copy_bytes'] += 1
        write(folder / 'installation_receipt.json', receipt)
        with self.assertRaisesRegex(ValueError, 'copied byte count'):
            check_installation(folder, self.p)

    def test_valid_exact_zero_proof_is_source_bound_without_derivative_replay(self):
        result = check_identity_preflights(self.save_proof(), self.p)
        self.assertEqual(result['VM_gradient_queries_checked'], 20)
        self.assertEqual(result['batchmatched_component_gradient_norm'], 0)
        self.assertTrue(result['all26_exact_zero_gradient_assertions_source_bound'])
        self.assertIn('does not rerun derivatives', result['limit'])

    def test_tiny_nonzero_matched_gradient_cannot_use_an_epsilon(self):
        def change(proof, main):
            proof['rows'][0]['batchmatched_component_gradient_norm'] = 1e-30
        with self.assertRaisesRegex(ValueError, 'exactly zero'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_tiny_nonzero_matched_penalty_is_rejected(self):
        def change(proof, main): proof['batchmatched_component_value'] = 1e-30
        with self.assertRaisesRegex(ValueError, 'exactly zero'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_changed_batch_order_is_rejected(self):
        def change(proof, main): proof['rows'][0]['ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'case order'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_all26_zero_gradient_assertion_cannot_be_false(self):
        def change(proof, main): proof['rows'][0]['all26_matched_gradient_tensors_exactly_zero'] = False
        with self.assertRaisesRegex(ValueError, 'Identity batch policy'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_missing_legacy_discrepancy_is_rejected(self):
        def change(proof, main): proof['legacy_component_gradient_norm'] = 0.
        with self.assertRaisesRegex(ValueError, 'positive gradient'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_changed_identity_margin_is_rejected(self):
        def change(proof, main): proof['identity_margin'] = 1e-6
        with self.assertRaisesRegex(ValueError, 'Identity proof policy'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_identity_weight_cannot_be_relaxed(self):
        def change(proof, main): proof['same_original_penalty_weight'] = 4
        with self.assertRaisesRegex(ValueError, 'Identity proof policy'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_additional_gradient_queries_cannot_enter_training_backward_count(self):
        def change(proof, main): proof['gradient_calls'] = 21
        with self.assertRaisesRegex(ValueError, 'Identity proof policy'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_identity_proof_and_main_state_must_match(self):
        def change(proof, main): main['initial_head_state'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'linkage'):
            check_identity_preflights(self.save_proof(change), self.p)

    def test_partial_preflight_stop_is_preserved_without_quality_acceptance(self):
        folder = self.root / 'partial'; folder.mkdir()
        result = check_identity_preflights(folder, self.p)
        self.assertFalse(result['proof_present'])
        self.assertFalse(result['completed_main_preflight'])
        self.assertNotIn('necessary_capacity_pass', result)

    def test_training_without_identity_proof_is_rejected(self):
        folder = self.root / 'missing'; (folder / 'outputs').mkdir(parents=True)
        write(folder / 'outputs/results.json', {'complete': True})
        with self.assertRaisesRegex(ValueError, 'lacks exact-zero identity proof'):
            check_identity_preflights(folder, self.p)

    def test_valid_artificial_early_failure_keeps_original_training_backward_count(self):
        result = check_execution(*self.execution_fixture())
        self.assertEqual(result['updates'], 50)
        self.assertEqual(result['backwards'], 51)
        self.assertTrue(result['neural_counts_verified'])
        self.assertNotIn('necessary_capacity_pass', result)

    def test_legacy_forward_counters_cannot_replace_new_proof_counts(self):
        def change(e, g, t, s, q): e['neural_forward_counts']['detail_head'] -= 10
        with self.assertRaisesRegex(ValueError, 'Terminal neural counts'):
            check_execution(*self.execution_fixture(change))

    def test_original_timing_projection_samples_remain_exact(self):
        def change(e, g, t, s, q): t['steady_sample_seconds'][0] += .001
        with self.assertRaisesRegex(ValueError, 'timing samples'):
            check_execution(*self.execution_fixture(change))

    def test_zero_supervised_exit_without_final_result_is_rejected(self):
        def change(e, g, t, s, q): s['trainer_exit_code'] = 0
        args = list(self.execution_fixture(change))
        (args[0] / 'trainer_exit_code.txt').write_text('0\n', encoding='ascii')
        copied = (args[0] / 'outputs/execution_receipt.json').resolve()
        self.assertTrue(copied.is_relative_to(self.root.resolve()))
        copied.unlink()
        args[4] = None
        with self.assertRaisesRegex(ValueError, 'Successful supervised exit lacks completed800 result'):
            check_execution(*args)

    def test_original_metric_capacity_and_replay_definitions_are_AST_identical(self):
        new = ast.parse((ROOT / 'scripts/audit_cctv_dgp_batchmatched_identity_v26.py').read_text())
        old = ast.parse((ROOT / 'scripts/audit_cctv_dgp_spatial_features_v25.py').read_text())
        names = ['rgb', 'raw_rgb', 'vector', 'observed', 'interior', 'feature_mask', 'pixel_metrics',
                 'detail_metric', 'png', 'aggregate', 'capacity', 'numeric_tree', 'state_hash',
                 'make_head', 'verify_initial_state', 'fixed_grid']
        for name in names:
            function = lambda tree: next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(function(new)), ast.dump(function(old)), name)
        self.assertEqual(audit.HEAD_RAW_TOLERANCE, 2e-6)
        self.assertEqual(audit.RECOGNIZER_VECTOR_TOLERANCE, 2e-5)
        self.assertEqual(audit.RECOGNIZER_COSINE_TOLERANCE, 5e-5)


if __name__ == '__main__':
    unittest.main()
