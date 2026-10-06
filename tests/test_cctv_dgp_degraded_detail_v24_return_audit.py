"""Transfer/receipt tamper and exact zero-initializer checks for V24; no fitting."""
import copy
import ast
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
import import_cctv_dgp_degraded_detail_v24 as transfer
import audit_cctv_dgp_degraded_detail_v24 as audit
from audit_cctv_dgp_degraded_detail_v24_execution import check_execution
from audit_cctv_dgp_degraded_detail_v24_cohort import check_receipt, audit_cohort


class V24ReturnChecks(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'scratch' / ('v24_return_fixture_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        # Shared unchanged head/execution schema fixture; not a V24 VM result.
        self.actual = ROOT / 'outputs/cctv_dgp_detail_skip_v23_return'

    def make_archive(self, extra=None, bad_hash=False):
        names = ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_degraded_detail_v24.py', 'scripts/run_v24.sh')
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
        for receipt in (execution, supervisor, failure):receipt['protocol_sha256'] = transfer.PIN
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
        self.assertEqual(len(p['assets_sha256']), 221)
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

    def test_reported_archive_binding_rejects_otherwise_valid_fixture(self):
        archive,sidecar,receipt=self.make_archive()
        with self.assertRaisesRegex(ValueError,'reported archive hash'):
            transfer.import_return(archive,sidecar,receipt,self.root/'imported',self.root,expected_sha='0'*64)
        self.assertFalse((self.root/'imported').exists())

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

    def test_all_original_auditor_logic_and_tolerances_remain_exact(self):
        source=(ROOT/'scripts/audit_cctv_dgp_degraded_detail_v24.py').read_text()
        for before,after in [('cctv_dgp_degraded_detail_v24','cctv_dgp_detail_skip_v23'),
                             ('cctv_dgp_degraded_detail_vm_v24','cctv_dgp_detail_skip_vm_v23'),
                             ('run_v24.sh','run_v23.sh'),('V24','V23'),
                             ('dgp-degraded-detail-cohort-capacity-v24','dgp-learned-detail-skip-capacity-v23'),
                             ("len(p['assets_sha256']) == 221","len(p['assets_sha256']) == 209")]:source=source.replace(before,after)
        original=ast.parse((ROOT/'scripts/audit_cctv_dgp_detail_skip_v23.py').read_text());revised=ast.parse(source)
        function=next(n for n in revised.body if isinstance(n,ast.FunctionDef) and n.name=='audit_return')
        extra=ast.parse("from audit_cctv_dgp_detail_skip_v23_cohort import audit_cohort\ncohort_audit = audit_cohort(bundle, returned, p, head, clock)").body
        for item in extra:
            matches=[n for n in function.body if ast.dump(n)==ast.dump(item)]
            self.assertEqual(len(matches),1);function.body.remove(matches[0])
        report=next(n.value for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets))
        for key in ['cohort_loss_audit','cohort_checker_sha256']:
            matches=[i for i,k in enumerate(report.keys) if isinstance(k,ast.Constant) and k.value==key]
            self.assertEqual(len(matches),1);index=matches[0];report.keys.pop(index);report.values.pop(index)
        self.assertEqual(ast.dump(original),ast.dump(revised))

    def cohort(self):
        reference=ROOT/'outputs/cctv_dgp_degraded_detail_v24_preparation/CPU_cohort_contract.json'
        self.assertEqual(transfer.sha(reference),'39e41b746ad1252bcde615541367b7ad3f7bf131f9fbcf4f44df6936c2c01816')
        receipt=transfer.read(reference)
        return receipt,copy.deepcopy(receipt['rows'])

    def test_cohort_baseline_receipt_passes_without_learning_claim(self):
        receipt,expected=self.cohort();result=check_receipt(receipt,expected)
        self.assertTrue(result['policy_and_all50_rows_verified']);self.assertEqual(result['maximum_independent_baseline_row_difference'],0)
        self.assertNotIn('necessary_capacity_pass',result)

    def test_changed_cohort_baseline_rejected_even_with_matching_mean(self):
        import numpy as np
        receipt,expected=self.cohort()
        for row in receipt['rows']:row['feature_MSE']*=1.001
        receipt['feature_normalizer']=float(np.float32(sum(r['feature_MSE'] for r in receipt['rows'] if not r['clear'])/40))
        with self.assertRaisesRegex(ValueError,'Independent cohort baseline differs'):check_receipt(receipt,expected)

    def test_clear_rows_cannot_enter_degraded_cohort_mean(self):
        import numpy as np
        receipt,expected=self.cohort();receipt['feature_normalizer']=float(np.float32(sum(r['feature_MSE'] for r in receipt['rows'])/50))
        with self.assertRaisesRegex(ValueError,'normalizer arithmetic differs'):check_receipt(receipt,expected)

    def test_cohort_row_relabel_or_omission_rejected(self):
        receipt,expected=self.cohort();receipt['rows'][0]['clear']=False
        with self.assertRaisesRegex(ValueError,'ID/control order differs'):check_receipt(receipt,expected)
        receipt,expected=self.cohort();receipt['rows'].pop()
        with self.assertRaisesRegex(ValueError,'row count differs'):check_receipt(receipt,expected)

    def test_clear_reward_and_pre_optimizer_policy_cannot_change(self):
        for key,value in [('clear_reward',True),('optimizer_constructed',True),('normalizer_floor',2e-6),('degraded_weight',1.0)]:
            receipt,expected=self.cohort();receipt[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Cohort policy differs'):check_receipt(receipt,expected)

    def test_missing_cohort_only_permitted_before_any_training_evidence(self):
        returned=self.root/'empty';returned.mkdir()
        result=audit_cohort(audit.BUNDLE,returned,{},None,lambda:None)
        self.assertFalse(result['receipt_present']);self.assertEqual(result['fixed_CPU_high_pass_calls'],0)
        (returned/'outputs').mkdir();(returned/'outputs/results.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'lacks frozen cohort setup'):audit_cohort(audit.BUNDLE,returned,{},None,lambda:None)


if __name__ == '__main__':
    unittest.main()
