"""Detached arithmetic and VM-boundary contracts; no model backward or training."""
import copy
import math
import json
from pathlib import Path
import shutil
import sys
from contextlib import contextmanager
import uuid
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_conflict_training_v5 import combine_gradients, rebuild_policy_summary, ARMS
import cctv_dgp_conflict_training_v5 as recipe


@contextmanager
def fixture_directory(label):
    # Windows mkdtemp's owner-only ACL excludes the sandbox principal. Ordinary
    # workspace folders inherit the accessible project ACL; retain audit fixtures.
    base=ROOT/'scratch/cctv-dgp-conflict-tests'
    base.mkdir(parents=True,exist_ok=True)
    directory=base/(label+'-'+uuid.uuid4().hex)
    assert directory.resolve().is_relative_to(base.resolve())
    directory.mkdir()
    yield directory


def recipe_fixture(directory):
    parent={'arms':[], 'references':[], 'training_epochs':{'1':[]}}
    for name in recipe.REQUIRED_ASSETS-{'v4_local_audit_for_training.json'}:
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    capsule=ROOT/'outputs/cctv_dgp_objective_return_v4/local_independent_audit.json'
    shutil.copyfile(capsule,directory/'v4_local_audit_for_training.json')
    diagnostic=ROOT/'outputs/cctv_dgp_objective_vm_v4'
    returned=ROOT/'outputs/cctv_dgp_objective_return_v4/outputs/cctv_dgp_objective_diagnostic_v4'
    fixed=copy.deepcopy(parent);fixed['arms']=ARMS
    manifest={'format':'cctv-two-objective-conflict-pilot-v5',
              'diagnostic_protocol_sha256':recipe.sha(diagnostic/'objective_diagnostic_protocol_v4.json'),
              'diagnostic_results_sha256':recipe.V4_RESULTS_SHA,'diagnostic_local_audit_sha256':recipe.V4_AUDIT_SHA,
              'starting_checkpoint_sha256':recipe.START_SHA,'starting_state_hash':recipe.START_STATE,
              'assets_sha256':{name:recipe.sha(directory/name) for name in recipe.REQUIRED_ASSETS},
              'protocol':fixed,'runtime_cap_seconds':1800,'expected_training_autograd_grad_calls':904,
              'expected_preflight_autograd_grad_calls':8,'native_cases_used':0,'native_reserved_used':False,
              'production_promotion_permitted':False}
    recipe.configure_paths(directory,perceptual_bundle_dir=diagnostic,diagnostic_bundle_dir=diagnostic,v4_return=returned)
    return parent,manifest,json.loads(capsule.read_text())


def seal(directory,manifest):
    file=directory/'conflict_protocol_v5.json'
    file.write_text(json.dumps(manifest),encoding='utf-8')
    (directory/'conflict_protocol_v5.sha256').write_bytes((recipe.sha(file)+'\n').encode('ascii'))


class ConflictPolicyContracts(unittest.TestCase):
    def tearDown(self):
        recipe.configure_paths(ROOT)

    def test_nonconflicting_gradients_remain_the_weighted_sum(self):
        r, i = (torch.tensor([2., 1.]),), (torch.tensor([1., 3.]),)
        for policy in ('weighted_sum', 'two_objective_pcgrad'):
            merged, receipt = combine_gradients(r, i, policy)
            torch.testing.assert_close(merged[0], r[0] + i[0], rtol=0, atol=0)
            self.assertFalse(receipt['conflict'])
            self.assertEqual(receipt['coefficients'], [1., 1.])
            self.assertFalse(merged[0].requires_grad)

    def test_conflicting_pair_projects_both_original_objectives(self):
        r, i = (torch.tensor([2., 0.]),), (torch.tensor([-1., 1.]),)
        merged, receipt = combine_gradients(r, i, 'two_objective_pcgrad')
        # Original dot=-2; projected r=(1,1), projected i=(0,1).
        torch.testing.assert_close(merged[0], torch.tensor([1., 2.]), rtol=0, atol=0)
        self.assertTrue(receipt['conflict'])
        self.assertEqual(receipt['coefficients'], [1.5, 2.])
        self.assertGreaterEqual(receipt['combined_identity_dot'], 0)
        self.assertGreaterEqual(receipt['combined_reconstruction_dot'], 0)
        self.assertEqual(receipt, rebuild_policy_summary('two_objective_pcgrad', 2, [[4., -2.], [-2., 2.]]))

    def test_weight_only_arm_does_not_silently_project_conflict(self):
        merged, receipt = combine_gradients((torch.tensor([2., 0.]),), (torch.tensor([-1., 1.]),), 'weighted_sum')
        torch.testing.assert_close(merged[0], torch.tensor([1., 1.]), rtol=0, atol=0)
        self.assertTrue(receipt['conflict'])
        self.assertEqual(receipt['coefficients'], [1., 1.])

    def test_projection_uses_global_parameters_not_per_tensor_decisions(self):
        r = (torch.tensor([2.]), torch.tensor([0.]))
        i = (torch.tensor([-1.]), torch.tensor([1.]))
        originals = tuple(v.clone() for v in r+i)
        merged, receipt = combine_gradients(r, i, 'two_objective_pcgrad')
        torch.testing.assert_close(torch.cat(merged), torch.tensor([1., 2.]), rtol=0, atol=0)
        for actual, before in zip(r+i, originals):
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
        self.assertEqual(receipt['dimension'], 2)

    def test_zero_or_antiparallel_identity_handles_denominators(self):
        r = (torch.tensor([2., 0.]),)
        merged, receipt = combine_gradients(r, (torch.zeros(2),), 'two_objective_pcgrad')
        torch.testing.assert_close(merged[0], r[0], rtol=0, atol=0)
        self.assertIsNone(receipt['cosine'])
        merged, receipt = combine_gradients(r, (torch.tensor([-1., 0.]),), 'two_objective_pcgrad')
        torch.testing.assert_close(merged[0], torch.zeros(2), rtol=0, atol=0)
        self.assertEqual(receipt['combined_l2'], 0.)

    def test_invalid_gradient_data_or_metadata_cannot_train(self):
        for r, i in (((), ()), ((torch.ones(2),), (torch.ones(3),)),
                     ((torch.tensor([float('nan')]),), (torch.ones(1),))):
            with self.assertRaises(ValueError):
                combine_gradients(r, i, 'two_objective_pcgrad')
        for gram in ([[1., 2.], [2., 1.]], [[1., 0.], [1., 1.]], [[1., float('nan')], [0., 1.]]):
            with self.assertRaises(ValueError):
                rebuild_policy_summary('two_objective_pcgrad', 2, gram)
        with self.assertRaises(ValueError):
            combine_gradients((torch.ones(2),), (torch.ones(2),), 'unknown')

    def test_two_changed_arms_keep_postactivation_and_declared_weights(self):
        self.assertEqual([a['lambda_identity'] for a in ARMS], [.4, .1])
        self.assertEqual([a['gradient_policy'] for a in ARMS], ['weighted_sum', 'two_objective_pcgrad'])
        self.assertTrue(all(a['feature_policy']=='postactivation' for a in ARMS))

    def test_audit_rejects_changed_projection_receipt_norm_counter_or_dimension(self):
        from audit_cctv_dgp_conflict_results_v5 import validate_gradient_trace
        arm=ARMS[1]
        summary=rebuild_policy_summary(arm['gradient_policy'],2,[[4.,-2.],[-2.,2.]])
        trace={'gradient_policy':arm['gradient_policy'],'autograd_grad_calls':2,
               'cumulative_autograd_grad_calls':10,'policy_gradients':summary,'preclip_norm':math.sqrt(5.)}
        self.assertEqual(validate_gradient_trace(trace,arm,2,1),summary)
        for key,value in (('preclip_norm',0.),('autograd_grad_calls',1),
                          ('cumulative_autograd_grad_calls',11),('gradient_policy','weighted_sum')):
            changed=copy.deepcopy(trace); changed[key]=value
            with self.assertRaises(ValueError):
                validate_gradient_trace(changed,arm,2,1)
        changed=copy.deepcopy(trace); changed['policy_gradients']['coefficients']=[1.,1.]
        with self.assertRaises(ValueError):
            validate_gradient_trace(changed,arm,2,1)
        with self.assertRaises(ValueError):
            validate_gradient_trace(trace,arm,3,1)

    def test_generated_sources_match_frozen_v3_derivation(self):
        from derive_cctv_dgp_conflict_v5 import derive_sources
        for name,expected in derive_sources().items():
            self.assertEqual((ROOT/name).read_text(encoding='utf-8'),expected)

    def test_fake_cuda_on_other_host_is_rejected(self):
        import run_cctv_dgp_conflict_vm_v5 as runner
        with patch('torch.cuda.is_available',return_value=True), \
             patch.object(runner.sys,'platform','linux'), \
             patch.object(runner.platform,'node',return_value='different-vm'):
            with self.assertRaisesRegex(RuntimeError,'Linux VM'):
                runner.require_vm(ROOT)

    def test_resealed_protocol_cannot_change_arms_budget_or_reserve_policy(self):
        scratch=ROOT/'scratch';scratch.mkdir(exist_ok=True)
        with fixture_directory('recipe') as name:
            directory=Path(name).resolve();self.assertTrue(directory.is_relative_to(scratch.resolve()))
            parent,manifest,receipt=recipe_fixture(directory)
            seal(directory,manifest)
            with patch.object(recipe,'verify_recipe',return_value=(parent,{})), \
                 patch('scripts.audit_cctv_dgp_objective_diagnostic_v4.audit',return_value=receipt):
                self.assertEqual(recipe.verify_protocol_v5(ROOT),manifest['protocol'])
                for key,value in (('runtime_cap_seconds',3600),('native_reserved_used',True),
                                  ('expected_training_autograd_grad_calls',452),('production_promotion_permitted',True)):
                    changed=copy.deepcopy(manifest);changed[key]=value;seal(directory,changed)
                    with self.assertRaises(ValueError):
                        recipe.verify_protocol_v5(ROOT)
                changed=copy.deepcopy(manifest);changed['protocol']['arms'][0]['lambda_identity']=.2;seal(directory,changed)
                with self.assertRaises(ValueError):
                    recipe.verify_protocol_v5(ROOT)

    def test_resealed_runtime_source_cannot_diverge_from_executed_source(self):
        scratch=ROOT/'scratch';scratch.mkdir(exist_ok=True)
        with fixture_directory('source') as name:
            directory=Path(name).resolve();self.assertTrue(directory.is_relative_to(scratch.resolve()))
            parent,manifest,_=recipe_fixture(directory)
            source=directory/'cctv_dgp_conflict_training_v5.py'
            source.write_bytes(source.read_bytes()+b'\n# altered fixture\n')
            manifest['assets_sha256']['cctv_dgp_conflict_training_v5.py']=recipe.sha(source)
            seal(directory,manifest)
            with patch.object(recipe,'verify_recipe',return_value=(parent,{})), self.assertRaisesRegex(ValueError,'Executed V5 source'):
                recipe.verify_protocol_v5(ROOT)

    def test_completed_v4_audit_must_exactly_reproduce(self):
        scratch=ROOT/'scratch';scratch.mkdir(exist_ok=True)
        with fixture_directory('lineage') as name:
            directory=Path(name).resolve();self.assertTrue(directory.is_relative_to(scratch.resolve()))
            parent,manifest,receipt=recipe_fixture(directory);seal(directory,manifest)
            receipt['training_references']=39
            with patch.object(recipe,'verify_recipe',return_value=(parent,{})), \
                 patch('scripts.audit_cctv_dgp_objective_diagnostic_v4.audit',return_value=receipt), \
                 self.assertRaisesRegex(ValueError,'does not reproduce'):
                recipe.verify_protocol_v5(ROOT)

    def test_runner_rejects_local_training_before_loading_or_creating_output(self):
        import run_cctv_dgp_conflict_vm_v5 as runner
        with patch.object(runner, 'require_vm', side_effect=RuntimeError('VM only')), \
             patch.object(runner, 'verify_bundle') as verify, patch.object(runner, 'DGPSynthesizer') as load:
            with self.assertRaisesRegex(RuntimeError, 'VM only'):
                runner.run(ROOT, ROOT/'scratch/forbidden_local_training_v5')
            verify.assert_not_called()
            load.assert_not_called()
        self.assertFalse((ROOT/'scratch/forbidden_local_training_v5').exists())


if __name__ == '__main__':
    unittest.main()
