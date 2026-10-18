"""Small saved-vector/ledger fixtures; no neural framework or real return read."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scripts import audit_codeformer_loss_response_return_state_v1 as a


class ReturnStateAuditFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads((a.ROOT/'outputs/cctv_codeformer_loss_response_v1_vm_r1/cases.json').read_text())['cases']

    def ledger(self):
        counts = dict.fromkeys(a.CAPS, 0)
        complete, previous, rows = 0, None, []
        for index,(name,context) in enumerate(a.expected_events(self.cases),start=1):
            if previous == 'optimizer_step_attempts':
                complete += 1;counts['disposable_optimizer_steps'] += 1
            counts[name] += 1
            rows.append({'event':index,'before_call':name,'context':context,'counts':dict(counts),
                         'actual_completed_Adam_steps':complete,'committed_training_updates':0,
                         'backward_calls':0,'seconds':index/1000})
            previous=name
        return rows

    def test_exact_ledger_semantics_separate_reserved_attempt_and_completed_step(self):
        rows=self.ledger();result=a.verify_ledger(rows,self.cases)
        self.assertEqual(result['events'],636)
        self.assertEqual(result['counts'],a.CAPS)
        first=next(i for i,row in enumerate(rows) if row['before_call']=='optimizer_step_attempts')
        self.assertEqual(rows[first]['actual_completed_Adam_steps'],0)
        self.assertEqual(rows[first+1]['actual_completed_Adam_steps'],1)
        self.assertEqual(result['completed_steps'],18)

    def test_stale_order_context_and_faked_step_completion_are_rejected(self):
        for kind in ('swap','context','complete','count','training'):
            rows=self.ledger()
            if kind=='swap':rows[0],rows[1]=rows[1],rows[0]
            elif kind=='context':rows[2]['context']={'reference_id':'final_fake'}
            elif kind=='complete':rows[0]['actual_completed_Adam_steps']=1
            elif kind=='count':rows[5]['counts']['HQ_teacher_forwards']+=1
            else:rows[-1]['committed_training_updates']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.verify_ledger(rows,self.cases)

    def test_eligibility_derives80component_queries_per_snapshot(self):
        self.assertEqual(sum(len(a.expected_terms(case)) for case in self.cases),80)
        self.assertEqual(sum('lpips_vgg' in a.expected_terms(case) for case in self.cases),12)
        self.assertEqual(sum('behavior_token_kl' in a.expected_terms(case) for case in self.cases),4)

    def test_component_deletion_and_explicit_zero_control(self):
        parts={key:None for key in a.COMPONENTS}
        parts['token_ce']=np.array([1.,2.],dtype=np.float32)
        parts['latent_mse']=np.array([3.,4.],dtype=np.float32)
        self.assertTrue(np.array_equal(a.compose_one(parts,None,False,np),[4.,6.]))
        self.assertTrue(np.array_equal(a.compose_one(parts,'token_ce',False,np),[3.,4.]))
        self.assertTrue(np.array_equal(a.compose_one(parts,None,True,np),[0.,0.]))
        self.assertIsNone(parts['lpips_vgg'])
        self.assertTrue(np.array_equal(parts['token_ce'],[1.,2.]))
        with self.assertRaises(ValueError):a.compose_one(dict.fromkeys(a.COMPONENTS),None,False,np)

    def test_named_vector_group_reduction_and_none_counts(self):
        stats=a.empty_stats()
        a.add_stats(stats,'encoder.x',np.array([3.,4.],dtype=np.float32),np)
        a.add_stats(stats,'ft_layers.x',None,np)
        a.add_stats(stats,'fuse_convs_dict.x',np.zeros(2,dtype=np.float32),np)
        result=a.finish_stats(stats)
        self.assertEqual(result['encoder']['l2'],5.)
        self.assertEqual(result['predictor']['none'],1)
        self.assertEqual(result['fusion']['none'],0)
        self.assertEqual(result['fusion']['nonzero'],0)
        a.compare_stats(result,copy.deepcopy(result))
        changed=copy.deepcopy(result);changed['encoder']['l2']=6.
        with self.assertRaises(ValueError):a.compare_stats(result,changed)
        rounded=copy.deepcopy(result)
        rounded['encoder']['l2']=np.nextafter(5.,6.)
        self.assertNotEqual(result,rounded)
        a.compare_stats(result,rounded)  # Independent reduction rounding, not a quality threshold.

    def test_fullsum_capture_tolerance_is_numeric_only_and_not_relaxable(self):
        record={'pass':True,'full_gradient_vector_exported':False,'atol':2e-5,'rtol':2e-4,
                'maximum_full_gradient_absolute':2.,'bound':.00042,'maximum_absolute':1e-5,'difference_l2':2e-5}
        a.check_sum_capture(record)
        record['atol']=3e-5
        with self.assertRaises(ValueError):a.check_sum_capture(record)

    def test_source_import_does_not_load_neural_framework_or_returned_module(self):
        self.assertFalse(any(name=='torch' or name.startswith('torch.') for name in sys.modules))
        self.assertNotIn('torch',a.__dict__)
        self.assertNotIn('research.codeformer_loss_response_v1.worker',sys.modules)


if __name__=='__main__':
    unittest.main()
