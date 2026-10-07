"""Malformed returns/frozen partitions/quality guards; synthetic, no VM results."""
import copy
import importlib.util
from pathlib import Path
import tarfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


AUDIT=module('v29_return',ROOT/'scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py')
TRAIN=module('v29_frozen_partition',ROOT/'outputs/cctv_dgp_mean_centered_decoder_vm_v29/cctv_dgp_mean_centered_v29_training.py')


def member(name='protocol.json',kind=tarfile.REGTYPE,size=2):
    m=tarfile.TarInfo(AUDIT.PREFIX+name);m.type=kind;m.size=size;return m


def rows():
    result=[]
    for source in ['source_a','source_b']:
        for profile in ['clear','blur','motion','lowlight','compound']:
            for i in range(5):
                result.append({'id':source+profile+str(i),'source':source,'profile':profile,
                    'metrics':{'MSE':.001,'SSIM':.8,'ArcFace_observed_fixed':.7,
                               'landmark_high_frequency_MSE':.002,'constant_mean_shift_only_MSE':.001}})
    return result


class V29ReturnSafetyTests(unittest.TestCase):
    def test_links_and_traversal_rejected(self):
        for m in [member(kind=tarfile.SYMTYPE),member('../protocol.json')]:
            with self.subTest(name=m.name),self.assertRaises(AssertionError):AUDIT.validate_members([m],{'protocol.json','../protocol.json'})

    def test_case_collisions_and_unexpected_weights_rejected(self):
        for values,allowed in [([member(),member('Protocol.json')],{'protocol.json','Protocol.json'}),([member('unknown.pth')],{'protocol.json'})]:
            with self.assertRaises(AssertionError):AUDIT.validate_members(values,allowed)

    def test_member_and_archive_size_rejected(self):
        with self.assertRaises(AssertionError):AUDIT.validate_members([member(size=5)],{'protocol.json'},total_cap=4)
        with self.assertRaises(AssertionError):AUDIT.validate_members([member(size=65*1024**2)],{'protocol.json'})

    def test_matrix_nonfinite_and_wrong_type_rejected(self):
        for a in [np.zeros((7,4),np.float32),np.full((7,4),np.nan,np.float64),np.zeros((6,4),np.float64)]:
            with self.assertRaises(AssertionError):AUDIT.matrix(a,(7,4))

    def test_gradient_partition_detects_changed_value(self):
        a=np.zeros((7,4),np.float64);a[0,0]=1;a[1,2]=2
        layout=[{'name':'left','start':0,'end':2},{'name':'right','start':2,'end':4}]
        before=AUDIT.statistics(a,layout)
        self.assertEqual(before[3]['left']['improvement_gradient_norm'],1)
        a[0,0]=3;after=AUDIT.statistics(a,layout)
        self.assertNotEqual(before[3],after[3])

    def test_frozen_hash_detects_head4_and_buffer_change(self):
        class State:
            def __init__(self):self.values={'head1.weight':torch.tensor([1.]),'head4.weight':torch.tensor([2.]),'norm.running_mean':torch.tensor([0.])}
            def state_dict(self):return self.values
        value=State();baseline=TRAIN.frozen_partition(value,{'head1.weight'})
        value.values['head1.weight']=torch.tensor([3.]);self.assertEqual(baseline,TRAIN.frozen_partition(value,{'head1.weight'}))
        for key in ['head4.weight','norm.running_mean']:
            current=State();current.values[key]=torch.tensor([9.])
            self.assertNotEqual(baseline,TRAIN.frozen_partition(current,{'head1.weight'}))

    def test_all17_groups_and_clear_controls_remain(self):
        values=AUDIT.groups(rows());self.assertEqual(len(values),17)
        self.assertEqual(values['all']['cases'],50);self.assertEqual(values['degraded']['cases'],40);self.assertEqual(values['clear']['cases'],10)
        self.assertEqual(values['source_a/clear']['cases'],5)

    def test_structure_gain_cannot_override_preservation_failure(self):
        base=AUDIT.groups(rows());candidate=copy.deepcopy(base)
        for group in candidate.values():
            group['MSE']=.0008;group['landmark_high_frequency_MSE']=.0016
        self.assertTrue(AUDIT.capacity(base,candidate)[4])
        for group in candidate.values():group['ArcFace_observed_fixed']=.69
        self.assertFalse(AUDIT.capacity(base,candidate)[4])

    def test_brightness_and_source_regression_remain_failures(self):
        base=AUDIT.groups(rows());candidate=copy.deepcopy(base)
        for group in candidate.values():group['MSE']=.0008;group['landmark_high_frequency_MSE']=.0016
        candidate['degraded']['constant_mean_shift_only_MSE']=.0007
        self.assertFalse(AUDIT.capacity(base,candidate)[4])
        candidate['degraded']['constant_mean_shift_only_MSE']=.001
        candidate['source_a/degraded']['landmark_high_frequency_MSE']=.0021
        self.assertFalse(AUDIT.capacity(base,candidate)[4])


if __name__=='__main__':unittest.main()
