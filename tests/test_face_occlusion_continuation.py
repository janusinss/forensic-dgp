"""Fixed weight-continuation budget, source identity and VM guard."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.train_face_occlusion_continuation_vm import main, continuation_schedule, validate_start, checkpoint_counters


class FaceOcclusionContinuationTests(unittest.TestCase):
    def test_linux_cuda_guard_precedes_files_and_model(self):
        with patch('scripts.train_face_occlusion_continuation_vm.require_vm_gpu',side_effect=RuntimeError('VM required')), \
             patch('scripts.train_face_occlusion_continuation_vm.load_adapter') as load:
            with self.assertRaisesRegex(RuntimeError,'VM required'):main(SimpleNamespace())
            load.assert_not_called()

    def test_exact_two_cycles_do_not_mutate_original_schedule(self):
        protocol=json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
        original=copy.deepcopy(protocol)
        result=continuation_schedule(protocol)
        self.assertEqual(len(result),20)
        self.assertEqual(sum(len(e) for e in result),420)
        self.assertEqual(result[:10],protocol['schedules']['extended']['batches'])
        self.assertEqual(result[10:],protocol['schedules']['extended']['batches'])
        result[0][0][0]=99999
        self.assertEqual(protocol,original)
        self.assertEqual(result[10:],original['schedules']['extended']['batches'])

    def test_only_verified_pretrained_epoch10_is_an_eligible_start(self):
        payload=dict(initialization='pretrained',epoch=10,optimizer_updates=210)
        validate_start(payload)
        for key,value in [('initialization','random'),('epoch',5),('optimizer_updates',209)]:
            with self.assertRaises(ValueError):validate_start({**payload,key:value})
        with self.assertRaises(ValueError):validate_start({**payload,'optimizer':{}})

    def test_new_and_cumulative_updates_are_distinct_and_bounded(self):
        self.assertEqual(checkpoint_counters(1),{'epoch':11,'optimizer_updates':231,'additional_epoch':1,'fresh_optimizer_updates':21})
        self.assertEqual(checkpoint_counters(20),{'epoch':30,'optimizer_updates':630,'additional_epoch':20,'fresh_optimizer_updates':420})
        for epoch in (0,21,True):
            with self.assertRaises(ValueError):checkpoint_counters(epoch)


if __name__=='__main__':unittest.main()
