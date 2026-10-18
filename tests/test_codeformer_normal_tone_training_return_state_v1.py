"""CPU-only saved-audit mutations; never import Torch or read checkpoints."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('normal_tone_return_state_test',ROOT/'scripts/audit_codeformer_normal_tone_training_return_state_v1.py')
A=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(A)


class SavedStateAuditFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=A.read(ROOT/'outputs/cctv_codeformer_normal_tone_training_v1_vm/protocol.json')
        cls.stage=A.read(ROOT/'outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json')
        cls.parity=A.read(Path(cls.stage['local_dependency_resolution']['training_parity_v1.json']))
        cls.refs=A.read(Path(cls.stage['local_dependency_resolution']['references.json']))['references']

    def test_exact_2075_plan_and_50_commits(self):
        rows,counts=A.events(self.p)
        self.assertEqual(len(rows),2075)
        self.assertEqual(counts['gradient_queries'],11)
        self.assertEqual(counts['optimizer_updates'],50)
        self.assertEqual(rows[-1]['pending_microbatches'],0)
        self.assertEqual(rows[-1]['committed_update'],50)

    def test_call_cap_mutation_refused(self):
        p=deepcopy(self.p);p['training']['call_caps']['gradient_queries']=9
        with self.assertRaises(ValueError):A.events(p)

    def test_sampler_replay_uses_391_references(self):
        ids=[r['id'] for r in self.refs if r['historical_role']=='train']
        states=A.sampler_states(ids,self.parity['updates'],np)
        self.assertEqual(states[50]['completed_updates'],50)
        self.assertEqual(states[50]['selection_count'],100)
        self.assertEqual(states[50]['next_update'],51)
        self.assertEqual(states[0]['permutation'],[])

    def test_sampler_identity_mutation_refused(self):
        ids=[r['id'] for r in self.refs if r['historical_role']=='train']
        rows=deepcopy(self.parity['updates']);rows[0]['TRAIN_selection']['id_a']='tr_ffhq_00000'
        with self.assertRaises(ValueError):A.sampler_states(ids,rows,np)

    def test_partial_import_refused_before_torch(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch') as tmp:
            path=Path(tmp)/'partial.json';path.write_text(json.dumps({'complete':False}))
            with self.assertRaisesRegex(ValueError,'completed root import'):A.Evidence(path)
        self.assertFalse(any(n=='torch' or n.startswith('torch.') for n in sys.modules))

    def test_duplicate_and_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch') as tmp:
            path=Path(tmp)/'bad.json'
            for value in ('{"x":1,"x":2}','{"x":NaN}'):
                path.write_text(value)
                with self.subTest(value=value),self.assertRaises(ValueError):A.read(path)


if __name__=='__main__':unittest.main()
