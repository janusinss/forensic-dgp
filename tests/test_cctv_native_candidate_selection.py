"""Keep rejected/baseline/corrupted VM candidates out of native output selection."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_cctv_dgp_native_pilot_review import validate_selection


class NativeSelectionChecks(unittest.TestCase):
    def setUp(self):
        self.returned={'branches':[
            {'arm':{'id':'camera_no_identity'},'selection':{'selected_epoch':0,'best_sha256':'a'*64}},
            {'arm':{'id':'camera_identity'},'selection':{'selected_epoch':2,'best_sha256':'b'*64}}]}
        self.execution={'candidates':[{'id':'camera_identity','selected_epoch':2,'sha256':'b'*64}]}

    def test_only_qualified_trained_epoch_is_admitted(self):
        self.assertEqual(validate_selection(self.execution,self.returned),self.execution['candidates'])

    def test_baseline_fallback_is_not_a_new_native_candidate(self):
        execution=copy.deepcopy(self.execution)
        execution['candidates'].insert(0,{'id':'camera_no_identity','selected_epoch':0,'sha256':'a'*64})
        with self.assertRaisesRegex(ValueError,'candidate selection'):
            validate_selection(execution,self.returned)

    def test_changed_hash_or_epoch_is_rejected(self):
        for key,value in [('sha256','c'*64),('selected_epoch',1)]:
            execution=copy.deepcopy(self.execution);execution['candidates'][0][key]=value
            with self.assertRaises(ValueError):
                validate_selection(execution,self.returned)

    def test_no_qualified_candidate_is_explicit_not_a_failed_read(self):
        returned=copy.deepcopy(self.returned);returned['branches'][1]['selection']['selected_epoch']=0
        self.assertEqual(validate_selection({'candidates':[]},returned),[])


if __name__=='__main__':
    unittest.main()
