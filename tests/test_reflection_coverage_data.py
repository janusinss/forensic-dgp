import copy
import importlib.util
import unittest


class QualificationContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.prepare_reflection_coverage_data'),
                             'Reviewed coverage-data preparation missing')

    def inputs(self):
        rows=[{'source_id':i,'path':f'dataset/{i}.png','sha256':f'{i:064x}',
               'usage':'pending','eyes':None,'eyes_reviewed':False,'split':'train'} for i in range(180)]
        for i in (0,4,101,102):rows[i].update(usage='paired_unoccluded',eyes=[[30.,40.],[70.,40.]],eyes_reviewed=True)
        ids=[6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,29,30,
             108,109,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,130,131,132]
        proposed=[{**rows[i],'decision':'proposed','eyes_native':[[30.,40.],[70.,40.]],'training_enabled':False} for i in ids]
        return {'records':rows},{'records':proposed,'native_review_complete':False,'training_enabled':False}

    def test_landmarks_need_declared_native_review_and_existing_source_membership(self):
        from scripts.prepare_reflection_coverage_data import qualify_sources
        prior,proposal=self.inputs();result=qualify_sources(prior,proposal)
        paired=[r for r in result if r['usage']=='paired_unoccluded']
        self.assertEqual(len(paired),28)
        by_id={r['source_id']:r for r in result}
        self.assertEqual(by_id[118]['usage'],'unpaired_occlusion')
        self.assertEqual(by_id[6]['usage'],'pending')
        self.assertFalse(by_id[6]['eyes_reviewed'])
        self.assertTrue(by_id[7]['eyes_reviewed'])
        self.assertFalse(by_id[7]['detector_training_enabled'])
        for bad_key in ('sha256','path'):
            bad=copy.deepcopy(proposal);bad['records'][1][bad_key]='changed'
            with self.assertRaises(ValueError):qualify_sources(prior,bad)

    def test_missing_or_unreliable_eye_proposal_cannot_enter_matched_pilot(self):
        from scripts.prepare_reflection_coverage_data import qualify_sources
        prior,proposal=self.inputs()
        for change in ('decision','eyes_native','eyes_reviewed','training_enabled'):
            bad=copy.deepcopy(proposal)
            row=next(r for r in bad['records'] if r['source_id']==7)
            row[change]={'decision':'needs_review','eyes_native':None,'eyes_reviewed':True,'training_enabled':True}[change]
            with self.assertRaises(ValueError):qualify_sources(prior,bad)


if __name__=='__main__':unittest.main()
