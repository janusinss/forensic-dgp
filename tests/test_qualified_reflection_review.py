import copy
import importlib.util
import unittest


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.prepare_qualified_reflection_review'),
                             'Versioned qualification registry builder missing')

    def cohort(self):
        return {'records':[{'source_id':i,'path':f'dataset/{i}.png','sha256':f'{i:064x}',
                            'source_pool':'test','clear_cases':[]} for i in (0,4,101,102,160,177,5)]}

    def test_only_native_reviewed_bases_are_eligible_and_suspected_coverings_stay_unpaired(self):
        from scripts.prepare_qualified_reflection_review import qualification_records
        records=qualification_records(self.cohort(),{'records':[
            {'source_id':160,'category':'likely_intrinsic_occlusion','rationale':'Opaque lens'},
            {'source_id':177,'category':'likely_intrinsic_occlusion','rationale':'Scene reflection'}]})
        by_id={r['source_id']:r for r in records}
        self.assertEqual([r['source_id'] for r in records if r['usage']=='paired_unoccluded'],[0,4,101,102])
        for i in (160,177):
            self.assertEqual(by_id[i]['usage'],'unpaired_occlusion')
            self.assertFalse(by_id[i]['detector_training_enabled'])
            self.assertIsNone(by_id[i]['accepted_mask'])
        self.assertEqual(by_id[5]['usage'],'pending')
        self.assertFalse(by_id[5]['eyes_reviewed'])

    def test_duplicate_source_id_or_bytes_cannot_create_independent_qualified_examples(self):
        from scripts.prepare_qualified_reflection_review import qualification_records
        for field in ('source_id','sha256','path'):
            cohort=self.cohort();cohort['records'][1][field]=cohort['records'][0][field]
            with self.assertRaises(ValueError):qualification_records(cohort,{'records':[]})

    def test_conflicting_prior_screen_cannot_be_silently_overruled_by_prototype_list(self):
        from scripts.prepare_qualified_reflection_review import qualification_records
        screening={'records':[{'source_id':0,'category':'likely_intrinsic_occlusion','rationale':'cover'}]}
        with self.assertRaises(ValueError):qualification_records(self.cohort(),screening)


if __name__=='__main__':unittest.main()
