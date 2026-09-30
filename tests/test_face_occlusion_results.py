"""Returned-result audit catches altered exposure and false eligibility."""
import copy
import unittest

from scripts.audit_face_occlusion_results import audit_history, canonical_member


class FaceOcclusionResultsTests(unittest.TestCase):
    def fixture(self):
        schedule = [[[0,1,2,3,73,74,75,76] for _ in range(21)] for _ in range(10)]
        real = dict(iou=.1,missed_fraction=.8,visible_false_positive=.02,
                    empty_mask_cases=1,covered_cases=15,negative_false_positive_cases=1,negative_cases=10)
        synthetic = dict(iou=.9,missed_fraction=.1,visible_false_positive=.01,
                         empty_mask_cases=0,covered_cases=320,negative_false_positive_cases=0,negative_cases=80)
        baseline = {'real':real,'synthetic':synthetic}
        steps = {arm:[dict(epoch=e+1,step=s+1,updates=e*21+s+1,indices=batch,
                           loss=1.,pre_clip_norm=.5)
                     for e,batches in enumerate(schedule) for s,batch in enumerate(batches)]
                 for arm in ('random','pretrained')}
        histories = {}
        for arm in steps:
            histories[arm]=[]
            for n,epoch in enumerate((1,5,10)):
                scores = copy.deepcopy(baseline)
                scores['real']['iou'] = .2 + n*.01 if arm=='pretrained' else .09
                histories[arm].append(dict(epoch=epoch,updates=epoch*21,mean_epoch_loss=1.,
                    real_gate=arm=='pretrained',synthetic_gate=True,selected=arm=='pretrained',**scores))
        complete = dict(complete=True,parent_unchanged=True,
                        arms={arm:dict(updates=210,selected_epochs=[1,5,10] if arm=='pretrained' else []) for arm in steps})
        return steps,histories,schedule,baseline,complete

    def test_fixed_420_updates_and_selection_are_reconstructed(self):
        result = audit_history(*self.fixture())
        self.assertEqual(result['total_logged_updates'],420)
        self.assertEqual(result['arms']['pretrained']['selected_epochs'],[1,5,10])
        self.assertEqual(result['arms']['random']['selected_epochs'],[])

    def test_changed_batch_or_nonfinite_gradient_fails(self):
        data = self.fixture();data[0]['random'][40]['indices'] = [99999]*8
        with self.assertRaisesRegex(ValueError,'schedule'):
            audit_history(*data)
        data = self.fixture();data[0]['pretrained'][0]['pre_clip_norm'] = float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):
            audit_history(*data)

    def test_retention_failure_cannot_be_selected(self):
        data = self.fixture();data[1]['pretrained'][0]['synthetic']['missed_fraction'] = .11
        with self.assertRaisesRegex(ValueError,'selection'):
            audit_history(*data)

    def test_archive_aliases_and_windows_paths_are_rejected(self):
        self.assertEqual(canonical_member('outputs/pilot/mask.png'),'outputs/pilot/mask.png')
        for name in ('../escape','/absolute','C:/absolute','outputs/../escape',
                     'outputs//alias','./outputs/alias','outputs\\windows'):
            with self.assertRaises(ValueError):canonical_member(name)


if __name__=='__main__':unittest.main()
