"""Replay diagnostics must aggregate pixels and preserve source/weight membership."""
import unittest
import numpy as np

from scripts.compare_replay_fit import counts
from scripts.diagnose_face_occlusion_replay import summarize_records


class FaceOcclusionReplayTests(unittest.TestCase):
    def records(self):
        return [dict(case=0,kind='lower',degraded=False,source='dataset/asian_faces',
                     counts=counts(np.ones(2,dtype=bool),np.ones(2,dtype=bool))),
                dict(case=1,kind='eyes',degraded=True,source='dataset/thumbnails128x128',
                     counts=counts(np.array([True]+[False]*7),np.ones(8,dtype=bool)))]

    def test_pixel_aggregation_and_source_strata_are_preserved(self):
        result=summarize_records(self.records())
        self.assertAlmostEqual(result['all']['iou'],.3)
        self.assertEqual(result['by_source']['dataset/asian_faces']['iou'],1.)
        self.assertEqual(result['by_source']['dataset/thumbnails128x128']['iou'],.125)
        self.assertEqual(result['by_kind_degradation']['eyes/True']['missed_fraction'],.875)

    def test_schedule_weights_differ_from_unique_case_means(self):
        result=summarize_records(self.records(),{0:3,1:1})
        self.assertAlmostEqual(result['all']['iou'],.5)
        self.assertEqual(result['all']['unique_cases'],2)
        self.assertEqual(result['all']['cases'],4)

    def test_duplicates_and_incomplete_or_invalid_weights_are_rejected(self):
        rows=self.records()
        with self.assertRaises(ValueError):summarize_records(rows+[rows[0]])
        for weights in ({0:1},{0:1,1:0},{0:1,1:1,2:1},{0:1,1:True}):
            with self.assertRaises(ValueError):summarize_records(rows,weights)


if __name__=='__main__':unittest.main()
