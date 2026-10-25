"""No-neural fixtures for the finite scientific data contract."""
from collections import Counter
import copy
from pathlib import Path
import unittest
from research.codeformer_skip_feature_adapter_v1 import data as d


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.order = ['tr_ffhq_%05d' % i for i in range(55)]

    def test_complete_exposures_no_implicit_epoch_extension(self):
        schedule = d.schedule_for(self.order)
        counts = Counter((s['role'], s['reference_id']) for u in schedule['updates'] for s in u['slots'])
        self.assertEqual(len(schedule['updates']), 110)
        self.assertEqual(sum(counts.values()), 440)
        for rid in self.order:
            self.assertEqual([counts[r,rid] for r in ('filter','clear','dark')], [4,2,2])
        sampler = d.FiniteSampler(schedule)
        for index in range(110):
            row = sampler.preview(); self.assertEqual(row['update'], index+1); sampler.commit(row)
        self.assertEqual(sampler.state()['completed_microbatches'],440)
        with self.assertRaises(ValueError):sampler.preview()

    def test_rejected_scope_and_duplicate_sources(self):
        bad = list(self.order); bad[0] = d.HELD[0]
        with self.assertRaises(ValueError):d.schedule_for(bad)
        bad = list(self.order); bad[-1] = bad[0]
        with self.assertRaises(ValueError):d.schedule_for(bad)

    def test_partial_or_mutated_commit_does_not_advance(self):
        sampler=d.FiniteSampler(d.schedule_for(self.order)); before=sampler.state()
        row=sampler.preview(); row['slots'].pop()
        with self.assertRaises(ValueError):sampler.commit(row)
        self.assertEqual(sampler.state(),before)
        row=sampler.preview(); sampler.commit(row)
        with self.assertRaises(ValueError):sampler.commit(row)

    def test_sampler_owns_schedule_copy(self):
        original=d.schedule_for(self.order); sampler=d.FiniteSampler(original)
        original['updates'][0]['slots'][0]['id']='mutated'
        self.assertNotEqual(sampler.preview()['slots'][0]['id'],'mutated')

    def test_portable_path_and_duplicate_json(self):
        for bad in ('../a','/a','a/../b','C:/x','a\\b'):
            with self.assertRaises(ValueError):d.canonical(bad)
        folder=Path(__file__).resolve().parents[1]/'scratch/skip_feature_data_fixtures'
        folder.mkdir(parents=True,exist_ok=True)
        for name,payload in (('duplicate.json',b'{"one":1,"one":2}'),('nonfinite.json',b'{"x":NaN}')):
            path=folder/name
            if not path.exists():
                with path.open('xb') as stream:stream.write(payload)
            self.assertEqual(path.read_bytes(),payload)
            with self.assertRaises(ValueError):d.read_json(path)


if __name__=='__main__':unittest.main()
