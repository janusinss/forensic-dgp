"""CPU metadata fixtures for two-source-held151-update prospective sampler."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_filter_fusion_training_v1 import data_v2 as d


def fixture():
    prep = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation'
    old = json.loads((prep / 'data_schedule_provisional_v1.json').read_text(encoding='utf-8'))
    rows = json.loads((prep / 'data_dark_linear_v1/manifest.json').read_text(encoding='utf-8'))['rows']
    return old, rows


class Metadata(unittest.TestCase):
    def test_import_no_torch(self):
        self.assertNotIn('torch', sys.modules)

    def test_301_unique_one_fixed_padding_no_held_sources(self):
        old, rows = fixture()
        schedule = d.revised_schedule(old, rows)
        sequence = [s for u in schedule['updates'] for s in u['slots'][:2]]
        self.assertEqual(len(sequence), 302)
        self.assertEqual(len({s['id'] for s in sequence}), 301)
        self.assertEqual([s['id'] for s in sequence if s['padding_duplicate']], [d.PADDING_ID])
        self.assertFalse(any(s['reference_id'] in d.HELD_SOURCES for u in schedule['updates'] for s in u['slots']))

    def test_subsequence_and_previous_evidence_unchanged(self):
        old, rows = fixture()
        before, before_rows = copy.deepcopy(old), copy.deepcopy(rows)
        schedule = d.revised_schedule(old, rows)
        expected = [s['id'] for u in old['updates'] for s in u['slots'][:2]
                    if s['reference_id'] not in d.HELD_SOURCES]
        actual = [s['id'] for u in schedule['updates'] for s in u['slots'][:2]]
        self.assertEqual(actual, expected)
        self.assertEqual(old, before)
        self.assertEqual(rows, before_rows)

    def test_samephoto_55_anchors_balanced(self):
        old, rows = fixture()
        schedule = d.revised_schedule(old, rows)
        self.assertEqual(len(schedule['anchor_source_order']), 55)
        counts = Counter(u['slots'][2]['reference_id'] for u in schedule['updates'])
        self.assertEqual(Counter(counts.values()), {3: 41, 2: 14})
        self.assertTrue(all(u['slots'][2]['reference_id'] == u['slots'][3]['reference_id']
                            for u in schedule['updates']))

    def test_sampler_exact151_604_and_no_extension(self):
        old, rows = fixture()
        sampler = d.FiniteSampler(d.revised_schedule(old, rows))
        for _ in range(151):
            preview = sampler.preview()
            self.assertEqual(preview, sampler.preview())
            sampler.commit(preview)
        self.assertEqual(sampler.state()['committed_microbatches'], 604)
        with self.assertRaises(ValueError):
            sampler.preview()

    def test_stale_partial_preview_rejects(self):
        old, rows = fixture()
        sampler = d.FiniteSampler(d.revised_schedule(old, rows))
        preview = sampler.preview()
        sampler.commit(preview)
        with self.assertRaises(ValueError):
            sampler.commit(preview)
        preview = sampler.preview()
        preview['slots'].pop()
        with self.assertRaises(ValueError):
            sampler.commit(preview)

    def test_held_or_duplicate_schedule_rejected(self):
        old, rows = fixture()
        schedule = d.revised_schedule(old, rows)
        schedule['updates'][0]['slots'][0]['reference_id'] = d.HELD_SOURCES[0]
        with self.assertRaises(ValueError):
            d.FiniteSampler(schedule)

    def test_anchor_gap_or_extra_requires_new_contract(self):
        old, rows = fixture()
        rid = next(r['reference_id'] for r in rows if r['reference_id'] not in d.HELD_SOURCES)
        with self.assertRaises(ValueError):
            d.revised_schedule(old, [r for r in rows if r['reference_id'] != rid])

    def test_separate_v2_admission_required(self):
        old, rows = fixture()
        schedule = d.revised_schedule(old, rows)
        with self.assertRaises(ValueError):
            d.require_admission({'format': 'codeformer-filter-fusion-explicit-admission-v1'}, 'a', 'b', schedule)


if __name__ == '__main__':
    unittest.main()
