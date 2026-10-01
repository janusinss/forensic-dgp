"""Leakage counterexamples for extending an old source-only eyewear queue."""
import unittest
from scripts.prepare_real_reflection_review import qualify_sources


class RealReflectionReviewTests(unittest.TestCase):
    def source(self, index, digest):
        return {'index': index, 'path': f'dataset/pool/{index}.jpg', 'sha256': digest*64,
                'partition': 'train', 'training_enabled': False, 'mask': None}

    def test_newly_used_source_is_excluded_even_if_old_queue_said_train(self):
        rows = [self.source(1, 'a'), self.source(2, 'b')]
        current = [{'source_sha256': 'a'*64, 'image_sha256': 'c'*64, 'split': 'train'}]
        eligible, excluded = qualify_sources(rows, {r['path'] for r in rows}, set(), current, set())
        self.assertEqual([r['index'] for r in eligible], [2])
        self.assertEqual(excluded[0]['reason'], 'already_reviewed_source_or_crop')

    def test_held_out_or_benchmark_sources_cannot_enter_extension(self):
        rows = [self.source(1, 'a'), self.source(2, 'b')]
        eligible, excluded = qualify_sources(rows, {rows[1]['path']}, {rows[0]['path']}, [], {'b'*64})
        self.assertEqual(eligible, [])
        self.assertEqual({r['reason'] for r in excluded}, {'outside_fixed_training_split', 'benchmark_source'})

    def test_duplicate_candidate_content_and_automatic_labels_are_rejected(self):
        a, b = self.source(1, 'a'), self.source(2, 'a')
        with self.assertRaises(ValueError):
            qualify_sources([a, b], {a['path'], b['path']}, set(), [], set())
        for changes in ({'training_enabled': True}, {'mask': 'invented.png'}, {'partition': 'validation'}):
            row = {**a, **changes}
            with self.assertRaises(ValueError):
                qualify_sources([row], {a['path']}, set(), [], set())


if __name__ == '__main__':
    unittest.main()
