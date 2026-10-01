"""Current replay membership must not restore quarantined or unregistered data."""
import unittest
from scripts.run_frozen_complementarity import current_core_ids


class FrozenComplementarityProtocolTests(unittest.TestCase):
    def fixture(self):
        return {'batches': [[[0, 1, 2, 3, 73, 78, 79, 73]]], 'quarantined_sources': [1]}

    def test_unique_current_replay_only_and_no_quarantined_source(self):
        self.assertEqual(current_core_ids(self.fixture(), {0, 5, 6}), [0, 5, 6])
        data = self.fixture(); data['batches'][0][0][-1] = 83
        with self.assertRaises(ValueError):
            current_core_ids(data, {0, 5, 6, 10})

    def test_unknown_negative_or_noninteger_input_cannot_become_training_case(self):
        for index in (80, -1, 73.0):
            data = self.fixture(); data['batches'][0][0][-1] = index
            with self.assertRaises(ValueError):
                current_core_ids(data, {0, 5, 6})


if __name__ == '__main__':
    unittest.main()
