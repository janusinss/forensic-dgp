import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.train_retention_vm import main, replay_weights, replay_losses, next_rejection_streak
from detector_training import segmentation_loss


class RetentionVMTests(unittest.TestCase):
    def test_vm_guard_precedes_load(self):
        with patch('scripts.train_retention_vm.require_vm_gpu', side_effect=RuntimeError('VM required')), patch('scripts.train_retention_vm.load_completion') as load:
            with self.assertRaisesRegex(RuntimeError, 'VM required'):
                main(SimpleNamespace())
            load.assert_not_called()

    def test_frozen_schedule_membership_and_weights(self):
        protocol = json.loads(Path('outputs/coverage_protocol_v1/protocol.json').read_text())
        ids = {i-73 for e in protocol['schedules']['extended']['batches'] for b in e for i in b if i >= 73}
        cache = {i: (None, torch.tensor([int(i % 5 != 0)])) for i in ids}
        weights = replay_weights(protocol, cache)
        self.assertEqual(sum(weights.values()), 840)
        self.assertEqual(len(weights), 638)
        del cache[next(iter(cache))]
        with self.assertRaises(ValueError):
            replay_weights(protocol, cache)

    def test_weighted_losses_keep_clear_and_covered_separate(self):
        class Fake:
            segmenter = torch.nn.Identity()
            def detect(self, x):
                return x[:, :1]
        cache = {}
        for i, value in enumerate((0, 127, 255)):
            target = torch.zeros(1, 2, 2)
            if i:
                target[0, 0, 0] = 1
            cache[i] = (torch.full((3, 2, 2), value, dtype=torch.uint8), target)
        weights = {0: 2, 1: 1, 2: 3}
        result = replay_losses(Fake(), cache, weights, 'cpu')
        direct = {i: float(segmentation_loss(x[:1][None].float()/255, m[None], .25, .1)) for i, (x, m) in cache.items()}
        self.assertAlmostEqual(result['clear'], direct[0])
        self.assertAlmostEqual(result['covered'], (direct[1]+3*direct[2])/4)

    def test_streak_resets_on_acceptance(self):
        streak = 0
        for accepted, expected in [(False, 1), (False, 2), (True, 0), (False, 1), (False, 2), (False, 3)]:
            streak = next_rejection_streak(streak, accepted)
            self.assertEqual(streak, expected)


if __name__ == '__main__':
    unittest.main()
