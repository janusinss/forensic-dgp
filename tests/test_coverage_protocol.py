import unittest
from coverage_protocol import CoverageBatches


class CoverageProtocolTests(unittest.TestCase):
    def test_replay_order_and_positions_match_across_real_pool_sizes(self):
        # Real index offsets differ, but replay case IDs must remain identical.
        a = [list(range(43)), list(range(43, 68)), list(range(68, 148)), list(range(148, 168))]
        b = [list(range(47)), list(range(47, 73)), list(range(73, 153)), list(range(153, 173))]
        x, y = CoverageBatches(a, 8, 21, 42), CoverageBatches(b, 8, 21, 42)
        for epoch in range(10):
            x.epoch = y.epoch = epoch
            for left, right in zip(x, y):
                self.assertEqual([(p, i-68) for p,i in enumerate(left) if i >= 68],
                                 [(p, i-73) for p,i in enumerate(right) if i >= 73])
                for batch, groups in ((left,a), (right,b)):
                    self.assertEqual([sum(i in g for i in batch) for g in groups], [2]*4)

    def test_reproducible_and_epoch_specific(self):
        x = CoverageBatches([list(range(i*10,(i+1)*10)) for i in range(4)], 8, 5, 42)
        before = list(x)
        self.assertEqual(before, list(x))
        x.epoch = 1
        self.assertNotEqual(before, list(x))

    def test_invalid_or_overlapping_groups_rejected(self):
        for groups, size in (([[0],[1],[2],[]],8), ([[0],[1],[2],[3]],6), ([[0],[1],[2],[0]],8)):
            with self.assertRaises(ValueError):
                CoverageBatches(groups,size,21,42)
