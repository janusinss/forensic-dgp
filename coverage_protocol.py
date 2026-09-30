"""Independent random streams for a matched dataset-coverage experiment."""
import numpy as np


class CoverageBatches:
    """Four balanced groups; real group lengths cannot perturb replay order."""
    def __init__(self, groups, batch_size, steps, seed):
        if len(groups) != 4 or not all(groups) or batch_size < 4 or batch_size % 4 or steps < 1:
            raise ValueError('Four nonempty groups, batch divisible by four, positive steps required')
        flat = [i for group in groups for i in group]
        if len(set(flat)) != len(flat):
            raise ValueError('Groups must contain unique, disjoint indices')
        self.groups = groups
        self.quarter = batch_size // 4
        self.steps, self.seed, self.epoch = steps, seed, 0

    def __len__(self):
        return self.steps

    def __iter__(self):
        streams = []
        needed = self.steps * self.quarter
        for group_id, group in enumerate(self.groups):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, group_id]))
            stream = []
            while len(stream) < needed:
                stream.extend(rng.permutation(group).tolist())
            streams.append(stream[:needed])
        order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, 4]))
        for step in range(self.steps):
            start = step * self.quarter
            batch = [i for stream in streams for i in stream[start:start + self.quarter]]
            yield order.permutation(batch).tolist()
