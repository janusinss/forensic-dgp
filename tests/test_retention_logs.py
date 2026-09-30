import unittest

from scripts.audit_retention_logs import audit_steps


class RetentionLogTests(unittest.TestCase):
    def fixture(self):
        run = dict(factors=[1., .5, .25, .125], tolerance=1e-6,
                   max_scheduled_batches=21, stop_rejection_streak=3,
                   preflight=False, ceilings={'covered': 1., 'clear': 1.})
        schedule = [list(range(i, i+8)) for i in range(21)]
        steps = [dict(step=i+1, indices=schedule[i], accepted=False, factor=None,
                      accepted_updates=0, rejection_streak=i+1,
                      attempts=[dict(factor=f, accepted=False, losses={'covered': 0., 'clear': 2.}) for f in run['factors']]) for i in range(3)]
        result = dict(complete=True, stop_reason='three_consecutive_rejected_batches',
                      scheduled_batches=3, accepted_updates=0, trials=12,
                      final_replay={'covered': 1., 'clear': 1.})
        return run, steps, result, schedule

    def test_expected_early_stop(self):
        result = audit_steps(*self.fixture())
        self.assertEqual(result['trials'], 12)
        self.assertEqual(result['warnings'], [])

    def test_invalid_policy_schedule_counter_and_final_loss(self):
        for mutate in [lambda r,s,o,q: r.update(tolerance=.1),
                       lambda r,s,o,q: s[0].update(indices=[99]),
                       lambda r,s,o,q: o.update(accepted_updates=1),
                       lambda r,s,o,q: o['final_replay'].update(clear=1.1),
                       lambda r,s,o,q: s[0]['attempts'][0].update(accepted=True)]:
            args = self.fixture()
            mutate(*args)
            with self.assertRaises(ValueError):
                audit_steps(*args)

    def test_successful_budget_allows_no_compensation_between_groups(self):
        run, _, result, schedule = self.fixture()
        steps = [dict(step=i+1, indices=schedule[i], accepted=True, factor=1.,
                      accepted_updates=i+1, rejection_streak=0,
                      attempts=[dict(factor=1., accepted=True, losses={'covered': .9, 'clear': .9})]) for i in range(21)]
        result.update(stop_reason='scheduled_budget_complete', scheduled_batches=21,
                      accepted_updates=21, trials=21)
        self.assertEqual(audit_steps(run, steps, result, schedule)['accepted_updates'], 21)
        steps[0]['attempts'][0]['losses'] = {'covered': 0., 'clear': 1.01}
        with self.assertRaises(ValueError):
            audit_steps(run, steps, result, schedule)


if __name__ == '__main__':
    unittest.main()
