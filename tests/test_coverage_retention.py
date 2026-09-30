import copy
import unittest

import torch

from coverage_retention import guarded_step


class RetentionTests(unittest.TestCase):
    def setup_case(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        opt = torch.optim.AdamW(model.parameters(), lr=.1)
        model.weight.grad = torch.ones_like(model.weight)
        # Populate AdamW moments using only this scalar test fixture.
        opt.step()
        model.weight.grad = torch.ones_like(model.weight)
        return model, opt

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_rejection_restores_parameters_moments_gradients_modes_rng(self):
        model, opt = self.setup_case()
        model.register_buffer('counter', torch.tensor(0.))
        before = copy.deepcopy(model.state_dict())
        optim = copy.deepcopy(opt.state_dict())
        rng = torch.get_rng_state()
        def evaluate():
            model.counter.add_(1)
            torch.rand(3)
            return {'covered': .5, 'clear': 2.}
        result = guarded_step(model, opt, evaluate, {'covered': 1., 'clear': 1.})
        self.assertFalse(result['accepted'])
        self.assertEqual(len(result['attempts']), 4)
        self.assert_nested_equal(before, model.state_dict())
        self.assert_nested_equal(optim, opt.state_dict())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.weight.grad, torch.ones_like(model.weight)))

    def test_backtracking_matches_one_direct_half_lr_update(self):
        model, opt = self.setup_case()
        reference = copy.deepcopy(model)
        reference_opt = torch.optim.AdamW(reference.parameters(), lr=.1)
        reference_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        reference.weight.grad = model.weight.grad.clone()
        reference_opt.param_groups[0]['lr'] = .05
        reference_opt.step()
        reference_opt.param_groups[0]['lr'] = .1
        before = model.weight.detach().clone()
        def evaluate():
            movement = float((model.weight - before).abs().max())
            return {'covered': movement, 'clear': movement}
        result = guarded_step(model, opt, evaluate, {'covered': .075, 'clear': .075})
        self.assertEqual(result['factor'], .5)
        self.assertEqual(len(result['attempts']), 2)
        self.assert_nested_equal(model.state_dict(), reference.state_dict())
        self.assert_nested_equal(opt.state_dict(), reference_opt.state_dict())

    def test_exception_rolls_back(self):
        model, opt = self.setup_case()
        before, optim = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
        def evaluate():
            raise RuntimeError('failed replay read')
        with self.assertRaisesRegex(RuntimeError, 'failed replay read'):
            guarded_step(model, opt, evaluate, {'covered': 1., 'clear': 1.})
        self.assert_nested_equal(before, model.state_dict())
        self.assert_nested_equal(optim, opt.state_dict())

    def test_nonfinite_loss_rejected_and_separate_groups_required(self):
        for losses in ({'covered': float('nan'), 'clear': 0.},
                       {'covered': 0., 'clear': 1.1}):
            model, opt = self.setup_case()
            result = guarded_step(model, opt, lambda: losses, {'covered': 1., 'clear': 1.})
            self.assertFalse(result['accepted'])
        model, opt = self.setup_case()
        with self.assertRaises(ValueError):
            guarded_step(model, opt, lambda: {'covered': 0.}, {'covered': 1., 'clear': 1.})

    def test_invalid_configuration_rejected_before_step(self):
        model, opt = self.setup_case()
        for factors in ((), (1., 1.), (0.,), (float('nan'),)):
            with self.assertRaises(ValueError):
                guarded_step(model, opt, lambda: {}, {'covered': 1., 'clear': 1.}, factors=factors)


if __name__ == '__main__':
    unittest.main()
