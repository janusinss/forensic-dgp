"""Prospective protocol mutations; standard library only, no neural imports."""
import copy
from pathlib import Path
import tempfile
import unittest

from research.codeformer_loss_response_v1 import contract as c

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/cctv_codeformer_loss_response_v1_vm'


class ContractBoundaries(unittest.TestCase):
    def setUp(self):
        self.p = c.read_json(STAGE / 'protocol.json')
        self.cases = c.read_json(STAGE / 'cases.json')

    def test_prior_frozen_protocol_and_cases_remain_valid(self):
        c.verify_protocol(self.p)
        c.verify_cases(self.cases)

    def test_no_promotion_or_automatic_execution(self):
        for key, value in [('automatic_app_promotion', True), ('model_qualified', True),
                           ('new_training_recipe', True), ('export_candidate_checkpoint', True),
                           ('manual_tmux_required', False), ('all_branches_disposable', False),
                           ('quality_gates_changed', True), ('status', 'COMPLETE')]:
            p = copy.deepcopy(self.p)
            p[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.verify_protocol(p)

    def test_rates_caps_losses_and_selection_cannot_be_enlarged(self):
        for key, child in [('optimizer', 'gradient_accumulation_denominator'),
                           ('limits', 'worker_seconds'), ('call_caps', 'disposable_optimizer_steps'),
                           ('loss_weights', 'token_ce')]:
            p = copy.deepcopy(self.p)
            p[key][child] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.verify_protocol(p)
        p = copy.deepcopy(self.p)
        p['source_updates'] = [1, 6, 15, 25]
        with self.assertRaises(ValueError):
            c.verify_protocol(p)

    def test_holdout_roles_paths_and_case_order_rejected(self):
        for field, value in [('reference_id', 'va_ffhq_00001'),
                             ('input_path', 'data/sources/va_ffhq_00001.png'),
                             ('hq_features_path', '../features.npy')]:
            cases = copy.deepcopy(self.cases)
            cases['cases'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.verify_cases(cases)
        cases = copy.deepcopy(self.cases)
        cases['cases'][0]['record']['role'] = 'final'
        with self.assertRaises(ValueError):
            c.verify_cases(cases)
        cases = copy.deepcopy(self.cases)
        cases['cases'].reverse()
        with self.assertRaises(ValueError):
            c.verify_cases(cases)

    def test_missing_behavior_target_cannot_silently_change_support(self):
        cases = copy.deepcopy(self.cases)
        del cases['cases'][2]['behavior_delivered_path']
        with self.assertRaises(ValueError):
            c.verify_cases(cases)

    def test_unsafe_paths_and_duplicate_nonfinite_JSON_rejected(self):
        for name in ('../a', '/a', 'a\\b', 'a:b', 'a//b', 'AUX', 'a/CON.txt', 'a. ', 'a/./b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                c.safe_relative(name)
        scratch = ROOT / 'scratch'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='loss-contract-', dir=scratch) as tmp:
            path = Path(tmp) / 'fixture.json'
            for payload in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(payload, encoding='utf-8')
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    c.read_json(path)


if __name__ == '__main__':
    unittest.main()
