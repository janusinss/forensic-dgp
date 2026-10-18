"""Small CPU scalar/metadata fixtures; no network, model or training execution."""
from copy import deepcopy
import math
import unittest

from research.codeformer_path_diagnostic_v1.return_state import (
    COSINE_PAIRS, GROUP_COUNTS, WEIGHTS, audit_cosines, audit_gradient, audit_family_norms, audit_execution,
    audit_stages, audit_trace, expected_calls, expected_stage_counts,
    weighted_scalar,
)


def cases():
    refs = ('a', 'a', 'b', 'b', 'c', 'c', 'd', 'd')
    return [{'id': f'train_u{u:04d}_slot{s}', 'update': u, 'slot': s,
             'reference_id': ref, 'image_eligible': u == 24 or s >= 3,
             'behavior': {} if s == 3 else None}
            for (u, s), ref in zip(((u, s) for u in (1, 24) for s in (1, 2, 3, 4)), refs)]


CAPS = {'codeformer_forwards': 104, 'hq_teacher_forwards': 4,
        'lpips_forwards': 10, 'gradient_queries': 64, 'optimizer_updates': 0}


def trace():
    counts = {k: 0 for k in CAPS}
    result = []
    for number, (name, context) in enumerate(expected_calls(cases()), 1):
        counts[name] += 1
        result.append({'event': number, 'before_call': name, 'context': context,
                       'counts': dict(counts), 'elapsed_seconds': number / 10})
    return result


def component():
    return {'state': 'original', 'case_id': cases()[0]['id'], 'term': 'coarse_rgb_l1',
            'weight': 4., 'accumulation_divisor': 4, 'raw_loss': .125,
            'weighted_loss_divided_by_four': .125, 'zero_optimizer_updates': True,
            'all_groups_nonzero_not_required': True,
            'groups': {group: {'parameters': count, 'absent_tensors': count,
                       'zero_tensors': 0, 'nonzero_tensors': 0, 'norm': 0., 'maximum_absolute': 0.}
                       for group, count in GROUP_COUNTS.items()}}


def cosines():
    return {'state': 'original', 'predicted_training_quality_not_implied': True,
            'comparisons': [{'a': list(a), 'b': list(b), 'groups': {
                group: {'dot': 6., 'norm_a': 2., 'norm_b': 3., 'cosine': 1.}
                for group in GROUP_COUNTS}} for a, b in COSINE_PAIRS]}


class ReturnStateTests(unittest.TestCase):
    def test_autocast_softmax_fp32_with_bf16_logits(self):
        row = {'logits_dtype': 'torch.bfloat16', 'native_probability_dtype': 'torch.float32',
            'execution': {'candidate_training': True, 'quantizer_training': False,
                'generator_training': False, 'grad_enabled': True, 'inference_mode': False}}
        audit_execution(row, 'TBF16')
        row['native_probability_dtype'] = 'torch.bfloat16'
        with self.assertRaises(ValueError): audit_execution(row, 'TBF16')

    def test_exact_trace_census_and_order(self):
        result, times = audit_trace(trace(), expected_calls(cases()), CAPS, 900)
        self.assertEqual(result['counts'], CAPS)
        self.assertEqual(result['events'], 182)
        self.assertEqual(len(times), 183)

    def test_no_optimizer_reservation(self):
        rows = trace(); rows[0]['before_call'] = 'optimizer_updates'
        with self.assertRaises(ValueError): audit_trace(rows, expected_calls(cases()), CAPS, 900)

    def test_exact_context_and_cumulative_counters(self):
        rows = trace(); rows[10]['counts']['gradient_queries'] += 1
        with self.assertRaises(ValueError): audit_trace(rows, expected_calls(cases()), CAPS, 900)
        rows = trace(); rows[10]['context'] += '/wrong'
        with self.assertRaises(ValueError): audit_trace(rows, expected_calls(cases()), CAPS, 900)

    def test_nonfinite_and_backward_clock_refused(self):
        for value in (math.nan, -1, 901):
            rows = trace(); rows[1]['elapsed_seconds'] = value
            with self.assertRaises(ValueError): audit_trace(rows, expected_calls(cases()), CAPS, 900)

    def test_stage_fingerprints_and_grad_fields(self):
        _, times = audit_trace(trace(), expected_calls(cases()), CAPS, 900)
        states = {key: 'a' * 64 for key in ('original', 'stopped', 'HQ_teacher', 'perceptual')}
        rows = [{'stage': name, 'counts': counts, 'states': states.copy(),
                 'parameter_grad_fields_all_none': True, 'elapsed_seconds': 20 + i}
                for i, (name, counts) in enumerate(expected_stage_counts())]
        self.assertEqual(audit_stages(rows, states, times, 900)['stage_count'], 13)
        rows[-1]['states']['stopped'] = 'b' * 64
        with self.assertRaises(ValueError): audit_stages(rows, states, times, 900)
        rows[-1]['states'] = states.copy(); rows[-1]['parameter_grad_fields_all_none'] = False
        with self.assertRaises(ValueError): audit_stages(rows, states, times, 900)

    def test_float32_weighting_and_wrong_weights_refused(self):
        # 0.125 is exact binary; float32(0.1)/32 has this exact stored value.
        self.assertEqual(weighted_scalar(.125, .1), .0031250000465661287)
        row = component()
        self.assertIs(audit_gradient(row, 'original', cases()[0], 'coarse_rgb_l1'), row)
        row['weighted_loss_divided_by_four'] += .001
        with self.assertRaises(ValueError): audit_gradient(row, 'original', cases()[0], 'coarse_rgb_l1')

    def test_absent_gradients_are_not_failures(self):
        row = component()
        audit_gradient(row, 'original', cases()[0], 'coarse_rgb_l1')

    def test_invalid_gradient_partition_or_norm_refused(self):
        for key, value in (('absent_tensors', 0), ('norm', 1), ('maximum_absolute', 1)):
            row = component(); row['groups']['encoder'][key] = value
            with self.assertRaises(ValueError): audit_gradient(row, 'original', cases()[0], 'coarse_rgb_l1')

    def test_cosine_sufficient_scalar_formula(self):
        row = cosines()
        self.assertEqual(len(audit_cosines(row, 'original')), 5)
        row['comparisons'][0]['groups']['encoder']['cosine'] = .5
        with self.assertRaises(ValueError): audit_cosines(row, 'original')

    def test_zero_norm_requires_null_cosine(self):
        row = cosines(); group = row['comparisons'][0]['groups']['encoder']
        group.update(dot=0., norm_a=0., cosine=None)
        audit_cosines(row, 'original')
        group['cosine'] = 0.
        with self.assertRaises(ValueError): audit_cosines(row, 'original')

    def test_impossible_cosine_refused(self):
        row = cosines(); row['comparisons'][0]['groups']['encoder']['dot'] = 10.
        with self.assertRaises(ValueError): audit_cosines(row, 'original')

    def test_gradient_identity_and_qualification_claim_refused(self):
        row = component(); row['zero_optimizer_updates'] = False
        with self.assertRaises(ValueError): audit_gradient(row, 'original', cases()[0], 'coarse_rgb_l1')
        row = cosines(); row['predicted_training_quality_not_implied'] = False
        with self.assertRaises(ValueError): audit_cosines(row, 'original')

    def test_sum_gradient_triangle_bounds(self):
        row = cosines()['comparisons'][0]
        parts = []
        for slot in (1, 2):
            for term in ('coarse_rgb_l1', 'pixel_l1_rgb'):
                item = component(); item['case_id'] = f'train_u0024_slot{slot}'; item['term'] = term
                for group in item['groups'].values(): group['norm'] = 1.5
                parts.append(item)
        self.assertEqual(audit_family_norms([row], parts), 12)
        row['groups']['encoder']['norm_a'] = 3.1
        with self.assertRaises(ValueError): audit_family_norms([row], parts)


if __name__ == '__main__':
    unittest.main()
