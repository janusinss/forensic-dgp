"""Pure source/contract/gate regressions; no Torch, neural or gradient execution."""
import ast
from collections import Counter
import copy
import builtins
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/codeformer_u96_decoder_readout_v1/worker.py'
HISTORICAL = ROOT / 'research/codeformer_u96_postfusion_smoothanchors_lr_v1/worker.py'


def load_pure():
    spec = importlib.util.spec_from_file_location('inert_decoder_readout_worker', SOURCE)
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'numpy', 'PIL', 'third_party', 'app'):
            raise AssertionError('Framework/runtime import forbidden during pure checks: ' + name)
        return real_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=guarded), \
            patch('subprocess.run', side_effect=AssertionError('No runtime probe permitted')):
        spec.loader.exec_module(module)
    return module


def contract_fixture(worker):
    learning = copy.deepcopy(worker.LEARNING)
    learning['optimization_reference_order'] = ['opt_' + str(index) for index in range(16)]
    learning['monitor_reference_order'] = ['monitor_' + str(index) for index in range(8)]
    cases = []
    for cohort, key, permitted in (('optimize', 'optimization_reference_order', True),
                                   ('monitor', 'monitor_reference_order', False)):
        for reference in learning[key]:
            for role in learning['slot_roles']:
                cases.append({'id': reference + '__' + role, 'reference_id': reference,
                              'role': role, 'cohort': cohort, 'split': 'TRAIN',
                              'optimization_permitted': permitted})
    guards = [{'id': 'guard_' + str(index) + '__' + role, 'reference_id': 'guard_' + str(index),
               'role': role} for index in range(8) for role in ('clear', 'dark', 'filter')]
    return {'learning': learning, 'cases': cases, 'guards': guards}


def monitor_fixture(worker):
    result = {}
    for role in worker.LEARNING['slot_roles']:
        for index in range(8):
            key = role + '_' + str(index)
            result[key] = {'id': key, 'role': role,
                'regional_metrics': {view: {region: {'mae': .1, 'adjacent_rgb_error': 1.0}
                    for region in ('whole', 'central128', 'outside')} for view in ('raw', 'native_png')},
                'U96_behavior_distance': {'raw': 0., 'native_png': 0.},
                'U96_illumination_distance': {'raw': 0., 'native_png': 0.}}
    return result


class ReadoutWorkerPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = load_pure()
        cls.text = SOURCE.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.text)
        cls.old_tree = ast.parse(HISTORICAL.read_text(encoding='utf-8'))

    def node(self, name, tree=None):
        return next(node for node in ast.walk(tree or self.tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    def pure_gate(self, name, **bindings):
        node = copy.deepcopy(self.node(name))
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        namespace = {'math': math, 'need': self.worker.need,
                     'ordered_float_sum': self.worker.ordered_float_sum,
                     'LEARNING': self.worker.LEARNING, **bindings}
        exec(compile(module, str(SOURCE), 'exec'), namespace)
        return namespace[name]

    def test_import_is_framework_probe_and_execution_free(self):
        self.assertEqual(load_pure().OUTPUT_RELATIVE, 'outputs/u96_decoder_readout_v1')
        top_imports = [node for node in self.tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any('torch' in ast.unparse(node) or 'numpy' in ast.unparse(node)
                             for node in top_imports))
        self.assertFalse(any(isinstance(node, ast.If) and '__main__' in ast.unparse(node.test)
                             for node in self.tree.body))

    def test_finite_optimizer_and_existing_readout_contract(self):
        learning = self.worker.LEARNING
        self.assertEqual(learning['optimizer'], {'type': 'Adam', 'lr': 5e-5, 'betas': [.9, .99],
            'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False, 'foreach': False, 'fused': False,
            'capturable': False, 'differentiable': False, 'maximize': False})
        self.assertEqual((learning['max_updates'], learning['microbatches_per_update'],
                          learning['warmup_updates'], learning['gradient_clip_max_norm']), (64, 4, 4, 1.))
        self.assertEqual(learning['trainable_names'], ['base.generator.blocks.24.weight',
                                                      'base.generator.blocks.24.bias'])
        self.assertEqual((learning['trainable_parameters'], learning['frozen_tensors']), (1731, 537))
        self.assertEqual(learning['snapshot_updates'], [0, 1, 4, 16, 32, 64])
        self.assertEqual(learning['loss'], {'image': 1., 'adjacent': .25,
                                          'local_inside': .5, 'local_outside': .5})

    def test_full_forward_and_backward_budget_arithmetic(self):
        limits = self.worker.LIMITS
        self.assertEqual(limits['full_forward_attempts'], 120 + 6 * 56 + 64 * 4)
        self.assertEqual(limits['candidate_snapshot_forwards'], 336)
        self.assertEqual(limits['candidate_training_forwards'], 256)
        self.assertEqual((limits['backward_attempts'], limits['optimizer_update_attempts']), (256, 64))
        expected = next(node.value for node in ast.walk(self.tree) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'expected'
                                for target in node.targets) and isinstance(node.value, ast.Dict))
        counters = ast.literal_eval(expected)
        self.assertEqual(counters['full_forward_attempts'], 712)
        self.assertEqual(counters['frozen_U96_adapter_site_reservations'], 712 * 4)
        self.assertEqual(counters['existing_RGB_readout_reservations'], 336 + 256)
        self.assertEqual(counters['gradient_queries'], 0)
        self.assertEqual(counters['controlled_hook_exception_probes'], 1)

    def test_resources_match_prospective_hard_bounds(self):
        limits = self.worker.LIMITS
        self.assertEqual((limits['internal_seconds'], limits['external_seconds'],
                          limits['per_phase_seconds']), (1800, 1860, 180))
        self.assertEqual(limits['initial_free_disk_bytes'], 8 * 1024 ** 3)
        self.assertEqual(limits['continuing_disk_reserve'], 4 * 1024 ** 3)
        self.assertEqual(limits['maximum_RSS_bytes'], 10 * 1024 ** 3)
        self.assertEqual(limits['maximum_CUDA_reserved_bytes'], 18 * 1024 ** 3)
        self.assertEqual(limits['maximum_output_bytes'], 2 * 1024 ** 3)

    def test_sampler_is_four_exact_passes_with_no_role_or_photo_collision(self):
        exposures = Counter()
        for update in range(1, 65):
            indices = self.worker.schedule_indices(update)
            self.assertEqual(len(set(indices)), 4)
            for role, index in zip(self.worker.LEARNING['slot_roles'], indices):
                exposures[role, index] += 1
        self.assertEqual(len(exposures), 64)
        self.assertEqual(set(exposures.values()), {4})
        for bad in (0, 65, True, 1.0, '1'):
            with self.subTest(update=bad), self.assertRaises(ValueError):
                self.worker.schedule_indices(bad)

    def test_TRAIN_admission_rejects_changed_policy_overlap_or_cohort(self):
        original = contract_fixture(self.worker)
        self.worker.validate_learning_contract(original)
        mutations = [lambda p: p['learning']['optimizer'].__setitem__('lr', 1e-4),
                     lambda p: p['learning']['monitor_reference_order'].__setitem__(0, 'opt_0'),
                     lambda p: p['cases'][0].__setitem__('optimization_permitted', False),
                     lambda p: p['cases'][64].__setitem__('optimization_permitted', True),
                     lambda p: p['cases'][0].__setitem__('split', 'DEV'),
                     lambda p: p['guards'][0].__setitem__('reference_id', 'opt_0'),
                     lambda p: p['cases'].pop(),
                     lambda p: p['learning'].__setitem__('automatic_extra_epochs', 5)]
        for mutation in mutations:
            changed = copy.deepcopy(original); mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.worker.validate_learning_contract(changed)

    def test_early_trend_requires_one_blur_role_to_improve_both_views(self):
        baseline = monitor_fixture(self.worker)
        unchanged = copy.deepcopy(baseline)
        self.assertFalse(self.worker.early_trend_gate(unchanged, baseline, 16)['pass'])
        for row in unchanged.values():
            if row['role'] == 'global_gaussian_1p2':
                row['regional_metrics']['raw']['whole']['adjacent_rgb_error'] = .999
        self.assertFalse(self.worker.early_trend_gate(unchanged, baseline, 16)['pass'])
        for row in unchanged.values():
            if row['role'] == 'global_gaussian_1p2':
                row['regional_metrics']['native_png']['whole']['adjacent_rgb_error'] = .999
        report = self.worker.early_trend_gate(unchanged, baseline, 16)
        self.assertTrue(report['pass'])
        self.assertTrue(report['groups']['global_gaussian_1p2']['pass'])
        self.assertFalse(report['groups']['local_gaussian_1p2']['pass'])
        self.assertTrue(self.worker.early_trend_gate({}, {}, 4)['pass'])
        self.assertFalse(self.worker.early_trend_gate({}, {}, 4)['evaluated'])

    def test_early_trend_local_alternative_and_missing_nonfinite_rejection(self):
        baseline = monitor_fixture(self.worker); current = copy.deepcopy(baseline)
        for row in current.values():
            if row['role'] == 'local_gaussian_1p2':
                for view in ('raw', 'native_png'):
                    row['regional_metrics'][view]['central128']['adjacent_rgb_error'] = .999
        self.assertTrue(self.worker.early_trend_gate(current, baseline, 16)['pass'])
        first = 'local_gaussian_1p2_0'
        del current[first]
        with self.assertRaises(ValueError):
            self.worker.early_trend_gate(current, baseline, 16)
        current = copy.deepcopy(baseline)
        current[first]['regional_metrics']['raw']['central128']['adjacent_rgb_error'] = float('nan')
        with self.assertRaises(ValueError):
            self.worker.early_trend_gate(current, baseline, 16)

    def test_all_historical_gate_and_delivery_metric_formulas_remain_exact(self):
        for name in ('legacy_gate', 'monitor_gate', 'region_statistics', 'regions', 'illumination',
                     'ordered_float_sum'):
            self.assertEqual(ast.dump(self.node(name), include_attributes=False),
                             ast.dump(self.node(name, self.old_tree), include_attributes=False), name)

    def test_legacy_gate_strict_case_threshold_was_not_relaxed(self):
        guards = []
        for role in ('clear', 'dark', 'filter'):
            for index in range(8):
                key = role + str(index)
                guards.append({'id': key, 'role': role,
                    'behavior_distance': {'raw': 0., 'native_png': 0.},
                    'metrics': {view: {'mae': 0., 'adjacent_rgb_error': 0.}
                                for view in ('raw', 'native_png')}})
        gate = self.pure_gate('legacy_gate', p={'guards': guards})
        bank = {row['id']: copy.deepcopy(row) for row in guards}
        self.assertTrue(gate(bank)['pass'])
        bank['clear0']['behavior_distance']['raw'] = .03
        self.assertTrue(gate(bank)['pass'])
        bank['clear0']['behavior_distance']['raw'] = .03000000000001
        self.assertFalse(gate(bank)['pass'])
        self.assertFalse(gate(bank)['thresholds_reset_to_U96'])

    def test_terminal_monitor_keeps_all_four_one_percent_and_dark_endpoints(self):
        baseline = monitor_fixture(self.worker); current = copy.deepcopy(baseline)
        gate = self.pure_gate('monitor_gate', baseline=baseline)
        self.assertTrue(gate(current, 0)['pass'])
        self.assertFalse(gate(current, 64)['pass'])
        for row in current.values():
            if row['role'] in ('local_gaussian_1p2', 'global_gaussian_1p2'):
                region = 'central128' if row['role'] == 'local_gaussian_1p2' else 'whole'
                for view in ('raw', 'native_png'):
                    row['regional_metrics'][view][region]['adjacent_rgb_error'] = .9899
        self.assertTrue(gate(current, 64)['pass'])
        current['lowlight_linear_0p35_0']['regional_metrics']['raw']['whole']['mae'] += .000001
        self.assertFalse(gate(current, 64)['pass'])

    def test_lazy_runtime_import_follows_verified_VM_and_active_source_guards(self):
        run = self.node('run'); source = ast.unparse(run)
        self.assertLess(source.index('packet.verify_static(root, expected_sha)'),
                        source.index('packet.require_vm(root)'))
        self.assertLess(source.index('packet.require_vm(root)'), source.index('import torch as torch_module'))
        self.assertIn("active_path = Path(__file__).resolve()", source)
        self.assertIn("'Active loaded worker source differs from sealed source'", source)
        self.assertIn("'Active model/objective source bytes differ'", source)
        self.assertIn("apps.stdout.strip()", source)
        self.assertIn("'NVIDIA L4'", source)

    def test_safe_initializer_only_loading_without_old_Adam_resume(self):
        loads = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'load']
        torch_loads = [node for node in loads if isinstance(node.func.value, ast.Name)
                       and node.func.value.id == 'torch']
        self.assertEqual(len(torch_loads), 2)
        for node in torch_loads:
            options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
            self.assertEqual(options, {'map_location': 'cpu', 'weights_only': True, 'mmap': True})
        construction = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == 'Adam']
        self.assertEqual(len(construction), 1)
        self.assertIn("need(not optimizer.state", self.text)

    def test_candidate_snapshots_use_grad_mode_and_full_initial_parity(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.With) and any('no_grad' in ast.unparse(item.context_expr) for item in node.items):
                self.assertNotIn('full_call(candidate', ast.unparse(node))
                self.assertNotIn('candidate(sample)', ast.unparse(node))
        snapshot = ast.unparse(self.node('snapshot'))
        self.assertIn('with torch.enable_grad()', snapshot)
        self.assertIn("record['stock'] == baseline[row['id']]['stock']", snapshot)
        self.assertIn('np.array_equal', snapshot)
        self.assertIn("checked(record['outputs']['png']).read_bytes()", snapshot)
        self.assertIn('early_trend_gate(bank, baseline, update)', snapshot)
        self.assertIn("legacy['pass'] and monitor['pass'] and trend['pass']", snapshot)

    def test_optimization_target_is_clean_photograph_without_behavior_copy(self):
        objective = ast.unparse(self.node('objective_for'))
        self.assertIn("row['optimization_permitted'] is True", objective)
        self.assertIn("target is not None", objective)
        self.assertIn('clean_objective.connected_clean_objective(raw, target', objective)
        self.assertNotIn('baseline', objective)
        self.assertNotIn('teacher', objective)
        self.assertNotIn('smooth_l1', objective)
        deliver = ast.unparse(self.node('deliver'))
        self.assertIn('prototype.connected_rgb256(native, torch=torch, manual_vm_root=root)', deliver)
        self.assertNotIn('.detach(', deliver)

    def test_no_extra_gradient_queries_and_reservation_precedes_optimizer(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        self.assertEqual(len([node for node in calls if isinstance(node.func, ast.Attribute)
                              and node.func.attr == 'backward']), 1)
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and
                             ast.unparse(node.func).startswith('torch.autograd.') for node in calls))
        source = ast.unparse(self.node('run'))
        self.assertLess(source.index("event('optimizer_reserved', update=update)"), source.index('optimizer.step()'))
        self.assertLess(source.index('optimizer.step()'), source.index("event('optimizer_returned', update=update)"))
        self.assertIn('g is not None and g.dtype == torch.float32', source)
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == 'any' and
                            'torch.count_nonzero(g).item() > 0' in ast.unparse(node)
                            for node in calls))
        self.assertNotIn('all(torch.count_nonzero(g)', source)

    def test_stopped_state_and_every_committed_state_preserve_evidence(self):
        source = ast.unparse(self.node('run'))
        for field in ('partial_gradients', 'optimizer_outcome_unknown', 'sampler_completed_backwards',
                      'CPU_RNG', 'CUDA_RNG', 'Python_random_state', 'NumPy_random_state'):
            self.assertIn(repr(field), source)
        self.assertIn("out / 'failed_state.pt'", source)
        self.assertIn('checkpoint(update)', source)
        self.assertIn('postclip_gradients.npz', source)
        self.assertIn("not_repaired_by_this_TRAIN_screen", source)
        self.assertIn('random.setstate(caller_rng_backup', source)
        self.assertIn('np.random.set_state(caller_rng_backup', source)
        self.assertIn('torch.cuda.set_rng_state_all(caller_rng_backup', source)
        self.assertIn('raise SystemExit(2)', source)

    def test_Linux_RSS_scalar_parser_refuses_ambiguous_data(self):
        self.assertEqual(self.worker.linux_current_rss_bytes('100 25 0 0 0', 4096), 102400)
        for text, page in (('100 unknown', 4096), ('100', 4096), ('100 25', 0), (None, 4096)):
            with self.subTest(text=text, page=page), self.assertRaises(ValueError):
                self.worker.linux_current_rss_bytes(text, page)


if __name__ == '__main__':
    unittest.main()
