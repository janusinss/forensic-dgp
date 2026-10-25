"""NumPy supplied-pixel fixtures; no model, neural framework or worker calls."""
import ast
import copy
import hashlib
import importlib.abc
import inspect
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


FRAMEWORK_ROOTS = frozenset(('torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'jaxlib', 'keras'))


class FrameworkImportBlocker(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FRAMEWORK_ROOTS | {'PIL'} or fullname.endswith(('.worker', '.model')):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden record-test import: ' + fullname)
        return None


BLOCKER = FrameworkImportBlocker()
sys.meta_path.insert(0, BLOCKER)
import numpy as np

from research.codeformer_matched_feature_learning_v1 import contract as C, objective, quality, records as R


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm/protocol.json'


def scoped(row, bank):
    result = copy.deepcopy(row)
    result.update(bank=bank, case_key=bank + ':' + row['id'])
    return result


class NewPixelRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = C.select_cases((PREP / 'data_admission_v1.json').read_bytes(),
            (PREP / 'candidate_data_v1.json').read_bytes(), PARENT.read_bytes())

    def setUp(self):
        self.row = scoped(self.selection['new_monitors'][0], 'new_monitors')
        self.raw = np.full((256, 256, 3), .5, dtype=np.float32)
        self.png = np.full((256, 256, 3), 127, dtype=np.uint8)
        self.target = np.full((256, 256, 3), .25, dtype=np.float32)

    def calculate(self, raw=None, png=None, target=None, row=None):
        return R.record_new(self.raw if raw is None else raw, self.png if png is None else png,
            self.target if target is None else target, self.row if row is None else row, np=np)

    def test_all160_real_selected_rows_have_correct_scope_and_unweighted_fields(self):
        for bank in ('new_optimization', 'new_monitors'):
            for source in self.selection[bank]:
                row = scoped(source, bank)
                result = self.calculate(row=row)
                with self.subTest(case=row['case_key']):
                    self.assertEqual(result['bank'], bank)
                    self.assertEqual(result['case_key'], row['case_key'])
                    self.assertEqual(result['input'], row['input'])
                    self.assertEqual(result['target'], row['target'])
                    self.assertEqual(result['metrics']['raw'], {
                        'mae': .25, 'adjacent_rgb_error': 0.0, 'coarse_rgb_mae': .25})
                    self.assertEqual(result['split'], 'TRAIN')
                    self.assertEqual(result['optimization_permitted'], bank == 'new_optimization')
                    self.assertEqual(tuple(result['metrics']), R.VIEWS)
                    self.assertTrue(all(tuple(fields) == R.FIELDS for fields in result['metrics'].values()))
                    self.assertEqual(result['metric_region'], 'whole256')
                    self.assertFalse(result['model_qualified'])

    def test_constant_offset_exact_mapping_and_FP32_PNG_normalization(self):
        result = self.calculate()
        for view, pixels in (('raw', self.raw),
                             ('native_png', self.png.astype(np.float32) / np.float32(255))):
            _, terms = objective.objective_numpy(pixels, self.target, self.row['role'], np=np)
            self.assertEqual(result['metrics'][view], {
                field: terms[name] for name, field in R.TERM_TO_FIELD})
        expected_png = float(np.float32(127) / np.float32(255)) - .25
        self.assertEqual(result['metrics']['native_png'], {
            'mae': expected_png, 'adjacent_rgb_error': 0.0, 'coarse_rgb_mae': expected_png})
        self.assertNotEqual(result['metrics']['raw']['mae'], result['metrics']['native_png']['mae'])
        self.assertNotEqual(result['metrics']['native_png']['mae'], 127 / 255 - .25)

    def test_nonuniform_equal_arrays_zero_and_no_generated_target(self):
        png = np.random.default_rng(502).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        photo = png.astype(np.float32) / np.float32(255)
        result = self.calculate(raw=photo.copy(), png=png, target=photo)
        self.assertEqual(result['metrics'], {
            view: dict.fromkeys(R.FIELDS, 0.0) for view in R.VIEWS})
        self.assertEqual(result['target'], self.row['target'])
        self.assertTrue(result['supplied_pixel_metrics_only'])
        self.assertFalse(result['file_bytes_or_pixel_provenance_verified_here'])
        self.assertTrue(result['inherited_complete_bank_gates_required'])

    def test_checkerboard_analytic_whole_RGB_adjacent_and_cancelled_block_means(self):
        pattern = (np.indices((256, 256)).sum(axis=0) % 2).astype(np.float32)
        raw = np.repeat(pattern[..., None], 3, axis=2)
        png = (raw * 255).astype(np.uint8)
        target = np.full_like(raw, .5)
        result = self.calculate(raw, png, target)
        for view in R.VIEWS:
            self.assertEqual(result['metrics'][view], {
                'mae': .5, 'adjacent_rgb_error': 1.0, 'coarse_rgb_mae': 0.0})

    def test_internal_impulse_analytic_edges_and_block16_mean(self):
        raw = np.zeros_like(self.raw); raw[15, 15, :] = 1.0
        png = (raw * 255).astype(np.uint8)
        result = self.calculate(raw, png, np.zeros_like(raw))
        for view in R.VIEWS:
            self.assertEqual(result['metrics'][view]['mae'], 1 / (256 * 256))
            self.assertEqual(result['metrics'][view]['adjacent_rgb_error'], 4 / (2 * 256 * 255))
            self.assertEqual(result['metrics'][view]['coarse_rgb_mae'], 1 / (256 * 256))

    def test_metrics_use_RGB_channels_when_signed_color_changes_cancel_RGB_channel_mean(self):
        raw = np.full_like(self.raw, .5)
        target = raw.copy()
        target[:, :, 0] = .75
        target[:, :, 1] = .25
        result = self.calculate(raw=raw, target=target)
        self.assertEqual(result['metrics']['raw']['mae'], 1 / 6)
        self.assertEqual(result['metrics']['raw']['coarse_rgb_mae'], 1 / 6)
        self.assertEqual(result['metrics']['raw']['adjacent_rgb_error'], 0.0)

    def test_local_role_is_whole256_not_only_central_region(self):
        row = scoped(next(r for r in self.selection['new_monitors']
                          if r['role'] == 'local_gaussian2p0_soft'), 'new_monitors')
        raw = np.zeros_like(self.raw); raw[:64] = 1.0
        result = self.calculate(raw=raw, png=(raw * 255).astype(np.uint8),
                                target=np.zeros_like(raw), row=row)
        self.assertEqual(result['metrics']['raw']['mae'], .25)
        self.assertEqual(result['metrics']['raw']['coarse_rgb_mae'], .25)
        self.assertGreater(result['metrics']['raw']['adjacent_rgb_error'], 0.0)

    def test_random_float32_fixture_matches_independent_saved_array_reference(self):
        rng = np.random.default_rng(603)
        raw = rng.random((256, 256, 3), dtype=np.float32)
        target = rng.random((256, 256, 3), dtype=np.float32)
        png = (raw * 255).astype(np.uint8)
        result = self.calculate(raw, png, target)
        for view, pixels in (('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255))):
            residual = pixels.astype(np.float64) - target.astype(np.float64)
            horizontal = np.abs(residual[:, 1:] - residual[:, :-1])
            vertical = np.abs(residual[1:] - residual[:-1])
            expected_adjacent = (horizontal.sum() + vertical.sum()) / (horizontal.size + vertical.size)
            expected_coarse = []
            for y in range(0, 256, 16):
                for x in range(0, 256, 16):
                    expected_coarse.extend(np.abs(residual[y:y+16, x:x+16].mean(axis=(0, 1))))
            self.assertEqual(result['metrics'][view]['mae'], float(np.abs(residual).mean()))
            self.assertAlmostEqual(result['metrics'][view]['adjacent_rgb_error'], expected_adjacent, places=15)
            self.assertAlmostEqual(result['metrics'][view]['coarse_rgb_mae'], np.asarray(expected_coarse).mean(), places=15)

    def test_png_is_measured_independently_and_no_floor_or_consistency_is_fabricated(self):
        result = self.calculate(png=np.full_like(self.png, 255))
        self.assertEqual(result['metrics']['raw']['mae'], .25)
        self.assertEqual(result['metrics']['native_png']['mae'], .75)
        self.assertFalse(result['raw_png_delivery_consistency_verified_here'])

    def test_raw_target_require_exact_FP32_HWC_RGB256(self):
        bad_arrays = [self.raw.astype(np.float64), self.raw.astype(np.float16),
            self.raw.astype(np.uint8), self.raw[None], self.raw.transpose(2, 0, 1),
            self.raw[:, :, :1], self.raw[:-1], self.raw.tolist(), None]
        for value in bad_arrays:
            for position in (0, 2):
                args = [self.raw, self.png, self.target, self.row]
                args[position] = value
                with self.subTest(position=position, shape=getattr(value, 'shape', None)), self.assertRaisesRegex(
                        ValueError, 'Finite FP32 HWC RGB256'):
                    R.record_new(*args, np=np)

    def test_every_raw_and_target_value_is_finite_and_in_range_without_clipping(self):
        for bad in (np.nan, np.inf, -np.inf, np.float32(-1e-8), np.nextafter(np.float32(1), np.float32(2))):
            for position in (0, 2):
                args = [self.raw, self.png, self.target, self.row]
                args[position] = args[position].copy()
                args[position][-1, -1, -1] = bad
                with self.subTest(position=position, bad=bad), self.assertRaisesRegex(
                        ValueError, 'Finite FP32 HWC RGB256'):
                    R.record_new(*args, np=np)

    def test_png_requires_actual_decoded_RGB8_not_float_or_encoded_bytes(self):
        for png in (self.png.astype(np.float32), self.png.astype(np.int16), self.png[None],
                    self.png.transpose(2, 0, 1), self.png[:, :, :1], self.png[:-1],
                    self.png.tolist(), self.png.tobytes(), None):
            with self.subTest(kind=type(png).__name__), self.assertRaisesRegex(ValueError, 'Exact decoded RGB8'):
                R.record_new(self.raw, png, self.target, self.row, np=np)

    def test_bad_scope_bank_case_key_or_split_cannot_enter_new_records(self):
        for field, value in (('bank', 'old_quality'), ('bank', 'unadmitted'),
                             ('case_key', 'old_quality:' + self.row['id']), ('case_key', self.row['id']),
                             ('split', 'DEV'), ('split', 'FINAL'), ('id', 'unknown')):
            row = copy.deepcopy(self.row); row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.calculate(row=row)
        for field in ('bank', 'case_key', 'id', 'reference_id', 'role', 'split', 'target'):
            row = copy.deepcopy(self.row); del row[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.calculate(row=row)

    def test_fixed_reference_partition_role_permissions_and_teacher_are_enforced(self):
        for field, value in (('reference_id', C.OPTIMIZATION_REFERENCES[0]),
                             ('reference_id', 'unadmitted'), ('role', 'global_gaussian_1p2'),
                             ('cohort', 'optimize'), ('cohort', 'adaptive_monitor'),
                             ('optimization_permitted', True), ('optimization_permitted', 0),
                             ('behavior_teacher', {'raw': 'forbidden'}), ('target', None)):
            row = copy.deepcopy(self.row); row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.calculate(row=row)
        row = scoped(self.selection['new_optimization'][0], 'new_optimization')
        row['optimization_permitted'] = False
        with self.assertRaises(ValueError):
            self.calculate(row=row)

    def test_exact_canonical_input_target_pins_and_stage_paths(self):
        for key in ('input', 'target'):
            for replacement in (None, {}, 'unbound', {**self.row[key], 'extra': True},
                                {**self.row[key], 'bytes': True}, {**self.row[key], 'bytes': 0},
                                {**self.row[key], 'sha256': 'A' * 64},
                                {**self.row[key], 'sha256': 'g' * 64},
                                {**self.row[key], 'sha256': '0' * 63}):
                row = copy.deepcopy(self.row); row[key] = replacement
                with self.subTest(key=key, replacement=replacement), self.assertRaises(ValueError):
                    self.calculate(row=row)
            for path in ('/escape.png', '../escape.png', 'data/../escape.png',
                         'data//escape.png', './data/escape.png', 'C:/escape.png',
                         'data\\escape.png', 'assets/unadmitted.png', 'data/wrong.png'):
                row = copy.deepcopy(self.row); row[key]['path'] = path
                with self.subTest(key=key, path=path), self.assertRaises(ValueError):
                    self.calculate(row=row)

    def test_readonly_noncontiguous_arrays_supported_without_mutation_or_aliases(self):
        raw = self.raw[::-1]; target = self.target[:, ::-1]; png = self.png[::-1, ::-1]
        for value in (raw, target, png): value.flags.writeable = False
        before = [value.tobytes() for value in (raw, target, png)]
        row_before = copy.deepcopy(self.row)
        result = self.calculate(raw, png, target)
        self.assertEqual([value.tobytes() for value in (raw, target, png)], before)
        self.assertEqual(self.row, row_before)
        result['input']['sha256'] = '0' * 64
        result['target']['bytes'] = 1
        result['metrics']['raw']['mae'] = 0.0
        self.assertEqual(self.row, row_before)
        self.assertEqual([value.tobytes() for value in (raw, target, png)], before)

    def test_new40_records_are_accepted_by_existing_additional_bank_evaluator(self):
        bank = {row['id']: self.calculate(row=scoped(row, 'new_monitors'))
                for row in self.selection['new_monitors']}
        result = quality.evaluate_new_bank(bank, copy.deepcopy(bank), self.selection, update=0)
        self.assertTrue(result['pass'])
        self.assertEqual(result['complete_new_monitor_cases'], 40)
        self.assertFalse(result['full_quality_pass_established'])

    def test_record_new_is_array_arithmetic_and_does_not_authenticate_or_import_old_recorder(self):
        with mock.patch.object(quality, 'inherited_metrics', side_effect=AssertionError('Old helper called')):
            self.assertEqual(self.calculate()['metrics']['raw']['mae'], .25)


class InheritedPixelRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads(PARENT.read_bytes())
        cls.module = quality.inherited_metrics()

    def setUp(self):
        self.raw = np.full((256, 256, 3), .5, dtype=np.float32)
        self.png = np.full((256, 256, 3), 127, dtype=np.uint8)
        self.target = np.full_like(self.raw, .25)
        self.baseline = np.full_like(self.raw, .375)
        self.baseline_png = np.full_like(self.png, 95)
        self.row = scoped(next(row for row in self.parent['cases']
                               if row['cohort'] == 'monitor' and row['role'] == 'clear'), 'old_quality')

    def calculate(self, *, row=None, target='default', **changes):
        kwargs = {'raw': self.raw, 'png': self.png,
            'target': self.target if type(target) is str and target == 'default' else target,
            'row': self.row if row is None else row, 'np': np,
            'frozen_U96_raw': self.baseline, 'frozen_U96_png': self.baseline_png}
        kwargs.update(changes)
        return R.record_inherited(**kwargs)

    def test_old_recorder_source_and_dependencies_are_exact_before_call(self):
        raw = (ROOT / quality.INHERITED_SOURCE).read_bytes()
        self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()),
                         (quality.INHERITED_BYTES, quality.INHERITED_SHA256))
        expected = self.module.record_metrics(self.raw, self.png, self.target, self.row, np=np,
            frozen_U96_raw=self.baseline, frozen_U96_png=self.baseline_png)
        result = self.calculate()
        for field, values in expected.items():
            self.assertEqual(result[field], values)
        self.assertEqual(tuple(result['metrics']['raw']), ('mae', 'adjacent_rgb_error'))
        self.assertNotIn('coarse_rgb_mae', result['metrics']['raw'])
        self.assertEqual(set(result['regional_metrics']['raw']), {'whole', 'central128', 'outside'})
        self.assertEqual(result['U96_behavior_distance']['raw'], .125)
        self.assertEqual(result['input'], self.row['input'])
        self.assertEqual(result['target'], self.row['target'])
        self.assertFalse(result['model_qualified'])

    def test_byte_pin_and_import_origin_failure_occurs_before_original_metric_call(self):
        with mock.patch.object(quality, 'INHERITED_SHA256', '0' * 64), \
             mock.patch.object(self.module, 'record_metrics', side_effect=AssertionError('Unpinned call')):
            with self.assertRaisesRegex(ValueError, 'unchanged inherited quality source bytes'):
                self.calculate()
        with mock.patch.object(self.module, '__file__', str(ROOT / 'wrong_metrics.py')):
            with self.assertRaisesRegex(ValueError, 'import origin differs'):
                self.calculate()
        name = next(iter(quality.INHERITED_DEPENDENCIES))
        changed = copy.deepcopy(quality.INHERITED_DEPENDENCIES)
        changed[name]['sha256'] = '0' * 64
        with mock.patch.object(quality, 'INHERITED_DEPENDENCIES', changed):
            with self.assertRaisesRegex(ValueError, 'quality dependency source bytes'):
                self.calculate()

    def test_null_legacy_dark_targets_stay_null_and_do_not_acquire_clean_metrics(self):
        row = scoped(next(row for row in self.parent['guards'] if row['role'] == 'dark'), 'old_quality')
        teacher = np.full_like(self.raw, .125)
        teacher_png = np.full_like(self.png, 31)
        result = self.calculate(row=row, target=None, teacher_raw=teacher, teacher_png=teacher_png)
        self.assertIsNone(result['target'])
        self.assertEqual(result['metrics'], {view: {'mae': None, 'adjacent_rgb_error': None} for view in R.VIEWS})
        self.assertEqual(result['regional_metrics'], dict.fromkeys(R.VIEWS, None))
        self.assertEqual(result['behavior_distance']['raw'], .375)
        with self.assertRaisesRegex(ValueError, 'target presence'):
            self.calculate(row=row)

    def test_original_teacher_presence_and_explicit_U96_controls_remain_mandatory(self):
        for key in ('frozen_U96_raw', 'frozen_U96_png'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.calculate(**{key: None})
        row = scoped(next(row for row in self.parent['guards'] if row['role'] == 'clear'), 'old_quality')
        with self.assertRaisesRegex(ValueError, 'teacher pair'):
            self.calculate(row=row)
        teacher = np.zeros_like(self.raw); teacher_png = np.zeros_like(self.png)
        result = self.calculate(row=row, teacher_raw=teacher, teacher_png=teacher_png)
        self.assertEqual(result['behavior_distance']['raw'], .5)

    def test_old_optimization_or_new_bank_rows_cannot_enter_inherited_records(self):
        row = scoped(next(row for row in self.parent['cases'] if row['cohort'] == 'optimize'), 'old_quality')
        with self.assertRaisesRegex(ValueError, 'evaluation-only'):
            self.calculate(row=row)
        for field, value in (('bank', 'new_monitors'), ('case_key', self.row['id']),
                             ('split', 'DEV'), ('optimization_permitted', True),
                             ('optimization_permitted', 0.0), ('cohort', 'optimize')):
            row = copy.deepcopy(self.row); row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.calculate(row=row)

    def test_raw_and_target_double_arrays_are_refused_before_inherited_arithmetic(self):
        with self.assertRaisesRegex(ValueError, 'FP32'):
            self.calculate(raw=self.raw.astype(np.float64))
        with self.assertRaisesRegex(ValueError, 'FP32'):
            self.calculate(target=self.target.astype(np.float64))

    def test_old_and_new_clear_ID_collision_keeps_banks_and_provenance_separate(self):
        selection = C.select_cases((PREP / 'data_admission_v1.json').read_bytes(),
            (PREP / 'candidate_data_v1.json').read_bytes(), PARENT.read_bytes())
        new_row = scoped(next(row for row in selection['new_monitors'] if row['id'] == self.row['id']), 'new_monitors')
        old = self.calculate()
        new = R.record_new(self.raw, self.png, self.target, new_row, np=np)
        self.assertEqual(old['id'], new['id'])
        self.assertNotEqual(old['case_key'], new['case_key'])
        self.assertNotEqual(old['input']['path'], new['input']['path'])
        self.assertNotEqual(old['target']['path'], new['target']['path'])
        self.assertNotEqual(set(old['metrics']['raw']), set(new['metrics']['raw']))
        wrong_old = scoped(new_row, 'old_quality')
        with self.assertRaisesRegex(ValueError, 'inherited asset namespace'):
            self.calculate(row=wrong_old)
        wrong_new = scoped(self.row, 'new_monitors')
        with self.assertRaisesRegex(ValueError, 'contract-normalized'):
            R.record_new(self.raw, self.png, self.target, wrong_new, np=np)

    def test_original_arrays_and_row_pins_remain_unchanged(self):
        values = (self.raw, self.png, self.target, self.baseline, self.baseline_png)
        before = [value.tobytes() for value in values]
        row_before = copy.deepcopy(self.row)
        for value in values: value.flags.writeable = False
        result = self.calculate()
        result['input']['path'] = 'changed'
        result['metrics']['raw']['mae'] = 0.0
        self.assertEqual([value.tobytes() for value in values], before)
        self.assertEqual(self.row, row_before)


class InertRecordSource(unittest.TestCase):
    def test_source_is_numpy_explicit_and_has_no_decoder_or_runtime_entrypoint(self):
        tree = ast.parse(inspect.getsource(R), feature_version=(3, 10))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom): imports.append(node.module or '')
        self.assertFalse(any(name.split('.')[0] in FRAMEWORK_ROOTS | {'numpy', 'PIL'}
                             or name.endswith(('.model', '.worker')) for name in imports))
        for function in (R.record_new, R.record_inherited):
            self.assertIn('np', inspect.signature(function).parameters)
            self.assertEqual(inspect.signature(function).parameters['np'].kind,
                             inspect.Parameter.KEYWORD_ONLY)
        self.assertFalse(hasattr(R, 'run'))
        self.assertEqual(BLOCKER.attempted, [])

    def test_clean_process_import_loads_no_numpy_framework_model_decoder_or_worker(self):
        code = '''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','PIL','tensorflow','jax','jaxlib','keras','torchvision','torchaudio'} or fullname.endswith(('.model','.worker')):
   raise RuntimeError('Forbidden:'+fullname)
  return None
sys.meta_path.insert(0,Block())
from research.codeformer_matched_feature_learning_v1 import records
assert not any(name.split('.')[0] in {'torch','numpy','PIL','tensorflow','jax','jaxlib','keras','torchvision','torchaudio'} or name.endswith(('.model','.worker')) for name in sys.modules)
print('INERT_EXPLICIT_NUMPY_RECORD_SOURCE')
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'INERT_EXPLICIT_NUMPY_RECORD_SOURCE')


if __name__ == '__main__':
    unittest.main()
