"""Pure source/receipt/metric guards; no models or synthetic VM returns."""
import ast
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts/run_u96_postfusion_lr_broader_preservation_v1.py'
BLOCKED = {'torch', 'psutil', 'PIL', 'numpy', 'skimage'}


def module_without_heavy_imports():
    original = builtins.__import__; oldpath = list(sys.path)
    def inert(name, *args, **kwargs):
        if name.split('.')[0] in BLOCKED:
            raise AssertionError('Heavy import during inert module load: ' + name)
        return original(name, *args, **kwargs)
    try:
        builtins.__import__ = inert
        sys.path.insert(0, str(ROOT / 'scripts'))
        spec = importlib.util.spec_from_file_location('_broader_runner_pure_test', RUNNER)
        result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
        return result
    finally:
        builtins.__import__ = original
        sys.path[:] = oldpath


R = module_without_heavy_imports()


def pure_metrics():
    spec = importlib.util.spec_from_file_location('_broader_metrics_pure_test', ROOT / R.METRICS_PATH)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    result.sum = R.ordered_sum
    return result


def gate_fixture():
    rows = []
    for profile, count in [('clear', 61), ('blur', 53), ('lowlight', 53), ('compound', 53), ('other', 138)]:
        rows.extend({'id': profile + f'_{i:03d}', 'source': 'FFHQ validation',
                     'profile': profile, 'evidence': 'paired_synthetic'} for i in range(count))
    rows.extend({'id': f'native_{i:03d}', 'source': 'native fixture', 'profile': 'native',
                 'evidence': 'native_unpaired'} for i in range(48))
    values = {}
    for row in rows:
        fields = {'mae': .1, 'mse': .02, 'psnr': 16., 'ssim': .5, 'luminance_bias': .1, 'coarse_rgb_mae': .1}
        values[row['id']] = {**row, 'metrics': None if row['evidence'] == 'native_unpaired' else
                            {view: dict(fields) for view in ('raw', 'native_png')}}
    return values


class BroaderPureTests(unittest.TestCase):
    def test_inert_import_restores_actual_import_callable(self):
        before = builtins.__import__; module_without_heavy_imports()
        self.assertIs(before, builtins.__import__)
        self.assertNotIn('torch', sys.modules)

    def test_python310_AST_and_help_are_inert(self):
        ast.parse(RUNNER.read_bytes(), feature_version=(3, 10))
        response = subprocess.run([sys.executable, '-B', str(RUNNER), '--help'],
                                  cwd=str(ROOT), text=True, capture_output=True, timeout=20)
        self.assertEqual(response.returncode, 0, response.stderr)
        self.assertIn('--teacher-audit', response.stdout)
        self.assertIn('--verify-only', response.stdout)

    def test_exact_historical716_slot_and90_sheet_plan(self):
        bundle = json.loads(R.BUNDLE.read_bytes())
        paired = [r for r in bundle['cases'][:406] if r['evidence'] == 'paired_synthetic']
        output = R.slots(paired)
        self.assertEqual(len(output), 716)
        self.assertEqual([r['new_output_index'] for r in output], list(range(1, 717)))
        self.assertEqual([r['case_id'] for r in output[::2]], [r['id'] for r in paired])
        self.assertEqual([r['arm'] for r in output[::2]], ['incumbent_U96'] * 358)
        self.assertEqual([r['arm'] for r in output[1::2]], ['postfusion_LR_u64'] * 358)
        sheets = R.sheet_plan(paired)
        self.assertEqual(len(sheets), 90)
        self.assertEqual(sum(s['generated_PNG_cells'] for s in sheets), 716)
        self.assertTrue(all(1 <= s['rows'] <= 4 for s in sheets))
        self.assertEqual([c for s in sheets for c in s['case_ids']], [r['id'] for r in paired])

    def test_real_teacher_schema_and_missing_failure_blocks(self):
        teacher = json.loads((R.PREP / 'independent_teacher_reference_audit_v1.json').read_bytes())
        historical = json.loads(R.BUNDLE.read_bytes())['cases'][:406]
        self.assertEqual(len(R.require_teacher(teacher, historical)), 358)
        bad = dict(teacher, complete=False)
        with self.assertRaises(ValueError): R.require_teacher(bad, historical)
        bad = dict(teacher, rows=teacher['rows'][:-1])
        with self.assertRaises(ValueError): R.require_teacher(bad, historical)
        bad = dict(teacher, gate_case_ids={})
        with self.assertRaises(ValueError): R.require_teacher(bad, historical)

    def test_source_review_not_a_future_boolean(self):
        with self.assertRaises(ValueError):
            R.reviewed_sources({'complete': True, 'source_review_pass': True, 'sources': []}, {'actual.py'})
        with self.assertRaises(ValueError):
            R.reviewed_sources({'complete': False, 'source_review_pass': False, 'sources': []}, {'actual.py'})

    def test_null_native_banks_never_accessed_by_legacy_gates(self):
        bank = gate_fixture(); M = pure_metrics()
        self.assertEqual(len(bank), 406)
        self.assertTrue(M.preservation(bank, bank)['pass'])
        self.assertTrue(M.dark_retention(bank, bank)['pass'])
        self.assertIsNone(R.blank_native_metrics())

    def test_clear_worst_and_mean_thresholds_both_views(self):
        M = pure_metrics(); initial = gate_fixture(); current = json.loads(json.dumps(initial))
        current['clear_000']['metrics']['raw']['mae'] += .031
        self.assertFalse(M.preservation(current, initial)['pass'])
        current = json.loads(json.dumps(initial))
        for row in current.values():
            if row['profile'] == 'clear': row['metrics']['native_png']['mae'] += .0101
        gate = M.preservation(current, initial)
        self.assertFalse(gate['groups']['clear']['views']['native_png']['mae']['pass'])
        self.assertTrue(gate['groups']['clear']['views']['raw']['mae']['pass'])

    def test_normal_blur_photometric_both_fields(self):
        M = pure_metrics(); initial = gate_fixture()
        for view, field in [('raw', 'luminance_bias'), ('native_png', 'coarse_rgb_mae')]:
            current = json.loads(json.dumps(initial))
            current['blur_000']['metrics'][view][field] += .031
            self.assertFalse(M.preservation(current, initial)['groups']['normal_blur']['views'][view][field]['pass'])

    def test_dark_no_epsilon_or_relaxed_mean(self):
        M = pure_metrics(); initial = gate_fixture()
        for profile, field, direction in [('lowlight', 'mae', 1), ('compound', 'ssim', -1)]:
            current = json.loads(json.dumps(initial))
            for row in current.values():
                if row['profile'] == profile: row['metrics']['native_png'][field] += direction * 1e-12
            self.assertFalse(M.dark_retention(current, initial)['profiles'][profile]['pass'])
        self.assertTrue(M.dark_retention(initial, initial)['pass'])
        self.assertEqual(R.ordered_sum([1e16, 1., -1e16]), 0.)

    def test_writer_exclusive_budget_and_append_only_numbering(self):
        with tempfile.TemporaryDirectory(prefix='broader_pure_', dir=ROOT / 'scratch') as temp:
            folder = Path(temp); writer = R.Writer(folder, 2048)
            first = writer.write(folder / 'kept.bin', b'exact')
            with self.assertRaises(FileExistsError): writer.write(folder / 'kept.bin', b'other')
            self.assertEqual((folder / 'kept.bin').read_bytes(), b'exact')
            self.assertEqual(first['sha256'], R.B.digest(b'exact'))
            journal = R.Journal(writer, folder / 'journal')
            journal.append({'label': 'before'}); journal.append({'label': 'after'})
            self.assertEqual([p.name for p in (folder / 'journal').iterdir()], ['000001.json', '000002.json'])
            self.assertEqual(json.loads((folder / 'journal/000002.json').read_bytes())['sequence'], 2)
            with self.assertRaises(ValueError): writer.write(folder / 'over.bin', b'x' * 2048)
            self.assertFalse((folder / 'over.bin').exists())

    def test_supervisor_completed_pass_vs_quality_stop(self):
        counters = {'forward_attempts': 716, 'forward_returns': 716, 'saved_outputs': 716, 'completed_paired_cases': 358}
        last = {'counters': counters, 'measurement_complete': True, 'forward_outcome_unknown': False}
        for code, status, passed in [(0, 'BROAD_PRESERVATION_GATE_PASS_NOT_QUALIFIED', True),
                                      (2, 'QUALITY_STOP_NOT_QUALIFIED', False)]:
            result = R.supervisor_outcome(code, False, dict(last, quality_status=status), [], None, False,
                {'complete': True, 'quality_status': status, 'counters': counters,
                 'gates': {R.ARMS[1]: {'preservation_vs_original': {'pass': passed},
                                      'dark_retention_vs_R2': {'pass': True}}}})
            self.assertTrue(result['all716_measurements_complete'])
            self.assertEqual(result['finite_preservation_gate_pass_not_qualification'], passed)
            self.assertEqual(result['quality_stop_recorded'], not passed)
            self.assertFalse(result['forward_outcome_unknown'])

    def test_supervisor_reservation_failure_empty_or_missing_status(self):
        old = {'counters': {'forward_attempts': 0, 'forward_returns': 0}, 'measurement_complete': False,
               'forward_outcome_unknown': False}
        failure = {'counters': {'forward_attempts': 1, 'forward_returns': 0}, 'forward_outcome_unknown': True}
        result = R.supervisor_outcome(1, False, old, [], failure, False, None)
        self.assertTrue(result['forward_outcome_unknown'])
        self.assertEqual(result['counters']['forward_attempts'], 1)
        self.assertEqual(result['last_durable_counters']['forward_attempts'], 0)
        for last, partial, unreadable in [(None, [], False), (old, [{'bytes': 0}], False), (old, [], True)]:
            self.assertTrue(R.supervisor_outcome(1, False, last, partial, None, unreadable, None)['forward_outcome_unknown'])

    def test_empty_and_wrong_shape_journal_bytes_preserved(self):
        with tempfile.TemporaryDirectory(prefix='broader_partial_', dir=ROOT / 'scratch') as temp:
            folder = Path(temp); (folder / 'journal').mkdir()
            (folder / 'journal/000001.json').write_bytes(b'')
            (folder / 'journal/000002.json').write_bytes(b'[]')
            with mock.patch.object(R, 'OUT', folder):
                parsed, partial = R.read_journal()
            self.assertEqual(parsed, [])
            self.assertEqual([r['bytes'] for r in partial], [0, 2])
            self.assertEqual(partial[0]['sha256'], R.B.digest(b''))
            self.assertEqual(partial[1]['sha256'], R.B.digest(b'[]'))
            self.assertEqual((folder / 'journal/000001.json').read_bytes(), b'')

    def test_delivery_AST_same_as_retained_tail(self):
        prior = ast.parse((ROOT / 'scripts/run_u96_smoothanchors_lr_fixed_DEV_tail_v1.py').read_bytes())
        current = ast.parse(RUNNER.read_bytes())
        def first_assignment(tree, name):
            worker_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
            for node in ast.walk(worker_node):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    if isinstance(node.value, ast.Constant) and node.value.value is None:
                        continue
                    return ast.dump(node.value, include_attributes=False)
            self.fail('Missing retained delivery assignment: ' + name)
        for name in ('sample', 'unclipped', 'raw', 'raw_array', 'pixels'):
            self.assertEqual(first_assignment(prior, name), first_assignment(current, name), name)
        worker = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
        ordered = sorted((n.lineno, ast.unparse(n)) for n in ast.walk(worker)
                         if isinstance(n, ast.Call) and (isinstance(n.func, ast.Subscript) or
                            isinstance(n.func, ast.Name) and n.func.id == 'status'))
        reserved = next(line for line, text in ordered if text == "status('forward_reserved')")
        forward = next(line for line, text in ordered if text.startswith('models[arm]('))
        returned = next(line for line, text in ordered if text == "status('forward_returned')")
        self.assertLess(reserved, forward); self.assertLess(forward, returned)


if __name__ == '__main__': unittest.main()
