"""Stdlib opaque-file transport fixtures; no image/tensor/framework execution."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from research.codeformer_fixed_state_response_v1 import contract as c, transport as t
from scripts import supervise_codeformer_fixed_state_response_v1_vm as supervisor

ROOT = Path(__file__).resolve().parents[1]
ZERO_KEYS = ('restorer_updates_this_run', 'head_updates_this_run', 'gradient_queries',
             'backward_calls', 'optimizer_constructions', 'optimizer_steps')


def encode(value): return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


class FixedResponseTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = ROOT / 'scratch' / ('fixed_response_transport_' + uuid.uuid4().hex)
        cls.parents = {alias: cls.base / alias for alias in ('tone_training', 'adain')}
        cls.deps, cls.protocols = {}, {}
        for alias, parent in cls.parents.items():
            parent.mkdir(parents=True)
            pin = put(parent / 'protocol.json', encode({'format': alias + '-opaque-fixture'}))
            cls.deps[alias] = {'protocol.json': pin}; cls.protocols[alias] = pin['sha256']
        cls.assets = {}
        for key, alias, name in [('initializer', 'adain', 'official.pth'),
                                  ('initial_composite', 'tone_training', 'initial.pt'),
                                  ('stopped_composite', 'tone_training', 'stopped.pt')]:
            pin = put(cls.parents[alias] / name, b'opaque read-only synthetic state; never load')
            cls.deps[alias][name] = pin; cls.assets[key] = {'root': alias, 'path': name, **pin}
        cls.ids = {'fitted': ['train_u%04d_slot%d' % (u, s) for u in range(1, 51) for s in range(1, 5)],
                   'heldout': ['heldout_tr_ffhq_%05d_%s' % (r, role) for r in range(32) for role in ('normal', 'dark')]}
        cls.pngs = {'inputs/heldout_tr_ffhq_%05d/%s/input.png' % (r, role)
                    for r in range(32) for role in ('normal', 'dark')}
        cls.pair_paths = cls.pngs | {name.replace('input.png', 'pair_record.json') for name in cls.pngs}

    def fixture(self, terminal='failure'):
        base = self.base / uuid.uuid4().hex; root, home = base / 'new', base / 'home'
        root.mkdir(parents=True); home.mkdir()
        files = {}
        for name in sorted(self.pair_paths): files[name] = put(root / name, b'{}\n' if name.endswith('.json') else b'opaque fixed PNG; never decode')
        for name, data in [('research/new.py', b'raise RuntimeError("neverexecute")\n'),
                           ('fitted_cases.json', encode({'cases': [{'id': v} for v in self.ids['fitted']]})),
                           ('heldout_pairs.json', encode({'cases': [{'id': v} for v in self.ids['heldout']]}))]:
            files[name] = put(root / name, data)
        p = {'files_sha256': {n: r['sha256'] for n, r in files.items()},
             'files_bytes': {n: r['bytes'] for n, r in files.items()},
             'parent_roots': dict.fromkeys(self.parents, 'synthetic_parent_name'),
             'parent_protocol_sha256': deepcopy(self.protocols), 'dependencies': deepcopy(self.deps),
             'assets': deepcopy(self.assets), 'fitted_index_file': 'fitted_cases.json', 'new_pairs_file': 'heldout_pairs.json'}
        sha = put(root / 'protocol.json', encode(p))['sha256']; counts = dict.fromkeys(c.COUNTS, 0)
        counts['controller_fitted_initial'] = 17
        value = {'protocol_sha256': sha, 'complete': True, 'failure': terminal == 'failure',
                 'measurement_complete': terminal == 'success', 'counts': counts,
                 **dict.fromkeys(ZERO_KEYS, 0)}
        if terminal == 'success': value['counts'] = dict(c.COUNTS)
        if terminal is not None: put(root / t.OUTPUTS / ('run_results.json' if terminal == 'success' else 'failure.json'), encode(value))
        put(root / 'outputs/supervision/run_receipt.json', encode({'worker_terminal': True, 'protocol_sha256': sha}))
        for name in ('preflight.log', 'diagnostic.log', 'export_invocation.json'): put(root / 'outputs/supervision' / name, b'{}\n')
        put(root / 'outputs/supervision/export.log', b'active export stdout boundary')
        return SimpleNamespace(root=root, home=home, p=p, sha=sha)

    @contextmanager
    def patches(self, f):
        # Authority/schema/platform replacements are fixture-only. Real byte,
        # path, count, coverage and streaming logic remains under test. The data
        # agent tests actual protocol and row/path schemas in a separate suite.
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(c, 'verify_protocol', side_effect=lambda p: p))
            stack.enter_context(mock.patch.object(c, 'resolve_parents', return_value=self.parents))
            stack.enter_context(mock.patch.object(c, 'expected_pair_paths', return_value=self.pair_paths))
            stack.enter_context(mock.patch.object(c, 'expected_input_pngs', return_value=self.pngs))
            stack.enter_context(mock.patch.object(c, 'expected_row_ids', return_value=deepcopy(self.ids)))
            stack.enter_context(mock.patch.object(c, 'manual_root', side_effect=lambda p: Path(p)))
            stack.enter_context(mock.patch.object(t, '_process_rows', return_value=[]))
            stack.enter_context(mock.patch.object(Path, 'home', return_value=f.home))
            stack.enter_context(mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=16 * 1024**3)))
            yield

    def complete_inventory(self, f):
        files = t._walk(f.root)
        for bank in ('initial', 'final'):
            for row in self.ids['fitted'] + self.ids['heldout']:
                for filename in ('raw.npy', 'native.png', 'record.json'):
                    name = t.OUTPUTS + '/' + bank + '/' + row + '/' + filename; files[name] = f.root / name
        for row in self.ids['heldout']:
            for filename in ('original_raw.npy', 'base_raw.npy'):
                name = t.OUTPUTS + '/new/' + row + '/' + filename; files[name] = f.root / name
        return files

    def test_dual_API_readonly_both_parents_exact_static_inputs_no_execution(self):
        f = self.fixture(); before = {str(p): p.stat().st_mtime_ns for p in f.root.rglob('*') if p.is_file()}
        with self.patches(f): a, b = t.verify_transfer(f.root, f.sha), t.verify_transfer(f.root, f.p, f.sha)
        self.assertEqual(a, b); self.assertEqual(a['dependencies_verified'], 5)
        self.assertEqual(a['dependencies_verified_by_parent'], {'tone_training': 3, 'adain': 2})
        self.assertEqual(a['static_input_PNGs_verified'], 64); self.assertEqual(a['neural_calls'], 0)
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in f.root.rglob('*') if p.is_file()})

    def test_changed_parent_or_fixed_input_refused(self):
        f = self.fixture(); path = self.parents['tone_training'] / 'stopped.pt'; old = path.read_bytes()
        try:
            path.write_bytes(b'changed')
            with self.patches(f), self.assertRaisesRegex(ValueError, 'Read-only parent'): t.verify_transfer(f.root, f.sha)
        finally: path.write_bytes(old)
        (f.root / next(iter(self.pngs))).write_bytes(b'changed')
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Static bytes'): t.verify_transfer(f.root, f.sha)

    def test_wrong_caller_protocol_extra_static_PNG_and_importable_refused(self):
        f = self.fixture(); p = deepcopy(f.p); p['new_pairs_file'] = 'wrong'
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Caller protocol'): t.verify_transfer(f.root, p, f.sha)
        for name in ('unexpected.png', 'research/unbound.py'):
            f = self.fixture(); put(f.root / name, b'opaque')
            with self.subTest(name=name), self.patches(f), self.assertRaisesRegex(ValueError, 'Unbound static'):
                t.verify_transfer(f.root, f.sha)

    def test_failure_exports_actual_reservations_and_explicit_missing_coverage(self):
        f = self.fixture()
        with self.patches(f): receipt = t.export_results(f.root, f.sha)
        self.assertTrue(receipt['failure_present']); self.assertFalse(receipt['complete_measurement'])
        self.assertEqual(receipt['worker_recorded_counts']['controller_fitted_initial'], 17)
        self.assertEqual(receipt['head_updates_this_run'], 0); self.assertFalse(receipt['output_coverage']['complete'])
        self.assertFalse(receipt['new_checkpoint_serialization']); self.assertEqual(receipt['checkpoint_members'], [])
        manifest = c.read_json(f.root / t.OUTPUTS / 'export_manifest.json')
        self.assertNotIn('stopped.pt', manifest['files_sha256'])
        self.assertNotIn('outputs/supervision/export.log', manifest['files_sha256'])
        self.assertTrue(self.pngs <= set(manifest['files_sha256']))
        archive = f.home / (t.PREFIX + '-results.tar.gz')
        self.assertEqual(Path(str(archive) + '.sha256').read_bytes(), (c.sha256(archive) + '  ' + archive.name + '\n').encode('ascii'))
        with self.patches(f), self.assertRaises(ValueError): t.export_results(f.root, f.sha)

    def test_missing_worker_receipt_preserves_unknown_instead_of_zero(self):
        f = self.fixture(None)
        with self.patches(f): value = t._terminal_evidence(f.root, t._walk(f.root), f.sha, f.p)
        self.assertIsNone(value['worker_recorded_counts']); self.assertTrue(value['counts_unknown_without_worker_receipt'])
        for key in ZERO_KEYS: self.assertIsNone(value[key])

    def test_export_JSON_declared2MiB_cap_refuses_before_write(self):
        f = self.fixture(); name = f.root / t.OUTPUTS / 'oversize.json'
        with self.patches(f), self.assertRaisesRegex(ValueError, 'declared2MiB'):
            t._write_reserved_json(name, {'payload': 'x' * c.LIMITS['one_JSON_bytes']})
        self.assertFalse(name.exists())
        self.assertEqual(c.LIMITS['one_JSON_bytes'], 2 * 1024**2)

    def test_complete656call_measurement_exact528head_and128parentcoverage(self):
        f = self.fixture('success'); files = self.complete_inventory(f)
        with self.patches(f): value = t._terminal_evidence(f.root, files, f.sha, f.p)
        self.assertTrue(value['complete_measurement']); self.assertEqual(value['worker_recorded_counts'], c.COUNTS)
        self.assertEqual(value['output_coverage']['head_RAW_PNG_pairs'], 528)
        self.assertEqual(value['output_coverage']['fresh_parent_RAW_arrays'], 128)
        self.assertEqual(value['output_coverage']['row_records'], 528)
        for name in (t.OUTPUTS + '/final/' + self.ids['fitted'][0] + '/record.json',
                     t.OUTPUTS + '/new/' + self.ids['heldout'][0] + '/original_raw.npy'):
            with self.subTest(name=name), self.patches(f), self.assertRaisesRegex(ValueError, 'Complete656call'):
                t._terminal_evidence(f.root, {n: p for n, p in files.items() if n != name}, f.sha, f.p)

    def test_zero_update_gradient_optimizer_schema_and_caps_refuse(self):
        for key in ZERO_KEYS:
            f = self.fixture(); path = f.root / t.OUTPUTS / 'failure.json'; value = c.read_json(path); value[key] = 1; put(path, encode(value))
            with self.subTest(key=key), self.patches(f), self.assertRaisesRegex(ValueError, 'Zero-update'):
                t._terminal_evidence(f.root, t._walk(f.root), f.sha, f.p)
        f = self.fixture(); path = f.root / t.OUTPUTS / 'failure.json'; value = c.read_json(path)
        value['counts']['controller_fitted_initial'] = 201; put(path, encode(value))
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Operation cap'): t._terminal_evidence(f.root, t._walk(f.root), f.sha, f.p)

    def test_checkpoint_and_partial_weight_serialization_forbidden(self):
        f = self.fixture()
        for name in ('checkpoint.pt', 'model.pth', 'model.pth.part', 'checkpoint.pt.tmp-' + 'f' * 32, 'data.pkl'):
            path = t.OUTPUTS + '/' + name; put(f.root / path, b'opaque')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'serialization forbidden'):
                t._measure_files(f.root, [path])

    def test_live_worker_lock_and_low_reserve_refuse(self):
        f = self.fixture()
        with self.patches(f), mock.patch.object(t, '_process_rows', return_value=[(123, 'python -m ' + t.MODULE + ' --run')]), self.assertRaises(ValueError):
            t.export_results(f.root, f.sha)
        put(f.root / c.LOCK_NAME, encode({'pid': 123}))
        with self.patches(f), mock.patch.object(t.os, 'kill', return_value=None), self.assertRaises(ValueError): t.require_worker_terminal(f.root)
        f = self.fixture()
        with self.patches(f), mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=t.RESERVE)), self.assertRaises(ValueError):
            t.export_results(f.root, f.sha)

    def test_exact_final_companion_log_finished_and_unknown_or_zero_counts(self):
        f = self.fixture()
        with self.patches(f): t.export_results(f.root, f.sha)
        terminal = {'complete': True, 'worker_terminal': True, 'protocol_sha256': f.sha, 'stages': []}
        put(f.root / 'outputs/supervision/finished.json', encode(terminal))
        log = b'exact closed stdout\n'; put(f.root / 'outputs/supervision/export.log', log)
        with self.patches(f): result = supervisor.final_companion(f.root, [])
        value = c.read_json(result['path']); self.assertEqual(base64.b64decode(value['export_log_base64']), log)
        self.assertEqual(value['head_updates_this_run'], 0); self.assertFalse(value['checkpoint_exported'])
        self.assertFalse(value['model_qualified']); self.assertEqual(value['finished_json'], terminal)
        with self.patches(f), self.assertRaises(FileExistsError): supervisor.final_companion(f.root, [])

    def test_supervisor_failed_measurement_still_exports_once_with_finitecaps(self):
        f = self.fixture(None); root = self.base / uuid.uuid4().hex / 'clean'; root.mkdir(parents=True)
        def result(root, sha, flag, seconds, grace, log):
            return {'stage': flag, 'exit_code': 1 if flag == '--run' else 0, 'worker_terminal': True}
        with mock.patch.object(c, 'manual_root', return_value=root), mock.patch.dict(supervisor.os.environ, {'TMUX': 'fixture'}), \
             mock.patch.object(supervisor.shutil, 'which', return_value='/usr/bin/timeout'), \
             mock.patch.object(supervisor, 'idle', return_value={}), mock.patch.object(supervisor, 'stage', side_effect=result) as stages, \
             mock.patch.object(supervisor, 'final_companion', return_value={}):
            self.assertEqual(supervisor.main(['--root', str(root), '--protocol-sha', f.sha]), 1)
        self.assertEqual([(x.args[2], x.args[3], x.args[4]) for x in stages.call_args_list],
                         [('--verify-transfer', 300, 30), ('--run', 900, 30), ('--export', 300, 30)])
        value = c.read_json(root / 'outputs/supervision/run_receipt.json')
        self.assertIsNone(value['head_updates_this_run']); self.assertEqual(value['maximum_head_updates'], 0)

    def test_recovered_supervisor_exception_exports_only_after_ownedgroup_terminal(self):
        f = self.fixture(); fake = SimpleNamespace(pid=4242, stdout=None)
        with mock.patch.object(supervisor.subprocess, 'Popen', return_value=fake), \
             mock.patch.object(supervisor.selectors, 'DefaultSelector', side_effect=RuntimeError('synthetic fixture')), \
             mock.patch.object(supervisor, '_stop_own_group') as stop, mock.patch.object(supervisor, '_group_alive', return_value=False):
            value = supervisor.stage(f.root, f.sha, '--run', 900, 30, f.root / 'synthetic.log')
        stop.assert_called_once_with(fake, 30); self.assertEqual(value['exit_code'], 125)
        with mock.patch.object(supervisor.subprocess, 'Popen', return_value=fake), \
             mock.patch.object(supervisor.selectors, 'DefaultSelector', side_effect=RuntimeError('fixture')), \
             mock.patch.object(supervisor, '_stop_own_group', side_effect=ValueError('not terminal')), self.assertRaisesRegex(ValueError, 'not terminal'):
            supervisor.stage(f.root, f.sha, '--run', 900, 30, f.root / 'synthetic-refusal.log')

    def test_portable_paths_hardlink_metadata_and_exact_sourcecaps(self):
        for name in ('../escape', 'x/CON.png', 'a\\b', 'a/./b', 'a.', '/abs'):
            with self.subTest(name=name), self.assertRaises(ValueError): t.canonical(name)
        f = self.fixture(); target = f.root / 'research/new.py'; lstat = Path.lstat
        def info(path, *args, **kwargs):
            value = lstat(path, *args, **kwargs)
            return SimpleNamespace(st_mode=value.st_mode, st_nlink=2, st_file_attributes=0) if path == target else value
        with mock.patch.object(Path, 'lstat', info), self.assertRaisesRegex(ValueError, 'Hardlinked|Hard-linked packet'):
            t._regular(f.root, 'research/new.py')
        self.assertEqual(sum(c.COUNTS.values()), 656)
        self.assertEqual((t.PAYLOAD_CAP, t.MEMBER_CAP, t.EXPORT_SECONDS), (1024**3, 3000, 300))


if __name__ == '__main__': unittest.main()
