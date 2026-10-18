"""CPU-only synthetic transfer/export fixtures; opaque states are never loaded."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import unittest
from unittest import mock
import uuid

from research.codeformer_tone_control_training_v1 import contract as c, transport as t, budget
from scripts import supervise_codeformer_tone_control_training_v1_vm as supervisor

ROOT = Path(__file__).resolve().parents[1]


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


class ToneControlTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep one immutable synthetic parent rather than duplicate1624tiny files
        # for every fixture. All fixtures stay under workspace scratch as evidence.
        cls.base = ROOT / 'scratch' / ('tone_control_transport_' + uuid.uuid4().hex)
        cls.parent = cls.base / 'parent'
        cls.parent.mkdir(parents=True)
        cls.deps = {}
        for name in ('references.json', 'comparison_cases.json', 'training_parity_v1.json'):
            cls.deps[name] = put(cls.parent / name, b'{}\n')
        cls.assets = {}
        for label in ('initializer', 'base'):
            name = label + '.pth'
            pin = put(cls.parent / name, b'opaque synthetic state; never load ' + label.encode())
            cls.deps[name] = pin
            cls.assets[label] = {'path': name, **pin}
        cls.old = {'files_sha256': {n: r['sha256'] for n, r in cls.deps.items()},
                   **deepcopy(c.PARENT_CONTROLS)}
        cls.oldpin = put(cls.parent / 'protocol.json', encode(cls.old))
        cls.deps['protocol.json'] = cls.oldpin
        cls.retained = {}
        cls.cases = ['case_' + str(i).zfill(4) for i in range(406)]
        for snapshot in ('0000', '0050'):
            for case in cls.cases:
                for name in ('raw.npy', 'native.png'):
                    relative = 'outputs/adain_gradient_isolation_v1/snapshot_' + snapshot + '/' + case + '/' + name
                    cls.retained[relative] = put(cls.parent / relative, b'opaque retained pixel fixture\n')

    def fixture(self, terminal='failure'):
        base = self.base / uuid.uuid4().hex
        root, home = base / 'new', base / 'home'
        root.mkdir(parents=True); home.mkdir()
        source = put(root / 'research/new.py', b'raise RuntimeError("must never execute")\n')
        p = {'outputs': t.OUTPUTS, 'files_sha256': {'research/new.py': source['sha256']},
             'files_bytes': {'research/new.py': source['bytes']},
             'dependencies': deepcopy(self.deps), 'retained_output_dependencies': deepcopy(self.retained),
             'historical_parent_protocol': deepcopy(self.old), 'parent_dir': 'parent',
             'parent_protocol_sha256': self.oldpin['sha256'], 'assets': deepcopy(self.assets),
             **deepcopy(c.PARENT_CONTROLS)}
        pin = put(root / 'protocol.json', encode(p))
        counts = dict.fromkeys(c.COUNTS, 0)
        counts.update(head_optimizer_step_attempts=6, head_backward_calls=21, controller_TRAIN=21)
        value = {'protocol_sha256': pin['sha256'], 'complete': False, 'completed_head_updates': 5,
                 'counts': counts, 'head_optimizer_inflight': True, 'partial_microbatches': 1,
                 'restorer_updates_this_run': 0}
        if terminal == 'success':
            value.update(complete=True, completed_head_updates=50, head_optimizer_inflight=False,
                         partial_microbatches=0, counts=dict(c.COUNTS))
        if terminal is not None:
            put(root / t.OUTPUTS / ('run_results.json' if terminal == 'success' else 'failure.json'), encode(value))
        put(root / c.CHECKPOINT_PATHS[0], b'opaque genuine synthetic initial state')
        if terminal == 'success':
            put(root / c.CHECKPOINT_PATHS[2], b'opaque genuine synthetic stopped state')
        else:
            put(root / (c.CHECKPOINT_PATHS[2] + '.tmp-' + 'a' * 32), b'partial actual stopped state bytes retained')
        for name in ('preflight.log', 'diagnostic.log', 'export_invocation.json'):
            put(root / 'outputs/supervision' / name, b'{}\n')
        put(root / 'outputs/supervision/run_receipt.json', encode({'worker_terminal': True, 'protocol_sha256': pin['sha256']}))
        put(root / 'outputs/supervision/export.log', b'active log boundary\n')
        return SimpleNamespace(root=root, parent=self.parent, home=home, p=p, sha=pin['sha256'])

    def full_banks(self, f):
        # Cardinality/terminal guards operate on a canonical inventory, so these
        # fixtures need no fake pixel arrays or thousands of physical files.
        inventory = t._walk(f.root)
        for bank in ('original0', 'head0000', 'head0050'):
            names = ['raw.npy', 'native.png', 'evidence.json']
            if bank != 'original0': names.append('base_raw.npy')
            for case in self.cases:
                for name in names:
                    relative = t.OUTPUTS + '/' + bank + '/' + case + '/' + name
                    inventory[relative] = f.root / relative
        for name in c.CHECKPOINT_PATHS[:2]:
            put(f.root / name, b'opaque composite endpoint'); inventory[name] = f.root / name
        return inventory

    @contextmanager
    def patches(self, f):
        # Only platform/protocol authority is replaced. The byte/path/coverage/
        # terminal/export code under test remains real. Contract fixtures are
        # independently tested in the data agent's own test namespace.
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(c, 'verify_protocol', side_effect=lambda p: p))
            stack.enter_context(mock.patch.object(c, 'resolve_parent', return_value=f.parent))
            stack.enter_context(mock.patch.object(c, 'manual_root', side_effect=lambda p: Path(p)))
            stack.enter_context(mock.patch.object(t, '_process_rows', return_value=[]))
            stack.enter_context(mock.patch.object(Path, 'home', return_value=f.home))
            stack.enter_context(mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=32 * 1024**3)))
            yield

    def test_transfer_dual_API_hashes_parent_and_retained_without_execution(self):
        f = self.fixture()
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in f.root.rglob('*') if p.is_file()}
        with self.patches(f):
            a = t.verify_transfer(f.root, f.sha)
            b = t.verify_transfer(f.root, f.p, f.sha)
        self.assertEqual(a, b)
        self.assertEqual(a['dependencies_verified'], 6)
        self.assertEqual(a['retained_output_dependencies_verified'], 1624)
        self.assertEqual(a['DEV_cases_verified'], 406)
        self.assertTrue(a['readonly']); self.assertEqual(a['neural_calls'], 0)
        self.assertEqual(before, {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in f.root.rglob('*') if p.is_file()})

    def test_changed_retained_file_and_parent_state_refused(self):
        f = self.fixture()
        for name in (next(iter(self.retained)), 'base.pth'):
            path = self.parent / name; previous = path.read_bytes()
            try:
                path.write_bytes(b'changed')
                with self.subTest(name=name), self.patches(f), self.assertRaisesRegex(ValueError, 'Read-only parent'):
                    t.verify_transfer(f.root, f.sha)
            finally: path.write_bytes(previous)

    def test_changed_source_unbound_source_and_caller_protocol_refused(self):
        f = self.fixture(); changed = deepcopy(f.p); changed['outputs'] = 'wrong'
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Caller protocol'):
            t.verify_transfer(f.root, changed, f.sha)
        put(f.root / 'research/unbound.py', b'raise RuntimeError("never")')
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Unbound static'):
            t.verify_transfer(f.root, f.sha)
        f = self.fixture(); put(f.root / 'research/new.py', b'changed')
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Static source'):
            t.verify_transfer(f.root, f.sha)

    def test_exact_retained_case_pairs_and_portable_paths(self):
        p = {'retained_output_dependencies': deepcopy(self.retained)}
        self.assertEqual(t.expected_case_ids(p), self.cases)
        del p['retained_output_dependencies'][next(iter(self.retained))]
        with self.assertRaises(ValueError): t.expected_case_ids(p)
        for name in ('../outside', 'A/CON.txt', 'a\\b', 'a/../b', 'a.', 'a:b', 'a/./b'):
            with self.subTest(name=name), self.assertRaises(ValueError): t.canonical(name)

    def test_failure_export_retains_actual_head_and_partial_composite(self):
        f = self.fixture()
        with self.patches(f): receipt = t.export_results(f.root, f.sha)
        self.assertTrue(receipt['failure_present']); self.assertFalse(receipt['complete_training'])
        self.assertEqual(receipt['committed_head_updates'], 5)
        self.assertEqual(receipt['restorer_updates_this_run'], 0)
        self.assertTrue(receipt['head_optimizer_inflight_at_failure'])
        self.assertEqual(receipt['partial_microbatches'], 1)
        self.assertEqual(receipt['candidate_checkpoint_members'], [c.CHECKPOINT_PATHS[0]])
        self.assertEqual(len(receipt['incomplete_checkpoint_temporary_members']), 1)
        self.assertFalse(receipt['DEV_output_coverage']['complete'])
        manifest = c.read_json(f.root / t.OUTPUTS / 'export_manifest.json')
        self.assertIn(c.CHECKPOINT_PATHS[0], manifest['files_sha256'])
        self.assertIn(c.CHECKPOINT_PATHS[2] + '.tmp-' + 'a' * 32, manifest['files_sha256'])
        self.assertNotIn('base.pth', manifest['files_sha256'])
        self.assertNotIn('outputs/supervision/export.log', manifest['files_sha256'])
        archive = f.home / (t.PREFIX + '-results.tar.gz')
        self.assertEqual((f.home / (archive.name + '.sha256')).read_bytes(),
                         (c.sha256(archive) + '  ' + archive.name + '\n').encode('ascii'))
        with tarfile.open(archive, 'r:gz') as bundle:
            self.assertIn(c.CHECKPOINT_PATHS[0], bundle.getnames())
            self.assertTrue(all(member.type == tarfile.REGTYPE and not member.pax_headers for member in bundle.getmembers()))
        with self.patches(f), self.assertRaises(ValueError): t.export_results(f.root, f.sha)

    def test_success_requires_all_case_pairs_base_RAW_and_evidence(self):
        f = self.fixture('success'); inventory = self.full_banks(f)
        with self.patches(f): value = t._terminal_evidence(f.root, inventory, f.sha, f.p)
        self.assertTrue(value['complete_training'])
        coverage = value['DEV_output_coverage']
        self.assertEqual((coverage['RAW_PNG_pairs'], coverage['base_RAW_arrays'], coverage['case_evidence_records']), (1218, 812, 1218))
        for name in ('raw.npy', 'base_raw.npy', 'evidence.json'):
            missing = t.OUTPUTS + '/head0050/' + self.cases[0] + '/' + name
            with self.subTest(name=name), self.patches(f), self.assertRaisesRegex(ValueError, 'all1218'):
                t._terminal_evidence(f.root, {n: p for n, p in inventory.items() if n != missing}, f.sha, f.p)
        with self.patches(f), self.assertRaisesRegex(ValueError, 'All3genuine'):
            t._terminal_evidence(f.root, {n: p for n, p in inventory.items() if n != c.CHECKPOINT_PATHS[2]}, f.sha, f.p)
        self.assertEqual(value['completed_head_updates'], 50)

    def test_completed50_technical_failure_exports_partial_bank_explicitly(self):
        f = self.fixture(); path = f.root / t.OUTPUTS / 'failure.json'; value = c.read_json(path)
        value.update(completed_head_updates=50, head_optimizer_inflight=False, partial_microbatches=0,
                     counts=dict(c.COUNTS)); put(path, encode(value))
        with self.patches(f): receipt = t.export_results(f.root, f.sha)
        self.assertEqual(receipt['committed_head_updates'], 50)
        self.assertTrue(receipt['failure_present']); self.assertFalse(receipt['complete_training'])
        self.assertFalse(receipt['DEV_output_coverage']['complete'])

    def test_missing_worker_receipt_keeps_unknown_counts(self):
        f = self.fixture(None)
        with self.patches(f): receipt = t.export_results(f.root, f.sha)
        self.assertIsNone(receipt['committed_head_updates'])
        self.assertIsNone(receipt['restorer_updates_this_run'])
        self.assertTrue(receipt['head_updates_unknown_without_worker_receipt'])

    def test_terminal_counts_types_stale_protocol_and_restorer_updates_refused(self):
        for key, value in (('completed_head_updates', True), ('partial_microbatches', 5),
                           ('restorer_updates_this_run', 1), ('head_optimizer_inflight', 1),
                           ('protocol_sha256', '0' * 64), ('counts', {})):
            f = self.fixture(); path = f.root / t.OUTPUTS / 'failure.json'; record = c.read_json(path)
            record[key] = value; put(path, encode(record))
            with self.subTest(key=key), self.patches(f), self.assertRaises(ValueError):
                t._terminal_evidence(f.root, t._walk(f.root), f.sha, f.p)
        f = self.fixture(); path = f.root / t.OUTPUTS / 'failure.json'; record = c.read_json(path)
        record['counts']['head_optimizer_step_attempts'] = 51; put(path, encode(record))
        with self.patches(f), self.assertRaises(ValueError): t._terminal_evidence(f.root, t._walk(f.root), f.sha, f.p)

    def test_checkpoint_exact_geometry_and_per_file_temporary_caps(self):
        self.assertEqual(t.checkpoint_kind(c.CHECKPOINT_PATHS[0]), 'final')
        self.assertEqual(t.checkpoint_kind(c.CHECKPOINT_PATHS[2] + '.tmp-' + 'f' * 32), 'temporary')
        for name in (t.OUTPUTS + '/arbitrary.pt', c.CHECKPOINT_PATHS[2] + '.tmp-bad',
                     t.OUTPUTS + '/composite_head0051/checkpoint.pt', t.OUTPUTS + '/model.pth.tmp-' + 'a' * 32):
            self.assertIsNone(t.checkpoint_kind(name))
            f = self.fixture(); put(f.root / name, b'opaque')
            with self.subTest(name=name), self.patches(f), self.assertRaises(ValueError): t._measure_files(f.root, t._walk(f.root))
        f = self.fixture(); design = deepcopy(c.DESIGN); design['state_and_return']['checkpoint_write_cap_bytes_each'] = 4
        with mock.patch.object(c, 'DESIGN', design), self.assertRaisesRegex(ValueError, 'checkpoint file cap'):
            t._measure_files(f.root, [c.CHECKPOINT_PATHS[2] + '.tmp-' + 'a' * 32])
        put(f.root / c.CHECKPOINT_PATHS[2], b'opaque second copy')
        with self.assertRaisesRegex(ValueError, 'both final and temporary'):
            t._measure_files(f.root, t._walk(f.root))

    def test_live_worker_lock_and_low_reserve_refuse_export(self):
        f = self.fixture()
        with self.patches(f), mock.patch.object(t, '_process_rows', return_value=[(123, 'python -m ' + t.MODULE + ' --run')]), self.assertRaises(ValueError):
            t.export_results(f.root, f.sha)
        put(f.root / c.LOCK_NAME, encode({'pid': 123}))
        with self.patches(f), mock.patch.object(t.os, 'kill', return_value=None), self.assertRaises(ValueError):
            t.require_worker_terminal(f.root)
        f = self.fixture()
        with self.patches(f), mock.patch.object(t.shutil, 'disk_usage', return_value=SimpleNamespace(free=t.RESERVE)), self.assertRaises(ValueError):
            t.export_results(f.root, f.sha)

    def test_no_native512_PNG_exception_in_256_only_packet(self):
        f = self.fixture(); name = t.OUTPUTS + '/first_training_example/clean512.png'
        put(f.root / name, b'x' * (c.LIMITS['one256PNG_bytes'] + 1))
        with self.patches(f), self.assertRaisesRegex(ValueError, 'PNG256 file cap'):
            t._measure_files(f.root, t._walk(f.root))

    def test_final_companion_exact_finished_protocol_and_log_bytes(self):
        f = self.fixture()
        with self.patches(f): t.export_results(f.root, f.sha)
        terminal = {'complete': True, 'worker_terminal': True, 'protocol_sha256': f.sha, 'stages': []}
        put(f.root / 'outputs/supervision/finished.json', encode(terminal))
        exact = b'closed export stdout\n\x00'; put(f.root / 'outputs/supervision/export.log', exact)
        with self.patches(f): result = supervisor.final_companion(f.root, [])
        record = c.read_json(result['path'])
        self.assertEqual(base64.b64decode(record['export_log_base64']), exact)
        self.assertEqual(record['finished_json'], terminal)
        self.assertEqual(record['finished_json_sha256'], c.sha256(f.root / 'outputs/supervision/finished.json'))
        self.assertEqual(record['committed_head_updates'], 5)
        self.assertEqual(record['restorer_updates_this_run'], 0)
        self.assertTrue(record['checkpoint_exported']); self.assertFalse(record['model_qualified'])
        with self.patches(f), self.assertRaises(FileExistsError): supervisor.final_companion(f.root, [])
        terminal['protocol_sha256'] = '0' * 64; put(f.root / 'outputs/supervision/finished.json', encode(terminal))
        with self.patches(f), self.assertRaisesRegex(ValueError, 'Final supervision'):
            supervisor.final_companion(f.root, [])

    def test_supervisor_competing_processes_and_exact_caps(self):
        rows = [(1, 'tmux new -d python -m ' + t.MODULE + ' --run'),
                (2, 'timeout 3000 python -B -m ' + t.MODULE + ' --run'),
                (3, 'python scripts/supervise_codeformer_other.py'),
                (4, 'python ordinary_data_audit.py')]
        self.assertEqual([v['pid'] for v in supervisor.competing_commands(rows, 99)], [2, 3])
        self.assertEqual(supervisor.LOG_CAP, 2097152)
        self.assertEqual(t.EXPORT_SECONDS, 600)
        self.assertEqual(sum(c.COUNTS.values()), 3090)
        self.assertEqual((t.PAYLOAD_CAP, t.ARCHIVE_CAP, t.MEMBER_CAP), (3758096384, 3865470566, 9000))

    def test_supervisor_exports_after_failed_run_with_exact_external_caps(self):
        root = self.base / uuid.uuid4().hex / 'new'; root.mkdir(parents=True)
        def stage_result(root, sha, flag, seconds, grace, log):
            return {'stage': flag, 'exit_code': 2 if flag == '--run' else 0,
                    'external_cap_seconds': seconds, 'grace_seconds': grace, 'worker_terminal': True}
        with mock.patch.object(c, 'manual_root', return_value=root), \
             mock.patch.dict(supervisor.os.environ, {'TMUX': 'synthetic CPU fixture'}), \
             mock.patch.object(supervisor.shutil, 'which', return_value='/usr/bin/timeout'), \
             mock.patch.object(supervisor, 'idle', return_value={}), \
             mock.patch.object(supervisor, 'stage', side_effect=stage_result) as stages, \
             mock.patch.object(supervisor, 'final_companion', return_value={}) as final:
            result = supervisor.main(['--root', str(root), '--protocol-sha', 'a' * 64])
        self.assertEqual(result, 1)
        self.assertEqual([(call.args[2], call.args[3], call.args[4]) for call in stages.call_args_list],
                         [('--verify-transfer', 300, 30), ('--run', 3000, 60), ('--export', 600, 30)])
        self.assertEqual(final.call_count, 1)
        receipt = c.read_json(root / 'outputs/supervision/run_receipt.json')
        self.assertIsNone(receipt['committed_head_updates'])
        self.assertTrue(receipt['training_success_not_implied'])
        with mock.patch.object(c, 'manual_root', return_value=root), \
             mock.patch.dict(supervisor.os.environ, {'TMUX': 'synthetic'}), \
             mock.patch.object(supervisor.shutil, 'which', return_value='/usr/bin/timeout'), \
             self.assertRaisesRegex(ValueError, 'no relaunch'):
            supervisor.main(['--root', str(root), '--protocol-sha', 'a' * 64])

    def test_new_root_hardlinks_and_platform_neural_guard(self):
        f = self.fixture()
        try: (f.root / 'linked.py').hardlink_to(f.root / 'research/new.py')
        except OSError as error: self.skipTest('Platform denied synthetic hardlink: ' + str(error))
        with self.assertRaisesRegex(ValueError, 'unlinked regular'): t._walk(f.root)
        with mock.patch.object(c.platform, 'system', return_value='Windows'), \
             self.assertRaisesRegex(ValueError, 'local neural calls forbidden'):
            c.manual_root(f.root)

    def test_platform_exact_home_root_and_tmux_guards_independent_of_link_privilege(self):
        f = self.fixture()
        with mock.patch.object(c.platform, 'system', return_value='Windows'), self.assertRaisesRegex(ValueError, 'local neural'):
            c.manual_root(f.root)
        with mock.patch.object(c.platform, 'system', return_value='Linux'), \
             mock.patch.object(Path, 'home', return_value=f.home), self.assertRaisesRegex(ValueError, 'Unexpected manual VM root'):
            c.manual_root(f.root)
        exact = f.home / 'forensic-dgp' / c.VM_NAME; exact.mkdir(parents=True)
        with mock.patch.object(c.platform, 'system', return_value='Linux'), \
             mock.patch.object(Path, 'home', return_value=f.home), \
             mock.patch.dict(c.os.environ, {'TMUX': ''}), self.assertRaisesRegex(ValueError, 'Manual tmux'):
            c.manual_root(exact)

    def test_hardlink_metadata_refusal_without_creating_a_link(self):
        f = self.fixture(); target = f.root / 'research/new.py'; actual_lstat = Path.lstat
        def linked_info(path, *args, **kwargs):
            info = actual_lstat(path, *args, **kwargs)
            if path == target:
                return SimpleNamespace(st_mode=info.st_mode, st_nlink=2,
                    st_file_attributes=getattr(info, 'st_file_attributes', 0) or 0)
            return info
        with mock.patch.object(Path, 'lstat', linked_info), self.assertRaisesRegex(ValueError, 'Hard-linked packet'):
            t._regular(f.root, 'research/new.py')

    def test_recovered_supervisor_exception_requires_terminal_owned_group(self):
        f = self.fixture(); log = f.root / 'outputs/supervision/synthetic-stage.log'
        fake = SimpleNamespace(pid=4242, stdout=None)
        with mock.patch.object(supervisor.subprocess, 'Popen', return_value=fake), \
             mock.patch.object(supervisor.selectors, 'DefaultSelector', side_effect=RuntimeError('synthetic selector error')), \
             mock.patch.object(supervisor, '_stop_own_group') as stop, \
             mock.patch.object(supervisor, '_group_alive', return_value=False):
            receipt = supervisor.stage(f.root, f.sha, '--run', 3000, 60, log)
        stop.assert_called_once_with(fake, 60)
        self.assertEqual(receipt['exit_code'], 125)
        self.assertEqual(receipt['supervisor_exception']['type'], 'RuntimeError')
        self.assertTrue(receipt['child_process_group_terminal'])
        with mock.patch.object(supervisor.subprocess, 'Popen', return_value=fake), \
             mock.patch.object(supervisor.selectors, 'DefaultSelector', side_effect=RuntimeError('synthetic')), \
             mock.patch.object(supervisor, '_stop_own_group', side_effect=ValueError('cannot prove terminal')), \
             self.assertRaisesRegex(ValueError, 'cannot prove terminal'):
            supervisor.stage(f.root, f.sha, '--run', 3000, 60, log.with_name('synthetic-refused.log'))


if __name__ == '__main__': unittest.main()
