"""Pure command/process-guard fixtures; no network, config secrets or file writes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import copy
import tarfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('storage_pass3_copy_tested', ROOT / 'scripts/storage_offload_pass3_20261018_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(ROOT / 'scripts'))
for alias, file in [('a', 'apply_storage_offload_pass3_20261018_v1.py'), ('p', 'prove_storage_offload_pass3_20261018_v1.py')]:
    spec = importlib.util.spec_from_file_location('storage_pass3_' + alias, ROOT / 'scripts' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals()[alias] = module


def plan():
    return {'destination': m.DESTINATION,
            'file_list': {'path': 'outputs/storage_offload_20261018_v1/pass3/pass3_files.txt'},
            'sha256_list': {'path': 'outputs/storage_offload_20261018_v1/pass3/pass3_sha256.txt'}}


def processes(extra=None):
    rows = [{'Name': 'python.exe', 'ProcessId': os.getpid(), 'ParentProcessId': os.getppid(),
             'ExecutablePath': str(Path(getattr(sys, '_base_executable', sys.executable)).absolute()),
             'CommandLine': 'python -B scripts/storage_offload_pass3_20261018_v1.py --plan-only'},
            {'Name': 'python.exe', 'ProcessId': 2, 'ParentProcessId': 0,
             'CommandLine': 'python -B app.py'},
            {'Name': 'powershell.exe', 'ProcessId': 3, 'ParentProcessId': 1,
             'CommandLine': 'Get-CimInstance Win32_Process'}]
    if extra:
        rows.append(extra)
    return types.SimpleNamespace(returncode=0, stdout=json.dumps(rows))


class Guards(unittest.TestCase):
    def test_exact_upload3scope_destination_and_finite_flags(self):
        command = m.operation_command('upload', plan(), m.OUT / 'fixture')
        self.assertIn('copy', command)
        self.assertIn(m.DESTINATION, command)
        self.assertIn('--immutable', command)
        self.assertIn('--checksum', command)
        self.assertEqual(command[command.index('--transfers') + 1], '2')
        self.assertEqual(command[command.index('--drive-chunk-size') + 1], '32M')
        self.assertEqual(command[command.index('--max-duration') + 1], '8h')
        self.assertEqual(command[command.index('--cutoff-mode') + 1], 'hard')
        self.assertEqual(command[command.index('--files-from-raw') + 1], str(m.ROOT / plan()['file_list']['path']))
        self.assertFalse(set(command) & {'move', 'sync', 'delete', 'purge'})

    def test_verify_is_downloaded_sha256_stream_not_secondcopy(self):
        command = m.operation_command('verify', plan(), m.OUT / 'fixture')
        self.assertIn('check', command)
        self.assertIn('--download', command)
        self.assertIn('--one-way', command)
        self.assertEqual(command[command.index('--checkfile') + 1], 'SHA256')
        self.assertIn('--combined', command)
        self.assertNotIn('copy', command)

    def test_command_cannot_change_destination_or_listpath(self):
        for key, value in [('destination', 'other:unsafe'), ('file_list', {'path': 'outside.txt'})]:
            wrong = plan()
            wrong[key] = value
            with self.assertRaises(ValueError):
                m.operation_command('upload', wrong, m.OUT / 'fixture')

    def test_three_exact_archive_candidates_no_filterfusion(self):
        self.assertEqual(len(m.ARCHIVES), 3)
        self.assertEqual(sum(r['bytes'] for r in m.ARCHIVES.values()), 6661664658)
        self.assertTrue(all('filter-fusion' not in name for name in m.ARCHIVES))

    def test_disjoint_app_allowed_processguard(self):
        with patch.object(m.subprocess, 'run', return_value=processes()):
            result = m.process_guard()
            self.assertTrue(result['complete'])
            self.assertFalse(result['raw_command_lines_exported'])
            self.assertNotIn('app.py', json.dumps(result))

    def test_candidateusage_blocks(self):
        extra = {'Name': 'python.exe', 'ProcessId': 4, 'ParentProcessId': 0,
                 'CommandLine': 'python audit.py outputs/cctv-dgp-bank-comparison-v1-r2-B-results.tar.gz'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            with self.assertRaises(ValueError):
                m.process_guard()

    def test_other_rclone_blocks(self):
        extra = {'Name': 'rclone.exe', 'ProcessId': 4, 'ParentProcessId': 0,
                 'CommandLine': 'rclone copy elsewhere'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            with self.assertRaises(ValueError):
                m.process_guard()

    def test_same_helper_other_PID_blocks(self):
        extra = {'Name': 'python.exe', 'ProcessId': os.getpid() + 9999, 'ParentProcessId': 0,
                 'CommandLine': 'python -B scripts/storage_offload_pass3_20261018_v1.py --upload'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            with self.assertRaises(ValueError):
                m.process_guard()

    def test_own_immediate_venv_parent_with_identical_CLI_allowed(self):
        extra = {'Name': 'python.exe', 'ProcessId': os.getppid(), 'ParentProcessId': 999,
                 'ExecutablePath': str(m.ROOT / 'venv/Scripts/python.exe'),
                 'CommandLine': 'python -B scripts/storage_offload_pass3_20261018_v1.py --plan-only'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            result = m.process_guard()
            parent = next(r for r in result['relevant_processes'] if r['pid'] == os.getppid())
            self.assertTrue(parent['verified_own_immediate_venv_parent_launcher'])

    def test_parent_helper_argument_or_image_changes_rejected(self):
        good = {'Name': 'python.exe', 'ProcessId': os.getppid(), 'ParentProcessId': 999,
                'ExecutablePath': str(m.ROOT / 'venv/Scripts/python.exe'),
                'CommandLine': 'python -B scripts/storage_offload_pass3_20261018_v1.py --plan-only'}
        for key, value in [('CommandLine', good['CommandLine'].replace('--plan-only', '--upload')),
                           ('ExecutablePath', 'C:/untrusted/python.exe')]:
            extra = {**good, key: value}
            with patch.object(m.subprocess, 'run', return_value=processes(extra)):
                with self.assertRaises(ValueError):
                    m.process_guard()

    def test_same_CLI_venv_not_immediate_parent_blocks(self):
        extra = {'Name': 'python.exe', 'ProcessId': os.getppid() + 9999, 'ParentProcessId': os.getppid(),
                 'ExecutablePath': str(m.ROOT / 'venv/Scripts/python.exe'),
                 'CommandLine': 'python -B scripts/storage_offload_pass3_20261018_v1.py --plan-only'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            with self.assertRaises(ValueError):
                m.process_guard()

    def test_own_parent_or_executable_metadata_mismatch_rejected(self):
        for key, value in [('ParentProcessId', os.getppid() + 1), ('ExecutablePath', None),
                           ('ExecutablePath', 'C:/untrusted/python.exe')]:
            result = processes()
            rows = json.loads(result.stdout)
            rows[0][key] = value
            result.stdout = json.dumps(rows)
            with patch.object(m.subprocess, 'run', return_value=result):
                with self.assertRaises(ValueError):
                    m.process_guard()

    def test_uninspectable_or_failed_CIM_rejects(self):
        extra = {'Name': 'python.exe', 'ProcessId': 4, 'ParentProcessId': 0, 'CommandLine': None}
        for result in [processes(extra), types.SimpleNamespace(returncode=1, stdout='')]:
            with patch.object(m.subprocess, 'run', return_value=result):
                with self.assertRaises(ValueError):
                    m.process_guard()

    def test_no_neural_import_or_apply_function(self):
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(any(hasattr(m, name) for name in ('apply', 'delete', 'remove', 'unlink')))


class Pass3Scope(unittest.TestCase):
    def test_only_these_new_three_scope_not_pass1_or_latest_return(self):
        self.assertEqual(set(m.ARCHIVES), set(p.PREFIXES))
        self.assertTrue(set(m.ARCHIVES).isdisjoint(a.PRIOR_PASS1_REMOVED))
        self.assertEqual(m.OUT, ROOT / 'outputs/storage_offload_20261018_v1/pass3')
        self.assertTrue(m.DESTINATION.endswith('/20261018_v1/pass3'))

    def test_exact_prefix_and_PAX_metadata_maps_whole_name(self):
        for prefix in p.PREFIXES.values():
            member = tarfile.TarInfo(prefix + '/data/source.png')
            member.pax_headers = {'mtime': '1791652636.12', 'path': member.name}
            self.assertEqual(p.mapped_member_name(member, prefix), 'data/source.png')

    def test_pax_link_traversal_and_wrong_prefix_rejected(self):
        prefix = next(iter(p.PREFIXES.values()))
        for kind in ('wrong_prefix', 'traversal', 'pax_link', 'symlink', 'nonfinite_time'):
            member = tarfile.TarInfo(prefix + '/file.txt')
            if kind == 'wrong_prefix': member.name = 'other/file.txt'
            if kind == 'traversal': member.name = prefix + '/../file.txt'
            if kind == 'pax_link': member.pax_headers = {'linkpath': 'x'}
            if kind == 'symlink': member.type = tarfile.SYMTYPE
            if kind == 'nonfinite_time': member.pax_headers = {'mtime': 'inf'}
            with self.assertRaises(ValueError): p.mapped_member_name(member, prefix)

    def test_namespace_requires_exact_all_six_absent_and_preserves_other_files(self):
        archives = set(m.ARCHIVES) | a.PRIOR_PASS1_REMOVED
        before = {name: {'bytes': 10, 'mtime_ns': 20} for name in archives}
        before['outputs/retained/checkpoint.pth'] = {'bytes': 30, 'mtime_ns': 40}
        before['outputs/storage_offload_20261018_v1/old_progress.json'] = {'bytes': 1, 'mtime_ns': 2}
        current = {k: dict(v) for k,v in before.items() if k not in archives}
        current['outputs/storage_offload_20261018_v1/old_progress.json']['bytes'] = 99
        self.assertTrue(a.namespace_delta(before, current)['complete'])
        for bad in ('restore_one', 'drop_checkpoint', 'change_checkpoint'):
            wrong = copy.deepcopy(current)
            if bad == 'restore_one': wrong[next(iter(archives))] = {'bytes': 10, 'mtime_ns': 20}
            if bad == 'drop_checkpoint': del wrong['outputs/retained/checkpoint.pth']
            if bad == 'change_checkpoint': wrong['outputs/retained/checkpoint.pth']['mtime_ns'] += 1
            self.assertFalse(a.namespace_delta(before, wrong)['complete'])

    def test_remote_three_schema_SHA_key_normalization_and_mismatch_stops(self):
        plan = {'files': [{'path':n, **r} for n,r in m.ARCHIVES.items()]}
        rows = [{'Path':r['path'], 'Size':r['bytes'], 'Hashes':{'SHA-256':r['sha256']},
                 'ID':'fixture-' + str(i), 'IsDir':False} for i,r in enumerate(plan['files'])]
        a.remote_rows_check(plan, rows)
        for bad in ('duplicate_id', 'duplicate_path', 'wrong_sha', 'directory'):
            wrong = copy.deepcopy(rows)
            if bad == 'duplicate_id': wrong[1]['ID'] = wrong[0]['ID']
            if bad == 'duplicate_path': wrong[1]['Path'] = wrong[0]['Path']
            if bad == 'wrong_sha': wrong[0]['Hashes']['SHA-256'] = '0'*64
            if bad == 'directory': wrong[0]['IsDir'] = True
            with self.assertRaises(ValueError): a.remote_rows_check(plan, wrong)

    def test_incomplete_or_other_scope_member_proof_cannot_enable_removal(self):
        rows = [{'archive_path':n,'archive_sha256':r['sha256'],'archive_bytes':r['bytes'],
                 'retained_root':r['retained_root'],'full_member_retention_verified':True} for n,r in m.ARCHIVES.items()]
        good = {'complete':True,'full_member_retention_verified':True,'archives':rows}
        with patch.object(m,'sha',return_value='a'*64), patch.object(m,'read',return_value=good), patch.object(m,'pin',return_value={'pin':True}):
            self.assertEqual(m.retention_proof(ROOT/'fixture.json','a'*64), {'pin':True})
        for bad in ('partial','one_missing','wrong_archive','wrong_retained_root'):
            wrong=copy.deepcopy(good)
            if bad=='partial':wrong['full_member_retention_verified']=False
            if bad=='one_missing':wrong['archives'].pop()
            if bad=='wrong_archive':wrong['archives'][0]['archive_path']='outputs/not-authorized.tar.gz'
            if bad=='wrong_retained_root':wrong['archives'][0]['retained_root']='outputs/other'
            with patch.object(m,'sha',return_value='a'*64), patch.object(m,'read',return_value=wrong):
                with self.assertRaises(ValueError):m.retention_proof(ROOT/'fixture.json','a'*64)

    def test_proof_all_three_flag_requires_actual_full_member_closures(self):
        good={'complete':True,'full_member_retention_verified':True}
        self.assertTrue(p.full_retention_verified([good.copy() for _ in range(3)]))
        self.assertFalse(p.full_retention_verified([good.copy(),good.copy()]))
        self.assertFalse(p.full_retention_verified([good.copy(),good.copy(),{'complete':False}]))


if __name__ == '__main__':
    unittest.main()
