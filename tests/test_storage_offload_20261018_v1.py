"""Pure command/process-guard fixtures; no network, config secrets or file writes."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('storage_copy_tested', ROOT / 'scripts/storage_offload_20261018_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plan():
    return {'destination': m.DESTINATION,
            'file_list': {'path': 'outputs/storage_offload_20261018_v1/pass1_files.txt'},
            'sha256_list': {'path': 'outputs/storage_offload_20261018_v1/pass1_sha256.txt'}}


def processes(extra=None):
    rows = [{'Name': 'python.exe', 'ProcessId': os.getpid(), 'ParentProcessId': os.getppid(),
             'ExecutablePath': str(Path(getattr(sys, '_base_executable', sys.executable)).absolute()),
             'CommandLine': 'python -B scripts/storage_offload_20261018_v1.py --plan-only'},
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
        self.assertEqual(sum(r['bytes'] for r in m.ARCHIVES.values()), 15463424100)
        self.assertTrue(all('filter-fusion' not in name for name in m.ARCHIVES))

    def test_disjoint_app_allowed_processguard(self):
        with patch.object(m.subprocess, 'run', return_value=processes()):
            result = m.process_guard()
            self.assertTrue(result['complete'])
            self.assertFalse(result['raw_command_lines_exported'])
            self.assertNotIn('app.py', json.dumps(result))

    def test_candidateusage_blocks(self):
        extra = {'Name': 'python.exe', 'ProcessId': 4, 'ParentProcessId': 0,
                 'CommandLine': 'python audit.py outputs/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz'}
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
                 'CommandLine': 'python -B scripts/storage_offload_20261018_v1.py --upload'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            with self.assertRaises(ValueError):
                m.process_guard()

    def test_own_immediate_venv_parent_with_identical_CLI_allowed(self):
        extra = {'Name': 'python.exe', 'ProcessId': os.getppid(), 'ParentProcessId': 999,
                 'ExecutablePath': str(m.ROOT / 'venv/Scripts/python.exe'),
                 'CommandLine': 'python -B scripts/storage_offload_20261018_v1.py --plan-only'}
        with patch.object(m.subprocess, 'run', return_value=processes(extra)):
            result = m.process_guard()
            parent = next(r for r in result['relevant_processes'] if r['pid'] == os.getppid())
            self.assertTrue(parent['verified_own_immediate_venv_parent_launcher'])

    def test_parent_helper_argument_or_image_changes_rejected(self):
        good = {'Name': 'python.exe', 'ProcessId': os.getppid(), 'ParentProcessId': 999,
                'ExecutablePath': str(m.ROOT / 'venv/Scripts/python.exe'),
                'CommandLine': 'python -B scripts/storage_offload_20261018_v1.py --plan-only'}
        for key, value in [('CommandLine', good['CommandLine'].replace('--plan-only', '--upload')),
                           ('ExecutablePath', 'C:/untrusted/python.exe')]:
            extra = {**good, key: value}
            with patch.object(m.subprocess, 'run', return_value=processes(extra)):
                with self.assertRaises(ValueError):
                    m.process_guard()

    def test_same_CLI_venv_not_immediate_parent_blocks(self):
        extra = {'Name': 'python.exe', 'ProcessId': os.getppid() + 9999, 'ParentProcessId': os.getppid(),
                 'ExecutablePath': str(m.ROOT / 'venv/Scripts/python.exe'),
                 'CommandLine': 'python -B scripts/storage_offload_20261018_v1.py --plan-only'}
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


if __name__ == '__main__':
    unittest.main()
