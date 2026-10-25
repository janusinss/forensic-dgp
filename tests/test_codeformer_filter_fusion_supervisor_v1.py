"""Pure supervisor refusal/closure fixtures; no process or neural launch."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fusion_supervisor', ROOT / 'scripts/supervise_codeformer_filter_fusion_training_v1_vm.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


class SupervisorFixtures(unittest.TestCase):
    def test_import_is_inert(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('third_party.codeformer.codeformer_arch', sys.modules)

    def test_nonlinux_refused_before_any_launch(self):
        with patch.object(S.platform, 'system', return_value='Windows'):
            with patch.object(S.subprocess, 'Popen') as launch:
                with self.assertRaises(ValueError): S.manual_root(ROOT)
                launch.assert_not_called()

    def test_wrong_linux_root_refused(self):
        with patch.object(S.platform, 'system', return_value='Linux'), patch.dict(os.environ, {'TMUX':'manual'}):
            with self.assertRaises((ValueError, FileNotFoundError)): S.manual_root(ROOT)

    def test_missing_process_group_is_terminal(self):
        with patch.object(S.os, 'killpg', create=True, side_effect=ProcessLookupError):
            self.assertFalse(S.group_alive(123456))

    def test_permission_denial_cannot_be_claimed_terminal(self):
        with patch.object(S.os, 'killpg', create=True, side_effect=PermissionError):
            with self.assertRaises(PermissionError): S.group_alive(123456)

    def test_outer_bound_leaves_internal_recovery_time(self):
        self.assertGreater(S.OUTER_WORKER_SECONDS,
                           S.w.STAGES['preflight'] + S.w.STAGES['worker'] + 20)
        self.assertLess(4*((S.LOG_CAP+2)//3)+65536, S.t.LIMITS['one_json_bytes'])


if __name__ == '__main__': unittest.main()
