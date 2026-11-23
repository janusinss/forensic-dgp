"""Independent fake-child supervisor tests. No VM checks, Torch or real child launch."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
WORKER=ROOT/'scripts/cctv_completion_codeformer_gradient_v1_vm.py'
SPEC=importlib.util.spec_from_file_location('gradient_supervisor_contract',WORKER)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class FakeChild:
    pid=424242
    def __init__(self,mode,handlers):self.mode=mode;self.returncode=None;self.waits=[];self.handlers=handlers
    def poll(self):return self.returncode
    def wait(self,timeout=None):
        self.waits.append(timeout)
        if len(self.waits)==1:
            if self.mode=='timeout':raise subprocess.TimeoutExpired(['fake-child'],timeout)
            if self.mode=='keyboard':raise KeyboardInterrupt()
            if self.mode=='term':self.handlers[signal.SIGTERM](signal.SIGTERM,None)
            self.returncode=2 if self.mode=='negative-gradient' else 0
        return self.returncode


class GradientSupervisorContract(unittest.TestCase):
    def exercise(self,mode):
        with tempfile.TemporaryDirectory(dir=r'C:\xampp\htdocs\YEAR 4\Testing\outputs') as temporary:
            root=Path(temporary);handlers={};child=FakeChild(mode,handlers);killed=[];previous=object()
            def install(sig,handler):handlers[sig]=handler
            def killpg(pid,sig):self.assertEqual(pid,child.pid);self.assertEqual(sig,signal.SIGKILL);killed.append(pid);child.returncode=-9
            popen=patch.object(M.subprocess,'Popen',side_effect=OSError('synthetic launch failure')) if mode=='launch-failure' else patch.object(M.subprocess,'Popen',return_value=child)
            argv=['worker','--root',str(root),'--protocol-sha','fake-protocol','--run']
            with patch.object(M,'verify',return_value={'files':[]}),patch.object(M,'require_vm',return_value={'gpu':'fake-no-VM'}),popen as launch,patch.object(M.os,'killpg',side_effect=killpg,create=True),patch.object(M.signal,'SIGKILL',9,create=True),patch.object(M.signal,'getsignal',return_value=previous),patch.object(M.signal,'signal',side_effect=install),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                code=M.main()
            receipt=json.loads((root/'supervisor.json').read_text())
            self.assertIs(handlers[signal.SIGTERM],previous)
            self.assertEqual(receipt['optimizer_updates'],0)
            self.assertTrue(receipt['owned_group_cleanup_finally'])
            self.assertTrue(receipt['training_success_not_implied'])
            if mode!='launch-failure':self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertNotIn('torch',sys.modules)
            return code,receipt,child,killed

    def test_normal_and_negative_gradient_statuses_retained(self):
        for mode,expected in [('normal',0),('negative-gradient',2)]:
            with self.subTest(mode=mode):
                code,r,c,k=self.exercise(mode);self.assertEqual(code,expected);self.assertEqual(r['child_exit'],expected);self.assertEqual(r['actual_child_returncode'],expected);self.assertFalse(r['timeout']);self.assertIsNone(r['interruption']);self.assertFalse(k);self.assertEqual(c.waits,[600])

    def test_timeout_kills_only_owned_group_and_records_actual_exit(self):
        code,r,c,k=self.exercise('timeout');self.assertEqual(code,124);self.assertTrue(r['timeout']);self.assertEqual(r['child_exit'],124);self.assertEqual(r['actual_child_returncode'],-9);self.assertEqual(k,[424242]);self.assertEqual(c.waits,[600,15])

    def test_keyboard_and_SIGTERM_cleanup_before_receipt(self):
        for mode,expected,etype in [('keyboard',130,'KeyboardInterrupt'),('term',128+int(signal.SIGTERM),'SystemExit')]:
            with self.subTest(mode=mode):
                code,r,c,k=self.exercise(mode);self.assertEqual(code,expected);self.assertEqual(r['child_exit'],expected);self.assertEqual(r['actual_child_returncode'],-9);self.assertEqual(r['interruption']['type'],etype);self.assertEqual(k,[424242]);self.assertEqual(c.waits,[600,15])

    def test_launch_failure_receipt_never_targets_an_unowned_group(self):
        code,r,c,k=self.exercise('launch-failure');self.assertEqual(code,1);self.assertIsNone(r['child_pid']);self.assertIsNone(r['actual_child_returncode']);self.assertEqual(r['interruption']['type'],'OSError');self.assertFalse(k)

    def test_gradient_math_unchanged_from_retained_unrun_packet(self):
        old=ROOT/'outputs/completion_codeformer_gradient_v1_packet/cctv_completion_codeformer_gradient_v1_vm/scripts/cctv_completion_codeformer_gradient_v1_vm.py'
        a=ast.parse(old.read_text());b=ast.parse(WORKER.read_text())
        def content(tree):return ast.dump(ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='main')],type_ignores=[]),include_attributes=False)
        self.assertEqual(content(a),content(b))


if __name__=='__main__':unittest.main(verbosity=2)
