"""Independent Windows guard probes: only test-owned harmless processes."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from ctypes import wintypes as W

ROOT=Path(__file__).resolve().parents[1]
S=importlib.util.spec_from_file_location('independent_guard_contract',ROOT/'completion_process_guard_v1.py')
G=importlib.util.module_from_spec(S);S.loader.exec_module(G)
K=ctypes.WinDLL('kernel32',use_last_error=True) if os.name=='nt' else None
if K:
 K.GetHandleInformation.argtypes=[W.HANDLE,ctypes.POINTER(W.DWORD)];K.GetHandleInformation.restype=W.BOOL
 K.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];K.OpenProcess.restype=W.HANDLE
 K.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD];K.WaitForSingleObject.restype=W.DWORD
 K.CloseHandle.argtypes=[W.HANDLE];K.CloseHandle.restype=W.BOOL


def live(pid):
 h=K.OpenProcess(0x00100000,False,pid)
 if not h:return False
 try:return K.WaitForSingleObject(h,0)==0x102
 finally:K.CloseHandle(h)


def ready(path):
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  try:return json.loads(path.read_text())
  except (OSError,ValueError):time.sleep(.02)
 raise AssertionError('Harmless owned test did not become ready')


@unittest.skipUnless(os.name=='nt','Windows lifecycle ownership')
class IndependentWindowsGuard(unittest.TestCase):
 def test_venv_actual_worker_and_descendant_stop_while_unrelated_survives(self):
  with tempfile.TemporaryDirectory(dir=r'C:\xampp\htdocs\YEAR 4\Testing\outputs') as folder:
   p=Path(folder);marker=p/'ready.json';unrelated=subprocess.Popen([sys._base_executable,'-B','-c','import time;time.sleep(60)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
   code="import json,os,pathlib,subprocess,sys,time;leaf=subprocess.Popen([sys.executable,'-B','-c','import time;allocation=bytearray(24*1024*1024);time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(json.dumps({'actual':os.getpid(),'leaf_launcher':leaf.pid}));time.sleep(60)"
   try:
    with G.OwnedWorker([sys.executable,'-B','-c',code,str(marker)]) as owned:
     ids=ready(marker);self.assertNotEqual(ids['actual'],owned.pid,'This test intentionally exercises the installed Windows venv redirector')
     self.assertTrue(live(ids['actual']));self.assertTrue(live(ids['leaf_launcher']));self.assertTrue(live(unrelated.pid));self.assertGreaterEqual(owned.active_processes(),3)
     time.sleep(.2);self.assertGreaterEqual(owned.memory_usage()['peak_job_private_bytes'],24*1024*1024)
     with self.assertRaises(subprocess.TimeoutExpired):owned.wait(timeout=.02)
     owned.kill();owned.wait_tree(timeout=5)
     self.assertEqual(owned.active_processes(),0);self.assertFalse(live(ids['actual']));self.assertFalse(live(ids['leaf_launcher']));self.assertTrue(live(unrelated.pid))
   finally:
    if unrelated.poll() is None:unrelated.kill()
    unrelated.wait(timeout=5)

 def test_job_process_handles_noninheritable_and_callers_flags_unchanged(self):
  with tempfile.TemporaryDirectory(dir=r'C:\xampp\htdocs\YEAR 4\Testing\outputs') as folder:
   logfile=Path(folder)/'out.txt'
   with logfile.open('wb') as out:
    caller=G.msvcrt.get_osfhandle(out.fileno());flags=W.DWORD();self.assertTrue(K.GetHandleInformation(caller,ctypes.byref(flags)));before=flags.value
    with G.OwnedWorker([sys.executable,'-B','-c',"print('NORMAL_OUTPUT',flush=True)"],stdout=out,stderr=subprocess.STDOUT) as owned:
     for h in (owned._job,owned._process):
      self.assertTrue(K.GetHandleInformation(h,ctypes.byref(flags)));self.assertEqual(flags.value&1,0)
     self.assertTrue(K.GetHandleInformation(caller,ctypes.byref(flags)));self.assertEqual(flags.value,before)
     self.assertEqual(owned.wait(timeout=10),0);self.assertEqual(owned.wait_tree(timeout=5),0)
    self.assertTrue(K.GetHandleInformation(caller,ctypes.byref(flags)));self.assertEqual(flags.value,before)
   self.assertEqual(logfile.read_text().strip(),'NORMAL_OUTPUT')


if __name__=='__main__':unittest.main()
