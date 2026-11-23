"""Harmless subprocess ownership tests. No model, inference or existing-PID kills."""
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("guard_under_test", ROOT / "completion_process_guard_v1.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
TEMP_PARENT = Path(r"C:\xampp\htdocs\YEAR 4\Testing\outputs") if os.name == "nt" else None


def wait_json(path, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            try: return json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError): pass
        time.sleep(.02)
    raise AssertionError("Harmless child did not publish readiness")


def running(pid):
    if os.name == "nt":
        from ctypes import wintypes as W
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]; kernel.OpenProcess.restype = W.HANDLE
        kernel.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]; kernel.WaitForSingleObject.restype = W.DWORD
        kernel.CloseHandle.argtypes = [W.HANDLE]; kernel.CloseHandle.restype = W.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle: return False
        try: return kernel.WaitForSingleObject(handle, 0) == 0x00000102
        finally: kernel.CloseHandle(handle)
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False


class GuardContractTests(unittest.TestCase):
    def test_normal_exit_quoting_env_and_log(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            path = Path(folder)
            env = dict(os.environ); env["GUARD_TEST_VALUE"] = "custom value"
            code = "import os,sys; print(repr(sys.argv[1]),os.environ['GUARD_TEST_VALUE'],flush=True); print('errline',file=sys.stderr,flush=True)"
            with (path / "log.txt").open("wb") as log:
                with GUARD.OwnedWorker([sys.executable, "-B", "-c", code, 'spaces and "quotes" and slash\\'], cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT) as child:
                    self.assertEqual(child.wait(timeout=10), 0)
                    self.assertEqual(child.wait_tree(timeout=5), 0)
                    self.assertEqual(child.active_processes(), 0)
            data = (path / "log.txt").read_text(encoding="utf-8")
            self.assertIn("custom value", data); self.assertIn("errline", data); self.assertIn("quotes", data)

    def test_timeout_kills_owned_child_and_grandchild(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            path = Path(folder); ready = path / "ready.json"
            code = "import json,os,pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(json.dumps({'root':os.getpid(),'child':p.pid}),encoding='utf-8'); time.sleep(60)"
            with GUARD.OwnedWorker([sys.executable, "-B", "-c", code, str(ready)]) as child:
                pids = wait_json(ready)
                self.assertTrue(running(pids["root"])); self.assertTrue(running(pids["child"]))
                if os.name == "nt": self.assertGreaterEqual(child.active_processes(), 2)
                with self.assertRaises(subprocess.TimeoutExpired): child.wait(timeout=.05)
                child.kill(); child.wait_tree(timeout=5)
                self.assertFalse(running(pids["root"])); self.assertFalse(running(pids["child"]))

    def test_close_kills_descendant_after_root_exit(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            path = Path(folder); ready = path / "ready.json"
            code = "import json,pathlib,subprocess,sys; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(json.dumps({'child':p.pid}),encoding='utf-8')"
            child = GUARD.OwnedWorker([sys.executable, "-B", "-c", code, str(ready)])
            try:
                pids = wait_json(ready); self.assertEqual(child.wait(timeout=10), 0)
                self.assertTrue(running(pids["child"]))
            finally: child.close()
            deadline = time.monotonic() + 5
            while running(pids["child"]) and time.monotonic() < deadline: time.sleep(.02)
            self.assertFalse(running(pids["child"]))
            child.close()  # Idempotent.

    def test_invalid_start_is_fail_closed(self):
        with self.assertRaises(ValueError): GUARD.OwnedWorker("python -c anything")
        with self.assertRaises(ValueError): GUARD.OwnedWorker(["python", "-c", "pass"])
        with self.assertRaises(ValueError): GUARD.OwnedWorker([sys.executable, "\0"])
        if os.name == "nt":
            with self.assertRaises(ValueError): GUARD.OwnedWorker([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object semantics")
    def test_assignment_failure_never_resumes_worker(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            marker = Path(folder) / "must_not_exist.txt"
            code = "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('USER_CODE_RAN')"
            with patch.object(GUARD._K, "AssignProcessToJobObject", return_value=0), patch.object(GUARD._K, "ResumeThread", wraps=GUARD._K.ResumeThread) as resumed:
                with self.assertRaises(OSError): GUARD.OwnedWorker([sys.executable, "-B", "-c", code, str(marker)])
                resumed.assert_not_called()
            self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "Windows Job Object kill-on-close semantics")
    def test_supervisor_crash_closes_job_and_kills_descendants(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            path = Path(folder); ready = path / "ready.json"
            leaf_code = "import json,os,pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(json.dumps({'root':os.getpid(),'child':p.pid}),encoding='utf-8'); time.sleep(60)"
            owner_code = "import importlib.util,sys,time; s=importlib.util.spec_from_file_location('owned_guard',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); w=m.OwnedWorker([sys.argv[2],'-B','-c',sys.argv[3],sys.argv[4]]); time.sleep(60)"
            # This harmless stdlib-only owner uses the actual interpreter rather than
            # a venv redirector so Popen.kill() here intentionally kills the owner.
            owner = subprocess.Popen([sys._base_executable, "-B", "-c", owner_code,
                                      str(ROOT / "completion_process_guard_v1.py"), sys.executable, leaf_code, str(ready)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                pids = wait_json(ready)
                self.assertTrue(running(pids["root"])); self.assertTrue(running(pids["child"]))
                owner.kill(); owner.wait(timeout=5)
                deadline = time.monotonic() + 5
                while (running(pids["root"]) or running(pids["child"])) and time.monotonic() < deadline: time.sleep(.02)
                self.assertFalse(running(pids["root"])); self.assertFalse(running(pids["child"]))
            finally:
                if owner.poll() is None: owner.kill()
                owner.wait(timeout=5)

    @unittest.skipUnless(os.name == "nt", "Windows explicit handle inheritance")
    def test_unrelated_inheritable_handle_is_not_forwarded(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            path = Path(folder)
            with (path / "unrelated.txt").open("wb") as sentinel, (path / "log.txt").open("wb") as log:
                handle = GUARD.msvcrt.get_osfhandle(sentinel.fileno())
                os.set_handle_inheritable(handle, True)
                code = "\n".join([
                    "import ctypes,os,sys", "from ctypes import wintypes as W",
                    "k=ctypes.WinDLL('kernel32',use_last_error=True)",
                    "k.GetFinalPathNameByHandleW.argtypes=[W.HANDLE,W.LPWSTR,W.DWORD,W.DWORD]",
                    "b=ctypes.create_unicode_buffer(32768)",
                    "try: n=k.GetFinalPathNameByHandleW(int(sys.argv[1]),b,len(b),0)",
                    "except OSError: n=0",  # Invalid noninherited handle under Windows verifier.
                    "same=bool(n) and os.path.basename(b.value).lower()=='unrelated.txt'",
                    "print('INHERITED' if same else 'NOT_INHERITED',flush=True)",
                    "raise SystemExit(1 if same else 0)"])
                try:
                    with GUARD.OwnedWorker([sys.executable, "-B", "-c", code, str(handle)], stdout=log, stderr=subprocess.STDOUT) as worker:
                        self.assertEqual(worker.wait(timeout=10), 0)
                finally: os.set_handle_inheritable(handle, False)
            self.assertIn("NOT_INHERITED", (path / "log.txt").read_text(encoding="utf-8"))
            self.assertEqual((path / "unrelated.txt").read_bytes(), b"")

    @unittest.skipUnless(os.name == "nt", "Windows Job Object memory accounting")
    def test_memory_usage_is_job_private_commit(self):
        with tempfile.TemporaryDirectory(dir=TEMP_PARENT) as folder:
            ready = Path(folder) / "ready.json"
            code = "import json,pathlib,sys,time; allocation=bytearray(12*1024*1024); pathlib.Path(sys.argv[1]).write_text('{}',encoding='utf-8'); time.sleep(60)"
            with GUARD.OwnedWorker([sys.executable, "-B", "-c", code, str(ready)]) as child:
                wait_json(ready)
                usage = child.memory_usage()
                self.assertGreaterEqual(usage["peak_process_private_bytes"], 12*1024*1024)
                self.assertGreaterEqual(usage["peak_job_private_bytes"], usage["peak_process_private_bytes"])
                self.assertIn("not RSS/minRAM certification", usage["scope"])
                child.kill(); child.wait_tree(timeout=5)
                self.assertGreaterEqual(child.memory_usage()["peak_job_private_bytes"], usage["peak_job_private_bytes"])

    @unittest.skipUnless(os.name == "nt", "Windows Job Object semantics")
    def test_job_handles_are_not_inherited_and_repeated_runs_close_handles(self):
        kernel = GUARD._K
        from ctypes import wintypes as W
        kernel.GetProcessHandleCount.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]; kernel.GetProcessHandleCount.restype = W.BOOL
        count = W.DWORD(); kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(count)); before = count.value
        for _ in range(6):
            with GUARD.OwnedWorker([sys.executable, "-B", "-c", "pass"]) as child:
                self.assertEqual(child.wait(timeout=10), 0)
        kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(count))
        self.assertLessEqual(count.value, before + 1)


if __name__ == "__main__":
    unittest.main()
