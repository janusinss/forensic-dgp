"""Own a local worker process tree, including Windows venv Python redirectors.

with OwnedWorker(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT) as worker:
    try:
        worker.wait(timeout=120)
    except subprocess.TimeoutExpired:
        worker.kill()                 # Terminates the owned tree, not just its root.
        worker.wait_tree(timeout=5)
        raise

Windows: CreateProcessW starts suspended; assign a private noninheritable Job
Object before resuming. KILL_ON_JOB_CLOSE also handles supervisor termination.
Only duplicated stdin/out/err handles are inherited. Job assignment failure is
fatal; there is no unguarded fallback. POSIX uses a new session and killpg.
This is lifecycle ownership for trusted workers, not a hostile-code sandbox.
No shell, process-name scans, PID-tree guesses, dependencies or neural imports.

Microsoft references:
https://learn.microsoft.com/windows/win32/procthread/job-objects
https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw
https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute
"""
import ctypes
import math
import os
from pathlib import Path
import signal
import subprocess
import time


if os.name == "nt":
    from ctypes import wintypes as W
    import msvcrt

    class _STARTUPINFO(ctypes.Structure):
        _fields_ = [("cb", W.DWORD), ("lpReserved", W.LPWSTR), ("lpDesktop", W.LPWSTR),
                    ("lpTitle", W.LPWSTR), ("dwX", W.DWORD), ("dwY", W.DWORD),
                    ("dwXSize", W.DWORD), ("dwYSize", W.DWORD), ("dwXCountChars", W.DWORD),
                    ("dwYCountChars", W.DWORD), ("dwFillAttribute", W.DWORD),
                    ("dwFlags", W.DWORD), ("wShowWindow", W.WORD), ("cbReserved2", W.WORD),
                    ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
                    ("hStdInput", W.HANDLE), ("hStdOutput", W.HANDLE), ("hStdError", W.HANDLE)]

    class _STARTUPINFOEX(ctypes.Structure):
        _fields_ = [("StartupInfo", _STARTUPINFO), ("lpAttributeList", ctypes.c_void_p)]

    class _PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [("hProcess", W.HANDLE), ("hThread", W.HANDLE),
                    ("dwProcessId", W.DWORD), ("dwThreadId", W.DWORD)]

    class _BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", W.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", W.DWORD), ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                     "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMIT), ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _ACCOUNTING(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in
                    ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")]
        _fields_ += [(name, W.DWORD) for name in
                     ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]

    _K = ctypes.WinDLL("kernel32", use_last_error=True)
    for _name, _args, _result in [
        ("CreateJobObjectW", [ctypes.c_void_p, W.LPCWSTR], W.HANDLE),
        ("SetInformationJobObject", [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD], W.BOOL),
        ("AssignProcessToJobObject", [W.HANDLE, W.HANDLE], W.BOOL),
        ("TerminateJobObject", [W.HANDLE, W.UINT], W.BOOL),
        ("QueryInformationJobObject", [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p], W.BOOL),
        ("GetCurrentProcess", [], W.HANDLE),
        ("OpenProcess", [W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
        ("DuplicateHandle", [W.HANDLE, W.HANDLE, W.HANDLE, ctypes.POINTER(W.HANDLE), W.DWORD, W.BOOL, W.DWORD], W.BOOL),
        ("InitializeProcThreadAttributeList", [ctypes.c_void_p, W.DWORD, W.DWORD, ctypes.POINTER(ctypes.c_size_t)], W.BOOL),
        ("UpdateProcThreadAttribute", [ctypes.c_void_p, W.DWORD, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p], W.BOOL),
        ("DeleteProcThreadAttributeList", [ctypes.c_void_p], None),
        ("CreateProcessW", [W.LPCWSTR, W.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, W.BOOL,
                            W.DWORD, ctypes.c_void_p, W.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(_PROCESS_INFORMATION)], W.BOOL),
        ("ResumeThread", [W.HANDLE], W.DWORD),
        ("TerminateProcess", [W.HANDLE, W.UINT], W.BOOL),
        ("WaitForSingleObject", [W.HANDLE, W.DWORD], W.DWORD),
        ("GetExitCodeProcess", [W.HANDLE, ctypes.POINTER(W.DWORD)], W.BOOL),
        ("CloseHandle", [W.HANDLE], W.BOOL),
    ]:
        getattr(_K, _name).argtypes = _args
        getattr(_K, _name).restype = _result

    def _checked(ok, operation):
        if not ok:
            error = ctypes.get_last_error()
            raise OSError(error, operation + ": " + ctypes.FormatError(error))

    def _duplicate_file(file, readable=False):
        # A new inheritable handle avoids changing the caller's handle flags.
        own_file = open(os.devnull, "rb" if readable else "wb") if file is None or file == subprocess.DEVNULL else None
        try:
            source = own_file if own_file is not None else file
            if not hasattr(source, "fileno"):
                raise ValueError("Use an open file or DEVNULL for stdout/stderr; PIPE is unsupported")
            if hasattr(source, "flush"):
                source.flush()
            duplicated = W.HANDLE()
            current = _K.GetCurrentProcess()
            _checked(_K.DuplicateHandle(current, msvcrt.get_osfhandle(source.fileno()), current,
                                        ctypes.byref(duplicated), 0, True, 2), "DuplicateHandle")
            return duplicated.value
        finally:
            if own_file is not None:
                own_file.close()


class OwnedWorker:
    """Popen-like poll/wait/kill plus explicit descendant ownership and close.

    argv[0] must be an existing absolute executable path. stdout/stderr accept
    open files or DEVNULL; stderr additionally accepts subprocess.STDOUT. None
    means DEVNULL. wait() waits for the root process; wait_tree() verifies all
    owned descendants exited. Always close() or use the context manager.
    """
    def __init__(self, argv, *, cwd=None, env=None, stdout=None, stderr=None):
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("A nonempty argument sequence is required; shell strings are unsupported")
        self.args = [os.fspath(arg) for arg in argv]
        if any(not isinstance(arg, str) or "\0" in arg for arg in self.args):
            raise ValueError("Arguments must be strings without NUL")
        executable = Path(self.args[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("argv[0] must be an existing absolute executable path")
        if cwd is not None and not Path(cwd).is_dir():
            raise ValueError("cwd must be an existing directory")
        self.pid, self.returncode = None, None
        self._process = self._job = self._popen = None
        self._member_handles = {}
        self._closed = False
        self.ownership = "windows-job-suspended-before-resume" if os.name == "nt" else "posix-new-session"
        try:
            if os.name == "nt":
                self._windows_start(cwd, env, stdout, stderr)
            else:
                self._popen = subprocess.Popen(self.args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                                stdout=subprocess.DEVNULL if stdout is None else stdout,
                                                stderr=subprocess.DEVNULL if stderr is None else stderr,
                                                start_new_session=True, close_fds=True)
                self.pid = self._popen.pid
        except BaseException:
            self.close()
            raise

    def _windows_start(self, cwd, env, stdout, stderr):
        self._job = _K.CreateJobObjectW(None, None)
        _checked(self._job, "CreateJobObjectW")
        limits = _EXTENDED_LIMIT()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE; no breakaway flags.
        _checked(_K.SetInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)), "SetInformationJobObject")
        handles, attributes, attribute_initialized = [], None, False
        process_info = _PROCESS_INFORMATION()
        assigned = False
        try:
            stdin_handle = _duplicate_file(None, readable=True); handles.append(stdin_handle)
            stdout_handle = _duplicate_file(stdout); handles.append(stdout_handle)
            stderr_handle = stdout_handle if stderr == subprocess.STDOUT else _duplicate_file(stderr)
            if stderr_handle not in handles: handles.append(stderr_handle)
            startup = _STARTUPINFOEX()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x00000101  # USESTDHANDLES | USESHOWWINDOW; hidden.
            startup.StartupInfo.wShowWindow = 0
            startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = stdin_handle, stdout_handle, stderr_handle
            size = ctypes.c_size_t()
            _K.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))  # Expected size-query failure.
            _checked(size.value, "Attribute-list size query")
            attributes = ctypes.create_string_buffer(size.value)
            _checked(_K.InitializeProcThreadAttributeList(attributes, 1, 0, ctypes.byref(size)), "InitializeProcThreadAttributeList")
            attribute_initialized = True
            startup.lpAttributeList = ctypes.cast(attributes, ctypes.c_void_p)
            inherited = (W.HANDLE * len(handles))(*handles)
            _checked(_K.UpdateProcThreadAttribute(attributes, 0, 0x00020002, inherited,
                                                  ctypes.sizeof(inherited), None, None), "Explicit handle inheritance list")
            environment = None
            if env is not None:
                entries = []
                for key, value in sorted(env.items(), key=lambda item: item[0].upper()):
                    if not isinstance(key, str) or not isinstance(value, str) or not key or "=" in key or "\0" in key + value:
                        raise ValueError("Environment must map valid string keys to string values")
                    entries.append(key + "=" + value)
                environment = ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(self.args))
            flags = 0x00000004 | 0x00000400 | 0x00080000 | 0x08000000
            _checked(_K.CreateProcessW(self.args[0], command, None, None, True, flags, environment,
                                       None if cwd is None else os.fspath(cwd), ctypes.byref(startup), ctypes.byref(process_info)), "CreateProcessW suspended")
            self._process, self.pid = process_info.hProcess, int(process_info.dwProcessId)
            # No application code (including venv redirector) runs before this succeeds.
            _checked(_K.AssignProcessToJobObject(self._job, self._process), "AssignProcessToJobObject before resume")
            assigned = True
            _checked(_K.ResumeThread(process_info.hThread) != 0xFFFFFFFF, "ResumeThread")
        except BaseException:
            if self._process:
                if assigned: _K.TerminateJobObject(self._job, 1)
                else: _K.TerminateProcess(self._process, 1)  # Still suspended, so no descendants exist.
                _K.WaitForSingleObject(self._process, 5000)
            raise
        finally:
            if process_info.hThread: _K.CloseHandle(process_info.hThread)
            if attribute_initialized: _K.DeleteProcThreadAttributeList(attributes)
            for handle in handles: _K.CloseHandle(handle)

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self._closed:
            raise RuntimeError("Worker is closed")
        if os.name != "nt":
            self.returncode = self._popen.poll()
        else:
            result = _K.WaitForSingleObject(self._process, 0)
            if result == 0x00000102: return None
            _checked(result == 0, "Poll process handle")
            code = W.DWORD(); _checked(_K.GetExitCodeProcess(self._process, ctypes.byref(code)), "GetExitCodeProcess")
            self.returncode = int(code.value)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None: return self.returncode
        if self._closed: raise RuntimeError("Worker is closed")
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        if os.name != "nt":
            self.returncode = self._popen.wait(timeout=timeout)
        else:
            milliseconds = 0xFFFFFFFF if timeout is None else min(0xFFFFFFFE, math.ceil(timeout * 1000))
            result = _K.WaitForSingleObject(self._process, milliseconds)
            if result == 0x00000102: raise subprocess.TimeoutExpired(self.args, timeout)
            _checked(result == 0, "Wait process handle")
            self.poll()
        return self.returncode

    def active_processes(self):
        if self._closed: return 0
        if os.name == "nt":
            accounting = _ACCOUNTING()
            _checked(_K.QueryInformationJobObject(self._job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None), "Query owned Job Object")
            return int(accounting.ActiveProcesses)
        try:
            os.killpg(self.pid, 0)
            return 1  # POSIX reports group existence, not an exact process count.
        except ProcessLookupError:
            return 0

    def memory_usage(self):
        """Peak private commit recorded by this job, including owned descendants."""
        if os.name != "nt": return None
        if self._closed: raise RuntimeError("Worker is closed; read job memory before close")
        limits = _EXTENDED_LIMIT()
        _checked(_K.QueryInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None), "Query job peak private commit")
        return {"peak_process_private_bytes": int(limits.PeakProcessMemoryUsed),
                "peak_job_private_bytes": int(limits.PeakJobMemoryUsed),
                "scope": "Job Object commit/private usage including descendants; not RSS/minRAM certification"}

    def _remember_members(self):
        """Retain process objects before termination; job accounting can settle first."""
        capacity = 16
        while True:
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", W.DWORD), ("count", W.DWORD),
                            ("ids", ctypes.c_size_t * capacity)]
            listing = ProcessIds()
            ok = _K.QueryInformationJobObject(self._job, 3, ctypes.byref(listing), ctypes.sizeof(listing), None)
            if ok: break
            if ctypes.get_last_error() != 234 or capacity >= 65536:
                _checked(False, "Query owned Job Object process IDs")
            capacity = min(65536, max(capacity * 2, listing.assigned + 8))
        for pid in listing.ids[:listing.count]:
            if pid not in self._member_handles:
                handle = _K.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only.
                if handle:
                    self._member_handles[pid] = handle
                elif ctypes.get_last_error() != 87:  # An exited process may already be gone.
                    _checked(False, "Open owned job member for exit verification")

    def _members_pending(self):
        for pid, handle in list(self._member_handles.items()):
            result = _K.WaitForSingleObject(handle, 0)
            if result == 0:
                _K.CloseHandle(handle)
                del self._member_handles[pid]
            elif result != 0x00000102:
                _checked(False, "Wait owned job member")
        return bool(self._member_handles)

    def wait_tree(self, timeout=5):
        if not math.isfinite(timeout) or timeout < 0: raise ValueError("timeout must be finite and nonnegative")
        deadline = time.monotonic() + timeout
        if os.name == "nt" and not self._closed:
            self._remember_members()
        while self.active_processes() or (os.name == "nt" and self._members_pending()):
            self.poll()
            if time.monotonic() >= deadline: raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(.02)
        return self.wait(timeout=max(0, deadline - time.monotonic()))

    def kill(self):
        if self._closed: return
        if os.name == "nt":
            if self._job:
                try: self._remember_members()
                finally: _checked(_K.TerminateJobObject(self._job, 1), "Terminate owned Job Object")
        elif self.pid is not None:
            try: os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError: pass

    terminate = kill

    def close(self):
        """Release ownership; terminate remaining descendants even after root exit."""
        if self._closed: return
        try:
            if os.name == "nt":
                try:
                    if self._job and self._process:
                        self.kill()
                        self.wait_tree(timeout=5)
                finally:
                    if self._job:
                        job, self._job = self._job, None
                        _checked(_K.CloseHandle(job), "Close owned Job Object")
                    if self._process:
                        process, self._process = self._process, None
                        _checked(_K.CloseHandle(process), "Close owned process handle")
            elif self._popen is not None:
                self.kill()
                self.wait(timeout=5)
        finally:
            if os.name == "nt":
                for handle in self._member_handles.values(): _K.CloseHandle(handle)
                self._member_handles.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        # Avoid blocking interpreter teardown; closing the last job handle kills its tree.
        if os.name == "nt":
            for handle in getattr(self, "_member_handles", {}).values(): _K.CloseHandle(handle)
            for attribute in ("_job", "_process"):
                handle = getattr(self, attribute, None)
                if handle:
                    _K.CloseHandle(handle)
                    setattr(self, attribute, None)
