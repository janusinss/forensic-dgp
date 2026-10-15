"""Read current Windows resident WorkingSet bytes; no optional dependency."""
import ctypes
from ctypes import wintypes
import os


class Counters(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
        ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
        ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]


def resident_bytes():
    assert os.name=='nt','This recorder memory reader is Windows-only'
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
    process=ctypes.WinDLL('psapi',use_last_error=True)
    process.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    process.GetProcessMemoryInfo.restype=wintypes.BOOL
    values=Counters();values.cb=ctypes.sizeof(Counters)
    if not process.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(values),values.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    assert values.WorkingSetSize>0
    return int(values.WorkingSetSize)
