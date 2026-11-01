"""Behavioral checks for installer-requested graceful desktop closure."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CSC = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'

FIXTURE = r"""
using System;
using System.Threading;
using System.Runtime.InteropServices;
internal static class LauncherControlFixture {
 [DllImport("shell32.dll", SetLastError=true)] static extern IntPtr CommandLineToArgvW([MarshalAs(UnmanagedType.LPWStr)] string text, out int count);
 [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr value);
 static void Need(bool value, string message) { if(!value) throw new Exception(message); }
 static int Main() {
  string first=@"C:\Test Apps\Forensic DGP";
  string second=@"D:\Other Folder\Forensic DGP";
  string name=Program.CloseRequestName(first);
  Need(name==Program.CloseRequestName(first.ToLowerInvariant()+@"\"),"close event must normalize equivalent paths");
  Need(name!=Program.CloseRequestName(second),"close request must be scoped to one installation");
  using(EventWaitHandle requested=new EventWaitHandle(false,EventResetMode.ManualReset,name))
  using(EventWaitHandle unrelated=new EventWaitHandle(false,EventResetMode.ManualReset,Program.CloseRequestName(second))) {
   Need(Program.SignalCloseRequest(first)==0,"valid close request failed");
   Need(requested.WaitOne(1000),"owned supervisor was not signaled");
   Need(!unrelated.WaitOne(0),"another installation received close request");
  }
  Need(Program.SignalCloseRequest(first)==0,"already closed app should permit setup to continue");
  string helper=@"C:\Test Person's Apps\Forensic DGP\implementation\desktop_start.ps1";
  string command="powershell.exe "+Program.StartupArguments(helper);
  int count; IntPtr values=CommandLineToArgvW(command,out count);
  Need(values!=IntPtr.Zero,"command parse failed");
  try {
   Need(count==7,"unexpected startup argument count");
   string[] args=new string[count];
   for(int i=0;i<count;i++) args[i]=Marshal.PtrToStringUni(Marshal.ReadIntPtr(values,i*IntPtr.Size));
   Need(args[1]=="-NoProfile" && args[2]=="-NonInteractive","startup isolation flags differ");
   Need(args[3]=="-ExecutionPolicy" && args[4]=="RemoteSigned","startup must retain the process-scoped local script policy");
   Need(args[5]=="-File" && args[6]==helper,"helper path did not round trip safely");
  } finally { LocalFree(values); }
  Console.WriteLine("PASS: scoped graceful close and transparent helper arguments");
  return 0;
 }
}
"""

@unittest.skipUnless(os.name=='nt' and CSC.is_file(), 'Windows Framework compiler required')
class LauncherControlTests(unittest.TestCase):
    def test_scoped_close_and_safe_transparent_startup(self):
        with tempfile.TemporaryDirectory(prefix='dgp-control-') as folder:
            base=Path(folder)
            fixture=base/'Fixture.cs'
            fixture.write_text(FIXTURE,encoding='utf-8')
            executable=base/'Fixture.exe'
            command=[str(CSC),'/nologo','/target:exe','/main:LauncherControlFixture',
                     '/reference:System.Windows.Forms.dll','/reference:System.Drawing.dll',
                     '/reference:System.Web.Extensions.dll','/out:'+str(executable),
                     str(ROOT/'packaging/launcher/Program.cs'),str(fixture)]
            built=subprocess.run(command,capture_output=True,text=True,timeout=45)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            checked=subprocess.run([str(executable)],capture_output=True,text=True,timeout=15)
            self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
            self.assertIn('scoped graceful close',checked.stdout)

if __name__=='__main__': unittest.main()
