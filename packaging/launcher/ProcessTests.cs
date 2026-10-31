// Non-UI ownership regression fixture; compile with Program.cs and /main:DesktopProcessTests.
using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
internal static class DesktopProcessTests
{
    public static int Main()
    {
        string ps = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), @"WindowsPowerShell\v1.0\powershell.exe");
        string path = Path.Combine(Path.GetTempPath(), "ForensicDGP-process-test-" + Guid.NewGuid().ToString("N") + ".log");
        using (FileStream output = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.ReadWrite))
        {
            Native.Check(Native.SetHandleInformation(output.SafeFileHandle.DangerousGetHandle(),1,1), "log inheritance");
            ProcessStartInfo controlInfo = new ProcessStartInfo(ps, "-NoProfile -NonInteractive -Command Start-Sleep -Seconds 30");
            controlInfo.UseShellExecute=false; controlInfo.CreateNoWindow=true;
            using (Process unrelated = Process.Start(controlInfo))
            {
                Native.Child child = null;
                try
                {
                    using (Native.Job owned = new Native.Job())
                    {
                        child = owned.Start(ps, "-NoProfile -NonInteractive -Command Start-Sleep -Seconds 30", output.SafeFileHandle.DangerousGetHandle());
                        if (!owned.ProcessIds().Contains(child.Id)) throw new Exception("Child not a member of owned job");
                        if (owned.ProcessIds().Contains((uint)unrelated.Id)) throw new Exception("Unrelated process acquired");
                        if (Native.HasAppWindow(owned.ProcessIds())) throw new Exception("Non-UI process reported an app window");
                    }
                    for (int i=0;i<30 && !child.Exited;i++) Thread.Sleep(100);
                    if (!child.Exited) throw new Exception("Owned job disposal did not stop child");
                    if (unrelated.HasExited) throw new Exception("Unrelated process was stopped");
                    Console.WriteLine("PASS: suspended child assigned before run; job disposal stops only owned child; unrelated child survives");
                }
                finally
                {
                    if (child!=null) child.Dispose();
                    if (!unrelated.HasExited) { unrelated.Kill(); unrelated.WaitForExit(3000); }
                }
            }
        }
        File.Delete(path);
        return 0;
    }
}
