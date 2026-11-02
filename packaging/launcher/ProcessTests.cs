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
        using (FileStream output = Native.CreateAppendLog(path))
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
                        // An executable starter may exit after spawning the actual
                        // application. Its exit must not imply all owned children exited.
                        string spawn = "-NoProfile -NonInteractive -Command \"Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList '-NoProfile -NonInteractive -Command Start-Sleep -Seconds 20' -WindowStyle Hidden\"";
                        using (Native.Child starter = owned.Start(ps, spawn, output.SafeFileHandle.DangerousGetHandle()))
                        {
                            DateTime deadline = DateTime.UtcNow.AddSeconds(15);
                            while (!starter.Exited && DateTime.UtcNow < deadline) Thread.Sleep(100);
                            if (!starter.Exited || starter.ExitCode != 0) throw new Exception("Starter fixture did not exit successfully");
                            for (int i=0;i<20 && owned.ProcessIds().Contains(starter.Id);i++) Thread.Sleep(50);
                            if (owned.ProcessIds().Contains(starter.Id) || owned.ProcessIds().Count < 2) throw new Exception("Starter descendant was not retained after the starter left its owned job");
                        }
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
