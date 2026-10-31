// Desktop supervisor. Built with the Windows .NET Framework compiler as a WinExe.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static class Program
{
    private const string DesktopMutex = @"Local\ForensicDGP-Desktop";
    private static readonly JavaScriptSerializer Json = new JavaScriptSerializer();
    private static string logPath;
    private static FileStream log;

    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        bool created;
        using (Mutex singleton = new Mutex(true, DesktopMutex, out created))
        {
            if (!created)
            {
                MessageBox.Show("Forensic DGP is already running. Close its window before launching another instance.", "Forensic DGP", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return 2;
            }
            string sessionDir = null;
            try
            {
                if (args.Length != 0) throw new InvalidOperationException("This launcher accepts no command-line options.");
                string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                InstallLocation location = InstallLocation.Resolve(Path.GetDirectoryName(Application.ExecutablePath), local);
                string root = location.ManagedRoot;
                Environment.SetEnvironmentVariable("DGP_INSTALL_ROOT", location.InstallRoot);
                VerifyManagedRoot(root, location.InstallRoot);
                string helper = VerifyHelper(root);
                string edge = FindEdge();
                string logs = Path.Combine(root, @"data\desktop\logs");
                EnsureOrdinaryDirectory(logs);
                logPath = Path.Combine(logs, "desktop-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss") + "-" + Guid.NewGuid().ToString("N") + ".log");
                log = Native.CreateAppendLog(logPath);
                Native.Check(Native.SetHandleInformation(log.SafeFileHandle.DangerousGetHandle(), 1, 1), "Cannot prepare the diagnostic log");
                string token = Guid.NewGuid().ToString("N");
                sessionDir = Path.Combine(Path.GetTempPath(), "ForensicDGP-" + token);
                if (Directory.Exists(sessionDir) || File.Exists(sessionDir)) throw new IOException("Desktop handshake directory already exists.");
                Directory.CreateDirectory(sessionDir);
                CheckOrdinary(sessionDir, true);
                string ready = Path.Combine(sessionDir, "ready.json");
                string eventName = @"Local\ForensicDGP-stop-" + token;
                string profile = Path.Combine(root, @"data\desktop\edge-profile");
                EnsureOrdinaryDirectory(profile);
                Environment.SetEnvironmentVariable("DGP_DESKTOP_READY_FILE", ready);
                Environment.SetEnvironmentVariable("DGP_DESKTOP_STOP_EVENT", eventName);
                Environment.SetEnvironmentVariable("DGP_DESKTOP_HELPER", helper);
                Environment.SetEnvironmentVariable("DGP_DESKTOP_MANAGED_ROOT", root);
                Environment.SetEnvironmentVariable("IMPLEMENTATION_NO_PAUSE", "1");
                using (EventWaitHandle stop = new EventWaitHandle(false, EventResetMode.ManualReset, eventName))
                using (Native.Job job = new Native.Job())
                {
                    string ps = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), @"WindowsPowerShell\v1.0\powershell.exe");
                    // Static code and environment values avoid interpolating user paths into shell syntax.
                    string command = "$ErrorActionPreference='Stop'; & $env:DGP_DESKTOP_HELPER -Action Launch -ManagedRoot $env:DGP_DESKTOP_MANAGED_ROOT -LaunchArgs @('--port','0','--no-browser','--no-pause','--desktop'); exit $LASTEXITCODE";
                    string psArgs = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + Convert.ToBase64String(Encoding.Unicode.GetBytes(command));
                    using (Native.Child server = job.Start(ps, psArgs, log.SafeFileHandle.DangerousGetHandle()))
                    {
                        string url = AwaitReady(server, ready, token, job);
                        WriteLog("Healthy application URL: " + url);
                        string edgeArgs = "--user-data-dir=" + Quote(profile) + " --no-first-run --no-default-browser-check --disable-background-mode --disable-extensions --app=" + Quote(url);
                        using (Native.Child browser = job.Start(edge, edgeArgs, log.SafeFileHandle.DangerousGetHandle(), true))
                        {
                            bool hadWindow = false;
                            bool starterExitLogged = false;
                            DateTime nextDiagnostic = DateTime.UtcNow;
                            DateTime windowDeadline = DateTime.UtcNow.AddSeconds(35);
                            try
                            {
                                for (;;)
                                {
                                    if (server.Exited) throw new InvalidOperationException("The application server stopped unexpectedly (exit " + server.ExitCode + ").");
                                    HashSet<uint> owned = job.ProcessIds();
                                    bool visible = Native.HasAppWindow(owned);
                                    if (browser.Exited && !starterExitLogged)
                                    {
                                        WriteLog("Edge starter exited: pid=" + browser.Id + ", code=" + browser.ExitCode + "; waiting for an owned app window until the bounded deadline.");
                                        starterExitLogged = true;
                                    }
                                    if (DateTime.UtcNow >= nextDiagnostic)
                                    {
                                        WriteLog("Owned app-window probe: visible=" + visible + "; pids=" + String.Join(",", owned));
                                        nextDiagnostic = DateTime.UtcNow.AddSeconds(2);
                                    }
                                    if (visible) hadWindow = true;
                                    else if (hadWindow) break;
                                    else if (DateTime.UtcNow > windowDeadline)
                                        throw new InvalidOperationException("The dedicated Edge window could not be owned within 35 seconds. Starter pid=" + browser.Id + ", exited=" + browser.Exited + ", exit code=" + (browser.Exited ? browser.ExitCode.ToString() : "running") + "; owned pids=" + String.Join(",", owned) + ". Close a stale Forensic DGP window and retry. No ordinary browser was opened.");
                                    Thread.Sleep(200);
                                }
                            }
                            finally
                            {
                                // This event exists only for this supervisor/server session.
                                stop.Set();
                                DateTime end = DateTime.UtcNow.AddSeconds(12);
                                while (!server.Exited && DateTime.UtcNow < end) Thread.Sleep(100);
                                if (!server.Exited) WriteLog("Graceful shutdown timed out; disposing the owned process job.");
                                else WriteLog("Owned server exited: " + server.ExitCode);
                            }
                        }
                    }
                }
                return 0;
            }
            catch (Exception error)
            {
                WriteLog(error.ToString());
                string suffix = logPath == null ? "" : "\n\nDiagnostics: " + logPath;
                MessageBox.Show(error.Message + suffix, "Forensic DGP could not start", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }
            finally
            {
                if (sessionDir != null)
                {
                    // Remove only this session's known handshake file and empty directory.
                    try { string f = Path.Combine(sessionDir, "ready.json"); if (File.Exists(f)) File.Delete(f); Directory.Delete(sessionDir, false); }
                    catch (IOException error) { WriteLog("Temporary handshake record retained: " + error.Message); }
                    catch (UnauthorizedAccessException error) { WriteLog("Temporary handshake record retained: " + error.Message); }
                }
                if (log != null) { log.Dispose(); log = null; }
                singleton.ReleaseMutex();
            }
        }
    }

    private static void VerifyManagedRoot(string root, string installRoot)
    {
        CheckOrdinary(root, true);
        string markerPath = Path.Combine(root, "owner.json");
        CheckOrdinary(markerPath, false);
        Dictionary<string, object> marker = ReadJson(markerPath);
        int schema = installRoot == null ? 1 : 2;
        if (marker.Count != (schema == 1 ? 2 : 4) || !marker.ContainsKey("repository") || !marker.ContainsKey("schema_version")
            || !(marker["schema_version"] is int) || (int)marker["schema_version"] != schema
            || Convert.ToString(marker["repository"]) != "janusinss/forensic-dgp")
            throw new InvalidOperationException("The managed installation ownership marker differs. Run the setup executable.");
        if (schema == 2 && (!marker.ContainsKey("layout") || !marker.ContainsKey("install_root")
            || Convert.ToString(marker["layout"]) != "desktop-runtime-v1" || Convert.ToString(marker["install_root"]) != installRoot))
            throw new InvalidOperationException("The managed installation marker is not bound to this application folder.");
    }

    private static string VerifyHelper(string root)
    {
        string pointer = Path.Combine(root, "current.json");
        CheckOrdinary(pointer, false);
        Dictionary<string, object> active = ReadJson(pointer);
        string commit = Convert.ToString(active["commit"]);
        if (Convert.ToInt32(active["schema"]) != 1 || !System.Text.RegularExpressions.Regex.IsMatch(commit, "^[0-9a-f]{40}$"))
            throw new InvalidOperationException("The managed active version is invalid.");
        string helper = Path.Combine(root, @"versions\" + commit + @"\implementation\distribution.ps1");
        string receiptPath = Path.Combine(root, @"receipts\" + commit + ".json");
        CheckOrdinary(helper, false); CheckOrdinary(receiptPath, false);
        Dictionary<string, object> receipt = ReadJson(receiptPath);
        Dictionary<string, object> files = receipt["files"] as Dictionary<string, object>;
        if (files == null || Convert.ToString(receipt["commit"]) != commit || Convert.ToInt32(receipt["schema"]) != 1)
            throw new InvalidOperationException("The managed version receipt differs.");
        string expected = Convert.ToString(files["implementation/distribution.ps1"]);
        using (SHA256 sha = SHA256.Create())
        using (FileStream input = File.OpenRead(helper))
        {
            string actual = BitConverter.ToString(sha.ComputeHash(input)).Replace("-", "").ToLowerInvariant();
            if (actual != expected) throw new InvalidOperationException("The managed installer helper differs from its receipt. Existing files were preserved.");
        }
        return helper;
    }

    private static Dictionary<string, object> ReadJson(string path)
    {
        if (new FileInfo(path).Length > 2 * 1024 * 1024) throw new IOException("Installation metadata is too large.");
        Dictionary<string, object> value = Json.DeserializeObject(File.ReadAllText(path)) as Dictionary<string, object>;
        if (value == null) throw new IOException("Installation metadata must be an object.");
        return value;
    }

    private static void CheckOrdinary(string path, bool directory)
    {
        string cursor = Path.GetFullPath(path);
        bool first = true;
        for (;;)
        {
            FileAttributes attributes = File.GetAttributes(cursor);
            if ((attributes & FileAttributes.ReparsePoint) != 0) throw new IOException("Application paths must not contain links or reparse points.");
            if (first && (((attributes & FileAttributes.Directory) != 0) != directory)) throw new IOException("Unexpected application path type.");
            first = false;
            string parent = Path.GetDirectoryName(cursor);
            if (String.IsNullOrEmpty(parent) || parent == cursor) break;
            cursor = parent;
        }
    }

    private static void EnsureOrdinaryDirectory(string path)
    {
        string cursor = path;
        while (!Directory.Exists(cursor))
        {
            if (File.Exists(cursor)) throw new IOException("An application directory is occupied by a file.");
            cursor = Path.GetDirectoryName(cursor);
            if (String.IsNullOrEmpty(cursor)) throw new IOException("No ordinary parent directory exists.");
        }
        CheckOrdinary(cursor, true);
        Directory.CreateDirectory(path);
        CheckOrdinary(path, true);
    }

    private static string FindEdge()
    {
        string[] roots = { Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData) };
        foreach (string root in roots)
        {
            string path = Path.Combine(root, @"Microsoft\Edge\Application\msedge.exe");
            if (File.Exists(path)) { CheckOrdinary(path, false); return path; }
        }
        throw new FileNotFoundException("Microsoft Edge is required for this desktop preview. Install Edge or use the developer batch launcher.");
    }

    private static string AwaitReady(Native.Child server, string ready, string token, Native.Job job)
    {
        DateTime end = DateTime.UtcNow.AddSeconds(90);
        while (DateTime.UtcNow < end)
        {
            if (server.Exited) throw new InvalidOperationException("The managed application could not start (exit " + server.ExitCode + "). Close any running app or installer before retrying.");
            if (File.Exists(ready))
            {
                Dictionary<string, object> state;
                try { state = ReadJson(ready); }
                catch (ArgumentException) { Thread.Sleep(100); continue; }
                catch (IOException) { Thread.Sleep(100); continue; }
                Uri uri;
                if (Convert.ToString(state["token"]) != token || Convert.ToInt32(state["schema"]) != 1 || !Uri.TryCreate(Convert.ToString(state["url"]), UriKind.Absolute, out uri)
                    || uri.Scheme != "http" || uri.Host != "127.0.0.1" || uri.Port <= 0 || !String.IsNullOrEmpty(uri.UserInfo) || uri.AbsolutePath != "/" || !String.IsNullOrEmpty(uri.Query) || !String.IsNullOrEmpty(uri.Fragment))
                    throw new IOException("The application readiness handshake differs.");
                uint pid = Convert.ToUInt32(state["pid"]);
                if (!job.ProcessIds().Contains(pid)) throw new IOException("The readiness process does not belong to this desktop instance.");
                try
                {
                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create(new Uri(uri, "/health"));
                    request.Proxy = null; request.Timeout = 1000; request.ReadWriteTimeout = 1000; request.AllowAutoRedirect = false;
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                        if (response.StatusCode == HttpStatusCode.OK) return uri.GetLeftPart(UriPartial.Authority);
                }
                catch (WebException) { }
            }
            Thread.Sleep(150);
        }
        throw new TimeoutException("The application did not become healthy within 90 seconds.");
    }

    private static void WriteLog(string text)
    {
        if (log == null) return;
        byte[] bytes = Encoding.UTF8.GetBytes(DateTime.UtcNow.ToString("O") + " " + text + "\r\n");
        try { log.Write(bytes, 0, bytes.Length); log.Flush(); } catch (IOException error) { Debug.WriteLine("Desktop diagnostic write failed: " + error.Message); }
    }

    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in value)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') result.Append('\\', slashes * 2 + 1);
            else result.Append('\\', slashes);
            slashes = 0; result.Append(c);
        }
        result.Append('\\', slashes * 2); result.Append('"');
        return result.ToString();
    }
}

internal sealed class InstallLocation
{
    public readonly string InstallRoot;
    public readonly string ManagedRoot;
    private InstallLocation(string installRoot, string managedRoot) { InstallRoot = installRoot; ManagedRoot = managedRoot; }
    private static readonly JavaScriptSerializer Json = new JavaScriptSerializer();

    public static InstallLocation Resolve(string executableDirectory, string localAppData)
    {
        string shell = Canonical(executableDirectory);
        string config = Path.Combine(shell, "install-location.json");
        if (!File.Exists(config))
        {
            string legacyShell = Canonical(Path.Combine(localAppData, @"Programs\Forensic DGP"));
            if (!String.Equals(shell, legacyShell, StringComparison.OrdinalIgnoreCase))
                throw new IOException("The installation location record is missing. Run setup for this application folder.");
            Ordinary(shell, true);
            return new InstallLocation(null, Canonical(Path.Combine(localAppData, @"DGP\app")));
        }
        ValidateShell(shell);
        Dictionary<string, object> marker = ReadObject(Path.Combine(shell, "shell-owner.json"));
        if (marker.Count != 2 || !Integer(marker, "schema", 1) || !Text(marker, "application", "Forensic DGP"))
            throw new IOException("The desktop shell ownership marker differs.");
        Dictionary<string, object> value = ReadObject(config);
        string root = Path.Combine(shell, "runtime");
        if (value.Count != 3 || !Integer(value, "schema", 1) || !Text(value, "install_root", shell) || !Text(value, "managed_root", root))
            throw new IOException("The installation record is not bound to this executable's folder.");
        return new InstallLocation(shell, root);
    }

    private static string Canonical(string value)
    {
        if (String.IsNullOrEmpty(value) || !System.Text.RegularExpressions.Regex.IsMatch(value, @"^[A-Za-z]:\\")
            || value.IndexOf('/') >= 0 || value.Substring(2).IndexOf(':') >= 0)
            throw new IOException("Choose an absolute local installation folder.");
        return Path.GetFullPath(value).TrimEnd('\\');
    }
    private static bool Integer(Dictionary<string, object> value, string key, int expected)
    { return value.ContainsKey(key) && value[key] is int && (int)value[key] == expected; }
    private static bool Text(Dictionary<string, object> value, string key, string expected)
    { return value.ContainsKey(key) && value[key] is string && (string)value[key] == expected; }
    private static Dictionary<string, object> ReadObject(string path)
    {
        Ordinary(path, false);
        if (new FileInfo(path).Length > 8192) throw new IOException("Installation metadata is too large.");
        Dictionary<string, object> value = Json.DeserializeObject(File.ReadAllText(path)) as Dictionary<string, object>;
        if (value == null) throw new IOException("Installation metadata must be an object.");
        return value;
    }
    private static void Ordinary(string path, bool directory)
    {
        bool first = true;
        for (string cursor = Path.GetFullPath(path); ; )
        {
            FileAttributes attributes = File.GetAttributes(cursor);
            if ((attributes & FileAttributes.ReparsePoint) != 0 || (first && (((attributes & FileAttributes.Directory) != 0) != directory)))
                throw new IOException("Installation paths must be ordinary files and folders without links.");
            first = false;
            string parent = Path.GetDirectoryName(cursor);
            if (String.IsNullOrEmpty(parent) || parent == cursor) break;
            cursor = parent;
        }
    }
    private static void ValidateShell(string shell)
    {
        if (shell.Length > 120) throw new IOException("The application folder exceeds the supported 120-character limit.");
        string driveRoot = Path.GetPathRoot(shell);
        DriveInfo drive = new DriveInfo(driveRoot);
        if (shell == driveRoot.TrimEnd('\\') || drive.DriveType != DriveType.Fixed || !drive.IsReady)
            throw new IOException("The application needs a dedicated local fixed-drive folder.");
        foreach (string part in shell.Substring(3).Split('\\'))
            if (part.Length == 0 || part == "." || part == ".." || part.EndsWith(".") || part.EndsWith(" ")
                || part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0
                || System.Text.RegularExpressions.Regex.IsMatch(part, @"^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)"))
                throw new IOException("Installation path contains an ambiguous directory name.");
        string[] protectedRoots = { Environment.GetEnvironmentVariable("SystemRoot"), Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) };
        foreach (string entry in protectedRoots)
        {
            if (String.IsNullOrEmpty(entry)) continue;
            string root = Canonical(entry);
            if (String.Equals(shell, root, StringComparison.OrdinalIgnoreCase) || shell.StartsWith(root + "\\", StringComparison.OrdinalIgnoreCase))
                throw new IOException("Windows and shared program folders cannot be the installation destination.");
        }
        string user = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] bases = { user, Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
            Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), Environment.GetFolderPath(Environment.SpecialFolder.MyMusic),
            Environment.GetFolderPath(Environment.SpecialFolder.MyVideos), String.IsNullOrEmpty(user) ? null : Path.Combine(user, "Downloads") };
        foreach (string entry in bases)
            if (!String.IsNullOrEmpty(entry) && String.Equals(shell, Canonical(entry), StringComparison.OrdinalIgnoreCase))
                throw new IOException("Choose a dedicated child folder rather than a personal directory root.");
        Ordinary(shell, true);
    }
}
internal static class Native
{
    [StructLayout(LayoutKind.Sequential)] private struct BasicLimit { public long PerProcessUserTimeLimit, PerJobUserTimeLimit; public uint LimitFlags; public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize; public uint ActiveProcessLimit; public UIntPtr Affinity; public uint PriorityClass, SchedulingClass; }
    [StructLayout(LayoutKind.Sequential)] private struct IoCounters { public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount; }
    [StructLayout(LayoutKind.Sequential)] private struct ExtendedLimit { public BasicLimit Basic; public IoCounters IO; public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed; }
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] private struct StartupInfo { public int cb; public string reserved, desktop, title; public uint x, y, xSize, ySize, xChars, yChars, fill, flags; public short showWindow, reserved2; public IntPtr reservedPtr, input, output, error; }
    [StructLayout(LayoutKind.Sequential)] private struct ProcessInfo { public IntPtr process, thread; public uint pid, tid; }
    [StructLayout(LayoutKind.Sequential)] private struct SecurityAttributes { public int length; public IntPtr descriptor; [MarshalAs(UnmanagedType.Bool)] public bool inherit; }
    [DllImport("kernel32.dll", SetLastError = true)] private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool QueryInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size, IntPtr length);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool CreateProcess(string application, StringBuilder command, IntPtr processAttributes, IntPtr threadAttributes, bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfo startup, out ProcessInfo process);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr CreateFile(string name, uint access, uint share, ref SecurityAttributes attributes, uint disposition, uint flags, IntPtr template);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool TerminateProcess(IntPtr process, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);
    private delegate bool EnumCallback(IntPtr window, IntPtr parameter);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumCallback callback, IntPtr parameter);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr window);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr window, out uint process);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] private static extern int GetClassName(IntPtr window, StringBuilder name, int count);

    public static FileStream CreateAppendLog(string path)
    {
        SecurityAttributes security = new SecurityAttributes(); security.length = Marshal.SizeOf(security); security.inherit = true;
        // FILE_APPEND_DATA ensures inherited native writers and managed writes
        // cannot overwrite one another through a shared file-position cache.
        IntPtr handle = CreateFile(path, 4, 3, ref security, 1, 0x80, IntPtr.Zero);
        Check(handle != new IntPtr(-1), "Cannot create append-only diagnostic log");
        return new FileStream(new Microsoft.Win32.SafeHandles.SafeFileHandle(handle, true), FileAccess.Write);
    }

    public static void Check(bool ok, string message)
    {
        if (!ok) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error(), message);
    }

    public sealed class Child : IDisposable
    {
        internal IntPtr handle;
        public readonly uint Id;
        internal Child(IntPtr value, uint id) { handle = value; Id = id; }
        public bool Exited { get { return WaitForSingleObject(handle, 0) == 0; } }
        public uint ExitCode { get { uint code; Check(GetExitCodeProcess(handle, out code), "Cannot read child process status"); return code; } }
        public void Dispose() { if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; } }
    }

    public sealed class Job : IDisposable
    {
        private IntPtr handle;
        public Job()
        {
            handle = CreateJobObject(IntPtr.Zero, null);
            Check(handle != IntPtr.Zero, "Cannot create owned application job");
            ExtendedLimit limits = new ExtendedLimit(); limits.Basic.LimitFlags = 0x2000;
            int size = Marshal.SizeOf(limits); IntPtr data = Marshal.AllocHGlobal(size);
            try { Marshal.StructureToPtr(limits, data, false); Check(SetInformationJobObject(handle, 9, data, (uint)size), "Cannot configure owned application shutdown"); }
            finally { Marshal.FreeHGlobal(data); }
        }
        public Child Start(string application, string arguments, IntPtr output, bool showWindow = false)
        {
            SecurityAttributes security = new SecurityAttributes(); security.length = Marshal.SizeOf(security); security.inherit = true;
            IntPtr input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero);
            Check(input != new IntPtr(-1), "Cannot create child input handle");
            StartupInfo info = new StartupInfo(); info.cb = Marshal.SizeOf(info); info.flags = showWindow ? 0x100u : 0x101u; info.showWindow = showWindow ? (short)5 : (short)0; info.input = input; info.output = output; info.error = output;
            ProcessInfo process;
            try { Check(CreateProcess(application, new StringBuilder("\"" + application + "\" " + arguments), IntPtr.Zero, IntPtr.Zero, true, 0x08000004, IntPtr.Zero, Path.GetDirectoryName(application), ref info, out process), "Cannot start owned application process"); }
            finally { CloseHandle(input); }
            try
            {
                if (!AssignProcessToJobObject(handle, process.process))
                {
                    int error = Marshal.GetLastWin32Error(); TerminateProcess(process.process, 1);
                    throw new System.ComponentModel.Win32Exception(error, "Cannot own the application process; no unmanaged process was allowed to continue");
                }
                if (ResumeThread(process.thread) == UInt32.MaxValue) { TerminateProcess(process.process, 1); Check(false, "Cannot resume owned application process"); }
                return new Child(process.process, process.pid);
            }
            catch { CloseHandle(process.process); throw; }
            finally { CloseHandle(process.thread); }
        }
        public HashSet<uint> ProcessIds()
        {
            const int size = 65536; IntPtr data = Marshal.AllocHGlobal(size);
            try
            {
                Check(QueryInformationJobObject(handle, 3, data, size, IntPtr.Zero), "Cannot inspect owned application processes");
                uint count = (uint)Marshal.ReadInt32(data, 4);
                if (count > (size - 8) / IntPtr.Size) throw new InvalidOperationException("Owned process list exceeds its limit.");
                HashSet<uint> ids = new HashSet<uint>();
                for (int i = 0; i < count; i++) ids.Add((uint)Marshal.ReadIntPtr(data, 8 + i * IntPtr.Size).ToInt64());
                return ids;
            }
            finally { Marshal.FreeHGlobal(data); }
        }
        public void Dispose() { if (handle != IntPtr.Zero) { CloseHandle(handle); handle = IntPtr.Zero; } }
    }

    public static bool HasAppWindow(HashSet<uint> owned)
    {
        bool found = false;
        EnumWindows(delegate(IntPtr window, IntPtr parameter)
        {
            uint pid; GetWindowThreadProcessId(window, out pid);
            if (owned.Contains(pid) && IsWindowVisible(window))
            {
                StringBuilder name = new StringBuilder(256); GetClassName(window, name, name.Capacity);
                if (name.ToString() == "Chrome_WidgetWin_1") { found = true; return false; }
            }
            return true;
        }, IntPtr.Zero);
        return found;
    }
}
