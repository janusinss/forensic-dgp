# Desktop lifecycle contract

Implementation status: prototype compiled and non-UI regressions passed. The root agent must verify the actual Edge application window before accepting this phase.

## Launch boundary

`packaging/launcher/Program.cs` builds as the x64 .NET Framework WinExe `ForensicDGP.exe`. It requires the Windows .NET Framework and an installed Microsoft Edge. It does not download a browser or silently fall back to ordinary browser mode.

The supervisor holds `Local\ForensicDGP-Desktop` for its entire lifetime. Inno can use this mutex to avoid replacing or uninstalling shell files while the desktop is open. The backend's existing `Local\DGP-app-*` mutex remains authoritative for source/environment installation and application launch.

The supervisor validates the expected managed root's ordinary paths, ownership marker, active commit, and the distribution helper's receipt hash before executing that helper. The helper verifies the complete source inventory and starts the configured environment using the existing launch path.

## Private host protocol

The supervisor creates a unique `%TEMP%\ForensicDGP-<32 lowercase hex>\` directory and Windows manual-reset event `Local\ForensicDGP-stop-<same token>`. It starts the backend with:

- `DGP_DESKTOP_READY_FILE`: the unique directory's `ready.json`.
- `DGP_DESKTOP_STOP_EVENT`: that event name.
- launch arguments: `--port 0 --no-browser --no-pause --desktop`.

`implementation.desktop.DesktopSession` refuses partial values, mismatching tokens, other readiness paths, links, and existing readiness records. A host-owned event must already exist. Python publishes `{schema:1,url,pid,token}` using the actual bound loopback port. The supervisor requires the reported process to belong to its owned job and `/health` to succeed before opening Edge. There is no HTTP shutdown endpoint.

A watcher on the named event sets `uvicorn.Server.should_exit`. This runs normal application shutdown handlers, including capture cleanup. Handles close on failed startup as well as normal exit.

## Ownership and closure

All server/worker processes start inside an unnamed Windows job configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. `CreateProcessW` starts the initial child suspended; it is assigned to the job before resumption. If assignment fails, that child is terminated before it may continue. Edge starts in the same owned job with its own profile under `data\desktop\edge-profile` and background mode/extensions disabled.

The supervisor enumerates visible `Chrome_WidgetWin_1` windows only for process IDs in its job. It first requires an owned window to appear; after the last owned app window closes, it signals graceful server shutdown. It allows 12 seconds before disposal of the owned job terminates remaining owned descendants. It never terminates a process by name. Unrelated Edge windows are outside the job.

Root must establish that Edge stays in the owned job and window enumeration remains reliable on the target machine, including Edge already running, window closure during capture/inference, and a repeated launch. A failed ownership proof blocks this shell rather than authorizing a different runtime.

## Diagnostics

The launcher uses hidden child processes and retains stdout/stderr in `data\desktop\logs\desktop-<timestamp>-<token>.log`. Startup failures show a dialog with the log location. Health retries are bounded to 90 seconds. Missing or unowned Edge windows fail within 35 seconds. Diagnostic logs and application records remain after closure. Only the current session's known temporary handshake file and empty directory are removed.

## Non-UI verification

1. `python -B -m unittest tests.test_desktop_launch -v`: 11 passing tests including a real Windows named-event signal and startup-error handle cleanup.
2. Compile `packaging/launcher/Program.cs` as WinExe with Framework64 `csc.exe`, references `System.Windows.Forms.dll`, `System.Web.Extensions.dll`, and `System.Drawing.dll`.
3. Compile `Program.cs` plus `ProcessTests.cs` with `/target:exe /main:DesktopProcessTests`. The fixture proves assignment, ownership enumeration, job-disposal cleanup, and survival of an unrelated control process.

These checks establish protocol and process primitives. They do not establish successful browser rendering, camera permissions, or the complete installed desktop flow.
