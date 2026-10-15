# Latest verified offload — 13 October 2026

The manual Drive connection has completed its first two backup batches.
Independent streamed checks and live cloud IDs/sizes/hashes verify 4,457
uploaded files. Exactly 4,432 local payload copies / 52.03 GB were removed
after source hashes, active-file protections and full namespace checks passed.
All retained research files are unchanged. Metadata and failure records stay
local; the original VM was not contacted. C: had 79.04 GB free at removal end.

[Completed audit and recovery guide](<C:/xampp/htdocs/YEAR 4/Testing/GOOGLE_DRIVE_OFFLOAD_COMPLETE_20261013.md>).
The working profile remains `%APPDATA%\rclone\forensic-dgp-drive-manual.conf`;
no private credential or token is in the research archive. Earlier statuses
below are retained history, not the current upload/removal state.

# Latest connection status — 12 October 2026

The user's from-scratch manual connection is now independently verified:
`thesis_archive`, My Drive, own Desktop client, limited `drive.file` scope.
The working profile is `%APPDATA%\rclone\forensic-dgp-drive-manual.conf`.
Quota: 5,497,558,138,880 bytes total; 5,477,794,986,957 bytes free at the check.
The earlier prepared `forensic-dgp-drive.conf` profile remains separate.

Two selected cold-backup batches are hash-verified and prepared for manual
upload/verification, with 52,028,568,639 logical bytes as potential later
local-removal candidates. App/current-model/diagnostic dependencies stay local.
No research payload has been uploaded or removed by this preparation.
[Five exact manual offload steps](<C:/xampp/htdocs/YEAR 4/Testing/GOOGLE_DRIVE_OFFLOAD_20261012_V1.md>).
Original guide bytes are retained under
`outputs/google_drive_offload_20261012_v1/documents_before/`.
The setup-attempt history below is superseded by this verified manual profile.

# Google Drive connection setup — 12 October 2026

Status: portable rclone **1.75.2** is downloaded and matches the official archive
SHA256. Browser sign-in is confirmed: **18.41 GB used of 5 TB**. The approved
`Forensic DGP Drive Archive` OAuth application identity is created in
`forensic-dgp-thesis`. The approved Desktop client `Forensic DGP archive uploader`
is also created. The client values were securely saved from Google's visible
created-client fields and independently validated; the private rclone profile
exists. Google authorization and the connection check remain **pending**.
No research file has been moved, deleted or uploaded.
Active-file separation and verified offload remain pending.

The attempted separate archive project was rejected because the account has
reached its project-creation quota. Keep the existing `forensic-dgp-thesis`
project for archive authorization even when moving training to another VM or
Cloud account. The Drive API is already enabled there. A separate project is
optional; no existing project needs to be deleted for this setup.

The direct uploader avoids creating a second full local project copy in Drive
for desktop's cache. The app stays at its existing Windows path. Connection
setup does not start a VM, model, training experiment or historical pilot.

## 1. Create your Google desktop credentials

The VM's Google Cloud account may change. Keep the existing archive application
and client in `forensic-dgp-thesis` under the account you will retain. The storage
account is selected at browser sign-in and can differ from the OAuth-project
account. Retain this project/client when moving training to another account.

1. Select the retained permanent `forensic-dgp-thesis` project in the
   [Google Cloud console](https://console.cloud.google.com/home/dashboard?project=forensic-dgp-thesis).
2. Enable **Google Drive API** on the
   [Drive API page](https://console.cloud.google.com/apis/library/drive.googleapis.com).
   Keep the permanent project selected in the top project picker.
3. Configure [Google Auth platform → Overview](https://console.cloud.google.com/auth/overview)
   if it is not already configured. Use app name `Forensic DGP Drive Archive`,
   your email for support/contact and **External** audience. Review Google's
   terms yourself. If an existing Internal audience excludes the storage
   account, preserve it and resolve the project choice before changing it.

   In this browser, direct Branding initially failed to load before the app was
   configured. **Overview → Get started** worked and app creation is now
   confirmed. Do not repeat that completed step or accept terms again.
4. The scope `https://www.googleapis.com/auth/drive.file` is already saved under
   [Data Access](https://console.cloud.google.com/auth/scopes?project=forensic-dgp-thesis).
5. The Desktop client is already created under
   [Clients → Create client](https://console.cloud.google.com/auth/clients):
   application type **Desktop app**, name `Forensic DGP archive uploader`.
   Do not create a duplicate. Google currently keeps its one-time secret download
   dialog open. Both automated download waits failed to confirm a file, and no
   matching recent JSON was found in Downloads. The already created values were
   instead saved privately at
   `%TEMP%\forensic-dgp-drive-20261012-client.json`, outside this project, without
   displaying them. Its format/project were independently validated before the
   one-time dialog was closed. No duplicate client or credential reset is needed.

If you already created the client in `forensic-dgp-thesis`, it can still work
while that project/client remains available. Keep its credentials and project;
changing the VM account alone does not require creating another archive client.
No Cloud project or client is created, deleted or changed by the local helper.

These are the current [Google desktop-client setup steps](https://developers.google.com/workspace/drive/api/quickstart/python).
The limited [drive.file scope](https://developers.google.com/workspace/drive/api/guides/api-specific-auth)
supports newly created archive files. It does not grant the uploader access to
every existing file in your Drive. Your own client is required because
[rclone's shared client is being retired during 2026](https://rclone.org/drive/#making-your-own-client-id).

## 2. Authorize the uploader

These commands use **Windows PowerShell**, not the VM or Google Cloud SDK Shell.
The normal destination is **My Drive**. If your capacity belongs to a Shared
drive, identify that drive before authentication so its ID can be supplied as
`-SharedDriveId`. No destination should be inferred from the quota alone.

1. Your signed-in **5 TB Drive account** is already the sole test user under
   [Audience → Test users](https://console.cloud.google.com/auth/audience)
   in the same permanent OAuth project.
2. The private client/profile is already prepared. No JSON download is needed
   for the existing Windows connection. Never paste credentials in chat.
3. Run the prepared connection command in **Windows PowerShell**:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File "C:\xampp\htdocs\YEAR 4\Testing\scripts\connect_google_drive_20261012_v1.ps1" -Mode Authenticate
   ```

4. Authorize your **5 TB account** in the Google browser window. Choose **Y**
   if the console asks whether to use a browser.
5. Send only the final quota/status report in chat.

The agent-started login attempt was stopped at the user's request to do these
steps manually. Its private profile is retained; no token/connection success
was confirmed. Use the fresh browser window opened by your command.

Confirm the final report contains `"Connected": true`. Share only that final
quota/status report in chat, never the credentials JSON or token.

The command creates a dedicated connection file at
`%APPDATA%\rclone\forensic-dgp-drive.conf`, outside `Testing`. It preserves any
existing file at that path and checks its scope before reuse. The uploader will
create its own archive folder later; a browser-created folder is not assumed
to be visible through `drive.file`.

For a new VM under another Cloud account, use the same archive OAuth application
and authorize the same storage account. The limited scope is tied to that
application's files; making an unrelated client for every VM can prevent access
to previously created archive files. Use the
[rclone headless-machine setup](https://rclone.org/remote_setup/) to authorize
the VM, or securely transfer the dedicated connection configuration. Keep those
credentials outside the research archive. Verify the new VM's project, zone,
instance, SSH user, GPU, Linux paths and restored asset hashes before generating
new pasteable commands. No migration or training is started by this guide.

Testing-mode OAuth grants can expire after seven days. Reauthorization uses the
same command and preserved profile. Publishing an OAuth app is a separate
choice; this setup does not publish anything or modify existing clients.

## 3. Independently check the connection

After authorization, this performs a read-only account/quota check:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\xampp\htdocs\YEAR 4\Testing\scripts\connect_google_drive_20261012_v1.ps1" -Mode Check
```

A missing quota field means unavailable information, not zero storage. Successful
authentication is not an upload, backup verification or permission to remove
unverified files. Actual offload will use a fresh active-dependency inventory,
bounded archives, cloud hash/member verification and an exact local-removal
ledger. App files, current work, checkpoints, splits, caches, provenance and
failure records remain preserved through that process.

The original installer precheck rejected the ZIP's normal empty root-directory
entry. `scratch/drive_setup_20261012_v1/setup_stop_01.json` retains that stop.
The corrected precheck accepts only that exact directory plus safe file members;
the original downloaded bytes and official checksum remain unchanged.

Installer evidence: `scratch/drive_setup_20261012_v1/installer_receipt.json`.
Official archive SHA256:
`8a9db2bad1dc79a65b3348fd857069598e2aab7cf4d118b77b1df23055aff818`.
