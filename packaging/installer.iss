#ifndef StageDir
  #error StageDir is required; use packaging/build_release.py
#endif
#include AddBackslash(StageDir) + "release.inc"

[Setup]
AppId={{FEE649E9-195D-4576-8F30-DAE06A5099C8}
AppName=Forensic DGP
AppVersion={#ReleaseVersion}
AppVerName=Forensic DGP {#ReleaseVersion} preview
AppPublisher=Forensic DGP Research
AppCopyright=Copyright (c) 2026 Forensic DGP Research
AppPublisherURL=https://forensic-dgp.pages.dev/
AppSupportURL=https://github.com/janusinss/forensic-dgp-downloads/releases
AppUpdatesURL=https://forensic-dgp.pages.dev/
AppComments=Experimental local face restoration
VersionInfoVersion={#ReleaseVersion}.0
VersionInfoCompany=Forensic DGP Research
VersionInfoDescription=Forensic DGP Offline Setup
VersionInfoTextVersion={#ReleaseVersion}
VersionInfoCopyright=Copyright (c) 2026 Forensic DGP Research
VersionInfoProductName=Forensic DGP
VersionInfoProductVersion={#ReleaseVersion}
VersionInfoProductTextVersion={#ReleaseVersion}
VersionInfoOriginalFileName=ForensicDGP_Setup.exe
DefaultDirName={localappdata}\Programs\Forensic DGP
DefaultGroupName=Forensic DGP
DisableDirPage=no
UsePreviousAppDir=yes
DisableWelcomePage=no
AlwaysShowDirOnReadyPage=yes
#ifdef OfflinePayload
ExtraDiskSpaceRequired={#OfflineExpandedBytes + 536870912}
#else
ExtraDiskSpaceRequired=3_221_225_472
#endif
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=ForensicDGP_Setup
#ifdef OfflinePayload
Compression=zip/1
SolidCompression=no
#else
Compression=lzma2
SolidCompression=yes
#endif
WizardStyle=modern
InfoBeforeFile={#StageDir}\LICENSE.txt
SetupIconFile={#StageDir}\app.ico
UninstallDisplayIcon={app}\ForensicDGP.exe
CloseApplications=no
RestartApplications=no
AllowCancelDuringInstall=no
SetupLogging=yes

[Tasks]
Name: desktopicon; Description: "Create a Desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#StageDir}\ForensicDGP.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\uninstall.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\release.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\shell-owner.json"; DestDir: "{app}"; Flags: ignoreversion uninsneveruninstall
Source: "{#StageDir}\gui_setup.ps1"; Flags: dontcopy
Source: "{#StageDir}\distribution.ps1"; Flags: dontcopy
Source: "{#StageDir}\app-source.zip"; Flags: dontcopy
Source: "{#StageDir}\app-source.json"; Flags: dontcopy
Source: "{#StageDir}\install_location.ps1"; Flags: dontcopy
Source: "{#StageDir}\install_state.ps1"; Flags: dontcopy
Source: "{#StageDir}\check_update.ps1"; Flags: dontcopy
#ifdef OfflinePayload
Source: "{#StageDir}\payload\*"; DestDir: "{tmp}\payload"; Flags: dontcopy recursesubdirs createallsubdirs
#endif

[Icons]
Name: "{userprograms}\Forensic DGP"; Filename: "{app}\ForensicDGP.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\Forensic DGP"; Filename: "{app}\ForensicDGP.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\ForensicDGP.exe"; Description: "Open Forensic DGP"; Flags: nowait postinstall skipifsilent

[Messages]
ConfirmUninstall=Uninstall %1 and permanently delete its models, saved records and other data in the installation folder?

[UninstallDelete]
Type: files; Name: "{app}\install-location.json"
Type: files; Name: "{app}\shell-owner.json"
Type: dirifempty; Name: "{app}"

[Code]
var
  Preparing, CancelRequested, BackendReady: Boolean;
  ProgressPage: TOutputProgressWizardPage;
  EventsFile, CancelFile, SetupLog, LastStage, ActiveDownloadStage: String;
  UninstallMutex: THandle;
  RequirementsPage: TOutputMsgWizardPage;
  CheckUpdateButton, DownloadUpdateButton: TNewButton;
  UpdateStatus: TNewStaticText;
  UpdateOutput, AvailableVersion: String;
  StepLabel, TransferLabel: TNewStaticText;
  DetailsButton: TNewButton;
  DetailsMemo: TNewMemo;
  PreviousInstallRoot, SelectedMode, InstalledVersion, StateOutput: String;
  ModePage: TOutputMsgWizardPage;
  OpenExistingButton: TNewButton;
  CleanupPrevious: TNewCheckBox;
  CleanupHint: TNewStaticText;
  OpenOnly, RelocationCompleted, PreviousLauncherVerified: Boolean;
  StepNumber: Integer;

function GetFileAttributes(Name: String): LongWord;
  external 'GetFileAttributesW@kernel32.dll stdcall';
function SetFileAttributes(Name: String; Attr: LongWord): Boolean;
  external 'SetFileAttributesW@kernel32.dll stdcall';
function DeleteFileW(Name: String): Boolean;
  external 'DeleteFileW@kernel32.dll stdcall';
function RemoveDirectoryW(Name: String): Boolean;
  external 'RemoveDirectoryW@kernel32.dll stdcall';
function CloseHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';
function CreateMutex(Security: LongWord; InitialOwner: Boolean; Name: String): THandle;
  external 'CreateMutexW@kernel32.dll stdcall';
function WaitForSingleObject(Handle: THandle; Milliseconds: LongWord): LongWord;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function ReleaseMutex(Handle: THandle): Boolean;
  external 'ReleaseMutex@kernel32.dll stdcall';

function OrdinaryShellPath: Boolean;
var
  Path, Parent: String;
  Attr: LongWord;
begin
  Result := False;
  Path := ExpandConstant('{app}');
  repeat
    Attr := GetFileAttributes(Path);
    if (Attr <> $FFFFFFFF) and (((Attr and $400) <> 0) or ((Attr and $10) = 0)) then Exit;
    Parent := ExtractFileDir(Path);
    if (Parent = Path) or (Parent = '') then Break;
    Path := Parent;
  until False;
  Result := True;
end;

function Q(S: String): String;
begin
  Result := '"' + S + '"';
end;

function PS: String;
begin
  Result := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
end;

function JsonField(S, Field: String): String;
var
  P, I: Integer;
begin
  Result := '';
  P := Pos('"' + Field + '"', S);
  if P = 0 then Exit;
  S := Trim(Copy(S, P + Length(Field) + 2, Length(S)));
  if (Length(S) = 0) or (S[1] <> ':') then Exit;
  S := Trim(Copy(S, 2, Length(S)));
  if (Length(S) > 0) and (S[1] = '"') then begin
    S := Copy(S, 2, Length(S));
    P := Pos('"', S);
    if P > 0 then Result := Copy(S, 1, P - 1);
    StringChangeEx(Result, '\\', '\', True);
  end else begin
    I := 1;
    while (I <= Length(S)) and (S[I] >= '0') and (S[I] <= '9') do I := I + 1;
    Result := Copy(S, 1, I - 1);
  end;
end;

function RegisteredInstallRoot: String;
begin
  Result := '';
  RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{FEE649E9-195D-4576-8F30-DAE06A5099C8}_is1', 'InstallLocation', Result);
  Result := RemoveBackslashUnlessRoot(Result);
end;

procedure StateLog(const S: String; const Error, FirstLine: Boolean);
begin
  Log(S);
  if (not Error) and (Pos('"mode"', S) > 0) then StateOutput := S;
end;

function IsReleaseVersion(const Version: String): Boolean;
var
  I, Digits, Parts: Integer;
begin
  Result := False;
  Digits := 0;
  Parts := 0;
  for I := 1 to Length(Version) do begin
    if Version[I] = '.' then begin
      if Digits = 0 then Exit;
      Parts := Parts + 1;
      Digits := 0;
    end else begin
      if (Version[I] < '0') or (Version[I] > '9') then Exit;
      Digits := Digits + 1;
      if Digits > 4 then Exit;
    end;
  end;
  Result := (Parts = 2) and (Digits > 0);
end;

procedure UpdateCheckLog(const S: String; const Error, FirstLine: Boolean);
begin
  if not Error then UpdateOutput := Trim(S);
end;

procedure CheckForUpdate(Sender: TObject);
var
  ReleaseText: AnsiString;
  CheckPath, RegisteredVersion, LocalVersion: String;
  Code: Integer;
begin
  AvailableVersion := '';
  DownloadUpdateButton.Visible := False;
  CheckUpdateButton.Enabled := False;
  UpdateStatus.Caption := 'Checking for a newer installer...';
  UpdateStatus.Update;
  try
    try
      ExtractTemporaryFile('release.json');
      ExtractTemporaryFile('check_update.ps1');
      CheckPath := ExpandConstant('{tmp}\check_update.ps1');
      if (GetSHA256OfFile(ExpandConstant('{tmp}\release.json')) <> '{#ReleaseManifestSHA256}') or
         (not LoadStringFromFile(ExpandConstant('{tmp}\release.json'), ReleaseText)) or
         (GetSHA256OfFile(CheckPath) <> JsonField(String(ReleaseText), 'check_update.ps1')) then
        RaiseException('The embedded update helper could not be verified.');
      LocalVersion := '{#ReleaseVersion}';
      RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{FEE649E9-195D-4576-8F30-DAE06A5099C8}_is1', 'DisplayVersion', RegisteredVersion);
      if not IsReleaseVersion(RegisteredVersion) then RegisteredVersion := LocalVersion;
      UpdateOutput := '';
      if (not ExecAndLogOutput(PS, '-NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File ' +
         Q(CheckPath) + ' -LocalVersion ' + Q(LocalVersion) + ' -InstalledVersion ' + Q(RegisteredVersion),
         ExpandConstant('{tmp}'), SW_HIDE, ewWaitUntilTerminated, Code, @UpdateCheckLog)) or (Code <> 0) then
        RaiseException('The update check could not run.');
      if UpdateOutput = 'DGP_CURRENT' then
        UpdateStatus.Caption := 'No newer published installer found.'
      else begin
        if Pos('DGP_UPDATE ', UpdateOutput) <> 1 then RaiseException('The release could not be checked.');
        AvailableVersion := Copy(UpdateOutput, 12, Length(UpdateOutput));
        if not IsReleaseVersion(AvailableVersion) then RaiseException('The available version could not be verified.');
        UpdateStatus.Caption := 'Forensic DGP ' + AvailableVersion + ' is available.';
        DownloadUpdateButton.Visible := True;
      end;
    except
      UpdateStatus.Caption := 'Unable to check for updates. You can continue with this installer.';
      Log(GetExceptionMessage);
    end;
  finally
    CheckUpdateButton.Enabled := True;
  end;
end;

procedure DownloadUpdate(Sender: TObject);
var
  Code: Integer;
  Url: String;
begin
  if not IsReleaseVersion(AvailableVersion) then Exit;
  Url := 'https://github.com/janusinss/forensic-dgp-downloads/releases/download/v' + AvailableVersion +
         '/ForensicDGP_Setup-' + AvailableVersion + '.exe';
  if ShellExec('open', Url, '', '', SW_SHOWNORMAL, ewNoWait, Code) then
    UpdateStatus.Caption := 'After downloading, close this setup and run the newer installer.'
  else
    UpdateStatus.Caption := 'Unable to open the update download.';
end;

procedure PrepareStateFiles;
var
  ReleaseText: AnsiString;
begin
  ExtractTemporaryFile('release.json');
  ExtractTemporaryFile('install_location.ps1');
  ExtractTemporaryFile('install_state.ps1');
  ExtractTemporaryFile('app-source.json');
  if GetSHA256OfFile(ExpandConstant('{tmp}\release.json')) <> '{#ReleaseManifestSHA256}' then
    RaiseException('The embedded release manifest differs.');
  if not LoadStringFromFile(ExpandConstant('{tmp}\release.json'), ReleaseText) then
    RaiseException('The embedded release manifest could not be read.');
  if GetSHA256OfFile(ExpandConstant('{tmp}\install_location.ps1')) <> JsonField(String(ReleaseText), 'install_location.ps1') then
    RaiseException('The installation location helper differs.');
  if GetSHA256OfFile(ExpandConstant('{tmp}\install_state.ps1')) <> JsonField(String(ReleaseText), 'install_state.ps1') then
    RaiseException('The installation lifecycle helper differs.');
  if GetSHA256OfFile(ExpandConstant('{tmp}\app-source.json')) <> JsonField(String(ReleaseText), 'app-source.json') then
    RaiseException('The source history manifest differs.');
end;

function RunState(Action: String): Boolean;
var
  Params: String;
  Code: Integer;
begin
  StateOutput := '';
  Params := '-NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File ' +
    Q(ExpandConstant('{tmp}\install_state.ps1')) + ' -Action ' + Action +
    ' -InstallRoot ' + Q(WizardDirValue) + ' -ReleaseVersion {#ReleaseVersion} -ReleaseCommit {#ReleaseCommit}' +
    ' -SourceManifest ' + Q(ExpandConstant('{tmp}\app-source.json'));
  if PreviousInstallRoot <> '' then Params := Params + ' -RegisteredRoot ' + Q(PreviousInstallRoot);
  Result := ExecAndLogOutput(PS, Params, ExpandConstant('{tmp}'), SW_HIDE,
    ewWaitUntilTerminated, Code, @StateLog) and (Code = 0);
end;

function InspectInstallation: Boolean;
begin
  PrepareStateFiles;
  Result := RunState('Inspect');
  if not Result then begin
    MsgBox('The selected folder or existing installation could not be verified. Existing files were preserved. Choose a dedicated empty folder or review the setup log.', mbError, MB_OK);
    Exit;
  end;
  SelectedMode := JsonField(StateOutput, 'mode');
  InstalledVersion := JsonField(StateOutput, 'installed_version');
  PreviousLauncherVerified := JsonField(StateOutput, 'launcher_verified') = '1';
  PreviousInstallRoot := JsonField(StateOutput, 'previous_root');
  if SelectedMode = 'newer' then begin
    MsgBox('A newer Forensic DGP version is already installed. Open that version or download a newer setup. Existing files were preserved.', mbInformation, MB_OK);
    Result := False;
    Exit;
  end;
  if SelectedMode = 'move' then
    ModePage.MsgLabel.Caption := 'Move Forensic DGP to:' + #13#10 + WizardDirValue + #13#10#13#10 +
      'Setup copies saved records unchanged, reuses verified models and recreates Python environments. The previous installation remains until the replacement is verified.'
  else if SelectedMode = 'update' then
    ModePage.MsgLabel.Caption := 'Update Forensic DGP ' + InstalledVersion + ' to {#ReleaseVersion}.' + #13#10#13#10 +
      'The existing folder and saved records are retained. Verified components are reused.'
  else if SelectedMode = 'current' then
    ModePage.MsgLabel.Caption := 'Forensic DGP {#ReleaseVersion} is installed and passed source, package and model checks.' + #13#10#13#10 +
      'Open the installed app, or choose Next to repair it from this setup.'
  else if SelectedMode = 'repair' then
    ModePage.MsgLabel.Caption := 'Repair the existing Forensic DGP installation.' + #13#10#13#10 +
      'Setup prepares verified replacement components while preserving saved records and previous files.'
  else
    ModePage.MsgLabel.Caption := 'Install Forensic DGP in:' + #13#10 + WizardDirValue + #13#10#13#10 +
      'This dedicated folder contains the application, private Python, libraries, models and saved records.';
  OpenExistingButton.Visible := SelectedMode = 'current';
  CleanupPrevious.Visible := SelectedMode = 'move';
  CleanupHint.Visible := CleanupPrevious.Visible;
  CleanupPrevious.Checked := False;
end;

function CloseAppForSetup: Boolean;
var
  Code, I: Integer;
begin
  Result := True;
  if not CheckForMutexes('Local\ForensicDGP-Desktop') then Exit;
  Result := False;
  if PreviousLauncherVerified and (PreviousInstallRoot <> '') and (InstalledVersion <> '') and (CompareText(InstalledVersion, '0.2.0') >= 0) then begin
    if MsgBox('Forensic DGP is running. Close its dedicated window and continue setup? Active work must finish or be stopped before installation.', mbConfirmation, MB_YESNO) <> IDYES then Exit;
    if Exec(AddBackslash(PreviousInstallRoot) + 'ForensicDGP.exe', '--request-close', PreviousInstallRoot,
      SW_HIDE, ewWaitUntilTerminated, Code) and (Code = 0) then begin
      for I := 1 to 150 do begin
        if not CheckForMutexes('Local\ForensicDGP-Desktop') then begin Result := True; Exit; end;
        Sleep(100);
      end;
    end;
  end;
  MsgBox('Close the Forensic DGP window, then retry setup. No unrelated browser or Python processes were stopped.', mbInformation, MB_OK);
end;

procedure OpenExisting(Sender: TObject);
var
  Code: Integer;
begin
  if not InspectInstallation then Exit;
  if SelectedMode <> 'current' then Exit;
  if Exec(AddBackslash(WizardDirValue) + 'ForensicDGP.exe', '', WizardDirValue,
    SW_SHOWNORMAL, ewNoWait, Code) then begin
    OpenOnly := True;
    WizardForm.Close;
  end else MsgBox('The installed app could not be opened. Choose Repair or review its desktop log.', mbError, MB_OK);
end;

function PhaseNumber(Stage: String): Integer;
begin
  Result := 1;
  if (Pos('python', Lowercase(Stage)) > 0) then Result := 2;
  if (Stage = 'environment') or (Pos('download:', Stage) = 1) or (Pos('copy:', Stage) = 1) or
     (Pos('models', Lowercase(Stage)) > 0) or (Stage = 'Application dependencies') or
     (Stage = 'Capture runtime') or (Stage = 'capture-runtime') then Result := 3;
  if Stage = 'verification' then Result := 4;
  if (Stage = 'activation') or (Stage = 'complete') then Result := 5;
end;

function PhaseText(Number: Integer): String;
begin
  case Number of
    1: Result := 'Preparing application files';
    2: Result := 'Preparing Python';
#ifdef OfflinePayload
    3: Result := 'Installing bundled components';
#else
    3: Result := 'Installing components';
#endif
    4: Result := 'Checking the installation';
    5: Result := 'Finishing setup';
  end;
end;

procedure UpdateProgress(const Event: String);
var
  Current, Total, Percent, Phase: Integer;
  Stage, Status, Message: String;
begin
  if Event <> '' then begin
    Stage := JsonField(Event, 'stage');
    Status := JsonField(Event, 'status');
    Message := JsonField(Event, 'message');
    if Stage <> '' then LastStage := Stage;
    Phase := PhaseNumber(Stage);
    if Phase > StepNumber then StepNumber := Phase;
    StepLabel.Caption := 'Step ' + IntToStr(StepNumber) + ' of 5';
    if (LastStage = 'activation') or (LastStage = 'complete') then WizardForm.CancelButton.Enabled := False;
    Current := StrToIntDef(JsonField(Event, 'current'), -1);
    Total := StrToIntDef(JsonField(Event, 'total'), -1);
    { Keep one measured transfer visible while parallel tasks continue in details. }
    if (Status = 'error') or (Status = 'cancelled') or (Stage = 'error') or
       (Stage = 'activation') or (Stage = 'complete') then ActiveDownloadStage := ''
    else if ActiveDownloadStage <> '' then begin
      if Stage <> ActiveDownloadStage then Exit;
      { Completion or verification of the visible file releases the display. }
      if (Status <> 'downloading') and (Status <> 'copying') then ActiveDownloadStage := '';
    end;
    if (Total > 0) and (Current >= 0) and (Current <= Total) and ((Status = 'downloading') or (Status = 'copying')) then begin
      ActiveDownloadStage := Stage;
      Percent := Round((Current * 100.0) / Total);
      ProgressPage.ProgressBar.Style := npbstNormal;
      ProgressPage.SetProgress(Percent, 100);
      TransferLabel.Caption := IntToStr(Percent) + '%  |  ' +
        Format('%.1f of %.1f MiB', [Current / 1048576.0, Total / 1048576.0]) + '  |  ' + Message;
      if Status = 'copying' then Log('DGP_PROGRESS copying ' + IntToStr(Percent) + ' ' + Stage)
      else Log('DGP_PROGRESS download ' + IntToStr(Percent) + ' ' + Stage);
    end else begin
      ProgressPage.SetProgress(0, 100);
      ProgressPage.ProgressBar.Style := npbstMarquee;
      TransferLabel.Caption := Message;
      if Status = 'complete' then TransferLabel.Caption := Message + ' ready';
      Log('DGP_PROGRESS activity ' + Stage);
    end;
    if Stage = 'complete' then begin
      ProgressPage.ProgressBar.Style := npbstNormal;
      ProgressPage.SetProgress(100, 100);
      TransferLabel.Caption := 'Application verified. Creating your shortcuts next.';
    end;
  end;
  ProgressPage.ProgressBar.Visible := True;
  if CancelRequested and (LastStage <> 'complete') then
    ProgressPage.SetText('Stopping safely', 'Finishing the current operation. Saved records are retained.')
  else
    ProgressPage.SetText(PhaseText(StepNumber), 'Keep this window open. You can view details below.');
end;

procedure ToggleDetails(Sender: TObject);
begin
  DetailsMemo.Visible := not DetailsMemo.Visible;
  if DetailsMemo.Visible then DetailsButton.Caption := 'Hide details'
  else DetailsButton.Caption := 'Show details';
end;

procedure SetupOutput(const S: String; const Error, FirstLine: Boolean);
begin
  Log(S);
  SaveStringToFile(SetupLog, S + #13#10, True);
  DetailsMemo.Lines.Add(S);
  if DetailsMemo.Lines.Count > 120 then DetailsMemo.Lines.Delete(0);
  if Pos('DGP_EVENT ', S) = 1 then UpdateProgress(Copy(S, 11, Length(S)));
  if (Pos('"mode"', S) > 0) and (Pos('"install_root"', S) > 0) then begin
    SelectedMode := JsonField(S, 'mode');
    PreviousInstallRoot := JsonField(S, 'previous_root');
  end;
end;

procedure InitializeWizard;
begin
  PreviousInstallRoot := RegisteredInstallRoot;
  RequirementsPage := CreateOutputMsgPage(wpWelcome, 'Before you install',
    'A local workspace for pictures and video',
#ifdef OfflinePayload
    'You need Windows 10 or 11 (64-bit), Microsoft Edge and .NET Framework 4.8 or newer.' + #13#10#13#10 +
    'This offline setup includes private Python, application libraries and verified models. No internet is needed after downloading this EXE.' + #13#10#13#10 +
#else
    'You need Windows 10 or 11 (64-bit), Microsoft Edge and internet access for first setup.' + #13#10#13#10 +
    'Setup downloads private Python, application libraries and verified models.' + #13#10#13#10 +
#endif
    'The folder you choose contains the full application, downloaded components and saved records. No administrator access is required.' + #13#10#13#10 +
    'This is an unsigned research preview. Generated facial details are estimates and cannot verify identity.');
  RequirementsPage.MsgLabel.Height := ScaleY(155);
  CheckUpdateButton := TNewButton.Create(WizardForm);
  CheckUpdateButton.Parent := RequirementsPage.Surface;
  CheckUpdateButton.SetBounds(0, ScaleY(161), ScaleX(145), ScaleY(28));
  CheckUpdateButton.Caption := 'Check for updates';
  CheckUpdateButton.OnClick := @CheckForUpdate;
  DownloadUpdateButton := TNewButton.Create(WizardForm);
  DownloadUpdateButton.Parent := RequirementsPage.Surface;
  DownloadUpdateButton.SetBounds(ScaleX(153), ScaleY(161), ScaleX(135), ScaleY(28));
  DownloadUpdateButton.Caption := 'Download update';
  DownloadUpdateButton.OnClick := @DownloadUpdate;
  DownloadUpdateButton.Visible := False;
  UpdateStatus := TNewStaticText.Create(WizardForm);
  UpdateStatus.Parent := RequirementsPage.Surface;
  UpdateStatus.AutoSize := False;
  UpdateStatus.WordWrap := True;
  UpdateStatus.SetBounds(0, ScaleY(194), RequirementsPage.SurfaceWidth, ScaleY(34));
  ModePage := CreateOutputMsgPage(wpSelectDir, 'Installation options', 'Keep your existing work', '');
  ModePage.MsgLabel.Height := ScaleY(125);
  OpenExistingButton := TNewButton.Create(WizardForm);
  OpenExistingButton.Parent := ModePage.Surface;
  OpenExistingButton.SetBounds(0, ScaleY(145), ScaleX(155), ScaleY(30));
  OpenExistingButton.Caption := 'Open installed app';
  OpenExistingButton.OnClick := @OpenExisting;
  OpenExistingButton.Visible := False;
  CleanupPrevious := TNewCheckBox.Create(WizardForm);
  CleanupPrevious.Parent := ModePage.Surface;
  CleanupPrevious.SetBounds(0, ScaleY(145), ModePage.SurfaceWidth, ScaleY(28));
  CleanupPrevious.Caption := 'Remove verified files from the previous folder after moving.';
  CleanupPrevious.Visible := False;
  CleanupHint := TNewStaticText.Create(WizardForm);
  CleanupHint.Parent := ModePage.Surface;
  CleanupHint.SetBounds(0, ScaleY(183), ModePage.SurfaceWidth, ScaleY(45));
  CleanupHint.WordWrap := True;
  CleanupHint.Caption := 'Unrecognized files and unverified legacy Python, environments or caches remain in the previous folder.';
  CleanupHint.Visible := False;
  ProgressPage := CreateOutputProgressPage('Installing Forensic DGP', 'Preparing your local workspace');
  ProgressPage.ProgressBar.Top := ScaleY(92);
  ProgressPage.ProgressBar.Visible := True;
  StepLabel := TNewStaticText.Create(WizardForm);
  StepLabel.Parent := ProgressPage.Surface;
  StepLabel.SetBounds(0, ScaleY(112), ProgressPage.SurfaceWidth, ScaleY(20));
  StepLabel.Caption := 'Step 1 of 5';
  TransferLabel := TNewStaticText.Create(WizardForm);
  TransferLabel.Parent := ProgressPage.Surface;
  TransferLabel.SetBounds(0, ScaleY(136), ProgressPage.SurfaceWidth, ScaleY(44));
  TransferLabel.WordWrap := True;
  DetailsButton := TNewButton.Create(WizardForm);
  DetailsButton.Parent := ProgressPage.Surface;
  DetailsButton.SetBounds(0, ScaleY(182), ScaleX(110), ScaleY(26));
  DetailsButton.Caption := 'Show details';
  DetailsButton.OnClick := @ToggleDetails;
  DetailsMemo := TNewMemo.Create(WizardForm);
  DetailsMemo.Parent := ProgressPage.Surface;
  DetailsMemo.SetBounds(0, ScaleY(216), ProgressPage.SurfaceWidth, ScaleY(60));
  DetailsMemo.ReadOnly := True;
  DetailsMemo.ScrollBars := ssVertical;
  DetailsMemo.Visible := False;
  WizardForm.SelectDirLabel.Caption := 'Choose a folder for the full app, Python, models and saved records.';
  WizardForm.FinishedLabel.Caption := 'Forensic DGP is ready. Open it from the Start Menu or your Desktop shortcut.' + #13#10#13#10 +
    'Uninstalling removes the app, models and saved records from the folder you selected.';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  FreeBytes, TotalBytes, RequiredBytes: Int64;
  Drive: String;
begin
  Result := True;
  if CurPageID = wpSelectDir then begin
#ifdef OfflinePayload
    RequiredBytes := {#OfflineExpandedBytes} + {#OfflinePayloadBytes} + 1073741824;
#else
    RequiredBytes := 3221225472;
#endif
    Drive := ExtractFileDrive(WizardDirValue);
    if (Length(Drive) <> 2) or (Drive[2] <> ':') then begin
      MsgBox('Choose a dedicated folder on a local fixed drive.', mbInformation, MB_OK);
      Result := False; Exit;
    end;
    if not GetSpaceOnDisk64(Drive + '\', FreeBytes, TotalBytes) then begin
      MsgBox('Setup could not check free disk space. Choose a ready local drive.', mbError, MB_OK);
      Result := False; Exit;
    end;
    if FreeBytes < RequiredBytes then begin
      MsgBox('Not enough free disk space for preparation and verified replacement files. Choose another drive or free disk space.', mbInformation, MB_OK);
      Result := False; Exit;
    end;
    Result := InspectInstallation;
  end;
  if (CurPageID = wpReady) and Result then Result := CloseAppForSetup;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpReady then begin
    if SelectedMode = 'move' then WizardForm.NextButton.Caption := 'Move and install'
    else if SelectedMode = 'update' then WizardForm.NextButton.Caption := 'Update'
    else if (SelectedMode = 'current') or (SelectedMode = 'repair') then WizardForm.NextButton.Caption := 'Repair';
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := 'Full application folder:' + NewLine + Space + WizardDirValue + NewLine + NewLine +
    'Includes:' + NewLine + Space + 'Application, private Python, libraries and verified models' + NewLine +
    Space + 'Saved records: ' + AddBackslash(WizardDirValue) + 'runtime\data' + NewLine + NewLine +
#ifdef OfflinePayload
    'Offline installation. Existing saved records are preserved.' + NewLine + NewLine + MemoTasksInfo;
#else
    'First setup requires internet access. Existing saved records are preserved.' + NewLine + NewLine + MemoTasksInfo;
#endif
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  if OpenOnly then begin Cancel := True; Confirm := False; Exit; end;
  if Preparing then begin
    Cancel := False;
    Confirm := False;
    if (LastStage = 'activation') or (LastStage = 'complete') then Exit;
    if not CancelRequested then begin
      CancelRequested := True;
      SaveStringToFile(CancelFile, 'cancel', False);
      WizardForm.CancelButton.Enabled := False;
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  Params, TempDrive: String;
  TempFree, TempTotal: Int64;
begin
  Result := '';
  if BackendReady then Exit;
  if not OrdinaryShellPath then begin
    Result := 'The launcher location contains a link or an unexpected file. Existing files were preserved.';
    Exit;
  end;
  if not CloseAppForSetup then begin Result := 'Close Forensic DGP before retrying setup.'; Exit; end;
#ifdef OfflinePayload
  TempDrive := ExtractFileDrive(ExpandConstant('{tmp}')) + '\';
  if (not GetSpaceOnDisk64(TempDrive, TempFree, TempTotal)) or
     (TempFree < {#OfflinePayloadBytes} + 268435456) then begin
    Result := 'Not enough space on the Windows temporary drive to unpack the offline installer. Free temporary-drive space before retrying.';
    Exit;
  end;
#endif
  ExtractTemporaryFile('gui_setup.ps1');
  ExtractTemporaryFile('check_update.ps1');
  ExtractTemporaryFile('distribution.ps1');
  ExtractTemporaryFile('app-source.zip');
  ExtractTemporaryFile('app-source.json');
  { Installed files also extracted privately for release input verification. }
  ExtractTemporaryFile('ForensicDGP.exe');
  ExtractTemporaryFile('uninstall.ps1');
  ExtractTemporaryFile('release.json');
  PrepareStateFiles;
  EventsFile := ExpandConstant('{tmp}\setup-events.jsonl');
  CancelFile := ExpandConstant('{tmp}\setup-cancel');
  SetupLog := ExpandConstant('{localappdata}\DGP-setup-{#ReleaseCommit}.log');
  LastStage := 'source verification';
  ActiveDownloadStage := '';
  StepNumber := 1;
  CancelRequested := False;
  DeleteFile(CancelFile);
  SaveStringToFile(EventsFile, '', False);
  Params := '-NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File ' + Q(ExpandConstant('{tmp}\gui_setup.ps1')) +
    ' -Stage ' + Q(ExpandConstant('{tmp}')) + ' -InstallRoot ' + Q(ExpandConstant('{app}')) + ' -EventsFile ' + Q(EventsFile) +
    ' -CancelFile ' + Q(CancelFile) + ' -LogPath ' + Q(SetupLog) +
    ' -ExpectedManifestHash {#ReleaseManifestSHA256}';
  if PreviousInstallRoot <> '' then Params := Params + ' -PreviousRoot ' + Q(PreviousInstallRoot);
  Preparing := True;
  ProgressPage.Show;
  try
    WizardForm.CancelButton.Enabled := True;
    ProgressPage.SetProgress(0, 100);
    ProgressPage.ProgressBar.Style := npbstMarquee;
    ProgressPage.ProgressBar.Visible := True;
#ifdef OfflinePayload
    ProgressPage.SetText('Preparing bundled files', 'Extracting verified offline components. No download is required.');
    ExtractTemporaryFiles('{tmp}\payload\*');
#endif
    if not ExecAndLogOutput(PS, Params, ExpandConstant('{tmp}'), SW_HIDE,
        ewWaitUntilTerminated, ResultCode, @SetupOutput) then begin
      Result := 'Could not run application setup. Log: ' + SetupLog;
      Exit;
    end;
    UpdateProgress('');
    if ResultCode = 0 then BackendReady := True
    else Result := 'Setup stopped before completion. Saved records were preserved. Review the active version and log: ' + SetupLog;
  finally
    Preparing := False;
    ProgressPage.Hide;
  end;
end;

var
  CheckOutput, CheckFailure: String;

procedure UninstallLog(const S: String; const Error, FirstLine: Boolean);
begin
  Log(S);
  if Error then CheckFailure := S;
  if not Error and (Pos('"mutex_name"', S) > 0) then CheckOutput := S;
end;

function InitializeUninstall: Boolean;
var
  Code: Integer;
  Name: String;
  WaitCode: LongWord;
begin
  Result := False;
  if not OrdinaryShellPath then Exit;
  if (RegisteredInstallRoot <> '') and (CompareText(RegisteredInstallRoot, RemoveBackslashUnlessRoot(ExpandConstant('{app}'))) <> 0) then begin
    MsgBox('A different Forensic DGP location is currently registered. This older uninstaller left all files unchanged.', mbInformation, MB_OK);
    Exit;
  end;
  if GetSHA256OfFile(ExpandConstant('{app}\uninstall.ps1')) <> '{#UninstallSHA256}' then begin
    MsgBox('The uninstall helper differs. Files were preserved.', mbError, MB_OK);
    Exit;
  end;
  CheckOutput := '';
  CheckFailure := '';
  if not ExecAndLogOutput(PS, '-NoProfile -NonInteractive -ExecutionPolicy RemoteSigned -File ' +
      Q(ExpandConstant('{app}\uninstall.ps1')) + ' -Action Check -InstallRoot ' + Q(ExpandConstant('{app}')) + ' -ManagedRoot ' +
      Q(ExpandConstant('{app}\runtime')), ExpandConstant('{app}'), SW_HIDE,
      ewWaitUntilTerminated, Code, @UninstallLog) or (Code <> 0) then begin
    if CheckFailure = '' then CheckFailure := 'The installation could not be verified.';
    SuppressibleMsgBox(CheckFailure + ' Files were preserved.', mbError, MB_OK, IDOK);
    Exit;
  end;
  if JsonField(CheckOutput, 'policy') <> 'remove_owned_runtime' then Exit;
  Name := JsonField(CheckOutput, 'mutex_name');
  if (Name = '') or (Pos('Local\DGP-app-', Name) <> 1) then Exit;
  UninstallMutex := CreateMutex(0, False, Name);
  if UninstallMutex = 0 then Exit;
  WaitCode := WaitForSingleObject(UninstallMutex, 0);
  if (WaitCode <> 0) and (WaitCode <> $80) then begin
    CloseHandle(UninstallMutex);
    UninstallMutex := 0;
    MsgBox('The application or installer is running. Close it before uninstalling.', mbError, MB_OK);
    Exit;
  end;
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssInstall) and not OrdinaryShellPath then
    RaiseException('The launcher path changed. Existing files were preserved.');
  if (CurStep = ssPostInstall) and (SelectedMode = 'move') then begin
    if RunState('CompleteMove') then begin
      RelocationCompleted := True;
      if CleanupPrevious.Checked then begin
        if not RunState('Cleanup') then
          MsgBox('The new installation is ready. Previous-file cleanup did not complete; the previous folder was preserved for review: ' + PreviousInstallRoot, mbInformation, MB_OK)
        else
          MsgBox('Verified previous app files were removed. Unrecognized files and unverified legacy Python, environments or caches remain in: ' + PreviousInstallRoot, mbInformation, MB_OK);
      end;
    end else
      MsgBox('The new installation is ready, but relocation completion could not be recorded. The previous folder remains unchanged: ' + PreviousInstallRoot, mbInformation, MB_OK);
  end;
end;

procedure DeinitializeUninstall;
begin
  if UninstallMutex <> 0 then begin
    ReleaseMutex(UninstallMutex);
    CloseHandle(UninstallMutex);
    UninstallMutex := 0;
  end;
end;

function MatchesUninstallRecord(Path, ExpectedHash: String): Boolean;
var
  Attr: LongWord;
begin
  Attr := GetFileAttributes(Path);
  Result := (Length(ExpectedHash) = 64) and (Attr <> $FFFFFFFF) and ((Attr and $410) = 0);
  if Result then Result := GetSHA256OfFile(Path) = ExpectedHash;
end;

function LongWinPath(const Path: String): String;
begin
  if Pos('\\?\', Path) = 1 then
    Result := Path
  else
    Result := '\\?\' + Path;
end;

function RemoveOwnedTree(const Path: String): Boolean;
var
  LongPath: String;
  Attr: LongWord;
  Entry: TFindRec;
begin
  Result := False;
  LongPath := LongWinPath(Path);
  Attr := GetFileAttributes(LongPath);
  if Attr = $FFFFFFFF then begin
    Result := True;
    Exit;
  end;
  if (Attr and $10) = 0 then begin
    if (Attr and 1) <> 0 then
      SetFileAttributes(LongPath, Attr and (not 1));
    Result := DeleteFileW(LongPath);
    Exit;
  end;
  if (Attr and $400) <> 0 then begin
    { Do not recurse into junctions or directory reparse points. }
    if (Attr and 1) <> 0 then
      SetFileAttributes(LongPath, Attr and (not 1));
    Result := RemoveDirectoryW(LongPath);
    Exit;
  end;
  if FindFirst(AddBackslash(LongPath) + '*', Entry) then begin
    try
      repeat
        if (Entry.Name <> '.') and (Entry.Name <> '..') then begin
          if not RemoveOwnedTree(AddBackslash(Path) + Entry.Name) then Exit;
        end;
      until not FindNext(Entry);
    finally
      FindClose(Entry);
    end;
  end;
  if (Attr and 1) <> 0 then
    SetFileAttributes(LongPath, Attr and (not 1));
  Result := RemoveDirectoryW(LongPath);
end;

procedure RemoveOwnedRuntime;
var
  Root, Owner, OwnerHash, Child: String;
  OwnerText: AnsiString;
  Attr: LongWord;
  Entry: TFindRec;
  Entries: TStringList;
  I: Integer;
begin
  Root := ExpandConstant('{app}\runtime');
  Owner := AddBackslash(Root) + 'owner.json';
  OwnerHash := JsonField(CheckOutput, 'runtime_owner_sha256');
  Attr := GetFileAttributes(LongWinPath(Root));
  if (Attr = $FFFFFFFF) and (OwnerHash = '') then Exit;
  if (Attr = $FFFFFFFF) or ((Attr and $410) <> $10) then
    RaiseException('The application runtime path changed. Uninstall stopped.');
  if OwnerHash = '' then begin
    if not RemoveDirectoryW(LongWinPath(Root)) then RaiseException('The runtime folder is no longer empty or cannot be removed.');
    Exit;
  end;
  if not MatchesUninstallRecord(Owner, OwnerHash) or not LoadStringFromFile(Owner, OwnerText) then
    RaiseException('The runtime ownership record changed. Uninstall stopped.');
  Entries := TStringList.Create;
  try
    if not FindFirst(AddBackslash(LongWinPath(Root)) + '*', Entry) then
      RaiseException('The runtime contents could not be read.');
    try
      repeat
        if (Entry.Name <> '.') and (Entry.Name <> '..') and (CompareText(Entry.Name, 'owner.json') <> 0) then
          Entries.Add(Entry.Name);
      until not FindNext(Entry);
    finally
      FindClose(Entry);
    end;
    for I := 0 to Entries.Count - 1 do begin
      if not OrdinaryShellPath or ((GetFileAttributes(LongWinPath(Root)) and $410) <> $10) then
        RaiseException('The application runtime path changed. Uninstall stopped.');
      Child := AddBackslash(Root) + Entries[I];
      Attr := GetFileAttributes(LongWinPath(Child));
      { RemoveOwnedTree handles >260-char paths, read-only files, and removes junctions without traversing targets. }
      if (Attr <> $FFFFFFFF) and not RemoveOwnedTree(Child) then
        RaiseException('Could not remove ' + Child + '. Close programs using that folder and retry uninstall.');
    end;
    if not MatchesUninstallRecord(Owner, OwnerHash) or not DeleteFileW(LongWinPath(Owner)) then
      RaiseException('Could not remove the runtime ownership record. Retry uninstall.');
    if not RemoveDirectoryW(LongWinPath(Root)) then begin
      if OrdinaryShellPath and ((GetFileAttributes(LongWinPath(Root)) and $410) = $10) and
          (GetFileAttributes(LongWinPath(Owner)) = $FFFFFFFF) then
        SaveStringToFile(Owner, OwnerText, False);
      RaiseException('Could not remove the runtime folder. Close programs using that folder and retry uninstall.');
    end;
  finally
    Entries.Free;
  end;
end;

procedure RemoveLocationBackups;
var
  Backups, RecordParts: TArrayOfString;
  I: Integer;
  Name, Path: String;
begin
  if not OrdinaryShellPath then RaiseException('The installation folder changed before cleanup.');
  Backups := StringSplit(JsonField(CheckOutput, 'location_backups'), ['|'], stExcludeEmpty);
  for I := 0 to GetArrayLength(Backups) - 1 do begin
    RecordParts := StringSplit(Backups[I], [':'], stExcludeEmpty);
    if GetArrayLength(RecordParts) <> 2 then Continue;
    Name := RecordParts[0];
    if (Name <> ExtractFileName(Name)) or (Pos('install-location.json.previous.', Name) <> 1) or
        (Length(Name) <> Length('install-location.json.previous.') + 32) then Continue;
    Path := AddBackslash(ExpandConstant('{app}')) + Name;
    if MatchesUninstallRecord(Path, RecordParts[1]) and not DeleteFile(Path) then
      RaiseException('Could not remove ' + Path + '. Close programs using that file and retry uninstall.');
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then begin
    if (UninstallMutex = 0) or not OrdinaryShellPath or
        ((RegisteredInstallRoot <> '') and (CompareText(RegisteredInstallRoot, RemoveBackslashUnlessRoot(ExpandConstant('{app}'))) <> 0)) or
        not MatchesUninstallRecord(ExpandConstant('{app}\shell-owner.json'), JsonField(CheckOutput, 'shell_owner_sha256')) or
        not MatchesUninstallRecord(ExpandConstant('{app}\install-location.json'), JsonField(CheckOutput, 'location_sha256')) then
      RaiseException('The installation changed before removal. Uninstall stopped.');
    RemoveOwnedRuntime;
    RemoveLocationBackups;
  end;
  if (CurUninstallStep = usPostUninstall) and OrdinaryShellPath then
    RemoveDir(ExpandConstant('{app}'));
end;
