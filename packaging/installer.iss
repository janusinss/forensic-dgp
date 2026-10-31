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
AppComments=Experimental local face restoration
DefaultDirName={localappdata}\Programs\Forensic DGP
DefaultGroupName=Forensic DGP
DisableDirPage=no
UsePreviousAppDir=yes
DisableWelcomePage=no
AlwaysShowDirOnReadyPage=yes
ExtraDiskSpaceRequired=3_221_225_472
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=ForensicDGP_Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
InfoBeforeFile={#StageDir}\LICENSE.txt
SetupIconFile={#StageDir}\app.ico
UninstallDisplayIcon={app}\ForensicDGP.exe
AppMutex=Local\ForensicDGP-Desktop
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

[Icons]
Name: "{userprograms}\Forensic DGP"; Filename: "{app}\ForensicDGP.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\Forensic DGP"; Filename: "{app}\ForensicDGP.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\ForensicDGP.exe"; Description: "Open Forensic DGP"; Flags: nowait postinstall skipifsilent

[Code]
var
  Preparing, CancelRequested, BackendReady: Boolean;
  ProgressPage: TOutputProgressWizardPage;
  EventsFile, CancelFile, SetupLog, LastStage, ActiveDownloadStage: String;
  UninstallMutex: THandle;
  RequirementsPage: TOutputMsgWizardPage;
  StepLabel, TransferLabel: TNewStaticText;
  DetailsButton: TNewButton;
  DetailsMemo: TNewMemo;
  PreviousInstallRoot: String;
  StepNumber: Integer;

function GetFileAttributes(Name: String): LongWord;
  external 'GetFileAttributesW@kernel32.dll stdcall';
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

function PhaseNumber(Stage: String): Integer;
begin
  Result := 1;
  if (Pos('python', Lowercase(Stage)) > 0) then Result := 2;
  if (Stage = 'environment') or (Pos('download:', Stage) = 1) or
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
    3: Result := 'Downloading and setting up components';
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
      if Status <> 'downloading' then ActiveDownloadStage := '';
    end;
    if (Total > 0) and (Current >= 0) and (Current <= Total) and (Status = 'downloading') then begin
      ActiveDownloadStage := Stage;
      Percent := Round((Current * 100.0) / Total);
      ProgressPage.ProgressBar.Style := npbstNormal;
      ProgressPage.SetProgress(Percent, 100);
      TransferLabel.Caption := IntToStr(Percent) + '%  |  ' +
        Format('%.1f of %.1f MiB', [Current / 1048576.0, Total / 1048576.0]) + '  |  ' + Message;
      Log('DGP_PROGRESS download ' + IntToStr(Percent) + ' ' + Stage);
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
end;

procedure InitializeWizard;
begin
  PreviousInstallRoot := RegisteredInstallRoot;
  RequirementsPage := CreateOutputMsgPage(wpWelcome, 'Before you install',
    'A local workspace for pictures and video',
    'You need Windows 10 or 11 (64-bit), Microsoft Edge and internet access for first setup.' + #13#10#13#10 +
    'Setup downloads a private Python runtime, application libraries and about 720 MiB of models. Allow at least 4 GB of free disk space.' + #13#10#13#10 +
    'The folder you choose contains the full application, downloaded components and saved records. No administrator access is required.' + #13#10#13#10 +
    'This is an unsigned research preview. Generated facial details are estimates and cannot verify identity.');
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
    'Your application data stays in the folder you selected. Uninstalling the Windows launcher keeps that data.';
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and (PreviousInstallRoot <> '') and
     (CompareText(RemoveBackslashUnlessRoot(WizardDirValue), PreviousInstallRoot) <> 0) then begin
    MsgBox('To choose a different folder, uninstall the Windows launcher first. Your existing models and saved records are retained. Setup does not move or delete the previous installation.', mbInformation, MB_OK);
    Result := False;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := 'Full application folder:' + NewLine + Space + WizardDirValue + NewLine + NewLine +
    'Includes:' + NewLine + Space + 'Application, private Python, libraries and verified models' + NewLine +
    Space + 'Saved records: ' + AddBackslash(WizardDirValue) + 'runtime\data' + NewLine + NewLine +
    'First setup requires internet access. Existing saved records are preserved.' + NewLine + NewLine + MemoTasksInfo;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
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
  Marker: AnsiString;
  Params: String;
begin
  Result := '';
  if BackendReady then Exit;
  if not OrdinaryShellPath then begin
    Result := 'The launcher location contains a link or an unexpected file. Existing files were preserved.';
    Exit;
  end;
  if (PreviousInstallRoot <> '') and
     (CompareText(RemoveBackslashUnlessRoot(ExpandConstant('{app}')), PreviousInstallRoot) <> 0) then begin
    Result := 'To choose a different folder, uninstall the Windows launcher first. Saved records remain in the previous location.';
    Exit;
  end;
  ExtractTemporaryFile('gui_setup.ps1');
  ExtractTemporaryFile('distribution.ps1');
  ExtractTemporaryFile('app-source.zip');
  ExtractTemporaryFile('app-source.json');
  { Installed files also extracted privately for release input verification. }
  ExtractTemporaryFile('ForensicDGP.exe');
  ExtractTemporaryFile('uninstall.ps1');
  ExtractTemporaryFile('release.json');
  ExtractTemporaryFile('install_location.ps1');
  EventsFile := ExpandConstant('{tmp}\setup-events.jsonl');
  CancelFile := ExpandConstant('{tmp}\setup-cancel');
  SetupLog := ExpandConstant('{localappdata}\DGP-setup-{#ReleaseCommit}.log');
  LastStage := 'source verification';
  ActiveDownloadStage := '';
  StepNumber := 1;
  CancelRequested := False;
  DeleteFile(CancelFile);
  SaveStringToFile(EventsFile, '', False);
  Params := '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File ' + Q(ExpandConstant('{tmp}\gui_setup.ps1')) +
    ' -Stage ' + Q(ExpandConstant('{tmp}')) + ' -InstallRoot ' + Q(ExpandConstant('{app}')) + ' -EventsFile ' + Q(EventsFile) +
    ' -CancelFile ' + Q(CancelFile) + ' -LogPath ' + Q(SetupLog) +
    ' -ExpectedManifestHash {#ReleaseManifestSHA256}';
  Preparing := True;
  ProgressPage.Show;
  try
    WizardForm.CancelButton.Enabled := True;
    ProgressPage.SetProgress(0, 100);
    ProgressPage.ProgressBar.Style := npbstMarquee;
    ProgressPage.ProgressBar.Visible := True;
    if not ExecAndLogOutput(PS, Params, ExpandConstant('{tmp}'), SW_SHOWNORMAL,
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
  CheckOutput: String;

procedure UninstallLog(const S: String; const Error, FirstLine: Boolean);
begin
  Log(S);
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
  if not ExecAndLogOutput(PS, '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File ' +
      Q(ExpandConstant('{app}\uninstall.ps1')) + ' -Action Check -InstallRoot ' + Q(ExpandConstant('{app}')) + ' -ManagedRoot ' +
      Q(ExpandConstant('{app}\runtime')), ExpandConstant('{app}'), SW_SHOWNORMAL,
      ewWaitUntilTerminated, Code, @UninstallLog) or (Code <> 0) then begin
    MsgBox('Close Forensic DGP or its installer before uninstalling. Shared files were preserved.', mbError, MB_OK);
    Exit;
  end;
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
end;

procedure DeinitializeUninstall;
begin
  if UninstallMutex <> 0 then begin
    ReleaseMutex(UninstallMutex);
    CloseHandle(UninstallMutex);
    UninstallMutex := 0;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if (CurUninstallStep = usDone) and not UninstallSilent then
    MsgBox('Windows launcher removed. Shared application files, models and saved records remain in ' +
      ExpandConstant('{app}\runtime') + '.', mbInformation, MB_OK);
end;
