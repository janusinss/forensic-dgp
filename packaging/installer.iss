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
DisableDirPage=yes
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
LicenseFile={#StageDir}\LICENSE.txt
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
Source: "{#StageDir}\shell-owner.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#StageDir}\gui_setup.ps1"; Flags: dontcopy
Source: "{#StageDir}\distribution.ps1"; Flags: dontcopy
Source: "{#StageDir}\app-source.zip"; Flags: dontcopy
Source: "{#StageDir}\app-source.json"; Flags: dontcopy

[Icons]
Name: "{userprograms}\Forensic DGP"; Filename: "{app}\ForensicDGP.exe"; WorkingDir: "{app}"
Name: "{userdesktop}\Forensic DGP"; Filename: "{app}\ForensicDGP.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\ForensicDGP.exe"; Description: "Open Forensic DGP"; Flags: nowait postinstall skipifsilent

[Code]
var
  Preparing, CancelRequested, BackendReady: Boolean;
  ProgressPage: TOutputProgressWizardPage;
  EventsFile, CancelFile, SetupLog, LastStage: String;
  UninstallMutex: THandle;

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

procedure UpdateProgress(const Event: String);
var
  Current, Total: Integer;
  Stage: String;
begin
  if Event <> '' then begin
    Stage := JsonField(Event, 'stage');
    if Stage <> '' then LastStage := Stage;
    if (LastStage = 'activation') or (LastStage = 'complete') then WizardForm.CancelButton.Enabled := False;
    Current := StrToIntDef(JsonField(Event, 'current'), 0);
    Total := StrToIntDef(JsonField(Event, 'total'), 0);
    if (Total > 0) and (Current >= 0) and (Current <= Total) then
      ProgressPage.SetProgress(Current, Total)
    else ProgressPage.SetProgress(0, 0);
  end;
  if CancelRequested and (LastStage <> 'complete') then
    ProgressPage.SetText('Cancelling at a safe boundary', 'Finishing the current operation. Saved records are retained.')
  else
    ProgressPage.SetText('Preparing application: ' + LastStage, 'Downloads and verification can take several minutes. Log: ' + SetupLog);
end;

procedure SetupOutput(const S: String; const Error, FirstLine: Boolean);
begin
  Log(S);
  SaveStringToFile(SetupLog, S + #13#10, True);
  if Pos('DGP_EVENT ', S) = 1 then UpdateProgress(Copy(S, 11, Length(S)));
end;

procedure InitializeWizard;
begin
  ProgressPage := CreateOutputProgressPage('Prepare Forensic DGP', 'Verified source, Python packages and models');
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
  Params, FixedDir: String;
begin
  Result := '';
  if BackendReady then Exit;
  if not OrdinaryShellPath then begin
    Result := 'The launcher location contains a link or an unexpected file. Existing files were preserved.';
    Exit;
  end;
  FixedDir := ExpandConstant('{localappdata}\Programs\Forensic DGP');
  if CompareText(RemoveBackslashUnlessRoot(ExpandConstant('{app}')), FixedDir) <> 0 then begin
    Result := 'Use the fixed per-user application location.';
    Exit;
  end;
  if DirExists(FixedDir) then begin
    if not LoadStringFromFile(FixedDir + '\shell-owner.json', Marker) then begin
      Result := 'An existing unrecognized launcher directory was preserved.';
      Exit;
    end;
    if Trim(String(Marker)) <> '{"schema":1,"application":"Forensic DGP"}' then begin
      Result := 'The launcher ownership marker differs. Existing files were preserved.';
      Exit;
    end;
  end;
  ExtractTemporaryFile('gui_setup.ps1');
  ExtractTemporaryFile('distribution.ps1');
  ExtractTemporaryFile('app-source.zip');
  ExtractTemporaryFile('app-source.json');
  { Installed files also extracted privately for release input verification. }
  ExtractTemporaryFile('ForensicDGP.exe');
  ExtractTemporaryFile('uninstall.ps1');
  ExtractTemporaryFile('release.json');
  EventsFile := ExpandConstant('{tmp}\setup-events.jsonl');
  CancelFile := ExpandConstant('{tmp}\setup-cancel');
  SetupLog := ExpandConstant('{localappdata}\DGP-setup-{#ReleaseCommit}.log');
  LastStage := 'source verification';
  CancelRequested := False;
  DeleteFile(CancelFile);
  SaveStringToFile(EventsFile, '', False);
  Params := '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File ' + Q(ExpandConstant('{tmp}\gui_setup.ps1')) +
    ' -Stage ' + Q(ExpandConstant('{tmp}')) + ' -EventsFile ' + Q(EventsFile) +
    ' -CancelFile ' + Q(CancelFile) + ' -LogPath ' + Q(SetupLog) +
    ' -ExpectedManifestHash {#ReleaseManifestSHA256}';
  Preparing := True;
  ProgressPage.Show;
  try
    WizardForm.CancelButton.Enabled := True;
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
  if CompareText(ExpandConstant('{app}'), ExpandConstant('{localappdata}\Programs\Forensic DGP')) <> 0 then Exit;
  if GetSHA256OfFile(ExpandConstant('{app}\uninstall.ps1')) <> '{#UninstallSHA256}' then begin
    MsgBox('The uninstall helper differs. Files were preserved.', mbError, MB_OK);
    Exit;
  end;
  CheckOutput := '';
  if not ExecAndLogOutput(PS, '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File ' +
      Q(ExpandConstant('{app}\uninstall.ps1')) + ' -Action Check -ManagedRoot ' +
      Q(ExpandConstant('{localappdata}\DGP\app')), ExpandConstant('{app}'), SW_SHOWNORMAL,
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
      ExpandConstant('{localappdata}\DGP\app') + '.', mbInformation, MB_OK);
end;
