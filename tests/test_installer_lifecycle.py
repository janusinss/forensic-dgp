"""Destination preflight behavior, without altering installed application folders."""
import json
import hashlib
import zipfile
import os
from pathlib import Path
import shutil
import re
import subprocess
import tempfile
import unittest
BASE=Path(__file__).resolve().parents[1]
PS=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
LOCATION=BASE/'packaging/install_location.ps1'

@unittest.skipUnless(PS.is_file(),'Windows PowerShell required')
class DestinationPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='preflight-',dir=BASE/'scratch')
        self.addCleanup(self.temp.cleanup)
        self.area=Path(self.temp.name)
        self.target=self.area/'chosen'
    def invoke(self,required=0):
        return subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(LOCATION),
            '-Action','Preflight','-InstallRoot',str(self.target),'-RequiredBytes',str(required)],capture_output=True,text=True,encoding='utf8',timeout=15)
    def test_preflight_returns_real_destination_without_creating_installation(self):
        before=set(self.area.iterdir())
        result=self.invoke(1)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        body=json.loads(result.stdout)
        self.assertEqual(Path(body['physical_install_root']),self.target)
        self.assertGreater(body['available_bytes'],0)
        self.assertEqual(set(self.area.iterdir()),before)
    def test_low_disk_is_rejected_without_consuming_space(self):
        free=shutil.disk_usage(self.area).free
        result=self.invoke(free+1024**3)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('space',result.stderr.lower())
        self.assertFalse(self.target.exists())
    def test_unknown_nonempty_destination_remains_untouched(self):
        self.target.mkdir(); sentinel=self.target/'unrelated.txt'; sentinel.write_bytes(b'preserve')
        result=self.invoke()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(sentinel.read_bytes(),b'preserve')
        self.assertEqual([p.name for p in self.target.iterdir()],['unrelated.txt'])

@unittest.skipUnless(PS.is_file(),'Windows PowerShell required')
class SetupWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='gui-life-',dir=BASE/'scratch')
        self.addCleanup(self.temp.cleanup)
        self.area=Path(self.temp.name)
        self.stage=self.area/'stage'; self.stage.mkdir()
        self.target=self.area/'target'
        self.log=self.area/'setup.log'
        for name in ('gui_setup.ps1','install_location.ps1','install_state.ps1'):
            shutil.copyfile(BASE/'packaging'/name,self.stage/name)
        for name in ('ForensicDGP.exe','uninstall.ps1'):
            (self.stage/name).write_bytes(b'inert embedded fixture')
        (self.stage/'app-source.zip').write_bytes(b'inert source bundle')
        (self.stage/'app-source.json').write_text(json.dumps({'schema':1,'commit':'b'*40,'ancestors':['b'*40],
             'files':{'app.py':'0'*40}}))
        (self.stage/'distribution.ps1').write_text(r"""param([string]$Action,[string]$ManagedRoot,[string]$Commit,[string]$BundlePath,[string]$BundleManifest,[switch]$Repair)
$ErrorActionPreference='Stop'
[IO.Directory]::CreateDirectory($ManagedRoot)|Out-Null
@{schema_version=2;repository='janusinss/forensic-dgp';layout='desktop-runtime-v1';install_root=$env:DGP_INSTALL_ROOT}|ConvertTo-Json -Compress|Set-Content -LiteralPath (Join-Path $ManagedRoot 'owner.json') -Encoding UTF8
@{payload=$env:DGP_OFFLINE_PAYLOAD;sha256=$env:DGP_OFFLINE_PAYLOAD_SHA256;repair=[bool]$Repair;install_root=$env:DGP_INSTALL_ROOT}|ConvertTo-Json -Compress|Set-Content -LiteralPath (Join-Path $ManagedRoot 'probe.json') -Encoding UTF8
exit 0
""")
        payload=self.stage/'payload'; (payload/'python').mkdir(parents=True)
        with zipfile.ZipFile(payload/'python/python-3.13.9-amd64.zip','w') as archive:
            archive.writestr('python.exe',b'inert fixture')
        python_zip=payload/'python/python-3.13.9-amd64.zip'
        (payload/'manifest.json').write_text(json.dumps({'schema':1,'platform':'win_amd64','files':[{'path':'python/python-3.13.9-amd64.zip','bytes':python_zip.stat().st_size,'sha256':hashlib.sha256(python_zip.read_bytes()).hexdigest()}],'models':[]}))
        files={name:hashlib.sha256((self.stage/name).read_bytes()).hexdigest() for name in
               ('app-source.zip','app-source.json','distribution.ps1','uninstall.ps1','gui_setup.ps1','ForensicDGP.exe','install_location.ps1','install_state.ps1')}
        files['offline-payload-manifest.json']=hashlib.sha256((payload/'manifest.json').read_bytes()).hexdigest()
        self.manifest={'schema':1,'version':'0.2.0','commit':'b'*40,'payload':'offline-bundled',
                       'payload_expanded_bytes':1,'files':files}
        (self.stage/'release.json').write_text(json.dumps(self.manifest))
        self.manifest_hash=hashlib.sha256((self.stage/'release.json').read_bytes()).hexdigest()
    def run_setup(self):
        return subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(self.stage/'gui_setup.ps1'),
           '-Stage',str(self.stage),'-InstallRoot',str(self.target),'-EventsFile',str(self.area/'events.jsonl'),
           '-CancelFile',str(self.area/'cancel'),'-LogPath',str(self.log),'-ExpectedManifestHash',self.manifest_hash,
           '-PreviousRoot',str(self.area/'missing')],
           capture_output=True,text=True,encoding='utf8',timeout=25,
           env=dict(os.environ,DGP_OFFLINE_PAYLOAD='untrusted-parent-path',DGP_OFFLINE_PAYLOAD_SHA256='untrusted-hash'))
    def test_offline_wrapper_pins_payload_and_regenerates_full_root_binding(self):
        result=self.run_setup()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        probe=json.loads((self.target/'runtime/probe.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(probe['payload'],str(self.stage/'payload'))
        self.assertEqual(probe['sha256'],self.manifest['files']['offline-payload-manifest.json'])
        config=json.loads((self.target/'install-location.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(config,{'schema':1,'install_root':str(self.target),'managed_root':str(self.target/'runtime')})
    def test_changed_embedded_payload_fails_before_destination_is_created(self):
        (self.stage/'payload/manifest.json').write_text('changed')
        result=self.run_setup()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.target.exists())
    def test_long_source_path_is_rejected_before_destination_changes(self):
        source=self.stage/'app-source.json'
        value=json.loads(source.read_text()); value['files']={'nested/'+'x'*210+'.py':'0'*40}
        source.write_text(json.dumps(value))
        self.manifest['files']['app-source.json']=hashlib.sha256(source.read_bytes()).hexdigest()
        (self.stage/'release.json').write_text(json.dumps(self.manifest))
        self.manifest_hash=hashlib.sha256((self.stage/'release.json').read_bytes()).hexdigest()
        result=self.run_setup()
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.target.exists())

@unittest.skipUnless((BASE/'scratch/tools/inno-7.1.1/ISCC.exe').is_file(),'Verified Inno compiler required')
class NativeCopyProgressTests(unittest.TestCase):
    def test_interleaved_copy_progress_displays_one_transfer_until_it_finishes(self):
        with tempfile.TemporaryDirectory(prefix='copy-progress-',dir=BASE/'scratch') as temporary:
            area=Path(temporary)
            source=(BASE/'packaging/installer.iss').read_text(encoding='utf8')
            json_function=source.split('function JsonField(',1)[1].split('function RegisteredInstallRoot',1)[0]
            progress_functions=source.split('function PhaseNumber(',1)[1].split('procedure ToggleDetails',1)[0]
            setup=('[Setup]\nAppId=ForensicDGP-CopyProgress-Fixture\nAppName=Copy progress fixture\nAppVersion=1.0.0\n'
                   'DefaultDirName={tmp}\nCreateAppDir=no\nUninstallable=no\nPrivilegesRequired=lowest\n'
                   'DisableWelcomePage=yes\nDisableDirPage=yes\nDisableProgramGroupPage=yes\n'
                   'OutputDir='+str(area)+'\nOutputBaseFilename=progress-fixture\n')
            declarations=('[Code]\nvar\nProgressPage: TOutputProgressWizardPage;\n'
                'StepLabel, TransferLabel: TNewStaticText;\n'
                'LastStage, ActiveDownloadStage: String;\nStepNumber: Integer;\nCancelRequested: Boolean;\n')
            initialize="""procedure InitializeWizard;
begin
  ProgressPage := CreateOutputProgressPage('Fixture', 'Copying');
  StepLabel := TNewStaticText.Create(WizardForm); StepLabel.Parent := ProgressPage.Surface;
  TransferLabel := TNewStaticText.Create(WizardForm); TransferLabel.Parent := ProgressPage.Surface;
  StepNumber := 1;
"""
            events=[
                ('copy:A','copying',0),('copy:B','copying',5),('copy:A','copying',50),
                ('copy:B','copying',10),('capture-runtime','running',None),('copy:A','copying',100),
                ('copy:A','complete',100),('copy:B','copying',50),('copy:B','copying',100),
                ('copy:B','complete',100),('copy:C','copying',40),('copy:D','copying',100),
                ('copy:D','complete',100),('copy:C','copying',100),('copy:C','complete',100)]
            for stage,status,current in events:
                row={'schema':1,'stage':stage,'status':status,'message':stage}
                if current is not None: row.update(current=current,total=100)
                initialize+="  UpdateProgress('"+json.dumps(row,separators=(',',':'))+"');\n"
            initialize+="end;\n"
            (area/'progress.iss').write_text(setup+declarations+'function JsonField('+json_function+
                'function PhaseNumber('+progress_functions+initialize,encoding='utf8')
            built=subprocess.run([str(BASE/'scratch/tools/inno-7.1.1/ISCC.exe'),str(area/'progress.iss')],
                capture_output=True,text=True,timeout=25)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            log=area/'progress.log'
            executed=subprocess.run([str(area/'progress-fixture.exe'),'/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART',
                '/LOG='+str(log)],capture_output=True,text=True,timeout=25)
            self.assertEqual(executed.returncode,0,executed.stdout+executed.stderr)
            body=log.read_text(encoding='utf-8-sig')
            selected=[(int(value),stage) for value,stage in re.findall(r'DGP_PROGRESS copying (\d+) (copy:[^\r\n ]+)',body)]
            self.assertEqual(selected,[(0,'copy:A'),(50,'copy:A'),(100,'copy:A'),
                (50,'copy:B'),(100,'copy:B'),(40,'copy:C'),(100,'copy:C')])
            self.assertIn('DGP_PROGRESS activity copy:C',body)
            self.assertNotIn('DGP_PROGRESS activity copy:D',body)

if __name__=='__main__': unittest.main()
