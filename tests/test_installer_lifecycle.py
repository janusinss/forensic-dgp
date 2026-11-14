"""Destination preflight behavior, without altering installed application folders."""
import json
import ctypes
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
COMPILER=Path(os.environ.get('DGP_INNO_COMPILER',BASE/'scratch/tools/inno-7.1.1/ISCC.exe'))

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
        (BASE/'scratch').mkdir(exist_ok=True)
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
    def run_setup(self, stage=None):
        stage=str(self.stage) if stage is None else str(stage)
        return subprocess.run([str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(self.stage/'gui_setup.ps1'),
           '-Stage',stage,'-InstallRoot',str(self.target),'-EventsFile',str(self.area/'events.jsonl'),
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
    def assert_offline_setup_succeeded(self, result, context=''):
        self.assertEqual(result.returncode,0,context+result.stdout+result.stderr)
        probe=json.loads((self.target/'runtime/probe.json').read_text(encoding='utf-8-sig'))
        self.assertTrue(Path(probe['payload']).samefile(self.stage/'payload'))
        self.assertEqual(probe['sha256'],self.manifest['files']['offline-payload-manifest.json'])
        self.assertTrue((self.target/'install-location.json').is_file())
    def test_short_83_stage_alias_accepts_the_exact_payload_inventory(self):
        spaced=self.area/'payload stage with spaces'
        self.stage.rename(spaced); self.stage=spaced
        get_short=ctypes.WinDLL('kernel32',use_last_error=True).GetShortPathNameW
        get_short.argtypes=(ctypes.c_wchar_p,ctypes.c_wchar_p,ctypes.c_uint32)
        get_short.restype=ctypes.c_uint32
        required=get_short(str(self.stage),None,0)
        self.assertGreater(required,0,'GetShortPathNameW failed for the existing fixture stage')
        output=ctypes.create_unicode_buffer(required)
        self.assertGreater(get_short(str(self.stage),output,required),0)
        alias=output.value
        if os.path.normcase(alias)==os.path.normcase(str(self.stage)):
            self.skipTest('This filesystem does not provide an 8.3 alias for the fixture stage')
        self.assertTrue(Path(alias).samefile(self.stage))
        result=self.run_setup(alias)
        self.assert_offline_setup_succeeded(result,'8.3 stage alias: '+alias+'; expanded stage: '+str(self.stage)+'\n')
    def test_spaced_unicode_stage_accepts_the_exact_payload_inventory(self):
        spaced=self.area/'payload stage Łódź'
        self.stage.rename(spaced); self.stage=spaced
        self.assert_offline_setup_succeeded(self.run_setup())
    def test_stage_with_trailing_separator_accepts_the_exact_payload_inventory(self):
        self.assert_offline_setup_succeeded(self.run_setup(str(self.stage)+'\\'))
    def test_mixed_case_stage_accepts_the_exact_payload_inventory(self):
        self.assert_offline_setup_succeeded(self.run_setup(str(self.stage).swapcase()))
    def test_unlisted_payload_file_is_named_and_rejected_before_destination_creation(self):
        extra=self.stage/'payload/python/unlisted-extra.bin'
        extra.write_bytes(b'not in the pinned payload inventory')
        result=self.run_setup()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Unexpected offline payload file',' '.join(result.stderr.split()))
        self.assertIn('python/unlisted-extra.bin',result.stderr)
        self.assertEqual(extra.read_bytes(),b'not in the pinned payload inventory')
        self.assertFalse(self.target.exists())
    def test_tampered_payload_file_is_rejected_before_destination_creation(self):
        payload=self.stage/'payload/python/python-3.13.9-amd64.zip'
        payload.write_bytes(b'changed payload')
        result=self.run_setup()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Offline payload file differs: python/python-3.13.9-amd64.zip',' '.join(result.stderr.split()))
        self.assertFalse(self.target.exists())
    def test_missing_payload_file_is_rejected_before_destination_creation(self):
        (self.stage/'payload/python/python-3.13.9-amd64.zip').unlink()
        result=self.run_setup()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('python-3.13.9-amd64.zip',result.stderr)
        self.assertFalse(self.target.exists())
    def test_linked_payload_directory_is_rejected_without_traversing_it(self):
        outside=self.area/'outside'; outside.mkdir()
        sentinel=outside/'keep.bin'; sentinel.write_bytes(b'outside payload')
        link=self.stage/'payload/linked-directory'
        quote=lambda value: "'"+str(value).replace("'","''")+"'"
        created=subprocess.run([str(PS),'-NoProfile','-NonInteractive','-Command',
            "$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+quote(link)+
            " -Target "+quote(outside)+" | Out-Null"],capture_output=True,text=True,encoding='utf8',timeout=15)
        self.assertEqual(created.returncode,0,created.stdout+created.stderr)
        result=self.run_setup()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Offline payload paths cannot contain links',' '.join(result.stderr.split()))
        self.assertEqual(sentinel.read_bytes(),b'outside payload')
        self.assertFalse(self.target.exists())
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

@unittest.skipUnless(COMPILER.is_file(),'Verified Inno compiler required')
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
            built=subprocess.run([str(COMPILER),str(area/'progress.iss')],
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
