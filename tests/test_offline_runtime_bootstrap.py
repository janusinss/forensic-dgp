"""Exercise only the offline archive selector, never download/extract Python."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.name=='nt','Windows Python bootstrap')
class OfflineBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='offline bootstrap '); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); (self.root/'python').mkdir()
        self.data=b'harmless archive fixture'
        (self.root/'python/python-3.13.9-amd64.zip').write_bytes(self.data)
        self.manifest=dict(schema=1,platform='win_amd64',python=dict(version='3.13.9',path='python/python-3.13.9-amd64.zip',bytes=len(self.data),sha256=hashlib.sha256(self.data).hexdigest()))
        content=json.dumps(self.manifest).encode(); (self.root/'manifest.json').write_bytes(content)
        self.env=os.environ.copy(); self.env.update(DGP_OFFLINE_PAYLOAD=str(self.root),DGP_OFFLINE_PAYLOAD_SHA256=hashlib.sha256(content).hexdigest(),FIXTURE_PYTHON_BYTES=str(len(self.data)),FIXTURE_PYTHON_SHA256=hashlib.sha256(self.data).hexdigest(),FIXTURE_BOOTSTRAP_SOURCE=str(ROOT/'implementation/bootstrap_python.ps1'))

    def run_selector(self):
        script="""$ErrorActionPreference='Stop'; $tokens=$null; $errors=$null; $ast=[Management.Automation.Language.Parser]::ParseFile($env:FIXTURE_BOOTSTRAP_SOURCE,[ref]$tokens,[ref]$errors); if ($errors.Count) { throw 'Parse failure' }; $names=@('Assert-OrdinaryDirectory','Offline-PythonHash','Offline-PythonArchive'); foreach ($name in $names) { $function=$ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true); if ($function.Count -ne 1) { throw 'Missing named selector dependency' }; . ([scriptblock]::Create($function[0].Extent.Text)) }; $expectedBytes=[long]$env:FIXTURE_PYTHON_BYTES; $expectedSha256=$env:FIXTURE_PYTHON_SHA256; Offline-PythonArchive"""
        return subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'),'-NoProfile','-NonInteractive','-Command',script],env=self.env,capture_output=True,text=True,timeout=20)

    def test_missing_private_python_reaches_separate_offline_rebuild(self):
        source=(ROOT/'implementation/bootstrap_python.ps1').read_text()
        start=source.index('        $info=$null')
        end=source.index("        $replaceExisting=$true",start)+len('        $replaceExisting=$true')
        self.env.update(FIXTURE_MISSING_EXE=str(self.root/'missing.exe'),FIXTURE_EXISTING_BLOCK=source[start:end])
        # Execute the exact production existing-runtime decision block with a
        # missing executable; no archive extraction or installation occurs.
        script="""$ErrorActionPreference='Stop'; $existing=$env:FIXTURE_MISSING_EXE; $VerifyOnly=$false; $offlineArchive='validated fixture'; $replaceExisting=$false; . ([scriptblock]::Create($env:FIXTURE_EXISTING_BLOCK)); if (!$replaceExisting) { throw 'Missing Python did not request separate offline replacement' }; 'replacement requested'"""
        result=subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'),'-NoProfile','-NonInteractive','-Command',script],env=self.env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('replacement requested',result.stdout)

    def test_verified_local_archive_is_selected(self):
        result=self.run_selector(); self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(Path(result.stdout.strip()),self.root/'python/python-3.13.9-amd64.zip')

    def test_bad_manifest_hash_fails_without_fallback(self):
        self.env['DGP_OFFLINE_PAYLOAD_SHA256']='0'*64
        result=self.run_selector(); self.assertNotEqual(result.returncode,0); self.assertIn('pinned SHA-256',result.stderr)

    def test_incomplete_offline_request_fails_without_fallback(self):
        self.env.pop('DGP_OFFLINE_PAYLOAD_SHA256')
        result=self.run_selector(); self.assertNotEqual(result.returncode,0); self.assertIn('No online fallback',result.stderr)

@unittest.skipUnless(os.name=='nt','Windows private Python publication')
class OfflinePythonPublishTests(unittest.TestCase):
    def run_publish(self, *, missing=False):
        with tempfile.TemporaryDirectory(prefix='offline Python publish ') as temporary:
            root=Path(temporary); base=root/'python'; base.mkdir()
            target=base/'python-3.13.9-amd64'; target.mkdir(); (target/'old.txt').write_bytes(b'old runtime')
            staged=base/'staging'/'python'
            if not missing:
                staged.mkdir(parents=True); (staged/'new.txt').write_bytes(b'new verified runtime')
            env=dict(os.environ,FIXTURE_BASE=str(base),FIXTURE_STAGED=str(staged),FIXTURE_TARGET=str(target),FIXTURE_BOOTSTRAP_SOURCE=str(ROOT/'implementation/bootstrap_python.ps1'))
            script="""$ErrorActionPreference='Stop'; $tokens=$null; $errors=$null; $ast=[Management.Automation.Language.Parser]::ParseFile($env:FIXTURE_BOOTSTRAP_SOURCE,[ref]$tokens,[ref]$errors); foreach ($name in @('Assert-OrdinaryDirectory','Publish-PythonRuntime')) { $function=$ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name},$true); if ($function.Count -ne 1) { throw 'Missing named publish dependency' }; . ([scriptblock]::Create($function[0].Extent.Text)) }; Publish-PythonRuntime $env:FIXTURE_STAGED $env:FIXTURE_TARGET $env:FIXTURE_BASE $true"""
            result=subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'),'-NoProfile','-NonInteractive','-Command',script],env=env,capture_output=True,text=True,timeout=20)
            if missing:
                self.assertNotEqual(result.returncode,0)
                self.assertEqual((target/'old.txt').read_bytes(),b'old runtime')
            else:
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual((target/'new.txt').read_bytes(),b'new verified runtime')
                backups=list(base.glob('python-3.13.9-amd64.previous-*'))
                self.assertEqual(len(backups),1)
                self.assertEqual((backups[0]/'old.txt').read_bytes(),b'old runtime')

    def test_repair_preserves_old_python_as_named_backup(self): self.run_publish()
    def test_missing_candidate_preserves_current_python(self): self.run_publish(missing=True)

if __name__=='__main__': unittest.main()
