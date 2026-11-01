"""Source repair replaces only a computed owned version and rolls back on failure."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

BASE = Path(__file__).resolve().parents[1]
SCRIPT = BASE / 'implementation' / 'distribution.ps1'
PS = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
COMMIT = 'a' * 40


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


@unittest.skipUnless(PS.is_file(), 'Windows PowerShell 5 is required')
class SourceRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BASE)
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.root = self.local / 'DGP' / 'app'
        self.version = self.root / 'versions' / COMMIT
        self.version.mkdir(parents=True)
        self.files = {
            'start.bat': b'@echo off\r\nif defined DGP_TEST_VERIFY_FAIL exit /b 1\r\nexit /b 0\r\n',
            'implementation/install_source.bat': b'@echo off\r\necho replacement > "%DGP_TEST_EXECUTED%"\r\nif defined DGP_TEST_FAIL exit /b 1\r\nexit /b 0\r\n',
            'implementation/file.txt': b'pristine\n',
        }
        for relative,data in self.files.items():
            target=self.version/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        (self.version/'implementation/file.txt').write_bytes(b'tampered')
        (self.version/'implementation/install_source.bat').write_bytes(b'@echo off\r\necho old-unverified > "%DGP_TEST_EXECUTED%"\r\nexit /b 0\r\n')
        (self.root/'owner.json').write_text(json.dumps(dict(schema_version=1,repository='janusinss/forensic-dgp')),encoding='utf8')
        (self.root/'receipts').mkdir()
        self.receipt=self.root/'receipts'/(COMMIT+'.json')
        self.receipt.write_bytes(json.dumps(dict(schema=1,commit=COMMIT,files={name:sha(data) for name,data in self.files.items()}),indent=2).encode())
        self.old_receipt=self.receipt.read_bytes()
        self.pointer=self.root/'current.json'
        self.old_pointer=json.dumps(dict(schema=1,commit=COMMIT),indent=2).encode()
        self.pointer.write_bytes(self.old_pointer)
        self.evidence=self.root/'data'/'saved-record.json'
        self.evidence.parent.mkdir()
        self.evidence.write_bytes(b'{"historical_path":"C:\\\\original","record":1}\r\n')
        self.old_evidence=self.evidence.read_bytes()
        self.old_source={p.relative_to(self.version).as_posix():p.read_bytes() for p in self.version.rglob('*') if p.is_file()}
        self.archive=self.local/'source.zip'
        with zipfile.ZipFile(self.archive,'w') as archive:
            for relative,data in self.files.items():
                archive.writestr('forensic-dgp-'+COMMIT+'/'+relative,data)
        self.manifest=self.local/'bundle.json'
        self.manifest.write_text(json.dumps(dict(schema=1,commit=COMMIT,source_zip_sha256=sha(self.archive.read_bytes()),
            source_zip_size=self.archive.stat().st_size,files={name:blob(data) for name,data in self.files.items()},ancestors=[COMMIT])),encoding='utf8')
        self.marker=self.local/'executed.txt'
        source=SCRIPT.read_text(encoding='utf8')
        api_start=source.index('function Api(')
        api_end=source.index('\n',api_start)
        source=source[:api_start]+"function Api([string]$url) { throw 'Network forbidden in repair fixture.' }"+source[api_end:]
        self.fixture=self.local/'distribution-fixture.ps1'
        self.fixture.write_text(source,encoding='utf-8-sig')
        self.env={key:value for key,value in os.environ.items() if not key.startswith(('DGP_','IMPLEMENTATION_'))}
        self.env.update(LOCALAPPDATA=str(self.local),DGP_TEST_EXECUTED=str(self.marker))

    def install(self,repair=False,fail=False,bundle=True,verify_fail=False):
        env=dict(self.env)
        if fail:
            env['DGP_TEST_FAIL']='1'
        if verify_fail:
            env['DGP_TEST_VERIFY_FAIL']='1'
        arguments=[str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(self.fixture),
                   '-Action','Install','-ManagedRoot',str(self.root),'-Commit',COMMIT]
        if bundle:
            arguments += ['-BundlePath',str(self.archive),'-BundleManifest',str(self.manifest)]
        if repair:
            arguments.append('-Repair')
        return subprocess.run(arguments,capture_output=True,text=True,encoding='utf8',errors='replace',env=env,timeout=30)

    def assert_original_preserved(self):
        self.assertEqual({p.relative_to(self.version).as_posix():p.read_bytes() for p in self.version.rglob('*') if p.is_file()},self.old_source)
        self.assertEqual(self.pointer.read_bytes(),self.old_pointer)
        self.assertEqual(self.evidence.read_bytes(),self.old_evidence)

    def test_normal_install_refuses_tampered_active_source(self):
        result=self.install()
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assert_original_preserved()
        self.assertEqual(self.receipt.read_bytes(),self.old_receipt)
        self.assertFalse(self.marker.exists())

    def test_explicit_repair_publishes_verified_bundle_and_preserves_backup(self):
        result=self.install(repair=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual((self.version/'implementation/file.txt').read_bytes(),b'pristine\n')
        self.assertEqual(self.marker.read_text().strip(),'replacement')
        self.assertEqual(self.evidence.read_bytes(),self.old_evidence)
        self.assertEqual(json.loads(self.pointer.read_text(encoding='utf-8-sig'))['commit'],COMMIT)
        backups=list((self.root/'repair-backups').glob('*/source/implementation/file.txt'))
        self.assertEqual(len(backups),1)
        self.assertEqual(backups[0].read_bytes(),b'tampered')

    def test_later_failure_restores_exact_prior_source_receipt_and_pointer(self):
        result=self.install(repair=True,fail=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assert_original_preserved()
        self.assertEqual(self.receipt.read_bytes(),self.old_receipt)
        self.assertEqual(self.marker.read_text().strip(),'replacement')
        failures=list((self.root/'repair-backups').glob('*/failed-replacement/implementation/file.txt'))
        self.assertEqual(len(failures),1)
        self.assertEqual(failures[0].read_bytes(),b'pristine\n')

    def test_owned_interrupted_source_without_receipt_can_be_repaired(self):
        self.receipt.unlink()
        result=self.install(repair=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual((self.version/'implementation/file.txt').read_bytes(),b'pristine\n')
        self.assertTrue(self.receipt.is_file())

    def test_final_verification_failure_rolls_back_source_and_receipt(self):
        result=self.install(repair=True,verify_fail=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assert_original_preserved()
        self.assertEqual(self.receipt.read_bytes(),self.old_receipt)
        self.assertEqual(self.marker.read_text().strip(),'replacement')

    def test_corrupt_owned_receipt_is_backed_up_without_being_trusted(self):
        self.receipt.write_bytes(b'not-json-but-preserved')
        result=self.install(repair=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        backups=list((self.root/'repair-backups').glob('*/receipt.json'))
        self.assertEqual(len(backups),1)
        self.assertEqual(backups[0].read_bytes(),b'not-json-but-preserved')
        self.assertEqual(json.loads(self.receipt.read_text(encoding='utf-8-sig'))['commit'],COMMIT)

    def test_corrupt_bundle_cannot_replace_existing_source(self):
        self.archive.write_bytes(b'altered bundle')
        result=self.install(repair=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assert_original_preserved()
        self.assertEqual(self.receipt.read_bytes(),self.old_receipt)
        self.assertFalse((self.root/'repair-backups').exists())
        self.assertFalse(self.marker.exists())

    def test_repair_without_validated_bundle_fails_before_execution(self):
        result=self.install(repair=True,bundle=False)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertIn('bundle',(result.stderr+result.stdout).lower())
        self.assert_original_preserved()
        self.assertFalse(self.marker.exists())

    def test_unrecognized_runtime_cannot_be_repaired(self):
        (self.root/'owner.json').unlink()
        result=self.install(repair=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assert_original_preserved()
        self.assertFalse(self.marker.exists())


if __name__ == '__main__':
    unittest.main()
