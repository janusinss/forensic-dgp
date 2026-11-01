"""Actual installer activation with tiny versioned models and no live app writes.

The PowerShell distribution transaction is real. Its package/bootstrap entry
points are isolated fixture scripts which exercise the production model store.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from tests import test_distribution_repair as repair_fixture

BASE = Path(__file__).resolve().parents[1]
A = 'a' * 40
B = 'b' * 40
OLD = {'implementation': b'old-restoration-weights', 'completion': b'old-completion-weights'}
NEW = dict(OLD, completion=b'new-completion-weights')
DRIVER = r"""import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from implementation.managed_paths import data_root
LAYOUT=__LAYOUT__
CONTENTS=__CONTENTS__
DATA=data_root(ROOT)
phase=sys.argv[1]
current=json.loads((DATA.parent/'current.json').read_text(encoding='utf-8-sig'))['commit']
with (DATA/'fixture-events.jsonl').open('a',encoding='utf-8') as log:
    log.write(json.dumps({'source':ROOT.name,'phase':phase,'active_before':current})+'\n')
for family,text in CONTENTS.items():
    expected=text.encode('ascii')
    if LAYOUT=='namespaced':
        from implementation.model_store import model_directory,prepare_model_directory
        folder=prepare_model_directory(ROOT,family) if phase=='install' else model_directory(ROOT,family)
    else:
        folder=DATA/'models'/family
    path=folder/('face-'+family+'.bin')
    if phase=='install':
        if family=='completion' and os.environ.get('DGP_TEST_INSTALL_FAIL'):
            (folder/'.tiny-model.failed.partial').write_bytes(b'incomplete')
            raise SystemExit(17)
        if path.exists():
            if path.read_bytes()!=expected:raise RuntimeError('Existing model changed')
        else:path.write_bytes(expected)
    if not path.is_file() or path.read_bytes()!=expected:raise RuntimeError('Model verification failed')
if phase=='verify' and os.environ.get('DGP_TEST_VERIFY_FAIL'):raise SystemExit(23)
"""


def source_files(layout, contents):
    driver=DRIVER.replace('__LAYOUT__',repr(layout)).replace('__CONTENTS__',repr({k:v.decode('ascii') for k,v in contents.items()}))
    files={
        'implementation/__init__.py': b'',
        'implementation/managed_paths.py': (BASE/'implementation/managed_paths.py').read_bytes(),
        'model-update-fixture.py': driver.encode('utf-8'),
        'start.bat': b'@echo off\r\n"%DGP_TEST_PYTHON%" -I -B "%~dp0model-update-fixture.py" verify\r\nexit /b %errorlevel%\r\n',
        'implementation/install_source.bat': b'@echo off\r\n"%DGP_TEST_PYTHON%" -I -B "%~dp0..\\model-update-fixture.py" install\r\nexit /b %errorlevel%\r\n',
    }
    if layout=='namespaced':
        files['implementation/model_store.py']=(BASE/'implementation/model_store.py').read_bytes()
    for family,content in contents.items():
        name='models.json' if family=='implementation' else 'completion_assets.json'
        manifest={'schema_version':1,'assets':{'face':{'filename':'face-'+family+'.bin',
                  'size':len(content),'sha256':repair_fixture.sha(content)}}}
        files['implementation/'+name]=json.dumps(manifest).encode('utf-8')
    return files


@unittest.skipUnless(repair_fixture.PS.is_file(),'Windows PowerShell required')
class ModelUpdateTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='model-update-',dir=BASE/'scratch')
        self.addCleanup(self.temp.cleanup)
        self.local=Path(self.temp.name)
        self.root=self.local/'DGP/app'
        self.old=self.root/'versions'/A
        self.new=self.root/'versions'/B
        self.old.mkdir(parents=True)
        (self.root/'receipts').mkdir()
        (self.root/'data').mkdir()
        self.pointer=self.root/'current.json'
        self.pointer.write_bytes(b'{"schema":1,"commit":"'+A.encode()+b'"}\n')
        self.original_pointer=self.pointer.read_bytes()
        (self.root/'owner.json').write_text(json.dumps({'schema_version':1,'repository':'janusinss/forensic-dgp'}))
        self.record=self.root/'data/saved-record.json'
        self.record.write_bytes(b'{"source":"historical","original_path":"C:\\\\original.png"}\r\n')
        self.original_record=self.record.read_bytes()
        self.env={k:v for k,v in os.environ.items() if not k.startswith(('DGP_','IMPLEMENTATION_'))}
        self.env.update(LOCALAPPDATA=str(self.local),DGP_TEST_PYTHON=sys.executable)
        source=(BASE/'implementation/distribution.ps1').read_text(encoding='utf-8-sig')
        start=source.index('function Api('); end=source.index('\n',start)
        source=source[:start]+"function Api([string]$url) { throw 'Network forbidden in model-update fixture.' }"+source[end:]
        self.fixture=self.local/'distribution-fixture.ps1'
        self.fixture.write_text(source,encoding='utf-8-sig')
        self.archive=self.local/'source.zip'
        self.manifest=self.local/'source.json'

    def prepare(self,layout):
        self.old_files=source_files(layout,OLD)
        for name,data in self.old_files.items():
            path=self.old/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        receipt={'schema':1,'commit':A,'files':{name:repair_fixture.sha(data) for name,data in self.old_files.items()}}
        (self.root/'receipts'/(A+'.json')).write_text(json.dumps(receipt))
        self.old_receipt=(self.root/'receipts'/(A+'.json')).read_bytes()
        # Fixture setup uses the same real model-store preparation code as the
        # candidate, but executes only against this tiny temporary installation.
        if layout=='legacy':
            for family,content in OLD.items():
                folder=self.root/'data/models'/family;folder.mkdir(parents=True)
                (folder/('face-'+family+'.bin')).write_bytes(content)
        else:
            run=self.run_driver(self.old,'install')
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.before_models=self.model_snapshot()
        self.candidate_files=source_files('namespaced',NEW)
        with zipfile.ZipFile(self.archive,'w') as archive:
            for name,data in self.candidate_files.items():archive.writestr('forensic-dgp-'+B+'/'+name,data)
        manifest={'schema':1,'commit':B,'files':{name:repair_fixture.blob(data) for name,data in self.candidate_files.items()},
                  'source_zip_sha256':repair_fixture.sha(self.archive.read_bytes()),'source_zip_size':self.archive.stat().st_size,
                  'ancestors':[B,A]}
        self.manifest.write_text(json.dumps(manifest))

    def run_driver(self,root,phase):
        env=dict(self.env,DGP_MANAGED_ROOT=str(self.root))
        return subprocess.run([sys.executable,'-I','-B',str(root/'model-update-fixture.py'),phase],
                              capture_output=True,text=True,env=env,timeout=15)

    def model_snapshot(self):
        folder=self.root/'data/models'
        return {p.relative_to(folder).as_posix():p.read_bytes() for p in folder.rglob('*') if p.is_file()}

    def install(self,fail_install=False,fail_verify=False):
        env=dict(self.env)
        if fail_install:env['DGP_TEST_INSTALL_FAIL']='1'
        if fail_verify:env['DGP_TEST_VERIFY_FAIL']='1'
        arguments=[str(repair_fixture.PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(self.fixture),
                   '-Action','Install','-ManagedRoot',str(self.root),'-Commit',B,
                   '-BundlePath',str(self.archive),'-BundleManifest',str(self.manifest)]
        return subprocess.run(arguments,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=30)

    def assert_old_preserved(self):
        self.assertEqual(self.record.read_bytes(),self.original_record)
        self.assertEqual((self.root/'receipts'/(A+'.json')).read_bytes(),self.old_receipt)
        self.assertEqual({p.relative_to(self.old).as_posix():p.read_bytes() for p in self.old.rglob('*') if p.is_file()},self.old_files)
        models=self.model_snapshot()
        for name,data in self.before_models.items():self.assertEqual(models[name],data)
        old_verify=self.run_driver(self.old,'verify')
        self.assertEqual(old_verify.returncode,0,old_verify.stdout+old_verify.stderr)

    def test_failed_candidate_install_keeps_legacy_release_and_records(self):
        self.prepare('legacy')
        result=self.install(fail_install=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertEqual(self.pointer.read_bytes(),self.original_pointer)
        self.assertTrue(any(p.name=='.tiny-model.failed.partial' for p in (self.root/'data/models').rglob('*')))
        self.assert_old_preserved()

    def test_failed_final_verification_keeps_namespaced_active_release(self):
        self.prepare('namespaced')
        result=self.install(fail_verify=True)
        self.assertNotEqual(result.returncode,0,result.stdout)
        self.assertEqual(self.pointer.read_bytes(),self.original_pointer)
        self.assertTrue(self.new.is_dir())
        self.assert_old_preserved()

    def test_success_activates_only_after_verification_and_reuses_unchanged_family(self):
        self.prepare('namespaced')
        result=self.install()
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads(self.pointer.read_text(encoding='utf-8-sig'))['commit'],B)
        self.assert_old_preserved()
        events=[json.loads(line) for line in (self.root/'data/fixture-events.jsonl').read_text().splitlines()]
        candidate_checks=[event for event in events if event['source']==B and event['phase']=='verify']
        self.assertTrue(candidate_checks,'Candidate must pass its actual verification entry point')
        self.assertTrue(all(event['active_before']==A for event in candidate_checks))
        families=self.root/'data/models'
        self.assertEqual(len(list((families/'implementation').glob('*/.model-set.json'))),1)
        self.assertEqual(len(list((families/'completion').glob('*/.model-set.json'))),2)
        self.assertEqual(sum(p.stat().st_size for p in families.rglob('*.bin')),len(OLD['implementation'])+len(OLD['completion'])+len(NEW['completion']))


if __name__=='__main__':unittest.main()
