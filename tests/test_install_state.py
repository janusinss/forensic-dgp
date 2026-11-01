"""Behavioral Windows installer-state and evidence-preserving relocation tests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil
import unittest

BASE=Path(__file__).resolve().parents[1]
PS=Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
HELPER=BASE/'packaging/install_state.ps1'
A='a'*40
B='b'*40

def sha(data): return hashlib.sha256(data).hexdigest()

@unittest.skipUnless(PS.is_file(),'Windows PowerShell required')
class InstallStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime_temp=tempfile.TemporaryDirectory(prefix='life-probe-',dir=BASE/'scratch')
        folder=Path(cls.runtime_temp.name)
        code=folder/'Probe.cs'
        code.write_text('internal static class Probe { public static int Main(string[] args) { return 0; } }')
        cls.probe=folder/'probe.exe'
        compiler=Path(os.environ['SystemRoot'])/'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
        made=subprocess.run([str(compiler),'/nologo','/target:exe','/out:'+str(cls.probe),str(code)],capture_output=True,text=True)
        if made.returncode: raise RuntimeError(made.stdout+made.stderr)
    @classmethod
    def tearDownClass(cls):
        cls.runtime_temp.cleanup()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='life-',dir=BASE/'scratch')
        self.addCleanup(self.temp.cleanup)
        self.area=Path(self.temp.name)
        self.old=self.area/'old'
        self.new=self.area/'new'
        self.env=dict(os.environ,LOCALAPPDATA=str(self.area/'local'))
        self.source_manifest=self.area/'candidate.json'
        self.source_manifest.write_text(json.dumps({'schema':1,'commit':B,'ancestors':[B,A]}))

    def call(self,action='Inspect',root=None,previous=None,version='0.2.0',commit=B,payload=None):
        cmd=[str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(HELPER),
             '-Action',action,'-InstallRoot',str(root or self.new),'-RegisteredRoot',str(previous or self.old),
             '-ReleaseVersion',version,'-ReleaseCommit',commit,'-SourceManifest',str(self.source_manifest)]
        if payload is not None:
            cmd.extend(['-PayloadManifest',str(payload)])
        if action in ('CompleteMove','Cleanup'):
            env=dict(self.env,DGP_FIXTURE_HELPER=str(HELPER),DGP_FIXTURE_ROOT=str(root or self.new),
                     DGP_FIXTURE_OLD=str(previous or self.old),DGP_FIXTURE_ACTION=action,
                     DGP_FIXTURE_VERSION=version,DGP_FIXTURE_COMMIT=commit,DGP_FIXTURE_SOURCE=str(self.source_manifest))
            command=(". $env:DGP_FIXTURE_HELPER -LibraryOnly -InstallRoot $env:DGP_FIXTURE_ROOT -RegisteredRoot $env:DGP_FIXTURE_OLD "
                     "-ReleaseVersion $env:DGP_FIXTURE_VERSION -ReleaseCommit $env:DGP_FIXTURE_COMMIT -SourceManifest $env:DGP_FIXTURE_SOURCE -Action $env:DGP_FIXTURE_ACTION; "
                     "function Registered-Location { return $env:DGP_FIXTURE_ROOT }; Invoke-Lifecycle")
            cmd=[str(PS),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-Command',command]
        else: env=self.env
        return subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',env=env,timeout=25)

    def result(self,*args,**kwargs):
        result=self.call(*args,**kwargs)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return json.loads(result.stdout)

    def install(self,root,version='0.1.1',commit=A):
        root.mkdir(parents=True,exist_ok=True)
        runtime=root/'runtime'; runtime.mkdir(exist_ok=True)
        def record(where,value): where.write_text(json.dumps(value),encoding='utf-8')
        record(root/'shell-owner.json',{'schema':1,'application':'Forensic DGP'})
        record(root/'install-location.json',{'schema':1,'install_root':str(root),'managed_root':str(runtime)})
        record(runtime/'owner.json',{'schema_version':2,'repository':'janusinss/forensic-dgp','layout':'desktop-runtime-v1','install_root':str(root)})
        executable=b'inert fixture launcher'
        (root/'ForensicDGP.exe').write_bytes(executable)
        record(root/'release.json',{'schema':1,'version':version,'commit':commit,'files':{'ForensicDGP.exe':sha(executable)}})
        record(runtime/'current.json',{'schema':1,'commit':commit})
        version_dir=runtime/'versions'/commit; version_dir.mkdir(parents=True,exist_ok=True)
        (version_dir/'app.txt').write_bytes(b'verified source fixture')
        (version_dir/'helper.txt').write_bytes(b'second verified source file')
        private=runtime/'python/python-3.13.9-amd64'; private.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(self.probe,private/'python.exe')
        config_dir=runtime/'data/installations'; config_dir.mkdir(parents=True,exist_ok=True)
        record(config_dir/(commit+'.json'),{'schema_version':1,'checkout_root':str(version_dir),
             'environment_root':str(runtime/'environments/fixture'),'python':str(runtime/'environments/fixture/Scripts/python.exe')})
        receipts=runtime/'receipts'; receipts.mkdir(exist_ok=True)
        record(receipts/(commit+'.json'),{'schema':1,'commit':commit,'files':{'app.txt':sha(b'verified source fixture'),'helper.txt':sha(b'second verified source file')}})

    def saved(self):
        path=self.old/'runtime/data/scratch/capture_requests/evidence.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        body=b'{"input_path":"C:\\historical\\original.png","audit":"unchanged"}\r\n'
        path.write_bytes(body)
        return path,body

    def test_stale_registration_is_fresh_and_inspection_is_read_only(self):
        data=self.result()
        self.assertEqual(data['mode'],'fresh')
        self.assertTrue(data['registration_stale'])
        self.assertFalse(self.new.exists())
        self.assertFalse(self.old.exists())

    def test_same_healthy_version_offers_open_or_repair(self):
        self.install(self.old,'0.2.0',B)
        data=self.result(root=self.old)
        self.assertEqual(data['mode'],'current')
        self.assertTrue(data['healthy'])

    def test_old_version_is_update_and_damaged_version_is_repair(self):
        self.install(self.old)
        self.assertEqual(self.result(root=self.old)['mode'],'update')
        (self.old/'runtime/versions'/A/'app.txt').write_bytes(b'corrupt')
        self.assertEqual(self.result(root=self.old)['mode'],'repair')

    def test_multi_file_source_receipt_is_healthy_without_count_wrapping(self):
        self.install(self.old,'0.2.0',B)
        data=self.result(root=self.old)
        self.assertEqual(data['mode'],'current')
        self.assertTrue(data['healthy'])

    def test_unexpected_extra_source_file_still_blocks_health(self):
        self.install(self.old,'0.2.0',B)
        extra=self.old/'runtime/versions'/B/'unexpected.txt'
        extra.write_bytes(b'unrecognized source bytes')
        data=self.result(root=self.old)
        self.assertEqual(data['mode'],'repair')
        self.assertIn('Unexpected installed source file',data['reason'])
        self.assertEqual(extra.read_bytes(),b'unrecognized source bytes')

    def test_tampered_launcher_never_claims_request_close_is_verified(self):
        self.install(self.old,'0.2.0',B)
        (self.old/'ForensicDGP.exe').write_bytes(b'tampered launcher')
        data=self.result(root=self.old)
        self.assertEqual(data['mode'],'repair')
        self.assertEqual(data['launcher_verified'],0)

    def test_missing_bound_private_runtime_is_repair(self):
        self.install(self.old,'0.2.0',B)
        (self.old/'runtime/python/python-3.13.9-amd64/python.exe').unlink()
        self.assertEqual(self.result(root=self.old)['mode'],'repair')

    def test_prepared_relocation_is_not_completed_or_cleanup_eligible(self):
        self.install(self.old); self.saved(); self.result('StageMove')
        self.install(self.new,'0.2.0',B); self.result('CommitMove')
        self.assertFalse((self.new/'runtime/data/relocations'/(B+'.json')).exists())
        self.assertNotEqual(self.call('Cleanup').returncode,0)
        self.assertTrue((self.old/'ForensicDGP.exe').exists())

    def test_newer_installed_version_blocks_downgrade(self):
        self.install(self.old,'9.0.0',A)
        self.assertEqual(self.result(root=self.old)['mode'],'newer')

    def test_relocation_rejects_unrelated_candidate_without_changing_records(self):
        self.install(self.old); source,body=self.saved()
        self.source_manifest.write_text(json.dumps({'schema':1,'commit':B,'ancestors':[B]}))
        self.assertNotEqual(self.call('StageMove').returncode,0)
        self.assertEqual(source.read_bytes(),body)
        self.assertFalse(self.new.exists())

    def test_different_empty_destination_is_move(self):
        self.install(self.old)
        self.assertEqual(self.result()['mode'],'move')
        self.assertFalse(self.new.exists())

    def test_relocation_rejects_destination_inside_previous_records(self):
        self.install(self.old); source,body=self.saved()
        nested=self.old/'runtime/data/new-installation'
        result=self.call('StageMove',root=nested)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('cannot contain',result.stderr)
        self.assertFalse(nested.exists())
        self.assertEqual(source.read_bytes(),body)

    def test_relocation_rejects_destination_containing_previous_installation(self):
        self.install(self.old); source,body=self.saved()
        original=set(self.area.iterdir())
        result=self.call('StageMove',root=self.area)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(set(self.area.iterdir()),original)
        self.assertEqual(source.read_bytes(),body)

    def test_unknown_destination_is_preserved(self):
        self.install(self.old)
        self.new.mkdir(); sentinel=self.new/'personal.txt'; sentinel.write_bytes(b'do not touch')
        result=self.call('StageMove')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(sentinel.read_bytes(),b'do not touch')
        self.assertTrue((self.old/'ForensicDGP.exe').exists())

    def test_move_preserves_record_bytes_and_rebinds_only_install_metadata(self):
        self.install(self.old)
        old_record,body=self.saved()
        stale=self.old/'runtime/data/installations'/('a'*40+'.json')
        stale.parent.mkdir(parents=True,exist_ok=True); stale.write_text('{"python":"old absolute interpreter"}')
        self.result('StageMove')
        new_record=self.new/'runtime/data/scratch/capture_requests/evidence.json'
        self.assertEqual(new_record.read_bytes(),body)
        self.assertEqual(old_record.read_bytes(),body)
        owner=json.loads((self.new/'runtime/owner.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(owner['install_root'],str(self.new))
        self.assertFalse((self.new/'runtime/data/installations').exists())
        self.assertFalse((self.new/'runtime/environments').exists())
        self.assertEqual(self.result('StageMove')['mode'],'move-staged')

    def model_payload(self):
        model=self.old/'runtime/data/models/completion/codeformer_completion_r1.pth'
        model.parent.mkdir(parents=True,exist_ok=True)
        body=b'verified fixture model bytes'; model.write_bytes(body)
        payload=self.area/'offline-models.json'
        payload.write_text(json.dumps({'models':[{'group':'completion','filename':model.name,
                           'path':'models/'+model.name}],
                           'files':[{'path':'models/'+model.name,'sha256':sha(body)}]}))
        return payload,model,body

    def test_namespaced_move_stages_records_without_duplicating_legacy_models(self):
        self.install(self.old); record,record_bytes=self.saved()
        payload,model,model_bytes=self.model_payload()
        manifest=json.loads(self.source_manifest.read_text())
        manifest['files']={'implementation/model_store.py':'c'*40}
        self.source_manifest.write_text(json.dumps(manifest))
        data=self.result('StageMove',payload=payload)
        copied=self.new/'runtime/data/scratch/capture_requests/evidence.json'
        self.assertEqual(copied.read_bytes(),record_bytes)
        self.assertEqual(record.read_bytes(),record_bytes)
        self.assertEqual(model.read_bytes(),model_bytes)
        self.assertFalse((self.new/'runtime/data/models').exists())
        receipt=json.loads(Path(data['receipt']).read_text(encoding='utf-8-sig'))
        self.assertFalse(any(item['path'].startswith('runtime\\data\\models\\') for item in receipt['copied_files']))

    def test_legacy_move_still_reuses_verified_flat_model(self):
        self.install(self.old); self.saved()
        payload,model,model_bytes=self.model_payload()
        manifest=json.loads(self.source_manifest.read_text())
        manifest['files']={'implementation/setup_models.py':'d'*40}
        self.source_manifest.write_text(json.dumps(manifest))
        self.result('StageMove',payload=payload)
        self.assertEqual((self.new/'runtime/data/models/completion'/model.name).read_bytes(),model_bytes)
        self.assertEqual(model.read_bytes(),model_bytes)

    def test_invalid_namespaced_source_pin_refuses_move_before_copy(self):
        self.install(self.old); record,record_bytes=self.saved()
        payload,model,model_bytes=self.model_payload()
        manifest=json.loads(self.source_manifest.read_text())
        manifest['files']={'implementation/model_store.py':'not-a-git-blob'}
        self.source_manifest.write_text(json.dumps(manifest))
        result=self.call('StageMove',payload=payload)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.new.exists())
        self.assertEqual(record.read_bytes(),record_bytes)
        self.assertEqual(model.read_bytes(),model_bytes)

    def test_move_rejects_candidate_identity_even_when_old_commit_matches(self):
        self.install(self.old,'0.2.0',B); record,record_bytes=self.saved()
        self.source_manifest.write_text(json.dumps({'schema':1,'commit':A,'ancestors':[A,B],
                                                  'files':{'implementation/model_store.py':'c'*40}}))
        result=self.call('StageMove')
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(self.new.exists())
        self.assertEqual(record.read_bytes(),record_bytes)

    def test_move_retry_refuses_changed_destination_record(self):
        self.install(self.old); old_record,body=self.saved(); self.result('StageMove')
        target=self.new/'runtime/data/scratch/capture_requests/evidence.json'
        target.write_bytes(b'another record')
        result=self.call('StageMove')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(target.read_bytes(),b'another record')
        self.assertEqual(old_record.read_bytes(),body)

    def test_interrupted_move_with_existing_bound_receipt_can_resume(self):
        self.install(self.old); source,body=self.saved(); self.result('StageMove')
        receipt=self.new/'runtime/relocation.pending.json'
        value=json.loads(receipt.read_text(encoding='utf-8-sig')); value['copied_files']=[]
        receipt.write_text(json.dumps(value))
        target=self.new/'runtime/data/scratch/capture_requests/evidence.json'; target.unlink()
        self.assertEqual(self.result('StageMove')['mode'],'move-staged')
        self.assertEqual(target.read_bytes(),body)
        self.assertEqual(source.read_bytes(),body)

    def test_changed_source_record_blocks_commit_without_deleting_either_copy(self):
        self.install(self.old); source,body=self.saved(); self.result('StageMove')
        self.install(self.new,'0.2.0',B)
        source.write_bytes(b'new source content')
        result=self.call('CommitMove')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(source.read_bytes(),b'new source content')
        self.assertEqual((self.new/'runtime/data/scratch/capture_requests/evidence.json').read_bytes(),body)
        self.assertTrue((self.old/'ForensicDGP.exe').exists())

    def test_new_source_record_blocks_commit_and_completion(self):
        self.install(self.old); source,body=self.saved(); self.result('StageMove')
        self.install(self.new,'0.2.0',B); self.result('CommitMove')
        extra=self.old/'runtime/data/scratch/capture_requests/new-record.json'
        extra.write_bytes(b'new evidence')
        self.assertNotEqual(self.call('CommitMove').returncode,0)
        self.assertNotEqual(self.call('CompleteMove').returncode,0)
        self.assertEqual(extra.read_bytes(),b'new evidence')
        self.assertEqual(source.read_bytes(),body)
        self.assertFalse((self.new/'runtime/data/relocations'/(B+'.json')).exists())

    def test_cleanup_requires_successfully_verified_destination(self):
        self.install(self.old); self.saved(); self.result('StageMove')
        result=self.call('Cleanup')
        self.assertNotEqual(result.returncode,0)
        self.assertTrue((self.old/'ForensicDGP.exe').exists())

    def test_cleanup_preserves_unrelated_files_and_never_runs_old_uninstaller(self):
        self.install(self.old); source,body=self.saved()
        unrelated=self.old/'personal.txt'; unrelated.write_bytes(b'keep me')
        self.result('StageMove'); self.install(self.new,'0.2.0',B)
        self.result('CommitMove'); self.result('CompleteMove'); data=self.result('Cleanup')
        self.assertEqual(data['mode'],'move-cleaned')
        self.assertEqual(unrelated.read_bytes(),b'keep me')
        self.assertFalse((self.old/'ForensicDGP.exe').exists())
        self.assertEqual((self.new/'runtime/data/scratch/capture_requests/evidence.json').read_bytes(),body)

    def test_cleanup_keeps_source_files_changed_after_relocation(self):
        self.install(self.old); source,_=self.saved(); self.result('StageMove')
        self.install(self.new,'0.2.0',B); self.result('CommitMove'); self.result('CompleteMove')
        source.write_bytes(b'new evidence after relocation')
        data=self.result('Cleanup')
        self.assertEqual(source.read_bytes(),b'new evidence after relocation')
        self.assertGreater(data['retained_files'],0)

if __name__=='__main__': unittest.main()
