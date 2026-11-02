"""Python3.10 compatibility contract. Hash bytes and AST only; no model/VM calls."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'outputs/completion_python310_repair_v1'
ORIGINAL=ROOT/'outputs/completion_codeformer_gradient_v1_packet_r1/cctv_completion_codeformer_gradient_v1_vm'
WORKER='scripts/cctv_completion_codeformer_gradient_v1_vm.py'
EXPORTER='scripts/export_completion_codeformer_gradient_v1.py'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def only_sha(source):
    tree=ast.parse(source,feature_version=(3,10))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='sha')
    namespace={'hashlib':hashlib,'Path':Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'isolated_sha_only','exec'),namespace)
    return namespace['sha']


class Python310RepairContract(unittest.TestCase):
    def test_only_two_hash_function_bodies_change(self):
        for name in (WORKER,EXPORTER):
            old=ast.parse((D/'original'/name).read_text(),feature_version=(3,10))
            new=ast.parse((D/'payload'/name).read_text(),feature_version=(3,10))
            stripped=lambda tree:ast.dump(ast.Module(body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='sha')],type_ignores=[]),include_attributes=False)
            self.assertEqual(stripped(old),stripped(new),name)
            self.assertNotIn('file_digest',(D/'payload'/name).read_text())

    def test_stream_hash_empty_and_chunk_boundaries_without_file_digest(self):
        saved=getattr(hashlib,'file_digest',None)
        if hasattr(hashlib,'file_digest'):delattr(hashlib,'file_digest')
        try:
            for name in (WORKER,EXPORTER):
                function=only_sha((D/'payload'/name).read_text())
                with tempfile.TemporaryDirectory(dir=r'C:\xampp\htdocs\YEAR 4\Testing\outputs') as temporary:
                    for length in (0,1,1024*1024-1,1024*1024,1024*1024+1,3*1024*1024+173):
                        content=(bytes(range(251))*((length+250)//251))[:length]
                        path=Path(temporary)/'fixture.bin';path.write_bytes(content)
                        self.assertEqual(function(path),hashlib.sha256(content).hexdigest(),(name,length))
        finally:
            if saved is not None:hashlib.file_digest=saved
        self.assertNotIn('torch',sys.modules)

    def test_protocol_changes_only_two_source_pins_and_compatibility_metadata(self):
        old=json.loads((D/'original/protocol.json').read_text());new=json.loads((D/'payload/protocol.json').read_text())
        self.assertEqual(len(old['files']),27);self.assertEqual(len(new['files']),27)
        allowed={WORKER,EXPORTER}
        for a,b in zip(old['files'],new['files']):
            self.assertEqual(a['path'],b['path'])
            if a['path'] not in allowed:self.assertEqual(a,b)
            else:
                path=D/'payload'/a['path'];self.assertEqual(b,{'path':a['path'],'bytes':path.stat().st_size,'sha256':digest(path)})
        old.pop('files');new.pop('files');metadata=new.pop('python_compatibility_repair')
        self.assertEqual(old,new)
        self.assertEqual(metadata['original_protocol_sha256'],digest(D/'original/protocol.json'))
        self.assertFalse(metadata['model_or_gradient_execution_from_repair'])

    def test_payload_is_exact_base64_and_original_R1(self):
        payload=json.loads((D/'repair_payload.json').read_text())
        self.assertEqual(payload['original_protocol_sha256'],'e1e4b9fd8c4bf71db806deca03a66e187e9726cafff65ff799acc25aa432387e')
        self.assertEqual(digest(D/'original/protocol.json'),payload['original_protocol_sha256'])
        self.assertEqual(digest(D/'payload/protocol.json'),payload['new_protocol_sha256'])
        self.assertEqual(set(payload['payload_base64']),{WORKER,EXPORTER,'protocol.json'})
        for name,encoded in payload['payload_base64'].items():self.assertEqual(base64.b64decode(encoded,validate=True),(D/'payload'/name).read_bytes())
        for name in (WORKER,EXPORTER,'protocol.json'):self.assertEqual((D/'original'/name).read_bytes(),(ORIGINAL/name).read_bytes())

    def test_all_seven_packet_python_files_parse_Python310_grammar(self):
        paths=sorted(ORIGINAL.rglob('*.py'));self.assertEqual(len(paths),7)
        for path in paths:
            relative=path.relative_to(ORIGINAL);patched=D/'payload'/relative
            ast.parse((patched if patched.exists() else path).read_text(),filename=str(relative),feature_version=(3,10))


class RepairApplyContract(unittest.TestCase):
    def setUp(self):
        path=ROOT/'scripts/repair_completion_codeformer_gradient_python310_v1.py'
        spec=importlib.util.spec_from_file_location('python310_repair_under_test',path)
        self.repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.repair)
        self.temp=tempfile.TemporaryDirectory(dir=r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
        self.addCleanup(self.temp.cleanup)
        self.parent=Path(self.temp.name);self.root=self.parent/self.repair.VM_NAME;self.root.mkdir()
        self.old_bytes={WORKER:b'old-worker-fixture',EXPORTER:b'old-exporter-fixture','weights/codeformer_inpainting.pth':b'INERT-NOT-A-MODEL','assets/source.png':b'INERT-NOT-AN-IMAGE'}
        self.new_bytes={WORKER:b'new-worker-fixture-compatible',EXPORTER:b'new-exporter-fixture-compatible'}
        files=[]
        for name,value in self.old_bytes.items():
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
            files.append({'path':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()})
        old={'format':'tiny-fixture-only','files':files,'cases':['unchanged-fixture'],'budgets':{'optimizer_updates':0,'seconds':600}}
        new=json.loads(json.dumps(old));rows={r['path']:r for r in new['files']}
        repairs=[]
        for name,value in self.new_bytes.items():
            before=next(r for r in files if r['path']==name);after={'path':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()};rows[name].update(after)
            repairs.append({'path':name,'old':{k:before[k]for k in ('bytes','sha256')},'new':{k:after[k]for k in ('bytes','sha256')}})
        oldraw=(json.dumps(old,indent=2)+'\n').encode();oldsha=hashlib.sha256(oldraw).hexdigest()
        new['python_compatibility_repair']={'original_protocol_sha256':oldsha,'model_or_gradient_execution_from_repair':False}
        newraw=(json.dumps(new,indent=2)+'\n').encode();newsha=hashlib.sha256(newraw).hexdigest();(self.root/'protocol.json').write_bytes(oldraw)
        self.originals={**self.old_bytes,'protocol.json':oldraw}
        self.payload={**self.new_bytes,'protocol.json':newraw}
        self.config={'original_protocol_sha256':oldsha,'new_protocol_sha256':newsha,'repair_files':repairs,'payload_base64':{name:base64.b64encode(value).decode()for name,value in self.payload.items()}}

    def assert_originals(self):
        for name,value in self.originals.items():self.assertEqual((self.root/name).read_bytes(),value,name)

    def test_success_preserves_all_originals_sibling_backup_and_second_apply_noop(self):
        r=self.repair.apply(self.root,self.config);self.assertTrue(r['complete']);self.assertEqual(r['files_changed'],3)
        backup=Path(r['backup']);self.assertEqual(backup.parent,self.parent)
        for name in self.payload:self.assertEqual((backup/'original'/name).read_bytes(),self.originals[name]);self.assertEqual((self.root/name).read_bytes(),self.payload[name])
        for name in self.old_bytes.keys()-self.new_bytes.keys():self.assertEqual((self.root/name).read_bytes(),self.old_bytes[name])
        self.assertTrue((backup/'repair_started.json').exists());self.assertTrue((backup/'repair_complete.json').exists())
        second=self.repair.apply(self.root,self.config);self.assertTrue(second['already_applied']);self.assertEqual(second['files_changed'],0)
        self.assertEqual(len(list(self.parent.glob(self.repair.VM_NAME+'_python310_repair_*'))),1)
        self.assertNotIn('torch',sys.modules)

    def test_wrong_protocol_and_changed_source_weight_or_data_refuse_before_backup(self):
        for name in ('protocol.json',WORKER,'weights/codeformer_inpainting.pth','assets/source.png'):
            with self.subTest(name=name):
                file=self.root/name;old=file.read_bytes();file.write_bytes(old+b'CHANGED')
                with self.assertRaises(ValueError):self.repair.apply(self.root,self.config)
                self.assertFalse(list(self.parent.glob(self.repair.VM_NAME+'_python310_repair_*')))
                file.write_bytes(old)
        self.assert_originals()

    def test_started_execution_namespace_refused_without_mutation(self):
        for name in ('outputs','supervisor.json','diagnostic.log','failure.json'):
            with self.subTest(name=name):
                p=self.root/name
                if name=='outputs':p.mkdir()
                else:p.write_text('retained evidence')
                with self.assertRaisesRegex(ValueError,'may have started'):self.repair.apply(self.root,self.config)
                self.assert_originals();self.assertFalse(list(self.parent.glob(self.repair.VM_NAME+'_python310_repair_*')))
                if p.is_dir():p.rmdir()
                else:p.unlink()

    def test_postcommit_verification_failure_rolls_back_known_bytes_and_retains_failure(self):
        original_verify=self.repair.verify_files
        def reject_new(root,protocol):
            if 'python_compatibility_repair'in protocol:raise OSError('synthetic final verification failure')
            return original_verify(root,protocol)
        with patch.object(self.repair,'verify_files',side_effect=reject_new):
            with self.assertRaisesRegex(OSError,'synthetic'):self.repair.apply(self.root,self.config)
        self.assert_originals()
        backups=list(self.parent.glob(self.repair.VM_NAME+'_python310_repair_*'));self.assertEqual(len(backups),1)
        failure=json.loads((backups[0]/'repair_failure.json').read_text());self.assertFalse(failure['complete']);self.assertEqual(set(failure['rolled_back']),set(self.payload));self.assertFalse((backups[0]/'repair_complete.json').exists())

    def test_changed_unrecognized_bytes_are_not_overwritten_by_rollback(self):
        original_replace=self.repair.os.replace;triggered=False
        unknown=self.root/EXPORTER
        def interrupt_replace(source,target):
            nonlocal triggered
            if Path(target).name=='protocol.json'and not triggered:
                triggered=True;unknown.write_bytes(b'EXTERNAL-UNKNOWN-CONTENT');raise OSError('synthetic concurrent source change')
            return original_replace(source,target)
        with patch.object(self.repair.os,'replace',side_effect=interrupt_replace):
            with self.assertRaisesRegex(OSError,'concurrent'):self.repair.apply(self.root,self.config)
        self.assertEqual(unknown.read_bytes(),b'EXTERNAL-UNKNOWN-CONTENT')
        self.assertEqual((self.root/WORKER).read_bytes(),self.originals[WORKER]);self.assertEqual((self.root/'protocol.json').read_bytes(),self.originals['protocol.json'])
        backup=next(self.parent.glob(self.repair.VM_NAME+'_python310_repair_*'));failure=json.loads((backup/'repair_failure.json').read_text());self.assertNotIn(EXPORTER,failure['rolled_back'])

    def test_apply_runs_with_file_digest_absent_and_wrong_self_pin_refuses(self):
        with patch.object(sys,'argv',['repair','--root',str(self.root),'--expected-sha','0'*64]):
            with self.assertRaisesRegex(ValueError,'Repair-script SHA256'):self.repair.main()
        self.assert_originals()
        saved=getattr(hashlib,'file_digest',None)
        if hasattr(hashlib,'file_digest'):delattr(hashlib,'file_digest')
        try:
            result=self.repair.apply(self.root,self.config)
            self.assertTrue(result['complete']);self.assertEqual(result['files_changed'],3)
        finally:
            if saved is not None:hashlib.file_digest=saved
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
