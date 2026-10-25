"""Pure static/fake-process transport checks; never launches a scientific worker."""
import builtins
import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'research/codeformer_u64_role_gradients_v1/runtime.py'
SPEC = importlib.util.spec_from_file_location('role_gradient_runtime_test', SOURCE)
R = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(R)


def raw_json(value):
    return (json.dumps(value, sort_keys=True)+'\n').encode()


def bytes_pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='role_gradient_transport_test_'); self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.root = self.home/R.STAGE; self.parent = self.home/R.PARENT_STAGE
        self.root.mkdir(); self.parent.mkdir()
        disk = mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=16*1024**3))
        disk.start(); self.addCleanup(disk.stop)
        self.launch_p = {'limits': {'initial_free_disk_bytes':8*1024**3}}

    def write(self, base, name, raw):
        p = base/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
        return {'path': name, **bytes_pin(raw)}

    def fixture(self):
        parent_files = {}
        standard = self.write(self.parent, 'evidence/standard.json', b'{}\n')
        parent_files[standard['path']] = {k: standard[k] for k in ('bytes', 'sha256')}
        for i in range(468):
            b = self.write(self.parent, 'assets/'+str(i)+'.txt', str(i).encode())
            parent_files[b['path']] = {k: b[k] for k in ('bytes', 'sha256')}
        parent_raw = raw_json({'files': parent_files, 'parent_protocol': standard})
        self.write(self.parent, 'protocol.json', parent_raw)
        parent_fixed = {'stage_name': R.PARENT_STAGE, **bytes_pin(parent_raw)}
        paths = ['research/__init__.py']+[R.PACKAGE+'/'+n for n in ('__init__.py','runtime.py','contract.py','analysis.py','worker.py')]
        paths += ['scripts/run_u64_role_gradients_v1.py','scripts/run_role_gradients.sh','scripts/build_u64_role_gradients_v1.py']
        files = {}
        for name in paths:
            b = self.write(self.root, name, b'# inert fixture\n')
            files[name] = {k: b[k] for k in ('bytes','sha256')}
        reviewed = [{'path': R.PACKAGE+'/run_role_gradients.sh' if n=='scripts/run_role_gradients.sh' else n, **v} for n,v in files.items()]
        proof = {'complete': True, 'source_review_pass': True, 'unresolved_blockers': [], 'sources': reviewed}
        tests = {'complete': True, 'all_tests_pass': True, 'neural_gradient_optimizer_training_VM_calls': 0, 'sources': reviewed}
        review = self.write(self.root, 'evidence/source_review.json', raw_json(proof))
        tested = self.write(self.root, 'evidence/source_tests.json', raw_json(tests))
        files.update({b['path']: {k:b[k] for k in ('bytes','sha256')} for b in (review,tested)})
        p = {'format': R.FORMAT, 'stage_name': R.STAGE, 'parent_protocol': parent_fixed, 'files': files,
             'manual_launch_only': True, 'automatic_retry_resume': False, 'model_qualified': False,
             'source_review': review, 'source_tests': tested, 'selection': {'fixture': True}, 'plan': [],
             'diagnostic': {'measurement': True}, 'limits': {'fixture': True}}
        contract = SimpleNamespace(select_cases=lambda a,b: {'fixture': True}, operation_plan=lambda s: [],
                                   prospective_contract=lambda: {'measurement': True}, LIMITS={'fixture':True})
        self.p, self.parent_fixed, self.contract = p, parent_fixed, contract
        return self.reseal()

    def reseal(self):
        raw = raw_json(self.p); self.write(self.root, 'protocol.json', raw)
        return hashlib.sha256(raw).hexdigest()

    def verify(self, sha):
        with mock.patch.object(R,'__file__',str(self.root/R.PACKAGE/'runtime.py')), mock.patch.object(R,'PARENT',self.parent_fixed), mock.patch.object(R,'_load_contract',return_value=self.contract):
            return R.verify_static(self.root,self.parent,sha)

    def test_import_and_help_are_framework_inert(self):
        old = builtins.__import__
        def blocked(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','numpy'):raise AssertionError('Forbidden framework import')
            return old(name,*args,**kwargs)
        with mock.patch('builtins.__import__',side_effect=blocked):
            spec=importlib.util.spec_from_file_location('fresh_inert_role_runtime',SOURCE)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_u64_role_gradients_v1.py'),'--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr); self.assertIn('--supervise',result.stdout)

    def test_static_verifies469_parent_files_and_review_join(self):
        sha=self.fixture(); self.assertEqual(self.verify(sha),self.p)
        (self.parent/'assets/467.txt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Consumed parent static'):self.verify(sha)

    def test_static_rejects_missing_parent_or_own_proof_and_unlisted_file(self):
        sha=self.fixture(); (self.parent/'assets/467.txt').unlink()
        with self.assertRaises(FileNotFoundError):self.verify(sha)
        self.write(self.parent,'assets/467.txt',b'467')
        self.write(self.root,'hidden.py',b'raise RuntimeError()')
        with self.assertRaisesRegex(ValueError,'Unexpected own-stage file'):self.verify(sha)
        (self.root/'hidden.py').unlink();self.p['source_review']['sha256']='0'*64;sha=self.reseal()
        with self.assertRaisesRegex(ValueError,'Pinned file differs'):self.verify(sha)

    def test_namespace_and_protocol_tampering_fail(self):
        sha=self.fixture();self.p['selection']={'tampered':True};sha=self.reseal()
        with self.assertRaisesRegex(ValueError,'scientific selection'):self.verify(sha)
        self.p['selection']={'fixture':True};self.p['files']['../escape']={'bytes':0,'sha256':'0'*64};sha=self.reseal()
        with self.assertRaises(ValueError):self.verify(sha)

    def test_root_research_bootstrap_is_pinned_before_import(self):
        target=self.home/'bootstrap';(target/'scripts').mkdir(parents=True);(target/'research').mkdir()
        script=ROOT/'scripts/run_u64_role_gradients_v1.py';(target/'scripts'/script.name).write_bytes(script.read_bytes())
        (target/'research/__init__.py').write_text("raise RuntimeError('UNPINNED_ROOT_IMPORTED')\n")
        p={'files':{'research/__init__.py':{'bytes':0,'sha256':'0'*64}}};raw=raw_json(p);(target/'protocol.json').write_bytes(raw)
        result=subprocess.run([sys.executable,'-B',str(target/'scripts'/script.name),'--root',str(target),'--protocol-sha',hashlib.sha256(raw).hexdigest(),'--run'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0);self.assertIn('Sealed bootstrap source differs: research/__init__.py',result.stderr)
        self.assertNotIn('UNPINNED_ROOT_IMPORTED',result.stderr)

    def test_run_child_changes_only_cwd_to_exact_parent(self):
        worker=SimpleNamespace(__file__=str(self.root/R.PACKAGE/'worker.py'))
        worker.run=mock.Mock(return_value=0)
        argv=['runtime','--root',str(self.root),'--protocol-sha','a'*64,'--run']
        env={R.ENV+'SUPERVISED':'1',R.ENV+'PROTOCOL':'a'*64,R.ENV+'SUPERVISOR_PID':str(R.os.getppid())}
        with mock.patch.object(sys,'argv',argv),mock.patch.object(R,'verify_static',return_value={}),mock.patch.object(R,'require_vm'),mock.patch.object(R.importlib,'import_module',return_value=worker),mock.patch.dict(R.os.environ,env),mock.patch.object(R.os,'chdir') as cwd:
            self.assertEqual(R.main(),0);cwd.assert_called_once_with(self.parent)
            worker.run.assert_called_once_with(self.root,self.parent,{},'a'*64)

    def write_status(self, counts=None, tail_counts=None):
        counts=dict.fromkeys(R.COUNTERS,0) if counts is None else counts
        self.write(self.root,R.OUTPUT+'/status.json',raw_json({'protocol_sha256':'a'*64,'counters':counts,'unknown_reservations':[],'operation_outcome_unknown':False}))
        self.write(self.root,R.OUTPUT+'/events.jsonl',raw_json({'counters':counts if tail_counts is None else tail_counts}))

    def test_status_missing_corrupt_or_stale_never_invents_zero(self):
        self.assertIsNone(R.status_snapshot(self.root,'a'*64)['counters'])
        self.assertTrue(R.status_snapshot(self.root,'a'*64)['operation_outcome_unknown'])
        self.write_status();self.assertFalse(R.status_snapshot(self.root,'a'*64)['operation_outcome_unknown'])
        counts=dict.fromkeys(R.COUNTERS,0);counts['full_forward_attempts']=counts['parity_forwards']=1
        self.write_status(tail_counts=counts);self.assertTrue(R.status_snapshot(self.root,'a'*64)['operation_outcome_unknown'])
        self.write(self.root,R.OUTPUT+'/status.json',b'{bad');self.assertIsNone(R.status_snapshot(self.root,'a'*64)['counters'])

    def test_status_pending_and_alarm_propagation(self):
        self.write_status();self.write(self.root,R.OUTPUT+'/status.json.pending',b'{}')
        self.assertTrue(R.status_snapshot(self.root,'a'*64)['operation_outcome_unknown'])
        with mock.patch.object(R,'read_json',side_effect=TimeoutError('alarm')):
            with self.assertRaises(TimeoutError):R.status_snapshot(self.root,'a'*64)

    def test_status_unmatched_returns_are_unknown_despite_matching_journal(self):
        for attempted,returned in (('full_forward_attempts','full_forward_returns'),('gradient_query_attempts','gradient_query_returns')):
            with self.subTest(kind=attempted):
                counts=dict.fromkeys(R.COUNTERS,0); counts[attempted]=1
                if attempted=='full_forward_attempts':counts['parity_forwards']=1
                self.write_status(counts)
                value=R.status_snapshot(self.root,'a'*64)
                self.assertTrue(value['journal_tail_reconciled']);self.assertTrue(value['operation_outcome_unknown'])
                counts[returned]=1;self.write_status(counts)
                self.assertFalse(R.status_snapshot(self.root,'a'*64)['operation_outcome_unknown'])

    def test_initial8GiB_admission_precedes_reservation_but_export_needs4GiB(self):
        with mock.patch.object(R,'require_vm'),mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=6*1024**3)),mock.patch.object(R,'_child') as child:
            with self.assertRaisesRegex(ValueError,'Initial8GiB'):R.supervise(self.root,self.parent,self.launch_p,'a'*64)
            child.assert_not_called();self.assertFalse((self.root/'supervisor.start.json').exists())
        with mock.patch.object(R,'require_vm'),mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=6*1024**3)),mock.patch.object(R,'verify_static',side_effect=lambda root,parent,sha,guard:(guard(), (_ for _ in ()).throw(RuntimeError('past4GiBguard')))[1]):
            with self.assertRaisesRegex(RuntimeError,'past4GiBguard'):R._export(self.root,self.parent,'a'*64)

    def test_freshness_blocks_orphan_runtime_before_spawn(self):
        for name in ('supervisor.worker.started.json','supervisor_failure.json','run.log'):
            with self.subTest(name=name):
                self.write(self.root,name,b'{}')
                with mock.patch.object(R,'require_vm'),mock.patch.object(R,'_child') as child:
                    with self.assertRaisesRegex(ValueError,'Fresh stage'):R.supervise(self.root,self.parent,{},'a'*64)
                    child.assert_not_called()
                (self.root/name).unlink()

    def test_unknown_cleanup_skips_export_but_known_failure_exports(self):
        worker={'exit_code':1,'spawn_returned':True,'cleanup':{'child_reaped':False,'process_group_absent':False}}
        with mock.patch.object(R,'require_vm'),mock.patch.object(R.signal,'getsignal'),mock.patch.object(R.signal,'signal'),mock.patch.object(R,'_child',return_value=worker) as child:
            self.assertEqual(R.supervise(self.root,self.parent,self.launch_p,'a'*64),1);self.assertEqual(child.call_count,1)
        second=self.home/'second';second.mkdir();worker['cleanup']={'child_reaped':True,'process_group_absent':True}
        with mock.patch.object(R,'require_vm'),mock.patch.object(R.signal,'getsignal'),mock.patch.object(R.signal,'signal'),mock.patch.object(R,'_child',side_effect=[worker,{'exit_code':0}]) as child:
            self.assertEqual(R.supervise(second,self.parent,self.launch_p,'a'*64),1);self.assertEqual(child.call_count,2)

    def test_fake_process_group_TERM_KILL_and_reap(self):
        child=SimpleNamespace(pid=123,wait=mock.Mock(side_effect=[subprocess.TimeoutExpired('fake',1),0]),poll=lambda:0)
        with mock.patch.object(R,'_group_alive',side_effect=[True,True,False]),mock.patch.object(R.os,'killpg',create=True) as kill,mock.patch.object(R.signal,'SIGKILL',SimpleNamespace(name='SIGKILL'),create=True):
            cleanup=R._stop_group(child)
        self.assertTrue(cleanup['child_reaped']);self.assertTrue(cleanup['process_group_absent']);self.assertEqual(kill.call_count,2)

    def test_export_requires_cleanup_even_if_supervisor_start_missing(self):
        self.write(self.root,'run.log',b'worker evidence')
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'verify_static',return_value={'files':{}}):
            with self.assertRaises(FileNotFoundError):R._export(self.root,self.parent,'a'*64)

    def test_export_streams_own_stage_only_and_LF_checksum(self):
        source=self.write(self.root,'source.txt',b'own source\n');self.write(self.root,'protocol.json',b'{}\n')
        self.write(self.parent,'secret.txt',b'never copy parent')
        parent_raw=raw_json({'files':{'secret.txt':bytes_pin(b'never copy parent')}})
        self.write(self.parent,'protocol.json',parent_raw)
        self.write_status();self.write(self.root,'run.log',b'worker log\n')
        self.write(self.root,'export.log',b'active log must not enter archive')
        self.write(self.root,'supervisor.worker.json',raw_json({'spawn_returned':True,'child_pid':123,'cleanup':{'child_reaped':True,'process_group_absent':True}}))
        p={'files':{'source.txt':{k:source[k] for k in ('bytes','sha256')}}}
        parent_fixed={'stage_name':R.PARENT_STAGE,**bytes_pin(parent_raw)}
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'verify_static',return_value=p),mock.patch.object(R,'_group_alive',return_value=False),mock.patch.object(Path,'home',return_value=self.home),mock.patch.object(R,'PARENT',parent_fixed),mock.patch.object(R.gzip,'GzipFile',wraps=gzip.GzipFile) as compressor:
            result=R._export(self.root,self.parent,'a'*64)
            self.assertEqual(compressor.call_args.kwargs['compresslevel'],1)
        archive=self.home/(R.PREFIX+'-results.tar.gz')
        with tarfile.open(archive,'r:gz') as t:
            names=t.getnames();self.assertTrue(all(n.startswith(R.STAGE+'/') for n in names))
            self.assertFalse(any('secret.txt' in n or n.endswith('/export.log') for n in names))
        with gzip.open(archive,'rb') as f:
            while f.read(8192):pass
        expected=result['archive']['sha256']+'  '+archive.name+'\n'
        self.assertEqual((self.home/(archive.name+'.sha256')).read_bytes(),expected.encode())
        self.assertFalse(result['parent_payload_included']);self.assertIn('secret.txt',result['parent_consumed_bindings'])
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'verify_static',return_value=p),mock.patch.object(R,'_group_alive',return_value=False),mock.patch.object(Path,'home',return_value=self.home):
            with self.assertRaisesRegex(ValueError,'immutable'):R._export(self.root,self.parent,'a'*64)

    def test_absolute_export_alarm_propagates_and_restores_timer(self):
        fake=SimpleNamespace(ITIMER_REAL=0,SIGALRM=14,getitimer=mock.Mock(return_value=(0.,0.)),getsignal=mock.Mock(),signal=mock.Mock(),setitimer=mock.Mock())
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'signal',fake),mock.patch.object(R,'_export',side_effect=TimeoutError('alarm')):
            with self.assertRaises(TimeoutError):R.export(self.root,self.parent,'a'*64)
            self.assertEqual(fake.setitimer.call_args_list[0].args[1],180);self.assertEqual(fake.setitimer.call_args_list[-1].args[1],0)


if __name__=='__main__':unittest.main()
