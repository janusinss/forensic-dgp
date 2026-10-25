"""Pure synthetic-file and stub-process tests; no worker, sealing, model or VM."""
import ast
import builtins
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'research/codeformer_u64_png_surrogate_measurement_v1/runtime.py'
SPEC=importlib.util.spec_from_file_location('png_surrogate_runtime_tests',SOURCE)
R=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(R)


def raw_json(value):return (json.dumps(value,sort_keys=True)+'\n').encode()
def payload(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def plan_fixture():
    plan=[]
    for slot in range(1,73):
        phase='historical_u64_parity' if slot<=4 else ('deterministic_same_u64_repeat' if slot<=8 else 'paired_TRAIN_gradient_diagnostic')
        plan.append({'slot':slot,'id':'case_'+str(slot),'phase':phase,
                     'gradient_queries':[] if slot<=8 else ['raw_mae','raw_adjacent','png_surrogate_mae','png_surrogate_adjacent']})
    return plan


def journal_fixture(plan):
    counters=dict.fromkeys(R.COUNTERS,0);rows=[]
    for token,(kind,call,query) in enumerate(R.operation_sequence(plan),1):
        data={'token':token,'plan_slot':call['slot'],'id':call['id']}
        if kind=='forward':data['category']=R.PHASE_CATEGORY[call['phase']];counters[R.PHASE_CATEGORY[call['phase']]]+=1
        if query is not None:data['query']=query
        counters[R.OPERATIONS[kind][0]]+=1
        rows.append(dict(data,kind=kind+'_reserved',counters=dict(counters),elapsed_seconds=len(rows)/100))
        counters[R.OPERATIONS[kind][1]]+=1
        rows.append(dict(data,kind=kind+'_returned',counters=dict(counters),elapsed_seconds=len(rows)/100,synchronized_return=True))
    return rows


class PNGSurrogateTransportTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='png_surrogate_transport_test_');self.addCleanup(temp.cleanup)
        self.home=Path(temp.name);self.root=self.home/R.STAGE;self.parent=self.home/R.PARENT_STAGE
        self.root.mkdir();self.parent.mkdir()
        m=mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=16*1024**3));m.start();self.addCleanup(m.stop)

    def write(self,base,name,data):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return {'path':name,**payload(data)}

    def fixture(self):
        files={};standard=self.write(self.parent,'evidence/standard.json',b'{}\n')
        for n in range(468):
            row=self.write(self.parent,'assets/'+str(n)+'.txt',str(n).encode());files[row['path']]={k:row[k]for k in ('bytes','sha256')}
        files[standard['path']]={k:standard[k]for k in ('bytes','sha256')}
        parent_raw=raw_json({'files':files,'parent_protocol':standard});self.write(self.parent,'protocol.json',parent_raw)
        self.parent_pin={'stage_name':R.PARENT_STAGE,**payload(parent_raw)}
        names=['research/__init__.py']+[R.PACKAGE+'/'+n for n in ('__init__.py','runtime.py','contract.py','surrogate.py','analysis.py','worker.py')]
        names+=['scripts/run_u64_png_surrogate_measurement_v1.py','scripts/run_png_surrogate.sh','scripts/build_u64_png_surrogate_measurement_v1.py']
        own={}
        for name in names:
            row=self.write(self.root,name,b'# inert synthetic source\n');own[name]={k:row[k]for k in ('bytes','sha256')}
        sources=[{'path':R.PACKAGE+'/run_png_surrogate.sh'if n=='scripts/run_png_surrogate.sh'else n,**v}for n,v in own.items()]
        proofs={}
        for key,extra in [('source_review',{'source_review_pass':True,'unresolved_blockers':[]}),('source_tests',{'all_tests_pass':True,'neural_gradient_optimizer_training_VM_calls':0})]:
            row=self.write(self.root,'evidence/'+key+'.json',raw_json({'complete':True,'sources':sources,**extra}));proofs[key]=row;own[row['path']]={k:row[k]for k in ('bytes','sha256')}
        parent_copy=self.write(self.root,'evidence/parent_protocol.json',parent_raw)
        standard_copy=self.write(self.root,'evidence/standard_protocol.json',b'{}\n')
        for binding in (parent_copy,standard_copy):own[binding['path']]={k:binding[k]for k in ('bytes','sha256')}
        self.p={'format':R.FORMAT,'stage_name':R.STAGE,'parent_protocol':self.parent_pin,'files':own,**proofs,
                'parent_protocol_copy':parent_copy,'standard_protocol_copy':standard_copy,
                'manual_launch_only':True,'automatic_retry_resume':False,'model_qualified':False,
                'selection':{'synthetic':True},'plan':plan_fixture(),'diagnostic':{'synthetic':True},'limits':{'synthetic':True}}
        self.contract=SimpleNamespace(select_cases=lambda *a:{'synthetic':True},operation_plan=lambda s:plan_fixture(),
             prospective_contract=lambda:{'synthetic':True},LIMITS={'synthetic':True})
        return self.reseal()

    def reseal(self):
        raw=raw_json(self.p);self.write(self.root,'protocol.json',raw);return hashlib.sha256(raw).hexdigest()

    def verify(self,sha):
        with mock.patch.object(R,'__file__',str(self.root/R.PACKAGE/'runtime.py')),mock.patch.object(R,'PARENT',self.parent_pin),mock.patch.object(R,'HELPER_PINS',{}),mock.patch.object(R,'_load_contract',return_value=self.contract):
            return R.verify_static(self.root,self.parent,sha)

    def test_runtime_and_builder_imports_and_bridge_help_are_inert(self):
        old=builtins.__import__
        def blocked(name,*a,**k):
            if name.split('.')[0]in ('torch','numpy','tensorflow','jax'):raise AssertionError('Forbidden import')
            return old(name,*a,**k)
        with mock.patch('builtins.__import__',side_effect=blocked):
            for path in (SOURCE,ROOT/'scripts/build_u64_png_surrogate_measurement_v1.py'):
                spec=importlib.util.spec_from_file_location('fresh_pure_'+path.stem,path);spec.loader.exec_module(importlib.util.module_from_spec(spec))
        result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_u64_png_surrogate_measurement_v1.py'),'--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--supervise',result.stdout)

    def test_python310_syntax_and_shell_FRESH_manual_stage(self):
        for path in (SOURCE,ROOT/'scripts/build_u64_png_surrogate_measurement_v1.py',ROOT/'scripts/run_u64_png_surrogate_measurement_v1.py'):
            ast.parse(path.read_text(),feature_version=(3,10))
        shell=(SOURCE.parent/'run_png_surrogate.sh').read_text();self.assertIn('TMUX',shell);self.assertIn(R.STAGE,shell);self.assertIn('--supervise',shell)

    def test_static_all469_and_source_admission(self):
        sha=self.fixture();self.assertEqual(self.verify(sha),self.p)
        (self.parent/'assets/467.txt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'Consumed parent static'):self.verify(sha)

    def test_missing_parent_alias_is_not_allowed_on_VM(self):
        sha=self.fixture();(self.parent/'assets/467.txt').unlink()
        with self.assertRaises(FileNotFoundError):self.verify(sha)

    def test_source_tamper_namespace_and_bad_approval_refused(self):
        sha=self.fixture();(self.root/R.PACKAGE/'worker.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Own static'):self.verify(sha)

    def test_unlisted_own_file_refused(self):
        sha=self.fixture();self.write(self.root,'hidden.py',b'bad')
        with self.assertRaisesRegex(ValueError,'Unexpected own-stage'):self.verify(sha)

    def test_exact_full72_plan328_operation_replay(self):
        plan=plan_fixture();self.assertEqual(len(plan),72)
        result=R.replay_journal(b''.join(raw_json(r)for r in journal_fixture(plan)),plan)
        self.assertEqual(result['completed_operation_pairs'],328);self.assertEqual(result['counters'],R.COUNTERS)
        self.assertTrue(result['full_exact_plan_completed']);self.assertFalse(result['operation_outcome_unknown'])

    def test_complete_failed_measurement_prefix_is_valid(self):
        plan=plan_fixture();rows=journal_fixture(plan);result=R.replay_journal(b''.join(raw_json(r)for r in rows[:120]),plan)
        self.assertTrue(result['valid_sealed_plan_prefix']);self.assertFalse(result['full_exact_plan_completed']);self.assertFalse(result['operation_outcome_unknown'])

    def test_unreturned_query_is_explicitly_unknown(self):
        plan=plan_fixture();rows=journal_fixture(plan);index=next(i for i,r in enumerate(rows)if r['kind']=='gradient_query_reserved')
        result=R.replay_journal(b''.join(raw_json(r)for r in rows[:index+1]),plan)
        self.assertTrue(result['operation_outcome_unknown']);self.assertEqual(result['counters']['gradient_query_attempts'],1)
        self.assertEqual(result['counters']['gradient_query_returns'],0)

    def test_query_order_counter_return_and_truncated_tail_refused(self):
        plan=plan_fixture();original=journal_fixture(plan)
        for label,mutate in [('query',lambda r:r[18].update(query='png_surrogate_adjacent')),('counter',lambda r:r[1]['counters'].update(full_forward_returns=0)),('return',lambda r:r[1].update(token=9))]:
            rows=copy.deepcopy(original);mutate(rows)
            with self.subTest(label=label),self.assertRaises(ValueError):R.replay_journal(b''.join(raw_json(r)for r in rows),plan)
        with self.assertRaises(ValueError):R.replay_journal(raw_json(original[0])[:-1],plan)

    def test_status_replay_reconciles_all_measurement_counters(self):
        plan=plan_fixture();rows=journal_fixture(plan);counts=R.replay_journal(b''.join(raw_json(r)for r in rows),plan)['counters']
        self.write(self.root,R.OUTPUT+'/events.jsonl',b''.join(raw_json(r)for r in rows))
        self.write(self.root,R.OUTPUT+'/status.json',raw_json({'protocol_sha256':'a'*64,'counters':counts,'unknown_reservations':[],'operation_outcome_unknown':False}))
        result=R.status_snapshot(self.root,'a'*64,{'plan':plan});self.assertFalse(result['operation_outcome_unknown'])
        self.write(self.root,R.OUTPUT+'/status.json.pending',b'{}');self.assertTrue(R.status_snapshot(self.root,'a'*64,{'plan':plan})['operation_outcome_unknown'])

    def test_unknown_status_never_invents_zero_and_alarm_propagates(self):
        self.assertIsNone(R.status_snapshot(self.root,'a'*64)['counters'])
        self.write(self.root,R.OUTPUT+'/status.json',b'bad');self.assertIsNone(R.status_snapshot(self.root,'a'*64)['counters'])
        with mock.patch.object(R,'read_json',side_effect=TimeoutError('alarm')):
            with self.assertRaises(TimeoutError):R.status_snapshot(self.root,'a'*64)

    def test_actual_contract_and_worker_ledger_match_transport_API(self):
        from research.codeformer_u64_png_surrogate_measurement_v1 import contract,worker
        for key,value in R.COUNTERS.items():self.assertEqual(contract.LIMITS[key],value)
        self.assertEqual(set(worker.COUNTER_KEYS),set(R.COUNTERS))
        for name in ('STAGE','PARENT_STAGE','PACKAGE','OUTPUT','PREFIX','FORMAT'):self.assertEqual(getattr(contract,name),getattr(R,name))
        parent_raw=(ROOT/'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/protocol.json').read_bytes()
        parent=json.loads(parent_raw);standard_raw=(ROOT/'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'/parent['parent_protocol']['path']).read_bytes()
        selection=contract.select_cases(parent_raw,standard_raw);plan=contract.operation_plan(selection)
        self.assertEqual(len(plan),72);self.assertEqual(len(R.operation_sequence(plan)),328)
        self.assertEqual(contract.prospective_contract()['limits'],contract.LIMITS)
        rows=[];ledger=worker.OperationLedger(lambda row:rows.append(copy.deepcopy(row)))
        for kind,call,query in R.operation_sequence(plan):
            data={'id':call['id'],'plan_slot':call['slot']}
            if kind=='forward':data['category']=R.PHASE_CATEGORY[call['phase']]
            if query is not None:data['query']=query
            token=ledger.reserve(kind,**data);ledger.returned(token,synchronized_return=True)
        result=R.replay_journal(b''.join(raw_json(r)for r in rows),plan)
        self.assertEqual(result['counters'],R.COUNTERS);self.assertFalse(result['operation_outcome_unknown'])

    def test_unknown_operation_and_overlapping_reservation_refused(self):
        plan=plan_fixture();rows=journal_fixture(plan)
        with self.assertRaises(ValueError):R.replay_journal(raw_json({'kind':'optimizer_reserved','token':1}),plan)
        with self.assertRaises(ValueError):R.replay_journal(raw_json(rows[0])+raw_json(rows[2]),plan)
        with self.assertRaises(ValueError):R.replay_journal(raw_json({'kind':'inference_reserved'}),plan)
        ordinary={'kind':'phase_returned','counters':dict.fromkeys(R.COUNTERS,0),'phase_seconds':.1}
        self.assertFalse(R.replay_journal(raw_json(ordinary),plan)['operation_outcome_unknown'])

    def test_run_bridge_rejects_unpinned_root_before_import(self):
        target=self.home/'bootstrap';(target/'scripts').mkdir(parents=True);(target/'research').mkdir()
        script=ROOT/'scripts/run_u64_png_surrogate_measurement_v1.py';(target/'scripts'/script.name).write_bytes(script.read_bytes())
        (target/'research/__init__.py').write_text("raise RuntimeError('UNPINNED_IMPORTED')\n")
        raw=raw_json({'files':{'research/__init__.py':payload(b'')}});(target/'protocol.json').write_bytes(raw)
        result=subprocess.run([sys.executable,'-B',str(target/'scripts'/script.name),'--root',str(target),'--protocol-sha',hashlib.sha256(raw).hexdigest(),'--run'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0);self.assertIn('Sealed bootstrap source differs',result.stderr);self.assertNotIn('UNPINNED_IMPORTED',result.stderr)

    def test_run_mode_changes_only_cwd_to_exact_readonly_parent(self):
        worker=SimpleNamespace(__file__=str(self.root/R.PACKAGE/'worker.py'),run=mock.Mock(return_value=2))
        env={R.ENV+'SUPERVISED':'1',R.ENV+'PROTOCOL':'a'*64,R.ENV+'SUPERVISOR_PID':str(R.os.getppid())}
        with mock.patch.object(sys,'argv',['runtime','--root',str(self.root),'--protocol-sha','a'*64,'--run']),mock.patch.object(R,'verify_static',return_value={}),mock.patch.object(R,'require_vm'),mock.patch.object(R.importlib,'import_module',return_value=worker),mock.patch.dict(R.os.environ,env),mock.patch.object(R.os,'chdir')as cwd:
            self.assertEqual(R.main(),2);cwd.assert_called_once_with(self.parent);worker.run.assert_called_once_with(self.root,self.parent,{},'a'*64)

    def test_supervisor_unknown_cleanup_skips_export_known_failure_exports(self):
        limits={'initial_free_disk_bytes':8*1024**3,'external_seconds':960,'export_external_seconds':330}
        worker={'exit_code':2,'spawn_returned':True,'cleanup':{'child_reaped':False,'process_group_absent':False}}
        with mock.patch.object(R,'require_vm'),mock.patch.object(R.signal,'getsignal'),mock.patch.object(R.signal,'signal'),mock.patch.object(R,'_child',return_value=worker)as child:
            self.assertEqual(R.supervise(self.root,self.parent,{'limits':limits,'plan':[]},'a'*64),2);self.assertEqual(child.call_count,1)
        second=self.home/'other_stage';second.mkdir();worker['cleanup']={'child_reaped':True,'process_group_absent':True}
        with mock.patch.object(R,'require_vm'),mock.patch.object(R.signal,'getsignal'),mock.patch.object(R.signal,'signal'),mock.patch.object(R,'_child',side_effect=[worker,{'exit_code':0}])as child:
            self.assertEqual(R.supervise(second,self.parent,{'limits':limits,'plan':[]},'a'*64),2);self.assertEqual(child.call_count,2)

    def test_builder_admission_requires_actual_every_reviewed_source(self):
        path=ROOT/'scripts/build_u64_png_surrogate_measurement_v1.py';spec=importlib.util.spec_from_file_location('builder_pure_test',path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        source=self.write(self.home,'source.py',b'# synthetic reviewed\n')
        review=self.home/'review.json';review.write_bytes(raw_json({'complete':True,'source_review_pass':True,'unresolved_blockers':[],'sources':[source]}))
        original_pin=builder.pin
        with mock.patch.object(builder,'ROOT',self.home),mock.patch.object(builder,'SOURCES',('source.py',)),mock.patch.object(builder,'pin',side_effect=lambda path:original_pin(path,self.home)):
            self.assertTrue(builder.admission(review)['source_review_pass'])
            (self.home/'source.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source changed after approval'):builder.admission(review)

    def test_external_watchdog_resource_breach_stops_owned_group(self):
        child=SimpleNamespace(pid=12,poll=mock.Mock(return_value=None),wait=mock.Mock(return_value=0))
        usage={'free_disk_bytes':3*1024**3,'output_bytes':0,'current_RSS_bytes':1,'archive_bytes':0}
        with mock.patch.object(R.subprocess,'Popen',return_value=child),mock.patch.object(R,'external_usage',return_value=usage),mock.patch.object(R,'_stop_group',return_value={'child_reaped':True,'process_group_absent':True})as stop:
            receipt=R._child(self.root,['inert_stub'],'worker',10,{})
        self.assertEqual(receipt['exit_code'],125);self.assertIn('External continuing4GiB',receipt['error']);stop.assert_called_once_with(child)

    def test_external_watchdog_records_stub_success(self):
        child=SimpleNamespace(pid=12,poll=mock.Mock(side_effect=[None,0]),wait=mock.Mock(return_value=0))
        usage={'free_disk_bytes':8*1024**3,'output_bytes':9,'current_RSS_bytes':10,'archive_bytes':0}
        with mock.patch.object(R.subprocess,'Popen',return_value=child),mock.patch.object(R,'external_usage',return_value=usage),mock.patch.object(R,'_stop_group',return_value={'child_reaped':True,'process_group_absent':True}):receipt=R._child(self.root,['inert_stub'],'worker',10,{})
        self.assertEqual(receipt['exit_code'],0);self.assertEqual(receipt['resource_samples'],1)

    def test_initial_space_and_freshness_precede_spawn(self):
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'_child')as child,mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=6*1024**3)):
            with self.assertRaisesRegex(ValueError,'Initial8GiB'):R.supervise(self.root,self.parent,{'limits':{'initial_free_disk_bytes':8*1024**3}},'a'*64)
            child.assert_not_called()
        self.write(self.root,'supervisor.worker.resources.jsonl',b'')
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'_child')as child:
            with self.assertRaisesRegex(ValueError,'Fresh stage'):R.supervise(self.root,self.parent,{},'a'*64)
            child.assert_not_called()

    def test_export_streams_python310_ustar_own_files_only(self):
        source=self.write(self.root,'source.txt',b'own synthetic bytes');self.write(self.root,'protocol.json',b'{}\n')
        parent_raw=raw_json({'files':{'never_included':payload(b'parent')}});self.write(self.parent,'protocol.json',parent_raw)
        parent={'stage_name':R.PARENT_STAGE,**payload(parent_raw)}
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'verify_static',return_value={'files':{'source.txt':payload(b'own synthetic bytes')},'plan':[]}),mock.patch.object(Path,'home',return_value=self.home),mock.patch.object(R,'PARENT',parent):result=R._export(self.root,self.parent,'a'*64)
        archive=self.home/(R.PREFIX+'-results.tar.gz')
        with tarfile.open(archive,'r:gz')as tar:
            self.assertEqual(set(tar.getnames()),{R.STAGE+'/source.txt',R.STAGE+'/protocol.json'})
        with gzip.open(archive,'rb')as f:
            while f.read(8192):pass
        self.assertEqual((self.home/(archive.name+'.sha256')).read_bytes(),(result['archive']['sha256']+'  '+archive.name+'\n').encode())
        self.assertFalse(result['parent_payload_included'])

    def test_zero_applications_and_optimizers_refused_in_plan_or_journal(self):
        plan=plan_fixture();plan[0]['kind']='parameter_application'
        with self.assertRaises(ValueError):R.operation_sequence(plan)
        for kind in ('parameter_application_reserved','optimizer_constructed','backward_returned'):
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                R.replay_journal(raw_json({'kind':kind,'counters':dict.fromkeys(R.COUNTERS,0)}),plan_fixture())

    def test_absolute_export_timeout_retains_bounded_failure_receipt(self):
        fake=SimpleNamespace(ITIMER_REAL=0,SIGALRM=14,getitimer=mock.Mock(return_value=(0.,0.)),getsignal=mock.Mock(),signal=mock.Mock(),setitimer=mock.Mock())
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'signal',fake),mock.patch.object(R,'_export',side_effect=TimeoutError('inert_alarm')):
            with self.assertRaises(TimeoutError):R.export(self.root,self.parent,'a'*64)
        self.assertEqual(fake.setitimer.call_args_list[0].args[1],300);self.assertEqual(fake.setitimer.call_args_list[1].args[1],5);self.assertEqual(fake.setitimer.call_args_list[-1].args[1],0)
        self.assertIn('inert_alarm',json.loads((self.root/'export_failure.json').read_bytes())['error'])


if __name__=='__main__':unittest.main()
