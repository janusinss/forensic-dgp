"""Pure synthetic-file/stub-process transport tests; no worker/model/VM run."""
import ast
import builtins
import copy
from contextlib import redirect_stdout
import gzip
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'research/codeformer_u64_png_checked_step_v1/runtime.py'
SPEC=importlib.util.spec_from_file_location('png_checked_runtime_tests',SOURCE)
R=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(R)


def raw_json(value):return (json.dumps(value,sort_keys=True)+'\n').encode()
def payload(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def plan_fixture():
    plan=[]
    def forward(category,trial=None):
        slot=len(plan)+1
        plan.append({'kind':'forward','slot':slot,'id':'case_'+str(slot),'trial_index':trial,
            'category':category,'gradient_queries':[],'phase':'synthetic'})
    def apply(action,trial):
        slot=len(plan)+1
        plan.append({'kind':'parameter_application','slot':slot,'id':action+'_'+str(trial),'trial_index':trial,
            'action':action,'gradient_queries':[],'phase':action})
    for _ in range(56):forward('initial_quality_forwards')
    for _ in range(4):forward('initial_repeat_forwards')
    for _ in range(64):forward('initial_optimization_forwards')
    for trial in range(4):
        apply('trial_apply',trial)
        for _ in range(64):forward('trial_optimization_forwards',trial)
        apply('trial_restore',trial)
    apply('selected_commit',None)
    for _ in range(56):forward('selected_quality_forwards')
    return plan


def metadata(call):
    result={'id':call['id'],'plan_slot':call['slot'],'trial_index':call['trial_index']}
    if call['kind']=='forward':result['category']=call['category']
    else:result['action']=call['action']
    return result


def returned_metadata(call,current):
    if call['kind']=='forward':return {'synchronized_return':True},current
    trial=call['trial_index'];action=call['action']
    after=R.ORIGINAL_HEAD_SHA if action=='trial_restore' else hashlib.sha256(('trial'+str(0 if trial is None else trial)).encode()).hexdigest()
    return {'synchronized_return':True,'actual_before_head_canonical_sha256':current,
            'actual_after_head_canonical_sha256':after},after


def journal_fixture(plan):
    rows=[];counts=dict.fromkeys(R.COUNTERS,0);current=R.ORIGINAL_HEAD_SHA
    for token,(kind,call,query) in enumerate(R.operation_sequence(plan),1):
        data={'token':token,**metadata(call)}
        counts[R.OPERATIONS[kind][0]]+=1
        if kind=='forward':counts[call['category']]+=1
        rows.append({**data,'kind':kind+'_reserved','counters':dict(counts),'elapsed_seconds':len(rows)/100})
        counts[R.OPERATIONS[kind][1]]+=1
        if kind=='parameter_application':counts[R.ACTION_COUNTERS[call['action']]]+=1
        extra,current=returned_metadata(call,current)
        rows.append({**data,**extra,'kind':kind+'_returned','counters':dict(counts),'elapsed_seconds':len(rows)/100})
    return rows


class CheckedTransportTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='png_checked_transport_test_');self.addCleanup(temp.cleanup)
        self.home=Path(temp.name);self.root=self.home/R.STAGE;self.parent=self.home/R.PARENT_STAGE
        self.root.mkdir();self.parent.mkdir()
        patch=mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=16*1024**3))
        patch.start();self.addCleanup(patch.stop)

    def write(self,base,name,data):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return {'path':name,**payload(data)}

    def fixture(self):
        standard=self.write(self.parent,'evidence/standard.json',b'{}\n');files={}
        for n in range(468):
            value=self.write(self.parent,'assets/'+str(n)+'.txt',str(n).encode());files[value['path']]=payload(str(n).encode())
        files[standard['path']]=payload(b'{}\n');parent_raw=raw_json({'files':files,'parent_protocol':standard})
        self.write(self.parent,'protocol.json',parent_raw);self.parent_pin={'stage_name':R.PARENT_STAGE,**payload(parent_raw)}
        names=['research/__init__.py']+[R.PACKAGE+'/'+n for n in ('__init__.py','runtime.py','contract.py','selection.py','quality.py','worker.py')]
        names+=['scripts/run_u64_png_checked_step_v1.py','scripts/run_png_checked.sh','scripts/build_u64_png_checked_step_v1.py']
        own={}
        for name in names:own[name]=payload(b'# synthetic inert source\n');self.write(self.root,name,b'# synthetic inert source\n')
        sources=[{'path':R.PACKAGE+'/run_png_checked.sh' if n=='scripts/run_png_checked.sh' else n,**v} for n,v in own.items()]
        proofs={}
        for key,extra in [('source_review',{'source_review_pass':True,'unresolved_blockers':[]}),
                          ('source_tests',{'all_tests_pass':True,'neural_gradient_optimizer_training_VM_calls':0})]:
            value=self.write(self.root,'evidence/'+key+'.json',raw_json({'complete':True,'sources':sources,**extra}))
            proofs[key]=value;own[value['path']]={k:value[k] for k in ('bytes','sha256')}
        for key,name,data in [('parent_protocol_copy','evidence/parent_protocol.json',parent_raw),
                              ('standard_protocol_copy','evidence/standard_protocol.json',b'{}\n')]:
            value=self.write(self.root,name,data);proofs[key]=value;own[name]=payload(data)
        self.p={'format':R.FORMAT,'stage_name':R.STAGE,'parent_protocol':self.parent_pin,'files':own,**proofs,
            'manual_launch_only':True,'automatic_retry_resume':False,'model_qualified':False,
            'selection':{'synthetic':True},'plan':plan_fixture(),'pilot':{'synthetic':True},'limits':{'synthetic':True}}
        self.contract=SimpleNamespace(select_cases=lambda *a:{'synthetic':True},operation_plan=lambda s:plan_fixture(),
            prospective_contract=lambda:{'synthetic':True},LIMITS={'synthetic':True})
        return self.reseal()

    def reseal(self):
        data=raw_json(self.p);self.write(self.root,'protocol.json',data);return hashlib.sha256(data).hexdigest()

    def verify(self,sha):
        with mock.patch.object(R,'__file__',str(self.root/R.PACKAGE/'runtime.py')),mock.patch.object(R,'PARENT',self.parent_pin),\
             mock.patch.object(R,'HELPER_PINS',{}),mock.patch.object(R,'_load_contract',return_value=self.contract),\
             mock.patch.object(R,'verify_retained_measurement',return_value=self.home/R.MEASUREMENT_STAGE) as dependency:
            result=R.verify_static(self.root,self.parent,sha);dependency.assert_called_once();return result

    def measurement_fixture(self):
        base=self.home/R.MEASUREMENT_STAGE;base.mkdir();static={}
        for n in range(21):
            name='static/'+str(n)+'.txt';self.write(base,name,b's');static[name]=payload(b's')
        calls=[{'slot':i,'id':'measured_'+str(i),'gradient_queries':[] if i<=8 else
                ['raw_mae','raw_adjacent','png_surrogate_mae','png_surrogate_adjacent']} for i in range(1,73)]
        m={'files':static,'plan':calls};protocol_raw=raw_json(m);self.write(base,'protocol.json',protocol_raw)
        self.measurement_protocol={'path':'protocol.json',**payload(protocol_raw)}
        files=dict(static)
        def put(name,value):
            data=raw_json(value) if not isinstance(value,bytes) else value
            self.write(base,name,data);files[name]=payload(data)
        counts={'full_forward_attempts':72,'full_forward_returns':72,'gradient_query_attempts':256,'gradient_query_returns':256,
                'parity_forwards':4,'repeat_forwards':4,'diagnostic_forwards':64}
        common={'complete':True,'protocol_sha256':self.measurement_protocol['sha256'],'counters':counts,
                'unknown_reservations':[],'operation_outcome_unknown':False,'model_qualified':False}
        result={**common,'optimizer_constructions':0,'optimizer_steps':0,'parameter_applications':0,
                'all575_parameters_bitwise_unchanged':True,'all539_frozen_exact':True,'gradient_fields_all_None':True}
        prefix=R.MEASUREMENT_OUTPUT+'/'
        for name,value in [('result.json',result),('status.json',common),('head_schema.json',{}),('operation_plan.json',{}),
            ('RNG_sampler_initial.json',{'RNG':{'synthetic':'same'},'optimizer':None}),
            ('RNG_sampler_closing.json',{'RNG':{'synthetic':'same'},'optimizer':None}),
            ('full_states/initial.json',{'canonical_sha256':R.ORIGINAL_FULL_SHA}),
            ('full_states/closing.json',{'canonical_sha256':R.ORIGINAL_FULL_SHA})]:put(prefix+name,value)
        for name,value in [('supervisor.start.json',{}),('supervisor.worker.started.json',{'child_pid':12,'process_group':12}),
            ('supervisor.worker.json',{'exit_code':0,'timeout':False,'error':None,'spawn_returned':True,'child_pid':12,
              'cleanup':{'child_reaped':True,'process_group_absent':True}})]:put(name,value)
        for call in calls[8:]:
            stem=prefix+'calls/'+str(call['slot']).zfill(4)+'/'
            put(stem+'record.json',{'call':call,'split':'TRAIN'})
            gradients={}
            for query in ('raw_mae','raw_adjacent'):
                name=stem+'gradients_'+query+'.npz';put(name,b'synthetic undecoded NPZ bytes')
                gradients[query]={'file':{'path':name,**files[name]},'head_state_canonical_sha256':R.ORIGINAL_HEAD_SHA}
            put(stem+'gradients.json',{'complete':True,'call':call,'gradients':gradients})
        dependency={'stage_name':R.MEASUREMENT_STAGE,'protocol':self.measurement_protocol,'files':files}
        p={'retained_measurement':dependency}
        for key,name,value in [('retained_measurement_manifest','evidence/measurement_manifest.json',raw_json(dependency)),
            ('measurement_protocol_copy','evidence/measurement_protocol.json',protocol_raw),
            ('measurement_result_copy','evidence/result.json',(base/prefix/'result.json').read_bytes()),
            ('measurement_initial_state_copy','evidence/initial.json',(base/prefix/'full_states/initial.json').read_bytes()),
            ('measurement_closing_state_copy','evidence/closing.json',(base/prefix/'full_states/closing.json').read_bytes())]:
            p[key]=self.write(self.root,name,value)
        return p,base

    def verify_measurement(self,p):
        with mock.patch.object(R,'MEASUREMENT_PROTOCOL',self.measurement_protocol):return R.verify_retained_measurement(self.root,p)

    def replay(self,rows,plan=None):return R.replay_journal(b''.join(raw_json(r) for r in rows),plan or plan_fixture())

    def test_python310_syntax_imports_and_bridge_help_inert(self):
        paths=[SOURCE,ROOT/'scripts/build_u64_png_checked_step_v1.py',ROOT/'scripts/run_u64_png_checked_step_v1.py']
        original=builtins.__import__
        def blocked(name,*a,**k):
            if name.split('.')[0] in {'numpy','torch','PIL','tensorflow','jax'}:raise AssertionError('Forbidden scientific import')
            return original(name,*a,**k)
        with mock.patch('builtins.__import__',side_effect=blocked):
            for path in paths:
                ast.parse(path.read_text(),feature_version=(3,10))
                spec=importlib.util.spec_from_file_location('inert_'+path.stem,path);spec.loader.exec_module(importlib.util.module_from_spec(spec))
        result=subprocess.run([sys.executable,'-B',str(paths[-1]),'--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--supervise',result.stdout)
        shell=(SOURCE.parent/'run_png_checked.sh').read_text();self.assertIn('TMUX',shell);self.assertIn(R.STAGE,shell)

    def test_static_all469_own_sources_and_retained_verifier_called(self):
        sha=self.fixture();self.assertEqual(self.verify(sha),self.p)
        (self.parent/'assets/467.txt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Consumed parent static'):self.verify(sha)

    def test_missing_parent_never_uses_VM_alias(self):
        sha=self.fixture();(self.parent/'assets/467.txt').unlink()
        with self.assertRaises(FileNotFoundError):self.verify(sha)

    def test_source_tamper_bad_admission_and_unlisted_namespace(self):
        sha=self.fixture();self.write(self.root,'hidden.py',b'bad')
        with self.assertRaisesRegex(ValueError,'Unexpected own-stage'):self.verify(sha)
        (self.root/'hidden.py').unlink();(self.root/R.PACKAGE/'worker.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Own static'):self.verify(sha)

    def test_all288_real_measurement_dependency_payloads_verified(self):
        p,base=self.measurement_fixture();self.assertEqual(len(p['retained_measurement']['files']),288)
        self.assertEqual(self.verify_measurement(p),base.resolve())

    def test_measurement_payload_tamper_missing_and_local_alias_refused(self):
        p,base=self.measurement_fixture();name=R.MEASUREMENT_OUTPUT+'/calls/0009/gradients_raw_mae.npz'
        (base/name).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Actual retained VM measurement'):self.verify_measurement(p)
        (base/name).unlink();self.write(self.home,'extracted/'+name,b'synthetic undecoded NPZ bytes')
        with self.assertRaises(FileNotFoundError):self.verify_measurement(p)

    def test_measurement_manifest_cannot_drop_or_add_witness(self):
        p,base=self.measurement_fixture();del p['retained_measurement']['files'][R.MEASUREMENT_OUTPUT+'/calls/0009/gradients_raw_mae.npz']
        with self.assertRaisesRegex(ValueError,'subset namespace'):self.verify_measurement(p)

    def test_measurement_protocol_and_metadata_copies_are_exact(self):
        p,base=self.measurement_fixture();(self.root/p['measurement_result_copy']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Pinned file differs'):self.verify_measurement(p)

    def test_namespace_controls_links_and_nonfinite_JSON_refused(self):
        for name in ('a\nfile','a\x00file','a\\file','../file','a//file','a:stream','a/CON.txt','a.','é'):
            with self.subTest(name=name),self.assertRaises(ValueError):R.relative(name)
        for data in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}'):
            with self.subTest(data=data),self.assertRaises(ValueError):R.strict_json(data)

    def test_full445_ordered_operations_canonical_chain_and_action_returns(self):
        x=self.replay(journal_fixture(plan_fixture()));self.assertEqual(x['counters'],R.COUNTERS)
        self.assertEqual(x['completed_operation_pairs'],445);self.assertTrue(x['full_exact_plan_completed'])
        self.assertEqual(x['application_returns_by_action'],{'trial_apply':4,'trial_restore':4,'selected_commit':1})

    def test_noeligible388pair_prefix_has380forwards8writes_original_restored(self):
        x=self.replay(journal_fixture(plan_fixture())[:776]);self.assertEqual(x['completed_operation_pairs'],388)
        self.assertEqual(x['counters']['full_forward_returns'],380);self.assertEqual(x['counters']['parameter_applications'],8)
        self.assertEqual(x['counters']['selected_parameter_commits'],0);self.assertTrue(x['no_eligible_original_restored_prefix'])

    def test_pending_write_unknown_cannot_be_erased_by_restore(self):
        rows=journal_fixture(plan_fixture());first=next(i for i,r in enumerate(rows) if r['kind']=='parameter_application_reserved')
        x=self.replay(rows[:first+1]);self.assertTrue(x['operation_outcome_unknown']);self.assertEqual(x['counters']['parameter_applications'],0)
        restore=next(r for r in rows if r.get('action')=='trial_restore' and r['kind'].endswith('reserved'))
        with self.assertRaises(ValueError):self.replay(rows[:first+1]+[restore])

    def test_replay_rejects_wrong_chain_restore_destination_action_or_counter(self):
        original=journal_fixture(plan_fixture());idx=next(i for i,r in enumerate(original) if r['kind']=='parameter_application_returned')
        restore=next(i for i,r in enumerate(original) if r.get('action')=='trial_restore' and r['kind'].endswith('returned'))
        for index,mutation in [(idx,{'actual_before_head_canonical_sha256':'0'*64}),(restore,{'actual_after_head_canonical_sha256':'1'*64}),
                               (idx,{'action':'selected_commit'}),(idx,{'counters':dict.fromkeys(R.COUNTERS,0)})]:
            rows=copy.deepcopy(original);rows[index].update(mutation)
            with self.subTest(index=index,mutation=mutation),self.assertRaises(ValueError):self.replay(rows)

    def test_selected_commit_cannot_form_unregistered_candidate(self):
        rows=journal_fixture(plan_fixture());index=next(i for i,r in enumerate(rows) if r.get('action')=='selected_commit' and r['kind'].endswith('returned'))
        rows[index]['actual_after_head_canonical_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'recorded candidate'):self.replay(rows)

    def test_new_gradient_optimizer_and_unknown_tokenless_operations_refused(self):
        for kind in ('gradient_query_reserved','optimizer_constructed','backward_returned','inference_reserved'):
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.replay([{'kind':kind}])

    def test_plan_queries_wrong_slots_and_truncated_journal_refused(self):
        p=plan_fixture();p[0]['gradient_queries']=['raw_mae']
        with self.assertRaises(ValueError):R.operation_sequence(p)
        p=plan_fixture();p[0]['slot']=2
        with self.assertRaises(ValueError):R.operation_sequence(p)
        with self.assertRaises(ValueError):R.replay_journal(raw_json(journal_fixture(plan_fixture())[0])[:-1],plan_fixture())

    def test_status_full_or_pending_atomic_receipt_reconciled(self):
        rows=journal_fixture(plan_fixture());counts=self.replay(rows)['counters']
        self.write(self.root,R.OUTPUT+'/events.jsonl',b''.join(raw_json(r) for r in rows))
        self.write(self.root,R.OUTPUT+'/status.json',raw_json({'protocol_sha256':'a'*64,'counters':counts,'unknown_reservations':[],'operation_outcome_unknown':False}))
        self.assertFalse(R.status_snapshot(self.root,'a'*64,{'plan':plan_fixture()})['operation_outcome_unknown'])
        self.write(self.root,R.OUTPUT+'/status.json.pending',b'{}')
        self.assertTrue(R.status_snapshot(self.root,'a'*64,{'plan':plan_fixture()})['operation_outcome_unknown'])

    def test_missing_status_not_zero_and_timeout_propagates(self):
        self.assertIsNone(R.status_snapshot(self.root,'a'*64)['counters'])
        self.write(self.root,R.OUTPUT+'/status.json',b'bad')
        self.assertIsNone(R.status_snapshot(self.root,'a'*64)['counters'])
        with mock.patch.object(R,'read_json',side_effect=TimeoutError('timer')):
            with self.assertRaises(TimeoutError):R.status_snapshot(self.root,'a'*64)

    def test_current_stdlib_contract_and_worker_ledger_API(self):
        from research.codeformer_u64_png_checked_step_v1 import contract,worker
        self.assertEqual(set(worker.COUNTER_KEYS),set(R.COUNTERS))
        for key,value in R.COUNTERS.items():self.assertEqual(contract.LIMITS[key],value)
        for key in ('STAGE','PARENT_STAGE','PACKAGE','OUTPUT','PREFIX','FORMAT','ENV'):self.assertEqual(getattr(contract,key),getattr(R,key))
        parent=(ROOT/'outputs'/R.PARENT_STAGE/'protocol.json').read_bytes();p=json.loads(parent)
        standard=(ROOT/'outputs'/R.PARENT_STAGE/p['parent_protocol']['path']).read_bytes()
        plan=contract.operation_plan(contract.select_cases(parent,standard));self.assertEqual(len(plan),445)
        self.assertEqual(contract.prospective_contract()['limits'],contract.LIMITS)
        rows=[];ledger=worker.OperationLedger(lambda row:rows.append(copy.deepcopy(row)));current=R.ORIGINAL_HEAD_SHA
        for kind,call,query in R.operation_sequence(plan):
            token=ledger.reserve(kind,**metadata(call));extra,current=returned_metadata(call,current);ledger.returned(token,**extra)
        x=self.replay(rows,plan);self.assertEqual(x['counters'],R.COUNTERS);self.assertFalse(x['operation_outcome_unknown'])

    def test_bridge_rejects_unpinned_sources_before_import(self):
        target=self.home/'bootstrap';(target/'scripts').mkdir(parents=True);(target/'research').mkdir()
        script=ROOT/'scripts/run_u64_png_checked_step_v1.py';(target/'scripts'/script.name).write_bytes(script.read_bytes())
        (target/'research/__init__.py').write_text("raise RuntimeError('UNPINNED_IMPORTED')\n")
        raw=raw_json({'files':{'research/__init__.py':payload(b'')}});(target/'protocol.json').write_bytes(raw)
        x=subprocess.run([sys.executable,'-B',str(target/'scripts'/script.name),'--root',str(target),'--protocol-sha',hashlib.sha256(raw).hexdigest(),'--run'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(x.returncode,0);self.assertIn('Sealed bootstrap source differs',x.stderr);self.assertNotIn('UNPINNED_IMPORTED',x.stderr)

    def test_run_mode_uses_parent_cwd_and_unchanged_worker_API_stub_only(self):
        worker=SimpleNamespace(__file__=str(self.root/R.PACKAGE/'worker.py'),run=mock.Mock(return_value=2))
        env={R.ENV+'SUPERVISED':'1',R.ENV+'PROTOCOL':'a'*64,R.ENV+'SUPERVISOR_PID':str(R.os.getppid())}
        with mock.patch.object(sys,'argv',['runtime','--root',str(self.root),'--protocol-sha','a'*64,'--run']),\
             mock.patch.object(R,'verify_static',return_value={}),mock.patch.object(R,'require_vm'),\
             mock.patch.object(R.importlib,'import_module',return_value=worker),mock.patch.dict(R.os.environ,env),mock.patch.object(R.os,'chdir') as cwd:
            self.assertEqual(R.main(),2);cwd.assert_called_once_with(self.parent);worker.run.assert_called_once_with(self.root,self.parent,{},'a'*64)

    def test_supervisor_unknown_cleanup_skips_export_known_stop_exports(self):
        limits={'initial_free_disk_bytes':8*1024**3,'external_seconds':1260,'export_external_seconds':330}
        worker={'exit_code':2,'spawn_returned':True,'cleanup':{'child_reaped':False,'process_group_absent':False}}
        with redirect_stdout(io.StringIO()),mock.patch.object(R,'require_vm'),mock.patch.object(R.signal,'getsignal'),mock.patch.object(R.signal,'signal'),mock.patch.object(R,'_child',return_value=worker) as child:
            self.assertEqual(R.supervise(self.root,self.parent,{'limits':limits,'plan':[]},'a'*64),2);self.assertEqual(child.call_count,1)
        second=self.home/'second';second.mkdir();worker['cleanup']={'child_reaped':True,'process_group_absent':True}
        with redirect_stdout(io.StringIO()),mock.patch.object(R,'require_vm'),mock.patch.object(R.signal,'getsignal'),mock.patch.object(R.signal,'signal'),mock.patch.object(R,'_child',side_effect=[worker,{'exit_code':0}]) as child:
            self.assertEqual(R.supervise(second,self.parent,{'limits':limits,'plan':[]},'a'*64),2);self.assertEqual(child.call_count,2)

    def test_watchdog_stub_resource_failure_and_success(self):
        child=SimpleNamespace(pid=12,poll=mock.Mock(return_value=None),wait=mock.Mock(return_value=0))
        usage={'free_disk_bytes':3*1024**3,'output_bytes':0,'current_RSS_bytes':1,'archive_bytes':0}
        with redirect_stdout(io.StringIO()),mock.patch.object(R.subprocess,'Popen',return_value=child),mock.patch.object(R,'external_usage',return_value=usage),\
             mock.patch.object(R,'_stop_group',return_value={'child_reaped':True,'process_group_absent':True}) as stop:
            x=R._child(self.root,['inert_stub'],'worker',10,{})
        self.assertEqual(x['exit_code'],125);stop.assert_called_once_with(child)

    def test_watchdog_stub_success_records_actual_samples(self):
        child=SimpleNamespace(pid=12,poll=mock.Mock(side_effect=[None,0]),wait=mock.Mock(return_value=0))
        usage={'free_disk_bytes':8*1024**3,'output_bytes':19,'current_RSS_bytes':20,'archive_bytes':0}
        with redirect_stdout(io.StringIO()),mock.patch.object(R.subprocess,'Popen',return_value=child),\
             mock.patch.object(R,'external_usage',return_value=usage),mock.patch.object(R,'_stop_group',return_value={'child_reaped':True,'process_group_absent':True}):
            x=R._child(self.root,['inert_stub'],'worker',10,{})
        self.assertEqual(x['exit_code'],0);self.assertEqual(x['resource_samples'],1);self.assertEqual(x['resource_maxima']['output_bytes'],19)

    def test_initial_disk_and_fresh_stage_before_any_spawn(self):
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'_child') as child,\
             mock.patch.object(R.shutil,'disk_usage',return_value=SimpleNamespace(free=6*1024**3)):
            with self.assertRaisesRegex(ValueError,'Initial8GiB'):R.supervise(self.root,self.parent,{'limits':{'initial_free_disk_bytes':8*1024**3}},'a'*64)
            child.assert_not_called()
        self.write(self.root,'run.log',b'')
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'_child') as child:
            with self.assertRaisesRegex(ValueError,'Fresh stage'):R.supervise(self.root,self.parent,{},'a'*64)
            child.assert_not_called()

    def test_export_Python310_USTAR_own_only_no_dependency_payloads(self):
        self.write(self.root,'source.txt',b'own');self.write(self.root,'protocol.json',b'{}\n')
        parent_raw=raw_json({'files':{'never_included':payload(b'parent')}});self.write(self.parent,'protocol.json',parent_raw)
        p={'files':{'source.txt':payload(b'own')},'plan':[],'retained_measurement':{'files':{'never_measurement':payload(b'kept')}}}
        with redirect_stdout(io.StringIO()),mock.patch.object(R,'require_vm'),mock.patch.object(R,'verify_static',return_value=p),\
             mock.patch.object(Path,'home',return_value=self.home),mock.patch.object(R,'PARENT',{'stage_name':R.PARENT_STAGE,**payload(parent_raw)}):
            result=R._export(self.root,self.parent,'a'*64)
        archive=self.home/(R.PREFIX+'-results.tar.gz')
        with tarfile.open(archive,'r:gz') as tar:self.assertEqual(set(tar.getnames()),{R.STAGE+'/source.txt',R.STAGE+'/protocol.json'})
        with gzip.open(archive,'rb') as stream:
            while stream.read(8192):pass
        self.assertEqual((self.home/(archive.name+'.sha256')).read_bytes(),(result['archive']['sha256']+'  '+archive.name+'\n').encode())
        self.assertFalse(result['parent_payload_included']);self.assertFalse(result['retained_measurement_payload_included'])

    def test_absolute_export_alarm_bounded_failure_receipt(self):
        fake=SimpleNamespace(ITIMER_REAL=0,SIGALRM=14,getitimer=mock.Mock(return_value=(0.,0.)),getsignal=mock.Mock(),signal=mock.Mock(),setitimer=mock.Mock())
        with mock.patch.object(R,'require_vm'),mock.patch.object(R,'signal',fake),mock.patch.object(R,'_export',side_effect=TimeoutError('inert_alarm')):
            with self.assertRaises(TimeoutError):R.export(self.root,self.parent,'a'*64)
        self.assertEqual([r.args[1] for r in fake.setitimer.call_args_list],[300,5,0])
        self.assertIn('inert_alarm',json.loads((self.root/'export_failure.json').read_bytes())['error'])

    def test_builder_review_requires_every_current_source(self):
        path=ROOT/'scripts/build_u64_png_checked_step_v1.py';spec=importlib.util.spec_from_file_location('builder_tests',path)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        source=self.write(self.home,'source.py',b'# synthetic source\n');review=self.home/'review.json'
        review.write_bytes(raw_json({'complete':True,'source_review_pass':True,'unresolved_blockers':[],'sources':[source]}))
        original=builder.pin
        with mock.patch.object(builder,'ROOT',self.home),mock.patch.object(builder,'SOURCES',('source.py',)),\
             mock.patch.object(builder,'pin',side_effect=lambda p:original(p,self.home)):
            self.assertTrue(builder.admission(review)['source_review_pass'])
            (self.home/'source.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source changed after approval'):builder.admission(review)


if __name__=='__main__':unittest.main()
