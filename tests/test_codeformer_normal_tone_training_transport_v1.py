"""Stdlib source/protocol/transport/checkpoint preservation fixtures; no training."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

from research.codeformer_normal_tone_training_v1 import contract as c,transport as t
from scripts import supervise_codeformer_normal_tone_training_v1 as supervisor

ROOT=Path(__file__).resolve().parents[1]


def encode(value):return (json.dumps(value,indent=2)+'\n').encode()


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


class TrainingTransport(unittest.TestCase):
    def fixture(self,terminal='failure'):
        base=ROOT/'scratch'/('normal_tone_training_transport_'+uuid.uuid4().hex)
        root,parent,home=[base/n for n in ('new','parent','home')]
        for p in (root,parent,home):p.mkdir(parents=True)
        deps={}
        for name in ('references.json','comparison_cases.json','training_parity_v1.json','hq_cache_parity_v1.json'):
            deps[name]=put(parent/name,b'{}\n')
        assets={'initializer':{'path':'old.pth',**put(parent/'old.pth',b'opaque never-load fixture')}}
        old={'files_sha256':{n:r['sha256'] for n,r in deps.items()},'assets':assets,**c.PARENT_CONTROLS}
        deps['old.pth']={k:assets['initializer'][k] for k in ('sha256','bytes')}
        dpin=put(parent/'protocol.json',encode(old));deps['protocol.json']=dpin
        source=put(root/'research/new.py',b'raise RuntimeError("never execute")\n')
        p={'outputs':t.OUTPUTS,'files_sha256':{'research/new.py':source['sha256']},
            'files_bytes':{'research/new.py':source['bytes']},'dependencies':deps,
            'parent_dir':'parent','parent_protocol_sha256':dpin['sha256'],'assets':assets,
            'data_manifest':'references.json','comparison_cases':'comparison_cases.json',**c.PARENT_CONTROLS}
        ppin=put(root/'protocol.json',encode(p));counts=dict.fromkeys(c.COUNTS,0)
        counts.update(optimizer_updates=5,optimizer_step_attempts=6,backward_calls=21,candidate_training_forwards=21)
        value={'protocol_sha256':ppin['sha256'],'complete':False,'completed_updates':5,
               'counts':counts,'optimizer_inflight':True}
        if terminal=='success':
            value.update(complete=True,completed_updates=50,optimizer_inflight=False)
            value['counts']={**c.COUNTS,'optimizer_updates':50}
        if terminal is not None:
            put(root/t.OUTPUTS/('failure.json' if terminal=='failure' else 'run_results.json'),encode(value))
        put(root/c.CHECKPOINT_PATHS[2],b'actual synthetic stopped checkpoint opaque bytes')
        if terminal=='success':
            for name in c.CHECKPOINT_PATHS[:2]:put(root/name,b'synthetic candidate endpoint checkpoint opaque bytes')
        put(root/(c.CHECKPOINT_PATHS[2]+'.tmp-'+'a'*32),b'partial checkpoint retained')
        for name in ('preflight.log','diagnostic.log','export_invocation.json'):put(root/'outputs/supervision'/name,b'{}\n')
        put(root/'outputs/supervision/run_receipt.json',encode({'worker_terminal':True,'protocol_sha256':ppin['sha256']}))
        put(root/'outputs/supervision/export.log',b'active export log not archived')
        return SimpleNamespace(root=root,parent=parent,home=home,p=p,sha=ppin['sha256'])

    @contextmanager
    def patches(self,f):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(c,'verify_protocol',side_effect=lambda p:p))
            stack.enter_context(mock.patch.object(c,'resolve_parent',return_value=f.parent))
            stack.enter_context(mock.patch.object(c,'manual_root',side_effect=lambda p:Path(p)))
            stack.enter_context(mock.patch.object(t,'_process_rows',return_value=[]))
            stack.enter_context(mock.patch.object(Path,'home',return_value=f.home))
            stack.enter_context(mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=32*1024**3)))
            yield

    def test_transfer_readonly_and_does_not_execute_source(self):
        f=self.fixture();before={str(p):p.stat().st_mtime_ns for p in f.root.rglob('*') if p.is_file()}
        with self.patches(f):r=t.verify_transfer(f.root,f.sha)
        self.assertEqual(r['dependencies_verified'],6);self.assertTrue(r['readonly'])
        self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in f.root.rglob('*') if p.is_file()})

    def test_changed_parent_unbound_source_and_named_asset_refused(self):
        f=self.fixture();(f.parent/'old.pth').write_bytes(b'changed')
        with self.patches(f),self.assertRaises(ValueError):t.verify_transfer(f.root,f.sha)
        f=self.fixture();put(f.root/'unbound.py',b'')
        with self.patches(f),self.assertRaises(ValueError):t.verify_transfer(f.root,f.sha)
        f=self.fixture();f.p['assets']={};f.sha=put(f.root/'protocol.json',encode(f.p))['sha256']
        with self.patches(f),self.assertRaises(ValueError):t.verify_transfer(f.root,f.sha)

    def test_failure_export_keeps_actual_commits_and_partial_checkpoint(self):
        f=self.fixture()
        with self.patches(f):r=t.export_results(f.root,f.sha)
        self.assertTrue(r['failure_present']);self.assertFalse(r['complete_training'])
        self.assertEqual(r['committed_training_updates'],5)
        self.assertTrue(r['worker_recorded_optimizer_inflight_at_failure'])
        self.assertEqual(r['candidate_checkpoint_members'],[c.CHECKPOINT_PATHS[2]])
        self.assertEqual(len(r['incomplete_checkpoint_temporary_members']),1)
        manifest=c.read_json(f.root/t.OUTPUTS/'export_manifest.json')
        self.assertIn(c.CHECKPOINT_PATHS[2],manifest['files_sha256'])
        self.assertIn(c.CHECKPOINT_PATHS[2]+'.tmp-'+'a'*32,manifest['files_sha256'])
        self.assertNotIn('old.pth',manifest['files_sha256'])
        self.assertNotIn('outputs/supervision/export.log',manifest['files_sha256'])
        with self.patches(f),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_success_requires50updates_and_all_caps(self):
        f=self.fixture('success')
        with self.patches(f):r=t.export_results(f.root,f.sha)
        self.assertTrue(r['complete_training']);self.assertEqual(r['committed_training_updates'],50)
        f=self.fixture('success');path=f.root/t.OUTPUTS/'run_results.json';v=c.read_json(path);v['counts']['gradient_queries']=9;put(path,encode(v))
        with self.patches(f),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_missing_worker_receipt_retains_unknown_actual_count(self):
        f=self.fixture(None)
        with self.patches(f):r=t.export_results(f.root,f.sha)
        self.assertIsNone(r['committed_training_updates']);self.assertTrue(r['training_updates_unknown_without_worker_receipt'])

    def test_arbitrary_checkpoint_and_pickle_refused(self):
        for name in (t.OUTPUTS+'/arbitrary.pt',t.OUTPUTS+'/model.pth',t.OUTPUTS+'/unsafe.pkl',
                     t.OUTPUTS+'/model.pth.tmp-'+'a'*32):
            f=self.fixture();put(f.root/name,b'opaque')
            with self.subTest(name=name),self.patches(f),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_live_worker_and_live_lock_refuse_export(self):
        f=self.fixture()
        with self.patches(f),mock.patch.object(t,'_process_rows',return_value=[(123,'python -m '+t.MODULE+' --run')]),self.assertRaises(ValueError):t.export_results(f.root,f.sha)
        put(f.root/'normal_tone_training_v1.lock',encode({'pid':123}))
        with self.patches(f),mock.patch.object(t.os,'kill',return_value=None),self.assertRaises(ValueError):t.require_worker_terminal(f.root)

    def test_stale_protocol_and_reserve_refused(self):
        f=self.fixture();path=f.root/t.OUTPUTS/'failure.json';v=c.read_json(path);v['protocol_sha256']='0'*64;put(path,encode(v))
        with self.patches(f),self.assertRaises(ValueError):t.export_results(f.root,f.sha)
        f=self.fixture()
        with self.patches(f),mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=4*1024**3)),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_final_companion_closes_checkpoint_count_and_exact_log(self):
        f=self.fixture()
        with self.patches(f):t.export_results(f.root,f.sha)
        put(f.root/'outputs/supervision/finished.json',encode({'complete':True,'stages':[]}))
        exact=b'closed export stdout\n';put(f.root/'outputs/supervision/export.log',exact)
        with self.patches(f):r=supervisor.final_companion(f.root,[])
        record=c.read_json(r['path']);self.assertEqual(base64.b64decode(record['export_log_base64']),exact)
        self.assertEqual(record['committed_training_updates'],5);self.assertTrue(record['checkpoint_exported'])
        self.assertFalse(record['model_qualified'])
        with self.patches(f),self.assertRaises(FileExistsError):supervisor.final_companion(f.root,[])

    def test_prospective_checkpoint_gate_geometry(self):
        self.assertEqual(t.checkpoint_kind(c.CHECKPOINT_PATHS[0]),'final')
        self.assertEqual(t.checkpoint_kind(c.CHECKPOINT_PATHS[2]+'.tmp-'+'f'*32),'temporary')
        for name in ('/outside/checkpoint.pt',t.OUTPUTS+'/snapshot_0051/checkpoint.pt',c.CHECKPOINT_PATHS[2]+'.tmp-bad'):
            self.assertIsNone(t.checkpoint_kind(name))

    def test_native_clean_target_PNG_narrow_exception_and_wrong_path_refusal(self):
        retained=ROOT/'outputs/codeformer_adain_gradient_isolation_v1_return/outputs/adain_gradient_isolation_v1/first_training_example/clean512.png'
        actual=retained.read_bytes();self.assertEqual(len(actual),388678)
        name=t.OUTPUTS+'/first_training_example/clean512.png'
        f=self.fixture();pin=put(f.root/name,actual)
        with self.patches(f):r=t.export_results(f.root,f.sha)
        self.assertTrue(r['complete'])
        self.assertEqual(c.read_json(f.root/t.OUTPUTS/'export_manifest.json')['files_sha256'][name],pin['sha256'])
        for wrong in (t.OUTPUTS+'/first_training_example/other.png',
                      t.OUTPUTS+'/route_witness/u0003_s1/native.png'):
            f=self.fixture();put(f.root/wrong,actual)
            with self.subTest(name=wrong),self.patches(f),self.assertRaisesRegex(ValueError,'PNG file cap'):t.export_results(f.root,f.sha)
        f=self.fixture();put(f.root/name,b'x'*(3*512*512+4097))
        with self.patches(f),self.assertRaisesRegex(ValueError,'PNG file cap'):t.export_results(f.root,f.sha)


class TrainingProtocol(unittest.TestCase):
    def protocol(self):
        deps=c.read_json(ROOT/'outputs/codeformer_loss_response_v1_return/protocol.json')['dependencies']
        name='instructions/CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md'
        files={**{'research/codeformer_normal_tone_training_v1/'+n:'1'*64 for n in
                  ('__init__.py','contract.py','transport.py','worker.py','model.py')},
               'research/__init__.py':'1'*64,'scripts/supervise_codeformer_normal_tone_training_v1.py':'1'*64,name:'1'*64}
        return {'format':c.FORMAT,'seed':20261015,'data_manifest':'references.json','comparison_cases':'comparison_cases.json',
            'outputs':c.OUTPUTS,'export_prefix':c.PREFIX,'parent_dir':c.PARENT_NAME,'parent_protocol_sha256':c.PARENT_PROTOCOL_SHA,
            'training':deepcopy(c.TRAINING),'evaluation':deepcopy(c.EVALUATION),'limits':deepcopy(c.LIMITS),
            'gradient_route':deepcopy(c.GRADIENT_ROUTE),'objective':deepcopy(c.OBJECTIVE),
            'normal_formula_prerequisite':deepcopy(c.NORMAL_FORMULA_PREREQUISITE),
            'assets':deepcopy(c.ASSETS),**deepcopy(c.PARENT_CONTROLS),'dependencies':deps,
            'files_sha256':files,'files_bytes':{n:0 for n in files},'design':{'path':name,'sha256':'1'*64},
            'status':'PREPARED_UNRUN','manual_tmux_required':True,'new_training_recipe':True,
            'actual_candidate_checkpoints_required':True,'model_qualified':False,'automatic_app_promotion':False,
            'automatic_training':False,'quality_gates_changed':False,'resume_authorized':False,'FINAL_used':False,'cached_gradient_reuse':False}

    def test_real_parent_metadata_and_source_derived_counts(self):
        p=self.protocol();self.assertIs(c.verify_protocol(p),p)
        self.assertEqual(c.expected_call_caps(),c.COUNTS);self.assertEqual(sum(c.COUNTS.values()),2075)
        self.assertEqual(sum(v for k,v in c.COUNTS.items() if k.endswith('_forwards')),1614)
        self.assertEqual(c.COUNTS['gradient_queries'],11)

    def test_schedule_rate_gate_formula_and_dependency_mutations_refused(self):
        p=self.protocol()
        for path,value in [(('training','max_updates'),51),(('training','warmup_updates'),0),
                           (('evaluation','mean_clear_mae_delta_stop'),.02),(('objective','normal_slots'),[2,4]),
                           (('normal_formula_prerequisite','absolute_tolerance'),1e-3),
                           (('limits','worker_seconds'),3000.),(('training','resume'),True)]:
            changed=deepcopy(p);changed[path[0]][path[1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):c.verify_protocol(changed)
        for key in ('automatic_training','cached_gradient_reuse','FINAL_used','model_qualified'):
            changed=deepcopy(p);changed[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):c.verify_protocol(changed)
        changed=deepcopy(p);name=next(n for n in changed['dependencies'] if n!='protocol.json');changed['dependencies'][name]['sha256']='0'*64
        with self.assertRaises(ValueError):c.verify_protocol(changed)


if __name__=='__main__':unittest.main()
