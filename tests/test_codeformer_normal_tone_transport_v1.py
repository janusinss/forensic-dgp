"""Bounded stdlib transfer/export/refusal fixtures; no neural or VM calls."""
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

from research.codeformer_normal_tone_v1 import transport as t
from research.codeformer_normal_tone_v1 import contract as c
from scripts import supervise_codeformer_normal_tone_v1 as supervisor

ROOT=Path(__file__).resolve().parents[1]


def put(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return {'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}


def encode(value):return (json.dumps(value)+'\n').encode('utf-8')


class TransferBoundary(unittest.TestCase):
    def fixture(self,failed=False):
        base=ROOT/'scratch'/('normal_tone_transport_'+uuid.uuid4().hex)
        root,adain,loss,home=[base/x for x in ('new','adain','loss','home')]
        for path in (root,adain,loss,home):path.mkdir(parents=True)
        adain_asset=put(adain/'old.pth',b'Never deserialize this synthetic fixture')
        ap={'files_sha256':{'old.pth':adain_asset['sha256']}}
        apin=put(adain/'protocol.json',encode(ap))
        adeps={'protocol.json':apin,'old.pth':adain_asset}
        cases=b'{"cases":[]}\n';cpin=put(loss/'cases.json',cases)
        assets={'initializer':{'path':'old.pth',**adain_asset}}
        lp={'files_sha256':{'cases.json':cpin['sha256']},'dependencies':adeps,'assets':assets}
        lpin=put(loss/'protocol.json',encode(lp))
        cache=put(loss/'cache.npy',b'saved-array opaque bytes, never loaded')
        control=put(loss/'prior_control.json',b'{}\n')
        manifest={'protocol_sha256':lpin['sha256'],'files_sha256':{'cache.npy':cache['sha256'],
            'prior_control.json':control['sha256']},'files_bytes':{'cache.npy':cache['bytes'],'prior_control.json':control['bytes']}}
        mpin=put(loss/'outputs/loss_response_v1/export_manifest.json',encode(manifest))
        ldeps={'protocol.json':lpin,'cases.json':cpin,'cache.npy':cache,'prior_control.json':control,
            'outputs/loss_response_v1/export_manifest.json':mpin}
        files={'cases.json':cases,'research/never_execute.py':b'raise RuntimeError("Forbidden execution")\n'}
        pins={name:put(root/name,payload) for name,payload in files.items()}
        p={'format':'fixture','outputs':t.OUTPUTS,'cases_file':'cases.json',
            'parent_roots':{'adain':{'protocol_sha256':apin['sha256']},'loss_response':{'protocol_sha256':lpin['sha256']}},
            'dependencies':{'adain':adeps,'loss_response':ldeps},
            'assets':{name:{'root':'adain',**value} for name,value in assets.items()},
            'files_sha256':{n:v['sha256'] for n,v in pins.items()},'files_bytes':{n:v['bytes'] for n,v in pins.items()}}
        ppin=put(root/'protocol.json',encode(p))
        completion={'protocol_sha256':ppin['sha256'],'complete':not failed,
            'counts':{'optimizer_step_attempts':2 if failed else 8},
            'actual_completed_disposable_Adam_steps':1 if failed else 8}
        put(root/t.OUTPUTS/('failure.json' if failed else 'completion.json'),encode(completion))
        put(root/t.OUTPUTS/'raw.npy',b'opaque synthetic output')
        for name in ('preflight.log','diagnostic.log','export_invocation.json'):
            put(root/'outputs/supervision'/name,b'{}\n')
        put(root/'outputs/supervision/run_receipt.json',encode({'worker_terminal':True,'protocol_sha256':ppin['sha256']}))
        put(root/'outputs/supervision/export.log',b'active log outside archive')
        return SimpleNamespace(root=root,adain=adain,loss=loss,home=home,p=p,sha=ppin['sha256'],
                               expected_loss=set(ldeps))

    @contextmanager
    def patches(self,f):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(t.c,'verify_protocol',side_effect=lambda p:p))
            stack.enter_context(mock.patch.object(t.c,'verify_cases',side_effect=lambda p:p))
            stack.enter_context(mock.patch.object(t.c,'resolve_parents',return_value={'adain':f.adain,'loss_response':f.loss}))
            stack.enter_context(mock.patch.object(t.c,'required_loss_response_paths',return_value=f.expected_loss))
            stack.enter_context(mock.patch.object(t.c,'manual_root',side_effect=lambda p:Path(p)))
            stack.enter_context(mock.patch.object(t,'_process_rows',return_value=[]))
            stack.enter_context(mock.patch.object(Path,'home',return_value=f.home))
            stack.enter_context(mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=32*1024**3)))
            yield

    def rewrite(self,f):f.sha=put(f.root/'protocol.json',encode(f.p))['sha256']

    def test_two_roots_hash_checked_readonly(self):
        f=self.fixture();before={str(p):p.stat().st_mtime_ns for p in f.root.rglob('*') if p.is_file()}
        with self.patches(f):r=t.verify_transfer(f.root,f.sha)
        self.assertTrue(r['readonly']);self.assertEqual(r['dependencies_verified'],{'adain':2,'loss_response':5})
        self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in f.root.rglob('*') if p.is_file()})

    def test_each_parent_corruption_refused(self):
        for alias,name in [('adain','old.pth'),('loss','cache.npy')]:
            f=self.fixture();(getattr(f,alias)/name).write_bytes(b'changed')
            with self.subTest(alias=alias),self.patches(f),self.assertRaises(ValueError):t.verify_transfer(f.root,f.sha)

    def test_omitted_control_and_root_substitution_refused(self):
        f=self.fixture();del f.p['dependencies']['loss_response']['prior_control.json'];self.rewrite(f)
        with self.patches(f),self.assertRaisesRegex(ValueError,'namespace'):t.verify_transfer(f.root,f.sha)
        f=self.fixture();f.p['parent_roots']['loss_response']['protocol_sha256']='0'*64;self.rewrite(f)
        with self.patches(f),self.assertRaisesRegex(ValueError,'parent protocol'):t.verify_transfer(f.root,f.sha)

    def test_unbound_code_and_wrong_protocol_refused(self):
        f=self.fixture()
        with self.patches(f),self.assertRaises(ValueError):t.verify_transfer(f.root,'0'*64)
        put(f.root/'unbound.py',b'')
        with self.patches(f),self.assertRaisesRegex(ValueError,'Unbound'):t.verify_transfer(f.root,f.sha)

    def test_named_asset_substitution_refused(self):
        f=self.fixture();f.p['assets']['initializer']['path']='protocol.json';self.rewrite(f)
        with self.patches(f),self.assertRaisesRegex(ValueError,'Named assets'):t.verify_transfer(f.root,f.sha)

    def test_canonical_paths(self):
        for name in ('../x','/x','a//b','a\\b','C:/x','a/NUL.txt','a/COM¹.txt','a/e\u0301','a/trailing.'):
            with self.subTest(name=name),self.assertRaises(ValueError):t.canonical(name)

    def test_live_worker_not_overridden_by_terminal_receipt(self):
        f=self.fixture()
        with self.patches(f),mock.patch.object(t,'_process_rows',return_value=[(123,'python -m '+t.MODULE+' --run')]),self.assertRaises(ValueError):
            t.export_results(f.root,f.sha)

    def test_lock_live_pid_refused(self):
        f=self.fixture();put(f.root/'normal_tone_v1.lock',encode({'pid':123}))
        with self.patches(f),mock.patch.object(t.os,'kill',return_value=None),self.assertRaises(ValueError):
            t.require_worker_terminal(f.root)

    def test_failure_export_preserves_actual_steps_and_namespace(self):
        f=self.fixture(failed=True)
        with self.patches(f):r=t.export_results(f.root,f.sha)
        self.assertTrue(r['failure_present']);self.assertEqual(r['actual_disposable_Adam_steps'],1)
        self.assertTrue((f.root/t.OUTPUTS/'failure.json').is_file())
        manifest=t.c.read_json(f.root/t.OUTPUTS/'export_manifest.json')
        self.assertNotIn('outputs/supervision/export.log',manifest['files_sha256'])
        self.assertTrue(all(not n.startswith(('old.pth','cache.npy')) for n in manifest['files_sha256']))
        checksum=f.home/(t.PREFIX+'-results.tar.gz.sha256')
        self.assertNotIn(b'\r',checksum.read_bytes());self.assertTrue(checksum.read_bytes().endswith(b'\n'))
        with self.patches(f),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_stale_protocol_terminal_refused(self):
        f=self.fixture();p=f.root/t.OUTPUTS/'completion.json';value=t.c.read_json(p);value['protocol_sha256']='0'*64;put(p,encode(value))
        with self.patches(f),self.assertRaisesRegex(ValueError,'protocol binding'):t.export_results(f.root,f.sha)

    def test_write_reserve_is_prospective(self):
        f=self.fixture()
        with self.patches(f),mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=4*1024**3)),self.assertRaises(ValueError):
            t.export_results(f.root,f.sha)

    def test_final_companion_exact_log_and_exclusive(self):
        f=self.fixture()
        with self.patches(f):exported=t.export_results(f.root,f.sha)
        finished={'complete':True,'stages':[]};put(f.root/'outputs/supervision/finished.json',encode(finished))
        final_log=b'export stdout exact final bytes\n';put(f.root/'outputs/supervision/export.log',final_log)
        with self.patches(f):r=supervisor.final_companion(f.root,[])
        record=t.c.read_json(Path(r['path']))
        self.assertEqual(base64.b64decode(record['export_log_base64']),final_log)
        self.assertEqual(record['archive_sha256'],exported['archive_sha256'])
        self.assertEqual(record['finished_json'],finished);self.assertFalse(record['model_qualified'])
        with self.patches(f),self.assertRaises(FileExistsError):supervisor.final_companion(f.root,[])


class ProtocolMetadataMutations(unittest.TestCase):
    """Real prior JSON pins only; no array/checkpoint bytes are opened."""
    @classmethod
    def setUpClass(cls):
        old=ROOT/'outputs/codeformer_loss_response_v1_return'
        prior=c.read_json(old/'protocol.json')
        returned=c.read_json(old/'outputs/loss_response_v1/export_manifest.json')
        imported=c.read_json(ROOT/'outputs/codeformer_loss_response_v1_return_review/return_import_v1.json')
        selected=c.required_loss_response_paths(prior,returned)
        cls.protocol={'format':c.FORMAT,'seed':20261015,'outputs':'outputs/normal_tone_v1',
            'parent_roots':deepcopy(c.ROOTS),'source_updates':deepcopy(c.SOURCE_UPDATES),
            'cases_file':'cases.json','snapshots':['original0','stopped50'],'branches':deepcopy(c.BRANCHES),
            'loss_weights':deepcopy(c.WEIGHTS),'optimizer':deepcopy(c.OPTIMIZER),
            'numerical':deepcopy(c.NUMERICAL),'objective':deepcopy(c.OBJECTIVE),
            'limits':deepcopy(c.LIMITS),'call_caps':deepcopy(c.COUNTS),
            'total_neural_forwards':224,'total_gradient_queries':48,
            'files_sha256':{'cases.json':imported['files_sha256']['cases.json']},
            'files_bytes':{'cases.json':imported['files_bytes']['cases.json']},
            'dependencies':{'adain':deepcopy(prior['dependencies']),
                'loss_response':{n:{'sha256':imported['files_sha256'][n],'bytes':imported['files_bytes'][n]}
                                 for n in selected}},
            'assets':{n:{'root':'adain',**v} for n,v in prior['assets'].items()},
            'cached_basis':c.cached_basis_paths(),'prior_controls':c.prior_controls(),
            'automatic_app_promotion':False,'model_qualified':False,'new_training_recipe':False,
            'export_candidate_checkpoint':False,'manual_tmux_required':True,
            'all_branches_disposable':True,'quality_gates_changed':False,'status':'PREPARED_UNRUN'}
        cls.cases=c.read_json(old/'cases.json')

    def test_real_metadata_reference_census(self):
        self.assertIs(c.verify_protocol(self.protocol),self.protocol)
        self.assertIs(c.verify_cases(self.cases),self.cases)
        self.assertEqual({k:len(v) for k,v in self.protocol['dependencies'].items()},
                         {'adain':2011,'loss_response':2619})

    def test_only_eight_metric_target_and_parity_pairs_selected(self):
        prefix='outputs/loss_response_v1/targets/'
        selected={n for n in self.protocol['dependencies']['loss_response'] if n.startswith(prefix)}
        refs={row['reference_id'] for row in self.cases['cases']}
        self.assertEqual(selected,{prefix+ref+'/'+suffix for ref in refs
                                   for suffix in ('clean256.npy','parity.json')})

    def test_resource_call_type_and_objective_mutations_refused(self):
        changes=[('seed',20261015.),('total_neural_forwards',224.),('total_gradient_queries',48.),
                 ('call_caps',dict(c.COUNTS,committed_training_updates=False)),
                 ('limits',dict(c.LIMITS,worker_seconds=1801)),
                 ('limits',dict(c.LIMITS,regular_file_cap=5000.)),
                 ('numerical',dict(c.NUMERICAL,fresh_all_coarse_cached_maximum_error=2e-5)),
                 ('objective',dict(c.OBJECTIVE,aggregation_divisor=8)),
                 ('objective',dict(c.OBJECTIVE,luminance_weights=[1/3,1/3,1/3]))]
        for key,value in changes:
            p=deepcopy(self.protocol);p[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):c.verify_protocol(p)

    def test_parent_manifest_control_and_promotion_mutations_refused(self):
        for key in ('automatic_app_promotion','model_qualified','new_training_recipe','export_candidate_checkpoint'):
            p=deepcopy(self.protocol);p[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):c.verify_protocol(p)
        p=deepcopy(self.protocol);p['dependencies']['adain'].pop(next(n for n in p['dependencies']['adain'] if n!='protocol.json'))
        with self.assertRaises(ValueError):c.verify_protocol(p)
        p=deepcopy(self.protocol);p['dependencies']['loss_response']['outputs/loss_response_v1/export_manifest.json']['sha256']='0'*64
        with self.assertRaises(ValueError):c.verify_protocol(p)
        p=deepcopy(self.protocol);p['prior_controls']['original0']['full']['branch_path']='outputs/other.json'
        with self.assertRaises(ValueError):c.verify_protocol(p)

    def test_case_join_order_and_eligibility_mutations_refused(self):
        for key,value in [('input_path',self.cases['cases'][1]['input_path']),
                          ('source_path','data/sources/tr_ffhq_other.png')]:
            document=deepcopy(self.cases);document['cases'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.verify_cases(document)
        document=deepcopy(self.cases);document['cases'][0],document['cases'][1]=document['cases'][1],document['cases'][0]
        with self.assertRaises(ValueError):c.verify_cases(document)
        document=deepcopy(self.cases)
        changed=not document['cases'][0]['parameters']['image_eligible']
        document['cases'][0]['parameters']['image_eligible']=changed
        document['cases'][0]['record']['parameters']['image_eligible']=changed
        with self.assertRaises(ValueError):c.verify_cases(document)


if __name__=='__main__':unittest.main()
