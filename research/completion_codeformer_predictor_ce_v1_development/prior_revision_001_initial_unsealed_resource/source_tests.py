"""Source preparation tests only. No model, Torch, optimizer, gradients or VM."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,math,sys,tempfile,types,unittest
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
contracts=load('research.completion_codeformer_predictor_ce_v1.contracts',HERE/'contracts.py')
topology=load('predictor_test_optimizer_topology',HERE/'optimizer_topology.py')
# Establish no executable parent package imports: owned helper import name only.
for name in ('research','research.completion_codeformer_predictor_ce_v1'):
 if name not in sys.modules:module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
budget_module=load('predictor_test_budget',HERE/'optimized_budget.py')
class Tests(unittest.TestCase):
 def source(self):return (HERE/'cctv_completion_codeformer_predictor_ce_v1_vm.py').read_text()
 def test_python310_all_sources(self):
  for p in HERE.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,10))
 def test_unreleased_execution_guard(self):
  source=self.source();tree=ast.parse(source);verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify');calls=[n for n in ast.walk(verify) if isinstance(n,ast.Call)];self.assertIn('UNBOUND_PREPARATORY_DESIGN_NOT_RELEASED',source);self.assertTrue(any(isinstance(v.func,ast.Name) and v.func.id=='need' for v in calls))
 def test_only_one_optimizer_constructor(self):
  tree=ast.parse(self.source());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Adam'];self.assertEqual(len(calls),1)
 def test_no_teacher_or_oracle_forward(self):
  source=self.source();self.assertNotIn('oracle_indices=',source);self.assertNotIn('hq_vqgan',source);self.assertNotIn('latent_loss=',source);self.assertNotIn('F.mse_loss(',source)
 def test_exact_metadata_partition(self):
  state=json.loads((ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results/outputs/state_before.json').read_text())['tensors'];selected=[n for n in state if contracts.predictor_name(n)];self.assertEqual(len(selected),114);self.assertEqual(len(state)-len(selected),383);self.assertEqual(sum(math.prod(state[n]['shape']) for n in selected),19450880)
 def test_balanced_fixed128_schedule(self):
  p=ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/balanced128_subset.json';self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),contracts.SUBSET_SHA);data=json.loads(p.read_text());rows=data['cases_in_reference_role_order'];flat=sum(data['schedule'],[]);self.assertEqual(len(rows),128);self.assertEqual(len(flat),len(set(flat)));self.assertEqual(len(data['schedule']),32);self.assertEqual({r['id'] for r in rows},set(flat));self.assertEqual(len({r['reference_id'] for r in rows}),32)
 def parent(self):return {'state':{77:{'step':32,'exp_avg':[1.],'exp_avg_sq':[2.]},8:{'step':32,'exp_avg':[3.],'exp_avg_sq':[4.]},99:{'step':32,'exp_avg':[5.],'exp_avg_sq':[6.]}},'param_groups':[{'params':[77,8,99],'lr':1e-5,'betas':(.9,.99),'eps':1e-8,'weight_decay':0}]}
 def test_projection_sparse_ids_and_reordered_active_names(self):
  parent=self.parent();old=deepcopy(parent);active=topology.project(parent,['encoder.a','ft.a','ft.b'],['ft.b','ft.a'],clone=deepcopy);self.assertEqual(active['state'][0],parent['state'][99]);self.assertEqual(active['state'][1],parent['state'][8]);self.assertEqual(active['param_groups'][0]['params'],[0,1]);self.assertEqual(parent,old);active['state'][0]['exp_avg'][0]=999.;self.assertEqual(parent,old)
 def test_merge_mixedsteps_preserves_inactive_and_parent(self):
  parent=self.parent();old=deepcopy(parent);active=topology.project(parent,['encoder.a','ft.a','ft.b'],['ft.b','ft.a'],clone=deepcopy)
  for v in active['state'].values():v['step']=64
  merged=topology.merge(parent,['encoder.a','ft.a','ft.b'],active,['ft.b','ft.a'],clone=deepcopy);self.assertEqual(merged['state'][77],old['state'][77]);self.assertEqual(merged['state'][8]['step'],64);self.assertEqual(merged['state'][99]['step'],64);self.assertEqual(merged['param_groups'],parent['param_groups']);self.assertEqual(parent,old);self.assertEqual(set(merged),{'state','param_groups'})
 def test_bad_optimizer_binding_refused(self):
  for names in (['encoder.a','ft.a','ft.a'],['encoder.a']):
   with self.assertRaises(ValueError):topology.ordered_ids(self.parent(),names)
  with self.assertRaises(ValueError):topology.project(self.parent(),['encoder.a','ft.a','ft.b'],['missing'],clone=deepcopy)
 def test_CE_meanall256_divisor_once_numpy_fixture(self):
  rng=np.random.default_rng(29);four=[]
  for _ in range(4):
   logits=rng.normal(size=(1,256,512)).astype(np.float32);labels=rng.integers(0,512,(1,256));l=logits.astype(np.float64);l-=l.max(-1,keepdims=True);ce=np.mean(np.log(np.exp(l).sum(-1))-np.take_along_axis(l,labels[...,None],axis=-1)[...,0]);four.append(ce)
  self.assertAlmostEqual(sum(v/4 for v in four),float(np.mean(four)),14);self.assertNotAlmostEqual(sum(v/16 for v in four),float(np.mean(four)),6)
 def gray(self):return {'records':[{'id':str(i),'style':'GRAY','mask_kind':contracts.KINDS[i%4],'GRAY_raw_mean_channel_spread':.02,'GRAY_PNG_mean_channel_spread':5.} for i in range(8)]}
 def test_gray_gate_all8macro4geometry(self):
  baseline=self.gray();now=deepcopy(baseline);self.assertTrue(contracts.compare_gray(now,baseline)['all_GRAY_macro_geometry_chroma_nonworse']);now['records'][0]['GRAY_PNG_mean_channel_spread']=6.;r=contracts.compare_gray(now,baseline);self.assertFalse(r['all_GRAY_macro_geometry_chroma_nonworse']);self.assertEqual({v['group'] for v in r['nonworse_failures']},{'macro','geometry/lower_face'})
 def test_gray_gate_missingcase_refused(self):
  now=self.gray();now['records'].pop()
  with self.assertRaises(ValueError):contracts.gray_style(now)
 def test_planned_counts(self):
  c=contracts.CONTRACT;self.assertEqual(sum(contracts.TYPED_CAPS.values()),384);self.assertEqual(8+192+56,256);self.assertEqual(32*4,128);self.assertEqual(192+64,256);self.assertEqual(256+4+128,388);self.assertEqual(4*40+8,168);self.assertEqual(c['cooperative_stop_seconds']+c['save_reserve_seconds'],c['worker_seconds'])
 def test_budget_exact_ownedextents_seek_and_overwrite_refusal(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1);p=out/'FIT_logits/a.npz'
   with b.open(p) as stream:stream.write(b'abcd');stream.seek(1);stream.write(b'X')
   self.assertEqual(b.total(),4);self.assertTrue(b.assert_ledger_consistent())
   with self.assertRaises(ValueError):b.open(p)
   with self.assertRaises(ValueError):b.open(out.parent/'escape.npz')
 def test_budget_terminalcap_before_write(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1)
   with b.open(out/'terminal_training_state.pth.partial',terminal=True) as stream:
    with self.assertRaises(ValueError):b.before_entry_write(stream.entry,contracts.CONTRACT['terminal_bytes']+1,terminal=True)
 def test_budget_workdeadline_and_savewindow(self):
  with tempfile.TemporaryDirectory() as temp:
   out=Path(temp);b=budget_module.OutputBudget(out,0,disk_free=lambda:20*1024**3,clock=lambda:1681)
   with self.assertRaises(ValueError):b.guard()
   b.guard(terminal=True)
 def test_no_scientific_execution(self):self.assertFalse(any(n=='torch' or n.startswith('torch.') for n in sys.modules))
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite);record={'format':'predictor-ce-v1-author-source-preparation-tests-v1','complete':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'local_neural_calls':0,'model_objects':0,'optimizer_objects':0,'backward_calls':0,'VM_calls':0,'prepared_source_only_not_runtime_or_packet_admission':True,'UTC':datetime.now(timezone.utc).isoformat()}
 target=HERE/'source_preparation_tests.json';target.write_bytes((json.dumps(record,indent=2)+'\n').encode());raise SystemExit(0 if result.wasSuccessful() else 1)
