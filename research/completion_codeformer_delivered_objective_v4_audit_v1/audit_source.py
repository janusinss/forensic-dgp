"""Independent source/packet audit; no model, gradient, optimizer or VM execution."""
from __future__ import annotations
import argparse,ast,contextlib,copy,hashlib,importlib.abc,importlib.util,io,json,sys,time
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2]
DESIGN_SHA='c565e8aa1362378b45c948463fe6018562858ff217f7f66b375161b6117ca289'
DESIGN_ADMISSION_SHA='888ae1368f62b7e150e152a6b16fc46aa47f1fb0fd8efd1a3e385db597190aea'
SOURCE_PATHS=('scripts/cctv_completion_codeformer_delivered_objective_v4_vm.py','research/completion_codeformer_delivered_objective_v4/contracts.py','research/completion_codeformer_delivered_objective_v4/optimized_budget.py','scripts/export_completion_codeformer_delivered_objective_v4.py','scripts/run_completion_codeformer_delivered_objective_v4.sh','scripts/prepare_completion_codeformer_delivered_objective_v4.py','research/completion_codeformer_delivered_objective_v4/delivered_loss.py')

DEVELOPMENT=ROOT/'research/completion_codeformer_delivered_objective_v4_development'
def source_path(name):
 if name in SOURCE_PATHS[:3] or name==SOURCE_PATHS[-1]:return DEVELOPMENT/Path(name).name
 return ROOT/name

def need(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as s:
  for block in iter(lambda:s.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
 p=Path(path);return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def named(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def callname(node):
 if isinstance(node,ast.Name):return node.id
 if isinstance(node,ast.Attribute):return callname(node.value)+'.'+node.attr
 return ''
class BlockScientific(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','PIL'):raise RuntimeError('Unexpected scientific import '+fullname)

def audit(stage,expected,output):
 started=time.monotonic();stage=Path(stage).resolve(strict=True);protocol_path=stage/'protocol.json';checks=[]
 def check(ok,label,details=None):
  need(ok,label);checks.append({'check':label,'pass':True,'details':details})
 check(sha(protocol_path)==expected,'Frozen protocol SHA')
 p=read(protocol_path);files={r['path']:r for r in p['files']}
 check(len(files)==len(p['files']) and len(files)>0,'Nonempty unique manifest')
 actual={q.relative_to(stage).as_posix() for q in stage.rglob('*') if q.is_file()}
 check(actual==set(files)|{'protocol.json'},'Exact staged namespace')
 payload=protocol_path.stat().st_size
 for name,row in files.items():
  rel=PurePosixPath(name);check(not rel.is_absolute() and '..' not in rel.parts and ':' not in name and '\\' not in name and str(rel)==name,'Safe member '+name)
  path=stage/name;check(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Exact payload '+name);payload+=row['bytes']
 check(payload<256*1024**2,'Execution payload under256MiB',{'bytes':payload,'files':len(files)})
 for name in SOURCE_PATHS:check((stage/name).read_bytes()==source_path(name).read_bytes(),'Staged/canonical source exact '+name)
 for name in SOURCE_PATHS:
  if name.endswith('.py'):
   t=ast.parse((stage/name).read_text(encoding='utf-8-sig'),feature_version=(3,10));check(not any(isinstance(n,ast.Attribute) and n.attr=='file_digest' for n in ast.walk(t)),'Python3.10 syntax/chunked hash '+name)
 check(sha(ROOT/'research/completion_codeformer_delivered_objective_v4_design_v1/root_design.json')==DESIGN_SHA,'Root design binding')
 check(sha(ROOT/'outputs/completion_codeformer_v3_next_objective_independent_review_v1/independent_V4_design_admission.json')==DESIGN_ADMISSION_SHA,'Independent design binding')
 source=(stage/SOURCE_PATHS[0]).read_text(encoding='utf-8');tree=ast.parse(source);worker=named(tree,'worker');verify=named(tree,'verify');main=named(tree,'main');save=named(worker,'save_state');evaluate=named(worker,'evaluate');forward=named(worker,'forward')
 calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call)];steps=[n for n in calls if callname(n.func).endswith('.step')];zeros=[n for n in calls if callname(n.func).endswith('.zero_grad')]
 check(len(steps)==1 and callname(steps[0].func)=='fusion_optimizer.step','Only fusion Adam step call')
 check(len(zeros)==2 and all(callname(n.func)=='fusion_optimizer.zero_grad' for n in zeros),'No code optimizer zero_grad')
 check(not any(callname(n.func) in ('net.encoder','net.quantize','F.cross_entropy','F.mse_loss') for n in calls),'No teacher/code CE/latent neural calls')
 check("active(fusion_groups)" in source and "len(code_parameters)==278" in source and "len(fusion_parameters)==54" in source and "==443" in source,'278code+165prior frozen;54fusion source checks')
 check("all(parameter.grad is None for parameter in code_parameters)" in source,'Frozen predictor receives no gradients')
 check("tree_digest(code_optimizer.state_dict())==code_optimizer_baseline" in source and "frozen_digest()==frozen_all_state" in source,'Frozen443 and codeAdam checked after learning')
 check("parent['optimizer_parameter_names']==optimizer_parameter_names" in source and "optimizer.load_state_dict(state)" in source and "torch.equal(restored[key].cpu(),prior[key].cpu())" in source and "==32" in source,'Named full R1 Adam restoration and moment equality')
 check("torch.load(parent_path,map_location='cpu',weights_only=True)" in source and "torch.load(checkpoint,map_location='cpu',weights_only=True)" in source,'Safe explicit parent and original loads')
 loop=next(n for n in worker.body if isinstance(n,ast.Try))
 check(source.index('Historical R1 baseline incompatible; retain stop before updates')<source.index('torch.set_rng_state(resume_rng')<source.index('for cycle,case_ids in enumerate(schedule,1):'),'48case compatibility then RNG then optimization order')
 check("len(compatibility_rows)==48" in source and "r['max_raw256_error']<=1e-6 and r['PNG_bytes_exact']" in source and "updates_before" in source,'All48 raw1e-6 and exactPNG admission before update')
 check("input_stride_exact" in source and "# Retain all four paired calls before aggregate parity assertions." in source and "max_raw256_error']<=1e-6" in source,'All4 canonicalparity pairs retained beforeunchangedgate')
 check("expected_rng_hashes==restored_rng_hashes" in source and "FullNumPyRNG restoration differs" in source and "len(restored_cuda)==len(resume_rng['CUDA'])==1" in source,'Fullfourfamily RNG equality/hash receipts')
 check("'forward_slots':528+counts['forward_slots']" in source and "'actual_whole_model_forward_calls':448+model_forwards['calls']" in source and "'optimizer_steps_total':64+counts['fusion_optimizer_steps']" in source,'Separate actual parent/new/cumulative counters')
 check("'forward_slots':480,'backward_calls':256,'code_optimizer_steps':0,'fusion_optimizer_steps':64,'empty_bypasses':168" in source and "model_forwards['calls']==480" in source,'Successful count assertions exactly480/256/64/168')
 check("clear_screen['prospective_numeric_screen_pass'] and challenge_screen['prospective_numeric_screen_pass']" in source and "conditions=True" in source,'Separate clear/challenge gates both required')
 check("torch.equal(out_tensor*(1-m),rgb*(1-m))" in source and "np.array_equal(array[~support],observed[~support])" in source,'Exact raw/PNG outside support preservation')
 check("model_forwards['calls']==before_calls" in source and "counts['empty_bypasses']+=1" in source,'Exact noNN empty controls')
 check("raw,_,_=forward('fit_RGB'" in source and "target=clean[case['teacher_key']][0]" in source and "prediction,rgb_loss,loss=delivered_hole_loss(raw,rgb,m,target,interpolate=F.interpolate,microbatches=4)" in source and "clip_grad_norm_(fusion_parameters,1.,error_if_nonfinite=True)" in source,'Exact clean256 delivered-hole loss and unchanged clip')
 check("seam=(" not in source and "loss=(rgb_loss+seam)" not in source and "clipping=clipping_diagnostic(raw,m512)" in source,'No separate seam loss; clipping remains detached diagnostic')
 check("return delivered_float(raw,rgb,m,interpolate=F.interpolate)" in source,'Training/evaluation share app delivery helper')
 graph=ROOT/'research/completion_codeformer_delivered_objective_v4_audit_v1/independent_loss_graph.json'
 check(sha(graph)=='084b1c14d9539505a1cc2ac47bd1334b9ba4a4ec968251fedbad82973d54b1cc' and sha(stage/SOURCE_PATHS[-1])==read(graph)['helper']['sha256'],'Exact tested helper loss graph binding')
 # Independently compare critical recovery, budget, gate and supervisor ASTs to sealed V3.
 v3source=(ROOT/'scripts/cctv_completion_codeformer_fusion_only_v3_vm.py').read_text();v3tree=ast.parse(v3source);v3worker=named(v3tree,'worker')
 class NormalizeStrings(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str):node.value=node.value.replace('fusion_only_v3','delivered_objective_v4').replace('fusion-only-v3','delivered-objective-v4')
   return node
 for func in ('save_state','guard','forward','empty_controls','phase_start','phase_end','tree_digest','cpu_copy','tensor_bytes','evaluate'):
  old=NormalizeStrings().visit(copy.deepcopy(named(v3worker,func)));new=named(worker,func)
  check(ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),'Critical inherited V3 AST exact after namespace changes '+func)
 oldmain=NormalizeStrings().visit(copy.deepcopy(named(v3tree,'main')))
 check(ast.dump(oldmain,include_attributes=False)==ast.dump(main,include_attributes=False),'Inherited supervisor/CLI AST exact after namespace changes')
 save_source=ast.get_source_segment(source,save)
 check("check_finite=False" in save_source and "retained_failure_gates" in save_source and "failed_frozen_prior_state" in save_source,'Failure state/nonfinite/priorchanges retained beforequalification')
 for key in ('trainable_state','code_optimizer','fusion_optimizer','partial_accumulated_gradients','torch_CPU_RNG','CUDA_RNG','python_random_state','numpy_random_state','sampler_schedule','progress','cumulative_counts'):
  check(key in save_source,'Complete terminal includes '+key)
 check("pre_save_receipt.json" in save_source and save_source.index("pre_save_receipt.json")<save_source.index('budget.open(partial,terminal=True)'),'Pre-save receipt before serialization')
 check(all(x in save_source for x in ('terminal_CPU_copy_and_assembly','terminal_serialization','flush_fsync_seconds','terminal_hash_and_atomic_commit','write_callbacks','planned_tensor_payload_bytes','remaining_hard_seconds')),'Assembly/serialize/fsync/commit timings and bytecallbacks')
 check('partial.unlink()' in save_source and save_source.index('os.link(partial,path)')<save_source.index('partial.unlink()'),'Successfulpartial alias only removed afterexclusivecommit')
 check('subprocess.PIPE' in source and "start_new_session=True" in source and "CONTRACT['trainer_log_bytes']-captured" in source and "os.killpg(child.pid,sig)" in source,'Supervisor boundedPIPE andownedprocessgroup')
 check('time.monotonic()-started>=1800' in source and "'external_cap_seconds':1800" in source and "Need7GiB" in source and "'g2-standard-4'" in source and "'L4'" in source,'Samehardcap/freefloor/manualL4root guards')
 check("not budget.terminal_committed" in ast.get_source_segment(source,forward),'No NN after terminal commit')
 # Pure import under a scientific-import blocker, using actual staged worker.
 blocker=BlockScientific();sys.meta_path.insert(0,blocker)
 try:
  spec=importlib.util.spec_from_file_location('independent_delivered_v4_source',stage/SOURCE_PATHS[0]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 finally:sys.meta_path.remove(blocker)
 c=module.CONTRACT
 check(c==p['contract'],'Actual pure-import contract equals sealed protocol')
 check(c['objective_resolution']==256 and c['clean_target_index']==0 and c['microbatch_divisor_once']==4 and c['additive_boundary_loss'] is False and c['training_PNG_rounding'] is False and c['saturation_gradient_surrogate'] is False,'Exact prospective objective contract; no seam/PNG/STE')
 check(c['forward_slots_planned']==c['whole_model_forwards_planned']==480 and c['forward_slots_max']==600 and c['backward_calls_max']==256 and c['optimizer_steps_max']==64 and c['code_optimizer_steps_max']==0 and c['teacher_component_calls_planned']==0,'Independent actual imported count caps')
 # Semantic metric boundary checks exercise exact comparator rather thansource text.
 metrics=module.METRICS;kinds=module.KINDS;rows=[dict(mask_kind=k,condition='a' if i%2 else 'b',**{m:.1 for m in metrics}) for i,k in enumerate(kinds)];baseline={'metrics':module.aggregate(rows,True)}
 good=[dict(r,PNG_hole_L1=.0989) for r in rows];check(module.compare_metrics({'metrics':module.aggregate(good,True)},baseline,conditions=True)['prospective_numeric_screen_pass'],'1.1percent nonworse gain passes')
 low=[dict(r,PNG_hole_L1=.0999) for r in rows];check(not module.compare_metrics({'metrics':module.aggregate(low,True)},baseline,conditions=True)['prospective_numeric_screen_pass'],'0.1percent gain fails1percent gate')
 good[0]['PNG_boundary_L1']=.11;check(not module.compare_metrics({'metrics':module.aggregate(good,True)},baseline,conditions=True)['prospective_numeric_screen_pass'],'Adverse geometry/condition boundary rejectsmeanwin')
 good[0]['raw_boundary_L1']=.3;check(module.compare_metrics({'metrics':module.aggregate(good,True)},baseline,conditions=True)['catastrophic_regression_stop'],'Adverse raw geometry triggerscatastrophe')
 optimized=(stage/SOURCE_PATHS[2]).read_bytes();original=(ROOT/'research/completion_codeformer_writer_v1/optimized_budget.py').read_bytes();expected_optimized=original.replace(b'from research.completion_codeformer_coverage_v2.contracts import CONTRACT,need',b'from research.completion_codeformer_delivered_objective_v4.contracts import CONTRACT,need').replace(b'Research-only O(1) evidence-budget ledger prototype; not used by any VM run.',b'O(1) evidence-budget ledger copy for the distinct manually launched V4 pilot.')
 check(optimized==expected_optimized,'ReviewedO1writer exactcopy exceptnewpolicyimport anddocumentedmodulelabel')
 # Verify the actual new scientific data is an exact re-expression of sealedV2.
 old=read(ROOT/'outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results/protocol.json');oldfiles={r['path']:r for r in old['files']}
 check(p['schedule']==old['schedule'],'Exactsame64x4sealedV2schedule')
 check(p['terms']==old['terms'] and p['development_excluded_before_generation']==old['development_excluded_before_generation'],'Exactterms and priorrefusal/exclusion membership')
 for key in ('fit_cases','monitor_cases','challenge_cases','development_cases'):
  check([r['id'] for r in p[key]]==[r['id'] for r in old[key]],'Fixedprotocolcaseorder '+key)
  for newer,prior in zip(p[key],old[key]):
   for field in ('id','reference_id','teacher_key','role','original_role','legacy_completion_split','condition','style','mask_kind','mask_pixels'):
    if field in prior:check(newer.get(field)==prior[field],'Fixedcase metadata '+newer['id']+' '+field)
   for field in ('input256','clean256','clean512','mask'):
    if field in prior:check(files[newer[field]]['sha256']==oldfiles[prior[field]]['sha256'],'Fixedcasebytes '+newer['id']+' '+field)
 for name in ('monitor','challenge','public_development'):
  historical=p['historical_control'];check({r['id'] for r in historical['joint_rows'][name]}=={r['id'] for r in p[{'monitor':'monitor_cases','challenge':'challenge_cases','public_development':'development_cases'}[name]]},'Historicaljoint exact case membership '+name)
  for slot,cycle in (('initial',0),('joint_terminal',64)):
   summary=read(stage/historical[slot][name]);source_summary=read(ROOT/('outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results/outputs/u%04d/%s/results.json'%(cycle,name)))
   check(summary==source_summary,'Historicalsummary exact original '+slot+' '+name)
 # Verify current app and original compare source assets are retainedunchanged.
 app=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json')
 for row in app['source_files']+[app['candidate_manifest'],app['checkpoint']]:
  q=Path(row['path']);check(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'CurrentR1app baselineexact '+q.name)
 # Sourcefiles rechecked last so source read and finalpins are samebytes.
 for name in SOURCE_PATHS:check((stage/name).read_bytes()==source_path(name).read_bytes(),'Finalstable source '+name)
 check(sha(protocol_path)==expected,'Final protocol stable')
 report={'format':'completion-codeformer-delivered-objective-v4-independent-final-source-admission-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':expected,'stage_root':str(stage),'protocol':pin(protocol_path),'source_files':[pin(stage/n) for n in SOURCE_PATHS],'checks':checks,'checks_count':len(checks),'payload_files':len(files),'payload_bytes_including_protocol':payload,'manual_review_scope':'Actual source read covers frozenroutes/fullAdamRNG,losses/cohorts,compatibility,savefailure/phasecaps,supervisor/export/installer; no learned-output qualification.','actual_VM_preflight':False,'actual_training_or_neural_calls':0,'local_execution':{'Torch_imports':0,'gradient_calls':0,'optimizer_steps':0,'VM_calls':0},'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
 need(not Path(output).exists(),'Preserve prioraudit');Path(output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'checks':len(checks),'path':str(output),'sha256':sha(output)}));return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.stage,a.expected_sha,a.output)
if __name__=='__main__':main()
