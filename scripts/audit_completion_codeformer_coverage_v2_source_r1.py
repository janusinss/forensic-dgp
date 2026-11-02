"""Independent source-only admission for completion coverage V2; no Torch or VM.
The one local verify fixture replaces only the nested VM-external-path resolver
with explicit existing local files, keeping every source/protocol/data check.
It is NOT an actual VM preflight and executes no worker/model/gradient code.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
NEW_SOURCES=('scripts/cctv_completion_codeformer_coverage_v2_vm.py','research/completion_codeformer_coverage_v2/contracts.py','scripts/export_completion_codeformer_coverage_v2.py','scripts/run_completion_codeformer_coverage_v2.sh','scripts/prepare_completion_codeformer_coverage_v2.py')
PINS={'data_manifest':'393bea14e4ba8a350644bc51d65c68a0eb57ba919b1c9d0954245c1f40935ea7','data_admission':'7c072ec5a916468275e4e5ba3e3124e9b472030014f1e7afed803f5beca3d63a','root_data_visual':'8488e9354a87ce186931c2770a9ce6a8de0da06b92baddba2c6d593aea54db72','root_data_milestone':'683766dc9f6d9acc7fab31a676f522345b39310e1504ebf279af14a5ab9f554c','original_source_admission':'188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab'}
def need(ok,label):
    if not ok: raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    started=time.monotonic();stage=a.stage_root.resolve(strict=True);out=a.out.resolve();need(not out.exists(),'Fresh independent source receipt required')
    p=read(stage/'protocol.json');protocol_sha=sha(stage/'protocol.json');need(p['format']=='completion-codeformer-coverage-v2','Protocol format')
    initial={name:sha(ROOT/name) for name in NEW_SOURCES};checks=[]
    def check(name,ok,details=None):need(ok,name);checks.append({'check':name,'pass':True,'details':details})
    rows=p['files'];expected={r['path'] for r in rows};check('Exact staged namespace',len(expected)==len(rows) and {q.relative_to(stage).as_posix() for q in stage.rglob('*') if q.is_file()}==expected|{'protocol.json'})
    for row in rows:
        file=stage/row['path'];need(not file.is_symlink() and file.is_file() and file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'Staged member drift '+row['path'])
    check('All staged payload hashes and sizes',True,{'files':len(rows),'payload_bytes':sum(r['bytes'] for r in rows)+(stage/'protocol.json').stat().st_size})
    for name in NEW_SOURCES:check('Staged final source matches canonical '+name,sha(stage/name)==initial[name])
    ast_pins={}
    for name in NEW_SOURCES:
        if name.endswith('.py'):
            tree=ast.parse((stage/name).read_text(encoding='utf-8-sig'),feature_version=(3,10));ast_pins[name]=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    check('All four Python sources parse as Python 3.10',len(ast_pins)==4)
    baseline=ROOT/'outputs/completion_codeformer_manual_v1/source_admission.json';check('Original source admission exact',sha(baseline)==PINS['original_source_admission'])
    original_sources=[]
    for row in read(baseline)['source_files']:
        file=Path(row['path']);need(file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'Original/app source changed '+row['path']);original_sources.append(row)
    check('Original completion/app/source baselines unchanged',True,{'count':len(original_sources)})
    for label,path in [('data_manifest',ROOT/'outputs/completion_codeformer_coverage_v2_data/manifest.json'),('data_admission',ROOT/'outputs/completion_codeformer_coverage_v2_data/independent_data_admission_v1.json'),('root_data_visual',ROOT/'outputs/completion_codeformer_coverage_v2_root_data_visual_review.json'),('root_data_milestone',ROOT/'outputs/completion_codeformer_coverage_v2_data_milestone.json')]:check('Prior authoritative '+label+' binding',sha(path)==PINS[label])
    spec=importlib.util.spec_from_file_location('independent_coverage_v2_worker_source',stage/NEW_SOURCES[0]);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    check('Pure source import has no Torch, NumPy or PIL',not any(k=='torch' or k.startswith('torch.') or k=='numpy' or k.startswith('numpy.') or k=='PIL' or k.startswith('PIL.') for k in sys.modules))
    command=subprocess.run([sys.executable,'-B',str(stage/NEW_SOURCES[0]),'--help'],capture_output=True,text=True,check=True,timeout=30)
    check('Actual unmodified help CLI exits without VM',all(s in command.stdout for s in ('--verify-transfer','--worker','--run')))
    # Audit-only local external mapping; the runtime worker remains unmodified.
    local_original=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth')
    local_parent=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth'
    mapped=[]
    def external_fixture(row,name,want_sha,want_bytes):
        expected_path='../cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth' if name=='codeformer_inpainting.pth' else '../cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth'
        file=local_original if name=='codeformer_inpainting.pth' else local_parent
        need(row['path']==expected_path and row['sha256']==want_sha and row['bytes']==want_bytes and file.name==name and file.is_file() and file.stat().st_size==want_bytes and sha(file)==want_sha,'Exact local mapped external evidence')
        mapped.append(pin(file));return file
    tree=ast.parse((stage/NEW_SOURCES[0]).read_text(encoding='utf-8-sig'));fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify'))
    replaced=0
    for node in ast.walk(fn):
        if isinstance(node,ast.FunctionDef) and node.name=='external':
            node.body=ast.parse('return external_fixture(row,name,want_sha,want_bytes)').body;replaced+=1
    need(replaced==1,'Only nested external VM resolver replaced in local fixture')
    namespace=dict(worker.__dict__);namespace['external_fixture']=external_fixture
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<independent-local-path-fixture>','exec'),namespace)
    returned,base,parent=namespace['verify'](stage,protocol_sha)
    check('All unmodified protocol/source/data admission checks pass with explicit local external mapping',returned==p and len(mapped)==2,{'runtime_VM_path_resolver_executed':False,'actual_VM_preflight_passed':False,'local_existing_external_assets':mapped})
    check('New phase counts and cumulative arithmetic',64*4*2==512 and 64*2==128 and 160+8+512+128+32+56==896 and 8+512+128+32+56==736 and 528+896==1424 and 448+736==1184)
    helper=namespace['compare_metrics'];metrics=namespace['METRICS'];kinds=namespace['KINDS']
    def cohort(value,conditions=False):
        summary={'metrics':{'macro':{k:value for k in metrics},'geometry':{g:{k:value for k in metrics} for g in kinds}}}
        if conditions:summary['metrics']['condition']={g:{k:value for k in metrics} for g in ('gray','RGB_irregular')}
        return summary
    check('Both cohorts require actual one percent and no subgroup regression',helper(cohort(.989),cohort(1))['prospective_numeric_screen_pass'] and not helper(cohort(.999),cohort(1))['prospective_numeric_screen_pass'] and helper(cohort(.989,True),cohort(1,True),conditions=True)['prospective_numeric_screen_pass'])
    bad=cohort(.98,True);bad['metrics']['condition']['gray']['raw_boundary_L1']=1.0000001
    check('One adverse challenge condition endpoint rejects favorable mean',not helper(bad,cohort(1,True),conditions=True)['prospective_numeric_screen_pass'])
    bad=cohort(.98);bad['metrics']['geometry'][kinds[0]]['PNG_boundary_L1']=1.0000001
    check('One adverse geometry endpoint rejects favorable mean',not helper(bad,cohort(1))['prospective_numeric_screen_pass'])
    check('Catastrophic raw regression stops despite favorable PNG',helper(cohort(1.11),cohort(1))['catastrophic_regression_stop'])
    source=(stage/NEW_SOURCES[0]).read_text(encoding='utf-8-sig');helper_source=(stage/NEW_SOURCES[1]).read_text(encoding='utf-8-sig')
    tree=ast.parse(source);worker_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    # Evaluate only the two cumulative-counter dictionary expressions, no function calls.
    cumulative=[]
    for node in ast.walk(worker_fn):
        if isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value in ('counts_cumulative','cumulative_counts'):cumulative.append(value)
    check('Terminal and final receipts both retain cumulative counters',len(cumulative)==2)
    sample={'forward_slots':896,'backward_calls':512,'code_optimizer_steps':64,'fusion_optimizer_steps':64,'empty_bypasses':168}
    for index,value in enumerate(cumulative):
        need(not any(isinstance(n,ast.Call) for n in ast.walk(value)),'Cumulative dictionary fixture must be pure arithmetic')
        actual=eval(compile(ast.Expression(value),'<counter-expression>','eval'),{'counts':sample,'model_forwards':{'calls':736}})
        need(actual=={'forward_slots':1424,'actual_whole_model_forward_calls':1184,'backward_calls':768,'code_optimizer_steps':96,'fusion_optimizer_steps':96,'optimizer_steps_total':192},'Completed cumulative values differ')
        partial=eval(compile(ast.Expression(value),'<partial-counter-expression>','eval'),{'counts':{'forward_slots':123,'backward_calls':23,'code_optimizer_steps':3,'fusion_optimizer_steps':2},'model_forwards':{'calls':43}})
        need(partial['forward_slots']==651 and partial['actual_whole_model_forward_calls']==491 and partial['backward_calls']==279 and partial['optimizer_steps_total']==69,'Partial cumulative values differ')
    check('Success and failure cumulative arithmetic uses actual new counters',True)
    check('Full four-family RNG equality and hash receipts present',all(s in source for s in ("actual_nr[0]==nr['algorithm']","actual_nr[2]==nr['position']","actual_nr[3]==nr['has_gauss']","actual_nr[4]==nr['cached_gaussian']","len(restored_cuda)==len(resume_rng['CUDA'])==1","expected_rng_hashes==restored_rng_hashes","'RNG_boundary_hashes'","'expected_hashes':expected_rng_hashes","'restored_hashes':restored_rng_hashes")))
    check('RNG restore occurs after baseline and before optimization loop',source.index("baseline=evaluate(0")<source.index("torch.set_rng_state(resume_rng")<source.index('for cycle,case_ids in enumerate(schedule,1)'))
    check('Original teacher generation occurs before loading trained R1',source.index("for row in p['teacher_targets']:")<source.index("parent=torch.load(parent_path")<source.index("R1weights_Adam_restored"))
    check('Strict original and resume safe state loads only',source.count('torch.load(')==2 and source.count('weights_only=True')==2)
    check('Original exact losses and clipping retained',all(s in source for s in ('F.cross_entropy(logits.reshape(-1,512),codes.reshape(-1))','F.mse_loss(lq,latent)','loss=(ce+latent_loss)/4','loss=(rgb_loss+seam)/4','clip_grad_norm_(code_parameters,1.,error_if_nonfinite=True)','clip_grad_norm_(fusion_parameters,1.,error_if_nonfinite=True)')) and not any(s in source for s in ('autocast(','.train(','.GradScaler(','.backward(create_graph=True')))
    check('Only two Adam step sites exist within the finite 64-cycle loop',source.count('code_optimizer.step()')==1 and source.count('fusion_optimizer.step()')==1 and "'optimizer_steps_max':128" in helper_source)
    check('Retained teacher NPZ names portable and actual labels finite bounded',"'cache_id'" in source and 'prequant_latent=' in source and 'nearest_codes=' in source and "'teacher_bytes':24*1024**2" in helper_source)
    check('Fresh app parity captures all four pairs before asserting 1e-6',source.index("for case in parity_cases:")<source.index("write(parity_folder/'summary.json'")<source.index("Fresh frozen app pipeline baseline parity failed") and "need(len(parity_cases)==4" in source and "row['max_raw256_error']<=1e-6" in source)
    check('Canonical unmodified wrapper used before optional display processing',"CodeFormerCompletion(net)" in source and 'to_grayscale' not in source and 'monochrome' not in source)
    check('Clear and challenge gates separate and both required',"clear_screen=compare(monitors[-1],baseline);challenge_screen=compare(challenge_final,challenge_baseline,conditions=True)" in source and "clear_screen['prospective_numeric_screen_pass'] and challenge_screen['prospective_numeric_screen_pass']" in source and "'cohorts_never_pooled':True" in source)
    check('Raw and PNG visible preservation plus no-NN empty controls enforced',"torch.equal(out_tensor*(1-m),rgb*(1-m))" in source and 'np.array_equal(array[~support],observed[~support])' in source and "model_forwards['calls']==before_calls" in source)
    check('One full terminal with both Adam, all RNG, schedule, progress and partial gradients',all(s in source for s in ("'code_optimizer':code_optimizer.state_dict()","'fusion_optimizer':fusion_optimizer.state_dict()","'numpy_random_state'","'sampler_schedule':schedule","'partial_accumulated_gradients'","'parent_optimizer_restored'","'parent_RNG_restored'","'earlier_snapshots_recoverable':False")))
    check('Perwrite caps reserve terminal, teacher, metadata, PNG, final receipt and disk',all(s in helper_source for s in ('self.terminal_committed','self.supervisor_reserve=1024**2',"CONTRACT['teacher_bytes']","CONTRACT['terminal_bytes']","CONTRACT['metadata_bytes']","CONTRACT['PNG_bytes_each']",'512*1024',"CONTRACT['stop_free_bytes']")) and 'def fileno(' not in helper_source)
    check('Supervisor uses bounded PIPE and its owned process group',all(s in source for s in ('stdout=subprocess.PIPE','start_new_session=True','os.getpgid(child.pid)==child.pid','os.killpg(child.pid,sig)',"remaining=CONTRACT['trainer_log_bytes']-captured",'log.write(chunk[:remaining])','hard_cap_SIGKILL',"with (root/'supervisor.json').open('xb')")))
    check('Hard and cooperative timing plus exact same L4 root remain strict',all(s in source for s in ("os.name=='posix'","Path.home()/'forensic-dgp'/VM_NAME","machine.rsplit('/',1)[-1]=='g2-standard-4'","len(gpu)==1 and 'L4' in gpu[0]","time.monotonic()-started>=1800")) and 'else 1680' in helper_source)
    exporter=(stage/NEW_SOURCES[2]).read_text(encoding='utf-8-sig');packager=(stage/NEW_SOURCES[4]).read_text(encoding='utf-8-sig');shell=(stage/NEW_SOURCES[3]).read_text(encoding='utf-8-sig')
    check('Exporter and installer independently tested final sources match package',sha(stage/NEW_SOURCES[2])=='eb9c17f3875ac1f576b81d1a58915c47fa8e7e44d5635eaf4c19a7e157bda0c2' and sha(stage/NEW_SOURCES[4])=='7cf37cd5140fd488b5a03bc963df1efba6b151275d2f3f3add9c95173635b10e')
    check('Export and package remain pure standard library and manual release only',not any(s in exporter+packager for s in ('import torch','import numpy','subprocess.Popen','compute ssh')) and "admission.get('complete') is True and admission.get('independent') is True and admission.get('protocol_sha256')==expected" in packager and "release.get('automatic_vm_launch') is False" in packager)
    check('Shell always preserves worker exit and attempts bounded export',shell.index('worker_exit=$?')<shell.index('scripts/export_completion_codeformer_coverage_v2.py')<shell.index('export_exit=$?') and 'exit "$worker_exit"' in shell and 'set -uo pipefail' in shell)
    findings=[
      {'scope':'teacher_and_resume','finding':'Original frozen b0d1 encoder/nearest512 code teacher cached for 40RGB+40GRAY before full R1 trainable332/frozen165 reconstruction; exact named Adam moments/step32 restored. RNG reset at declared post-parity/post-baseline boundary, all fields and expected/restored hashes retained. No inference-candidate-only resume.'},
      {'scope':'finite_learning','finding':'Two routes alternate on identical materialized 64x4 schedules. Each cycle2anchors+2augmented/4source refs;256cases once perroute. Same CE+latent and unclamped512hole+kernel7boundary losses, FP32 Adam LR1e-5, clip1, frozen generator/codebook;128new Adam steps/512backwards. Newtyped896/whole736; cumulative1424/1184,192Adam/768backwards,step96each at success.'},
      {'scope':'quality_and_evidence','finding':'Four raw canonical R1 app parity pairs retained before unchanged1e-6 assert. Same-L4 frozenR1 baselines for original32clear and16challenge, >=1%PNGholegain each separately plus allraw/PNGmacro,geometry,andchallengecondition nonworse1e-12; catastrophe10% earlystop. OriginalCodeFormer cumulative comparison reported separately. Public32 remain unpaired, excludedfromlearningtargets. Exactoutside-maskraw/PNG and168emptybypasses;graychroma diagnostic only.'},
      {'scope':'resources_failures','finding':'1800s external owned child/1680s cooperative guard, oneL4/g2 andnoexistingworkload,20GiBCUDA,7GiBstart/2GiBfloor. 2304MiB global/rawtar/compressed,1GiB terminal,24MiBteacher,8MiBmetadata including capped4MiBPIPElog/protocol/1MiBsupervisor,256KiBPNG. Perwrite reservations with terminalcommit release and finalfailureheadroom. Full current terminal or failedpartial with moments/RNG/sampler/progress/partialgrads retained; noauto-resume/snapshotrescue.'},
      {'scope':'release_boundary','finding':'Fresh stage source/protocol namespace and hashes exact. Archive requires this independently bound protocol admission and separate root manual-transfer release; no VM or automatic training. Installer stages exclusive regular/pinned members and refuses existing pilot, namespace drift or insufficient disk. Export preserves failedstate/archivepartials and originalparentassets.'}
    ]
    root_tests=[]
    for name in ('completion_codeformer_coverage_v2_root_export_tests.json','completion_codeformer_coverage_v2_root_installer_tests.json','completion_codeformer_coverage_v2_root_contract_tests_v2.json'):
        root_tests.append(pin(ROOT/'outputs'/name))
    check('Final canonical sources unchanged during independent audit',initial=={name:sha(ROOT/name) for name in NEW_SOURCES})
    report={'format':'completion-codeformer-coverage-v2-independent-final-source-admission-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol_sha,'stage_root':str(stage),'source_files':[pin(stage/name) for name in NEW_SOURCES],'protocol':pin(stage/'protocol.json'),'checks':checks,'review_findings':findings,'function_AST_sha256':ast_pins,'retained_root_boundary_tests':root_tests,'repaired_before_admission':['Missing cumulative typed/whole forwards in terminal/final receipt','Partial NumPy/CUDA RNG boundary validation and missing RNG boundary hashes','Post-terminal reserve double-accounting and receipt deadline','CappedWriter interface for real NumPy/Pillow/Torch serialization','Supervisor bounded PIPE metadata accounting and immediate owned PID status'],'local_verify_fixture':{'modified_runtime_source':False,'only_test_override':'AST-cloned verify() nested external VM path resolver mapped to explicitly hashed existing local base/parent. All other verify checks unmodified.','VM_path_boundary_runtime_tested':False,'actual_VM_preflight_passed':False},'limits':{'model_or_gradient_or_optimizer_calls':0,'Torch_imports_in_this_audit':0,'VM_or_network_calls':0,'reserved_FINAL_access':0,'new_archive_built':False,'training_run':False,'actual_VM_execution_pending':True,'returned_tensor_pixel_quality_audit_pending':True,'numeric_screen_not_model_qualification':True,'automatic_VM_launch':False,'automatic_app_promotion':False,'model_qualified':False,'goal_complete':False},'seconds':time.monotonic()-started,'auditor_source':pin(Path(__file__))}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'independent':True,'checks':len(checks),'protocol_sha256':protocol_sha,'receipt':pin(out),'model_qualified':False,'actual_VM_preflight_passed':False}),flush=True)
if __name__=='__main__':main()

