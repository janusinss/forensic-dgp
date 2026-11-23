"""Independent full packet admission and actual import-blocked transfer CLI.
Creates private same-volume hardlinks for exact source/old-control/original/R1 bytes.
Never invokes a model, NN, gradient, optimizer, cloud or external command.
"""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import ast,builtins,contextlib,hashlib,io,json,os,runpy,socket,subprocess,sys,time,urllib.request
ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_004/cctv_completion_codeformer_spatial_code_intervention_v1_vm'
OLD=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results'
DESIGN=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design/root_design.json'
PROTOCOL='b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123'
OUTPUTS=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def ip(path):
 p=Path(path).resolve();return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(path):
 h=hashlib.sha256()
 with ip(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve()),'bytes':ip(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(ip(path).read_text(encoding='utf-8-sig'))
def samefile(src,dst):
 a=ip(src).stat();b=ip(dst).stat();need(a.st_dev==b.st_dev and a.st_ino==b.st_ino and a.st_nlink>=2 and b.st_nlink>=2 and a.st_size==b.st_size,'Exact sameinode private fixturehardlink');return {'source':str(Path(src).resolve()),'destination':str(Path(dst).resolve()),'same_device_inode':True,'bytes':a.st_size}

def main():
 started=time.monotonic();need(sha(STAGE/'protocol.json')==PROTOCOL,'Exact finaldraft004protocol');p=read(STAGE/'protocol.json');d=read(DESIGN);need(sha(DESIGN)=='1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657','Exact admitted rootdesign');files={r['path']:r for r in p['files']};need(len(files)==len(p['files'])==166,'All166unique stagedpayloads')
 # Extended prefix paths do not compare directly with basic STAGE; normalize via suffix from resolved staging root.
 actual={q.relative_to(ip(STAGE)).as_posix() for q in ip(STAGE).rglob('*') if q.is_file()};need(actual==set(files)|{'protocol.json'},'Closed167regular stage namespace')
 payloadpins=[]
 for rel,row in files.items():
  q=STAGE/rel;need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and ':' not in rel and '\\' not in rel,'Safe staged relativepath');need(ip(q).stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Every166payload exact');payloadpins.append(pin(q))
  if q.suffix=='.py':ast.parse(ip(q).read_text(encoding='utf-8-sig'),feature_version=(3,10))
 worker=STAGE/'scripts/cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';need(sha(worker)=='ffe76c1cf2c0329c8b719687be21db4d775081f9df47d4369137a84bd1dab411','Latest21-fixture worker included');runtime=read(ROOT/'research/completion_codeformer_spatial_code_intervention_v1_independent_review/independent_runtime_source_semantics.json');need(runtime['complete'] is True and runtime['source_pins'][worker.name]['sha256']==sha(worker),'Latest independently admittedsource')
 shellpath=STAGE/'scripts/run_completion_codeformer_spatial_code_intervention_v1.sh';shell=ip(shellpath).read_text();guard='command -v timeout >/dev/null || exit 1';exportline='timeout --signal=KILL 900s "$PY" -B -u scripts/export_completion_codeformer_spatial_code_intervention_v1.py --root "$ROOT" --protocol-sha "$EXPECTED"';need(guard in shell and exportline in shell and shell.index(guard)<shell.index('"$PY" -B -u scripts/cctv_')<shell.index(exportline),'ExternalGNUtimeout900s guard before anywork and ownedexport command');need('export_exit=$?' in shell and 'exit "$export_exit"' in shell and 'exit "$worker_exit"' in shell,'Export failure/timeout status propagated without automaticretry')
 oldp=read(OLD/'protocol.json');need(p['monitor_cases']==oldp['monitor_cases'] and p['challenge_cases']==oldp['challenge_cases'] and p['parity_cases']==oldp['parity_cases'] and p['references']==oldp['references'],'Exact olddata/case/reference rows carried unchanged');need([r['id'] for r in p['bridge_cases']]==d['data']['bridge_case_ids'],'Exact4freshC IDs');need(p['checkpoint']==d['models']['original_checkpoint'] and p['parent']=={k:d['models']['exact_full_R1_parent'][k] for k in p['parent']},'Exact original/R1 byte bindings')
 labels={r['teacher_key']:r for r in p['oracle_labels']};want={r['teacher_key']:r for r in d['data']['original16labels_exact_only']};need(len(labels)==16 and set(labels)==set(want),'Exact16samebook labels');need(all(labels[k]['sha256']==want[k]['sha256'] and labels[k]['bytes']==want[k]['bytes'] and labels[k]['clean512_sha256']==want[k]['clean512_sha256'] for k in labels),'No teacher/preprior swap')
 ext=p['external_controls'];ef={r['path']:r for r in ext['files']};oldfiles={r['relative_output_path']:r for r in d['external_controls']['files']};need(set(ef)==set(oldfiles) and len(ef)==196,'Exact196oldA/C namespace')
 for rel,row in ef.items():need(row['sha256']==oldfiles[rel]['sha256'] and row['bytes']==oldfiles[rel]['bytes'] and sha(OLD/'outputs'/rel)==row['sha256'],'Every196retainedoldA/C binding')
 for arm,folder in [('A','ACTUAL_A_R1_w1'),('C','ORACLE_C_unavailable_clean_information_w1')]:
  for kind in ('monitor','challenge'):need(ext['arms'][arm][kind]==read(OLD/'outputs'/folder/kind/'results.json'),'Exact4historicalsummary objects')
 need(p['manual_execution_only'] is True and p['automatic_resume_or_promotion'] is False,'Manualonly/noautomaticresume')
 # Private fixture location is short to avoid Windows MAX_PATH; retain every link as audit evidence.
 for i in range(1,1000):
  home=OUTPUTS/('sv%03d'%i)
  if not home.exists():break
 else:raise AssertionError('No free private fixture namespace')
 need(home.resolve().is_relative_to(OUTPUTS.resolve()),'Private fixture confined originalTestingoutputs');home.mkdir();forensic=home/'forensic-dgp';new=forensic/'cctv_completion_codeformer_spatial_code_intervention_v1_vm';links=[]
 def link(src,dst):
  need(not ip(dst).exists(),'No fixture overwrite');ip(dst.parent).mkdir(parents=True,exist_ok=True);os.link(ip(src),ip(dst));links.append(samefile(src,dst))
 for rel in sorted(actual):link(STAGE/rel,new/rel)
 original=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth');parent=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth';oldroot=forensic/'cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs'
 link(original,forensic/'cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth');link(parent,forensic/'cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth')
 for rel in ef:link(OLD/'outputs'/rel,oldroot/rel)
 need(len(links)==167+2+196==365,'365privateexacthardlinks');before={str(q.relative_to(ip(home))):q.stat().st_size for q in ip(home).rglob('*') if q.is_file()};need(len(before)==365,'Closed365fixture files beforeCLI')
 forbidden_attempts=[]
 def forbidden(*args,**kwargs):forbidden_attempts.append(str(args[:1]));raise AssertionError('Network/subprocess/OSexecution forbidden in verify-only fixture')
 realimport=builtins.__import__
 def blocked(name,*args,**kwargs):
  if name.split('.')[0] in ('torch','numpy','PIL','cv2','scipy'):forbidden_attempts.append('scientificimport '+name);raise AssertionError('Scientificmodule forbidden in verify-only fixture '+name)
  return realimport(name,*args,**kwargs)
 stdout=io.StringIO();privateworker=ip(new/'scripts/cctv_completion_codeformer_spatial_code_intervention_v1_vm.py');startCLI=time.monotonic();exitcode=None
 with patch.object(Path,'home',return_value=ip(home)),patch.object(sys,'argv',[str(privateworker),'--root',str(ip(new)),'--protocol-sha',PROTOCOL,'--verify-transfer']),patch.object(builtins,'__import__',blocked),patch.object(subprocess,'Popen',forbidden),patch.object(subprocess,'run',forbidden),patch.object(subprocess,'check_output',forbidden),patch.object(subprocess,'check_call',forbidden),patch.object(urllib.request,'urlopen',forbidden),patch.object(socket,'socket',forbidden),patch.object(os,'system',forbidden),contextlib.redirect_stdout(stdout):
  try:runpy.run_path(str(privateworker),run_name='__main__')
  except SystemExit as result:exitcode=result.code
 CLI_seconds=time.monotonic()-startCLI
 need(exitcode==0 and not forbidden_attempts,'Actual unmodified --verify-transfer exits0 with forbidden actions activelyblocked');printed=json.loads(stdout.getvalue());need(printed['complete'] is True and printed['files']==166 and printed['external_controls_files']==196 and printed['total_model_calls_planned']==108 and printed['neural_calls']==printed['optimizer_updates']==printed['teacher_neural_calls']==0,'Actual verify-only printed scope exact')
 after={str(q.relative_to(ip(home))):q.stat().st_size for q in ip(home).rglob('*') if q.is_file()};need(after==before and not ip(new/'outputs').exists() and not ip(new/'supervisor.json').exists(),'ActualCLIcreated nooutputs/processstate/pycache');need(not any(n in sys.modules for n in ('torch','numpy','PIL','cv2')),'No scientificmodule ever imported inactualCLIprocess')
 for row in payloadpins:need(pin(row['path'])==row,'Stagedsource unchangedafterCLI')
 for rel,row in ef.items():need(sha(OLD/'outputs'/rel)==row['sha256'],'Historicalcontrol unchangedafterCLI')
 sourceadmission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');r1pins=sourceadmission['source_files']+[sourceadmission['candidate_manifest'],sourceadmission['checkpoint']]
 for row in r1pins:need(pin(row['path'])==row,'CurrentprotectedR1app/model unchanged')
 need(sha(STAGE/'protocol.json')==PROTOCOL and sha(original)==p['checkpoint']['sha256'] and sha(parent)==p['parent']['sha256'],'Original/R1/protocol stableafterCLI')
 result={'format':'independent-spatial-code-intervention-v1-packet-and-actual-transfer-CLI-admission','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'protocol_sha256':PROTOCOL,'protocol':pin(STAGE/'protocol.json'),'checks_passed':CHECKS,'staged_payloads_all166_exact':True,'closed_regular_namespace167':True,'allVMruntime_python310syntax_parsed':True,'exactold48cases4parity8monitorreferences16labels196controls':True,'exactall4reusedsummaries':True,'latest_worker_source_and_independent_semantics_bound':True,'external_export_hard_cap':{'shell':pin(shellpath),'command_v_timeout_guard_precedes_work':True,'GNUtimeout_owned_export_child_KILL900s':True,'export_status_propagated_noautomaticretry':True,'VMexecution_not_performed':True},'actual_unmodified_verify_transfer_CLI':{'exit_code':exitcode,'printed_receipt':printed,'seconds':CLI_seconds,'private_fixture':str(home.resolve()),'sameinode_exacthardlinks':len(links),'fixture365regular_files_before_after_exact':True,'scientificimports_network_subprocess_OSexecution_activelyblocked':True,'forbidden_attempts':forbidden_attempts,'noNPZTorchPIL_oractualmodel_loaded':True,'nooutputs_supervisor_orpycache_created':True,'Path_home_overridden_onlywithinfixtureprocess':True},'source_pins':payloadpins,'private_hardlink_ledger':links,'all22_current_R1pins_exact':True,'original_parent_andcontrols_unchanged':True,'archive_admitted':False,'released_for_root_manual_archive_preparation_only':True,'actual_VM_launch_authorized_by_this_receipt':False,'manual_transfer_release_pending_final_archive_checks':True,'model_qualified':False,'full_goal_complete':False,'local_actions':{'model_orcheckpoint_loads':0,'neural_calls':0,'optimizer_objects_or_steps':0,'gradient_calls':0,'VM_or_cloud_calls':0,'external_process_calls':0,'actual_worker_run_orhistorical_training':False},'seconds':time.monotonic()-started}
 target=Path(__file__).parent/'independent_source_packet_admission_draft004.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'checks':CHECKS,'actualCLIexit':exitcode,'payloads':166,'external_controls':196,'neural_calls':0},indent=2),flush=True)
if __name__=='__main__':main()
