"""Run the unmodified same-codebook --verify-transfer CLI in an isolated hardlink fixture.
No VM, neural model, optimizer, training, weight deserialization or network call.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth')
PARENT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results/outputs/terminal_training_state.pth'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic();stage=a.stage.resolve(strict=True);protocol=stage/'protocol.json';need(sha(protocol)==a.expected_sha,'Sealed protocol differs');d=json.loads(protocol.read_text());need(not a.output.exists(),'Preserve prior CLI receipt')
 tempbase=Path(tempfile.gettempdir()).resolve(strict=True);temporary=Path(tempfile.mkdtemp(prefix='completion-samecode-pure-cli-',dir=tempbase)).resolve(strict=True);need(temporary.parent==tempbase and temporary.name.startswith('completion-samecode-pure-cli-'),'Unexpected fixture root');home=temporary/'home';vm=home/'forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm';vm.mkdir(parents=True)
 expected={r['path']:r for r in d['files']};expected['protocol.json']={'bytes':protocol.stat().st_size,'sha256':a.expected_sha};hardlinks=[]
 for rel,row in expected.items():
  source=stage/rel;target=vm/rel;need(source.is_file() and not source.is_symlink() and source.stat().st_size==row['bytes'] and sha(source)==row['sha256'],'Fixture source drift '+rel);target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target);need(os.path.samefile(source,target),'Payload not a hardlink')
 for kind,source in [('checkpoint',ORIGINAL),('parent',PARENT)]:
  row=d[kind];target=(vm/row['path']).resolve();need(target.is_relative_to(home/'forensic-dgp') and not target.is_relative_to(vm),'Dependency fixture escaped sibling evidence scope');target.parent.mkdir(parents=True,exist_ok=True);need(source.stat().st_size==row['bytes'] and sha(source)==row['sha256'],'Original/parent bytes drift');os.link(source,target);need(os.path.samefile(source,target),'Dependency not a hardlink');hardlinks.append({'kind':kind,'original':pin(source),'fixture_path':str(target),'samefile_hardlink':True})
 worker=vm/'scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py'
 wrapper=temporary/'guarded_cli.py';wrapper.write_text('''import importlib.abc,pathlib,runpy,sys
class BlockScientific(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','PIL'):raise RuntimeError('Scientific import forbidden in transfer-only CLI: '+fullname)
def audit(event,args):
 if event in ('subprocess.Popen','os.system','socket.connect','socket.bind'):raise RuntimeError('External operation forbidden in transfer-only CLI: '+event)
sys.addaudithook(audit)
sys.meta_path.insert(0,BlockScientific())
worker,root,home,protocol=sys.argv[1:]
pathlib.Path.home=classmethod(lambda cls:pathlib.Path(home))
sys.argv=[worker,'--root',root,'--protocol-sha',protocol,'--verify-transfer']
runpy.run_path(worker,run_name='__main__')
assert not any(n.split('.')[0] in ('torch','numpy','PIL') for n in sys.modules)
''',encoding='utf-8')
 child=subprocess.run([sys.executable,'-I','-B',str(wrapper),str(worker),str(vm),str(home),a.expected_sha],capture_output=True,text=True,timeout=60)
 need(child.returncode==0,'Actual guarded CLI failed '+child.stdout+' '+child.stderr);result=json.loads(child.stdout);need(result['complete'] is True and result['files']==len(d['files']) and result['oracle_label_sets']==16 and result['diagnostic_outputs_planned']==192 and result['optimizer_objects']==0 and result['monitor_cases']==32 and result['challenge_cases']==16 and result['optimizer_updates']==0 and result['neural_calls']==0,'Actual CLI result differs')
 actual={x.relative_to(vm).as_posix() for x in vm.rglob('*') if x.is_file()};need(actual==set(expected) and not (vm/'outputs').exists() and not (vm/'supervisor.json').exists(),'CLI wrote unexpected runtime namespace')
 for rel,row in expected.items():need((vm/rel).stat().st_size==row['bytes'] and sha(vm/rel)==row['sha256'] and sha(stage/rel)==row['sha256'],'CLI altered linked evidence '+rel)
 for kind,source in [('checkpoint',ORIGINAL),('parent',PARENT)]:need(sha(source)==d[kind]['sha256'],'CLI changed model dependency')
 need(temporary.resolve().parent==tempbase and temporary.name.startswith('completion-samecode-pure-cli-'),'Refuse cleanup outside exact fixture directory')
 # Every created path is confined to this fresh fixed-prefix temp root; only fixture hardlink aliases are removed.
 shutil.rmtree(temporary)
 need(not temporary.exists(),'Fixture cleanup incomplete')
 report={'format':'independent-same-codebook-probe-v1-actual-guarded-pure-transfer-CLI-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'protocol_sha256':a.expected_sha,'worker_sha256':sha(stage/'scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py'),'actual_unmodified_verify_transfer_CLI_branch':True,'staged_payload_files':len(d['files']),'fixture_uses_byte_exact_hardlinks':True,'original_parent_hardlinks':hardlinks,'Path_home_fixture_only':True,'Torch_NumPy_PIL_imports_actively_blocked':True,'subprocess_socket_system_calls_actively_blocked_in_child':True,'returncode':child.returncode,'stdout':result,'stderr':child.stderr,'actual_VM_preflight':False,'real_source_dependency_bytes_unchanged':True,'no_runtime_outputs_or_supervisor_created':True,'safe_resolved_temp_fixture_removed':True,'runtime_python':sys.version,'Python310_syntax_checked_separately':True,'neural_gradient_optimizer_VM_calls':0,'model_qualified':False,'seconds':time.monotonic()-started}
 a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'receipt':pin(a.output),'returncode':0,'payload_files':len(d['files']),'seconds':report['seconds']},indent=2))
if __name__=='__main__':main()
