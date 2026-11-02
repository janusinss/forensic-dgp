"""Independent tiny-file adversarial export fixture. No neural/scientific imports."""
import contextlib,hashlib,importlib.util,io,json,os,tarfile
from pathlib import Path
from unittest.mock import patch
BASE=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');HERE=BASE/'research/completion_spatial_code_intervention_v1_transport_review'
FIXTURE=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs\spatial_transport_review_alias_003')
SOURCE=BASE/'scripts/export_completion_codeformer_spatial_code_intervention_v1.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
spec=importlib.util.spec_from_file_location('independent_spatial_exporter',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert not FIXTURE.exists();FIXTURE.mkdir();home=FIXTURE/'home';root=FIXTURE/'pilot';home.mkdir();root.mkdir();(root/'outputs').mkdir();(root/'asset.txt').write_bytes(b'pinned tiny source asset\n')
allowed=['outputs/'+n for n in ('state_before.json','oracle_label_receipt.json','reused_control_binding.json','external_control_compatibility.json','arm_H_complete.json','arm_V_complete.json','state_after.json','results.json','failure.json','state_on_failure.json','state_receipt_failure.json','failed_actual_model_state.pth','failed_actual_model_state.pth.partial','failure_state_receipt.json')]
allowed += ['outputs/fresh_app_parity/summary.json','outputs/external_bridge/summary.json']
p={'format':'completion-codeformer-spatial-code-intervention-v1','manual_execution_only':True,'automatic_resume_or_promotion':False,'contract':{'output_bytes':1024**3,'stop_free_bytes':2*1024**3},'files':[{'path':'asset.txt','bytes':(root/'asset.txt').stat().st_size,'sha256':sha(root/'asset.txt')}],'arms':[],'parity_cases':[],'bridge_cases':[],'monitor_cases':[],'challenge_cases':[],'export':{'allowed_output_paths':sorted(allowed)}}
write(root/'protocol.json',p);write(root/'supervisor.json',{'complete':True,'worker_exit':1,'timeout':False});(root/'trainer.log').write_bytes(b'tiny failed-state exporter fixture only\n');write(root/'outputs/failure.json',{'complete':False,'fixture':True})
committed=root/'outputs/failed_actual_model_state.pth';partial=root/'outputs/failed_actual_model_state.pth.partial';committed.write_bytes(b'tiny actual failed-state placeholder; NEVER a real checkpoint\n');os.link(committed,partial);before=(committed.stat().st_ino,partial.stat().st_ino,committed.stat().st_nlink)
original_tar_open=tarfile.open;mutated=[]
def during_tar(*args,**kwargs):
    if kwargs.get('mode')=='w|' and not mutated:
        assert partial.resolve().is_relative_to(FIXTURE.resolve()) and partial.name=='failed_actual_model_state.pth.partial'
        payload=partial.read_bytes();partial.unlink();partial.write_bytes(payload);mutated.append({'partial_inode_after':partial.stat().st_ino,'committed_inode_after':committed.stat().st_ino,'same_hash':sha(partial)==sha(committed)})
    return original_tar_open(*args,**kwargs)
error=None;receipt=None
try:
    with patch.object(module.tarfile,'open',during_tar),contextlib.redirect_stdout(io.StringIO()):receipt=module.export(root,sha(root/'protocol.json'),home,enforce_vm=False)
except BaseException as exc:error=repr(exc)
result={'format':'independent-spatial-export-alias-drift-fixture-v1','complete':True,'source':{'path':str(SOURCE),'bytes':SOURCE.stat().st_size,'sha256':sha(SOURCE)},'fixture_root':str(FIXTURE),'initial_same_inode_and_nlink':before,'mutation':mutated,'export_returned_complete':bool(receipt and receipt.get('complete')),'error':error,'finding_confirmed':bool(receipt and receipt.get('complete') and partial.stat().st_ino!=committed.stat().st_ino and receipt.get('same_inode_failed_state_aliases')),'repair_refuses_before_publication':bool(error and not receipt and not (home/(module.STEM+'-results.tar.gz')).exists() and (home/(module.STEM+'-results.tar.gz.partial')).is_file()),'scope':'One tiny-file owned fixture replaces only its own partial alias with equal bytes on a distinct inode during tar streaming. No real research/VM/model/archive inputs modified.','neural_calls':0,'VM_calls':0,'real_checkpoint_reads':0}
path=HERE/'independent_alias_drift_repair_fixture.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
