"""Independent packet/source/dependency/deadline and actual Windows rejection check."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

from diagnose_cctv_dgp_active_original_decoder_v28_mean_control import sha,read,write

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_vm'
PREP=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_preparation'
OLD=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'
RETURNED=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return'
PARENT=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'
WORKER='scripts/cctv_dgp_v28_preservation_diagnostic_v1_vm.py'


def main():
    start=time.monotonic();p=read(BUNDLE/'protocol.json');prepared=read(PREP/'preparation.json')
    assert sha(BUNDLE/'protocol.json')==prepared['protocol_sha256'];pin=prepared['protocol_sha256']
    assert p['format']=='own-DGP-final-v28-preservation-diagnostic-v1'
    assert p['optimizer_updates']==p['epochs']==p['backwards']==0 and p['component_gradient_calls']==100
    assert p['gradient_layout']==[10,498627] and p['gradient_saved_values_bound']==54848970
    assert len(p['case_rows'])==50 and all(c['role']=='train' for c in p['case_rows'])
    assert prepared['preparer_sha256']==sha(ROOT/'scripts/prepare_cctv_dgp_v28_preservation_diagnostic_v1.py')
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest
    for name,digest in p['closed_V28_evidence_sha256'].items():assert sha(RETURNED/name)==digest
    assert len(p['closed_V28_evidence_sha256'])==307
    base=read(PARENT/'protocol.json')
    for name,digest in base['assets_sha256'].items():assert sha(PARENT/name)==digest
    original=read(OLD/'protocol.json');assert sha(OLD/'protocol.json')==p['closed_V28_protocol_sha256']
    assert p['retained_capacity_gates']==original['retained_capacity_gates']==base['prospective_gates']
    assert p['parameter_layout']==original['parameter_layout'] and p['terms'][:7]==original['terms']
    assert p['decoder_parameters']==498627 and p['decoder_parameter_tensors']==12
    worker=(BUNDLE/WORKER).read_text(encoding='utf-8');tree=ast.parse(worker,feature_version=(3,10))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['AdamW','Adam','SGD','backward','step','zero_grad'] for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='save' and isinstance(n.value,ast.Name) and n.value.id=='torch' for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Name) and n.id in ['optimizer','train'] for n in ast.walk(tree))
    fn=lambda t,name:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    old_tree=ast.parse((OLD/'scripts/cctv_dgp_active_original_decoder_v28_vm.py').read_text(encoding='utf-8'))
    for name in ['parent_check','original_functions']:
        assert ast.dump(fn(tree,name),include_attributes=False)==ast.dump(fn(old_tree,name),include_attributes=False)
    run=fn(tree,'run');assert run.body[0].value.func.id=='vm_scope'
    for literal in ['signal.alarm(300)',"'cap_seconds':330","'within_external_bound':elapsed<=360",
                    "'optimizer_constructed':False","'new_checkpoint_created':False",'candidate.net.load_state_dict(saved_state,strict=True)',
                    "terms['diagnostic_clear_SSIM_regression']=b['clear_weight']*terms['SSIM_regression']/5",
                    "terms['diagnostic_clear_ArcFace_regression']=b['clear_weight']*terms['ArcFace_regression']/5"]:assert literal in worker,literal
    archive=ROOT/'outputs/cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz'
    assert archive.stat().st_size==prepared['archive_bytes']==28332 and sha(archive)==prepared['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==3 and all(m.isfile() for m in members)
        for member in members:
            prefix=BUNDLE.name+'/';assert member.name.startswith(prefix)
            name=member.name[len(prefix):];stream=tar.extractfile(member);assert stream
            with stream:data=stream.read()
            assert data==(BUNDLE/name).read_bytes()
    shell=(BUNDLE/'scripts/run_v28_preservation.sh').read_text(encoding='utf-8')
    assert '330s' in shell and '150s' in shell and '--kill-after=30s' in shell
    assert '--run' in shell and '--export' in shell and '--record-supervision' in shell
    assert '--verify-transfer' not in shell and 'test ! -e "$pilot_root/outputs"' in shell
    # Actual forbidden-host attempt ends before imports, outputs, CUDA or gradients.
    actual=subprocess.run([sys.executable,'-B',str(BUNDLE/WORKER),'--root',str(BUNDLE),'--protocol-sha',pin,'--run'],capture_output=True,text=True,timeout=20)
    assert actual.returncode!=0 and 'Existing Linux VM only; no local gradients' in actual.stderr
    assert not (BUNDLE/'outputs').exists()
    write(PREP/'actual_Windows_rejection.json',{'complete':True,'exit_code':actual.returncode,'stderr':actual.stderr,
        'neural_or_gradient_calls':0,'outputs_created':False,'optimizer_updates':0})
    tests=subprocess.run([sys.executable,'-B',str(ROOT/'tests/test_cctv_dgp_v28_preservation_diagnostic_v1.py')],capture_output=True,text=True,timeout=30)
    assert tests.returncode==0 and 'Ran 9 tests' in tests.stderr and 'OK' in tests.stderr
    write(PREP/'test_receipt.json',{'complete':True,'test_source_sha256':sha(ROOT/'tests/test_cctv_dgp_v28_preservation_diagnostic_v1.py'),
        'tests_passed':9,'exit_code':tests.returncode,'output':tests.stderr,'synthetic_arrays_are_not_VM_evidence':True})
    runbook=(ROOT/'CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',runbook,flags=re.MULTILINE)==['1','2','3','4','5']
    lines=[s for s in runbook.splitlines() if s.startswith('gcloud compute scp ')]
    assert len(lines)==5 and all('--project=forensic-dgp-thesis --zone=us-central1-a' in s for s in lines)
    assert all(s.count('janusdominic0@forensic-dgp-thesis:')==1 for s in lines)
    assert 'tmux new-session -A -s dgp_v28_preservation_diagnostic' in runbook
    assert '--protocol-sha '+pin+' --verify-transfer' in runbook and 'bash scripts/run_v28_preservation.sh '+pin in runbook
    assert p['budgets']['worker_seconds']==300 and p['budgets']['external_seconds']==330
    record={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':pin,
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'packet_files_verified':3,'parent_assets_verified':len(base['assets_sha256']),
        'closed_V28_evidence_bindings_verified':307,'old_seven_losses_and_failed_gates_retained':True,
        'actual_Windows_rejection_before_neural_or_gradients':True,'return_regressions_passed':9,'five_manual_steps_verified':True,
        'prospective_return_auditor_sha256':sha(ROOT/'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return.py'),
        'prospective_return_tests_sha256':sha(ROOT/'tests/test_cctv_dgp_v28_preservation_diagnostic_v1.py'),
        'runbook_sha256':sha(ROOT/'CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md'),
        'optimizer_updates':0,'local_gradient_calls':0,'actual_VM_execution_started':False,'new_training_recipe_created':False,
        'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(PREP/'independent_packet_audit.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
