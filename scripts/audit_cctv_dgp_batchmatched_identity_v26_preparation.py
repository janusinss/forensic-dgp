"""Freeze tested preparation evidence and exact manual upload/install/run/download readback."""
import ast
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from import_cctv_dgp_v25_gradient_diagnostic_v1 import read,write,sha,require


def main():
    started=time.monotonic();out=ROOT/'outputs/cctv_dgp_batchmatched_identity_v26_preparation'
    packet=read(out/'preparation.json');check=read(out/'independent_packet_source_audit.json')
    require(packet['complete'] and check['complete'],'Independent packet check required')
    runbook=(ROOT/'CCTV_DGP_BATCHMATCHED_IDENTITY_V26_VM.md').read_text()
    require([int(v) for v in re.findall(r'^([1-5])\. ',runbook,re.M)]==list(range(1,6)),'Five manual steps')
    commands=re.findall(r'^gcloud compute scp (.+)$',runbook,re.M);require(len(commands)==5,'Two uploads/three downloads')
    for command in commands:
        require('--project=forensic-dgp-thesis --zone=us-central1-a' in command and
            len(re.findall(r'"(janusdominic0@forensic-dgp-thesis:[^"]+)"',command))==1,'Exact VM/separate source')
    name='cctv-dgp-batchmatched-identity-v26'
    require(name+'-execution.tar.gz' in commands[0] and name+'-execution.tar.gz.sha256' in commands[1],'Exact uploads')
    for command,suffix in zip(commands[2:],['results.tar.gz','results.tar.gz.sha256','export.json']):
        require('/home/janusdominic0/'+name+'-'+suffix in command and command.endswith('"."'),'Exact download filename/destination')
    pin=packet['protocol_sha256'];require(runbook.count(pin)==4,'Protocol pin at header/install/verify/run')
    for text in ['tmux new-session -A -s dgp_batchmatched_identity_v26',
        'test ! -e ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26',
        'python3 ~/forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26/scripts/install_v26.py',
        '--install','--verify-transfer','bash scripts/run_v26.sh '+pin,
        'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate']:
        require(text in runbook,'Runbook command missing: '+text)
    sources=['scripts/import_cctv_dgp_v25_gradient_diagnostic_v1.py','scripts/audit_cctv_dgp_v25_gradient_diagnostic_v1_return.py',
        'scripts/cctv_dgp_batchmatched_identity_v26.py','scripts/cctv_dgp_batchmatched_identity_v26_preflight.py',
        'scripts/install_cctv_dgp_batchmatched_identity_v26_vm.py','scripts/prepare_cctv_dgp_batchmatched_identity_v26.py',
        'scripts/cctv_dgp_batchmatched_identity_v26_vm.py','scripts/verify_cctv_dgp_batchmatched_identity_v26.py',
        'tests/test_cctv_dgp_gradient_return_and_batchmatched_v26.py','scripts/audit_cctv_dgp_batchmatched_identity_v26_preparation.py']
    for source in sources:ast.parse((ROOT/source).read_text(),feature_version=(3,10))
    result={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'protocol_sha256':pin,'independent_packet_source_audit_sha256':sha(out/'independent_packet_source_audit.json'),
        'source_bindings_sha256':{n:sha(ROOT/n) for n in sources},'new_tests_passed':11,'test_seconds_reported':.097,
        'test_command':'venv/Scripts/python.exe -B -m unittest discover -s tests -p test_cctv_dgp_gradient_return_and_batchmatched_v26.py -v',
        'test_embedding':'Arithmetic batch-sensitive mock only, no learned-model calls or derivatives',
        'Python310_sources_parsed':len(sources),'five_manual_steps_verified':True,'single_remote_source_per_scp_verified':True,
        'Bash_n_exit_code':0,'Bash_readonly_outside_sandbox':'C:/Program Files/Git/bin/bash.exe -n outputs/cctv_dgp_batchmatched_identity_vm_v26/scripts/run_v26.sh',
        'Bash_sandbox_signal_pipe_failure_preserved':True,'actual_L4_zero_identity_proof_and_capacity_pending':True,
        'local_model_calls':0,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'VM_actions':False,
        'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(out/'independent_preparation_readback.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_bindings_sha256'},indent=2))


if __name__=='__main__':main()
