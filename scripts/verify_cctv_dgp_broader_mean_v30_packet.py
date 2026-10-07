"""Independent metadata/transfer/guard/runbook check before human VM training."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_broader_mean_vm_v30'
PREP=ROOT/'outputs/cctv_dgp_broader_mean_v30_preparation'
PIN='b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    started=time.monotonic();assert sha(BUNDLE/'protocol.json')==PIN
    p=read(BUNDLE/'protocol.json');old=read(ROOT/'outputs/cctv_dgp_mean_centered_decoder_vm_v29/protocol.json')
    for key in ['optimizer','optimizer_updates','terms','parameter_layout','retained_capacity_gates',
                'initial_preservation_terms_exact_zero','original_checkpoint_sha256','original_DGP_state','frozen_recognizer_state']:
        assert p[key]==old[key],key
    assert p['initial_proof_case_rows']==old['case_rows'] and len(p['case_rows'])==3905 and len(p['training_references'])==781
    assert p['epochs']==800/781 and p['training_batches_per_epoch']==781 and p['component_gradient_calls']==70
    assert p['budgets']['cache_seconds']==900 and p['budgets']['fit_seconds']==3600 and p['budgets']['worker_seconds']==4500
    assert p['budgets']['minimum_free_disk_bytes']==6*1024**3 and p['budgets']['peak_vram_bytes']==20*1024**3
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    mixed=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2';v=read(mixed/'mixed_protocol_v9.json')
    assert sha(mixed/'mixed_protocol_v9.json')==p['mixed_data_protocol_sha256']
    assert p['training_references']==[r for r in v['references'] if r['role']=='train']
    byid={c['id']:c for c in p['case_rows']};assert len(byid)==3905
    for c in v['training_cases']:
        assert all(byid[c['id']][k]==value for k,value in c.items()) and byid[c['id']]['role']=='train'
    assert len(p['mixed_TRAIN_assets_sha256'])==5467 and all(name.startswith('data/') for name in p['mixed_TRAIN_assets_sha256'])
    for name,digest in p['mixed_TRAIN_assets_sha256'].items():assert sha(mixed/name)==digest==v['assets_sha256'][name]
    dev=read(ROOT/'outputs/cctv_dgp_generalization_vm_v15/generalization_protocol_v15.json')['references']
    for key in ['id','source_sha256','target_rgb_sha256']:
        assert {r[key] for r in dev}.isdisjoint({r[key] for r in p['training_references']}),key
    assert {r['hq_source_sha256'] for r in dev if 'hq_source_sha256' in r}.isdisjoint({r['hq_source_sha256'] for r in p['training_references'] if 'hq_source_sha256' in r})
    parent=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'
    for c in p['initial_proof_case_rows']:
        fresh=byid[c['id']]
        for key in ['input','target','observed']:assert sha(parent/c[key])==sha(mixed/fresh[key])
        assert fresh['landmarks5_canvas_xy']==c['landmarks5_canvas_xy']
    for folder,key in [('outputs/cctv_dgp_feature_skips_v27_return','closed_V27_evidence_sha256'),
                       ('outputs/cctv_dgp_original_decoder_gradient_v1_r2_return','closed_R2_evidence_sha256'),
                       ('outputs/cctv_dgp_active_original_decoder_v28_return','closed_V28_evidence_sha256'),
                       ('outputs/cctv_dgp_v28_preservation_diagnostic_v1_return','closed_diagnostic_evidence_sha256'),
                       ('outputs/cctv_dgp_mean_centered_decoder_v29_return','closed_V29_evidence_sha256')]:
        for name,digest in p[key].items():assert sha(ROOT/folder/name)==digest,(folder,name)
    schedule=read(BUNDLE/'schedule.json');batches=schedule['batches']
    assert len(batches)==800 and all(len(b)==len(set(b))==5 and all(0<=i<3905 for i in b) for b in batches)
    assert sorted(i for b in batches[:781] for i in b)==list(range(3905))
    assert len({i for b in batches[781:] for i in b})==95
    assert sha(BUNDLE/'cctv_dgp_mean_centered_decoder_v29.py')==old['assets_sha256']['cctv_dgp_mean_centered_decoder_v29.py']
    assert sha(BUNDLE/'cctv_dgp_app_input_v28.py')==old['assets_sha256']['cctv_dgp_app_input_v28.py']
    parsed=[]
    for path in [*BUNDLE.rglob('*.py'),ROOT/'scripts/audit_cctv_dgp_broader_mean_v30_return.py']:
        ast.parse(path.read_text(),feature_version=(3,10));parsed.append(path.relative_to(ROOT).as_posix())
    archive=ROOT/'outputs/cctv-dgp-broader-mean-v30-execution.tar.gz';prep=read(PREP/'preparation.json')
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']==746356
    assert Path(str(archive)+'.sha256').read_text().strip().split()==[sha(archive),archive.name]
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==8 and all(m.isfile() for m in members)
        for m in members:
            assert m.name.startswith(BUNDLE.name+'/')
            name=m.name[len(BUNDLE.name)+1:]
            with tar.extractfile(m) as f: digest=hashlib.sha256(f.read()).hexdigest()
            assert digest==sha(BUNDLE/name)
    worker=BUNDLE/'scripts/cctv_dgp_broader_mean_v30_vm.py'
    rejected=subprocess.run([sys.executable,'-B',str(worker),'--root',str(BUNDLE),'--protocol-sha',PIN,'--run'],capture_output=True,text=True,timeout=30)
    assert rejected.returncode!=0 and 'Existing Linux VM only; no local gradients' in rejected.stderr and not (BUNDLE/'outputs').exists()
    write(PREP/'windows_training_guard.json',{'complete':True,'worker_sha256':sha(worker),'exit_code':rejected.returncode,
          'failure_expected':True,'before_neural_or_gradient_or_optimizer_work':True,'outputs_created':False,
          'stderr':rejected.stderr,'VM_actions':False})
    bash=Path('C:/Program Files/Git/bin/bash.exe');assert bash.is_file()
    shell=BUNDLE/'scripts/run_v30.sh';syntax=subprocess.run([str(bash),'-n',str(shell)],capture_output=True,text=True,timeout=30)
    assert syntax.returncode==0
    write(PREP/'bash_syntax_receipt.json',{'complete':True,'shell_sha256':sha(shell),'exit_code':0,'read_only':True,'stdout':syntax.stdout,'stderr':syntax.stderr})
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_cctv_dgp_broader_mean_v30_return.py'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert tests.returncode==0 and 'Ran 7 tests' in tests.stderr
    write(PREP/'test_receipt.json',{'complete':True,'tests_passed':7,'test_sha256':sha(ROOT/'tests/test_cctv_dgp_broader_mean_v30_return.py'),
          'checker_sha256':sha(ROOT/'scripts/audit_cctv_dgp_broader_mean_v30_return.py'),'stdout':tests.stdout,'stderr':tests.stderr,'local_neural_or_gradient_calls':0})
    runbook=ROOT/'CCTV_DGP_BROADER_MEAN_V30_VM.md';text=runbook.read_text()
    assert all('\n'+str(i)+'. ' in text for i in range(1,6)) and '\n6. ' not in text
    assert text.count('gcloud compute scp ')==5 and text.count('--zone=us-central1-a')==5
    assert 'tmux new-session -A -s dgp_v30' in text and text.count(PIN)==3 and sha(archive) in text
    for line in text.splitlines():
        if 'gcloud compute scp ' in line:assert line.count('janusdominic0@forensic-dgp-thesis:')==1
    for name in ['cctv-dgp-broader-mean-v30-results.tar.gz','cctv-dgp-broader-mean-v30-results.tar.gz.sha256','cctv-dgp-broader-mean-v30-export.json']:assert name in text
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':PIN,'archive_sha256':sha(archive),
             'archive_bytes':archive.stat().st_size,'packet_files_verified':8,'TRAIN_assets_verified':5467,
             'initial50_source_parity_exact':True,'104_DEV_references_disjoint_by_reference_source_and_target':True,
             'historical_person_pretrained_overlap_unknown':True,'optimizer_losses_model_and_numeric_gates_unchanged':True,
             'all3905_cases_exposed_once_before_second_epoch':True,'finite_updates':800,'Python310_sources_parsed':parsed,
             'return_boundary_regressions_passed':7,'actual_Windows_training_rejection_before_neural_work':True,
             'read_only_Bash_syntax_pass':True,'five_manual_steps_verified':True,'runbook_sha256':sha(runbook),
             'prospective_return_auditor_sha256':sha(ROOT/'scripts/audit_cctv_dgp_broader_mean_v30_return.py'),
             'local_neural_or_gradient_calls':0,'local_optimizer_updates':0,'VM_actions':False,
             'actual_VM_training_started':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(PREP/'independent_packet_audit.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
