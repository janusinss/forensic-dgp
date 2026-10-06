"""Thin manual-L4 diagnostic from audited failed V25 states; no optimization."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from cctv_dgp_v25_gradient_diagnostic_v1_vm import sha,read,write

NEW=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm'
OUT=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_preparation'
PARENT=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25'
RETURN=ROOT/'outputs/cctv_dgp_spatial_features_v25_return'
NAME='cctv-dgp-v25-gradient-diagnostic-v1'
TERMS=['degraded_landmark_detail','degraded_observed_detail','degraded_pixel',
       'clear_baseline_anchor','pixel_regression','SSIM_regression','ArcFace_regression']
SHELL=r'''#!/usr/bin/env bash
set -u
set -o pipefail
diagnostic_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
diagnostic_pin="${1:?Supply the frozen protocol SHA256}"
diagnostic_python="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
test -x "$diagnostic_python" || exit 2
test ! -e "$diagnostic_root/outputs" || exit 3
test ! -e "$diagnostic_root/trainer.log" || exit 4
cd "$diagnostic_root" || exit 5
start_mono="$("$diagnostic_python" -c 'import time;print(time.monotonic())')"
timeout --signal=TERM --kill-after=30s 480s "$diagnostic_python" -B -u scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py --root "$diagnostic_root" --protocol-sha "$diagnostic_pin" --run 2>&1 | tee trainer.log
diagnostic_status=${PIPESTATUS[0]}
end_mono="$("$diagnostic_python" -c 'import time;print(time.monotonic())')"
printf 'Diagnostic exit code: %s\n' "$diagnostic_status"
printf '%s\n' "$diagnostic_status" > trainer_exit_code.txt
"$diagnostic_python" -B scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py --root "$diagnostic_root" --protocol-sha "$diagnostic_pin" --record-supervision --supervisor-start "$start_mono" --supervisor-end "$end_mono" --trainer-exit "$diagnostic_status"
supervision_status=$?
if [ "$supervision_status" -ne 0 ]; then exit "$supervision_status"; fi
timeout --signal=TERM --kill-after=10s 60s "$diagnostic_python" -B -u scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py --root "$diagnostic_root" --protocol-sha "$diagnostic_pin" --export
export_status=$?
printf 'Export exit code: %s\n' "$export_status"
if [ "$export_status" -ne 0 ]; then exit "$export_status"; fi
exit "$diagnostic_status"
'''


def main():
    start=time.monotonic();assert not NEW.exists() and not OUT.exists(),'Preserve previous diagnostic preparation'
    auditpath=ROOT/'outputs/cctv_dgp_spatial_features_v25_independent_audit.json';audit=read(auditpath)
    visualpath=ROOT/'outputs/cctv_dgp_spatial_features_v25_diagnostic/visual_review.json';visual=read(visualpath)
    losscheck=ROOT/'outputs/cctv_dgp_spatial_features_v25_loss_audit_v1/independent_saved_loss_audit.json'
    assert audit['complete'] and audit['VM_failure_present'] and not audit['early_structure_stop']['pass']
    assert visual['complete'] and visual['cases_reviewed']==50 and read(losscheck)['complete']
    assert sha(PARENT/'protocol.json')==audit['protocol_sha256']==sha(RETURN/'protocol.json')
    old=read(PARENT/'protocol.json')
    for name,digest in old['assets_sha256'].items():assert sha(PARENT/name)==digest,name
    source=ROOT/'scripts/cctv_dgp_v25_gradient_diagnostic_v1_vm.py'
    ast.parse(source.read_text(),feature_version=(3,10))
    NEW.mkdir();(NEW/'scripts').mkdir()
    shutil.copy2(source,NEW/'scripts'/source.name)
    with (NEW/'scripts/run_gradient.sh').open('x',encoding='utf-8',newline='\n') as f:f.write(SHELL)
    assets={n:sha(NEW/n) for n in ['scripts/'+source.name,'scripts/run_gradient.sh']}
    inputs=['outputs/failure.json','outputs/early_structure_stop.json','outputs/cohort_loss_setup.json',
        'outputs/frozen_DGP_features.json','outputs/one_batch_gradient_preflight.json','outputs/feature_path_gradient_update2.json',
        'outputs/execution_receipt.json','outputs/stopped_head.pth','supervisor_receipt.json','trainer_exit_code.txt']
    for update in [0,50]:
        inputs+=['outputs/update'+str(update)+'/'+n for n in ['head.pth','metrics.json']]
        inputs+=['outputs/update'+str(update)+'/'+c['id']+'.npy' for c in old['cases']]
    bindings={n:sha(RETURN/n) for n in inputs}
    p={'format':'own-DGP-V25-saved-state-gradient-diagnostic-v1','date':'2026-10-06',
        'purpose':'Diagnose actual unchanged V25 fixed-state gradients before another recipe; no quality acceptance',
        'assets_sha256':assets,'closed_V25_protocol_sha256':audit['protocol_sha256'],
        'closed_V25_return_sha256':'d32b3d68f5bc4e91858ab05c6cda1efda6109520a6776420cede603ba4d33434',
        'closed_V25_inputs_sha256':bindings,'snapshots':[0,50],'head_parameters':53781,'head_batches':20,
        'component_gradient_calls':140,'recognizer_forwards':120,'DGP_forwards':0,'optimizer_updates':0,
        'head_states':{'0':audit['VM_initial_state'],'50':audit['stopped_head_state']},
        'frozen_recognizer_state':'9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963',
        'terms':TERMS,'cohort_policy':'All same50 exposed TRAIN cases; two snapshots; ten sequential five-case batches each state; mean/10 per term to equal full50-case gradient.',
        'source_policy':'Compile only pinned head, host guard and five original objective/metric functions. Never import/execute the V25 optimizer or launcher.',
        'frozen_policy':'No optimizer, backward(), parameter update, new checkpoint, input/feature changes, threshold change or automatic follow-on.',
        'timing_limits':{'worker_seconds':420,'external_seconds':480,'external_kill_grace_seconds':30,
            'export_seconds':30,'external_export_seconds':60,'export_kill_grace_seconds':10,'free_disk_bytes':1024**3,'allocated_VRAM_bytes':20*1024**3},
        'prospective_return_checks':{'head_raw_parity_maximum':2e-6,'gradient_shape':[7,53781],'gradient_dtype':'float64',
            'finite_gradients':True,'head_and_frozen_states_unchanged':True,'exact_saved_raw_and_cache_bindings':True,
            'counts_and_timing_checked':True,'original_V25_failure_retained':True,
            'saved_gradient_arithmetic_only_local':True,'local_gradient_calls':0,'local_optimizer_updates':0,
            'CPU_saved_loss_scalar_tolerances':{'nonidentity_terms':2e-5,'ArcFace_regression':5e-4},
            'scalar_tolerance_scope':'Numerical fixed-state CPU/CUDA replay only; ArcFace5e-4 derives from original cosine5e-5 twice, multiplied by fixed penalty5. No quality threshold changes.'},
        'limitations':'Saved-state gradients do not reconstruct AdamW moments or its optimization trajectory. Exposed paired photographic TRAIN evidence only; no generalization, CCTV/local performance, hidden identity or quality acceptance.',
        'local_basis_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in [auditpath,visualpath,losscheck,
            ROOT/'outputs/cctv_dgp_spatial_features_v25_diagnostic/independent_saved_diagnostic_audit.json',
            ROOT/'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json']},
        'manual_existing_L4_only':True,'new_training_recipe':False,'app_promotion':False,'goal_complete':False}
    write(NEW/'protocol.json',p);pin=sha(NEW/'protocol.json')
    with (NEW/'protocol.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(pin+'  protocol.json\n')
    archive=ROOT/'outputs'/(NAME+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for file in sorted(NEW.rglob('*')):
            if not file.is_file():continue
            info=tar.gettarinfo(str(file),NEW.name+'/'+file.relative_to(NEW).as_posix())
            info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o755 if file.suffix=='.sh' else 0o644
            with file.open('rb') as f:tar.addfile(info,f)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    OUT.mkdir();receipt={'complete':True,'date':'2026-10-06','protocol_sha256':pin,'archive_sha256':digest,
        'archive_bytes':archive.stat().st_size,'archive_members':4,'assets':2,'parent_saved_input_bindings':len(bindings),
        'data_or_weights_reuploaded':False,'new_training_recipe':False,'optimizer_updates':0,'local_gradient_calls':0,
        'neural_calls':0,'VM_actions':False,'seconds':time.monotonic()-start,'app_promotion':False,'goal_complete':False}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
