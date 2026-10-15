"""Seal finite manual gradient diagnosis after audited frozen counterfactuals."""
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
PARENT=OUT/'cctv_dgp_bank_photometric_guard_v1_return'
PATH_ANALYSIS=OUT/'cctv_dgp_bank_path_ablation_v1_analysis'
PARTITION=OUT/'cctv_dgp_bank_partition_v1_analysis'
NAME='cctv_dgp_bank_loss_alignment_v1_vm'; STEM='cctv-dgp-bank-loss-alignment-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    dest=OUT/NAME; archive=OUT/(STEM+'-execution.tar.gz'); receipt=OUT/'cctv_dgp_bank_loss_alignment_v1_preparation.json'
    assert not any(path.exists() for path in [dest,archive,receipt])
    evidence=[PATH_ANALYSIS/n for n in ['import.json','independent_audit.json','limited_frozen_replay.json','visual_review.json','partition_state_audit.json']]
    evidence += [PARTITION/n for n in ['run_results.json','independent_audit.json','visual_review.json']]
    math_path=OUT/'cctv_dgp_bank_loss_alignment_v1_math_audit.json';evidence.append(math_path)
    assert all(read(f)['complete'] for f in evidence)
    audit=read(PATH_ANALYSIS/'independent_audit.json'); partition=read(PARTITION/'independent_audit.json')
    assert not any(g['raw']['pass'] or g['png']['pass'] for g in audit['partial_preview_gate_diagnostics'].values())
    assert not any(g['raw']['pass'] or g['png']['pass'] for g in partition['partial_preview_gate_diagnostics'].values())
    assert read(PATH_ANALYSIS/'visual_review.json')['viewed_sheets']==read(PARTITION/'visual_review.json')['viewed_sheets']==26
    previous=read(OUT/'cctv_dgp_bank_path_ablation_v1_return/protocol.json')
    old=read(PARENT/'protocol.json');required=previous['parent_required_sha256']
    assert len(required)==5852
    for name,expected in required.items():assert sha(PARENT/name)==expected,name
    math=read(math_path);states=read(PATH_ANALYSIS/'partition_state_audit.json')['variants']
    dest.mkdir();(dest/'scripts').mkdir()
    worker=ROOT/'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py'
    helper=ROOT/'scripts/cctv_dgp_bank_loss_alignment_v1.py'
    assert sha(helper)==math['helper_sha256']
    shutil.copyfile(worker,dest/'scripts'/worker.name);shutil.copyfile(helper,dest/helper.name)
    shell='''#!/usr/bin/env bash
set -uo pipefail
test -n "${TMUX:-}" || exit 1
test $# -eq 1 || exit 1
test ! -e outputs || { echo 'Immutable used diagnostic; no unchanged rerun'; exit 1; }
test ! -e review.log || exit 1
command -v timeout >/dev/null || exit 1
python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$1" --verify-transfer || exit 1
timeout --signal=TERM --kill-after=30 930 python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$1" --run 2>&1 | tee review.log
statuses=("${PIPESTATUS[@]}")
diagnostic_exit=${statuses[0]}
printf '%s\\n' "$diagnostic_exit" > review_exit_code.txt
printf 'Diagnostic exit code: %s\\n' "$diagnostic_exit"
timeout --signal=TERM --kill-after=30 330 python -B -u scripts/cctv_dgp_bank_loss_alignment_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha "$1" --export 2>&1 | tee export.log
statuses=("${PIPESTATUS[@]}")
export_exit=${statuses[0]}
printf '%s\\n' "$export_exit" > export_exit_code.txt
printf 'Export exit code: %s\\n' "$export_exit"
test "$export_exit" -eq 0 || exit "$export_exit"
exit "$diagnostic_exit"
'''
    (dest/'scripts/run_alignment.sh').write_text(shell,encoding='utf-8',newline='\n')
    p=dict(format='manual-loss-gradient-and-hypothetical-Adam-direction-diagnostic-v1',name=NAME,
        parent_name='cctv_dgp_bank_photometric_guard_v1_vm',parent_protocol_sha256=previous['parent_protocol_sha256'],
        parent_required_sha256=required,assets_sha256={f.relative_to(dest).as_posix():sha(f) for f in dest.rglob('*') if f.is_file()},
        states=['initial','joint50'],model_states={'initial':states['initial']['model_state'],'joint50':states['joint']['model_state']},
        case_ids=previous['case_ids'],paired_TRAIN_cases=100,TRAIN_references=20,
        sample_gradient_begins=previous['parity_begins'],sample_full_gradient_batches=4,
        learning_names=math['learning_names'],uncompressed_sampled_gradient_upper_bound_bytes=math['uncompressed_sampled_gradient_upper_bound_bytes'],
        component_names=['MSE','SSIM','ArcFace','detail','anchors','barrier','brightness','clear_MSE','degraded_MSE'],
        maximum_gradient_queries=360,model_forward_cap=80,recognizer_forward_cap=40,original_forward_cap=0,
        optimizer_updates=0,backwards=0,full_epochs=0,native_pixels_used=0,final_identity_pixels=0,
        worker_seconds=900,measurement_seconds=600,export_seconds=300,external_seconds=930,export_external_seconds=330,kill_grace_seconds=30,
        peak_VRAM_bytes=20*1024**3,minimum_free_bytes=4*1024**3,disk_reserve_bytes=1024**3,output_byte_cap=1024**3,
        projection_reference_batches=2,projection_safety_factor=1.5,gradient_vs_frozen_raw_parity_tolerance=1e-6,
        same_L4_frozen_control_raw_replay_exact=True,
        preserved_scientific_thresholds=old['scientific_thresholds'],original_checkpoint_sha256=old['original_checkpoint_sha256'],
        loss_and_optimizer_direction_configuration_unchanged=True,
        hypothetical_single_reference_Adam_direction_not_actual_accumulated_batch_step=True,
        failed_checkpoint_gradients_are_diagnosis_not_training_resume=True,
        full_gradient_audit_scope_is_sampled_not_all40=True,
        fixed_guarded_parent_model_source_only=True,no_training_worker_import=True,
        no_strength_optimization_or_new_checkpoint=True,manual_tmux_required=True,
        failed_run_resume_allowed=False,app_promotion=False,automatic_follow_on=False,
        native_unpaired=True,independent_final_review_pending=True,all_seven_completion_families_pending=True,goal_complete=False,
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in evidence})
    write(dest/'protocol.json',p);pin=sha(dest/'protocol.json')
    spec=importlib.util.spec_from_file_location('alignment_transfer_validator',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(dest,PARENT,pin)
    tree=ast.parse(worker.read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['step','backward','Adam','AdamW','SGD'] for n in ast.walk(tree))
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(dest.rglob('*')):
            if f.is_file():tar.add(f,arcname=NAME+'/'+f.relative_to(dest).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),name=NAME,stem=STEM,
        protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']),parent_required_assets=len(required),
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0,
        manual_VM_gradient_diagnostic_pending=True,no_agent_VM_launch=True,full_goal_incomplete=True))
    print(dict(complete=True,protocol_sha256=pin,archive_bytes=archive.stat().st_size,manual_VM_run_pending=True))


if __name__=='__main__':main()
