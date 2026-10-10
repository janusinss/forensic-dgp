"""Preserve an unrun draft; fix prospective replay paths in a distinct R1 packet."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/cctv_dgp_group_guard_probe_v35_preparation'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def main():
    source=ROOT/'scripts/audit_cctv_dgp_group_guard_probe_v35_return.py'
    old=source.read_text(encoding='utf-8')
    pinned=(ROOT/'scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r2.py').read_text(encoding='utf-8')
    assert "PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'" in pinned
    assert "ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'" in pinned
    assert "PARENT=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'" in old
    assert "ACTIVE=ROOT/'outputs/cctv_dgp_batchmatched_identity_vm_v26'" in old
    corrected=old.replace("PARENT=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'", "PARENT=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'")
    corrected=corrected.replace("ACTIVE=ROOT/'outputs/cctv_dgp_batchmatched_identity_vm_v26'", "ACTIVE=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'")
    corrected=corrected.replace('cctv_dgp_group_guard_probe_v35_vm','cctv_dgp_group_guard_probe_v35_r1_vm')
    corrected=corrected.replace('cctv_dgp_group_guard_probe_v35_return','cctv_dgp_group_guard_probe_v35_r1_return')
    corrected=corrected.replace('cctv-dgp-group-guard-probe-v35','cctv-dgp-group-guard-probe-v35-r1')
    corrected=corrected.replace('cctv_dgp_group_guard_probe_v35_independent_audit','cctv_dgp_group_guard_probe_v35_r1_independent_audit')
    checker=ROOT/'scripts/audit_cctv_dgp_group_guard_probe_v35_r1_return.py'
    with checker.open('x',encoding='utf-8',newline='\n') as stream:stream.write(corrected)
    builder=ROOT/'scripts/prepare_cctv_dgp_group_guard_probe_v35.py'
    text=builder.read_text(encoding='utf-8')
    text=text.replace('cctv_dgp_group_guard_probe_v35_vm','cctv_dgp_group_guard_probe_v35_r1_vm')
    text=text.replace('cctv-dgp-group-guard-probe-v35','cctv-dgp-group-guard-probe-v35-r1')
    text=text.replace('cctv_dgp_group_guard_probe_v35_preparation','cctv_dgp_group_guard_probe_v35_r1_preparation')
    text=text.replace('audit_cctv_dgp_group_guard_probe_v35_return.py','audit_cctv_dgp_group_guard_probe_v35_r1_return.py')
    text=text.replace('CCTV_DGP_GROUP_GUARD_PROBE_V35_VM.md','CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_VM.md')
    revised=ROOT/'scripts/prepare_cctv_dgp_group_guard_probe_v35_r1.py'
    ast.parse(corrected,feature_version=(3,10));ast.parse(text,feature_version=(3,10))
    with revised.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    files=[source,builder,ROOT/'scripts/prepare_cctv_dgp_v35_prospective_auditor.py',ROOT/'CCTV_DGP_GROUP_GUARD_PROBE_V35_VM.md',
        ROOT/'outputs/cctv-dgp-group-guard-probe-v35-execution.tar.gz',ROOT/'outputs/cctv-dgp-group-guard-probe-v35-execution.tar.gz.sha256']
    files.extend(q for q in sorted((ROOT/'outputs/cctv_dgp_group_guard_probe_v35_vm').rglob('*')) if q.is_file())
    write(PREP/'unrun_draft_readback_failure.json',{'complete':True,'cause':'Two prospective CPU replay directories differ from the pinned, previously validated V33 replay basis',
        'failed_location':source.relative_to(ROOT).as_posix(),'detected_before_any_VM_transfer_or_model_call':True,
        'invalid_assumption':'Shared checkpoint digest implies the parent and active module directories may be substituted',
        'original_files_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in files},
        'corrected_replay_checker_sha256':sha(checker),'revision_builder_sha256':sha(revised),
        'recipe_changed':False,'quality_gates_changed':False,'original_draft_to_be_run':False,
        'VM_calls':0,'neural_or_gradient_calls':0,'optimizer_updates':0})
    print(json.dumps({'complete':True,'draft_retained':True,'R1_recipe_changed':False}))


if __name__=='__main__':main()
