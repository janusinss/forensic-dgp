"""Preserve full history and record the audited diagnostic plus one manual pilot."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_v28_preservation_diagnostic_and_mean_centered_v29_milestone'
PREVIOUS=ROOT/'outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone/milestone.json'
DOCS=('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    started=time.monotonic();assert not OUT.exists()
    assert sha(PREVIOUS)=='5e163e129b438081214b0914845a3204431abc70751457b101dcd368ce87ac86'
    previous=read(PREVIOUS);assert verify(previous['new_evidence_sha256'])==1041
    assert read(PREVIOUS.with_name('independent_readback.json'))['complete']
    audit=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit_r1.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['members_verified']==322
    assert audit['saved_gradient_values_checked']==54848970 and audit['optimizer_updates']==audit['local_gradient_calls']==0
    analysis=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis/results.json')
    assert analysis['mean_anchor_increases_along_plain_SGD_direction_at_final_state']
    prep=ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_preparation'
    packet=read(prep/'independent_packet_audit.json');bash=read(prep/'bash_syntax_receipt.json')
    assert packet['complete'] and packet['return_and_projection_regressions_passed']==17
    assert packet['exact_app_input_encoder_and_initial50_projection_parity'] and packet['five_manual_steps_verified']
    assert bash['complete'] and bash['exit_code']==0 and bash['read_only']
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(app_path)
    assert sha(app_path)==previous['app_record_sha256'] and verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    OUT.mkdir();before=OUT/'before_docs';before.mkdir();mapping={}
    prefix='''**Latest research milestone — 6 October 2026: V28 final-state diagnostic audited; distinct mean-centered V29 finite pilot verified for human VM execution.**

The downloaded211,650,756-byte zero-update diagnostic matches SHA256
7ed57b598096e2bf52302b5acba45626a12ec66f17ca3d94f565e704d170cc75.
The R1 independent audit verifies322 regular files and54,848,970 saved gradient
values, displacement and per-component/per-tensor arithmetic. CPU replay uses
50 original-DGP,50 final-DGP and100 recognizer forwards with no local derivatives.
All50 fresh VM initial/final raw and vector parity maxima are exactly zero;
final PNGs match exactly. The VM performs100 gradient queries in30.042s,
zero optimizer updates/backwards/epochs and no new checkpoint. All states remain.

The original local audit stopped on two one-ULP Linux/Windows norm differences
(max5.55e-17). Its original source and failure are retained. The distinct R1
checker allows rtol1e-12/atol1e-14 only for derived arithmetic. Saved arrays,
displacement, archive/source hashes and PNGs still require exact matches.
Four repair regressions pass. No model-quality gate changes.

At the final V28 state, the existing seven-term objective's plain negative
gradient increases the measured RGB-mean penalty (directional derivative
+1.998307315), while clear SSIM and ArcFace hinge penalties decrease.
Preservation gradients are active. This is local first-order evidence, not a
finite-step/AdamW guarantee or a unique historical cause. V28's18.0595% TRAIN
structure gain still fails three gates; the post-training mean control still
fails five. Both remain rejected, with no app promotion or earlier selection.

V29 tests one changed spatial path: subtract the observed RGB mean of candidate
minus frozen same-input original-DGP baseline BEFORE unchanged losses and final
clamping. All profiles use that path without target/source/identity routing.
No learned projection parameters, strength selection, loss-weight change or
surrogate derivatives. Clipping can reintroduce mean shift; the original
brightness and all17 preservation groups still decide acceptance. Original
selected12 decoder tensors train from the original checkpoint in a separate
copy; encoder/FPN/inactive head4 and all evaluation buffers stay frozen.

The schedule/optimizer remain800 updates/80 epochs, AdamW lr0.00003/WD0.01,
clip1, snapshots0/50/400/800, selected12 initial70-query proof before optimizer.
Early50 structure gain>=1%, final>=10%, both source gains>=0, all17 groups and
brightness fraction<=20% remain. Final800 only. No unchanged historical run,
resume, gate weakening, competing-task termination or automatic follow-on.

V29's seven-file30,293-byte packet uploads no images or weights. Protocol:
77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f.
Archive:de158e52c44494638c0477cd7fca2a67ed12f18ad40a9fc38e9ff45cdac67e4d.
It verifies246 original assets plus11 closed V28 and9 diagnostic dependencies.
Python3.10 parsing,17 forward-only/return regressions, all50 exact initial
projection cases, float64 reference arithmetic(max5.96e-8), actual Windows
pre-neural/gradient rejection, readonly Bash syntax and five manual steps pass.
Two local unissued test setup failures remain recorded; no VM work or local
gradients occurred. Actual V29 training has not started. Human gcloud upload,
SSH and tmux remain the only training execution path. Preflight300s, fit1500s,
worker1800s, external2100s+30s grace, export120s/external150s+30s grace;
idle existing L4/g2-standard-4,3GiB free, allocatedVRAM<=20GiB.

All prior1041 bindings and deeper586/50/668/cleanup60/309/66/692/299/697/513
history, three full document bodies and app22 bindings remain preserved.
No new native or reserved-final pixels are opened. These50 paired photographic
TRAIN cases do not establish native CCTV, ethnicity or Zamboanga performance.
V29 return audit and all50-face review must precede separately frozen native/app
qualification. Useful DGP restoration, all seven automatic/assisted covering
families, full app flow and independent final review remain required. The
existing app model/design stays in place. Goal active/incomplete.

[Diagnostic results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_RESULTS.md>) ·
[V29 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md>)

Earlier bodies below are retained history. Diagnostic-ready/pending wording is
superseded by this audited return; only the distinct V29 pilot is ready for
manual execution. V28 and the fixed mean control remain rejected.

'''
    for name in DOCS:
        original=(ROOT/name).read_bytes()
        with (before/name).open('xb') as f:f.write(original)
        mapping[name]=(before/name).relative_to(ROOT).as_posix()
        split=original.index(b'\n')+1
        (ROOT/name).write_bytes(original[:split]+b'\n'+prefix.encode('utf-8')+original[split:])
    assert verify(previous['new_evidence_sha256'],mapping)==1041
    evidence={}
    def bind(path):evidence[path.relative_to(ROOT).as_posix()]=sha(path)
    names=[*DOCS,'CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_RESULTS.md','CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md',
        'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return_r1.py',
        'scripts/analyse_cctv_dgp_v28_preservation_diagnostic_v1.py',
        'scripts/cctv_dgp_mean_centered_decoder_v29.py','scripts/prepare_cctv_dgp_mean_centered_decoder_v29.py',
        'scripts/verify_cctv_dgp_mean_centered_decoder_v29_packet.py','scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py',
        'tests/test_cctv_dgp_v28_preservation_audit_r1.py','tests/test_cctv_dgp_mean_centered_decoder_v29.py',
        'tests/test_cctv_dgp_mean_centered_decoder_v29_audit.py',
        'scripts/record_cctv_dgp_v28_diagnostic_and_mean_centered_v29_milestone.py',
        'scripts/verify_cctv_dgp_v28_diagnostic_and_mean_centered_v29_milestone.py',
        'outputs/cctv-dgp-v28-preservation-diagnostic-v1-results.tar.gz',
        'outputs/cctv-dgp-v28-preservation-diagnostic-v1-results.tar.gz.sha256',
        'outputs/cctv-dgp-v28-preservation-diagnostic-v1-export.json',
        'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return_import.json',
        'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit_r1.json',
        'outputs/cctv-dgp-mean-centered-decoder-v29-execution.tar.gz',
        'outputs/cctv-dgp-mean-centered-decoder-v29-execution.tar.gz.sha256',
        PREVIOUS.relative_to(ROOT).as_posix(),PREVIOUS.with_name('independent_readback.json').relative_to(ROOT).as_posix()]
    for name in names:bind(ROOT/name)
    for directory in [ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return',
        ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_audit_repair_r1',
        ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis',
        ROOT/'outputs/cctv_dgp_mean_centered_v29_draft_regression_failure',
        ROOT/'outputs/cctv_dgp_mean_centered_v29_return_test_setup_failure',
        ROOT/'outputs/cctv_dgp_mean_centered_decoder_vm_v29',prep,before]:
        for path in directory.rglob('*'):
            if path.is_file():bind(path)
    milestone={'complete':True,'date':'2026-10-06','scope':'V28 diagnostic independently audited; single changed mean-centered V29 finite pilot ready for human execution',
        'new_evidence_sha256':evidence,'previous_milestone_sha256':sha(PREVIOUS),'previous1041_original_locations':mapping,
        'original_document_bodies_preserved':3,'app_record_sha256':sha(app_path),'app22_bindings_preserved':True,
        'diagnostic_archive_sha256':audit['archive_sha256'],'diagnostic_archive_bytes':211650756,'diagnostic_members_verified':322,
        'diagnostic_gradient_queries':100,'diagnostic_saved_values_audited':54848970,'diagnostic_optimizer_updates':0,
        'original_cross_platform_audit_failure_retained':True,'diagnostic_distinct_R1_audit_pass':True,
        'V28_rejected_gates_retained':3,'mean_control_rejected_gates_retained':5,
        'V29_protocol_sha256':packet['protocol_sha256'],'V29_archive_sha256':packet['archive_sha256'],'V29_archive_bytes':30293,
        'V29_update_bound':800,'V29_epoch_bound':80,'V29_preflight_gradient_queries_bound':70,
        'V29_gates_optimizer_schedule_and_loss_weights_unchanged':True,'V29_regressions_passed':17,
        'V29_actual_VM_training_started':False,'human_manual_VM_execution_required':True,'new_finite_training_protocol_created':True,
        'all_old_failed_gates_preserved':True,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'VM_actions':False,'native_or_reserved_used':False,'independent_final_review_complete':False,'app_promotion':False,
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-started}
    write(OUT/'milestone.json',milestone)
    print(json.dumps({k:milestone[k] for k in ['complete','diagnostic_saved_values_audited','V29_archive_bytes','V29_actual_VM_training_started','goal_status','seconds']},indent=2))


if __name__=='__main__':main()
