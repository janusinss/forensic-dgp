"""Preserve history and publish the audited failed V28 endpoint plus zero-update diagnostic."""
import json
from pathlib import Path
import time

from diagnose_cctv_dgp_active_original_decoder_v28_mean_control import sha,read,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone'
PREVIOUS=ROOT/'outputs/dgp_original_decoder_r2_audit_and_active_decoder_v28_milestone/milestone.json'
DOCS=('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md')


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    start=time.monotonic();assert not OUT.exists()
    assert sha(PREVIOUS)=='8347872326a0f9cd2c79b9ebfa86c8ce46b0698deb966f5eaba427f1847118cf'
    previous=read(PREVIOUS);assert verify(previous['new_evidence_sha256'])==586
    assert read(PREVIOUS.with_name('independent_readback.json'))['complete']
    audit=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json')
    assert audit['complete'] and audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert audit['members_verified']==838 and audit['saved_gradient_values_checked']==38394279
    visual=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_visual_review/visual_review.json')
    assert visual['complete'] and visual['cases_reviewed']==50 and visual['exact_source_cells']==200
    control=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/independent_readback.json')
    assert control['complete'] and control['fixed_control_is_insufficient'] and control['gate_failures_retained']==5
    prep=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_preparation'
    packet=read(prep/'independent_packet_audit.json');bash=read(prep/'bash_syntax_receipt.json')
    assert packet['complete'] and packet['return_regressions_passed']==9 and packet['actual_Windows_rejection_before_neural_or_gradients']
    assert bash['complete'] and bash['exit_code']==0 and bash['read_only']
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(app_path)
    assert sha(app_path)==previous['app_record_sha256'] and verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    OUT.mkdir();before=OUT/'before_docs';before.mkdir();mapping={}
    prefix='''**Latest research milestone — 6 October 2026: V28 completed800 but failed acceptance; all50 final faces reviewed; distinct zero-update preservation diagnostic verified for manual VM execution.**

The downloaded291,926,255-byte V28 archive matches
3196e8ab797763e7ced75a57411b973afc3b1b12654632b7662fab3e2e168aac.
Independent audit passes838 regular files,246 original assets,38,394,279
saved gradient values and every checkpoint/raw/PNG/metric at0/50/400/800.
CPU replay performs50 original-DGP,200 candidate-DGP and250 recognizer
forwards without local derivatives or updates. All selected12 initial CUDA
parity/gradient requirements pass. The VM completes800 updates/80 epochs:
fit91.850s, worker124.967s, terminal125.789s, peak allocated1,240,671,744 bytes.
The original checkpoint, encoder/FPN, inactive head4 and evaluation buffers
remain unchanged. The historical R2 all14 failure remains failed.

Final delivered degraded landmark-HF MSE improves18.0595%; both photograph
source gains are nonnegative. V28 still fails the fixed requirements: one
clear-source group losesSSIM and frozen ArcFace similarity, and a constant
RGB-mean shift explains71.3988% of the degraded pixel-MSE gain against20% max.
This fraction refers to pixel-MSE gain, not recovered identity or a percentage
of structural gain. necessary_capacity_pass remains false. Final800 only;
no earlier checkpoint selection, unchanged rerun or weakened gate is allowed.

All50 final TRAIN faces are viewed in ten exact-size sheets;200 saved cells
match their source arrays. Several degraded faces have clearer coarse eyes,
nose and mouth boundaries, with unresolved or altered finer eyes, glasses,
gaze and expression in strong degradation. Training capacity and this primary
assistant review do not establish generalization or independent final review.
Source labels do not imply ethnicity. Native unpaired evidence remains separate.

One fixed target/profile-independent RGB-mean control retains18.0237% landmark-HF
gain and reduces brightness fraction to1.6955%, but fails five preservation
checks. Independent readback verifies all50 arrays/PNGs/recognizer vectors and
17 groups. Only two control examples are visually inspected; no complete
control visual acceptance is claimed. This processing control is insufficient
and is not adopted. All original raw outputs, checkpoints and failures remain.

The next distinct VM diagnostic measures the final V28 original seven loss
gradients plus RGB-mean and clear-only SSIM/ArcFace diagnostic components.
It permits100 queries in ten fixed5-case batches, zero optimizer updates/epochs
or backwards, and no new model checkpoint. Extra components are measurements,
not a new training objective or selected weights. First-order derivatives do
not identify a unique trajectory cause. No actual VM run has occurred yet.
The300s worker/330s external+30s grace and120s export/150s external+30s grace
are finite; idle existing L4/g2-standard-4,2GiB free and allocatedVRAM<=20GiB.
Every source/state/parity/finite/time failure is retained; no resume or follow-on.

The28,332-byte three-file packet uploads no images or weights. It reuses307
closed V28 evidence bindings and246 original assets. Protocol SHA256:
a270f4631f00c55e8c0377f082bc5ec7d593b633c53f91126078a5452e3cf5b0.
Archive SHA256:d700acbb0bf6cdc03de9b6389132e77830937af518bc6ae563fee67698f11a51.
Python3.10 parsing, nine malformed-return/no-training regressions, actual
Windows rejection before neural/gradient work, read-only Bash syntax and five
manual command steps pass. The initial local AST save-check failure occurred
before packet creation/VM work; its source/evidence is preserved. The corrected
check specifically rejects Torch checkpoint/optimizer calls. Human transfer,
SSH and tmux remain the execution path; no assistant cloud action occurs.

Previous586 bindings and deeper50/668/cleanup60/309/66/692/299/697/513 history,
three full document bodies and app22 bindings remain intact. The existing app
model/design stays in place. Original checkpoints, splits, scientific caches,
logs and all prior gate failures remain. No new native or reserved-final pixels
are opened. Useful native restoration, all seven automatic/assisted covering
families, meaningful full app verification and independent final review remain
required. No Zamboanga or exact hidden-identity claim. Goal active/incomplete.

[Audited V28 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_RESULTS.md>) ·
[Next five manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md>)

Earlier bodies below are retained history. V28-ready/pending wording and its
training commands are superseded by the rejected final return; only the distinct
zero-update diagnostic is ready for manual execution.

'''
    for name in DOCS:
        original=(ROOT/name).read_bytes()
        with (before/name).open('xb') as f:f.write(original)
        mapping[name]=(before/name).relative_to(ROOT).as_posix()
        split=original.index(b'\n')+1
        (ROOT/name).write_bytes(original[:split]+b'\n'+prefix.encode('utf-8')+original[split:])
    verify(previous['new_evidence_sha256'],mapping)
    evidence={}
    def bind(path):evidence[path.relative_to(ROOT).as_posix()]=sha(path)
    names=[*DOCS,'CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_RESULTS.md','CCTV_DGP_V28_PRESERVATION_DIAGNOSTIC_V1_VM.md',
        'scripts/prepare_cctv_dgp_active_original_decoder_v28_visual_review.py',
        'scripts/record_cctv_dgp_active_original_decoder_v28_visual_review.py',
        'scripts/diagnose_cctv_dgp_active_original_decoder_v28_mean_control.py',
        'scripts/verify_cctv_dgp_active_original_decoder_v28_mean_control.py',
        'scripts/prepare_cctv_dgp_v28_preservation_diagnostic_v1.py',
        'scripts/verify_cctv_dgp_v28_preservation_diagnostic_v1_packet.py',
        'scripts/audit_cctv_dgp_v28_preservation_diagnostic_v1_return.py',
        'tests/test_cctv_dgp_v28_preservation_diagnostic_v1.py',
        'scripts/record_cctv_dgp_v28_return_and_preservation_diagnostic_v1_milestone.py',
        'scripts/verify_cctv_dgp_v28_return_and_preservation_diagnostic_v1_milestone.py',
        'outputs/cctv-dgp-active-original-decoder-v28-results.tar.gz',
        'outputs/cctv-dgp-active-original-decoder-v28-results.tar.gz.sha256',
        'outputs/cctv-dgp-active-original-decoder-v28-export.json',
        'outputs/cctv_dgp_active_original_decoder_v28_return_import.json',
        'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json',
        'outputs/cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz',
        'outputs/cctv-dgp-v28-preservation-diagnostic-v1-execution.tar.gz.sha256',
        PREVIOUS.relative_to(ROOT).as_posix(),PREVIOUS.with_name('independent_readback.json').relative_to(ROOT).as_posix()]
    for name in names:bind(ROOT/name)
    for directory in [ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return',
        ROOT/'outputs/cctv_dgp_active_original_decoder_v28_visual_review',
        ROOT/'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1',
        ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_vm',
        ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_prepare_regression_failure',prep,before]:
        for path in directory.rglob('*'):
            if path.is_file():bind(path)
    m={'complete':True,'date':'2026-10-06','scope':'V28 completed800 rejected; whole50-case review; fixed mean control rejected; new zero-update manual diagnostic ready',
       'new_evidence_sha256':evidence,'previous_milestone_sha256':sha(PREVIOUS),'previous586_original_locations':mapping,
       'original_document_bodies_preserved':3,'app_record_sha256':sha(app_path),'app22_bindings_preserved':True,
       'V28_archive_sha256':audit['archive_sha256'],'V28_archive_bytes':audit['archive_bytes'],
       'V28_updates':800,'V28_epochs':80,'V28_necessary_capacity_pass':False,'V28_gate_failures_retained':3,
       'V28_training_cases_visually_reviewed':50,'V28_saved_gradient_values_audited':38394279,
       'mean_control_gate_pass':False,'mean_control_gate_failures_retained':5,'mean_control_independent_readback_pass':True,
       'diagnostic_protocol_sha256':packet['protocol_sha256'],'diagnostic_archive_sha256':packet['archive_sha256'],
       'diagnostic_archive_bytes':packet['archive_bytes'],'diagnostic_gradient_queries_bound':100,'diagnostic_optimizer_updates':0,
       'diagnostic_actual_VM_execution_started':False,'human_manual_VM_execution_required':True,'return_regressions_passed':9,
       'new_finite_training_protocol_created':False,'all_old_failed_gates_preserved':True,
       'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'VM_actions':False,
       'native_or_reserved_used':False,'independent_final_review_complete':False,'app_promotion':False,
       'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    write(OUT/'milestone.json',m)
    print(json.dumps({k:m[k] for k in ['complete','V28_necessary_capacity_pass','diagnostic_archive_bytes','diagnostic_actual_VM_execution_started','goal_status','seconds']},indent=2))


if __name__=='__main__':main()
