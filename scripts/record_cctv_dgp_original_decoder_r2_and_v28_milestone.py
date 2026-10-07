"""Close the genuine R2 failure and publish only the new finite manual V28 pilot."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_original_decoder_r2_audit_and_active_decoder_v28_milestone'
PREVIOUS=ROOT/'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json'
DOCS=('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def verify(bindings,mapping=None):
    for name,digest in bindings.items():
        path=(ROOT/(mapping or {}).get(name,name)).resolve()
        assert path.is_relative_to(ROOT) and sha(path)==digest,name
    return len(bindings)


def main():
    start=time.monotonic();assert not OUT.exists()
    assert sha(PREVIOUS)=='6b97fcbaf2f4e3709b7b2a4989ff60ffd86464c5120af6ef8752c82fd41a8e4f'
    previous=read(PREVIOUS);assert verify(previous['new_evidence_sha256'])==50
    assert read(PREVIOUS.with_name('independent_readback.json'))['complete']
    audit=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json')
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['VM_proof_complete']
    head=read(ROOT/'outputs/cctv_dgp_original_decoder_head4_review_v1/results.json')
    head_checked=read(ROOT/'outputs/cctv_dgp_original_decoder_head4_review_v1/independent_readback.json')
    assert head['complete'] and head_checked['complete'] and head['unchanged_CPU_head4_outputs_exact_zero_cases']==50
    normalized=read(ROOT/'outputs/cctv_dgp_decoder_app_normalization_v1/independent_readback.json')
    assert normalized['complete'] and normalized['exact_CPU_byte_encoding_equivalence']
    prep=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_preparation'
    packet=read(prep/'independent_packet_audit_final.json');tests=read(prep/'test_receipt.json')
    assert packet['complete'] and tests['complete'] and tests['tests_passed']==9 and tests['bash_syntax_exit_code']==0
    cpu=read(prep/'cpu_initial_replay_regression.json');assert cpu['complete'] and not cpu['actual_V28_VM_result']
    prototype=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_review/results.json')
    assert prototype['complete'] and prototype['initial_actual_app_reference_parity_cases']==50
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(app_path)
    assert sha(app_path)==previous['app_record_sha256'] and verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    OUT.mkdir();before=OUT/'before_docs';before.mkdir();mapping={}
    prefix='''**Latest research milestone — 6 October 2026: R2 all14 failure audited; inactive head4 traced; distinct V28 active-decoder pilot verified for manual execution.**

The downloaded93,557,954-byte R2 archive matches
dfd8e661126d144e87875f236a7eedb339f7803a65f57b92aa0fd128110d4e3e.
Safe import verifies172 regular files. Independent audit checks46,909,863 saved
gradient values,246 original assets,50 original-DGP CPU forwards,50 recognizer
forwards and50 cohort rows. All70 L4 queries ran, then the unchanged all14 gate
failed at line263. Head4's two tensors are exactly zero in every component and
every batch; the other12 aggregate improvement gradients are nonzero. The four
initial preservation terms/gradients are correctly zero. No optimizer, updates,
epochs or new checkpoint occurred. Worker25.232s/supervisor26.416s; no timeout.
Export completion is packaging only. R2 remains failed and must not rerun.

Forward-only trace: every110,592 head4 kernel value is nonzero float32 subnormal
(maximum about6.31e-40). The fourth map is nonzero, but head4 outputs exactly zero
in all50 cases. Separate float64 calculations give2.49e-76–4.29e-76 maxima, below
float32's smallest positive value1.40e-45. Independent NumPy readback verifies
350 activation arrays/100 convolutions. No local derivatives or model changes.
No unique checkpoint-history cause or explanation of all prior failures is claimed.
The fourth map still contributes through the FPN top-down path to other heads.

The actual app encoder divides in NumPy before device transfer; R2 used the older
Torch division after transfer. Both are exactly equal on all256 byte values and
all50 current CPU inputs/raw/PNGs: no CPU normalization defect or gain. CUDA
encoding equivalence was not measured. Earlier canonical wording is corrected
in scope without editing old receipts. V28 ships the exact actual app encoder AST.

The selected separate-original-decoder direction continues as distinct V28:
train head1–head3,smooth,smooth2,final only; keep head4, the encoder/FPN and all
stored buffers frozen. The unchanged original forward has498,627 trainable
parameters in12 tensors. Initial actual-app raw/PNG parity passes all50 CPU
inputs; original state remains unchanged. This does not waive or accept R2 all14.
V28 first requires70 fresh same5-case CUDA gradient queries: every selected12
improvement gradient finite/connected/nonzero, all initial preservation values
and gradients exactly zero, exact raw/PNG baseline parity before any optimizer.

V28 is bounded800 updates/80 epochs; AdamW fixed0.00003, weight decay0.01 and
clip1. Snapshots0/50/400/800; stop at50 if structure gain<1%. Final gain>=10%, both
source nonregression, original17-group MSE/SSIM/ArcFace bounds and brightness
fraction<=20% remain. No checkpoint selection before final800; even capacity
pass requires all50-face review and separately frozen broader/native/app work.
Require idle existing L4/g2-standard-4 and3GiB free. Preflight300s, fit1500s,
worker1800s; supervisor2100s+30s grace; export120s/external150s+30s grace;
torch allocated VRAM<=20GiB. Preserve every failure; no automatic follow-on,
overwrite, resume, failed-recipe rerun or threshold search. Training remains
human transfer/SSH/tmux only. No assistant VM/cloud action occurs.

The29,527-byte eight-file packet uploads no original data or weights.
Archive SHA256:e1484a675ad4330e4615d2f58a70f66ba8a8ad287b78cae66f8555ec4c4b1614.
Protocol SHA256:27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8.
Python3.10 parsing, source/packet/app-encoder contracts, actual Windows pre-neural
rejection, unchanged historical shell deadlines, Bash syntax and nine malformed
return/frozen-partition/group/quality regressions pass. A prospective complete
return audit is prepared. Its initial CPU replay was exercised on genuine closed
R2 evidence:50 DGP/50 recognizer/50 cohort rows; this is not a V28 VM result.
The missing CPU device argument found before release was corrected; preliminary
auditor source/checks and failure evidence remain. Actual V28 training is pending.

Previous50 preparation bindings, deeper668/cleanup60/309/66/692/299/697/513,
the three complete document bodies and app22 bindings remain intact. Cleanup
remains previously verified14 duplicate archives/1.64GiB; last measured7.48GiB
free, not a new live disk claim. Original checkpoints, scientific caches, splits,
logs and all earlier failed gates remain. No native or reserved-final pixels are
opened. Photograph source labels remain separate and do not imply ethnicity.
Useful whole-face native restoration, all seven automatic/assisted covering
families, canonical app flow/regressions and independent final review remain
required. No Zamboanga or hidden-identity claim follows. Goal active/incomplete.

[R2 results and branch diagnosis](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_RESULTS.md>) ·
[Five manual V28 steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_VM.md>) ·
[App-input clarification](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DECODER_APP_NORMALIZATION_V1.md>)

Earlier bodies below remain preserved history. R2-ready/pending and its launch
commands are superseded by the closed failed return; only distinct V28 is ready.

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
    files=[*DOCS,'CCTV_DGP_ORIGINAL_DECODER_GRADIENT_V1_R2_RESULTS.md','CCTV_DGP_ACTIVE_ORIGINAL_DECODER_V28_VM.md',
      'CCTV_DGP_DECODER_APP_NORMALIZATION_V1.md','scripts/review_cctv_dgp_decoder_app_normalization_v1.py',
      'scripts/verify_cctv_dgp_decoder_app_normalization_v1.py','scripts/review_cctv_dgp_original_decoder_head4_v1.py',
      'scripts/verify_cctv_dgp_original_decoder_head4_v1.py','scripts/cctv_dgp_active_original_decoder_v28.py',
      'scripts/review_cctv_dgp_active_original_decoder_v28.py','scripts/verify_cctv_dgp_active_original_decoder_v28.py',
      'scripts/cctv_dgp_active_decoder_v28_training.py','scripts/prepare_cctv_dgp_active_original_decoder_v28.py',
      'scripts/verify_cctv_dgp_active_original_decoder_v28_packet.py','scripts/audit_cctv_dgp_active_original_decoder_v28_return.py',
      'tests/test_cctv_dgp_active_original_decoder_v28_audit.py',
      'scripts/record_cctv_dgp_original_decoder_r2_and_v28_milestone.py',
      'scripts/verify_cctv_dgp_original_decoder_r2_and_v28_milestone.py',
      'outputs/cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz',
      'outputs/cctv-dgp-original-decoder-gradient-v1-r2-results.tar.gz.sha256',
      'outputs/cctv-dgp-original-decoder-gradient-v1-r2-export.json',
      'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return_import.json',
      'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json',
      'outputs/cctv-dgp-active-original-decoder-v28-execution.tar.gz',
      'outputs/cctv-dgp-active-original-decoder-v28-execution.tar.gz.sha256',
      'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json',
      'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/independent_readback.json']
    for name in files:bind(ROOT/name)
    for directory in [ROOT/'outputs/cctv_dgp_decoder_app_normalization_v1',ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return',
       ROOT/'outputs/cctv_dgp_original_decoder_head4_review_v1',ROOT/'outputs/cctv_dgp_active_original_decoder_v28_review',
       ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28',prep,before]:
        for path in directory.rglob('*'):
            if path.is_file():bind(path)
    m={'complete':True,'date':'2026-10-06','scope':'Genuine failed R2 all14 audit/underflow trace and new finite active12 V28 manual preparation',
       'new_evidence_sha256':evidence,'previous_milestone_sha256':sha(PREVIOUS),'previous50_original_locations':mapping,
       'original_document_bodies_preserved':3,'app_record_sha256':sha(app_path),'app22_bindings_preserved':True,
       'R2_return_sha256':audit['archive_sha256'],'R2_archive_bytes':audit['archive_bytes'],'R2_proof_remains_failed':True,
       'R2_saved_gradient_values_audited':46909863,'R2_optimizer_updates':0,'inactive_tensors':head['zero_VM_gradient_tensors_all_batches'],
       'head4_CPU_zero_output_cases':50,'head4_CPU_nonzero_upstream_cases':50,'head4_float64_subnormal_bound_cases':50,
       'actual_app_CPU_encoding_difference':0,'V28_initial_actual_app_CPU_parity_cases':50,
       'V28_trainable_parameters':498627,'V28_trainable_tensors':12,'V28_updates_bound':800,'V28_final_epochs_bound':80,
       'V28_protocol_sha256':packet['protocol_sha256'],'V28_archive_sha256':packet['archive_sha256'],'V28_archive_bytes':packet['archive_bytes'],
       'V28_actual_training_started':False,'human_manual_VM_execution_required':True,'prospective_full_return_audit_prepared':True,
       'V28_return_regressions_passed':9,'new_finite_training_protocol_created':True,'all_old_failed_gates_preserved':True,
       'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'VM_actions':False,
       'native_or_reserved_used':False,'independent_final_review_complete':False,'app_promotion':False,
       'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(m,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:m[k] for k in ['complete','R2_proof_remains_failed','V28_archive_bytes','V28_actual_training_started','goal_status','seconds']},indent=2))


if __name__=='__main__':main()
