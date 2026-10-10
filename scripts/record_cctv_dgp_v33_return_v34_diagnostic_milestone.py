"""Preserve the complete previous handoff and record the verified V33/V34 boundary."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone'
PREVIOUS=ROOT/'outputs/completion_input_footprints_v1_milestone'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert not OUT.exists();old=read(PREVIOUS/'milestone.json')
    for n,h in old['new_evidence_sha256'].items():assert sha(ROOT/n)==h,n
    assert read(PREVIOUS/'independent_closure_audit.json')['complete']
    audit=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json')
    visual=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/visual_review.json')
    packet=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json')
    prep=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_preparation/preparation_receipt.json')
    checked=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_preparation/independent_packet_audit_r1.json')
    assert audit['complete'] and audit['finite_probe_complete'] and audit['CPU_replay']['outputs']==280
    assert visual['all4_pages_actually_viewed_at_original_detail'] and not visual['app_adoption']
    assert checked['complete'] and checked['protocol_sha256']==prep['protocol_sha256']==sha(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json')
    assert checked['VM_launches']==0 and not (ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/outputs').exists()
    for n,h in packet['local_basis_sha256'].items():assert sha(ROOT/n)==h,n
    imported=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_return_import.json')
    for n,h in imported['files_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_return'/n)==h,n
    addition='''
**Research milestone - 7 October 2026: V33 output probe audited; group-preservation diagnostic V34 prepared.**

V33 completed its finite human-run probe in192.75seconds: eight reset first-step
AdamW proposals, zero committed trajectory updates, new gradient queries or new
checkpoints. The independent R2 audit verifies5,914files, all1,400 raw/PNG
measurements, moment/displacement/projection proofs and280 frozen CPU replay
outputs. Maximum raw error0.000002414, PNG difference one byte, vector error
0.000001349 and component error0.000003875 pass the original tolerances. Local
gradients, backwards and optimizer updates are zero. The audit took569.79seconds.

The initial checker stopped on exact float group-dictionary equality; independent
readback measured rounding differences around0.00000000000000017. Its source and
failure are retained. R2 preserves exact saved-row aggregation and checks fresh
group arithmetic with the existing case tolerance, retaining identical failure
keys, source-gain signs and brightness decisions. No qualification gate changed.

No proposal qualifies a training recipe or app replacement. At the original state,
the four preservation terms have zero gradients; cone scale1 therefore equals
the unconstrained proposal and fails15 group/metric checks in each TRAIN subset.
All smaller original-state scales also fail preservation. At stopped50, cone
scale1 improves detail0.37288% in exposed and0.44015% in unexposed matched TRAIN
subsets. The exposed subset passes17 group checks, but the unexposed subset
still fails two ArcFace checks. It repairs one existing blur failure while
introducing a clear-group failure; the compound degradation failure remains.
Neither subset gain replaces V32's full3,905-case0.970672% failed1% gate at50.
Do not resume that checkpoint or repeat the failed projection recipe unchanged.

All four prospectively metadata-selected comparison pages were actually inspected
at256px cell resolution: first reference per source, all five profiles, both
matched TRAIN subsets,20 cases/160 input-target-comparison cells. The difficult
rows remain soft around eyes, nose and mouth, with little convincing all-feature
clarity gain. This is a bounded development review, not review of all1,400 images
or independent final quality. All1,400 measurements were audited separately.
Unexposed means untouched during the first50 V32 updates, not held-out DEV/final.

GEM's locally linear gradient approximation and recent epsilon-constraint work
support investigating the protected functions directly; they do not prove CCTV
usefulness or authorize relaxing preservation bounds. The research rationale
and primary-paper links are in CCTV_DGP_LOSS_CONE_PROBE_V33_RESULTS.md.

V34 measures original-state, non-hinged raw MSE, one-minus-SSIM and one-minus-
fixed-ArcFace derivatives for each of the same100 TRAIN cases. The same five-case
batch context and23 selected own-DGP tensors remain. There are exactly300
gradient queries, zero optimizer/parameter updates and no new checkpoint. Saved
case matrices allow an independent assembly of all17 source/profile group
derivatives, rather than relying on a single average that can conceal opposing
directions. This is a changed diagnostic measurement, not a changed training
loss or an automatic continuation. Finite outputs must still pass the original
PNG, source, brightness and capacity requirements before any later training.

The three-file32,433-byte packet passes independent provenance, Python3.10,
read-only Bash syntax, Windows pre-neural rejection, three wrong-root/instance,
eight unsafe-return and six group-cancellation/reordering/count regressions.
The first verifier's sandbox signal-pipe failure is retained separately; its
read-only Bash syntax check passes outside that restriction. The VM packet and
gates are unchanged. Protocol SHA256:
d03050e18e4fd2369a6dd0803632bebfa0c709cc95cc8f7a6c8776785b0923a1.
Execution archive SHA256:
db822ba0b17f941d8e20783e5ee73ed97586dfdfb0f53a4d4909d392bd4b2656.

V34 remains unrun. Follow the five manual Google Cloud SDK upload/install/tmux/
launch/download steps on the existing running L4/g2-standard-4 VM. Require6GiB
free and an idle GPU. Estimated diagnostic1-6minutes plus export1-3minutes;
worker600s, external630s plus30s grace; export300s/external330s plus30s grace;
VRAM20GiB and uncompressed return1.5GiB stops are enforced. Original assets,
failed gates, splits and research-cache backup remain. No VM start, new training
or cleanup was performed in this milestone. Prior disk/VM snapshots are historical.

The local own-DGP remains the primary restorer with its existing Auto/override
workflow and design. All14 current app bindings remain hash matched. The prior
full app/inline Playwright ordering evidence and negative full-family completion
reviews remain binding. No new native or reserved-final pixels were opened;
public native CCTV development and separate labeled final identities stay frozen.
This new probe is paired synthetic photographic TRAIN evidence, separate from
unpaired CCTV evidence. Source labels imply neither ethnicity nor Zamboanga
performance. Useful native restoration, automatic and assisted covering-family
quality and independent final review remain outstanding. Goal active/incomplete.

[V33 findings and actual review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LOSS_CONE_PROBE_V33_RESULTS.md>)
[V34 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md>)
[Independent V33 return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json>)

The complete previous handoff follows. Its V33-audit-pending statements are historical and superseded by this verified return. Its covering-family quality failures remain current.

'''
    OUT.mkdir();(OUT/'before_docs').mkdir();handoff=ROOT/'PROJECT_HANDOFF.md';before=handoff.read_bytes()
    saved=OUT/'before_docs/PROJECT_HANDOFF.md';saved.write_bytes(before)
    at=before.index(b'\n')+1;handoff.write_bytes(before[:at]+addition.encode('utf-8')+before[at:])
    names=['record_cctv_dgp_v33_return_v34_diagnostic_milestone.py','verify_cctv_dgp_v33_return_v34_diagnostic_milestone.py',
        'analyze_cctv_dgp_loss_cone_probe_v33_return_v1.py','record_cctv_dgp_v33_review_v1.py',
        'audit_cctv_dgp_loss_cone_probe_v33_return_r2.py','supervise_cctv_dgp_v33_return_audit_r2_v1.py',
        'cctv_dgp_group_guard_grad_v34_vm.py','prepare_cctv_dgp_group_guard_grad_v34.py',
        'audit_cctv_dgp_group_guard_grad_v34_return.py','verify_cctv_dgp_group_guard_grad_v34_packet.py',
        'verify_cctv_dgp_group_guard_grad_v34_packet_r1.py','preserve_cctv_dgp_v34_packet_check_failure_v1.py']
    files=[ROOT/'scripts'/n for n in names]+[saved,handoff,ROOT/'CCTV_DGP_LOSS_CONE_PROBE_V33_RESULTS.md',
        ROOT/'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md',PREVIOUS/'milestone.json',PREVIOUS/'independent_closure_audit.json',
        ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_return_import.json',ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json']
    for folder in ['outputs/cctv_dgp_loss_cone_probe_v33_return','outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1',
        'outputs/cctv_dgp_loss_cone_probe_v33_audit_r2_run_v1','outputs/cctv_dgp_loss_cone_probe_v33_audit_failure_v1',
        'outputs/cctv_dgp_loss_cone_probe_v33_group_diagnostic_v1','outputs/cctv_dgp_group_guard_grad_v34_vm',
        'outputs/cctv_dgp_group_guard_grad_v34_preparation']:
        files.extend(q for q in sorted((ROOT/folder).rglob('*')) if q.is_file())
    for stem,suffixes in [('cctv-dgp-loss-cone-probe-v33',['-results.tar.gz','-results.tar.gz.sha256','-export.json']),
        ('cctv-dgp-group-guard-grad-v34',['-execution.tar.gz','-execution.tar.gz.sha256'])]:
        files.extend(ROOT/'outputs'/(stem+suffix) for suffix in suffixes)
    evidence={q.relative_to(ROOT).as_posix():sha(q) for q in files}
    m={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'new_evidence_sha256':evidence,
        'previous_milestone_sha256':sha(PREVIOUS/'milestone.json'),'previous_handoff_path':saved.relative_to(ROOT).as_posix(),
        'document':{'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(handoff),'addition_bytes':len(addition.encode('utf-8'))},
        'V33_return_independently_verified':True,'V33_saved_outputs_audited':1400,'V33_CPU_replays':280,
        'V33_reviewed_cases':20,'V33_reviewed_pages':4,'all1400_visually_reviewed':False,
        'V34_packet_verified':True,'V34_prepared_gradient_queries':300,'V34_VM_execution_started':False,
        'local_gradient_or_optimizer_calls':0,'app_adopted':False,'covering_family_failures_retained':True,
        'native_or_reserved_final_used':False,'independent_final_quality_review':False,
        'goal_status':'active','goal_complete':False}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'complete':True,'bindings':len(evidence),'milestone_sha256':sha(OUT/'milestone.json'),'V34_VM_execution_started':False}))


if __name__=='__main__':main()
