"""Record audited returned gradients and an independently checked, unlaunched V33 probe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_gradient_return_v33_probe_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_post_v32_learning_review_v1_milestone'
PREP = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation'
PACKET = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_vm'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists()
    old = read(PREVIOUS / 'milestone.json')
    for name, digest in old['new_evidence_sha256'].items(): assert sha(ROOT / name) == digest, name
    returned = read(ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json')
    packet, prep, checked = map(read, [PACKET / 'protocol.json', PREP / 'preparation_receipt.json', PREP / 'independent_packet_audit_r1.json'])
    assert returned['complete'] and returned['diagnostic_complete'] and returned['members_verified'] == 756
    assert returned['CPU_replay']['cases_at_both_states'] == 200 and returned['optimizer_updates'] == 0
    assert checked['complete'] and checked['protocol_sha256'] == prep['protocol_sha256'] == sha(PACKET / 'protocol.json')
    assert checked['actual_optimizer_updates'] == checked['VM_launches'] == 0 and checked['unsafe_return_boundaries_rejected'] == 7
    assert read(PREP / 'Bash_syntax_audit.json')['exit_code'] == 0 and read(PREP / 'metric_arithmetic_audit_v1/audit.json')['complete']
    assert not (PACKET / 'outputs').exists() and packet['optimizer_updates'] == 8 and packet['committed_trajectory_updates'] == 0
    addition = '''
**Latest research milestone - 7 October 2026: V32 loss-gradient return audited; finite V33 output probe ready.**

The human downloaded the completed diagnostic. Archive SHA256
529acd5750c64e7320260c09f55cdd3a115c2f465afd02abc4101009155143e3
matches721,569,318bytes and both sidecars. Independent audit verifies756files,
301,298,844saved gradient values and200frozen CPU restoration replays. Maximum
raw error0.000002563, PNG erroronebyte and vector error0.0000005253 pass unchanged
tolerances. VM diagnostic45.58s/280queries/0optimizer updates; local audit282.12s
with zero gradients, backwards or optimizer calls. No new checkpoint was created.
The sidecar read race and both successful download transports are preserved;
completed canonical sidecars match their staged copies without repeat downloads.

At stopped50, preservation/restoration gradient-norm ratios are7.5746exposed and
4.2689unoptimized, with negative cosines−0.24866/−0.25694. The negative raw
seven-loss sum increases facial-detail loss in both aggregate TRAIN cohorts.
This measures endpoint conflicts, not actual unsaved AdamW trajectories or
curvature. Existing protection counters actual regressions and remains necessary.
V32's50-update/0.970672% versus1% failed structure gate remains; no resume exists.

Own cone arithmetic keeps all seven nonzero gradient constraints. A restoration
proposal descends in all three restoration terms in the stopped aggregates;
projecting the full seven-loss sum can leave a detail derivative at zero. Both
methods on44matrices pass88independent primal checks, analytic/invalid/random
fixtures, with no neural work or parameter assignments. This does not prove
finite face improvement, capacity or useful native output. Primary research and
its mathematical adaptation/curvature/performance limits are recorded separately.

The distinct V33 probe is prepared for manual execution on the existing L4 VM.
It reuses four saved aggregate matrices and makes0new gradient queries. Each
original/stopped state and exposed/unoptimized cohort makes two fresh disposable
AdamW proposal steps, eight total, resetting before every output trial. The
original loss values, normalization, learning rate, clipping, decay and all
quality gates stay. No continuing trajectory, epoch or new checkpoint is created.
The actual restoration AdamW displacement is projected against all seven nonzero
loss gradients, then tested at fixed scales1,1/2,1/4,1/8 beside both controls.
Zero-gradient hinges and float32 rounding can still cause finite regressions.
Actual outputs, not linear projections, decide them. The unconstrained control
cannot qualify a training recipe. No automatic800-update follow-on is permitted.

The42,130byte/five-file packet passes414basis bindings, Python3.10 parsing,
pre-neural Windows update rejection, seven unsafe-import checks, eight independent
primal displacement fixtures and actual read-only Bash syntax. The raw metric
reviewer reproduces the first six components on200saved cases with maximum
error0.00000016393. First syntax/solver/serialization preparation failures remain.
The distinct R1 local reviewer fixes solver coordinates and tightens convergence;
VM packet/protocol, projection and all tolerances remain unchanged. Its source is
frozen separately and supersedes the retained initial return reviewer.

V33 protocol SHA256:ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee.
Archive SHA256:282931a3b265c834a5736ac7f70c0816e08b74a15948d4c94645679a309eeace.
Require an idle L4/g2-standard-4 with6GiB free. Estimated3–10minute probe plus
1–4minute export; actual V33 timing remains unmeasured. Worker900s,
external930s+30s grace, export300s/external330s+30s grace, VRAM20GiB and return2GiB
are enforced. Save all1,200trial plus200before outputs. Prospective independent
audit checks their composition/metrics, eight actual moment/displacement proofs,
four independent projections and280metadata-selected frozen CPU replays.
The guide gives five exact gcloud upload/install/tmux/launch/download steps.
Agent VM launches, uploads, new training and app changes are zero in this stage.

This is paired synthetic photographic TRAIN evidence. Source labels are not
ethnicity, native CCTV or Zamboanga performance. Native development/reserved
pixels remain unopened. Original checkpoints, splits, caches, failures and actual
Windows backup remain. The own-DGP app checkpoint, selector, override and design
are unchanged. Converted MAT is still unqualified. Useful native restoration,
automatic/assisted quality for all seven covering families and independent final
review remain outstanding. Goal active/incomplete; actual training stays manual.

[Audited gradient review and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V32_LOSS_GRADIENT_RETURN_REVIEW_V1.md>)
[Five exact manual V33 steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LOSS_CONE_PROBE_V33_VM.md>)

The complete previous handoff is retained below. Its previously unlaunched V32
diagnostic is now completed by the human; the new V33 probe remains unlaunched.

'''
    OUT.mkdir(); (OUT / 'before_docs').mkdir(); handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    before_path = OUT / 'before_docs/PROJECT_HANDOFF.md'
    with before_path.open('xb') as stream: stream.write(before)
    split = before.index(b'\n') + 1; handoff.write_bytes(before[:split] + addition.encode('utf-8') + before[split:])
    names = ['record_cctv_dgp_v32_gradient_return_v33_probe_milestone.py', 'verify_cctv_dgp_v32_gradient_return_v33_probe_milestone.py',
             'cctv_dgp_loss_cone_v33.py', 'analyze_cctv_dgp_v32_loss_directions_v1.py', 'analyze_cctv_dgp_v32_restoration_cone_v1.py',
             'audit_cctv_dgp_loss_cone_arithmetic_v1.py', 'inspect_cctv_dgp_v32_loss_gradient_status_v1.py',
             'download_cctv_dgp_v32_loss_gradient_sidecars_v1.py', 'run_cctv_dgp_v32_loss_gradient_return_audit_v1.py',
             'prepare_cctv_dgp_loss_cone_probe_v33.py', 'cctv_dgp_loss_cone_probe_v33_vm.py', 'cctv_dgp_loss_cone_probe_v33_metrics.py',
             'audit_cctv_dgp_loss_cone_probe_v33_return.py', 'audit_cctv_dgp_loss_cone_probe_v33_return_r1.py',
             'verify_cctv_dgp_loss_cone_probe_v33_packet.py', 'verify_cctv_dgp_loss_cone_probe_v33_packet_r1.py',
             'diagnose_cctv_dgp_v33_primal_audit_v1_r1.py', 'verify_cctv_dgp_v33_metric_arithmetic_v1.py']
    files = [ROOT / 'scripts' / name for name in names] + [handoff, before_path,
             ROOT / 'CCTV_DGP_V32_LOSS_GRADIENT_RETURN_REVIEW_V1.md', ROOT / 'CCTV_DGP_LOSS_CONE_PROBE_V33_VM.md',
             ROOT / 'outputs/cctv-dgp-loss-cone-probe-v33-execution.tar.gz', ROOT / 'outputs/cctv-dgp-loss-cone-probe-v33-execution.tar.gz.sha256',
             ROOT / 'outputs/cctv-dgp-v32-loss-gradient-v1-results.tar.gz', ROOT / 'outputs/cctv-dgp-v32-loss-gradient-v1-results.tar.gz.sha256',
             ROOT / 'outputs/cctv-dgp-v32-loss-gradient-v1-export.json', ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return_import.json', PREVIOUS / 'milestone.json',
             PREVIOUS / 'independent_closure_audit.json', PREVIOUS / 'final_readback.json']
    for folder in [PACKET, PREP, ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_audit_execution',
                   ROOT / 'outputs/cctv_dgp_v32_loss_gradient_status_v1', ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_sidecar_download',
                   ROOT / 'outputs/cctv_dgp_v32_loss_directions_v1', ROOT / 'outputs/cctv_dgp_v32_restoration_cone_v1',
                   ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation_failure_v1',
                   ROOT / 'outputs/cctv_dgp_v32_gradient_return_v33_closure_preparation_failure_v1']:
        files.extend(path for path in sorted(folder.rglob('*')) if path.is_file())
    m = {'complete': True, 'datetime_UTC': datetime.now(timezone.utc).isoformat(), 'recorder_sha256': sha(Path(__file__)),
         'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'), 'previous_document_locations': {'PROJECT_HANDOFF.md': before_path.relative_to(ROOT).as_posix()},
         'new_evidence_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
         'document': {'name': handoff.name, 'before_path': before_path.relative_to(ROOT).as_posix(), 'before_sha256': hashlib.sha256(before).hexdigest(),
                      'after_sha256': sha(handoff), 'addition_bytes': len(addition.encode('utf-8'))},
         'human_V32_gradient_diagnostic_complete': True, 'independent_return_audit_pass': True, 'gradient_arrays_analyzed_without_parameter_changes': True,
         'user_design_discussion_satisfied': True, 'V33_packet_prepared': True, 'V33_VM_execution_started': False, 'V33_optimizer_proposal_steps_maximum': 8,
         'V33_committed_trajectory_updates': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'agent_VM_launches': 0,
         'native_or_reserved_used': False, 'old_failure_and_preparation_failures_retained': True, 'gates_or_protection_removed': False,
         'app_changes': False, 'app_promotion': False, 'manual_VM_execution_required': True, 'independent_final_review': False,
         'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8') as stream: stream.write(json.dumps(m, indent=2) + '\n')
    print(json.dumps({'complete': True, 'bindings': len(m['new_evidence_sha256']), 'milestone_sha256': sha(OUT / 'milestone.json'), 'V33_VM_execution_started': False}))


if __name__ == '__main__': main()
