"""Archive the entire handoff and bind V36's failures and prepared V37 evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone'
PREVIOUS = ROOT / 'outputs/completion_conditioning_union_v1_milestone'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    started = time.monotonic(); assert not OUT.exists()
    old = read(PREVIOUS / 'milestone.json'); prior_audit = read(PREVIOUS / 'independent_closure_audit.json')
    assert old['complete'] and prior_audit['complete'] and prior_audit['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    for name, digest in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
        assert time.monotonic() - started < 300
    audit = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json')
    analysis = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json')
    visual = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/visual_review.json')
    forward = read(ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1/analysis.json')
    packet = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_preparation/independent_packet_audit_r1.json')
    assert audit['complete'] and audit['finite_probe_complete'] and not analysis['jointly_eligible_subset_variants']
    assert visual['actually_viewed_cases'] == 100 and visual['actually_viewed_pages'] == 20 and not visual['app_adoption']
    assert forward['forward_cases'] == 500 and forward['PNG_failures_without_same_group_raw_failure'] == 9
    assert packet['complete'] and packet['VM_launches'] == packet['actual_gradient_queries'] == packet['actual_optimizer_updates'] == 0
    addition = '''
**DGP milestone - 8 October 2026: V36 independently audited and rejected; delivered-PNG path verified; V37 zero-update diagnostic prepared.**

The human's returned V36 archive contains2,135 verified regular files. The
independent local audit verifies all500 saved raw/PNG outputs,170 group receipts,
108 projection constraints and100 frozen CPU replay outputs. Audit runtime is
226.393 seconds; its bounded external supervisor completes in227.784 seconds.
No returned code is executed and no local gradients or optimizer run. VM V36
takes92.611 seconds for four independently reset finite displacements, zero
optimizer updates, zero new gradient queries and no checkpoint or trajectory.

Preservation failures fall from V35's41 to11, but no V36 scale passes both
TRAIN cohorts. At scale1, structure gains are0.277816% and0.410191%; one cohort
has no PNG failures and the other has two. Across scales there are ten delivered
ArcFace and one SSIM group failures. Positive source gains and brightness passes
do not override preservation. All100 V35/V36 original raw and PNG files are
byte-identical. Existing1% at50/10% at800 capacity gates and all development
qualification failures remain. No V36 trial is promoted or resumed.

All100 TRAIN cases and20 pages are actually viewed at original256-pixel cell
detail; a separate audit verifies all700 input/target/output cells. Severe blur
and compound inputs still lack clear eyelid, nostril and lip/tooth structure;
the four trials remain visually close to the original DGP. Softness alone is
not a failure. This is implementing-assistant development review, not independent
final review. Both cohorts are now design data. The historical 'unexposed' label
does not make the second cohort held-out development or final evaluation.

Nine of11 PNG failures have no same-group raw-metric failure; two fail both.
A frozen-output arithmetic diagnostic reproduces all500 actual PNG-valued
forwards exactly, with maximum MSE error4.163336e-17 and SSIM error2.220446e-16.
Its separate7.170-second check verifies1,154 source bindings, all500 encodings,
all11 classifications and four boundary/quantization/filter fixtures. No model,
neural call, gradient, backward, optimizer or VM connection runs for this check.
Primary research and the official floor derivative motivate an explicitly
approximate surrogate; they do not guarantee nonlinear DGP preservation.

After the required architecture discussion, the user selected the best approach
with research where needed. V37 therefore measures the delivered PNG guard path
before any new update design. It keeps the original own-trained DGP, the same
100 photographic TRAIN cases, five-case context and23 tensors. The maximum is
300 coarse autograd gradient queries, zero optimizer/parameter updates, zero
epochs and no new checkpoint. Custom backward machinery is used within grad;
zero backward API calls does not mean zero VM gradient computation. All17 finite
PNG checks, useful-structure gates, source and brightness requirements remain.

The four-file35,166-byte packet is independently verified: Python3.10 syntax,
412 V36 dependency hashes,26 local bindings, Windows/root/instance rejection,
eight unsafe-archive tests, six group tests and read-only Bash syntax. Windows
initially denied Bash's signal pipe; the original checker/failure is retained
and a distinct R1 consumes the successful read-only external Bash-n evidence.
The frozen packet was not changed. Protocol SHA256:
7e3c9ae20fe2f4faa2d0e83ba628294c5b91a8d0f43093ea53bffcc6320d0090.
Execution archive SHA256:
444bea1b2f5f5b79a0d0912b2f06e5a01132a015525e66ff58af30ae38f07503.

V37 is PREPARED, NOT RUN. Actual gradients/training stay under the human's
manual L4/g2-standard-4 tmux workflow. Require6GiB free; expected diagnostic
3-10 minutes, export1-3 minutes. Worker/external stops are900/930 seconds plus
30-second kill grace, export300/330 seconds plus30-second grace, allocated VRAM
20GiB and uncompressed return1.5GiB. A forward mismatch or any finite/state/count/
resource violation stops and exports failure evidence. No historical worker,
follow-on optimizer, cleanup or VM connection is executed by the agent here.

The own-trained DGP remains primary and all14 app/checkpoint bindings are
unchanged. Existing design, Auto/On/Off, mask review and PNG/bundle downloads
remain. The earlier real inline Playwright flow is preserved; no frontend
change requires another browser run here. Completion conditioning improvements
and failures remain separately recorded and neither automatic nor assisted
whole covering-family scope is qualified. Original checkpoints, source/splits,
research caches/local backups, reports and all failure evidence remain.

The24 unpaired native development crops and520 paired synthetic development
cases are unchanged; the45 reserved-final identities/58 crops remain unopened.
No real Zamboanga CCTV samples or ethnicity/local-performance claims follow
from these photographic TRAIN diagnostics. Useful native structure, all seven
covering families and independent final review remain. Goal active/incomplete.

[V36 audited findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_RESULTS.md>)
[PNG-path evidence and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V36_DELIVERED_METRIC_REVIEW.md>)
[V37 five exact manual upload/install/tmux/launch/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_GUARD_GRAD_V37_VM.md>)

The complete preceding handoff follows. Its V36-not-returned language describes
the earlier milestone and is superseded by this audited human return.

'''
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    saved = OUT / 'before_docs/PROJECT_HANDOFF.md'; saved.write_bytes(before)
    at = before.index(b'\n') + 1; addition_bytes = addition.encode('utf-8')
    handoff.write_bytes(before[:at] + addition_bytes + before[at:])
    p = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_vm/protocol.json')
    bindings = dict(p['local_basis_sha256'])
    for name, digest in read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return_import.json')['files_sha256'].items():
        bindings['outputs/cctv_dgp_finite_clearance_probe_v36_return/' + name] = digest
    bindings.update(analysis['bindings_sha256']); bindings.update(forward['bindings_sha256'])
    folders = ['outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1', 'outputs/cctv_dgp_finite_clearance_probe_v36_audit_run_v1',
               'outputs/cctv_dgp_v36_delivered_metric_path_v1', 'outputs/cctv_dgp_delivered_guard_grad_v37_preparation',
               'outputs/cctv_dgp_delivered_guard_grad_v37_vm']
    for folder in folders:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file(): bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    names = ['PROJECT_HANDOFF.md', saved.relative_to(ROOT).as_posix(), 'CCTV_DGP_DELIVERED_GUARD_GRAD_V37_VM.md',
             'outputs/cctv_dgp_finite_clearance_probe_v36_return_import.json',
             'outputs/cctv-dgp-finite-clearance-probe-v36-results.tar.gz',
             'outputs/cctv-dgp-finite-clearance-probe-v36-results.tar.gz.sha256',
             'outputs/cctv-dgp-finite-clearance-probe-v36-export.json',
             'outputs/cctv-dgp-delivered-guard-grad-v37-execution.tar.gz',
             'outputs/cctv-dgp-delivered-guard-grad-v37-execution.tar.gz.sha256',
             'outputs/completion_conditioning_union_v1_milestone/milestone.json',
             'outputs/completion_conditioning_union_v1_milestone/independent_closure_audit.json',
             'scripts/supervise_cctv_dgp_v36_return_audit_v1.py',
             'scripts/analyze_cctv_dgp_finite_clearance_probe_v36_return_v1.py', 'scripts/record_cctv_dgp_v36_visual_review_v1.py',
             'scripts/verify_cctv_dgp_v36_analysis_and_pages_v1.py', 'scripts/cctv_dgp_delivered_png_guard_v37.py',
             'scripts/build_cctv_dgp_delivered_guard_v37_worker_v1.py', 'scripts/cctv_dgp_delivered_guard_grad_v37_vm.py',
             'scripts/prepare_cctv_dgp_delivered_guard_grad_v37.py', 'scripts/verify_cctv_dgp_delivered_guard_grad_v37_packet.py',
             'scripts/verify_cctv_dgp_delivered_guard_grad_v37_packet_r1.py',
             'scripts/record_cctv_dgp_v36_return_v37_diagnostic_milestone.py',
             'scripts/verify_cctv_dgp_v36_return_v37_diagnostic_milestone.py']
    for name in names: bindings[name] = sha(ROOT / name)
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    m = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'previous_handoff_path': saved.relative_to(ROOT).as_posix(), 'new_evidence_sha256': bindings,
         'previous_bindings': len(old['new_evidence_sha256']),
         'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff), 'addition_bytes': len(addition_bytes)},
         'V36_qualification_failed': True, 'V36_outputs_visually_reviewed': 500, 'V37_prepared_only': True,
         'V37_VM_launches': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls_here': 0,
         'app_changes': False, 'app_adoption': False, 'automatic_quality_qualification': False,
         'assisted_quality_qualification': False, 'independent_final_review': False, 'goal_complete': False,
         'seconds': time.monotonic() - started, 'cap_seconds': 300}
    assert m['seconds'] < 300
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(m, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'bindings': len(bindings), 'seconds': m['seconds'],
                      'milestone_sha256': sha(OUT / 'milestone.json')}), flush=True)


if __name__ == '__main__': main()
