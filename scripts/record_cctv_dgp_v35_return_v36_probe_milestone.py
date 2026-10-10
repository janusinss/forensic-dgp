"""Retain the full preceding handoff and freeze V35 return/V36 preparation evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v35_return_v36_probe_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_v34_return_v35_probe_milestone'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    old = read(PREVIOUS / 'milestone.json')
    assert read(PREVIOUS / 'independent_closure_audit.json')['complete']
    for name, digest in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
    audit = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2.json')
    pages = read(ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1/independent_analysis_page_audit.json')
    prep = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_preparation/independent_packet_audit.json')
    assert audit['complete'] and audit['finite_probe_complete'] and audit['members_verified'] == 2135
    assert pages['complete'] and pages['all700_cells_exact']
    assert prep['complete'] and prep['all65_empirical_clearances_independently_recalibrated'] and prep['VM_launches'] == 0
    addition = '''
**Research milestone - 8 October 2026: V35 R1 audited; all scales fail preservation; changed V36 probe prepared.**

The human returned V35 R1. Its388,519,155-byte archive and2,135 members are
independently verified. The L4 probe completed93.899seconds, export26.162seconds:
four disposable original-state copies, zero optimizer/gradient/backward/epoch
or committed training updates, and no new checkpoint. Independent R2 audit
rederives500 raw/PNG measurements and replays100 frozen CPU outputs in191.532
seconds; all original tolerances pass and states are restored. The prefix-only
checker correction retains the original failed checker/log; no VM rerun needed.

Every scale1,1/2,1/4,1/8 fails preservation in both matched TRAIN subsets.
Scale1 delivered-detail gain is0.277588% exposed and0.418626% other TRAIN,
with4 and8 failed group/metric checks. Across all scales there are40 PNG
ArcFace failures and one clear MSE failure. Every scale also has raw MSE/
ArcFace failures, so PNG conversion alone does not explain the outcome.
Positive source gains and brightness passes do not override preservation.
The subset probe does not satisfy the unchanged1%-at50/10%-at800 capacity gates.

All100 cases/500 model outputs were actually viewed at original256px cell
detail across20 pages. Severe blur and compound inputs retain poorly defined
eyes, nostrils and mouths; trials remain close to the original DGP. Independent
readback verifies all700 input/target/output cells exactly. This is implementing
assistant TRAIN review, not independent final review. A renderer helper-name
collision is retained with its10 partial pages; distinct R1 preserves those
pages exactly and completes review preparation without changing returned data.

V36 changes the direction rather than rerunning the zero-clearance recipe.
All108 rows remain;65 raw MSE/ArcFace rows reserve twice the measured positive
V35 scale1 raw/PNG departure from their linear prediction. SSIM and six
restoration rows retain zero clearance because the saved skimage SSIM differs
from the differentiable VM guard. This is empirical, not a proven nonlinear
bound. Actual17 PNG groups, source, brightness and capacity gates are unchanged.
Both matched subsets now guide TRAIN design; unexposed is not DEV/final.

The array-only primal/dual check and independent full-row KKT/recalibration
pass. Candidate magnitude is100.703574% of the original mean proposal, below
the2x cap; this is not quality gain. Float32 copies retain small reported
clearance residuals. V36 remains an unrun four-scale finite image probe:
100 baselines plus400 trial outputs, no new gradients/optimizer updates,
trajectory, epochs, checkpoints, historical run or app promotion. Its9-file
12,190,362-byte packet passes Python3.10/read-only Bash syntax, all65 empirical
recalibrations,108-row certificate, host/root/platform and unsafe-return tests,
exact export/checker prefix parity and unchanged snapshot AST. Protocol SHA256:
9af4cbf10d7c141e2cbef2248bc282c135a9cc6117f37b47feaef76d71dadd38.
Execution archive SHA256:
ee51b5bc9e888dec32e0dfd90c96a19f9e93e21acd1fd4bef3e8500d12f0a67e.

Use the five manual Google Cloud SDK/tmux steps on the existing idle running
NVIDIA L4/g2-standard-4. Require6GiB free; estimate2-4minutes plus1-2minutes
export. Worker900s/external930s+30s grace, export300s/external330s+30s grace,
VRAM20GiB and uncompressed return768MiB enforce finite work. No VM connection,
start, cleanup or actual new training was performed in this milestone.

All14 current own-DGP Auto/override app bindings, original checkpoints, splits,
research caches, historical gates, instructions and the local backup remain.
No new native development, paired DEV or reserved-final pixels were opened.
These photographic paired TRAIN findings establish neither ethnicity nor
Zamboanga performance. Prior automatic and assisted covering-family failures
remain. Useful native structure, all seven covering families, independent final
review and the full DGP-led scope remain outstanding; request clearer or less-
covered crops when usable structure is insufficient. Goal active/incomplete.

[V35 audited findings and actual review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_RESULTS.md>)
[V36 five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_VM.md>)

The entire previous handoff follows. Its V35-unrun wording is historical and
superseded by this verified return; earlier quality failures remain binding.

'''
    OUT.mkdir()
    (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    saved = OUT / 'before_docs/PROJECT_HANDOFF.md'
    saved.write_bytes(before)
    at = before.index(b'\n') + 1
    handoff.write_bytes(before[:at] + addition.encode('utf-8') + before[at:])
    files = [handoff, saved, ROOT / 'CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_RESULTS.md',
        ROOT / 'CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_VM.md', PREVIOUS / 'milestone.json',
        PREVIOUS / 'independent_closure_audit.json', PREVIOUS / 'final_readback.json',
        ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_return_import.json',
        ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_independent_audit_r2.json']
    scripts = ['supervise_cctv_dgp_v35_r1_return_audit_v1.py',
        'preserve_cctv_dgp_v35_r1_prefix_failure_and_prepare_r2.py',
        'supervise_cctv_dgp_v35_r1_return_audit_r2.py',
        'audit_cctv_dgp_group_guard_probe_v35_r1_return_r2.py',
        'analyze_cctv_dgp_group_guard_probe_v35_r1_return_v1.py',
        'analyze_cctv_dgp_group_guard_probe_v35_r1_return_v1_r1.py',
        'record_cctv_dgp_v35_r1_visual_review_v1.py', 'verify_cctv_dgp_v35_r1_analysis_and_pages_v1.py',
        'analyze_cctv_dgp_v35_finite_clearance_v1.py', 'cctv_dgp_finite_clearance_geometry_v36.py',
        'prepare_cctv_dgp_finite_clearance_probe_v36.py', 'cctv_dgp_finite_clearance_probe_v36_vm.py',
        'audit_cctv_dgp_finite_clearance_probe_v36_return.py', 'verify_cctv_dgp_finite_clearance_probe_v36_packet.py',
        'record_cctv_dgp_v35_return_v36_probe_milestone.py', 'verify_cctv_dgp_v35_return_v36_probe_milestone.py']
    files.extend(ROOT / 'scripts' / name for name in scripts)
    for folder in ['cctv_dgp_group_guard_probe_v35_r1_return', 'cctv_dgp_group_guard_probe_v35_r1_audit_run_v1',
        'cctv_dgp_group_guard_probe_v35_r1_audit_run_r2', 'cctv_dgp_group_guard_probe_v35_r1_audit_correction_r2',
        'cctv_dgp_group_guard_probe_v35_r1_analysis_v1', 'cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1',
        'cctv_dgp_v35_finite_clearance_v1', 'cctv_dgp_finite_clearance_probe_v36_vm',
        'cctv_dgp_finite_clearance_probe_v36_preparation']:
        files.extend(q for q in sorted((ROOT / 'outputs' / folder).rglob('*')) if q.is_file())
    files.extend(ROOT / 'outputs' / ('cctv-dgp-group-guard-probe-v35-r1' + suffix)
                 for suffix in ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json'])
    files.extend(ROOT / 'outputs' / ('cctv-dgp-finite-clearance-probe-v36' + suffix)
                 for suffix in ['-execution.tar.gz', '-execution.tar.gz.sha256'])
    evidence = {q.relative_to(ROOT).as_posix(): sha(q) for q in files}
    result = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'new_evidence_sha256': evidence, 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
        'previous_handoff_path': saved.relative_to(ROOT).as_posix(),
        'document': {'before_sha256': hashlib.sha256(before).hexdigest(),
                     'after_sha256': sha(handoff), 'addition_bytes': len(addition.encode('utf-8'))},
        'V35_members_verified': 2135, 'V35_CPU_replays': 100, 'V35_all100_cases_actually_viewed': True,
        'V35_all4_scales_failed_both_TRAIN_subsets': True, 'V35_new_checkpoint': False,
        'all108_affine_constraints_and65_calibrations_verified': True,
        'V36_packet_verified': True, 'V36_VM_execution_started': False,
        'all14_app_bindings_unchanged': True, 'local_neural_training_or_autograd_calls': 0,
        'new_VM_calls': 0, 'native_or_reserved_final_used': False, 'failure_records_preserved': True,
        'covering_family_failures_remain_binding': True, 'independent_final_review': False,
        'app_promotion': False, 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'bindings': len(evidence), 'milestone_sha256': sha(OUT / 'milestone.json')}))


if __name__ == '__main__':
    main()
