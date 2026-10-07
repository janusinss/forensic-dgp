"""Record the audited diagnostic and frozen manual pilot; preserve prior documents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_sampling_review_and_profile_batches_v31_milestone'
PREVIOUS = ROOT / 'outputs/dgp_v30_return_and_sampling_gradient_v1_milestone'
DOCUMENTS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists()
    old = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == '2dddd0b8bd9f4c443e16a103afdecbf9e940266aaed554a68156bd5211594a09'
    for row in old['documents']: assert sha(ROOT / row['name']) == row['after_sha256']
    audit = read(ROOT / 'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json')
    packet = read(ROOT / 'outputs/cctv_dgp_profile_batches_v31_preparation/independent_packet_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['optimizer_updates'] == 0
    assert packet['complete'] and packet['regressions_passed'] == 10 and not packet['actual_VM_training_started']
    addition = '''
**Latest research milestone - 7 October 2026: sampling diagnostic audited; V31 paired-batch pilot ready.**

The human returned the distinct V30 sampling-gradient diagnostic. The independent
local audit verifies all 756 files, 280 saved component-gradient queries and 200
CPU inference outputs across original/stopped states. VM diagnostic runtime was
34.729 seconds, with zero optimizer updates, zero backwards and no new checkpoint.
All 12 selected decoder tensors have finite nonzero improvement gradients.
The audit used no local autograd or training. Original models and V30's failed
50-update/0.805717% structure requirement remain preserved.

The stopped candidate drifts on four unexposed clear TRAIN controls: the largest
raw observed-pixel MSE increase is 16.2719%. One motion case also activates the
pixel-regression hinge. Preservation gradients therefore protect a measured
regression. The stopped negative total-objective direction increases the
whole-observed-detail term in both measured cohorts. These are local gradient
and raw-loss observations, not a reconstruction of AdamW steps or a unique causal
proof. The cohorts are TRAIN data; source labels are not ethnicity. Raw objective
drift is separate from delivered PNG gates and paired synthetic PSNR/SSIM.

V31 changes batch formation only: each approved reference's clear control and
four degradations share one batch. The full 781-reference/3,905-case TRAIN corpus,
original DGP initialization, mean-centered decoder path, 12 trainable tensors,
seven losses, fixed initial50 normalizers, AdamW and all numerical gates stay
fixed. The frozen V30 permutation determines reference order. The first 781
updates cover every TRAIN case once; 19 further batches complete the finite
800-update maximum. The unchanged early gate stops at50 unless structure gain
reaches1%; final gates require10%, all17 preservation groups, nonnegative source
gains and mean-only fraction<=20%. No V30 continuation or unchanged failed rerun.

The nine-file 751,432-byte transfer packet is independently verified, including
10 regressions, Python3.10 syntax, read-only Bash syntax and rejection of local
training before model imports. Protocol SHA256:
ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e.
Execution archive SHA256:
bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8.
Actual V31 VM training remains unstarted. Follow the five manual upload/install/
tmux/launch/download steps on the existing L4/g2-standard-4 VM at ~/forensic-dgp.
Require6GiB free; timing, VRAM and export stops are enforced. Historical storage
measurements are not a fresh free-space reading. The local research-cache backup,
original checkpoints, splits, provenance and all failures remain retained.

Returned V31 results require an independent audit and all50 TRAIN preview review.
Only passing capacity justifies repeating the fixed520 paired DEV cases and24
unpaired native development crops. V29 development/native failures remain binding;
reserved final pixels remain unopened. No new native or evaluation images were
used in this diagnostic or packet. No real Zamboanga CCTV evidence exists yet.

The DGP-led app is unchanged. Useful native structure, all seven covering families
with separate automatic/assisted review, independent final review and the full
bundled inline Playwright app flow remain required. Preserve clear glasses,
non-obstructing hair and all visible facial appearance; request a clearer or
less-covered crop when usable information is insufficient. Goal active/incomplete.

[Audited diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V30_SAMPLING_GRADIENT_V1_RESULTS.md>)
[V31 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_VM.md>)

The complete earlier document body is preserved below. Its diagnostic-unrun
language is historical and superseded by this audited human return.

'''.encode('ascii')
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    rows = []
    for name in DOCUMENTS:
        path = ROOT / name; before = path.read_bytes(); archived = OUT / 'before_docs' / name
        with archived.open('xb') as stream: stream.write(before)
        split = before.index(b'\n') + 1; after = before[:split] + addition + before[split:]
        path.write_bytes(after)
        assert after[:split] + after[split + len(addition):] == before
        rows.append({'name': name, 'before_path': archived.relative_to(ROOT).as_posix(), 'before_sha256': sha(archived),
                     'after_sha256': sha(path), 'addition_bytes': len(addition), 'complete_previous_body_preserved': True})
    files = [ROOT / name for name in DOCUMENTS] + [OUT / 'before_docs' / name for name in DOCUMENTS]
    for directory in ['outputs/cctv_dgp_profile_batches_vm_v31', 'outputs/cctv_dgp_profile_batches_v31_preparation',
                      'outputs/cctv_dgp_v30_sampling_gradient_v1_analysis', 'outputs/cctv_dgp_v30_sampling_gradient_v1_audit_execution_v1']:
        files.extend(f for f in (ROOT / directory).rglob('*') if f.is_file())
    names = ['CCTV_DGP_V30_SAMPLING_GRADIENT_V1_RESULTS.md', 'CCTV_DGP_PROFILE_BATCHES_V31_VM.md',
             'outputs/cctv-dgp-profile-batches-v31-execution.tar.gz', 'outputs/cctv-dgp-profile-batches-v31-execution.tar.gz.sha256',
             'outputs/cctv-dgp-v30-sampling-gradient-v1-results.tar.gz', 'outputs/cctv-dgp-v30-sampling-gradient-v1-results.tar.gz.sha256',
             'outputs/cctv-dgp-v30-sampling-gradient-v1-export.json', 'outputs/cctv_dgp_v30_sampling_gradient_v1_return_import.json',
             'outputs/cctv_dgp_v30_sampling_gradient_v1_independent_audit.json',
             'scripts/analyze_cctv_dgp_v30_sampling_gradient_v1.py', 'scripts/prepare_cctv_dgp_profile_batches_v31.py',
             'scripts/cctv_dgp_profile_batches_v31_schedule.py', 'scripts/verify_cctv_dgp_profile_batches_v31_packet.py',
             'scripts/cctv_dgp_profile_batches_v31_return_audit_template.py', 'scripts/audit_cctv_dgp_profile_batches_v31_return.py',
             'tests/test_cctv_dgp_profile_batches_v31.py', Path(__file__).relative_to(ROOT).as_posix(),
             'scripts/verify_cctv_dgp_sampling_review_and_profile_batches_v31_milestone.py']
    files.extend(ROOT / name for name in names)
    milestone = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
                 'scope': 'Audited zero-update TRAIN diagnostic and prepared single-variable manual VM pilot; no app qualification',
                 'documents': rows, 'new_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(set(files))},
                 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
                 'previous56_original_locations': {row['name']: row['before_path'] for row in rows},
                 'previous_final_readback_sha256': sha(PREVIOUS / 'final_readback.json'),
                 'diagnostic_archive_sha256': audit['archive_sha256'], 'diagnostic_gradient_queries': 280,
                 'diagnostic_optimizer_updates': 0, 'CPU_inference_outputs_replayed': 200,
                 'V31_protocol_sha256': packet['protocol_sha256'], 'V31_archive_sha256': packet['archive_sha256'],
                 'V31_finite_updates': 800, 'V31_actual_VM_training_started': False,
                 'V30_failed_early_gate_retained': True, 'manual_VM_execution_required': True,
                 'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
                 'new_VM_actions': [], 'native_or_reserved_used': False, 'app_promotion': False,
                 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(milestone, stream, indent=2)
    print(json.dumps({'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'),
                      'new_bindings': len(milestone['new_evidence_sha256']), 'documents_updated': 3,
                      'VM_training_started': False, 'goal_complete': False}, indent=2))


if __name__ == '__main__': main()
