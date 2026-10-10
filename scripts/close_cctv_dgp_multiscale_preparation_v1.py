"""Record audited design/preparation while preserving every original/failure binding."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_multiscale_calibration_contract_v1 import NAME, read, write, sha, verify


def main():
    out = ROOT / 'outputs'; dest = out / 'cctv_dgp_multiscale_calibration_v1_milestone'
    assert not dest.exists()
    prep = read(out / 'cctv_dgp_multiscale_calibration_v1_preparation.json')
    transfer = read(out / 'cctv_dgp_multiscale_calibration_v1_transfer_audit_r1/independent_audit.json')
    parity = read(out / 'cctv_dgp_multiscale_calibration_v1_parity/independent_audit.json')
    learning = read(out / 'cctv_dgp_head4_learning_signal_review_v1/independent_audit.json')
    assert transfer['complete'] and parity['complete'] and learning['complete']
    assert transfer['archive_sha256'] == prep['archive_sha256'] and transfer['protocol_sha256'] == prep['protocol_sha256']
    packet = verify(out / NAME, prep['protocol_sha256'])
    previous = read(out / 'cctv_dgp_head4_capacity_vm_v1/protocol.json')
    protected = dict(previous['local_sources']); protected.update(packet['local_sources'])
    checkpoints = read(out / 'cctv_dgp_head4_capacity_v1_return_milestone/closure.json')['original_checkpoints_sha256']
    protected.update(checkpoints)
    rejected = out / 'cctv_dgp_head4_capacity_vm_v1_return'
    manifest = read(rejected / 'export_manifest.json')
    failure_paths = ['outputs/failure.json', 'outputs/early_gate.json', 'outputs/results.json',
        'outputs/update50/candidate.pth', 'outputs/update50/training_state.pt', 'outputs/stopped_training_state.pt']
    for n in failure_paths:
        protected[(rejected / n).relative_to(ROOT).as_posix()] = manifest['files_sha256'][n]
    for n, d in protected.items():
        assert sha(ROOT / n) == d, n
    assert not (out / NAME / 'outputs').exists(), 'Manual run has not begun locally'
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']
    assert not any(n in protected for n in docs)
    prefix = '''**Latest verified status — 10 October 2026: failed Head4 capacity retained; balanced multiscale/rate diagnostic prepared and independently checked. Manual VM run pending; full goal incomplete.**

The downloaded 50-update Head4 pilot remains rejected: raw structure gain
0.00608187% versus 1%, two blurred-source group MSE regressions and 53.2186%
brightness-only share versus the 20% maximum. Its checkpoint, full stopped state,
original assertions, all saved outputs and failure remain; no failed-state resume.

The saved-gradient review independently checks 1680 products with zero neural,
gradient or optimizer calls. A favorable averaged initial direction did not
validate the per-reference recipe: 8/20 individual directions predict worse
detail in at least one source/cohort group. This is initial TRAIN evidence,
not a unique causal explanation or a forecast across fifty updates/epochs.
See CCTV_DGP_POST_HEAD4_CAPACITY_DESIGN_REVIEW.md.

The new initializer preserves all100 current-DGP CPU outputs exactly; ten cases
are independently replayed. The frozen finite calibration compares deep3 with
the complete original decoder15, three rates and two balanced exposed TRAIN
pools. Twelve independent arms have one update/50 fitting exposures each,
280 preceding gradient queries, separate raw/PNG results and all24 native
development crops as unpaired review evidence. No final identity pixels enter
fitting, rate choice or tuning. All64 planned comparison pages require review.
Archive/member/source/data/launch guards pass. Actual L4 gradients, finite
outputs, useful native appearance and longer training are still unverified.

The existing VM was API-verified stopped; guest disk space is unknown. Manual
start/upload/tmux/download commands are in CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md.
Require7GiB after installation, runtime projection, a1GiB reserve and2.5GiB
output cap. Model work stops at1800seconds and export at600seconds; external
supervision is also bounded. Local work used no new gradients or updates.

Original1%/10% and appearance requirements remain. Current app checkpoint/model
selection is unchanged. The proposed additional-epoch1/2/5 study, five milestones,
independent final review and allseven completion families remain incomplete.
Native and paired synthetic evidence remain separate; no ethnicity, hidden
identity or Zamboanga performance claim follows. Historical statuses below are
superseded; closed pilot commands must not be automatically rerun.

---

'''
    dest.mkdir(); (dest / 'before').mkdir(); before = {}; after = {}
    write(dest / 'plan.json', {'protected_sha256': protected, 'canonical_documents': docs,
        'prefix_utf8': prefix, 'manual_run_pending': True, 'model_qualification': False, 'goal_complete': False})
    for n in docs:
        f = ROOT / n; old = f.read_bytes(); before[n] = sha(f)
        with (dest / 'before' / n).open('xb') as stream:
            stream.write(old)
        f.write_bytes(prefix.encode('utf-8') + old); after[n] = sha(f)
    for n, d in protected.items():
        assert sha(ROOT / n) == d, n
    write(dest / 'closure.json', {'complete': True, 'before_sha256': before, 'after_sha256': after,
        'protected_sha256': protected, 'failure_artifacts': failure_paths,
        'preparation_sha256': sha(out / 'cctv_dgp_multiscale_calibration_v1_preparation.json'),
        'transfer_audit_sha256': sha(out / 'cctv_dgp_multiscale_calibration_v1_transfer_audit_r1/independent_audit.json'),
        'guide_sha256': sha(ROOT / 'CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md'),
        'design_review_sha256': sha(ROOT / 'CCTV_DGP_POST_HEAD4_CAPACITY_DESIGN_REVIEW.md'),
        'manual_VM_run_pending': True, 'local_optimizer_updates': 0, 'local_gradient_queries': 0,
        'model_qualification': False, 'goal_complete': False})
    # Independent readback uses complete bytes, not only a matching final prefix.
    closure = read(dest / 'closure.json'); plan = read(dest / 'plan.json')
    for n in docs:
        saved = dest / 'before' / n
        assert sha(saved) == closure['before_sha256'][n]
        assert (ROOT / n).read_bytes() == plan['prefix_utf8'].encode('utf-8') + saved.read_bytes()
        assert sha(ROOT / n) == closure['after_sha256'][n]
    for n, d in plan['protected_sha256'].items():
        assert sha(ROOT / n) == d
    write(dest / 'independent_closure_audit.json', {'complete': True, 'canonical_documents_checked': 3,
        'original_document_bytes_preserved': True, 'protected_bindings_unchanged': len(protected),
        'original_checkpoints_verified': 2, 'rejected_checkpoint_and_full_failure_state_retained': True,
        'prepared_archive_transfer_audited': True, 'local_optimizer_updates': 0, 'local_gradient_queries': 0,
        'manual_VM_run_pending': True, 'model_qualification': False, 'goal_complete': False})
    print({'complete': True, 'canonical_documents': 3, 'protected_bindings': len(protected), 'manual_run_pending': True}, flush=True)


if __name__ == '__main__':
    main()
