"""Publish the audited zero-update return, preserving exact previous document bytes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit'
OUT = AUDIT / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
GUIDE = 'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md'
PREFIX = '''# 23 October 2026: PNG-surrogate diagnostic completed and independently audited

[Audited return and interpretation](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md).
[Next finite-step design](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md).

STATUS: EXECUTED_PNG_SURROGATE_MEASUREMENT_V1_AUDITED_ZERO_UPDATES.
The user's manual L4 diagnostic completed: worker exit0, no timeout, known
child/group cleanup;128.140s worker status/130.919s supervisor. All673 archived
members authenticate. Return979,202,030 bytes SHA256
`0fe590ba344a6d0fe11a318f54dd66d0f1814ad751b7142927d471c29e7af3b2`;
protocolSHA256 `0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c`.
Independent replay confirms992 events,328 known operation pairs,72 forwards,
256 gradient queries,ZERO parameter applications,ZERO optimizer constructions
and no unknown pending operation. Empty run.log is legal. Final export-process
receipts are outside this archive; no finalexport exit is inferred.

All575 initial/closing tensors remain exactly original trainedu64,including36
heads and539 R2/U96 tensors. All256 gradient witnesses and1,642 saved-pixel
checks authenticate:72 exactRAW/floorPNG pairs,64 exactactualPNGdecodedFP32
surrogate forwards,256 target-bound scalars,4historicalparities+4fullrepeats.
Independent geometry matches18,306 values within3.55e-15 arithmetic readback
error. RNG/fixedsampler/nulloptimizer records agree. Root reviewed all12 exact
RGB sheets;204cells and4byte-identical repeat aliases cover all72outputs.
This completed measurement did not train or improve a model.

RAW and same-gate PNG-surrogate signals are strongly aligned. Low-light versus
nondark MAE conflicts persist. The independent saved-array full16-vector check
examined all65,535 simplex faces with synthetic/KKT checks. Crucially,the prior
eight-RAW direction ALREADY descends all16 pooled RAW/PNG-surrogate signals;
the full16direction has cosine0.99999338198 with that prior direction. It has
233negative/23positive individualquery dots. Adding near-duplicate identity
surrogates is not a demonstrated fix for the prior finitePNG monitor failure.
No near-identical16-solver training pilot is prepared or launched.

Next design changes finite-step selection: fixed4 independent-original trials,
actualRAW/PNG eligibility on the64optimizationTRAINcases,then one unchanged
56case monitor/guard screen. Proposed caps2e-5/1.5e-5/1e-5/5e-6 preserve old
2.5e-5per-application and1e-4cumulative bounds withfixed.99headroom. At mostone
selected update,nine explicittrial/restore/commit applications,436forwards,
zero newautogradqueries; fatalinvariants stop,qualityineligible trials remain
retained,no monitor-driven fallback. Worker1200s/external1260s+20cleanup,
phases180s,export300/330s,8GiBinitial/4GiBreserve,RAM10GiB/CUDA18GiB and2GiB
output/archive caps are prospective requirements,not implemented enforcement.
This follow-up currently consists of local design/pure-array prototype work.
NO runnable nextworker,sealedtransfer or nextmanualguide is ready. Independent
source/tests/archive review is required before any manual trial/application run.
Do not rerun this completed diagnostic or resume the failed gate-aligned pilot.

Originalu64 remains retained challenger;U96 remains experimental app incumbent.
Prior failed update2 and earlieru1 remain unselected. No qualified successor or
automatic goal advancement. All checkpoints,splits,failures,provenance,archives
and broaderDEV failures remain retained/binding. Native/wild evidence stays
unpaired;FINAL remains unopened/execution-unfrozen,independentreviewerspending.
ReviewedProtect/Apply/bypass remains separate;automaticmatching and completion
development remain deferred. All actualtraining/application work stays manualL4.
Full goal ACTIVE/incomplete. Exact historical document bytes follow.

---

'''
GUIDE_PREFIX = '''# Completed diagnostic: audited zero-update return; do not rerun

23 October 2026. Worker exit0;72forwards,256queries,zero weight updates.
All575 tensors remain exactly originalu64. Independent archive,pixels,gradient,
state and12-sheet reviews passed. Measurement success is not model improvement.

[Return review](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md).
[Next source-only design](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md).
The upload/tmux commands below are historical for the now-completed diagnostic.
Do not repeat them. No next runnable worker or manual transfer guide is ready.
U96 stays app incumbent;originalu64 remains retained. Full goal ACTIVE/incomplete.
Exact preceding guide bytes follow.

---

'''


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    if OUT.exists():
        raise ValueError('Retain any preceding/partial publication; fresh directory required')
    required = {
        'outputs/codeformer_u64_png_surrogate_measurement_v1_return_transport/receipt_v1.json': 'transport_audit_pass',
        'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels/receipt_v1.json': 'saved_pixel_scalar_consistency_pass',
        'outputs/codeformer_u64_png_surrogate_measurement_v1_return_gradients/receipt_v1.json': 'saved_state_gradient_geometry_audit_pass',
        'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/visual_readback_v1.json': 'visual_readback_pass',
        'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/root_visual_review_v1.json': 'root_personally_inspected_all_planned_sheets',
        'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/report_design_readback_v1.json': 'report_design_review_pass',
    }
    consumed = []
    for name, key in required.items():
        path = ROOT / name; value = json.loads(path.read_bytes())
        if value.get('complete') is not True or value.get(key) is not True or value.get('unresolved_blockers'):
            raise ValueError('Unresolved returned-evidence review: ' + name)
        consumed.append(pin(path))
    for name in ('feasibility_v1.json', 'inventory_v1.json'):
        path = AUDIT / name; value = json.loads(path.read_bytes())
        if value.get('complete') is not True:
            raise ValueError('Incomplete saved analysis: ' + name)
        consumed.append(pin(path))
    for name in ('CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md',
                 'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md'):
        consumed.append(pin(ROOT / name))
    prior_path = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_preparation/publication_v1/publication_receipt_v1.json'
    previous = json.loads(prior_path.read_bytes()); consumed.append(pin(prior_path))
    oldpins = {row['after']['path']: row['after'] for row in previous['changes']}
    guide_review_path = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_preparation/guide_review_v1.json'
    oldpins[GUIDE] = json.loads(guide_review_path.read_bytes())['guide']; consumed.append(pin(guide_review_path))
    for name in (*DOCS, GUIDE):
        if pin(ROOT / name) != oldpins[name]:
            raise ValueError('Current document differs from exact prior publication: ' + name)
    OUT.mkdir()
    changes = []
    for name in (*DOCS, GUIDE):
        path = ROOT / name; before = path.read_bytes()
        if pin(path) != oldpins[name]:
            raise ValueError('Concurrent document change: ' + name)
        backup = OUT / (name + '.before')
        with backup.open('xb') as stream:
            stream.write(before); stream.flush(); os.fsync(stream.fileno())
        prefix = (GUIDE_PREFIX if name == GUIDE else PREFIX).encode('utf8')
        staged = OUT / (name + '.new')
        with staged.open('xb') as stream:
            stream.write(prefix + before); stream.flush(); os.fsync(stream.fileno())
        if staged.read_bytes() != prefix + before or pin(path) != oldpins[name]:
            raise ValueError('Current/staged document changed before replace')
        os.replace(staged, path)
        if path.read_bytes()[len(prefix):] != backup.read_bytes():
            raise ValueError('Historical suffix changed')
        changes.append({'before': oldpins[name], 'backup': pin(backup), 'after': pin(path),
                        'prefix_bytes': len(prefix), 'prior_bytes_preserved_exactly': True})
    if any(pin(ROOT / row['path']) != row for row in consumed):
        raise ValueError('Audited evidence changed during publication')
    receipt = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'EXECUTED_PNG_SURROGATE_MEASUREMENT_V1_AUDITED_ZERO_UPDATES',
        'changes': changes, 'consumed_before_after_exact': consumed,
        'all_historical_bytes_retained': True, 'all575_original_unchanged': True,
        'actual_training_or_parameter_applications_local': 0, 'VM_calls': 0,
        'model_qualified': False, 'next_worker_or_guide_ready': False,
        'automatic_goal_advancement_triggered': False, 'full_goal': 'ACTIVE/incomplete',
        'source': pin(Path(__file__).resolve())}
    with (OUT / 'publication_receipt_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'complete': True, 'documents': len(changes), 'next_worker_ready': False,
           'model_qualified': False, 'full_goal': 'ACTIVE/incomplete'})


if __name__ == '__main__':
    main()
