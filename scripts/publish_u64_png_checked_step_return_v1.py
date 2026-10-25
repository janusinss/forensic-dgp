"""Record the independently audited failed return without altering historical bytes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit'
OUT = AUDIT / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
GUIDE = 'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_VM.md'
PREFIX = '''# 23 October 2026: PNG-checked step executed; audited quality stop

[Audited return](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_RETURN_REVIEW.md).
[Three-route diagnostic matrix](CCTV_CODEFORMER_U64_POST_CHECKED_STEP_DIAGNOSTIC_MATRIX_V1.md).

STATUS: EXECUTED_PNG_CHECKED_STEP_V1_AUDITED_QUALITY_STOP.
The manual L4 run executed all four fixed trials, selected trial 0 by the
optimization TRAIN rule, and failed the separate unchanged final monitor
screen. Worker exit 2, no timeout, known child/group cleanup; 281.732 seconds
worker status / 285.157 seconds supervisor. The result archive is 1,112,595,923
bytes, SHA256
`f2d8f2670c0c4b7fd9310317d1a020ffb9f0b1211e07b9d0ed92023e7832fafc`.
All 1,469 archive members, 33 static and 757 dependency bindings verify.
Final export-process exit receipts are outside this archive; no exit inferred.

Independent audits verify 436 RAW/floor-PNG pairs, 1,341 journal events and
445 completed operations: four trial applications, four original restorations
and one selected commit. Zero new queries and zero optimizer updates; the
128 retained RAW gradient witnesses and complete original u64 initializer
authenticate. All 539 frozen R2/U96 tensors remain bitwise exact. The failed
selected state and all original/trial/restore/RNG/sampler records are retained.

Trials 0/1/2 passed the 64-case optimization role-mean RAW/PNG screen; trial 3
did not. First eligible trial 0 was selected before any monitor evaluation.
Inherited and original terminal gates and strict clear/low-light MAE checks
passed, but final local-blur PNG adjacent error rose from 0.01697094396257641
to 0.01697106960719266, delta +1.2564461624939294e-7. Original/prior duplicate
checks concern the same endpoint. All eight local monitor RAW errors improved,
but five PNG errors worsened. Saved-code arithmetic confirms 882 changed RGB
components and net +25 target-edge residual units. No tolerance or quantizer
change; no monitor-driven fallback to trials 1/2. This tiny error increase is
not a replacement for the original u64's historical 1.627% global gain.

Root reviewed all 21 exact saved RGB256 sheets; 664 cells and four identical
repeat aliases cover all 436 forwards. No convincing useful visible gain over
original u64; shared smoothing and severe-blur detail uncertainty persist.
Independent saved-cell and integer-endpoint readbacks pass. Audit success does
not overrule quality failure. There was no local model/autograd/training/
parameter application, VM connection or browser run during this return audit.

Three u64 preservation recipes have now failed. AGENTS.md's circuit breaker
requires a diagnostic discussion before further model/training changes. The
pending choice is training examples/loss design (recommended first review)
versus trainable reconstruction architecture. The evidence matrix is read-only
analysis, not a new training plan or runnable packet. Do not resume, append
epochs, rerun an unchanged route, or use historical launch commands below.

Original u64 remains retained; U96 stays experimental app incumbent. No model
promotion or automatic goal advancement. Checkpoints, splits, provenance,
failures and broader DEV limits remain binding. Native/wild remain unpaired;
FINAL stays unopened/execution-unfrozen, independent reviewers pending.
Protect/Apply preparation and bypass remain separate; automatic matching and
completion development stay deferred. The full goal remains ACTIVE/incomplete.
Exact preceding document bytes follow as historical evidence.

---

'''
GUIDE_PREFIX = '''# Completed failed experiment: do not rerun this guide

23 October 2026. The manual run reached its final monitor screen and stopped
with worker exit 2. Archive, state, saved-pixel and all 21-sheet audits are
complete; localized-blur PNG preservation failed. No qualified successor.

[Return review](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_RETURN_REVIEW.md).
[Diagnostic matrix](CCTV_CODEFORMER_U64_POST_CHECKED_STEP_DIAGNOSTIC_MATRIX_V1.md).
U96 remains app incumbent; original u64 remains retained. A diagnostic choice
is pending under AGENTS.md after three failed preservation recipes. No next
runnable worker or upload guide is ready. The commands below are historical:
do not resume, add epochs or rerun this completed recipe.
Exact preceding guide bytes follow.

---

'''


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    if OUT.exists():
        raise ValueError('Preserve prior/partial publication; fresh directory required')
    required = {
        'outputs/codeformer_u64_png_checked_step_v1_return_transport/receipt_v1.json': 'transport_audit_pass',
        'outputs/codeformer_u64_png_checked_step_v1_return_states/receipt_v1.json': 'saved_state_direction_application_audit_pass',
        'outputs/codeformer_u64_png_checked_step_v1_return_pixels/receipt_v1.json': 'saved_pixel_metric_selection_consistency_pass',
        'outputs/codeformer_u64_png_checked_step_v1_return_audit/visual_readback_v1.json': 'visual_readback_pass',
        'outputs/codeformer_u64_png_checked_step_v1_return_audit/root_visual_review_v1.json': 'root_personally_inspected_all_planned_sheets',
        'outputs/codeformer_u64_png_checked_step_v1_return_audit/report_design_readback_v1.json': 'report_design_review_pass',
        'outputs/codeformer_u64_png_checked_step_v1_return_audit/endpoint_diagnosis_v1.json': 'saved_endpoint_diagnosis_pass',
    }
    consumed = []
    for name, key in required.items():
        path = ROOT / name; value = json.loads(path.read_bytes())
        if value.get('complete') is not True or value.get(key) is not True:
            raise ValueError('Incomplete return review: ' + name)
        for blocker in ('unresolved_blockers', 'unresolved_transport_blockers', 'failed_checks'):
            if value.get(blocker):
                raise ValueError('Unresolved evidence review: ' + name)
        consumed.append(pin(path))
    review = json.loads((AUDIT / 'report_design_readback_v1.json').read_bytes())
    approved = {row['path']: row for row in review['reviewed_documents']}
    document_names = ('CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_RETURN_REVIEW.md',
                      'CCTV_CODEFORMER_U64_POST_CHECKED_STEP_DIAGNOSTIC_MATRIX_V1.md')
    if set(approved) != set(document_names):
        raise ValueError('Review must bind exactly the report and diagnosis matrix')
    for name in document_names:
        current = pin(ROOT / name)
        if current != approved[name]:
            raise ValueError('Document changed since independent review: ' + name)
        consumed.append(current)
    if pin(AUDIT / 'root_visual_review_v1.json') != review['reviewed_root_visual']:
        raise ValueError('Root visual review changed since independent review')
    previous_path = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation/publication_v1/publication_receipt_v1.json'
    previous = json.loads(previous_path.read_bytes()); consumed.append(pin(previous_path))
    oldpins = {row['after']['path']: row['after'] for row in previous['changes']}
    guide_path = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation/guide_review_v1.json'
    oldpins[GUIDE] = json.loads(guide_path.read_bytes())['guide']; consumed.append(pin(guide_path))
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
            raise ValueError('Concurrent document/stage change before replacement')
        os.replace(staged, path)
        if path.read_bytes()[len(prefix):] != backup.read_bytes():
            raise ValueError('Historical suffix changed')
        changes.append({'before': oldpins[name], 'backup': pin(backup), 'after': pin(path),
                        'prefix_bytes': len(prefix), 'prior_bytes_preserved_exactly': True})
    if any(pin(ROOT / row['path']) != row for row in consumed):
        raise ValueError('Audited evidence changed during publication')
    receipt = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'EXECUTED_PNG_CHECKED_STEP_V1_AUDITED_QUALITY_STOP',
        'changes': changes, 'consumed_before_after_exact': consumed,
        'all_historical_bytes_retained': True, 'original_u64_retained': True,
        'app_incumbent': 'U96 experimental; unchanged', 'model_qualified': False,
        'next_worker_or_guide_ready': False, 'diagnostic_choice_pending': True,
        'actual_training_or_parameter_applications_local': 0, 'VM_calls': 0,
        'automatic_goal_advancement_triggered': False, 'full_goal': 'ACTIVE/incomplete',
        'source': pin(Path(__file__).resolve())}
    with (OUT / 'publication_receipt_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'complete': True, 'documents': len(changes), 'next_worker_ready': False,
           'model_qualified': False, 'full_goal': 'ACTIVE/incomplete'})


if __name__ == '__main__':
    main()
