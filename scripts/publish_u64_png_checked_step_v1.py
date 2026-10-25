"""Publish a verified, unrun PNG-checked finite test; preserve exact prior documents."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation'
OUT = PREP / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    if OUT.exists():
        raise ValueError('Fresh publication required; retain any previous partial work')
    required = {
        'source_review_v1.json': 'source_review_pass',
        'source_tests_v1.json': 'all_tests_pass',
        'archive_readback_v1.json': 'archive_review_pass',
        'guide_review_v1.json': 'guide_review_pass',
    }
    consumed = []
    for name, field in required.items():
        path = PREP / name
        value = json.loads(path.read_bytes())
        if value.get('complete') is not True or value.get(field) is not True or value.get('unresolved_blockers'):
            raise ValueError('Independent verification incomplete: ' + name)
        consumed.append(pin(path))
    build_path = PREP / 'build_receipt_v1.json'
    build = json.loads(build_path.read_bytes())
    consumed.append(pin(build_path))
    archive = ROOT / build['archive']['path']
    checksum = ROOT / build['checksum']['path']
    stage = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
    if pin(archive) != build['archive'] or pin(checksum) != build['checksum']:
        raise ValueError('Sealed transfer changed')
    protocol = pin(stage / 'protocol.json')
    if {k: protocol[k] for k in ('bytes', 'sha256')} != {k: build['protocol'][k] for k in ('bytes', 'sha256')}:
        raise ValueError('Sealed protocol changed')
    consumed.extend((pin(archive), pin(checksum), protocol))
    guide = ROOT / 'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_VM.md'
    reviewed_guide = json.loads((PREP / 'guide_review_v1.json').read_bytes())['guide']
    if pin(guide) != reviewed_guide:
        raise ValueError('Guide changed after independent review')
    consumed.append(pin(guide))
    for row in build['sources']:
        if pin(ROOT / row['path']) != row:
            raise ValueError('Reviewed source changed: ' + row['path'])
        consumed.append(row)
    previous_path = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/publication_v1/publication_receipt_v1.json'
    previous = json.loads(previous_path.read_bytes())
    expected = {row['after']['path']: row['after'] for row in previous['changes']}
    consumed.append(pin(previous_path))
    for name in DOCS:
        if pin(ROOT / name) != expected[name]:
            raise ValueError('Concurrent document change before publication: ' + name)
    prefix = f'''# 23 October 2026: actual-PNG checked finite step prepared for manual L4 run

[Verified five-step VM guide](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_VM.md).
[Audited zero-update diagnostic](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md).
[Selection design](CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md).

STATUS: PREPARED_PNG_CHECKED_STEP_V1_UNRUN.
The completed PNG-surrogate measurement passed transport, saved pixels, states,
gradient and all12sheet audits. It applied zero updates; all575 tensors remain
exact originalu64. Its oldRAW direction already descends all16pooled signals;
adding near-duplicate identity-surrogate gradients is not a demonstrated fix.
The previous gate-aligned update2 remains failed and its earlieru1 unselected.

One distinct actual-PNG finite selection test is now source-reviewed, tested,
sealed and independently archive/guide-audited for MANUAL L4 execution. Transfer
{build['archive']['bytes']:,} bytes, SHA256
`{build['archive']['sha256']}`;
protocol {build['protocol']['bytes']:,} bytes, SHA256
`{build['protocol']['sha256']}`. The actual34member archive,
33staticbindings,16approvedsources and78puretests pass. All469parentfiles and
288retainedmeasurementfiles rehash locally. Two local parent model aliases and
267runtimeextractionaliases document provenance; the VM requires actual real
files in both original sibling stages, with no alias substitution.

Initialize exact complete original trainedu64, not a failed successor. Reuse128
retainedRAW gradient witnesses from the completed72/256zero-update measurement;
compute no new autograd queries and construct no optimizer. Four fixed maximum
coordinate scales2e-5/1.5e-5/1e-5/5e-6 usefixed.99headroom. Each trial independently
starts originalu64, saves64actualoptimizationTRAIN RAW/PNG cases, then restores
original. Every actual rounded application retains original2.5e-5per-step and
1e-4cumulative bounds, all539R2/U96 tensors exact, and candidate/commit8RAWnegative
dots. Restores are explicit reversals, never hidden as inference or descent.

All four complete banks precede first-eligible selection: all16 RAW/PNG rolemean
MAE/adjacent endpoints nonincrease versus originalu64; both blurPNGadjacent means
strictly improve, with zero quality tolerance. Selector API excludes monitors
and guards. At mostone selected training update;436fullforwards andnine explicit
applications (4trial/4restore/1commit),zero newgradientqueries. Noeligibleprefix
stops after380forwards/eightapplications with originalhead restored. Final56case
monitor/guard screen binds all inherited teachers/individualbehavior/1percent
structure and clear/low-light requirements once. Failure rejects entiretest;
no nextcandidate or oldstate fallback. Fatal invariants stop immediately; any
partial copied write remains unknown and cannot be hidden by a later restore.

Require8GiB free before launch and retain4GiB. Worker1200s/external1260s plus20s
cleanup,phases180s;export300/330s. RAM10GiB,reservedCUDA18GiB,newoutput/archive
2GiB each. Estimates are unmeasured for this route. All initial/final575 states,
trial/restore/commithead/checkpointvalues,actualroundeddisplacements,64case
individualregressions,selection,RNG/sampler/nulloptimizer,journal/resources and
failures are retained. No automatic retry,resume or historicalpilotlaunch.

No VM connection,actualmodelcall,newautogradquery or parameterapplication took
place during this preparation. Actual tests remain user-manualL4. A passing
finiteTRAIN screen is not useful restoration or model qualification; returned
pixels/states/selection/gates and every planned visualsheet require independent
audit, then broaderDEV/user acceptance/FINAL/app verification remain mandatory.
Originalu64 remains retained challenger;U96 remains experimental app incumbent.
No qualified successor or automatic goal advancement has been declared.

All checkpoints,splits,provenance,archives and gatefailures remain retained;
broaderDEV failures bind. Native/wild stayunpaired with no ethnicity or local
Zamboanga-performance claim. FINAL staysunopened/execution-unfrozen,reviewers
pending. ReviewedProtect/Apply/bypass remains separate;automaticmatching and
completion development staydeferred. Full goal ACTIVE/incomplete. The retained
design's earlier source-only status and statuses below are historical. Exact
prior document bytes follow.

---

'''.encode('utf8')
    OUT.mkdir()
    changes = []
    for name in DOCS:
        path = ROOT / name
        before = path.read_bytes()
        if pin(path) != expected[name]:
            raise ValueError('Concurrent publication: ' + name)
        backup = OUT / (name + '.before')
        with backup.open('xb') as stream:
            stream.write(before); stream.flush(); os.fsync(stream.fileno())
        staged = OUT / (name + '.new')
        with staged.open('xb') as stream:
            stream.write(prefix + before); stream.flush(); os.fsync(stream.fileno())
        if staged.read_bytes() != prefix + before or pin(path) != expected[name]:
            raise ValueError('Staged/current publication changed')
        os.replace(staged, path)
        if path.read_bytes()[len(prefix):] != backup.read_bytes():
            raise ValueError('Historical byte suffix differs')
        changes.append({'before': expected[name], 'backup': pin(backup), 'after': pin(path),
                        'prefix_bytes': len(prefix), 'prior_bytes_preserved_exactly': True})
    if any(pin(ROOT / row['path']) != row for row in consumed):
        raise ValueError('Consumed evidence changed during publication')
    receipt = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'PREPARED_PNG_CHECKED_STEP_V1_UNRUN', 'changes': changes,
        'consumed_before_after_exact': consumed, 'all_historical_bytes_retained': True,
        'manual_guide_ready': True, 'neural_gradient_optimizer_training_VM_calls': 0,
        'actual_parameter_applications': 0, 'model_qualified': False,
        'automatic_goal_advancement_triggered': False, 'full_goal': 'ACTIVE/incomplete',
        'source': pin(Path(__file__).resolve())}
    with (OUT / 'publication_receipt_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print({'complete': True, 'documents': len(changes), 'manual_guide_ready': True, 'launched': False})


if __name__ == '__main__':
    main()
