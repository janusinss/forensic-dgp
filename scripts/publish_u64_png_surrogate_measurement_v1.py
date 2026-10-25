"""Publish a verified, unrun manual diagnostic; preserve exact prior documents."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_preparation'
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
    stage = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
    if pin(archive) != build['archive'] or pin(checksum) != build['checksum']:
        raise ValueError('Sealed transfer changed')
    protocol = pin(stage / 'protocol.json')
    if {k: protocol[k] for k in ('bytes', 'sha256')} != {k: build['protocol'][k] for k in ('bytes', 'sha256')}:
        raise ValueError('Sealed protocol changed')
    consumed.extend((pin(archive), pin(checksum), protocol))
    guide = ROOT / 'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md'
    reviewed_guide = json.loads((PREP / 'guide_review_v1.json').read_bytes())['guide']
    if pin(guide) != reviewed_guide:
        raise ValueError('Guide changed after independent review')
    consumed.append(pin(guide))
    for row in build['sources']:
        if pin(ROOT / row['path']) != row:
            raise ValueError('Reviewed source changed: ' + row['path'])
        consumed.append(row)
    previous_path = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/publication_v1/publication_receipt_v1.json'
    previous = json.loads(previous_path.read_bytes())
    expected = {row['after']['path']: row['after'] for row in previous['changes']}
    consumed.append(pin(previous_path))
    for name in DOCS:
        if pin(ROOT / name) != expected[name]:
            raise ValueError('Concurrent document change before publication: ' + name)
    prefix = f'''# 23 October 2026: zero-update RAW/PNG diagnostic ready for manual L4 execution

[Five-step upload, install, tmux and download guide](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md).
[Audited prior failure](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md).

STATUS: PREPARED_PNG_SURROGATE_MEASUREMENT_V1_UNRUN.
The returned gate-aligned training pilot is audited and failed at update two;
its localized PNG preservation regression remains binding. Neither that failed
state nor its earlier update one is selected. Do not resume or append epochs.
The exact original trained u64 remains retained; U96 remains experimental app
incumbent. No successor or automatic goal advancement is declared.

One distinct diagnostic is now source-reviewed, tested and sealed for the user's
manual L4 run. Transfer {build['archive']['bytes']:,} bytes, archive SHA256
`{build['archive']['sha256']}`;
protocol {build['protocol']['bytes']:,} bytes, SHA256
`{build['protocol']['sha256']}`. Actual gzip/USTAR, all22 members,
21 static bindings, 12 reviewed sources, 51 pure tests and the exact five-step
guide independently pass. All469 parent static files were rehashed locally;
two explicitly bound local model aliases do not substitute for real VM paths.
No VM connection, live-space admission, model forward, actual gradient query,
parameter application or new training occurred while preparing this packet.

Initialize the complete exact original trained-u64 state, not failed update two
or earlier update one. Execute at most72 full forwards and256 separate gradient
queries on the16 admitted optimization TRAIN photographs across four roles.
Four historical parity and four same-state repeats precede64 measured cases.
All575 tensors must remain bitwise unchanged: ZERO parameter applications,
ZERO optimizer constructions, no weight update, no common-direction solver.
RAW and actual decoded-PNG forward metrics produce16 separately named pooled
vectors. The PNG custom backward is explicitly a biased identity surrogate,
not the true derivative, a finite-step certificate or proof of useful detail.
The exact saved PNG floor exporter and every quality requirement stay intact.

Require8GiB free before launch and retain4GiB. Worker900s, external960s plus
20s cleanup; phases180s. Export300s internally/330s externally plus cleanup.
RAM10GiB, reserved CUDA18GiB, new outputs/archive2GiB each. No automatic retry,
historical launch or resume. The completed cleananchors parent is read-only;
the VM requires all469 actual installed static files. Retain full initial and
closing575 values, all256 gradient witnesses, all72 RAW/PNG pairs, eight
native/logits/LQ triples, RNG/sampler/resource/journal/failure receipts.
Returned bytes and outputs must be independently audited before a training
design is selected. A completed diagnostic creates no newly trained model.

All original checkpoints, splits, failures, provenance and prior return archives
remain retained. Broader DEV failures remain binding. Native/wild evidence stays
unpaired; no ethnicity or Zamboanga-performance claim. FINAL remains unopened
and execution-unfrozen, independent reviewers pending. Reviewed Protect/Apply/
bypass is a separate preparation study; automatic matching and completion
development remain deferred. All actual training remains manual L4. Full goal
ACTIVE/incomplete. Prior statuses below are historical; exact prior bytes follow.

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
        'status': 'PREPARED_PNG_SURROGATE_MEASUREMENT_V1_UNRUN', 'changes': changes,
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
