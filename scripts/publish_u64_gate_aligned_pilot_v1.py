"""Publish a reviewed manual packet, preserving exact preceding documents."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_preparation'
DEST = PREP / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    if DEST.exists():
        raise ValueError('Retain prior publication; do not overwrite')
    requirements = {'source_tests_v1.json': 'all_tests_pass',
                    'source_review_v1.json': 'source_review_pass',
                    'build_receipt_v1.json': 'complete',
                    'archive_review_v1.json': 'archive_review_pass',
                    'guide_review_v1.json': 'guide_review_pass',
                    'historical_scalar_integration_v1.json': 'pass'}
    consumed, receipts = [], {}
    for name, required in requirements.items():
        path = PREP / name
        value = json.loads(path.read_bytes())
        if value.get('complete') is not True or value.get(required) is not True or value.get('unresolved_blockers'):
            raise ValueError('Unresolved admission: ' + name)
        consumed.append(pin(path))
        receipts[name] = value
    build = receipts['build_receipt_v1.json']
    for row in build['sources']:
        if pin(ROOT / row['path']) != row:
            raise ValueError('Reviewed source changed: ' + row['path'])
        consumed.append(row)
    for key in ('archive', 'checksum'):
        row = build[key]
        if pin(ROOT / row['path']) != row:
            raise ValueError('Reviewed transfer changed')
        consumed.append(row)
    guide = ROOT / 'CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md'
    guide_pin = pin(guide)
    if receipts['guide_review_v1.json']['guide'] != guide_pin:
        raise ValueError('Reviewed guide changed')
    consumed.append(guide_pin)
    prefix = f'''# 23 October 2026: four-update original-u64 pilot ready for manual transfer

[Five-step upload, installation, tmux and download guide](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md).
[Audited diagnosis](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md).

STATUS: PREPARED_UNRUN_MANUAL_GATE_ALIGNED_PILOT_V1. The new runner and its
bounded supervisor are implemented, independently source-reviewed and sealed.
Its 62 source/array/file/stub-process tests pass with neural imports blocked.
They do not establish actual VM behavior or useful restoration. Archive and
command-guide reviews pass. No VM connection, upload, worker, model, gradient,
optimizer, training or parameter application occurred during this preparation.

Transfer archive {build['archive']['bytes']:,} bytes, SHA256
`{build['archive']['sha256']}`;
protocol SHA256 `{build['protocol']['sha256']}`.
The 26 static bindings plus protocol are sealed. Existing parent data/model
bindings were freshly rehashed: 469 files, including two explicitly local-only
model aliases. Actual VM admission requires all 469 installed parent paths.
The completed parent's failed eight-update training is never resumed.

This distinct pilot starts from complete original trained u64, keeps all 539
R2/U96 tensors frozen and permits at most four updates to its existing 36 heads.
It directly uses eight pooled photographic fidelity/structure gradients, with
normalization and a certified shared descent direction. A single fixed 0.99
factor provides numerical headroom; no metric-based line search or retries.
Actual FP32-rounded displacement must pass all eight dot checks and coordinate
caps of 2.5e-5 per update and 1e-4 from original u64. No Adam object or momentum
is constructed. These finite gradient-descent applications are actual training
and remain manual L4 work.

Maximum 540 full forwards, 512 gradient queries and four parameter applications.
Initial 56-case bank plus four repeats establish original-u64 parity. Every
update receives all 64 optimization cases, then all 56 quality cases. All RAW
and PNG role-mean fidelity/structure endpoints must not increase versus original
u64 AND the preceding accepted state. Original individual/legacy/teacher gates
remain unchanged. At terminal four, original 1% blur-structure gates bind and
clear/dark MAE must improve. Every case's changes and failed state are retained;
no earlier failed-pilot checkpoint is selected retrospectively.

Estimate 8-16 minutes is unmeasured. Require 8 GiB free and retain 4 GiB. Worker
limit 1800s; external 1860s plus bounded 20s cleanup; phase limit 180s. Export
300s internally/330s externally plus cleanup. RAM 10 GiB, reserved CUDA 18 GiB,
new outputs and return archive 2 GiB each. External writer cleanup must be known
before export. Complete state, head, gradient, RNG/sampler, null optimizer,
RAW/PNG and failure evidence are retained within declared caps.

Return review must verify actual updates, all saved outputs/metrics, regressions
and every planned visual sheet before any longer training or qualification.
Original u64 remains the retained challenger; U96 remains experimental app
incumbent. No new model was qualified and automatic goal advancement did not
trigger. All original checkpoints, splits, provenance and broader DEV failures
remain binding. Reviewed Protect/Apply/bypass remains separate; automatic blur
matching and later completion development remain deferred. Native/wild evidence
is unpaired; FINAL remains unopened/execution-unfrozen with reviewers pending.
Full goal ACTIVE/incomplete. Historical source-only and executed-guide statuses
below are retained snapshots; this prepared-packet status supersedes them.
Exact preceding document bytes follow.

---

'''.encode('utf-8')
    pending = [(ROOT / name, (ROOT / name).read_bytes(), pin(ROOT / name)) for name in DOCS]
    DEST.mkdir()
    changes = []
    for path, original, before in pending:
        if pin(path) != before:
            raise ValueError('Concurrent document change')
        backup = DEST / (path.name + '.before')
        with backup.open('xb') as stream:
            stream.write(original)
        if backup.read_bytes() != original:
            raise ValueError('Prior document copy differs')
        path.write_bytes(prefix + original)
        if path.read_bytes()[len(prefix):] != original:
            raise ValueError('Prior document suffix differs')
        changes.append({'before': before, 'backup': pin(backup), 'after': pin(path),
                        'prefix_bytes': len(prefix), 'prior_bytes_preserved_exactly': True})
    if any(pin(ROOT / row['path']) != row for row in consumed):
        raise ValueError('Consumed evidence changed during publication')
    receipt = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
               'status': 'PREPARED_UNRUN_MANUAL_GATE_ALIGNED_PILOT_V1',
               'changes': changes, 'consumed_before_after_exact': consumed,
               'all_historical_document_bytes_retained': True, 'VM_launched': False,
               'actual_parameter_applications': 0, 'model_qualified': False,
               'automatic_goal_advancement_triggered': False, 'full_goal': 'ACTIVE/incomplete',
               'source': pin(Path(__file__).resolve())}
    with (DEST / 'publication_receipt_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': True, 'documents': len(changes), 'manual_packet_ready': True,
                      'VM_launched': False, 'model_qualified': False}))


if __name__ == '__main__':
    main()
