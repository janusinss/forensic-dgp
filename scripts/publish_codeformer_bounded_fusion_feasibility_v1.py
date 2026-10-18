"""Publish a verified manual measurement packet, retaining prior goal history."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_preparation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    archive = read(PREP / 'execution_archive_v1.json')
    audit = read(PREP / 'independent_archive_readback_v1.json')
    commands = read(PREP / 'independent_commands_readback_v1.json')
    assert archive['complete'] and audit['complete'] and commands['complete'] and commands['approved']
    assert archive['members'] == 293 and archive['bytes'] == 43604878 and archive['optimizer_updates'] == 0
    assert archive['protocol_sha256'] == 'c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb'
    assert archive['archive_sha256'] == '9e355da89976885ad4d9935b483a1df8f449c916fe9dd0d480973ffa58af77d6'
    assert sha(ROOT / archive['archive']) == archive['archive_sha256']
    for name, digest in commands['reviewed_files_sha256'].items():
        assert sha(ROOT / name) == digest
    prefix = '''# Bounded-fusion CodeFormer measurement ready — 16 October 2026

**CURRENT: VERIFIED / PREPARED / UNRUN; ZERO OPTIMIZER UPDATES.**
The new packet tests input-conditioned, bounded fusion on 32 saved TRAIN
observations before choosing another restoration training recipe. It measures
170 full model forwards and 960 fusion-block evaluations, with zero gradient,
backward, teacher or optimizer calls. No new model has been trained or promoted.

[Five exact manual upload/tmux/launch/download steps](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_VM.md).
[Preparation and finite limits](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_PREPARATION.md).
[Previous actual FP32 training review](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md).

The previous FP32 comparison completed 50 actual Adam updates and improves
many paired blurred/dark cases, but keeps the same clear/normal-blur failures
as V2. Precision alignment alone was insufficient. Preserve useful moderate
brightening while protecting visible structure, tone and contrast. Do not
resume the stopped recipe unchanged or append epochs past its quality stop.

The fixed gate is an observed-input luminance probe, not an exposure detector.
Its normal/dark ambiguity and a learned gate's initial zero-delta limitation
justify this finite no-update test. Compare all five paths and overlap stress
canaries; input/clean targets, original/stopped tensors and failures are bound.
A positive measurement cannot qualify a model or automatically start training.

Upload: 43,604,878 bytes; all 293 members independently hash-verified. The
existing FP32 original/stopped weights remain read-only external dependencies.
44 CPU fixtures and independent source/data/resource/numerical reviews passed.
Require 8 GiB free after installation; preserve 4 GiB. Runtime caps are
300 seconds initialization, 600 measurement, 900 whole worker and 300 export,
with finite supervisor grace. Actual new-route runtime remains unmeasured.

No actual neural/gradient/training execution, VM connection, cleanup or Drive
offload occurred during preparation. Earlier storage offload remains complete.
All model execution stays human-launched on the existing L4 in tmux. Keep
original DGP/CodeFormer checkpoints, research assets, splits and gate failures.

Native ChokePoint/QMUL DEV evidence remains separate and unpaired; no reserved
FINAL pixels, ethnicity, recovered identity or Zamboanga claim. Request clearer
or less-covered crops when usable structure is insufficient. User restoration
acceptance, independent final review, primary app integration and full bundled
inline Playwright Auto/override/download checks remain outstanding. Completion/
detector development is deferred until restoration acceptance. The full goal
remains active; earlier statuses below are unchanged historical records.

---

'''.encode('utf-8')
    out = PREP / 'publication_v1'
    out.mkdir(exist_ok=False); (out / 'prior').mkdir()
    rows = []
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        path = ROOT / name
        before = path.read_bytes()
        assert not before.startswith(prefix)
        backup = out / 'prior' / name
        with backup.open('xb') as stream:
            stream.write(before); stream.flush(); os.fsync(stream.fileno())
        staged = path.with_name(name + '.bounded-feasibility-stage')
        with staged.open('xb') as stream:
            stream.write(prefix + before); stream.flush(); os.fsync(stream.fileno())
        assert staged.read_bytes() == prefix + before and path.read_bytes() == before
        os.replace(staged, path)
        assert path.read_bytes()[len(prefix):] == backup.read_bytes() == before
        rows.append({'path': name, 'before_sha256': sha(backup), 'after_sha256': sha(path),
            'prior_bytes': len(before), 'retained_suffix_offset': len(prefix),
            'backup': str(backup.relative_to(ROOT)), 'prior_bytes_retained_exactly': True})
    receipt = {'complete': True, 'status': 'VERIFIED / PREPARED / UNRUN',
        'documents': rows, 'reports_sha256': commands['reviewed_files_sha256'],
        'archive_sha256': archive['archive_sha256'], 'protocol_sha256': archive['protocol_sha256'],
        'evidence_sha256': {name: sha(PREP / name) for name in ('execution_archive_v1.json',
            'independent_archive_readback_v1.json', 'independent_commands_readback_v1.json',
            'final_review_approval_v1.json', 'independent_data_resource_v1.json', 'numerical_policy_v1.json')},
        'neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False, 'goal_complete': False,
        'source_sha256': sha(__file__)}
    with (out / 'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'complete': True, 'documents': len(rows), 'manual_measurement_unrun': True}))


if __name__ == '__main__':
    main()
