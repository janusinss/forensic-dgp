"""Publish the audited return status, retaining every previous document byte."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_path_diagnostic_v1_return_review'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    chain = load(REVIEW / 'closed_chain_v1.json')
    assert chain['complete'] and chain['visual_conditions_reviewed'] == 104
    assert chain['optimizer_updates_in_diagnostic'] == 0 and not chain['model_qualified']
    reports = ['CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_RETURN_REVIEW.md',
               'CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md']
    independent = []
    for filename in ('independent_report_design_readback_pixels_v3.json',
                     'independent_report_design_readback_state_v1.json'):
        path = REVIEW / filename
        row = load(path)
        assert row['complete'] and row['approved']
        for name in reports:
            assert row['reviewed_files_sha256'][name] == digest((ROOT / name).read_bytes())
        independent.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path.read_bytes())})
    out = REVIEW / 'publication_v1'
    out.mkdir(exist_ok=False)
    (out / 'prior').mkdir()
    top = '''# CodeFormer path diagnostic V1 audited — 16 October 2026

**CURRENT: MEASUREMENT COMPLETE; ZERO OPTIMIZER UPDATES; NO MODEL PROMOTION.**
The actual download and all 1,243 returned files passed independent hash,
state, scalar and saved-pixel checks. Root reviewed all eight sheets covering
104 conditions plus eight native256 PNGs. The supervised measurement took
145.792 seconds. The original and stopped checkpoints remain unchanged.

[Audited results and limitations](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_RETURN_REVIEW.md).
[All saved visual comparisons](outputs/codeformer_path_diagnostic_v1_return_review/visual_sheets/INDEX.md).
[Prospective FP32 training comparison](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md).

FP32 inference and FP32 autograd/training match exactly on the eight measured
TRAIN observations at both states. BF16 changes 5–41 of 256 selected codes
per observation. This supports an inference-matched FP32 training comparison;
it does not prove the cause of every V2 regression. Returning original fusion
removes helpful exposure and worsens all four dark TRAIN cases, so blanket
fusion rollback is unsupported. Moderate useful brightening remains accepted;
preserve clear features, contrast and visible appearance.

The next design is a NEW 50-update comparison from untouched CodeFormer,
retaining V2's first 200 observations, Adam, first50 learning rates, losses,
trainable groups, comparison images and quality guards. The source600 schedule
horizon and finite50 execution ceiling are separate. It is DESIGN ONLY: no
worker, protocol, transfer archive or launch commands have been published for
that next experiment. Do not run the old diagnostic or V2 training again.

V1 u200 and V2 u50 remain quality-stopped, with their original states and gate
failures retained. The diagnostic does not create an improved trained model.
It uses four photographic TRAIN references; no native CCTV or reserved FINAL
qualification follows. All new training remains human-launched on L4 in tmux,
with verified transfers, finite timing/resource/stop rules and return review.

The earlier local storage offload remains complete; no cleanup or further
Drive transfer occurred in this audit. User restoration acceptance, independent
native/final review, integration and full app verification remain outstanding.
The full thesis goal remains active. Further detector/completion development
stays deferred until accepted restoration. Earlier statuses below are retained
unchanged history and superseded by this current audited-return status.

---

'''
    historical = '''# This diagnostic has returned — 16 October 2026

**COMPLETED AND AUDITED. Do not relaunch these historical commands.**
The diagnostic performed zero optimizer updates and created no new model.
[Current results](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_RETURN_REVIEW.md) and
[next design](CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md) supersede the old
PREPARED/UNRUN status below. The next design has no runnable packet yet.
All previous preparation/guide bytes are retained unchanged as history.

---

'''
    targets = {name: top for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                                     'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')}
    targets.update({name: historical for name in ('CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_VM.md',
                                                 'CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_PREPARATION.md')})
    updates = []
    for name, prefix in targets.items():
        path = ROOT / name
        before = path.read_bytes()
        assert not before.startswith(prefix.encode('utf-8'))
        backup = out / 'prior' / name
        backup.write_bytes(before)
        new = prefix.encode('utf-8') + before
        stage = ROOT / (name + '.path-return-staged')
        with stage.open('xb') as stream:
            stream.write(new); stream.flush(); os.fsync(stream.fileno())
        assert stage.read_bytes() == new and path.read_bytes() == before
        os.replace(stage, path)
        after = path.read_bytes()
        assert after[len(prefix.encode('utf-8')):] == before
        updates.append({'path': name, 'before_sha256': digest(before), 'after_sha256': digest(after),
                        'prior_bytes': len(before), 'retained_suffix_offset': len(prefix.encode('utf-8')),
                        'prior_bytes_retained_exactly': True, 'backup': str(backup.relative_to(ROOT))})
    result = {'complete': True, 'date': '2026-10-16', 'closed_chain_sha256': digest((REVIEW / 'closed_chain_v1.json').read_bytes()),
              'reports_sha256': {name: digest((ROOT / name).read_bytes()) for name in reports},
              'independent_reviews': independent, 'documents': updates,
              'model_qualified': False, 'new_training_packet_prepared': False,
              'goal_complete': False, 'local_neural_gradient_optimizer_VM_calls': 0,
              'publisher_sha256': digest(Path(__file__).read_bytes())}
    (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'documents_updated': len(updates), 'new_packet': False}))


if __name__ == '__main__':
    main()
