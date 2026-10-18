"""Publish verified FP32 return; preserve exact earlier workflow bytes."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_fp32_comparison_v1_return_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    chain = read(REVIEW / 'closed_chain_v1.json')
    assert chain['complete'] and chain['actual_updates'] == 50 and chain['visual_pages_root_reviewed'] == 52
    assert not chain['clear_guard_pass'] and not chain['normal_blur_photometric_guard_pass']
    assert not chain['model_qualified'] and not chain['goal_complete']
    for name, digest in chain['reports_sha256'].items():
        assert sha(ROOT / name) == digest
    for name, digest in chain['receipts_sha256'].items():
        assert sha(REVIEW / name) == digest
    out = REVIEW / 'publication_v1'
    out.mkdir(exist_ok=False)
    (out / 'prior').mkdir()
    current = '''# FP32 CodeFormer comparison returned and audited — 16 October 2026

**CURRENT: 50 ADAM UPDATES VERIFIED; QUALITY STOP RETAINED; NO MODEL PROMOTION.**
All 6,501 returned members and original packet bindings passed independent
hash checks. All 350 trainable tensors changed; all 165 frozen tensors stayed
exact. Root reviewed all 52 sheets covering 406 cases plus the three failed
clear cases at actual 256 pixels. The full thesis goal remains active.

[Actual return review](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md).
[All visual comparisons](outputs/codeformer_fp32_comparison_v1_return_review/visual_sheets/INDEX.md).
[Next design review](CCTV_CODEFORMER_POST_FP32_DESIGN_REVIEW_V1.md).

FP32 resolves the initial numerical mismatch but still fails the same three
clear controls and the normal-blur RAW/PNG photometric guards. The issue is
a quality tradeoff after real training, not a transfer or optimizer crash.
Paired blur/lowlight/compound SSIM improves on 50/53, 51/53 and 52/53 cases
against original CodeFormer; differences from BF16 V2 are small and mixed.
Moderate helpful brightening stays accepted; preserve normal-face appearance,
contrast and visible structure. A passing audit or export is not acceptance.

The user selected the best supported next approach after the three-attempt
design discussion. Review input-conditioned, bounded exposure/fusion adaptation
with delivered-256 preservation accounted for. Gate learnability, input-only
conditioning and the exact resource ledger still need resolution. This is
DESIGN REVIEW ONLY: no next worker, packet or launch command is published.
Do not resume the stopped FP32/V2 recipe or append unchanged epochs.

Untouched CodeFormer remains the initializer/benchmark; original DGP and all
failed states, checkpoints, inputs, splits and gates remain retained. Actual
training stays human-launched on the L4 in tmux, with verified transfers and
finite limits. No VM connection, training, cleanup, app adoption or new Drive
offload occurred during this audit. Earlier local offload remains complete.

All 24 ChokePoint and 24 QMUL-SurvFace crops remain separate unpaired DEV
evidence; no aligned clean native targets, ethnicity, recovered-identity or
Zamboanga claim. Several remain insufficient: request clearer/more frontal
or less-covered inputs. Reserved FINAL pixels remain untouched. User
restoration acceptance, independent final review, integration and full app
Auto/override/download verification remain outstanding. Detector/completion
model development stays deferred until restoration acceptance. Earlier
PREPARED/UNRUN text below is retained history, superseded by this return status.

---

'''
    history = '''# Historical FP32 comparison commands — returned 16 October 2026

**COMPLETED AND AUDITED AT 50 UPDATES; QUALITY STOP RETAINED. Do not relaunch.**
[Actual results](CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md) supersede
the PREPARED/UNRUN preparation below. All earlier bytes are retained as history.
The next [design review](CCTV_CODEFORMER_POST_FP32_DESIGN_REVIEW_V1.md) has no
prepared executable/transfer or next launch command. No unchanged resume.

---

'''
    targets = {name: current for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')}
    targets.update({name: history for name in ('CCTV_CODEFORMER_FP32_COMPARISON_V1_VM.md',
        'CCTV_CODEFORMER_FP32_COMPARISON_V1_PREPARATION.md')})
    rows = []
    for name, prefix in targets.items():
        path = ROOT / name
        before = path.read_bytes(); prefix = prefix.encode('utf-8')
        assert not before.startswith(prefix)
        backup = out / 'prior' / name
        with backup.open('xb') as stream:
            stream.write(before); stream.flush(); os.fsync(stream.fileno())
        after = prefix + before
        staged = path.with_name(name + '.FP32-return-publication-stage')
        with staged.open('xb') as stream:
            stream.write(after); stream.flush(); os.fsync(stream.fileno())
        assert staged.read_bytes() == after and path.read_bytes() == before
        os.replace(staged, path)
        assert path.read_bytes()[len(prefix):] == backup.read_bytes() == before
        rows.append({'path': name, 'before_sha256': sha(backup), 'after_sha256': sha(path),
            'prior_bytes': len(before), 'retained_suffix_offset': len(prefix),
            'backup': str(backup.relative_to(ROOT)), 'prior_bytes_retained_exactly': True})
    receipt = {'complete': True, 'date': '2026-10-16', 'documents': rows,
        'closed_chain_sha256': sha(REVIEW / 'closed_chain_v1.json'), 'reports_sha256': chain['reports_sha256'],
        'actual_updates': 50, 'quality_stop_retained': True, 'new_training_packet_prepared': False,
        'goal_complete': False, 'model_qualified': False, 'local_neural_gradient_optimizer_VM_calls': 0,
        'source_sha256': sha(__file__)}
    with (out / 'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps({'complete': True, 'documents_updated': len(rows), 'prior_bytes_retained': True}))


if __name__ == '__main__':
    main()
