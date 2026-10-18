"""Publish the audited return milestone while preserving all instruction history."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_return_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, payload):
    with Path(path).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    chain_path = REVIEW / 'closed_chain_v1.json'
    chain = read(chain_path)
    assert chain['complete'] and not chain['model_qualified'] and not chain['goal_complete']
    assert chain['freshly_rehashed_imported_members'] == 2231 and chain['local_neural_gradient_optimizer_VM_calls'] == 0
    for name, digest in chain['evidence_sha256'].items():
        assert sha(ROOT / name) == digest
    report = ROOT / 'CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_RETURN_REVIEW.md'
    assert sha(report) == chain['report_sha256']
    old = read(ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_preparation/publication_v1/receipt.json')
    before = {}
    for row in old['documents']:
        path = ROOT / row['path']
        assert sha(path) == row['after_sha256']
        before[row['path']] = path.read_bytes()
    assert len(before) == 4
    prefix = '''# Bounded-fusion CodeFormer return audited — 17 October 2026

**CURRENT: MEASUREMENT COMPLETE / AUDITED; FIXED BRIGHTNESS GATE REJECTED.**
All2,231 returned files (761,506,959 payload bytes) and the original293-file
packet passed fresh hash checks. The manual L4 run completed170 forwards and
960 fusion evaluations in178.910 supervised seconds, with zero optimizer or
gradient updates. Root reviewed all18 sheets /32 TRAIN observations and eight
individual256 canary outputs. This is not a newly trained or accepted model.

[Actual return review](CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_RETURN_REVIEW.md).
[All comparisons](outputs/codeformer_bounded_fusion_feasibility_v1_return_review/visual_sheets/INDEX.md).
[Closed evidence chain](outputs/codeformer_bounded_fusion_feasibility_v1_return_review/closed_chain_v1.json).

The fixed brightness gate lowers normal tone error but worsens dark PNG MAE
on13/16 cases versus the constant bound. Both bounds also sacrifice some useful
stopped-model dark improvement; severe facial-detail errors remain. Reject the
unchanged fixed gate and keep its evidence. No threshold/beta grid, unchanged
epochs, app adoption or model promotion follows. The full61-clear/53-blur DEV
guards were not evaluated in this32-case TRAIN measurement and remain unchanged.

The next controlled hypothesis is AdaIN image-gradient isolation: detach only
the encoder style argument during candidate training, leaving fixed-state
forward values and CE/HQ/token/fusion learning intact. Its prospective manual
VM witness must prove the intended route before Adam. The planned fresh run
starts from untouched CodeFormer and permits at most50 updates; it does not
resume failed weights. Source/packet preparation is in review and unrun here.
[Prospective design](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md).
No new sealed upload/launch guide is established by this milestone header.

Input-policy correction: all200 retained PNG/255 float32 hashes match the
original canonical TRAIN input pixels. The camera renderer already quantizes
to RGB8 before normalization. Earlier continuous-input wording is a retained
documentation error, not a numerical input change or an explanation of the
quality stops. Exact regeneration, records and the full200 schedule remain.

Keep original pretrained CodeFormer as initializer/benchmark and retain all
original DGP/CodeFormer checkpoints, research caches, provenance, splits and
failure states. All actual training remains human-launched on the L4 in tmux
with verified transfers and finite resource/quality stops. No local NN/gradient/
optimizer, VM connection, cleanup, Drive offload or app change occurred during
this return audit. Earlier verified local offload remains complete.

Restoration remains first: preserve visible appearance and useful moderate
brightening, request clearer/more frontal/less-covered input when information
is insufficient, and preserve the app design. Native sources stay separate
and unpaired; no reserved FINAL pixels, ethnicity, recovered-identity or local
Zamboanga claim. User acceptance, independent final review and full local
Auto/override/PNG verification remain outstanding. Detector/completion model
development stays deferred until restoration acceptance. The full goal remains
active and incomplete; all earlier instruction bytes below remain history.

---

'''.encode('utf-8')
    out = REVIEW / 'publication_v1'
    out.mkdir(exist_ok=False)
    (out / 'prior').mkdir()
    stages = {}
    for name, payload in before.items():
        assert (ROOT / name).read_bytes() == payload
        put(out / 'prior' / name, payload)
        staged = ROOT / (name + '.bounded-return-stage')
        put(staged, prefix + payload)
        assert staged.read_bytes() == prefix + payload
        stages[name] = staged
    rows = []
    for name, staged in stages.items():
        path = ROOT / name
        assert path.read_bytes() == before[name]
        os.replace(staged, path)
        assert path.read_bytes()[len(prefix):] == before[name] == (out / 'prior' / name).read_bytes()
        rows.append({'path': name, 'before_sha256': sha(out / 'prior' / name), 'after_sha256': sha(path),
            'prior_bytes': len(before[name]), 'retained_suffix_offset': len(prefix),
            'backup': str((out / 'prior' / name).relative_to(ROOT)), 'prior_bytes_retained_exactly': True})
    receipt = {'complete': True, 'documents': rows, 'closed_chain_sha256': sha(chain_path),
        'report_sha256': sha(report), 'source_sha256': sha(__file__),
        'status': 'ACTUAL RETURN AUDITED; NEW TRAINING SOURCE UNDER REVIEW / UNRUN',
        'model_qualified': False, 'goal_complete': False, 'local_neural_gradient_optimizer_VM_calls': 0}
    put(out / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'complete': True, 'documents': len(rows), 'model_qualified': False, 'new_training_unrun': True}))


if __name__ == '__main__':
    main()
