"""Publish audited actual50 state without changing any historical bytes or model."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_normal_tone_training_v1_return_review'
PUBLICATION = REVIEW / 'publication_v1'
NAMES = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
         'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md',
         'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md',
         'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_PREPARATION.md')
PREFIX = '''# Actual CodeFormer model exported; preservation failed — 17 October 2026

**CURRENT: 50 TRAINING UPDATES VERIFIED / MODEL CREATED / QUALITY NOT ACCEPTED.**
[Return review](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_RETURN_REVIEW.md).
[Trained candidate](outputs/codeformer_normal_tone_training_v1_candidate_unqualified/README.md).
[All 34 comparison sheets](outputs/codeformer_normal_tone_training_v1_return_review/pixels_v1/visual_sheets/INDEX.md).

The manual L4 run committed 50 Adam updates and 200 microbatches. All 515 exported
model tensors exactly match the retained trained update-50 checkpoint; all 165
frozen tensors remain exact. The new .pth is 376,609,465 bytes, SHA-256
`5f2460d332764421fc998c00e0dda1e0316545e7e3fcd484fcc57773b0212d7a`.
It is our finite trained adaptation of CodeFormer, not a from-scratch architecture.
Its params_ema container key is compatibility only; EMA was not used.

All 6,304 return members, four companions, checkpoints, 812 new RAW/PNG pairs,
historical controls and state/optimizer/sampler records passed independent
integrity readback. Root personally reviewed all 34 sheets / 406 cases / 2,030
exact256 cells. Integrity and model export do not establish restoration quality.
Clear preservation FAILED: 60/61 cases exceed the unchanged 0.03 limit, versus
three previously. All 53 normal-blur cases fail both tone metrics in RAW/PNG.
Dark paired profiles improve, but clear/light/blur/JPEG regress and severe native
block-like crops still produce missing or invented features. Native is unpaired;
all clean-reference metrics remain null. Sources and final identities stay separate.

Reject normal-only luminance-centering for this recipe. Retain the genuine model,
all original states, split/terms/provenance/caches and failures. Do not add epochs,
resume this optimizer, repeat its guide unchanged, weaken gates or switch the app.
Normal TRAIN tone drift is supported by saved evidence, not a transfer failure.

A bounded CPU clean-reference oracle found scalar tone-correction capacity on
saved256 outputs for both parents. It is not an input-only method, a trained
controller, preserved dark benefit or structural recovery. Clipping and contrast
changes remain material. The next design review separates input-conditioned tone
control from the retained full-coarse restoration path; no next runnable packet
is sealed and no controller has been trained. New actual training remains the
user's manual finite L4/tmux workflow after verified source/data/transfer review.

The full goal stays restoration first: useful 256x256 for one cropped frontal or
mild-turn face, visible appearance and existing design preserved, clearer-input
requests when information is insufficient, user acceptance, independent final
review and full app Auto/override/PNG verification through bundled inline
Playwright. Keep untouched CodeFormer as credited benchmark/initializer and all
original DGP/research evidence. No ethnicity, hidden-identity or Zamboanga claim.
Later separate seven-family detector/mask-review/completion/download scope stays
deferred until restoration acceptance. Storage offload is already complete and
was not repeated. This model is research-only; the overall goal is incomplete.

All earlier preparation status and launch commands below are historical.

---

'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2); stream.write('\n')


def main():
    evidence_names = ('return_import_v1.json', 'independent_transport_v1.json',
        'state_v1/state_audit_v1.json', 'pixels_v1/audit.json', 'root_visual_review_v1.json',
        'independent_candidate_export_v1.json', 'state_v1/source_and_outcome_review_v1.json',
        'pixels_v1/tone_feasibility_v1/audit.json',
        'pixels_v1/tone_feasibility_v1/independent_scalar_readback_v1.json',
        'pixels_v1/tone_feasibility_v1/closure_v1.json')
    evidence = {name: read(REVIEW / name) for name in evidence_names}
    assert all(row['complete'] is True for row in evidence.values())
    archive = evidence['return_import_v1.json']['archive_sha256']
    protocol = evidence['return_import_v1.json']['protocol_sha256']
    for name in ('independent_transport_v1.json', 'state_v1/state_audit_v1.json',
                 'pixels_v1/audit.json', 'root_visual_review_v1.json', 'independent_candidate_export_v1.json'):
        assert evidence[name]['archive_sha256'] == archive and evidence[name]['protocol_sha256'] == protocol
    candidate = ROOT / 'outputs/codeformer_normal_tone_training_v1_candidate_unqualified'
    provenance = read(candidate / 'provenance.json')
    assert provenance['committed_training_updates'] == 50 and provenance['model_qualified'] is False
    assert provenance['app_promoted'] is False and sha(candidate / 'codeformer_u0050_unqualified.pth') == provenance['exported_sha256']
    assert evidence['independent_candidate_export_v1.json']['provenance_sha256'] == sha(candidate / 'provenance.json')
    assert evidence['pixels_v1/audit.json']['clear_guard_pass'] is False
    assert evidence['pixels_v1/audit.json']['blur_guard_pass'] is False
    report = ROOT / 'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_RETURN_REVIEW.md'
    prefix = PREFIX.encode('utf-8')
    if not PUBLICATION.exists():
        PUBLICATION.mkdir()
        files = {}
        for name in NAMES:
            before = (ROOT / name).read_bytes()
            backup = PUBLICATION / (name + '.before')
            with backup.open('xb') as stream:
                stream.write(before)
            files[name] = {'before_sha256': sha(backup), 'before_bytes': len(before),
                          'after_sha256': hashlib.sha256(prefix + before).hexdigest(),
                          'preserved_suffix_offset_bytes': len(prefix)}
        with (PUBLICATION / 'prefix.txt').open('xb') as stream:
            stream.write(prefix)
        new_json(PUBLICATION / 'plan.json', {'documents': files, 'prefix_sha256': sha(PUBLICATION / 'prefix.txt'),
            'report_sha256': sha(report), 'evidence_sha256': {name: sha(REVIEW / name) for name in evidence_names},
            'candidate_provenance_sha256': sha(candidate / 'provenance.json'), 'publisher_sha256': sha(__file__)})
    plan = read(PUBLICATION / 'plan.json')
    assert plan['publisher_sha256'] == sha(__file__) and plan['report_sha256'] == sha(report)
    assert plan['prefix_sha256'] == sha(PUBLICATION / 'prefix.txt')
    for name, expected in plan['evidence_sha256'].items():
        assert sha(REVIEW / name) == expected
    assert (PUBLICATION / 'prefix.txt').read_bytes() == prefix
    for name, row in plan['documents'].items():
        path = ROOT / name
        backup = PUBLICATION / (name + '.before')
        assert sha(backup) == row['before_sha256']
        if sha(path) == row['before_sha256']:
            temporary = PUBLICATION / (name + '.new')
            if temporary.exists():
                assert sha(temporary) == row['after_sha256']
            else:
                with temporary.open('xb') as stream:
                    stream.write(prefix + backup.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        assert sha(path) == row['after_sha256']
        assert path.read_bytes()[row['preserved_suffix_offset_bytes']:] == backup.read_bytes()
    receipt = PUBLICATION / 'receipt.json'
    if not receipt.exists():
        new_json(receipt, {**plan, 'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
            'archive_sha256': archive, 'protocol_sha256': protocol,
            'status': 'ACTUAL50_MODEL_EXPORTED_FAILED_PRESERVATION',
            'model_created': True, 'model_qualified': False, 'app_promoted': False,
            'overall_goal_complete': False, 'local_neural_gradient_optimizer_VM_calls': 0,
            'documents_updated': len(NAMES), 'historical_bytes_preserved_exactly': True})
    print({'complete': True, 'model_created': True, 'model_qualified': False, 'documents_updated': len(NAMES)})


if __name__ == '__main__':
    main()
