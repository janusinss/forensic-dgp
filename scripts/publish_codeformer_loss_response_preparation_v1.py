"""Publish verified, unrun preparation while retaining exact instruction history."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_loss_response_v1 import contract as c

REVIEW = ROOT / 'outputs/codeformer_loss_response_design_v1'
DOCS = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'PROJECT_HANDOFF.md']

PREFIX = '''# CodeFormer loss response V1 sealed for manual diagnostic — 17 October 2026

**CURRENT: VERIFIED SOURCE/DATA PREPARATION / UNRUN / NO MODEL PROMOTION.**
[Five manual steps](CCTV_CODEFORMER_LOSS_RESPONSE_V1_VM.md).
[Preparation and limits](CCTV_CODEFORMER_LOSS_RESPONSE_V1_PREPARATION.md).
[Prospective measurement design](CCTV_CODEFORMER_LOSS_RESPONSE_V1_DESIGN.md).
[Previous actual AdaIN result](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md).

The source-only upload is632,930 bytes,17 regular members, SHA-256
`5644c4a16113ccba29f63b9f3008db6118f553491ff7c467a3d35b61121c0cdb`.
Protocol SHA-256 is
`23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab`.
The revised stage preserves the original unsealed attempt and every checker
failure. It freshly binds2,011 read-only parent files and16 fixed TRAIN
observations with unchanged normal/dark pairs, clean targets and noise fields.
CPU evidence:56 tests passed, three physical Windows-link checks skipped.
This establishes local source/data preparation, not Linux/GPU feasibility.
Five local full-camera reconstructions differed under newer packages; all16
must reproduce exactly on the compatible manual VM before model construction.
No tolerance or quality requirement was relaxed.

The new question is which retained objective and saved Adam state changes
normal-face colour/exposure while supplying useful dark-face restoration.
Original0 and stopped50 are separate diagnostic measurement states. Each of
their ten conditions restores the same saved model/Adam/RNG; proposals never
chain. Fixed update50 rates are a measurement control, not a resumed schedule.
Maximum425 neural forwards,193 gradient queries and18 REAL DISPOSABLE Adam
steps; ZERO committed training updates and no candidate checkpoint export.
No new recipe, sole cause, accepted model or useful native output is established.
The prior50-update AdaIN experiment remains a preservation failure: the same
three clear cases fail and normal-blur photometric guards fail. Its compound/
lowlight gains over original CodeFormer remain evidence, not qualification.

Only the user launches the new diagnostic on the existing NVIDIA L4 in tmux.
Require14 GiB free after install; keep4 GiB reserve. Worker1,800 seconds,
preflight300 and export600 seconds have finite external stop controls. RSS8 GiB,
CUDA reserved22 GiB and payload/archive4/4.5 GiB caps remain prospective.
Runtime estimate8–18 minutes excluding transfer is unmeasured. Retain any
provenance/resource/numerical failure; do not retry or extend it automatically.
No upload, VM connection, neural/gradient/optimizer call, training launch,
app change, cleanup, Drive transfer or deletion occurred during preparation.
Prior storage offload stays complete and separate; no freed bytes are counted again.

**Latest usable-input requirement remains active before generation.** Assess
visible facial relationships, native/effective resolution, blur/blocking,
exposure/clipping, coverage and supported pose from the input alone. Request
a clearer, more frontal or less-covered crop when usable information is
insufficient. Brightness or blur alone must not reject usable faces; generated
detail and enlargement do not establish input usability. A covered mouth/nose
alone does not reject sufficient visible upper-face context for later completion.
The app still has operator review/basic checks, not a calibrated automatic
structure gate. Calibrate prospectively on DEV-only input labels, with uncertain
review, preserved historical cases/refusal coverage and no FINAL tuning. A
confirmed hard-insufficiency refusal must not be bypassed by On/Off or masks.

The full goal stays active: useful, accurately credited CodeFormer-based256×256
restoration fine-tuned by us for one already cropped frontal/mildly turned face,
preserving visible appearance and the existing app design. Untouched CodeFormer
remains initializer/benchmark. Preserve original DGP/checkpoints, splits,
provenance, failures, caches and migration backups. Useful moderate brightening
is allowed with appearance/structure preserved. Keep paired synthetic evidence
separate from public native/unpaired CCTV and its source/terms/overlap limits.
No aligned clean native target, ethnicity, exact recovered identity or Zamboanga
performance claim follows. Reserved FINAL identities/pixels remain untouched.
User-visible improvement/acceptance, independent final review, preservation,
useful native outputs and CodeFormer app Auto/override/PNG flow verified through
bundled inline Playwright remain open. Diagnostic/transfer completion is insufficient.

Separate covering detection/completion improvement stays deferred until the
user accepts restoration. Later scope retains masks, sunglasses, strong lens
glare, hands, obstructing hair, scarves and objects; automatic removal-area
preview/correction; visible-pixel, clear-glasses and non-obstructing-hair
preservation; one plausible estimate, original/mask/result and PNG/optional
bundle downloads; automatic/assisted evidence separated; no exact hidden identity.
Too little visible face requires a less-covered crop.

All previous instruction bytes remain exactly below and in publication backups.
The archive's instruction snapshots are the pre-publication preserved history;
its bound protocol and current five-step guide specify this new measurement.
Earlier no-next-packet/PREPARED/UNRUN headers describe their historical stage.
The overall goal is incomplete.

---

'''


def pin(path):
    return {'sha256': c.sha256(path), 'bytes': Path(path).stat().st_size}


def exclusive(path, payload):
    c.no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def main():
    publication = REVIEW / 'publication_v1'
    c.require(not publication.exists(), 'Retain prior publication; do not overwrite')
    seal = c.read_json(REVIEW / 'seal_v1.json')
    release = c.read_json(REVIEW / 'independent_release_review_r1_v1.json')
    c.require(seal['complete'] and release['complete'], 'Closed seal and independent release review required')
    c.require(release['status'] == 'SEALED_PREPARED_UNRUN_RELEASE_REVIEW_PASS' and
              release['protocol_sha256'] == seal['protocol_sha256'], 'Current passing release identity required')
    for name, binding in release['source_pins'].items():
        c.require(pin(c.safe_path(ROOT, name)) == binding, 'Reviewed release input changed: ' + name)
    c.require(seal['archive_sha256'] == '5644c4a16113ccba29f63b9f3008db6118f553491ff7c467a3d35b61121c0cdb' and
              c.sha256(seal['archive']) == seal['archive_sha256'], 'Archive differs before publication')
    member_map = {row['path']: {'sha256': row['sha256'], 'bytes': row['bytes']} for row in seal['members']}
    for name in DOCS:
        c.require(pin(ROOT / name) == member_map[c.VM_NAME + '/instructions/' + name],
                  'Instruction changed since sealing')
    publication.mkdir()
    prior = {}
    for name in DOCS:
        data = (ROOT / name).read_bytes()
        target = publication / 'prior' / name
        exclusive(target, data)
        c.require(target.read_bytes() == data, 'Instruction backup readback differs')
        prior[name] = pin(target)
    evidence_files = sorted(path for path in REVIEW.rglob('*') if path.is_file() and
                            publication not in path.parents)
    artifacts = [ROOT / name for name in
        ('CCTV_CODEFORMER_LOSS_RESPONSE_V1_VM.md', 'CCTV_CODEFORMER_LOSS_RESPONSE_V1_DESIGN.md',
         'CCTV_CODEFORMER_LOSS_RESPONSE_V1_PREPARATION.md')]
    artifacts += sorted((ROOT / 'research/codeformer_loss_response_v1').glob('*.py'))
    artifacts += sorted((ROOT / 'tests').glob('test_codeformer_loss_response_*_v1.py'))
    artifacts += sorted((ROOT / 'scripts').glob('*codeformer_loss_response*v1.py'))
    artifacts += [Path(seal['archive']), Path(seal['checksum'])]
    chain = {'complete': True, 'scope': 'Closed local preparation evidence before instruction publication',
        'files': {path.relative_to(ROOT).as_posix(): pin(path) for path in [*evidence_files, *artifacts]},
        'prior_instruction_bytes': prior, 'protocol_sha256': seal['protocol_sha256'],
        'actual_neural_gradient_optimizer_VM_calls': 0, 'prepared_unrun': True, 'model_qualified': False}
    c.write_new_json(publication / 'evidence_chain.json', chain)
    prefix = PREFIX.encode('utf-8')
    exclusive(publication / 'prefix.md', prefix)
    changed = {}
    for name in DOCS:
        path, backup = ROOT / name, publication / 'prior' / name
        c.require(pin(path) == prior[name], 'Instruction drift after backup')
        payload = prefix + backup.read_bytes()
        path.write_bytes(payload)
        c.require(path.read_bytes() == payload and path.read_bytes()[len(prefix):] == backup.read_bytes(),
                  'Exact instruction suffix preservation failed')
        changed[name] = {'after': pin(path), 'prior': prior[name], 'prior_backup': str(backup),
                         'preserved_suffix_offset_bytes': len(prefix)}
    receipt = {'complete': True, 'status': 'PUBLISHED_PREPARED_UNRUN', 'UTC': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': seal['protocol_sha256'], 'archive_sha256': seal['archive_sha256'],
        'evidence_chain': pin(publication / 'evidence_chain.json'), 'prefix': pin(publication / 'prefix.md'),
        'instructions': changed, 'source_sha256': c.sha256(__file__),
        'actual_neural_gradient_optimizer_VM_calls': 0, 'no_app_or_training_change': True,
        'input_threshold_implemented_or_calibrated': False, 'model_qualified': False, 'overall_goal_complete': False}
    c.write_new_json(publication / 'receipt.json', receipt)
    print({'complete': True, 'instructions_updated': len(changed), 'status': receipt['status'],
           'receipt_sha256': c.sha256(publication / 'receipt.json')})


if __name__ == '__main__':
    main()
