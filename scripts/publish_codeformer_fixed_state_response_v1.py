"""Publish the verified unrun measurement with exact historical document suffixes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_fixed_state_response_v1_preparation'
PUB = PREP / 'publication_v1'
NAMES = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
    'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_VM.md',
    'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md',
    'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_PREPARATION.md',
    'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN_REVIEW.md')
GUIDES = ('CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_VM.md', 'CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_PREPARATION.md')
EVIDENCE = ('stage_v1.json', 'seal_v1.json', 'independent_source_closure_v1.json',
    'independent_worker_review_v1.json', 'independent_preseal_v1.json',
    'independent_sealer_source_v1.json', 'independent_sealed_transport_v1.json', 'independent_manual_guide_v1.json')
PROTOCOL = '3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089'
ARCHIVE = 'cb1fd0552665ad09a2ddfa59a1ab2ce5e9ae5746f72d2a1490a5091c1c2c1d1d'
PREFIX = '''# Frozen-response measurement sealed; manual launch pending — 17 October 2026

**CURRENT: VERIFIED PACKAGE PREPARED / UNRUN / ZERO TRAINING UPDATES.**
[Use these five manual steps](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_VM.md).
[Purpose and finite limits](CCTV_CODEFORMER_FIXED_STATE_RESPONSE_V1_PREPARATION.md).

The genuine trained CodeFormer-plus-controller composite already exists. All
519 exported tensor values match its retained stopped checkpoint, but it failed
preservation and dark retention and remains research-only. The head darkens all
406 DEV cases and already leaked unwanted correction into 98/100 recorded dark
TRAIN responses. No fixed-final fitted-response evaluation was returned.
[Actual model and failed results](CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_RETURN_REVIEW.md).

The new manual measurement freezes both saved heads and both parent restorers.
It reuses 200 fitted input/base/original caches and adds 64 matched observations
from 32 unused approved TRAIN photographs. Both heads run on all 264 observations:
528 head plus 128 new parent forwards, exactly 656 maximum, with zero gradients,
backwards, optimizers, updates or new checkpoints. The target is evidence about
fitting versus response on unused sources, not a smaller replacement for the
restoration goal. Photograph/source IDs do not prove separate human identities.
DEV/FINAL pixels remain unused by this measurement; no quality gate changes.

The upload is 4,909,256 bytes, SHA-256
`cb1fd0552665ad09a2ddfa59a1ab2ce5e9ae5746f72d2a1490a5091c1c2c1d1d`;
protocol SHA-256
`3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089`.
All 153 source/input members and physical gzip/USTAR/checksum bindings passed
independent readback. The 1,055 minimal existing parent file bindings and all
264 input joins were freshly checked locally. There are 49 source-bound CPU
passes across separate receipts. No local neural work or VM connection ran.

Require 8 GiB free after installation, preserving a 4 GiB reserve. Enforced
worker time is 900 seconds; data/init/measurement phase caps are 300/300/600
seconds within it. Preflight/export each have 300 seconds plus supervised grace.
Only the user's manual L4/tmux launch is authorized; no automatic retry, cleanup,
historical pilot or additional training is launched. Runtime outcomes remain
unknown until the returned outputs are independently audited.

The full goal stays useful 256x256 restoration of one cropped frontal/mild-turn
face, visible appearance and app design preserved, clearer/less-covered input
requests when insufficient, credited untouched CodeFormer benchmark and retained
original DGP/checkpoints/caches/splits/provenance/terms/failures. Keep native
unpaired evidence separate from paired synthetic measurements and sources;
reserved FINAL remains separate. No ethnicity, Zamboanga performance or exact
hidden-identity claim is supported. User acceptance, independent FINAL review
and app Auto/override/PNG flow with bundled inline Playwright remain required.
Separate seven-family covering detection, reviewable masks and plausible
completion stay deferred until restoration acceptance. Storage offload was
already completed and was not repeated. The overall goal remains incomplete.

Earlier current-status statements and commands below are historical. Exact
previous bytes and sealed instruction snapshots remain preserved for auditing.

---

'''


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    rows = {name: read(PREP / name) for name in EVIDENCE}
    assert all(row['complete'] is True for row in rows.values())
    seal = rows['seal_v1.json']
    assert seal['archive_sha256'] == ARCHIVE and seal['protocol_sha256'] == PROTOCOL
    assert seal['archive_bytes'] == 4909256 and seal['static_members'] == 153
    archive = ROOT / 'outputs/cctv-codeformer-fixed-state-response-v1-execution.tar.gz'
    assert sha(archive) == ARCHIVE and archive.stat().st_size == 4909256
    assert Path(str(archive) + '.sha256').read_bytes() == (ARCHIVE + '  ' + archive.name + '\n').encode('ascii')
    stage = read(PREP / 'stage_v1.json'); folder = Path(stage['stage'])
    for name, expected in seal['members_sha256'].items():
        assert sha(folder / name) == expected
        if name.startswith(('research/codeformer_', 'scripts/', 'tests/')): assert sha(ROOT / name) == expected
    assert sha(folder / 'protocol.json') == PROTOCOL
    assert seal['dependencies_freshly_verified'] == {'tone_training': 807, 'adain': 248}
    assert len(read(folder / 'fitted_cases.json')['cases']) == 200 and len(read(folder / 'heldout_pairs.json')['cases']) == 64
    prefix = PREFIX.encode('utf-8')
    if not PUB.exists():
        PUB.mkdir(); documents = {}
        for name in NAMES:
            before = (ROOT / name).read_bytes()
            snapshot = seal['members_sha256'].get('instructions/' + name)
            if snapshot: assert hashlib.sha256(before).hexdigest() == snapshot
            with (PUB / (name + '.before')).open('xb') as stream: stream.write(before)
            documents[name] = {'before_sha256': hashlib.sha256(before).hexdigest(), 'before_bytes': len(before),
                'after_sha256': hashlib.sha256(prefix + before).hexdigest(), 'preserved_suffix_offset_bytes': len(prefix),
                'frozen_instruction_snapshot_sha256': snapshot}
        with (PUB / 'prefix.txt').open('xb') as stream: stream.write(prefix)
        new_json(PUB / 'plan.json', {'documents': documents, 'prefix_sha256': sha(PUB / 'prefix.txt'),
            'evidence_sha256': {name: sha(PREP / name) for name in EVIDENCE},
            'guide_sha256': {name: sha(ROOT / name) for name in GUIDES}, 'publisher_sha256': sha(__file__)})
    plan = read(PUB / 'plan.json')
    assert plan['publisher_sha256'] == sha(__file__) and (PUB / 'prefix.txt').read_bytes() == prefix
    for name, expected in plan['evidence_sha256'].items(): assert sha(PREP / name) == expected
    for name, expected in plan['guide_sha256'].items(): assert sha(ROOT / name) == expected
    for name, row in plan['documents'].items():
        path, backup = ROOT / name, PUB / (name + '.before'); assert sha(backup) == row['before_sha256']
        if sha(path) == row['before_sha256']:
            temporary = PUB / (name + '.new')
            if not temporary.exists():
                with temporary.open('xb') as stream:
                    stream.write(prefix + backup.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            assert sha(temporary) == row['after_sha256']; os.replace(temporary, path)
        assert sha(path) == row['after_sha256'] and path.read_bytes()[len(prefix):] == backup.read_bytes()
    if not (PUB / 'receipt.json').exists():
        new_json(PUB / 'receipt.json', {**plan, 'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
            'protocol_sha256': PROTOCOL, 'archive_sha256': ARCHIVE, 'status': 'PREPARED_UNRUN',
            'measurement_launched': False, 'new_training_updates': 0, 'model_qualified': False,
            'app_promoted': False, 'overall_goal_complete': False, 'documents_updated': len(NAMES),
            'historical_bytes_preserved_exactly': True, 'local_neural_gradient_optimizer_VM_calls': 0})
    print({'complete': True, 'status': 'PREPARED_UNRUN', 'documents_updated': len(NAMES), 'new_training_updates': 0})


if __name__ == '__main__': main()
