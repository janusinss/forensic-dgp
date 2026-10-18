"""Publish the sealed, still-unrun pilot without changing historical evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_tone_control_training_v1_preparation'
PUB = PREP / 'publication_v1'
STAGE = ROOT / 'outputs/cctv_codeformer_tone_control_training_v1_vm'
PROTOCOL = 'f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1'
ARCHIVE_SHA = 'a5bdd3dbaa5bf8376b04e2e38a3f1beddfef7367aa3ab4318527db6660ae7e79'
NAMES = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
         'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
         'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md',
         'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md',
         'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md',
         'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_PREPARATION.md')
GUIDES = ('CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_VM.md',
          'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_PREPARATION.md')
EVIDENCE = ('stage_v1.json', 'seal_v1.json', 'data_CPU_tests_v6.json',
            'independent_worker_review_v1.json', 'independent_preseal_source_v1.json',
            'independent_sealer_source_review_v1.json', 'independent_sealed_transport_v1.json',
            'independent_manual_guide_v1.json', 'resource_ledger_v3.json')
PREFIX = '''# Next CodeFormer tone-control pilot sealed; manual launch pending — 17 October 2026

**CURRENT: SOURCE PACKAGE VERIFIED / TRAINING PREPARED / UNRUN.**
[Use these five manual steps](CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_VM.md).
[Preparation and acceptance limits](CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_PREPARATION.md).

The downloaded normal-tone run already created a genuine trained CodeFormer
model, independently exported and verified. It failed clear/normal-blur
preservation and remains research-only. Do not repeat its old launch commands,
add epochs to that recipe, resume its optimizer or adopt it in the app.
[Actual model and failed results](CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_RETURN_REVIEW.md).

The new finite pilot freezes the earlier full-coarse fine-tuned CodeFormer and
trains an 8,257-parameter input-only tone controller for at most 50 Adam updates.
It tests normal-face preservation while retaining useful dark-face brightening;
it does not establish new facial detail or structural recovery. Clean-reference
oracle capacity is not an input-only method or successful learned restoration.
All existing RAW/PNG preservation limits remain, with paired dark-retention and
known TRAIN stress checks. Native CCTV remains unpaired and FINAL stays unused.

The source-only upload is 1,040,383 bytes, SHA-256
`a5bdd3dbaa5bf8376b04e2e38a3f1beddfef7367aa3ab4318527db6660ae7e79`;
protocol SHA-256
`f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1`.
All 24 archive members, full gzip/TAR structure, portable checksum and source
bindings passed independent readback. Locally, all 3,635 existing parent file
bindings and the 200 TRAIN / 406 DEV input index were freshly verified. VM
availability remains subject to preflight. Require 16 GiB free after install;
preserve a 4 GiB reserve. Only the human's manual finite L4/tmux run is authorized.
No local neural work or automatic VM launch was performed during preparation.

The full goal remains useful 256x256 restoration of one frontal/mild-turn crop,
preserved visible appearance and app design, clearer-input refusal when needed,
credited untouched CodeFormer benchmark, retained original DGP checkpoints and
research assets, separate sources/terms/splits and paired versus unpaired claims.
User acceptance, independent final review and app Auto/override/PNG verification
with bundled inline Playwright remain required. Seven-family covering detection,
mask review and plausible completion remain deferred until restoration acceptance.
No ethnicity, Zamboanga performance or exact hidden-identity claim is supported.
Storage offload was already complete and was not repeated. The goal is incomplete.

All earlier current-status statements and commands below are historical. Exact
pre-publication bytes and frozen instruction snapshots are retained for auditing.

---

'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    evidence = {name: read(PREP / name) for name in EVIDENCE}
    assert all(row['complete'] is True for row in evidence.values())
    seal = evidence['seal_v1.json']
    transport = evidence['independent_sealed_transport_v1.json']
    assert seal['protocol_sha256'] == transport['protocol_sha256'] == PROTOCOL
    assert seal['archive_sha256'] == transport['archive_sha256'] == ARCHIVE_SHA
    assert seal['archive_bytes'] == transport['archive_bytes'] == 1040383
    archive = ROOT / 'outputs/cctv-codeformer-tone-control-training-v1-execution.tar.gz'
    assert sha(archive) == ARCHIVE_SHA and archive.stat().st_size == 1040383
    assert Path(str(archive) + '.sha256').read_bytes() == (ARCHIVE_SHA + '  ' + archive.name + '\n').encode('ascii')
    assert sha(STAGE / 'protocol.json') == PROTOCOL and len(seal['members_sha256']) == 24
    for name, expected in seal['members_sha256'].items():
        assert sha(STAGE / name) == expected
        if name.startswith(('research/codeformer_', 'scripts/', 'tests/')):
            assert sha(ROOT / name) == expected
    index = seal['all_TRAIN_DEV_bindings_checked']
    assert index['TRAIN_cases'] == 200 and index['DEV_cases'] == 406 and index['DEV_native_unpaired'] == 48
    assert seal['dependencies_freshly_verified'] == 2011 and seal['retained_outputs_freshly_verified'] == 1624
    for name in GUIDES:
        assert (ROOT / name).is_file()
    prefix = PREFIX.encode('utf-8')
    if not PUB.exists():
        # Every historical byte is backed up before the first document mutation.
        PUB.mkdir()
        documents = {}
        for name in NAMES:
            before = (ROOT / name).read_bytes()
            expected_snapshot = seal['members_sha256'].get('instructions/' + name)
            if expected_snapshot:
                assert hashlib.sha256(before).hexdigest() == expected_snapshot
            with (PUB / (name + '.before')).open('xb') as stream:
                stream.write(before)
            documents[name] = {'before_sha256': hashlib.sha256(before).hexdigest(), 'before_bytes': len(before),
                'after_sha256': hashlib.sha256(prefix + before).hexdigest(), 'preserved_suffix_offset_bytes': len(prefix),
                'frozen_instruction_snapshot_sha256': expected_snapshot}
        with (PUB / 'prefix.txt').open('xb') as stream:
            stream.write(prefix)
        new_json(PUB / 'plan.json', {'documents': documents, 'prefix_sha256': sha(PUB / 'prefix.txt'),
            'evidence_sha256': {name: sha(PREP / name) for name in EVIDENCE},
            'guide_sha256': {name: sha(ROOT / name) for name in GUIDES}, 'publisher_sha256': sha(__file__)})
    plan = read(PUB / 'plan.json')
    assert plan['publisher_sha256'] == sha(__file__) and (PUB / 'prefix.txt').read_bytes() == prefix
    for name, expected in plan['evidence_sha256'].items():
        assert sha(PREP / name) == expected
    for name, expected in plan['guide_sha256'].items():
        assert sha(ROOT / name) == expected
    for name, row in plan['documents'].items():
        path, backup = ROOT / name, PUB / (name + '.before')
        assert sha(backup) == row['before_sha256']
        if sha(path) == row['before_sha256']:
            temporary = PUB / (name + '.new')
            if not temporary.exists():
                with temporary.open('xb') as stream:
                    stream.write(prefix + backup.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            assert sha(temporary) == row['after_sha256']
            os.replace(temporary, path)
        assert sha(path) == row['after_sha256']
        assert path.read_bytes()[len(prefix):] == backup.read_bytes()
    if not (PUB / 'receipt.json').exists():
        new_json(PUB / 'receipt.json', {**plan, 'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
            'archive_sha256': ARCHIVE_SHA, 'archive_bytes': 1040383, 'protocol_sha256': PROTOCOL,
            'status': 'PREPARED_UNRUN', 'training_executed': False, 'new_controller_trained': False,
            'restoration_qualified': False, 'app_promoted': False, 'overall_goal_complete': False,
            'documents_updated': len(NAMES), 'historical_bytes_preserved_exactly': True,
            'local_neural_gradient_optimizer_VM_calls': 0})
    print({'complete': True, 'status': 'PREPARED_UNRUN', 'documents_updated': len(NAMES), 'model_qualified': False})


if __name__ == '__main__':
    main()
