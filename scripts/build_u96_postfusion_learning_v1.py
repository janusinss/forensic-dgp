"""Seal a reviewed manual pilot only after explicit new input admission.

This builder copies inert source/assets and hashes bytes. It never imports Torch
or invokes any VM, worker, model, optimizer, gradient or training command.
"""
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'research/codeformer_u96_postfusion_learning_v1'
DATA = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_preparation_r1'
REVIEWS = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_design_review'
STAGE = ROOT / 'outputs/cctv_codeformer_u96_postfusion_learning_v1_vm'
PREFIX = 'cctv-codeformer-u96-postfusion-learning-v1'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def pin(path, root=ROOT):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def doc(path):
    return json.loads(path.read_bytes())


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert 'torch' not in sys.modules
    assert not STAGE.exists(), 'Earlier stage is immutable'
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    assert not archive.exists() and not checksum.exists()
    assert shutil.disk_usage(ROOT).free >= 6 * 1024**3
    # Every review/admission is required before the first stage mutation.
    source_reviews = []
    for name in ('packet_helper_review_v1_r2.json', 'worker_source_review_v1.json'):
        path = REVIEWS / name
        review = doc(path)
        assert review['complete'] and review['source_review_pass'], 'Unresolved source review: ' + name
        for binding in review['sources']:
            assert pin(ROOT / binding['path']) == {key: binding[key] for key in ('path', 'bytes', 'sha256')}
        source_reviews.append(path)
    admission_path = DATA / 'input_admission_v1.json'
    admission = doc(admission_path)
    manifest_path = DATA / 'candidate_manifest_v1.json'
    manifest = doc(manifest_path)
    preparation_audit = DATA / 'independent_preparation_audit_v1.json'
    assert admission['complete'] and admission['optimization_permitted'] is True
    assert admission['candidate_manifest'] == pin(manifest_path)
    assert doc(preparation_audit)['independent_preparation_audit_pass'] is True
    spec = importlib.util.spec_from_file_location('inert_learning_packet', PACKAGE / 'packet.py')
    packet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packet)
    assert 'torch' not in sys.modules
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_bytes(), feature_version=(3, 10))
    assert b'\r' not in (PACKAGE / 'run_learning.sh').read_bytes()
    STAGE.mkdir()
    sources = []
    def copy(source, relative):
        source = Path(source)
        original = pin(source)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        installed = pin(target, STAGE)
        assert original['bytes'] == installed['bytes'] and original['sha256'] == installed['sha256']
        assert pin(source) == original
        sources.append({'workspace_source': original, 'stage': installed})
        return installed
    for name in ('__init__.py', 'packet.py', 'worker.py', 'entry.py', 'CONTRACT.md', 'install_weights_from_r1.py'):
        copy(PACKAGE / name, 'research/codeformer_u96_postfusion_learning_v1/' + name)
    copy(PACKAGE / 'stage_entry.py', 'scripts/run_u96_postfusion_learning_v1.py')
    copy(PACKAGE / 'run_learning.sh', 'scripts/run_learning.sh')
    for relative, payload in (
        ('research/__init__.py', b'"""Inert stage research namespace."""\n'),
        ('third_party/__init__.py', b'"""Retained third-party namespace."""\n'),
        ('third_party/codeformer/__init__.py', b'"""Retained CodeFormer architecture and terms."""\n'),
    ):
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    for name in ('codeformer_arch.py', 'vqgan_arch.py', 'registry.py'):
        copy(ROOT / 'third_party/codeformer' / name, 'third_party/codeformer/' + name)
    terms = [path for path in (ROOT / 'third_party/codeformer').iterdir()
             if path.is_file() and ('license' in path.name.lower() or 'notice' in path.name.lower())]
    assert terms
    for path in terms:
        copy(path, 'third_party/codeformer/' + path.name)
    prototype = {}
    for key, name in (('source', 'model.py'), ('design', 'architecture_design_v1_r1.json'),
                      ('review', 'independent_source_review_v1_r1.json')):
        prototype[key] = copy(ROOT / 'research/codeformer_u96_postfusion_v1' / name,
                              'research/codeformer_u96_postfusion_v1/' + name)
    copy(ROOT / 'research/codeformer_u96_postfusion_v1/__init__.py',
         'research/codeformer_u96_postfusion_v1/__init__.py')
    for path in source_reviews:
        copy(path, 'evidence/' + path.name)
    data = {}
    for key, path in (
        ('provenance', manifest_path), ('admission', admission_path),
        ('audit', preparation_audit),
        ('legacy_protocol', ROOT / 'outputs/cctv_codeformer_spatial_continuation_v1_vm/protocol.json'),
    ):
        data[key] = copy(path, 'evidence/data/' + ('legacy_protocol.json' if key == 'legacy_protocol' else path.name))
    # Preserve all input-review decisions, original V1 exclusions and data-revision evidence.
    for path in sorted(DATA.rglob('*.json')):
        if path in (manifest_path, admission_path, preparation_audit):
            continue
        copy(path, 'evidence/data_revision/' + path.relative_to(DATA).as_posix())
    copy(ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_PREFLIGHT_V1_R1_REVIEW.md',
         'evidence/prior_preflight_review.md')
    for name in ('saved_return_audit_v1.json', 'root_visual_review_v1.json'):
        copy(ROOT / 'outputs/codeformer_u96_postfusion_preflight_v1_r1_return_audit' / name,
             'evidence/prior_preflight/' + name)
    copy(ROOT / 'outputs/codeformer_u96_postfusion_preflight_v1_r1_return_peer/independent_return_review_v1.json',
         'evidence/prior_preflight/independent_return_review_v1.json')
    copy(REVIEWS / 'preservation_gate_readback_v1.json', 'evidence/preservation_gate_readback_v1.json')
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'):
        copy(ROOT / name, 'evidence/current_instructions/' + name)
    model = {
        'r2': copy(ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth',
                   'weights/codeformer_r2.pth'),
        'u96': copy(ROOT / 'outputs/codeformer_spatial_continuation_v1_models/codeformer_spatial_u_u96.pth',
                    'weights/codeformer_u96_adapter.pth'),
    }
    order = admission['optimization_reference_order'] + admission['monitor_reference_order']
    cohorts = {reference: ('optimize' if reference in admission['optimization_reference_order'] else 'monitor')
               for reference in order}
    lookup = {row['id']: row for row in manifest['cases']}
    assert len(lookup) == 96
    cases = []
    from PIL import Image
    def copy_bound(binding, relative):
        source = ROOT / binding['path']
        assert pin(source) == binding
        return copy(source, relative)
    for reference in order:
        first = lookup[reference + '__clear']
        target = copy_bound(first['target'], 'assets/targets/' + reference + '.png')
        with Image.open(STAGE / target['path']) as image:
            assert image.mode == 'RGB' and image.size == (256, 256)
            target_rgb = hashlib.sha256(image.tobytes()).hexdigest()
        for role in packet.SLOT_ROLES:
            row = lookup[reference + '__' + role]
            assert row['target'] == first['target']
            inp = target if role == 'clear' else copy_bound(row['input'], 'assets/inputs/' + row['id'] + '.png')
            with Image.open(STAGE / inp['path']) as image:
                assert image.mode == 'RGB' and image.size == (256, 256)
                input_rgb = hashlib.sha256(image.tobytes()).hexdigest()
            cases.append({'id': row['id'], 'reference_id': reference, 'role': role, 'split': 'TRAIN',
                          'cohort': cohorts[reference], 'optimization_permitted': cohorts[reference] == 'optimize',
                          'input': inp, 'target': target, 'input_RGB8_sha256': input_rgb,
                          'target_RGB8_sha256': target_rgb})
    legacy = doc(ROOT / 'outputs/cctv_codeformer_spatial_continuation_v1_vm/protocol.json')
    guards = legacy['guards']
    leaves = {}
    def visit(value):
        if type(value) is dict:
            if set(value) == {'path', 'bytes', 'sha256'}:
                if value['path'] in leaves:
                    assert leaves[value['path']] == value
                leaves[value['path']] = value
            else:
                for child in value.values():
                    visit(child)
        elif type(value) is list:
            for child in value:
                visit(child)
    visit(guards)
    for relative, binding in sorted(leaves.items()):
        source = ROOT / 'outputs/cctv_codeformer_spatial_continuation_v1_vm' / relative
        assert source.stat().st_size == binding['bytes'] and sha(source) == binding['sha256']
        copy(source, relative)
    learning = {**packet.LEARNING_FIXED,
                'optimization_reference_order': admission['optimization_reference_order'],
                'monitor_reference_order': admission['monitor_reference_order']}
    files = {path.relative_to(STAGE).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(STAGE.rglob('*')) if path.is_file()}
    protocol = {'format': packet.FORMAT, 'stage_name': packet.STAGE_NAME,
                'created_UTC': datetime.now(timezone.utc).isoformat(), 'files': files,
                'model': model, 'prototype': prototype, 'data': data,
                'cases': cases, 'guards': guards, 'learning': learning, 'limits': packet.LIMITS,
                'manual_launch_only': True, 'automatic_retry_resume': False,
                'model_qualified': False, 'training_success_not_implied': True,
                'source_status': 'PREPARED_UNRUN_FINITE_IMAGE_LEARNING_PILOT',
                'return_policy': 'All static nonweight files plus runtime/state evidence; complete frozen initializer files retained separately.',
                'frozen_initializer_files_omitted_from_execution_upload': True}
    write_new(STAGE / 'protocol.json', protocol)
    protocol_sha = sha(STAGE / 'protocol.json')
    packet.verify(STAGE, protocol_sha)
    expected = {name: row for name, row in files.items() if not name.startswith('weights/')}
    expected['protocol.json'] = {'bytes': (STAGE / 'protocol.json').stat().st_size, 'sha256': protocol_sha}
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(expected):
                    tar.add(STAGE / name, arcname=STAGE.name + '/' + name, recursive=False)
    digest = sha(archive)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name.startswith(STAGE.name + '/')
            name = member.name[len(STAGE.name) + 1:]
            assert name in expected and name not in seen
            seen.add(name)
            payload = tar.extractfile(member).read()
            assert len(payload) == expected[name]['bytes'] and hashlib.sha256(payload).hexdigest() == expected[name]['sha256']
    assert seen == set(expected) and 'torch' not in sys.modules
    receipt_dir = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_packet'
    receipt_dir.mkdir(exist_ok=False)
    receipt = {'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_TRAINING_PACKET',
               'builder': pin(Path(__file__).resolve()), 'stage': str(STAGE),
               'protocol': pin(STAGE / 'protocol.json'), 'archive': pin(archive), 'checksum': pin(checksum),
               'static_files': len(files), 'archive_members': len(seen), 'source_copies': sources,
               'full_stream_archive_readback': True, 'limits': packet.LIMITS,
               'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    write_new(receipt_dir / 'build_receipt_v1.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('complete', 'status', 'protocol', 'archive', 'archive_members')}))


if __name__ == '__main__':
    main()
