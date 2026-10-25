"""Seal a new manual-only packet after exact source/test reviews.

This builder performs only file operations and static checks. It does not load
checkpoints, import Torch/NumPy, connect to a VM or run any training/inference.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'research/codeformer_u96_spatial_feasibility_v1'
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm'
INITIALIZERS = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
PREP = ROOT / 'outputs/codeformer_u96_spatial_feasibility_v1_preparation'
DESIGN = ROOT / 'outputs/codeformer_u96_next_spatial_design_v1'
REVIEW = PREP / 'runner_review/new_runner_source_review_v1.json'
TESTS = PREP / 'root_runner_source_tests_v1.json'
STAGE = ROOT / 'outputs/cctv_codeformer_u96_spatial_feasibility_v1_vm'
PREFIX = 'cctv-codeformer-u96-spatial-feasibility-v1'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, base=ROOT):
    path, base = Path(path).absolute(), Path(base).absolute()
    need(path == path.resolve(strict=True) and base == base.resolve(strict=True) and
         path.is_relative_to(base), 'Canonical source must remain inside its declared base')
    for ancestor in (path.parent, *path.parent.parents, base):
        info = ancestor.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary source ancestry required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode) and
         not getattr(before, 'st_file_attributes', 0) & 1024, 'Exact regular source required')
    digest = sha(path)
    after = path.lstat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hash')
    return {'path': path.relative_to(base).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def doc(path):
    return json.loads(path.read_bytes())


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_review(review, tests, source_files):
    """No source freezing from a stale green receipt or omitted executable."""
    need(review['complete'] is True and review['source_review_pass'] is True and
         review['unresolved_blockers'] == [], 'Independent complete-runner review required')
    need(tests['complete'] is True and tests['all_tests_pass'] is True and
         tests['neural_gradient_optimizer_training_VM_calls'] == 0,
         'Pure current-source tests required')
    approved = {}
    for record in (review, tests):
        pins = record['sources']
        need(type(pins) is list and len({row['path'] for row in pins}) == len(pins),
             'Unique exact reviewed/tested source bindings required')
        bound = {row['path'] for row in pins}
        need(source_files <= bound, 'Every executable and builder must be both reviewed and tested')
        for row in pins:
            need(pin(ROOT / row['path']) == row, 'Reviewed or tested source changed: ' + row['path'])
            need(row['path'] not in approved or approved[row['path']] == row,
                 'Review/test source bindings disagree')
            approved[row['path']] = row
    return approved


def approved_copy_binding(source, approved, *, expected_content=None):
    """Rejoin the actual copy bytes to approval, not merely a fresh hash."""
    before = pin(source)
    if source.is_relative_to(PACKAGE):
        need(before['path'] in approved and before == approved[before['path']],
             'Executable changed or appeared after review: ' + before['path'])
    if expected_content is not None:
        need((before['bytes'], before['sha256']) ==
             (expected_content['bytes'], expected_content['sha256']),
             'Parent-bound content changed before copy: ' + before['path'])
    return before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Create the exclusive reviewed execution packet')
    args = parser.parse_args()
    need(args.build, 'Use --build only after current independent source review and pure tests')
    need('torch' not in sys.modules and 'numpy' not in sys.modules, 'Static builder only')
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    need(not any(path.exists() for path in (STAGE, archive, checksum)), 'Prior stage/archives immutable')
    need(shutil.disk_usage(ROOT).free >= 3 * 1024**3, 'At least3GiB local build headroom')
    required = {path.relative_to(ROOT).as_posix() for path in PACKAGE.iterdir() if path.is_file()}
    required.add(Path(__file__).resolve().relative_to(ROOT).as_posix())
    approved = validate_review(doc(REVIEW), doc(TESTS), required)
    # Import the inert package through its ordinary qualified namespace.
    sys.path.insert(0, str(ROOT))
    from research.codeformer_u96_spatial_feasibility_v1 import contract, packet
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_bytes(), feature_version=(3, 10))
    need(b'\r' not in (PACKAGE / 'run_feasibility.sh').read_bytes(), 'LF-only VM shell source')
    parent_path = PARENT / 'protocol.json'
    parent_bytes = parent_path.read_bytes()
    selection = contract.select_parent_cases(parent_bytes)
    parent = doc(parent_path)
    scope_path = DESIGN / 'prospective_scope_readback_v1.json'
    scope = doc(scope_path)
    need(scope['complete'] and scope['selected_cases'] == selection and
         scope['parent_protocol'] == pin(parent_path), 'Exact retained prospective selection/readback')
    scope_peer = doc(DESIGN / 'independent_scope_readback_v1.json')
    need(scope_peer['complete'] is True and scope_peer['root_receipt'] == pin(scope_path),
         'Independent data-scope readback must bind current retained receipt')
    for row in parent['model'].values():
        need(pin(INITIALIZERS / row['path'], INITIALIZERS) == row, 'Exact initializer bytes required')
    STAGE.mkdir()
    copies = []

    def copy(source, relative, *, expected_content=None):
        before = approved_copy_binding(source, approved, expected_content=expected_content)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.exists(), 'Exclusive build target')
        shutil.copyfile(source, target)
        installed = pin(target, STAGE)
        need((before['bytes'], before['sha256']) == (installed['bytes'], installed['sha256']) and
             pin(source) == before, 'Source changed during copy')
        copies.append({'source': before, 'installed': installed})
        return installed

    for path in sorted(PACKAGE.iterdir()):
        if path.is_file() and path.name not in ('script_entry.py', 'run_feasibility.sh'):
            copy(path, 'research/codeformer_u96_spatial_feasibility_v1/' + path.name)
    copy(PACKAGE / 'script_entry.py', 'scripts/run_u96_spatial_feasibility_v1.py')
    copy(PACKAGE / 'run_feasibility.sh', 'scripts/run_feasibility.sh')
    for name, row in parent['files'].items():
        if name.startswith(('assets/', 'evidence/', 'third_party/')):
            need(pin(PARENT / name, PARENT) == {'path': name, **row}, 'Parent source bytes differ')
            copy(PARENT / name, name, expected_content=row)
    namespace = STAGE / 'research/__init__.py'
    with namespace.open('xb') as stream:
        stream.write(b'"""Inert research namespace."""\n')
    parent_pin = copy(parent_path, 'evidence/spatial_parent_protocol.json', expected_content={
        'bytes': contract.PARENT_PROTOCOL_BYTES, 'sha256': contract.PARENT_PROTOCOL_SHA256})
    copy(scope_path, 'evidence/spatial_scope_readback_v1.json')
    copy(DESIGN / 'independent_scope_readback_v1.json', 'evidence/spatial_scope_peer_v1.json')
    copy(REVIEW, 'evidence/spatial_runner_source_review_v1.json')
    copy(TESTS, 'evidence/spatial_root_runner_tests_v1.json')
    for name in ('independent_path_review_v1.json', 'independent_path_review_v1.md',
                 'lowlight_tradeoff_review_v1.json', 'lowlight_tradeoff_review_v1.md',
                 'feasibility_contract_review_v1.json', 'feasibility_contract_review_v1.md',
                 'inert_source_review_v1.json', 'direction_independent_review_v1.json',
                 'root_inert_source_tests_v1.json'):
        copy(DESIGN / name, 'evidence/spatial_design/' + name)
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_CODEFORMER_U96_SPATIAL_FEASIBILITY_V1_PREPARATION.md'):
        copy(ROOT / name, 'evidence/spatial_instructions/' + name)
    prototype = {key: pin(STAGE / ('research/codeformer_u96_spatial_feasibility_v1/' + key + '.py'), STAGE)
                 for key in ('model', 'objective', 'direction', 'contract')}
    files = {path.relative_to(STAGE).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(STAGE.rglob('*')) if path.is_file()}
    for row in parent['model'].values():
        files[row['path']] = {key: row[key] for key in ('bytes', 'sha256')}
    need(required == {path.relative_to(ROOT).as_posix() for path in PACKAGE.iterdir() if path.is_file()}
         | {Path(__file__).resolve().relative_to(ROOT).as_posix()}, 'Executable namespace changed during build')
    need(validate_review(doc(REVIEW), doc(TESTS), required) == approved,
         'Approval changed during build')
    protocol = {
        'format': packet.FORMAT, 'stage_name': packet.STAGE_NAME,
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'files': files,
        'model': parent['model'], 'prototype': prototype, 'data': parent['data'],
        'parent_protocol': parent_pin, 'cases': parent['cases'], 'guards': parent['guards'],
        'inherited_DEV_evidence': parent['inherited_DEV_evidence'],
        'quality_policy': parent['quality_policy'],
        'feasibility': contract.prospective_contract(), 'limits': contract.LIMITS,
        'manual_launch_only': True, 'automatic_retry_resume': False,
        'model_qualified': False, 'training_success_not_implied': True,
        'source_status': 'PREPARED_UNRUN_DISTINCT_EXISTING_SPATIAL_DECODER_FEASIBILITY',
        'frozen_initializer_files_omitted_from_execution_upload': True,
    }
    write_new(STAGE / 'protocol.json', protocol)
    protocol_sha = sha(STAGE / 'protocol.json')
    packet.verify_static(STAGE, protocol_sha, missing_weights_for_install=True)
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
            need(member.isfile() and member.name.startswith(STAGE.name + '/'), 'Regular namespaced members only')
            name = member.name[len(STAGE.name) + 1:]
            need(name in expected and name not in seen, 'Exact unique reviewed members')
            seen.add(name)
            payload = tar.extractfile(member).read()
            need(len(payload) == expected[name]['bytes'] and
                 hashlib.sha256(payload).hexdigest() == expected[name]['sha256'], 'Archive payload differs')
    need(seen == set(expected) and 'torch' not in sys.modules and 'numpy' not in sys.modules,
         'Complete pure/static archive readback')
    need(checksum.read_bytes() == (digest + '  ' + archive.name + '\n').encode('ascii'), 'Exact LF checksum')
    need(validate_review(doc(REVIEW), doc(TESTS), required) == approved,
         'Approval/source changed during archive readback')
    receipt = {'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_FEASIBILITY_PACKET',
               'builder': pin(Path(__file__).resolve()), 'protocol': pin(STAGE / 'protocol.json'),
               'archive': pin(archive), 'checksum': pin(checksum), 'source_review': pin(REVIEW),
               'static_files': len(files), 'archive_members': len(seen), 'copies': copies,
               'full_stream_payload_readback': True, 'initializer_copies_omitted_locally': True,
               'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    write_new(PREP / 'build_receipt_v1.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('complete', 'status', 'protocol', 'archive', 'archive_members')}))


if __name__ == '__main__':
    main()
