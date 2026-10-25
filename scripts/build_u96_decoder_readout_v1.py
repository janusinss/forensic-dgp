"""Seal byte-bound manual files only after independent current-source review.

This builder has no framework/model/training/VM execution capability.
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
PACKAGE = ROOT / 'research/codeformer_u96_decoder_readout_v1'
PARENT = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
REVIEW = ROOT / 'outputs/codeformer_u96_next_training_decision_v1/runner_review/new_runner_source_review_v1.json'
PREP = ROOT / 'outputs/codeformer_u96_decoder_readout_v1_preparation'
STAGE = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm'
PREFIX = 'cctv-codeformer-u96-decoder-readout-v1'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def pin(path, base=ROOT):
    return {'path': path.relative_to(base).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def doc(path):
    return json.loads(path.read_bytes())


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert 'torch' not in sys.modules
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    assert not any(path.exists() for path in (STAGE, archive, checksum)), 'Prior stage/archives immutable'
    assert shutil.disk_usage(ROOT).free >= 3 * 1024**3
    review = doc(REVIEW)
    assert review['complete'] is True and review['source_review_pass'] is True
    assert not review.get('unresolved_blockers'), 'Independent runner review has blockers'
    for row in review['sources']:
        assert pin(ROOT / row['path']) == row, 'Reviewed source changed: ' + row['path']
    required = {path.relative_to(ROOT).as_posix() for path in PACKAGE.iterdir() if path.is_file()}
    required.add(Path(__file__).resolve().relative_to(ROOT).as_posix())
    assert required <= {row['path'] for row in review['sources']}, 'All executable/builder sources require review'
    reuse_path = PREP / 'readout_conditional_TRAIN_reuse_v1.json'
    reuse = doc(reuse_path)
    assert pin(reuse_path) in review['sources'], 'Independent new-scope data-reuse review required'
    assert reuse['complete'] and reuse['optimization_permitted'] and not reuse['monitor_is_independent_holdout']
    test_path = PREP / 'root_source_tests_v1.json'
    tests = doc(test_path)
    assert tests['complete'] and tests['all_tests_pass'] and tests['neural_gradient_optimizer_training_VM_calls'] == 0
    for row in tests['sources']:
        assert pin(ROOT / row['path']) == row, 'Tested source changed: ' + row['path']
    spec = importlib.util.spec_from_file_location('inert_readout_packet', PACKAGE / 'packet.py')
    packet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packet)
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_bytes(), feature_version=(3, 10))
    assert b'\r' not in (PACKAGE / 'run_learning.sh').read_bytes()
    parent_path = PARENT / 'protocol.json'
    assert (parent_path.stat().st_size, sha(parent_path)) == packet.PARENT_PROTOCOL_FIXED
    parent = doc(parent_path)
    assert reuse['optimization_reference_order'] == parent['learning']['optimization_reference_order']
    assert reuse['monitor_reference_order'] == parent['learning']['monitor_reference_order']
    # Rehash model bytes without loading/duplicating the initializer locally.
    for row in parent['model'].values():
        assert pin(PARENT / row['path'], PARENT) == row
    STAGE.mkdir()
    copies = []
    def copy(source, relative):
        before = pin(source)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        installed = pin(target, STAGE)
        assert (before['bytes'], before['sha256']) == (installed['bytes'], installed['sha256'])
        assert pin(source) == before
        copies.append({'source': before, 'installed': installed})
        return installed
    for path in sorted(PACKAGE.iterdir()):
        if path.is_file() and path.name not in ('stage_entry.py', 'run_learning.sh'):
            copy(path, 'research/codeformer_u96_decoder_readout_v1/' + path.name)
    copy(PACKAGE / 'stage_entry.py', 'scripts/run_u96_decoder_readout_v1.py')
    copy(PACKAGE / 'run_learning.sh', 'scripts/run_learning.sh')
    # Preserve exact parent dataset, admission, terms, guard assets and evidence.
    for name, row in parent['files'].items():
        if name.startswith(('assets/', 'evidence/', 'third_party/')):
            assert pin(PARENT / name, PARENT) == {'path': name, **row}
            copy(PARENT / name, name)
    for name in ('research/__init__.py',):
        target = STAGE / name
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as stream:
            stream.write(b'"""Inert research namespace."""\n')
    parent_pin = copy(parent_path, 'evidence/readout_parent_protocol.json')
    reuse_pin = copy(reuse_path, 'evidence/readout_conditional_TRAIN_reuse_v1.json')
    inherited_pin = copy(ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_v3_interruption_audit_v1/partial_counterexample_analysis_v1.json',
                         'evidence/inherited_DEV/partial_counterexample_analysis_v1.json')
    copy(REVIEW, 'evidence/readout_source_review_v1.json')
    copy(test_path, 'evidence/readout_root_source_tests_v1.json')
    for name in ('DECISION_MEMO_V1.md', 'GATE_READABILITY_ADDENDUM_V1.md',
                 'prospective_pilot_contract_v1.json', 'prospective_TRAIN_reuse_readback_v1.json',
                 'decision_review_by_saved_lineage_peer_v1.json', 'root_inert_prototype_review_v1.json'):
        copy(ROOT / 'outputs/codeformer_u96_next_training_decision_v1' / name, 'evidence/readout_design/' + name)
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'):
        copy(ROOT / name, 'evidence/readout_instructions/' + name)
    model = parent['model']
    prototype = {key: pin(STAGE / ('research/codeformer_u96_decoder_readout_v1/' + name), STAGE)
                 for key, name in (('source', 'model.py'), ('objective', 'objective.py'))}
    files = {path.relative_to(STAGE).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(STAGE.rglob('*')) if path.is_file()}
    for row in model.values():
        files[row['path']] = {key: row[key] for key in ('bytes', 'sha256')}
    learning = {**packet.LEARNING_FIXED,
                'optimization_reference_order': reuse['optimization_reference_order'],
                'monitor_reference_order': reuse['monitor_reference_order']}
    protocol = {'format': packet.FORMAT, 'stage_name': packet.STAGE_NAME,
                'created_UTC': datetime.now(timezone.utc).isoformat(), 'files': files,
                'model': model, 'prototype': prototype, 'data': parent['data'],
                'parent_protocol': parent_pin, 'readout_reuse': reuse_pin,
                'inherited_DEV_evidence': inherited_pin, 'cases': parent['cases'], 'guards': parent['guards'],
                'learning': learning, 'limits': packet.LIMITS, 'quality_policy': packet.QUALITY_POLICY,
                'manual_launch_only': True, 'automatic_retry_resume': False,
                'model_qualified': False, 'training_success_not_implied': True,
                'source_status': 'PREPARED_UNRUN_DISTINCT_EXISTING_DECODER_READOUT',
                'frozen_initializer_files_omitted_from_execution_upload': True}
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
            assert member.isfile() and member.name.startswith(STAGE.name + '/')
            name = member.name[len(STAGE.name) + 1:]
            assert name in expected and name not in seen
            seen.add(name)
            payload = tar.extractfile(member).read()
            assert len(payload) == expected[name]['bytes'] and hashlib.sha256(payload).hexdigest() == expected[name]['sha256']
    assert seen == set(expected) and 'torch' not in sys.modules
    assert checksum.read_bytes() == (digest + '  ' + archive.name + '\n').encode('ascii')
    result = {'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_TRAINING_PACKET',
              'builder': pin(Path(__file__).resolve()), 'protocol': pin(STAGE / 'protocol.json'),
              'archive': pin(archive), 'checksum': pin(checksum), 'source_review': pin(REVIEW),
              'static_files': len(files), 'archive_members': len(seen), 'copies': copies,
              'complete_stream_readback': True, 'initializer_copies_omitted_locally': True,
              'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    write_new(PREP / 'build_receipt_v1.json', result)
    print(json.dumps({key: result[key] for key in ('complete', 'status', 'protocol', 'archive', 'archive_members')}))


if __name__ == '__main__':
    main()
