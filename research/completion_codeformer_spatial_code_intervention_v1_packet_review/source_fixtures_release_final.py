"""Local pure-file fixtures for the actual safe-installer and bounded exporter.
Uses tiny fake source/metadata files, never a checkpoint, model, NN or VM.
"""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import contextlib, gzip, hashlib, importlib.util, io, json, os, tarfile, tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIXTURES = Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs\spcode_v5')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


builder = load('spatial_builder_fixtures', ROOT / 'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py')
exporter = load('spatial_exporter_fixtures', ROOT / 'scripts/export_completion_codeformer_spatial_code_intervention_v1.py')
rows = []


def sha(value): return hashlib.sha256(value).hexdigest()


def addtar(stream, name, body=b'', kind='regular'):
    info = tarfile.TarInfo(name); info.size = len(body)
    if kind == 'symlink': info.type = tarfile.SYMTYPE; info.linkname = '../../escaped'; info.size = 0
    stream.addfile(info, io.BytesIO(body) if kind == 'regular' else None)


def installer_case(label, *, extra=None, omit=False, duplicate=False, symlink=False, traversal=False, existing=False, wrong_hash=False, space=None):
    home = FIXTURES / ('i_' + label); home.mkdir(); base = home / 'forensic-dgp'; base.mkdir()
    relative = '../escaped.bin' if traversal else 'data/source.bin'
    body = b'fixed-source-payload'
    protocol = {'format': builder.FORMAT, 'files': [{'path': relative, 'bytes': len(body), 'sha256': sha(body)}]}
    encoded = (json.dumps(protocol) + '\n').encode(); digest = ('0' * 64 if wrong_hash else sha(encoded))
    target = base / builder.VM_NAME
    if existing: target.mkdir(); (target / 'retain.txt').write_bytes(b'preserved')
    archive = home / (builder.STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        addtar(tar, builder.VM_NAME + '/protocol.json', encoded)
        if not omit: addtar(tar, builder.VM_NAME + '/' + relative, body, 'symlink' if symlink else 'regular')
        if duplicate: addtar(tar, builder.VM_NAME + '/' + relative, body)
        if extra: addtar(tar, builder.VM_NAME + '/' + extra, b'unlisted')
    code = builder.INSTALL_CODE.replace('__PROTOCOL_SHA__', digest)
    failed = None
    with patch.object(Path, 'home', return_value=home), patch.object(builder.shutil, 'disk_usage', return_value=SimpleNamespace(free=space if space is not None else 100 * 1024 ** 3)), contextlib.redirect_stdout(io.StringIO()):
        try: exec(compile(code, 'exact_installer_literal', 'exec'), {})
        except Exception as exc: failed = repr(exc)
    expected_failure = any((extra, omit, duplicate, symlink, traversal, existing, wrong_hash, space is not None))
    assert bool(failed) == bool(expected_failure), (label, failed)
    assert not (home / 'escaped.bin').exists() and not (base / 'escaped.bin').exists()
    if not expected_failure: assert (target / relative).read_bytes() == body
    if existing: assert (target / 'retain.txt').read_bytes() == b'preserved'
    rows.append({'case': 'installer/' + label, 'passed': True, 'expected_failure': bool(expected_failure), 'actual_error': failed, 'archive_bytes': archive.stat().st_size, 'installer_literal_sha256': sha(builder.INSTALL_CODE.encode()), 'no_VM_or_models': True})


def export_case(label, *, state=None, alias=False, distinct=False, extra=False):
    home = FIXTURES / ('e_' + label); home.mkdir(); root = home / exporter.VM_NAME; root.mkdir(); outputs = root / 'outputs'; outputs.mkdir()
    source = b'pinned-source'; (root / 'source.txt').write_bytes(source)
    p = {'format': builder.FORMAT, 'manual_execution_only': True, 'automatic_resume_or_promotion': False,
         'contract': {'output_bytes': 1024 ** 3, 'stop_free_bytes': 2 * 1024 ** 3}, 'parity_cases': [{'id': 'parity'}], 'bridge_cases': [{'id': 'bridge'}],
         'arms': [{'id': 'H', 'filename_tag': 'ORACLE_H_mask_touched_same_codebook'}, {'id': 'V', 'filename_tag': 'ORACLE_V_visible_context_same_codebook'}],
         'monitor_cases': [{'id': 'monitor'}], 'challenge_cases': [{'id': 'challenge'}],
         'files': [{'path': 'source.txt', 'bytes': len(source), 'sha256': sha(source)}], 'export': {}}
    p['export']['allowed_output_paths'] = builder.allowed_output_paths(p)
    encoded = (json.dumps(p) + '\n').encode(); (root / 'protocol.json').write_bytes(encoded)
    exit_code = 1 if state in ('partial', 'committed') else 0
    (root / 'supervisor.json').write_text(json.dumps({'complete': True, 'worker_exit': exit_code, 'timeout': False}))
    (root / 'trainer.log').write_bytes(b'quiet fixture\n')
    final = {'complete': True, 'protocol_sha256': sha(encoded), 'model_qualified': False, 'diagnostic_only': True, 'new_model_created': False}
    (outputs / ('failure.json' if exit_code else 'results.json')).write_text(json.dumps(final))
    path = outputs / 'failed_actual_model_state.pth'; partial = Path(str(path) + '.partial')
    if state == 'partial': partial.write_bytes(b'NOT-A-CHECKPOINT malformed retained partial fixture')
    elif state in ('committed', 'successful_state'):
        path.write_bytes(b'NOT-A-CHECKPOINT failure serialization fixture')
        if alias: os.link(path, partial)
        if distinct: partial.write_bytes(path.read_bytes())
    if extra: (outputs / 'unlisted.bin').write_bytes(b'must-refuse')
    failed, receipt = None, None
    with patch.object(exporter.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 * 1024 ** 3)), contextlib.redirect_stdout(io.StringIO()):
        try: receipt = exporter.export(root, sha(encoded), home, enforce_vm=False)
        except Exception as exc: failed = repr(exc)
    should_fail = distinct or extra or state == 'successful_state'
    assert bool(failed) == bool(should_fail), (label, failed)
    if not should_fail:
        archive = home / (exporter.STEM + '-results.tar.gz')
        with tarfile.open(archive, 'r:gz') as tar:
            names = tar.getnames(); manifest = json.load(tar.extractfile(exporter.MEMBER_ROOT + '/export_manifest.json'))
        assert len(names) == len(set(names))
        if alias:
            assert len(manifest['same_inode_failed_state_aliases']) == 1
            assert exporter.MEMBER_ROOT + '/outputs/failed_actual_model_state.pth' in names
            assert exporter.MEMBER_ROOT + '/outputs/failed_actual_model_state.pth.partial' not in names
            assert partial.exists() and partial.stat().st_ino == path.stat().st_ino
        if state == 'partial': assert exporter.MEMBER_ROOT + '/outputs/failed_actual_model_state.pth.partial' in names
        assert archive.stat().st_ino == Path(str(archive) + '.partial').stat().st_ino
    rows.append({'case': 'export/' + label, 'passed': True, 'expected_failure': should_fail, 'actual_error': failed, 'successful_export': receipt is not None, 'dummy_serialized_bytes_never_deserialized': True, 'no_VM_or_models': True})


def main():
    need = lambda ok: None if ok else (_ for _ in ()).throw(ValueError('Fixture destination exists'))
    need(not (OUT / 'source_fixtures_release.json').exists())
    FIXTURES.mkdir(exist_ok=False)
    installer_case('valid')
    installer_case('wrong_sha', wrong_hash=True)
    installer_case('missing_payload', omit=True)
    installer_case('unlisted_payload', extra='extra.bin')
    installer_case('duplicate_payload', duplicate=True)
    installer_case('symlink_payload', symlink=True)
    installer_case('traversal_manifest', traversal=True)
    installer_case('existing_target', existing=True)
    installer_case('six_GiB_without_payload_space', space=6 * 1024 ** 3)
    export_case('valid_success')
    export_case('unlisted_output_refused', extra=True)
    export_case('success_checkpoint_refused', state='successful_state')
    export_case('failure_partial_retained', state='partial')
    export_case('failure_committed_retained', state='committed')
    export_case('failure_same_inode_alias_once', state='committed', alias=True)
    export_case('failure_distinct_pair_refused', state='committed', distinct=True)
    old_limit = exporter.MAX_BYTES; exporter.MAX_BYTES = 8
    output = io.BytesIO(); writer = exporter.CountedWriter(output, lambda: None)
    try:
        try: writer.write(b'oversized')
        except ValueError: pass
        else: raise AssertionError('Pre-write bound not enforced')
        assert output.getvalue() == b''
    finally: exporter.MAX_BYTES = old_limit
    rows.append({'case': 'export/pre_write_cap_before_IO', 'passed': True, 'actual_source_Class_used': True, 'isolated_tiny_limit_fixture': True, 'production_constants_unchanged': exporter.MAX_BYTES == 1280 * 1024 ** 2})
    report = {'format': 'completion-spatial-code-intervention-v1-author-pure-file-fixtures-v1', 'complete': True, 'checks': len(rows), 'all_pass': True, 'cases': rows, 'installer_literal_sha256': sha(builder.INSTALL_CODE.encode()), 'source_pins': [builder.pin(ROOT / 'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py'), builder.pin(ROOT / 'scripts/export_completion_codeformer_spatial_code_intervention_v1.py')], 'model_imports': 0, 'checkpoint_deserializations': 0, 'neural_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0}
    with (OUT / 'source_fixtures_release.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'complete': True, 'checks': len(rows), 'receipt': builder.pin(OUT / 'source_fixtures_release.json')}))


if __name__ == '__main__': main()
