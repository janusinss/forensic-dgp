"""Tiny opaque-archive regression fixtures; no model or VM execution."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import import_cctv_dgp_decoder_activation_v1_return as importer

PREPARATION = ROOT/'outputs/cctv_dgp_decoder_activation_v1_preparation'
FAILURES = (AssertionError, tarfile.TarError, EOFError, OSError, json.JSONDecodeError)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')


@contextmanager
def fixture():
    base = ROOT/'scratch/decoder_activation_importer_fixtures'
    base.mkdir(parents=True, exist_ok=True)
    target = base/('fixture-'+uuid.uuid4().hex)
    target.mkdir()  # Ordinary inherited Windows ACL; never tempfile mode0700.
    try:
        assert target.resolve().parent == base.resolve()
        assert target.resolve().is_relative_to(ROOT.resolve())
        yield target
    finally:
        resolved = target.resolve()
        assert resolved.parent == base.resolve() and resolved.is_relative_to(ROOT.resolve())
        assert not target.is_symlink()
        shutil.rmtree(target)


def package(base, *, partial=False, extra=None, manifest_change=None, payload_change=None,
            asset_pin=None, protocol_payload=None, newline=b'\n', sidecar_filename=None,
            tar_member_type=None):
    asset = b'{"opaque_fixture":true}\n'
    protocol = dict(name='tiny-test-fixture', budgets=dict(output_byte_cap=1024*1024),
                    assets_sha256={'design.json':asset_pin or digest_bytes(asset)})
    protocol_bytes = encoded(protocol)
    pin = digest_bytes(protocol_bytes)
    payloads = {'protocol.json':protocol_payload if protocol_payload is not None else protocol_bytes,
                'design.json':asset, importer.STAGE+'/payload.bin':bytes(range(256)),
                'run_exit.json':encoded(dict(exit_code=-9 if partial else 0, timeout=partial, kill_sent=partial))}
    if partial:
        payloads[importer.STAGE+'/progress.jsonl'] = b'{"counts":{"gradient_queries":35}}\n'
    else:
        payloads[importer.STAGE+'/run_results.json'] = encoded(dict(complete=True,counts=dict(gradient_queries=70,optimizer_updates=0)))
    for name, value in (extra or []):
        payloads[name] = value
    if payload_change:
        payload_change(payloads)
    manifest = dict(complete=True, protocol_sha256=pin, optimizer_updates=0, model_qualified=False,
        training_success_not_implied=True, gradient_queries=None if partial else 70,
        partial_terminal_without_worker_receipt=partial,
        files={name:dict(bytes=len(value),sha256=digest_bytes(value)) for name,value in payloads.items()})
    if manifest_change:
        manifest_change(manifest)
    archive = base/'cctv-dgp-decoder-activation-v1-results.tar.gz'
    with tarfile.open(archive,'x:gz') as stream:
        for name, value in [('result_manifest.json',encoded(manifest))]+list(payloads.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(value), 0o644, 0
            if tar_member_type and name == tar_member_type[0]:
                member.type = tar_member_type[1]
                member.linkname = '../../outside.bin'
            stream.addfile(member,io.BytesIO(value))
    sidecar = Path(str(archive)+'.sha256')
    export_path = base/'export.json'
    exported = dict(complete=True, archive_sha256=digest_file(archive), bytes=archive.stat().st_size,
        protocol_sha256=pin, optimizer_updates=0, model_qualified=False, training_success_not_implied=True,
        partial_terminal_without_worker_receipt=partial, gradient_queries=None if partial else 70)
    sidecar.write_bytes((exported['archive_sha256']+'  '+(sidecar_filename or archive.name)).encode('utf-8')+newline)
    export_path.write_bytes(encoded(exported))
    return dict(archive=archive,sidecar=sidecar,export_path=export_path,target=base/'returned',
                protocol=protocol,pin=pin,payloads=payloads,exported=exported)


def refresh_transport(obj):
    value=json.loads(obj['export_path'].read_text(encoding='utf-8'))
    value.update(archive_sha256=digest_file(obj['archive']),bytes=obj['archive'].stat().st_size)
    obj['sidecar'].write_bytes((value['archive_sha256']+'  '+obj['archive'].name+'\n').encode('utf-8'))
    obj['export_path'].write_bytes(encoded(value))


class ImporterArchiveFixtures(unittest.TestCase):
    def call(self, obj):
        return importer.import_verified(**{key:obj[key] for key in ('archive','sidecar','export_path','target','protocol','pin')})

    def source_snapshot(self, obj):
        return {key:digest_file(obj[key]) for key in ('archive','sidecar','export_path')}

    def reject_before_target(self, obj):
        before=self.source_snapshot(obj); protocol=copy.deepcopy(obj['protocol'])
        with patch.object(Path,'mkdir',side_effect=RuntimeError('Unexpected target creation')) as mkdir:
            with self.assertRaises(FAILURES):self.call(obj)
            mkdir.assert_not_called()
        self.assertFalse(obj['target'].exists())
        self.assertEqual(before,self.source_snapshot(obj))
        self.assertEqual(protocol,obj['protocol'])

    def test_valid_complete_exact_payload_and_source_objects(self):
        with fixture() as base:
            obj=package(base); before=self.source_snapshot(obj); protocol=copy.deepcopy(obj['protocol'])
            result=self.call(obj)
            self.assertTrue(result['complete']);self.assertTrue(result['all_hashes_before_extraction'])
            self.assertTrue(result['extracted_payload_rehashed']);self.assertTrue(result['casefold_unique_namespace'])
            self.assertFalse(result['model_qualified']);self.assertEqual(result['exact_gradient_queries'],70)
            self.assertEqual(result['protocol_sha256'],obj['pin'])
            self.assertEqual(before,self.source_snapshot(obj));self.assertEqual(protocol,obj['protocol'])
            self.assertEqual({p.relative_to(obj['target']).as_posix() for p in obj['target'].rglob('*') if p.is_file()},set(obj['payloads'])|{'result_manifest.json'})
            for rel,value in obj['payloads'].items():self.assertEqual((obj['target']/rel).read_bytes(),value)
            self.assertEqual((obj['target']/importer.STAGE/'payload.bin').read_bytes(),bytes(range(256)))

    def test_valid_partial_keeps_unknown_gradient_count(self):
        with fixture() as base:
            obj=package(base,partial=True);result=self.call(obj)
            self.assertTrue(result['partial_terminal_without_worker_receipt'])
            self.assertIsNone(result['exact_gradient_queries'])
            self.assertFalse((obj['target']/importer.STAGE/'run_results.json').exists())
            self.assertEqual((obj['target']/importer.STAGE/'payload.bin').read_bytes(),bytes(range(256)))

    def test_LF_CRLF_and_BOM_sidecars(self):
        for newline in (b'\n',b'\r\n'):
            with self.subTest(newline=newline),fixture() as base:
                obj=package(base,newline=newline);self.assertTrue(self.call(obj)['complete'])
        with fixture() as base:
            obj=package(base);obj['sidecar'].write_bytes(b'\xef\xbb\xbf'+obj['sidecar'].read_bytes())
            self.assertTrue(self.call(obj)['complete'])

    def test_wrong_sidecar_filename_rejected(self):
        with fixture() as base:self.reject_before_target(package(base,sidecar_filename='other-results.tar.gz'))

    def test_download_corruption_rejected(self):
        with fixture() as base:
            obj=package(base);value=bytearray(obj['archive'].read_bytes());value[len(value)//2]^=0x20;obj['archive'].write_bytes(value)
            self.reject_before_target(obj)

    def test_truncated_archive_rejected_even_with_updated_transport_hash(self):
        with fixture() as base:
            obj=package(base);value=obj['archive'].read_bytes();obj['archive'].write_bytes(value[:len(value)//2]);refresh_transport(obj)
            self.reject_before_target(obj)

    def test_member_hash_mismatch_rejected(self):
        with fixture() as base:
            obj=package(base,manifest_change=lambda m:m['files'][importer.STAGE+'/payload.bin'].update(sha256='0'*64))
            self.reject_before_target(obj)

    def test_required_asset_hash_is_not_only_manifest_hash(self):
        with fixture() as base:self.reject_before_target(package(base,asset_pin='0'*64))

    def test_protocol_payload_must_match_pinned_SHA(self):
        with fixture() as base:self.reject_before_target(package(base,protocol_payload=b'changed protocol payload\n'))

    def test_casefold_duplicate_names_rejected(self):
        with fixture() as base:
            obj=package(base,extra=[(importer.STAGE+'/A.bin',b'a'),(importer.STAGE+'/a.bin',b'b')])
            self.reject_before_target(obj)

    def test_traversal_backslash_absolute_and_windows_names_rejected(self):
        names=['../outside.bin',importer.STAGE+'/../outside.bin',importer.STAGE+'\\outside.bin',
               '/absolute.bin','C:/absolute.bin',importer.STAGE+'/CON.txt',importer.STAGE+'/NUL',
               importer.STAGE+'/com1.log',importer.STAGE+'/LPT9.ext',importer.STAGE+'/trailing.',importer.STAGE+'/trailing ']
        for name in names:
            with self.subTest(name=name),fixture() as base:self.reject_before_target(package(base,extra=[(name,b'opaque')]))

    def test_NUL_and_noncanonical_names_rejected(self):
        for name in ('',importer.STAGE+'/a\x00b.bin',importer.STAGE+'/control\x1f.bin',
                     importer.STAGE+'//double.bin','./'+importer.STAGE+'/dot.bin'):
            with self.subTest(safe_name=name),self.assertRaises(AssertionError):importer.safe_name(name)
        for name in (importer.STAGE+'/a\x00b.bin',importer.STAGE+'//double.bin','./'+importer.STAGE+'/dot.bin'):
            with self.subTest(archive_name=name),fixture() as base:self.reject_before_target(package(base,extra=[(name,b'opaque')]))

    def test_windows_invalid_punctuation_rejected_before_target(self):
        for character in '<>"|?*':
            with self.subTest(character=character),fixture() as base:
                name=importer.STAGE+'/bad'+character+'name.bin'
                self.reject_before_target(package(base,extra=[(name,b'opaque')]))

    def test_regular_file_cannot_be_another_member_parent(self):
        for parent,child in [('parent','parent/child.bin'),('Parent','parent/child.bin')]:
            with self.subTest(parent=parent,child=child),fixture() as base:
                obj=package(base,extra=[(importer.STAGE+'/'+parent,b'parent file'),(importer.STAGE+'/'+child,b'child file')])
                self.reject_before_target(obj)

    def test_symlink_hardlink_and_directory_tar_members_forbidden(self):
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
            with self.subTest(kind=kind),fixture() as base:
                rel=importer.STAGE+'/forbidden-link'
                obj=package(base,extra=[(rel,b'')],tar_member_type=(rel,kind));self.reject_before_target(obj)

    def test_exact_manifest_namespace_required(self):
        with fixture() as base:
            obj=package(base,manifest_change=lambda m:m['files'].update({'missing.bin':dict(bytes=0,sha256=digest_bytes(b''))}))
            self.reject_before_target(obj)
        with fixture() as base:
            obj=package(base,manifest_change=lambda m:m['files'].pop(importer.STAGE+'/payload.bin'))
            self.reject_before_target(obj)

    def test_duplicate_existing_return_is_not_touched(self):
        with fixture() as base:
            obj=package(base);self.call(obj)
            before={p.relative_to(obj['target']).as_posix():digest_file(p) for p in obj['target'].rglob('*') if p.is_file()}
            archive_before=self.source_snapshot(obj)
            with patch.object(Path,'mkdir',side_effect=RuntimeError('Unexpected target mkdir')) as mkdir:
                with self.assertRaises(AssertionError):self.call(obj)
                mkdir.assert_not_called()
            self.assertEqual(before,{p.relative_to(obj['target']).as_posix():digest_file(p) for p in obj['target'].rglob('*') if p.is_file()})
            self.assertEqual(archive_before,self.source_snapshot(obj))


def main():
    started=time.monotonic();log=io.StringIO()
    PREPARATION.mkdir(parents=True,exist_ok=True)
    final_json,final_txt=PREPARATION/'local_importer_preparation_r1.json',PREPARATION/'local_importer_preparation_r1.txt'
    assert not final_json.exists() and not final_txt.exists(),'Preserve existing completed local importer evidence'
    text=Path(__file__).read_text(encoding='utf-8');ast.parse(text,feature_version=(3,10));compile(text,str(__file__),'exec')
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    attempt=1
    while (PREPARATION/f'local_importer_r1_attempt_{attempt:03d}.json').exists():attempt+=1
    receipt=dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,
        seconds=time.monotonic()-started,Python=sys.version,fixture_kind='ordinary UUID workspace-local directories; resolved containment before recursive cleanup',
        source_sha256={p.relative_to(ROOT).as_posix():digest_file(p) for p in [Path(__file__),Path(importer.__file__),ROOT/'scripts/cctv_dgp_decoder_activation_v1_contract.py']},
        log_sha256=digest_bytes(log.getvalue().encode('utf-8')),model_forwards=0,gradient_queries=0,optimizer_updates=0,VM_connections=0,
        archived_fixture_payload_bytes=256,scientific_or_model_qualification=False,VM_archive_resealed=False)
    for path,value in [(PREPARATION/f'local_importer_r1_attempt_{attempt:03d}.txt',log.getvalue().encode('utf-8')),
                       (PREPARATION/f'local_importer_r1_attempt_{attempt:03d}.json',encoded(receipt))]:
        with path.open('xb') as stream:stream.write(value)
    if result.wasSuccessful():
        with final_txt.open('xb') as stream:stream.write(log.getvalue().encode('utf-8'))
        with final_json.open('xb') as stream:stream.write(encoded(receipt))
    print(log.getvalue(),end='');print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
