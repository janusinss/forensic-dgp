"""Synthetic source-only archive sealing fixtures; no model/array loading."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import zlib

from scripts import seal_codeformer_loss_response_v1 as s

ROOT = Path(__file__).resolve().parents[1]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path, value):
    write(path, (json.dumps(value, indent=2) + '\n').encode('utf-8'))


class SourceSealerTests(unittest.TestCase):
    def fixture(self):
        repo = ROOT / 'scratch' / ('loss_sealer_' + uuid.uuid4().hex)
        stage = repo / 'outputs' / (s.c.VM_NAME + '_r1')
        review = repo / s.REVIEW
        review.mkdir(parents=True)
        stage.mkdir(parents=True)
        source = {name: ('Synthetic source file ' + name + '\n').encode() for name in s.SOURCE_MAP}
        source['research/__init__.py'] = b''
        deps = {'protocol.json': {'sha256': s.c.PARENT_PROTOCOL_SHA, 'bytes': 8}}
        rows = []
        a_variants = ['severe24', 'severe28', 'severe32', 'severe40']
        b_variants = ['clear256', 'clear128', 'light256', 'light64']
        for index, update in enumerate(s.c.SOURCE_UPDATES):
            for slot in (1,2,3,4):
                rid = 'tr_ffhq_%05d' % (100 + index * 2 + (slot > 2))
                variant = (a_variants if slot <= 2 else b_variants)[index]
                edge = int(''.join(char for char in variant if char.isdigit()))
                p = {'behavior_anchor': slot == 3, 'image_eligible': edge >= 32,
                     'variant': variant, 'long_edge': edge}
                prefix = 'outputs/adain_gradient_isolation_v1/training_inputs/update_%04d/slot_%d/' % (update,slot)
                hq = 'outputs/adain_gradient_isolation_v1/HQ_cache/' + rid + '/'
                source_path = 'data/sources/' + rid + '.png'
                source_sha = s.digest(source_path.encode()); input_sha = s.digest(prefix.encode())
                row = {'id': 'u%04d_s%d' % (update,slot), 'update': update, 'slot': slot,
                    'reference_id': rid, 'profile': variant, 'parameters': p,
                    'source_path': source_path, 'input_path': prefix + 'input.png',
                    'pair_record_path': prefix + 'pair_record.json', 'hq_indices_path': hq + 'indices.npy',
                    'hq_features_path': hq + 'features.npy', 'hq_record_path': hq + 'record.json',
                    'record': {'role': 'train', 'reference_id': rid, 'update': update, 'slot': slot,
                               'profile': variant, 'parameters': p, 'source_path': source_path,
                               'source_sha256': source_sha, 'saved_input_sha256': input_sha}}
                if slot == 3:
                    anchor = 'outputs/adain_gradient_isolation_v1/behavior_anchors/update_%04d/' % update
                    row.update(behavior_delivered_path=anchor+'delivered256.npy',
                               behavior_logits_path=anchor+'logits.npy', behavior_record_path=anchor+'record.json')
                for key, value in row.items():
                    if key.endswith('_path'):
                        deps[value] = {'sha256': source_sha if key == 'source_path' else
                                       input_sha if key == 'input_path' else s.digest(value.encode()), 'bytes': 8}
                rows.append(row)
        cases = {'format': s.c.CASES_FORMAT, 'source_updates': s.c.SOURCE_UPDATES, 'cases': rows}
        source['cases.json'] = (json.dumps(cases, indent=2) + '\n').encode()
        assets = {}
        for name in ('initializer','teacher','perceptual_features','perceptual_calibration',
                     'original_checkpoint','stopped_checkpoint'):
            path = 'retained/' + name + '.pth'
            deps[path] = {'sha256': s.digest(name.encode()), 'bytes': 8}
            assets[name] = {'path': path, **deps[path]}
        own = {name: s.digest(payload) for name, payload in source.items()}
        sizes = {name: len(payload) for name, payload in source.items()}
        protocol = {'format': s.c.FORMAT, 'seed': 20261015, 'outputs': 'outputs/loss_response_v1',
            'parent_directory': s.c.PARENT_NAME, 'parent_protocol_sha256': s.c.PARENT_PROTOCOL_SHA,
            'source_updates': s.c.SOURCE_UPDATES, 'cases_file': 'cases.json', 'snapshots': ['original0','stopped50'],
            'branches': s.c.BRANCHES, 'loss_weights': s.c.WEIGHTS, 'optimizer': s.c.OPTIMIZER,
            'numerical': copy.deepcopy(s.c.NUMERICAL), 'limits': copy.deepcopy(s.c.LIMITS), 'call_caps': copy.deepcopy(s.c.COUNTS),
            'total_neural_forwards': 425, 'total_gradient_queries': 193, 'files_sha256': own,
            'files_bytes': sizes, 'dependencies': deps, 'assets': assets, 'automatic_app_promotion': False,
            'model_qualified': False, 'new_training_recipe': False, 'export_candidate_checkpoint': False,
            'manual_tmux_required': True, 'all_branches_disposable': True, 'quality_gates_changed': False,
            'status': 'PREPARED_UNRUN'}
        source['protocol.json'] = (json.dumps(protocol, indent=2) + '\n').encode()
        for name, payload in source.items():
            write(stage/name, payload)
        for name, relative in s.SOURCE_MAP.items():
            write(repo/relative, source[name])
        write(repo/s.BUILDER, b'# synthetic builder; never executed\n')
        receipt = {'complete': True, 'stage': str(stage), 'protocol_sha256': s.digest(source['protocol.json']),
            'source_files': own, 'source_bytes': sizes, 'dependencies': deps,
            'source_sha256': s.digest((repo/s.BUILDER).read_bytes()),
            'execution_archive_sealed': False, 'model_qualified': False,
            'actual_neural_gradient_optimizer_VM_calls': 0}
        receipt_path = review/'stage_v1_r1.json'
        write_json(receipt_path, receipt)
        readiness = {'complete': True, 'source_data_review_pass': True, 'CPU_only': True, 'manual_unrun': True,
            'protocol_sha256': receipt['protocol_sha256'], 'stage_receipt_sha256': s.digest(receipt_path.read_bytes()),
            'actual_neural_gradient_optimizer_VM_calls': 0, 'optional_extra_audit': {'no_schema_assumed': True}}
        readiness_path = review/'readiness.json'
        write_json(readiness_path, readiness)
        return SimpleNamespace(repo=repo, stage=stage, review=review, source=source, p=protocol, cases=cases,
                               receipt=receipt, receipt_path=receipt_path, ready=readiness, readiness=readiness_path)

    def validate(self, f):
        return s.validate_stage(f.repo, f.receipt_path, f.readiness)

    def rebind_protocol(self, f):
        write_json(f.stage/'protocol.json', f.p)
        f.receipt['protocol_sha256'] = s.digest((f.stage/'protocol.json').read_bytes())
        write_json(f.receipt_path, f.receipt)
        f.ready['protocol_sha256'] = f.receipt['protocol_sha256']
        f.ready['stage_receipt_sha256'] = s.digest(f.receipt_path.read_bytes())
        write_json(f.readiness, f.ready)

    def test_exact_synthetic_schema_and_source_binding_passes(self):
        f = self.fixture()
        result = self.validate(f)
        self.assertEqual(len(result['source']), 17)
        self.assertEqual(result['source']['research/__init__.py'], b'')
        self.assertEqual(result['readiness_sha256'], s.digest(f.readiness.read_bytes()))

    def test_stale_or_non_cpu_readiness_refused(self):
        for key, value in [('protocol_sha256','0'*64), ('stage_receipt_sha256','1'*64),
                           ('complete',False), ('CPU_only',False), ('manual_unrun',False),
                           ('source_data_review_pass',False), ('actual_neural_gradient_optimizer_VM_calls',True)]:
            f = self.fixture(); f.ready[key] = value; write_json(f.readiness, f.ready)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(f)

    def test_mutated_protocol_or_cases_or_current_sources_refused(self):
        for target in ('protocol', 'cases', 'repo_source', 'builder', 'empty_init', 'source_size'):
            f = self.fixture()
            if target == 'protocol':
                f.p['limits']['worker_seconds'] = 1801
                self.rebind_protocol(f)
            elif target == 'cases':
                write(f.stage/'cases.json', b'{"stale":true}\n')
            elif target == 'repo_source':
                write(f.repo/'research/codeformer_loss_response_v1/worker.py', b'changed')
            elif target == 'builder':
                write(f.repo/s.BUILDER, b'changed builder')
            elif target == 'empty_init':
                write(f.stage/'research/__init__.py', b'#unexpected')
            else:
                f.p['files_bytes']['cases.json'] += 1
                self.rebind_protocol(f)
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.validate(f)

    def test_semantically_invalid_but_rehashed_cases_refused(self):
        f = self.fixture()
        f.cases['cases'][0]['record']['slot'] = 4
        write_json(f.stage/'cases.json', f.cases)
        payload = (f.stage/'cases.json').read_bytes()
        f.p['files_sha256']['cases.json'] = f.receipt['source_files']['cases.json'] = s.digest(payload)
        f.p['files_bytes']['cases.json'] = f.receipt['source_bytes']['cases.json'] = len(payload)
        self.rebind_protocol(f)
        with self.assertRaises(ValueError):
            self.validate(f)

    def test_unlisted_output_directory_binary_or_collision_refused(self):
        for name in ('outputs', 'unexpected.npy', 'CASES.json'):
            f = self.fixture()
            if name == 'outputs':
                (f.stage/name).mkdir()
            else:
                write(f.stage/name, b'forbidden')
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(f)

    def test_original_stage_is_never_sealed(self):
        f = self.fixture()
        f.receipt['stage'] = str(f.stage.with_name(s.c.VM_NAME))
        f.stage.rename(f.stage.with_name(s.c.VM_NAME))
        write_json(f.receipt_path, f.receipt)
        f.ready['stage_receipt_sha256'] = s.digest(f.receipt_path.read_bytes()); write_json(f.readiness,f.ready)
        with self.assertRaisesRegex(ValueError, 'revised stage'):
            self.validate(f)

    def test_link_metadata_guards_without_physical_link_privilege(self):
        f = self.fixture()
        with mock.patch.object(s.c,'no_links',side_effect=lambda p:Path(p)), \
                mock.patch.object(Path,'lstat',return_value=SimpleNamespace(st_mode=stat.S_IFREG|0o644,st_nlink=2)), \
                self.assertRaisesRegex(ValueError,'singly linked'):
            s.exact_regular(f.stage/'cases.json')
        for mode, attributes in ((stat.S_IFLNK|0o777,0),(stat.S_IFREG|0o644,1024)):
            with self.subTest(mode=mode,attributes=attributes), \
                    mock.patch.object(Path,'lstat',return_value=SimpleNamespace(st_mode=mode,st_file_attributes=attributes)), \
                    self.assertRaisesRegex(ValueError,'Linked/reparse'):
                s.c.no_links(f.stage/'cases.json')

    def test_physical_hardlink_refused_when_available(self):
        f = self.fixture()
        path = f.stage/'cases.json'
        try:
            os.link(path, f.repo/'hardlink.json')
        except OSError as error:
            self.skipTest('Host hardlink privilege unavailable: ' + str(error))
        with self.assertRaisesRegex(ValueError,'singly linked'):
            self.validate(f)

    def test_deterministic_archive_and_independent_headers_payload_readback(self):
        f = self.fixture()
        one, two = f.repo/'first.tar.gz', f.repo/'second.tar.gz'
        s.write_archive(one,f.source); s.write_archive(two,f.source)
        self.assertEqual(one.read_bytes(),two.read_bytes())
        receipt = s.readback_archive(one,f.source)
        self.assertEqual(receipt['regular_members'],17)
        self.assertTrue(receipt['independent_payload_readback_pass'])
        self.assertEqual(receipt['member_payload_bytes'],sum(map(len,f.source.values())))
        with self.assertRaisesRegex(ValueError,'overwrite forbidden'):
            s.write_archive(one,f.source)

    def test_corrupted_archive_header_or_payload_refused(self):
        for corruption in ('gzip_header','gzip_payload','tar_uid','extra_member','second_gzip_frame'):
            f = self.fixture(); path = f.repo/'test.tar.gz'; s.write_archive(path,f.source)
            data = bytearray(path.read_bytes())
            if corruption == 'gzip_header':
                data[4] = 1; path.write_bytes(data)
            elif corruption == 'gzip_payload':
                data[len(data)//2] ^= 1; path.write_bytes(data)
            elif corruption == 'second_gzip_frame':
                path.write_bytes(data + gzip.compress(b'',mtime=0))
            else:
                raw = io.BytesIO()
                with gzip.GzipFile(filename='',mode='wb',mtime=0,fileobj=raw) as gz:
                    with tarfile.open(fileobj=gz,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
                        for name in sorted(f.source):
                            h=tarfile.TarInfo(s.c.VM_NAME+'/'+name);h.size=len(f.source[name]);h.mode=0o644
                            h.uid=1 if corruption=='tar_uid' else 0
                            archive.addfile(h,io.BytesIO(f.source[name]))
                        if corruption=='extra_member':
                            h=tarfile.TarInfo(s.c.VM_NAME+'/extra.json');h.size=2
                            archive.addfile(h,io.BytesIO(b'{}'))
                path.write_bytes(raw.getvalue())
            with self.subTest(corruption=corruption), self.assertRaises((ValueError,OSError,EOFError,tarfile.TarError,zlib.error)):
                s.readback_archive(path,f.source)

    def test_caps_refuse_before_payload_overflow(self):
        stream=io.BytesIO(); capped=s.CappedWriter(stream,4)
        capped.write(b'abcd')
        with self.assertRaisesRegex(ValueError,'finite cap'):
            capped.write(b'e')
        f=self.fixture()
        with mock.patch.object(s,'SOURCE_CAP',1), self.assertRaises(ValueError):
            self.validate(f)

    def test_seal_receipt_checksum_and_overwrite_refusal(self):
        f=self.fixture()
        result=s.seal(f.repo,f.receipt_path,f.readiness)
        archive=f.repo/'outputs'/s.ARCHIVE_NAME
        checksum=Path(str(archive)+'.sha256')
        self.assertEqual(checksum.read_bytes(),(s.digest(archive.read_bytes())+'  '+archive.name+'\n').encode('ascii'))
        self.assertEqual(result['readiness_sha256'],s.digest(f.readiness.read_bytes()))
        self.assertEqual(result['actual_neural_gradient_optimizer_VM_calls'],0)
        self.assertFalse(result['training_launched']); self.assertFalse(result['model_qualified'])
        before=archive.read_bytes()
        with self.assertRaisesRegex(ValueError,'Retain existing'):
            s.seal(f.repo,f.receipt_path,f.readiness)
        self.assertEqual(before,archive.read_bytes())

    def test_no_neural_modules_imported(self):
        self.assertFalse(any(name=='torch' or name.startswith('torch.') for name in sys.modules))


if __name__=='__main__':
    unittest.main()
