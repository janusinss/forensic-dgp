"""Synthetic CPU archive/readiness boundaries; never seal the actual packet."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import uuid

from scripts import seal_codeformer_bounded_fusion_feasibility_v1 as s

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def directory():
    base = ROOT / 'scratch'; base.mkdir(exist_ok=True)
    root = base / ('bounded-fusion-sealer-fixture-' + uuid.uuid4().hex)
    root.mkdir(); yield root


def archive(path, rows):
    with tarfile.open(path, 'w:gz') as tar:
        for name, data, kind in rows:
            member = tarfile.TarInfo(name)
            member.type = kind; member.size = len(data) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE): member.linkname = 'a.txt'
            tar.addfile(member, io.BytesIO(data) if kind == tarfile.REGTYPE else None)


def ledger():
    return {'resource_ledger':{'success_call_caps':deepcopy(s.c.CALL_CAPS),
        'result_payload_bytes':1000000000,'archive_upper_bytes':1100000000,
        'metadata_bytes':20000000,'regular_files':2400,'temporary_bytes':128*1024**2}}


class SealingBoundaryTests(unittest.TestCase):
    def test_exact_regular_namespace_all_hashes_and_bytes(self):
        with directory() as root:
            (root/'a.txt').write_bytes(b'one');(root/'b.txt').write_bytes(b'two')
            expected={name:s.c.sha256(root/name) for name in ('a.txt','b.txt')}
            file=root/'packet.tar.gz'
            archive(file,[(s.PACKAGE+'/'+name,(root/name).read_bytes(),tarfile.REGTYPE) for name in expected])
            proof=s.archive_readback(file,root,expected)
            self.assertEqual(proof['members'],2);self.assertEqual(proof['payload_bytes'],6)
            self.assertFalse(proof['returned_code_executed'])

    def test_nonportable_traversal_and_package_escape_rejected(self):
        for name in ('../a.txt','/a.txt',s.PACKAGE+'/../a.txt',s.PACKAGE+'/CON.txt',
                     s.PACKAGE+'/a\\b',s.PACKAGE+'/a:stream','other/a.txt'):
            with self.subTest(name=name),directory() as root:
                (root/'a.txt').write_bytes(b'one');file=root/'bad.tar.gz'
                archive(file,[(name,b'one',tarfile.REGTYPE)])
                with self.assertRaises(ValueError):s.archive_readback(file,root,{'a.txt':s.c.sha256(root/'a.txt')})

    def test_links_directories_duplicates_and_case_collisions_rejected(self):
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE):
            with self.subTest(kind=kind),directory() as root:
                (root/'a.txt').write_bytes(b'one');file=root/'bad.tar.gz'
                archive(file,[(s.PACKAGE+'/a.txt',b'one',kind)])
                with self.assertRaises(ValueError):s.archive_readback(file,root,{'a.txt':s.c.sha256(root/'a.txt')})
        for second in ('a.txt','A.txt'):
            with self.subTest(second=second),directory() as root:
                (root/'a.txt').write_bytes(b'one');file=root/'bad.tar.gz'
                archive(file,[(s.PACKAGE+'/a.txt',b'one',tarfile.REGTYPE),(s.PACKAGE+'/'+second,b'one',tarfile.REGTYPE)])
                with self.assertRaises(ValueError):s.archive_readback(file,root,{'a.txt':s.c.sha256(root/'a.txt')})

    def test_member_hash_size_missing_and_extra_namespace_fail(self):
        for variant in ('hash','size','missing','extra'):
            with self.subTest(variant=variant),directory() as root:
                (root/'a.txt').write_bytes(b'one');(root/'b.txt').write_bytes(b'two')
                expected={'a.txt':s.c.sha256(root/'a.txt')};rows=[(s.PACKAGE+'/a.txt',b'one',tarfile.REGTYPE)]
                if variant=='hash':rows[0]=(s.PACKAGE+'/a.txt',b'bad',tarfile.REGTYPE)
                if variant=='size':rows[0]=(s.PACKAGE+'/a.txt',b'toolong',tarfile.REGTYPE)
                if variant=='missing':expected['b.txt']=s.c.sha256(root/'b.txt')
                if variant=='extra':rows.append((s.PACKAGE+'/b.txt',b'two',tarfile.REGTYPE))
                file=root/'bad.tar.gz';archive(file,rows)
                with self.assertRaises(ValueError):s.archive_readback(file,root,expected)

    def test_finite_compressed_payload_and_file_caps(self):
        with directory() as root:
            (root/'a.txt').write_bytes(b'one');file=root/'packet.tar.gz'
            archive(file,[(s.PACKAGE+'/a.txt',b'one',tarfile.REGTYPE)])
            expected={'a.txt':s.c.sha256(root/'a.txt')}
            with self.assertRaises(ValueError):s.archive_readback(file,root,expected,limit_bytes=1)
            with patch.object(s,'MAX_UPLOAD_PAYLOAD_BYTES',2),self.assertRaises(ValueError):s.archive_readback(file,root,expected)
            with patch.object(s,'MAX_UPLOAD_FILES',0),self.assertRaises(ValueError):s.archive_readback(file,root,expected)
        stream=io.BytesIO();writer=s.FiniteArchive(stream,deadline=100)
        with patch.object(s.time,'monotonic',return_value=101),self.assertRaises(ValueError):writer.write(b'x')
        with patch.object(s.time,'monotonic',return_value=0),patch.object(s,'MAX_UPLOAD_BYTES',1):
            writer.write(b'x')
            with self.assertRaises(ValueError):writer.write(b'y')

    def test_resource_proof_exact_calls_caps_and_physical_minimum(self):
        s.validate_resource_proof(ledger())
        for field,value in [('result_payload_bytes',True),('result_payload_bytes',1),
                            ('regular_files',1),('metadata_bytes',s.c.LIMITS['metadata_bytes']+1),
                            ('temporary_bytes',s.c.LIMITS['temporary_bytes']+1),
                            ('archive_upper_bytes',s.c.LIMITS['archive_bytes']+1),
                            ('success_call_caps',{**s.c.CALL_CAPS,'optimizer_updates':1})]:
            changed=ledger();changed['resource_ledger'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):s.validate_resource_proof(changed)

    def test_unapproved_or_modified_readiness_blocks_before_any_append(self):
        with directory() as root:
            prep=root/'outputs/codeformer_bounded_fusion_feasibility_v1_preparation';prep.mkdir(parents=True)
            file=prep/'final_review_approval_v1.json'
            s.c.write_new_json(file,{'complete':True,'approved':False,'findings':[]})
            with self.assertRaises(ValueError):s.freeze_and_seal(root,s.c.sha256(file))
            self.assertFalse((root/'outputs'/s.PACKAGE).exists())
            with self.assertRaises(ValueError):s.freeze_and_seal(root,'a'*64)

    def test_import_constructs_no_framework_or_vm_action(self):
        code=('import sys;from scripts import seal_codeformer_bounded_fusion_feasibility_v1;'
              'assert not any(n in sys.modules for n in ("torch","numpy","PIL","cv2"))')
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
