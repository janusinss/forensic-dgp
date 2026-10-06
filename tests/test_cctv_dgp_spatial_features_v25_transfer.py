"""V25 return-integrity/path/failure evidence checks; no neural imports."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import import_cctv_dgp_spatial_features_v25 as transfer
BUNDLE=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25'
REQUIRED=['protocol.json','schedule.json','scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/run_v25.sh',
    'cctv_dgp_spatial_features_v25.py','cctv_dgp_spatial_features_v25_prepare.py','cctv_dgp_degraded_objective_v24.py']


class ReturnTransfer(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'scratch'/('v25_transfer_'+uuid.uuid4().hex);self.root.mkdir(parents=True)
    def fixture(self,extra=None,omit=None,alter_manifest=False,failed=False):
        files={n:(BUNDLE/n).read_bytes() for n in REQUIRED if n!=omit}
        if failed:files['outputs/failure.json']=b'{"complete":false,"fixture":true}\n'
        declared={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
        if alter_manifest:declared['schedule.json']='0'*64
        files['export_manifest.json']=json.dumps({'complete':True,'protocol_sha256':transfer.PIN,'files_sha256':declared}).encode()
        archive=self.root/transfer.ARCHIVE_NAME
        with tarfile.open(archive,'x:gz') as stream:
            for n,v in files.items():
                info=tarfile.TarInfo(transfer.PREFIX+'/'+n);info.size=len(v);stream.addfile(info,io.BytesIO(v))
            if extra is not None:
                stream.addfile(extra,io.BytesIO(b'x') if extra.isreg() else None)
        digest=transfer.sha(archive);side=Path(str(archive)+'.sha256');side.write_text(digest+'  '+archive.name+'\n',encoding='ascii')
        receipt=self.root/'export.json';transfer.write(receipt,{'complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,
            'training_success_not_implied':True,'run_results_present':False,'failure_present':failed})
        return archive,side,receipt
    def test_failed_return_can_be_imported_without_success_claim(self):
        paths=self.fixture(failed=True);result=transfer.import_return(*paths,self.root/'imported',self.root)
        self.assertTrue(result['complete']);self.assertFalse(result['goal_complete']);self.assertEqual(result['neural_calls'],0)
        self.assertIn('outputs/failure.json',result['files_sha256'])
    def test_wrong_sidecar_rejected_before_extraction(self):
        a,s,r=self.fixture();s.write_text('0'*64+'  '+a.name+'\n',encoding='ascii')
        with self.assertRaisesRegex(ValueError,'checksum'):transfer.import_return(a,s,r,self.root/'imported',self.root)
        self.assertFalse((self.root/'imported').exists())
    def test_reported_VM_hash_required_if_supplied(self):
        paths=self.fixture()
        with self.assertRaisesRegex(ValueError,'reported archive hash'):transfer.import_return(*paths,self.root/'imported',self.root,expected_sha='0'*64)
    def test_missing_new_feature_helper_rejected(self):
        a,_,_=self.fixture(omit='cctv_dgp_spatial_features_v25_prepare.py')
        with self.assertRaisesRegex(ValueError,'Missing returned frozen source'):transfer.inspect_archive(a)
    def test_manifest_payload_tamper_rejected(self):
        a,_,_=self.fixture(alter_manifest=True)
        with self.assertRaisesRegex(ValueError,'member hash differs'):transfer.inspect_archive(a)
    def test_traversal_rejected(self):
        info=tarfile.TarInfo(transfer.PREFIX+'/../escape');info.size=1;a,_,_=self.fixture(extra=info)
        with self.assertRaisesRegex(ValueError,'Unsafe archive path'):transfer.inspect_archive(a)
    def test_windows_duplicate_case_rejected(self):
        info=tarfile.TarInfo(transfer.PREFIX+'/PROTOCOL.JSON');info.size=1;a,_,_=self.fixture(extra=info)
        with self.assertRaisesRegex(ValueError,'Duplicate Windows archive'):transfer.inspect_archive(a)
    def test_links_rejected(self):
        info=tarfile.TarInfo(transfer.PREFIX+'/link');info.type=tarfile.SYMTYPE;info.linkname='protocol.json';a,_,_=self.fixture(extra=info)
        with self.assertRaisesRegex(ValueError,'regular files'):transfer.inspect_archive(a)
    def test_oversized_member_rejected_without_reading(self):
        a,_,_=self.fixture();old=transfer.MAX_FILE_BYTES
        try:
            transfer.MAX_FILE_BYTES=1
            with self.assertRaisesRegex(ValueError,'member size exceeds cap'):transfer.inspect_archive(a)
        finally:transfer.MAX_FILE_BYTES=old
    def test_receipt_cannot_mislabel_failure_presence(self):
        a,s,r=self.fixture(failed=True);data=transfer.read(r);data['failure_present']=False;r.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'presence receipt differs'):transfer.import_return(a,s,r,self.root/'imported',self.root)
        self.assertFalse((self.root/'imported').exists())


if __name__=='__main__':unittest.main()
