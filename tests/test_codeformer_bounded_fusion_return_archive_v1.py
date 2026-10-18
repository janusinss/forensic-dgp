"""Synthetic finite stdlib transportation fixtures; no arrays/models execute."""
from contextlib import contextmanager,ExitStack
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch
import uuid

from research.codeformer_bounded_fusion_feasibility_v1 import return_archive as a

ROOT=Path(__file__).resolve().parents[1]


def payload(value):return (json.dumps(value,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()


@contextmanager
def fixture(variant=None):
    root=ROOT/'scratch'/('bounded-return-fixture-'+uuid.uuid4().hex);root.mkdir(parents=True)
    packet=root/'packet';packet.mkdir();parent=root/'parent';parent.mkdir();review=root/'review';review.mkdir()
    caps={**a.CALL_CAPS,'codeformer_forwards':20,'fusion_block_evaluations':120}
    (parent/'protocol.json').write_bytes(payload({'fixture_parent':True}));parent_sha=a.sha(parent/'protocol.json')
    asset_paths={};assets=deepcopy(a.ASSETS);asset_dir=root/'assets';asset_dir.mkdir()
    for key in assets:
        file=asset_dir/(key+'.weights');file.write_bytes(('inert '+key+' weights').encode());asset_paths[key]=file
        assets[key].update(sha256=a.sha(file),bytes=file.stat().st_size,parent_protocol_sha256=parent_sha)
    frozen=b'raise RuntimeError("returned source must NEVER execute")\n';(packet/'source.py').write_bytes(frozen)
    p={'format':'codeformer-bounded-fusion-feasibility-finite-manual-v1','outputs':a.OUTPUTS,'export_prefix':a.PREFIX,
       'limits':deepcopy(a.LIMITS),'assets':assets,'call_caps':caps,'files_sha256':{'source.py':digest(frozen)},
       'export_files':['source.py'],'cases':[{'id':'train_u0001_slot1'},{'id':'train_u0001_slot2'}],
       'canary_ids':['train_u0001_slot1','train_u0001_slot2']}
    (packet/'protocol.json').write_bytes(payload(p));protocol_sha=a.sha(packet/'protocol.json')
    with ExitStack() as stack:
        stack.enter_context(patch.object(a,'CALL_CAPS',caps));stack.enter_context(patch.object(a,'ASSETS',assets))
        stack.enter_context(patch.object(a,'PROTOCOL_SHA256',protocol_sha));stack.enter_context(patch.object(a,'RESERVE_BYTES',0))
        entries={name:(payload({}) if name.endswith('.json') else b'inert byte payload') for name in a.output_namespace(p)}
        launch={'protocol_sha256':protocol_sha,'pid':42,'manual_tmux':True,'automatic_training':False,'optimizer_updates':0}
        entries[a.OUTPUTS+'/manual_launch.json']=payload(launch)
        entries[a.OUTPUTS+'/worker_complete.json']=payload({'complete':True,'protocol_sha256':protocol_sha,'pid':42,
            'model_qualified':False,'next_training_authorized':False,'automatic_app_promotion':False,'counts':caps})
        stages=[{'stage':flag,'exit_code':0,'seconds':1.0,'external_cap_seconds':cap,'grace_seconds':grace,
                 'log':'outputs/supervision/'+log,'log_capped':False} for flag,cap,grace,log in
                [('--verify-transfer',300,30,'preflight.log'),('--run',900,60,'diagnostic.log')]]
        supervisor={'protocol_sha256':protocol_sha,'manual_tmux':True,'no_automatic_retry':True,
            'training_success_not_implied':True,'model_qualified':False,'automatic_app_promotion':False,
            'optimizer_updates':0,'neural_measurement_complete':True,'stages':stages}
        entries['outputs/supervision/run_receipt.json']=payload(supervisor)
        entries['outputs/supervision/preflight.log']=entries['outputs/supervision/diagnostic.log']=b'closed log\n'
        entries['protocol.json']=payload(p);entries['source.py']=frozen
        if variant=='changed_source':entries['source.py']=b'changed frozen source'
        if variant=='missing_path':del entries[next(n for n in sorted(entries) if n.endswith('/raw256.npy'))]
        if variant=='missing_operational':del entries['outputs/supervision/preflight.log']
        if variant=='live_export_log':entries['outputs/supervision/export.log']=b'live log forbidden'
        if variant=='weights':entries[a.OUTPUTS+'/weights.pth']=b'forbidden'
        if variant=='oversized_log':entries['outputs/supervision/diagnostic.log']=b'x'*(2*1024**2+1)
        if variant=='wrong_worker_count':
            worker=json.loads(entries[a.OUTPUTS+'/worker_complete.json']);worker['counts']={**caps,'optimizer_updates':1}
            entries[a.OUTPUTS+'/worker_complete.json']=payload(worker)
        if variant=='nonfinite_json':entries[a.OUTPUTS+'/worker_complete.json']=b'{"x":1e999}'
        if variant=='duplicate_json':entries[a.OUTPUTS+'/worker_complete.json']=b'{"complete":true,"complete":false}'
        if variant=='path_escape':entries['../escaped.py']=b'forbidden escape'
        if variant=='case_collision':entries['SOURCE.py']=frozen
        m={'format':'codeformer-bounded-fusion-feasibility-return-v1','protocol_sha256':protocol_sha,
           'files_sha256':{n:digest(v) for n,v in entries.items()},'files_bytes':{n:len(v) for n,v in entries.items()},
           'parent_assets':deepcopy(assets),'retained_not_exported_sha256':{},'complete_measurement':True,
           'failure_present':False,'optimizer_updates':0,'model_qualified':False,'native_CCTV_qualification':False,
           'automatic_training':False}
        if variant=='retained_input':m['retained_not_exported_sha256']={'unbound_input.png':'a'*64}
        if variant=='wrong_parent':m['parent_assets']['stopped']['sha256']='a'*64
        if variant=='wrong_manifest_bytes':m['files_bytes']['source.py']+=1
        if variant=='unlisted_member':m['files_sha256'].pop('source.py');m['files_bytes'].pop('source.py')
        manifest_bytes=payload(m);entries[a.MANIFEST_NAME]=manifest_bytes
        archive=root/a.RESULT_ARCHIVE
        with tarfile.open(archive,'w:gz') as tar:
            for name,value in sorted(entries.items()):
                member=tarfile.TarInfo(name);member.size=len(value)
                if variant=='symlink' and name=='source.py':member.type=tarfile.SYMTYPE;member.linkname='protocol.json';member.size=0
                if variant=='hardlink' and name=='source.py':member.type=tarfile.LNKTYPE;member.linkname='protocol.json';member.size=0
                tar.addfile(member,io.BytesIO(value) if member.isreg() else None)
                if variant=='duplicate_member' and name=='source.py':tar.addfile(member,io.BytesIO(value))
        archive_sha=a.sha(archive);archive_size=archive.stat().st_size
        stack.enter_context(patch.object(a,'ARCHIVE_SHA256',archive_sha));stack.enter_context(patch.object(a,'ARCHIVE_BYTES',archive_size))
        stack.enter_context(patch.object(a,'MANIFEST_SHA256',digest(manifest_bytes)))
        sidecar=Path(str(archive)+'.sha256');sidecar.write_bytes((archive_sha+'  '+archive.name+'\n').encode())
        export_path=root/a.EXPORT_RECEIPT
        export={'complete':True,'archive_sha256':archive_sha,'bytes':archive_size,'manifest_member':a.MANIFEST_NAME,
            'manifest_sha256':digest(manifest_bytes),'protocol_sha256':protocol_sha,'measurement_complete':True,
            'failure_present':False,'seconds':1.0,'optimizer_updates':0,'model_qualified':False,'training_success_not_implied':True}
        export_path.write_bytes(payload(export))
        if variant=='crlf_sidecar':sidecar.write_bytes(sidecar.read_bytes().replace(b'\n',b'\r\n'))
        if variant=='damaged_archive':
            with archive.open('ab') as stream:stream.write(b'changed')
        if variant=='changed_retained_packet':(packet/'source.py').write_bytes(b'changed on disk')
        if variant=='changed_parent_asset':asset_paths['stopped'].write_bytes(b'changed')
        args=(archive,sidecar,export_path,packet,root/'return',review/'transport_v1.json')
        kwargs={'parent_packet':parent,'original_asset':asset_paths['original'],'stopped_asset':asset_paths['stopped']}
        yield root,args,kwargs


class ReturnTransportTests(unittest.TestCase):
    def test_positive_full_verify_import_and_no_qualification(self):
        with fixture() as (root,args,kwargs):
            result=a.inspect_and_import(*args,**kwargs)
            self.assertTrue(result['complete']);self.assertTrue(result['whole_archive_and_all_member_hashes_verified_before_import'])
            self.assertFalse(result['returned_code_executed']);self.assertFalse(result['model_qualified'])
            self.assertEqual(result['files_sha256'][a.MANIFEST_NAME],a.MANIFEST_SHA256)
            self.assertEqual(len(result['files_sha256']),len(result['files_bytes']))
            self.assertEqual((root/'return/source.py').read_bytes(),(root/'packet/source.py').read_bytes())
            with self.assertRaises(ValueError):a.inspect_and_import(*args,**kwargs)

    def test_all_corruption_and_status_variants_fail_before_import(self):
        variants=('changed_source','missing_path','missing_operational','live_export_log','weights','oversized_log',
            'wrong_worker_count','nonfinite_json','duplicate_json','path_escape','case_collision','retained_input',
            'wrong_parent','wrong_manifest_bytes','unlisted_member','symlink','hardlink','duplicate_member',
            'crlf_sidecar','damaged_archive','changed_retained_packet','changed_parent_asset')
        for variant in variants:
            with self.subTest(variant=variant),fixture(variant) as (root,args,kwargs):
                with self.assertRaises((ValueError,KeyError,tarfile.TarError)):a.inspect_and_import(*args,**kwargs)
                self.assertFalse((root/'return').exists());self.assertTrue((root/'review/transport_v1_failure.json').is_file())

    def test_partial_import_is_preserved_after_io_failure(self):
        with fixture() as (root,args,kwargs):
            original=tarfile.TarFile.extractfile;seen=[0]
            def interrupted(owner,member):
                if member.name=='source.py':
                    seen[0]+=1
                    if seen[0]==2:raise OSError('synthetic import write boundary')
                return original(owner,member)
            with patch.object(tarfile.TarFile,'extractfile',interrupted),self.assertRaises(OSError):a.inspect_and_import(*args,**kwargs)
            self.assertTrue((root/'return').is_dir());self.assertTrue((root/'return_import_journal.jsonl').is_file())
            failure=a.read(root/'review/transport_v1_failure.json')
            self.assertEqual(failure['phase'],'exclusive_import');self.assertTrue(failure['partial_evidence_retained'])
            self.assertGreater(failure['imported_members'],0);self.assertFalse((root/'review/transport_v1.json').exists())

    def test_canonical_paths_and_prefix_collisions(self):
        for name in ('../a','/a','a//b','a\\b','x:stream','CON.txt','x/COM¹','x/trail.','x/a\x00'):
            with self.subTest(name=name),self.assertRaises(ValueError):a.canonical(name)
        for names in (['a','a/b'],['a','A']):
            with self.assertRaises(ValueError):a.namespace(names)

    def test_finite_deadline_and_disk_stop_before_import(self):
        with fixture() as (root,args,kwargs),patch.object(a,'MAX_SECONDS',-1):
            with self.assertRaises(ValueError):a.inspect_and_import(*args,**kwargs)
            self.assertFalse((root/'return').exists())
        with fixture() as (root,args,kwargs),patch.object(a.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaises(ValueError):a.inspect_and_import(*args,**kwargs)
            self.assertFalse((root/'return').exists())

    def test_import_time_does_not_load_frameworks_or_touch_vm(self):
        code='import sys;from research.codeformer_bounded_fusion_feasibility_v1 import return_archive;assert not any(n in sys.modules for n in ("torch","numpy","PIL","cv2"))'
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
