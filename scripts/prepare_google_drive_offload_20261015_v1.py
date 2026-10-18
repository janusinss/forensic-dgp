"""Freeze exact local copies already backed by a fully audited Drive TAR."""
import gzip
import hashlib
import json
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/google_drive_offload_20261015_v1'
MANIFEST = 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/recovery_manifest.json'
PIN = '1111c9495a3dfcdcecd73c9ac8d6c21218163edd5db120465da30ff24e225a84'
ARCHIVE_SHA = '5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c'
GROUPS = {'cctv_dgp_actual_step_review_v1_vm_return','cctv_dgp_feature_fusion_v32_r2_return',
          'cctv_dgp_head4_capacity_vm_v1_return','cctv_dgp_loss_cone_probe_v33_return',
          'cctv_dgp_pcgrad_fit_v41_return','cctv_dgp_profile_batches_v31_return',
          'cctv_dgp_spatial_fit_v40_return'}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def bounded(name):
    rel = Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name
    path = ROOT/rel
    assert path.resolve(strict=True).is_relative_to(ROOT.resolve())
    for item in [path, *path.parents]:
        if item == ROOT.parent:
            break
        assert not item.is_symlink() and not item.lstat().st_file_attributes & 0x400, name
    assert stat.S_ISREG(path.lstat().st_mode), name
    return path

def metadata(path):
    s = path.lstat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,dev=s.st_dev,
                nlink=s.st_nlink,mode=s.st_mode,attributes=s.st_file_attributes)

def main():
    start=time.monotonic()
    assert not (OUT/'plan.json').exists()
    assert sha(ROOT/MANIFEST)==PIN
    source=read(ROOT/MANIFEST)
    before=OUT/'namespace_before_r1.json.gz'
    receipt=read(OUT/'namespace_before_r1_receipt.json')
    assert receipt['complete'] and sha(before)==receipt['sha256']
    with gzip.open(before,'rt',encoding='utf-8') as f:
        namespace=json.load(f)
    cloud=read(OUT/'cloud_archive_current.json')
    assert cloud['ID']=='1dAb7nZCps6K1dKIcX_IbXUUNrixK0MJ0' and cloud['Size']==12579983360
    assert {k.lower():v for k,v in cloud['Hashes'].items()}['sha256']==ARCHIVE_SHA
    assert read(OUT/'actual_restore_proof.json')['complete']
    candidates=[]
    for row in source['candidates']:
        name=row['local_backup'];parts=Path(name).parts
        assert len(parts)>3 and parts[0]=='outputs' and parts[1] in GROUPS and parts[2]=='outputs'
        assert name.endswith(('.png','.npy','.npz'))
        assert row['payload_kind'] in {'PNG_image','raw_RGB_array','lossless_RGB_output_pack'},row['payload_kind']
        path=bounded(name);meta=metadata(path)
        assert meta==namespace[name] and meta['bytes']==row['metadata'][2] and meta['nlink']==1,name
        assert row['sha256']==row['local_backup_sha256']
        candidates.append(dict(path=name,sha256=row['sha256'],**meta,
                               recovery_member=row['recovery_member'],payload_kind=row['payload_kind']))
    assert len(candidates)==76973 and sum(r['bytes'] for r in candidates)==12394485961
    names={r['path'] for r in candidates};assert len(names)==len(candidates)
    protected=set()
    for d in ['checkpoints','weights','models','templates','static','docs','scripts']:
        base=ROOT/d
        for folder in base.rglob('*'):
            if folder.is_file() and not folder.is_symlink():
                protected.add(folder.relative_to(ROOT).as_posix())
    protected.update(p.name for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in {'.py','.md','.json','.html','.bat','.ps1','.txt'})
    for name in ['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
                 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth',
                 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth']:
        protected.add(name)
    comparison=ROOT/'outputs/cctv_dgp_checkpoint_comparison_v1'
    for path in comparison.rglob('*'):
        if path.is_file() and not path.is_symlink():
            protected.add(path.relative_to(ROOT).as_posix())
    old=read(ROOT/'outputs/google_drive_offload_20261012_v1/protected_active_bindings.json')
    protected.update(name for name in old if (ROOT/name).is_file())
    for row in source['retained_non_image_arrays']:
        protected.add(row['local_backup'])
    assert not names & protected
    hashes={}
    for n,name in enumerate(sorted(protected),1):
        hashes[name]=sha(bounded(name))
        if n%1000==0: print({'protected_hashed':n,'of':len(protected)},flush=True)
        assert time.monotonic()-start<600,'Finite preparation limit'
    remote='thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1/image/outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/dgp-image-output-recovery-20261010-v1.tar'
    helper_names=['scripts/inventory_google_drive_offload_20261015_v1.py',
                  'scripts/snapshot_google_drive_offload_20261015_v1.py',
                  'scripts/prepare_google_drive_offload_20261015_v1.py',
                  'scripts/apply_google_drive_offload_20261015_v1.py',
                  'scripts/restore_google_drive_offload_20261015_v1.py']
    additions={name:sha(ROOT/name) for name in helper_names if name not in namespace}
    plan=dict(format='existing-drive-tar-local-output-offload-v1',root=str(ROOT),
              destination='thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261015_v1',
              recovery_manifest=dict(path=MANIFEST,sha256=PIN),
              archive=dict(remote=remote,id=cloud['ID'],bytes=cloud['Size'],sha256=ARCHIVE_SHA),
              candidates=candidates,protected_hashes=hashes,
              inventory_before_file=str(before.relative_to(ROOT)).replace('\\','/'),
              inventory_before_sha256=sha(before),candidate_bytes=sum(r['bytes'] for r in candidates),
              allowed_namespace_additions=additions,
              candidate_count=len(candidates),protected_files=len(hashes),
              preserves_all_metadata_checkpoints_caches_splits_failures=True,
              existing_archive_reused_without_duplicate_payload_upload=True,
              new_training_or_VM_calls=0,seconds=time.monotonic()-start)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    (OUT/'plan.sha256').write_text(sha(OUT/'plan.json')+'  plan.json\n',encoding='ascii')
    print(json.dumps({'complete':True,'plan_sha256':sha(OUT/'plan.json'),
                     'candidate_count':len(candidates),'candidate_bytes':plan['candidate_bytes'],
                     'protected_files':len(hashes),'seconds':plan['seconds']}),flush=True)

if __name__=='__main__':main()
