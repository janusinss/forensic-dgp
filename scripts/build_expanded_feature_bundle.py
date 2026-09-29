"""Create a standalone verified VM archive; never train or run a GPU forward."""
import hashlib,json,tarfile
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    if Path.cwd().resolve()!=root:raise RuntimeError('Run from repository root')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    folder=Path('outputs/expanded_feature_data_v1');manifest=json.loads((folder/'manifest.json').read_text())
    check=json.loads((folder/'matched_check.json').read_text())
    if not check['complete'] or check['manifest_sha256']!=sha(folder/'manifest.json'):raise ValueError('Matched data check invalid')
    for p,digest in check['current_code_sha256'].items():
        if sha(p)!=digest:raise ValueError('Changed checked code: '+p)
    files={}
    names=['expanded_feature_data.py','feature_disk_cache.py','anatomical_augmentation.py','feature_vm_runtime.py',
           'completion_data.py','completion.py','completion_inference.py','detector_training.py',
           'scripts/train_expanded_feature_vm.py','scripts/run_expanded_feature_vm.sh',
           'scripts/compare_pixel_heads_vm.py','scripts/compare_presence_heads_vm.py',
           'EXPANDED_FEATURE_VM.md','EXPANDED_FEATURE_DATA.md','ANATOMICAL_AUGMENTATION.md',
           'outputs/sam2.1_hiera_tiny.pt','dataset/detector_glare_review_v3/manifest.json',manifest['split_path']]
    names.extend(str(folder/name) for name in ('manifest.json','verification.json','rejections.json','matched_check.json'))
    for p in names:files[Path(p).as_posix()]=Path(p)
    for p,digest in manifest['evidence_sha256'].items():
        if sha(p)!=digest:raise ValueError('Changed source evidence: '+p)
        files[Path(p).as_posix()]=Path(p)
    for r in manifest['sources']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed source')
        files[r['path']]=Path(r['path'])
    real=Path('dataset/detector_glare_review_v3')
    if sha(real/'manifest.json')!=manifest['v3_manifest_sha256']:raise ValueError('Changed labels')
    for r in json.loads((real/'manifest.json').read_text())['records']:
        for key in ('image','mask'):
            p=real/r[key]
            if sha(p)!=r[key+'_sha256']:raise ValueError('Changed reviewed image/mask')
            files[p.as_posix()]=p
    parents={'outputs/expanded_parents/mixed.pth':('outputs/downloaded_feature_mixed_vm/outputs/feature_mixed_training/final_epoch_20.pth','eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4'),
             'outputs/expanded_parents/spatial.pth':('outputs/downloaded_presence_comparison/spatial_epoch_20.pth','b7af5b8568fc1fd1a6be289a6599794c6c2976e162a351be90a53344997b40c4')}
    for name,(p,digest) in parents.items():
        if sha(p)!=digest:raise ValueError('Parent mismatch')
        files[name]=Path(p)
    vendor=Path('outputs/vendor_sam2')
    for p in (vendor/'sam2').rglob('*'):
        if p.suffix in ('.py','.yaml') and '__pycache__' not in p.parts:files[p.as_posix()]=p
    for p in vendor.iterdir():
        if p.is_file() and p.name.startswith(('LICENSE','NOTICE')):files[p.as_posix()]=p
    inventory={name:sha(p) for name,p in sorted(files.items())}
    archive=Path('outputs/expanded-feature-vm-bundle.tar.gz')
    if archive.exists():raise RuntimeError('Bundle exists; preserve instead of replacing')
    inv=Path('outputs/expanded_feature_inventory.json');inv.write_text(json.dumps(inventory,indent=2))
    with tarfile.open(archive,'w:gz') as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname='expanded_feature_bundle/'+name)
        tar.add(inv,arcname='expanded_feature_bundle/expanded_inventory.json')
    with tarfile.open(archive) as tar:
        if len(tar.getmembers())!=len(files)+1:raise ValueError('Archive member mismatch')
        for m in tar.getmembers():
            name=m.name.removeprefix('expanded_feature_bundle/')
            if not m.isfile() or '..' in Path(name).parts:raise ValueError('Unsafe member')
            expected=sha(inv) if name=='expanded_inventory.json' else inventory[name]
            if hashlib.sha256(tar.extractfile(m).read()).hexdigest()!=expected:raise ValueError('Archive hash mismatch')
    digest=sha(archive)
    archive.with_name(archive.name+'.sha256').write_bytes((digest+'  '+archive.name+'\n').encode('ascii'))
    report={'archive':str(archive),'bytes':archive.stat().st_size,'members':len(files)+1,'sha256':digest,'all_members_verified':True,'gpu_preflight_pending':True}
    Path('outputs/expanded_feature_bundle_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
