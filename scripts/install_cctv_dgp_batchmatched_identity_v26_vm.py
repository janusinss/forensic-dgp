"""Thin packet reconstruction from fingerprinted existing V25; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import platform
import shutil
import signal
import sys
import time
import traceback


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')
def safe(root,name):
    posix=PurePosixPath(name)
    assert not posix.is_absolute() and '..' not in posix.parts and '\\' not in name and ':' not in name and posix.as_posix()==name
    file=(root/name).resolve();assert file.is_relative_to(root.resolve()) and not (root/name).is_symlink(),name
    return file


def verify(root,pin):
    assert sha(root/'protocol.json')==pin,'V26 protocol changed'
    p=read(root/'protocol.json');assert p['format']=='dgp-spatial-batchmatched-identity-capacity-v26'
    assert len(p['transfer_assets_sha256'])==5 and len(p['inherited_assets'])==235 and len(p['assets_sha256'])==240
    for name,digest in p['transfer_assets_sha256'].items():assert sha(safe(root,name))==digest,'Transfer file changed: '+name
    return p


def install(root,p,pin):
    assert sys.platform=='linux' and platform.node().split('.')[0]=='forensic-dgp-thesis','Existing Linux VM only'
    assert root.is_relative_to((Path.home()/'forensic-dgp').resolve()),'~/forensic-dgp only'
    assert not (root/'installation_receipt.json').exists() and not (root/'installation_failure.json').exists(),'Preserve completed/partial installation'
    assert not (root/'outputs').exists() and not (root/'trainer.log').exists(),'Preserve previous training'
    started=time.monotonic();parent=Path.home()/'forensic-dgp/cctv_dgp_spatial_features_vm_v25';copied=[]
    try:
        assert sha(parent/'protocol.json')==p['closed_V25_protocol_sha256'],'Existing V25 protocol changed/missing'
        old=read(parent/'protocol.json')
        assert len(old['assets_sha256'])==235 and old['format']=='dgp-spatial-feature-capacity-v25'
        assert sha(parent/'outputs/early_structure_stop.json')==p['closed_V25_early_failure_sha256'],'Preserve original V25 failure'
        assert read(parent/'outputs/early_structure_stop.json')['pass'] is False
        expected_bytes=0
        for name,row in p['inherited_assets'].items():
            assert old['assets_sha256'][row['source']]==row['sha256']==p['assets_sha256'][name],'Inherited mapping differs'
            source=safe(parent,row['source']);assert sha(source)==row['sha256'],'Existing V25 asset changed: '+row['source']
            assert not safe(root,name).exists(),'Preserve partial inherited file: '+name
            expected_bytes+=source.stat().st_size
        assert shutil.disk_usage(root).free>=3*1024**3+expected_bytes,'Need3GiB free after preserving/copied original assets'
        for name,row in p['inherited_assets'].items():
            assert time.monotonic()-started<60,'Install cap60s'
            source=safe(parent,row['source']);destination=safe(root,name);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination);assert sha(destination)==row['sha256'],'Copy differs: '+name;copied.append(name)
        for name,digest in p['assets_sha256'].items():assert sha(safe(root,name))==digest,'Reconstructed asset changed: '+name
        assert time.monotonic()-started<60,'Install cap60s'
        receipt={'complete':True,'protocol_sha256':pin,'inherited_assets_verified_and_copied':235,'assets_verified':240,
            'inherited_copy_bytes':expected_bytes,'old_V25_protocol_sha256':p['closed_V25_protocol_sha256'],
            'old_V25_failure_preserved':True,'seconds':time.monotonic()-started,'cap_seconds':60,
            'model_or_gradient_calls':0,'optimizer_updates':0,'original_files_changed':False,'quality_acceptance_not_implied':True}
        write(root/'installation_receipt.json',receipt);print(json.dumps(receipt))
    except BaseException as exc:
        write(root/'installation_failure.json',{'complete':False,'protocol_sha256':pin,'copied':copied,'cause':str(exc),
            'traceback':traceback.format_exc(),'seconds':time.monotonic()-started,'resume_permitted':False,'original_files_changed':False})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--verify-transfer',action='store_true');modes.add_argument('--install',action='store_true')
    args=parser.parse_args();root=args.root.resolve();p=verify(root,args.protocol_sha)
    if args.verify_transfer:print(json.dumps({'complete':True,'thin_assets':5,'inherited_assets':235,'model_or_gradient_calls':0}));return
    if sys.platform=='linux':
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Install cap60s')));signal.alarm(60)
    install(root,p,args.protocol_sha)


if __name__=='__main__':main()
