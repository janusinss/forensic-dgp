"""Build a self-contained, non-training R1 installer from sealed V1 assets."""
from pathlib import Path
import base64, copy, gzip, hashlib, json
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/completion_codeformer_finetune_v1_packet/cctv_completion_codeformer_finetune_v1_vm'
OUT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_packet'
RETURN=ROOT/'outputs/completion_codeformer_finetune_v1_return/cctv_completion_codeformer_finetune_v1_results'
OLD_SHA='0dfe963bd29856ea6201d6397aa25d651ca0291661fa3ec26539c9c770e31b8f'
def digest(b):return hashlib.sha256(b).hexdigest()
old_body=(OLD/'protocol.json').read_bytes();assert digest(old_body)==OLD_SHA
old=json.loads(old_body);new=copy.deepcopy(old)
new['frozen_UTC']=datetime.now(timezone.utc).isoformat()
new['export']['stem']='cctv-completion-codeformer-finetune-v1-r1'
new['compatibility_repair']={'parent_protocol_sha256':OLD_SHA,'failed_return_archive_sha256':'7d51d190d7cba45f6c0f2f99e90e32d06fb5ef43444bf55b07e973ba0ba91bd6','reason':'Canonical app active-index input layout and indexed output composition; retain parity arrays/layouts before unchanged 1e-6 gate','failed_parent_optimizer_steps':0,'scientific_contract_unchanged':True,'exact_failed_CUDA_cause_proven':False,'new_directory':'cctv_completion_codeformer_finetune_v1_r1_vm','old_directory_preserved':True,'automatic_launch':False}
replacements={
'scripts/cctv_completion_codeformer_finetune_v1_vm.py':'scripts/cctv_completion_codeformer_finetune_v1_r1_vm.py',
'scripts/export_completion_codeformer_finetune_v1.py':'scripts/export_completion_codeformer_finetune_v1_r1.py',
'scripts/run_completion_codeformer_finetune_v1.sh':'scripts/run_completion_codeformer_finetune_v1_r1.sh'}
blobs={}
for row in new['files']:
    before=row['path']
    if before in replacements:
        after=replacements[before];body=(ROOT/after).read_bytes();row.update(path=after,bytes=len(body),sha256=digest(body));blobs[after]=base64.b64encode(body).decode('ascii')
new_body=(json.dumps(new,indent=2,allow_nan=False)+'\n').encode();new_sha=digest(new_body)
assert new['contract']==old['contract'] and len(new['files'])==329
parent_evidence=[]
for name in ('supervisor.json','trainer.log','outputs/failure.json','outputs/terminal_training_state.pth'):
    path=RETURN/name
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    parent_evidence.append({'path':name,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
payload={'old_protocol_b64':base64.b64encode(old_body).decode(),'new_protocol_b64':base64.b64encode(new_body).decode(),'old_sha':OLD_SHA,'new_sha':new_sha,'replacements':replacements,'new_sources':blobs,'parent_evidence':parent_evidence}
encoded=base64.b85encode(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0)).decode('ascii')
template=r'''"""Install the isolated R1 completion pilot; verifies/clones existing assets only.
Python3.10 compatible. No Torch, GPU, gradients, training, network, or deletion.
Old failed V1 and original checkpoint remain untouched. Never automatically run.
"""
import argparse,base64,gzip,hashlib,json,os,shutil,stat,time,uuid
from pathlib import Path,PurePosixPath
PAYLOAD_B85=__PAYLOAD__
OLD_NAME='cctv_completion_codeformer_finetune_v1_vm'
NEW_NAME='cctv_completion_codeformer_finetune_v1_r1_vm'
MAX_SECONDS=180
class Refused(ValueError):pass
def need(ok,message):
    if not ok:raise Refused(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def package():
    raw=gzip.decompress(base64.b85decode(PAYLOAD_B85));need(len(raw)<1024*1024,'Installer payload cap')
    data=json.loads(raw);old=base64.b64decode(data['old_protocol_b64'],validate=True);new=base64.b64decode(data['new_protocol_b64'],validate=True)
    need(hashlib.sha256(old).hexdigest()==data['old_sha'] and hashlib.sha256(new).hexdigest()==data['new_sha'],'Embedded protocol digest differs')
    return data,old,new
def member(root,relative):
    need(isinstance(relative,str) and relative and '\\' not in relative,'Invalid relative member')
    name=PurePosixPath(relative);need(not name.is_absolute() and '..' not in name.parts and ':' not in relative,'Member traversal')
    path=root.joinpath(*name.parts)
    for item in [path,*path.parents]:
        need(not item.is_symlink(),'Symlink refused: '+str(item))
        if item==root:break
    need(path.resolve().is_relative_to(root.resolve()),'Member escaped root')
    return path
def check_file(root,row):
    path=member(root,row['path']);info=path.lstat()
    need(stat.S_ISREG(info.st_mode) and info.st_size==row['bytes'],'Regular file/size differs: '+row['path'])
    need(sha(path)==row['sha256'],'SHA differs: '+row['path']);return path
def install(family,*,enforce_vm=True):
    started=time.monotonic();family=Path(family).resolve(strict=True)
    if enforce_vm:need(os.name=='posix' and family==(Path.home()/'forensic-dgp').resolve(strict=True),'Install only existing VM ~/forensic-dgp')
    def guard():need(time.monotonic()-started<MAX_SECONDS,'Installer180second stop; retain any staging directory')
    data,old_bytes,new_bytes=package();old=json.loads(old_bytes);new=json.loads(new_bytes)
    need(new['contract']==old['contract'] and new['checkpoint']==old['checkpoint'],'Scientific/checkpoint contract changed')
    source=member(family,OLD_NAME);target=member(family,NEW_NAME)
    need(source.is_dir() and not target.exists(),'Existing V1 required; preserve any existing R1 without overwrite')
    need((source/'protocol.json').read_bytes()==old_bytes,'Existing V1 protocol differs')
    for row in old['files']:guard();check_file(source,row)
    for row in data['parent_evidence']:guard();check_file(source,row)
    failure=json.loads((source/'outputs/failure.json').read_text());supervisor=json.loads((source/'supervisor.json').read_text())
    counts=failure['counts'];need(failure['exception']=="ValueError('Fresh frozen app pipeline baseline parity failed')",'Expected retained baseline-parity failure')
    need(counts['backward_calls']==counts['code_optimizer_steps']==counts['fusion_optimizer_steps']==0,'Parent is not zero-update failure')
    need(supervisor['complete'] is True and supervisor['worker_exit']==1 and not supervisor['timeout'],'Terminal parent failure receipt differs')
    weight=(source/old['checkpoint']['path']).resolve(strict=True)
    need(weight.is_relative_to(family),'Shared checkpoint escaped forensic-dgp')
    check_file(family,{'path':weight.relative_to(family).as_posix(),'bytes':old['checkpoint']['bytes'],'sha256':old['checkpoint']['sha256']})
    guard();need(shutil.disk_usage(family).free>=new['contract']['minimum_free_bytes']+sum(r['bytes'] for r in new['files'])+len(new_bytes),'Need4GiB plus30MiB before R1 installation')
    stage=member(family,NEW_NAME+'.installing-'+uuid.uuid4().hex);stage.mkdir()
    for row in old['files']:
        guard();relative=row['path']
        if relative in data['replacements']:continue
        src=check_file(source,row);dst=member(stage,relative);dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as incoming,dst.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing,8*1024*1024)
    for relative,encoded in data['new_sources'].items():
        guard();dst=member(stage,relative);dst.parent.mkdir(parents=True,exist_ok=True)
        with dst.open('xb') as stream:stream.write(base64.b64decode(encoded,validate=True))
    with (stage/'protocol.json').open('xb') as stream:stream.write(new_bytes)
    expected={r['path'] for r in new['files']}|{'protocol.json'}
    actual={path.relative_to(stage).as_posix() for path in stage.rglob('*') if path.is_file()}
    need(actual==expected,'New R1 file namespace differs')
    for row in new['files']:guard();check_file(stage,row)
    # Final source/evidence recheck before publishing; no operation writes to V1.
    for row in old['files']+data['parent_evidence']:guard();check_file(source,row)
    need((source/'protocol.json').read_bytes()==old_bytes,'Parent protocol changed while installing')
    need(not target.exists(),'R1 appeared while installing; preserve staging')
    guard();stage.rename(target)
    return {'complete':True,'installed_root':str(target),'protocol_sha256':data['new_sha'],'files':len(new['files']),'parent_protocol_sha256':data['old_sha'],'parent_retained':True,'optimizer_updates':0,'neural_calls':0,'training_launched':False,'seconds':time.monotonic()-started}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-sha',required=True)
    args=parser.parse_args();need(sha(Path(__file__).resolve())==args.expected_sha,'Installer self SHA differs')
    print(json.dumps(install(Path.home()/'forensic-dgp')),flush=True)
if __name__=='__main__':main()
'''
installer=template.replace('__PAYLOAD__',repr(encoded)).encode('utf-8')
OUT.mkdir(exist_ok=False)
(OUT/'protocol.json').write_bytes(new_body)
name='install_cctv_completion_codeformer_finetune_v1_r1.py'
(OUT/name).write_bytes(installer)
installer_sha=digest(installer)
(OUT/(name+'.sha256')).write_bytes((installer_sha+'  '+name+'\n').encode('ascii'))
summary={'complete':True,'installer':name,'bytes':len(installer),'sha256':installer_sha,'protocol_sha256':new_sha,'files':len(new['files']),'payload_bytes':sum(r['bytes'] for r in new['files']),'data_and_contract_unchanged':True,'manual_training_only':True,'parent_evidence':parent_evidence,'local_neural_calls':0,'optimizer_updates':0}
(OUT/'build_receipt.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))