"""Bounded evidence export for the distinct manual CodeFormer spatial-code-intervention-v1 pilot.
No model imports, neural work, cleanup, overwrite or automatic training action.
Ordinary success writes no checkpoint; unexpected mutation failure may retain a
bounded actual state/partial, never an adopted or automatically resumable model.
"""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,os,shutil,stat,tarfile,time
from pathlib import Path,PurePosixPath
STEM='cctv-completion-codeformer-spatial-code-intervention-v1'
VM_NAME='cctv_completion_codeformer_spatial_code_intervention_v1_vm'
MEMBER_ROOT='cctv_completion_codeformer_spatial_code_intervention_v1_results'
MAX_BYTES=1280*1024**2
OUTPUT_BYTES=1024**3
MAX_SECONDS=900
STOP_FREE=2*1024**3

def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path,guard=None):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            if guard:guard()
            digest.update(block)
    return digest.hexdigest()
def read(path):
    def reject(x):raise ValueError('NonfiniteJSON '+x)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def regular(path):
    info=path.lstat();need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'Only regular evidence files may be exported');return info
def member(root,relative):
    need(isinstance(relative,str) and relative and '\\' not in relative and ':' not in relative,'Invalid member spelling')
    name=PurePosixPath(relative);need(not name.is_absolute() and '..' not in name.parts and str(name)==relative and len(relative.encode('utf-8'))<=240,'Member traversal/noncanonical spelling/length')
    path=root.joinpath(*name.parts)
    for item in [path,*path.parents]:
        need(not item.is_symlink(),'Evidence symlink refused')
        if item==root:break
    need(path.resolve().is_relative_to(root.resolve()),'Evidence escaped root');return path

def allowed_outputs(protocol):
    names={'outputs/'+name for name in ('state_before.json','oracle_label_receipt.json','reused_control_binding.json','external_control_compatibility.json','arm_H_complete.json','arm_V_complete.json','state_after.json','results.json','failure.json','state_on_failure.json','state_receipt_failure.json','failed_actual_model_state.pth','failed_actual_model_state.pth.partial','failure_state_receipt.json')}
    for folder,cases in [('fresh_app_parity',protocol['parity_cases']),('external_bridge',protocol['bridge_cases'])]:
        names.add('outputs/'+folder+'/summary.json')
        for case in cases:
            for suffix in ('.npz','.json'):names.add('outputs/'+folder+'/'+case['id']+suffix)
    for arm in protocol['arms']:
        for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
            prefix='outputs/'+arm['filename_tag']+'/'+kind+'/'
            names.add(prefix+'results.json')
            for case in protocol[key]:
                for suffix in ('.npz','.png'):names.add(prefix+case['id']+'__ORACLE_unavailable_clean_information'+suffix)
    need(protocol['export']['allowed_output_paths']==sorted(names),'Exact closed prospective output namespace differs')
    return names
class CountedWriter:
    def __init__(self,stream,guard,*,disk_directory=None):self.stream=stream;self.guard=guard;self.disk_directory=disk_directory;self.count=0
    def write(self,data):
        self.guard();need(self.count+len(data)<=MAX_BYTES,'Raw/compressed return exceeds1280MiB')
        if self.disk_directory is not None:need(shutil.disk_usage(self.disk_directory).free-len(data)>=STOP_FREE,'Preserve2GiB free during export')
        result=self.stream.write(data);need(result==len(data),'Short evidence archive write');self.count+=result;return result
    def flush(self):return self.stream.flush()

def export(root,expected,home,*,enforce_vm=True):
    root=Path(root).resolve(strict=True);home=Path(home).resolve(strict=True);started=time.monotonic()
    if enforce_vm:need(os.name=='posix' and root==(home/'forensic-dgp'/VM_NAME).resolve(),'Export only named manual VM root')
    def guard():need(time.monotonic()-started<MAX_SECONDS,'Export900second cooperative stop')
    need(shutil.disk_usage(home).free>=STOP_FREE,'Need2GiB export free-space floor')
    ppath=member(root,'protocol.json');regular(ppath);need(sha(ppath,guard)==expected,'Protocol SHA differs')
    protocol=read(ppath);need(protocol.get('format')=='completion-codeformer-spatial-code-intervention-v1','Wrong pilot protocol')
    need(isinstance(protocol.get('files'),list) and len(protocol['files'])>0,'Empty source manifest refused')
    need(protocol.get('manual_execution_only') is True and protocol.get('automatic_resume_or_promotion') is False,'Manual-only protocol differs')
    contract=protocol.get('contract',{});need(contract.get('output_bytes')==OUTPUT_BYTES and contract.get('stop_free_bytes')==STOP_FREE,'Export/output/disk contract differs')
    permitted=allowed_outputs(protocol)
    permitted_directories={PurePosixPath(name).parent.as_posix() for name in permitted}
    permitted_directories|={parent.as_posix() for name in permitted for parent in PurePosixPath(name).parents if parent.as_posix()!='.'}
    seen=set()
    for record in protocol['files']:
        guard();relative=record['path'];need(relative not in seen,'Duplicate source manifest member');seen.add(relative)
        path=member(root,relative);info=regular(path);need(info.st_size==record['bytes'] and sha(path,guard)==record['sha256'],'Installed source/data drift '+relative)
    files=[]
    for name in ('protocol.json','supervisor.json','trainer.log'):
        path=member(root,name);regular(path);files.append(path)
    supervisor=read(root/'supervisor.json');need(supervisor.get('complete') is True and type(supervisor.get('worker_exit')) is int,'Terminal supervisor receipt required')
    output=member(root,'outputs')
    if output.exists():
        need(output.is_dir() and not output.is_symlink(),'Invalid output root')
        for path in sorted(output.rglob('*')):
            guard();need(not path.is_symlink(),'Output symlink refused');need(path.resolve().is_relative_to(output),'Output escaped evidence root')
            relative=path.relative_to(root).as_posix()
            if path.is_dir():need(relative in permitted_directories,'Unlisted output directory');continue
            need(relative in permitted,'Unlisted output evidence '+relative)
            regular(path);files.append(path)
    need(len(files)>=3 and len(set(files))==len(files),'Empty/duplicate return manifest refused')
    need(len(files)+1<=400,'Return regular member cap400 including manifest')
    state_path=output/'failed_actual_model_state.pth';state_partial=Path(str(state_path)+'.partial');aliases=[]
    if state_path in files and state_partial in files:
        a,b=regular(state_path),regular(state_partial)
        need((a.st_dev,a.st_ino,a.st_uid,a.st_size)==(b.st_dev,b.st_ino,b.st_uid,b.st_size) and a.st_nlink>=2 and b.st_nlink>=2 and sha(state_path,guard)==sha(state_partial,guard),'Distinct failed-state pair refused; retain VM evidence')
        aliases.append({'path':state_partial.relative_to(root).as_posix(),'committed_path':state_path.relative_to(root).as_posix(),'same_dev_inode_uid_size_hash':True,'dev':a.st_dev,'inode':a.st_ino,'uid':a.st_uid,'nlink':a.st_nlink,'bytes':a.st_size,'sha256':sha(state_partial,guard),'source_alias_retained':True,'payload_exported_once':True})
        files.remove(state_partial)
    checkpoints=[path for path in files if any(s.lower() in ('.pth','.pt') for s in path.suffixes)]
    need(all(path in (state_path,state_partial) for path in checkpoints),'Only named actual failure state allowed')
    need(not checkpoints or supervisor['worker_exit']!=0,'Successful frozen diagnostic must not return checkpoint/partial')
    need(sum(regular(path).st_size for path in checkpoints)<=384*1024**2,'Cumulative actual failed-state/partial cap384MiB')
    need(sum(regular(path).st_size for path in files if path.is_relative_to(output))<=OUTPUT_BYTES,'Whole output cap1GiB')
    need(sum(path.suffix.lower()=='.png' for path in files)<=96,'Diagnostic PNG member cap96')
    need(sum('.npz' in [suffix.lower() for suffix in path.suffixes] for path in files)<=104,'Diagnostic NPZ member cap104')
    need(all(len(path.relative_to(root).as_posix().encode('utf-8'))<=240 for path in files),'Bounded return path length240')
    results=output/'results.json'
    if supervisor['worker_exit'] in (0,2):
        need(results.is_file(),'Completed diagnostic needs final results')
        summary=read(results);need(summary.get('complete') is True and summary.get('protocol_sha256')==expected and summary.get('model_qualified') is False and summary.get('diagnostic_only') is True and summary.get('new_model_created') is False,'Results/protocol diagnostic scope differs')
    entries=[]
    for path in files:
        guard();before=regular(path);digest=sha(path,guard);after=regular(path)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Evidence changed while hashing')
        entries.append({'path':path.relative_to(root).as_posix(),'bytes':after.st_size,'sha256':digest})
    payload_bytes=sum(row['bytes'] for row in entries);need(payload_bytes<=MAX_BYTES,'Uncompressed payload exceeds1280MiB')
    partials=[r['path'] for r in entries if r['path'].endswith('.partial')]+[a['path'] for a in aliases];committed=[row for row in entries if row['path']=='outputs/failed_actual_model_state.pth']
    manifest={'format':'completion-codeformer-spatial-code-intervention-v1-return-manifest-v1','protocol_sha256':expected,'files':entries,'worker_exit':supervisor['worker_exit'],'worker_timeout':supervisor.get('timeout'),'worker_interruption':supervisor.get('interruption'),'terminal_state_committed':bool(committed),'terminal_state_pin':committed[0] if committed else None,'partial_files_retained':partials,'retained_teacher_label_arrays':len([r for r in entries if r['path'].startswith('outputs/teacher_labels/') and r['path'].endswith('.npz')]),'original_base_weights_exported':False,'parent_training_state_exported':False,'state_recoverability':'No checkpoint or optimizer is created by this frozen-state diagnostic; independent arrays/state-invariance audit remains required.','diagnostic_only':True,'new_model_created':False,'oracle_information_is_not_deployable':True,'model_qualified':False,'training_success_not_implied':True,'raw_and_compressed_cap_bytes':MAX_BYTES,'export_cooperative_seconds':MAX_SECONDS,'minimum_free_bytes':STOP_FREE}
    manifest.update({'same_inode_failed_state_aliases':aliases,'retained_teacher_label_arrays':0,'whole_output_cap_bytes':OUTPUT_BYTES,'state_recoverability':'Ordinary frozen success writes no checkpoint. Unexpected mutation failure may retain one bounded actual497state+RNG or partial without optimizer, never qualified or automatically resumable. Independent arrays/state-invariance audit remains required.'})
    body=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode('utf-8');need(len(body)<=2*1024**2,'Return manifest metadata cap')
    # PAX filenames, blocks and end padding are counted again by streaming writer.
    prebound=sum(((row['bytes']+511)//512)*512+2048 for row in entries)+((len(body)+511)//512)*512+2048+10240
    need(prebound<=MAX_BYTES,'Payload plus conservative tar/header bound exceeds1280MiB')
    archive=home/(STEM+'-results.tar.gz');partial=Path(str(archive)+'.partial');checksum=Path(str(archive)+'.sha256');receipt_path=home/(STEM+'-export.json')
    for path in (archive,partial,checksum,receipt_path):need(not path.exists(),'Preserve existing export; refusing overwrite')
    guard();raw_count=0;compressed_count=0
    try:
        with partial.open('xb') as raw:
            compressed=CountedWriter(raw,guard,disk_directory=home)
            with gzip.GzipFile(filename='',mode='wb',fileobj=compressed,compresslevel=1,mtime=0) as zipped:
                uncompressed=CountedWriter(zipped,guard)
                with tarfile.open(fileobj=uncompressed,mode='w|',bufsize=1024*1024,format=tarfile.PAX_FORMAT) as tar:
                    for path,row in zip(files,entries):
                        guard();need(regular(path).st_size==row['bytes'] and sha(path,guard)==row['sha256'],'Evidence changed before tar member')
                        info=tarfile.TarInfo(MEMBER_ROOT+'/'+row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                        with path.open('rb') as stream:tar.addfile(info,stream)
                        need(sha(path,guard)==row['sha256'],'Evidence changed during tar member')
                    info=tarfile.TarInfo(MEMBER_ROOT+'/export_manifest.json');info.size=len(body);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(body))
                raw_count=uncompressed.count
            compressed_count=compressed.count;raw.flush();os.fsync(raw.fileno())
        guard();need(raw_count<=MAX_BYTES and compressed_count==partial.stat().st_size<=MAX_BYTES,'Final raw/compressed cap')
        digest=sha(partial,guard);need(shutil.disk_usage(home).free>=STOP_FREE,'Export free space below2GiB');guard();need(not archive.exists(),'Archive appeared before publication')
        # Same-directory exclusive hardlink publication avoids replacing a raced target.
        os.link(partial,archive);need(archive.stat().st_ino==partial.stat().st_ino,'Archive publication identity differs')
        # The successful.partial is retained as a hardlink, not a second payload copy.
        with checksum.open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
        receipt={'format':'completion-codeformer-spatial-code-intervention-v1-export-receipt-v1','complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,'raw_tar_bytes':raw_count,'payload_bytes':payload_bytes,'seconds':time.monotonic()-started,'protocol_sha256':expected,'worker_exit':supervisor['worker_exit'],'terminal_state_committed':bool(committed),'partial_files_retained':partials,'successful_archive_partial_hardlink_retained':str(partial),'diagnostic_arrays_state_invariance_pending_independent_audit':True,'diagnostic_only':True,'new_model_created':False,'training_success_not_implied':True,'model_qualified':False,'payload_files':len(entries),'export_only_neural_calls':0}
        receipt['same_inode_failed_state_aliases']=aliases
        with receipt_path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps(receipt),flush=True);return receipt
    except BaseException as exc:
        failure=home/(STEM+'-export-failure.json')
        if not failure.exists():
            with failure.open('x',encoding='utf-8',newline='\n') as stream:json.dump({'complete':False,'error':repr(exc),'partial_archive':str(partial),'partial_retained':partial.exists(),'seconds':time.monotonic()-started,'raw_tar_bytes_observed':raw_count,'compressed_bytes_observed':partial.stat().st_size if partial.exists() else 0,'training_success_not_implied':True},stream,indent=2);stream.write('\n')
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args();export(args.root,args.protocol_sha,Path.home())
if __name__=='__main__':main()
