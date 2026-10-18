"""Freeze an independently reviewed AdaIN-isolation overlay and transfer TAR.

Pure local source/metadata/hash work. Never connects, executes old packets,
constructs a model, installs dependencies or starts the manual training pilot.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import prepare_codeformer_adain_gradient_isolation_v1 as prep

MAX_UPLOAD_BYTES=32*1024**2
MAX_UPLOAD_PAYLOAD_BYTES=32*1024**2
MAX_FILES=1000
MAX_SECONDS=300


def archive_readback(file,packet,expected):
    prep.require(file.stat().st_size<=MAX_UPLOAD_BYTES and 0<len(expected)<=MAX_FILES,'Finite upload cap exceeded')
    seen=set();folded=set();total=0
    with tarfile.open(file,'r:gz') as bundle:
        for member in bundle:
            name=prep.canonical(member.name);parts=PurePosixPath(name).parts
            prep.require(member.type in (tarfile.REGTYPE,tarfile.AREGTYPE) and member.isfile() and member.sparse is None and
                not member.linkname and not any(k.startswith('GNU.sparse') for k in member.pax_headers),'Regular nonsparse upload members only')
            prep.require(len(parts)>1 and parts[0]==prep.PACKAGE,'Exact package installation prefix required')
            name=PurePosixPath(*parts[1:]).as_posix()
            prep.require(name in expected and name not in seen and name.casefold() not in folded,'Unexpected/duplicate/case-colliding upload member')
            source=prep.bounded(packet,name)
            prep.require(type(member.size) is int and member.size==source.stat().st_size,'Upload member size differs')
            total+=member.size;prep.require(total<=MAX_UPLOAD_PAYLOAD_BYTES,'Upload payload cap exceeded')
            h=hashlib.sha256();count=0
            with bundle.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk);count+=len(chunk)
            prep.require(count==member.size and h.hexdigest()==expected[name],'Upload member hash differs')
            seen.add(name);folded.add(name.casefold())
    prep.require(seen==set(expected),'Missing upload member')
    return {'members':len(seen),'payload_bytes':total,'all_member_hashes_and_sizes_exact':True,
            'exact_namespace':True,'returned_or_packet_code_executed':False}


def resource_proof(record,p):
    prep.require(record.get('complete') is True and record.get('approved') is True,'Independent complete resource approval required')
    ledger=record.get('ledger',{})
    for field,limit in [('conservative_payload_bytes','result_payload_bytes'),('conservative_tar_gzip_upper_bytes','archive_bytes')]:
        number=ledger.get(field);prep.require(type(number) is int and 0<number<=p['limits'][limit],'Finite resource proof differs: '+field)
    files=ledger.get('planned_file_count_upper')
    prep.require(type(files) is int and 0<files<=7000 and ledger.get('regular_file_count_cap')==7000 and
        ledger.get('payload_cap_bytes')==p['limits']['result_payload_bytes'] and
        ledger.get('archive_cap_bytes')==p['limits']['archive_bytes'] and
        ledger.get('atomic_checkpoint_temporary_bytes')==p['limits']['checkpoint_temporary_bytes'] and
        ledger.get('startup_free_bytes_after_install')==p['limits']['initial_free_bytes'] and
        ledger.get('reserve_bytes')==p['limits']['reserve_free_bytes'] and
        ledger.get('extra_safe_checkpoint_copy_allowed') is False and
        ledger.get('last_safe_checkpoint_must_be_a_reference_not_fourth_serialized_file') is True,
        'Checkpoint/file/disk reservation proof differs')


class FiniteArchive:
    def __init__(self,stream,deadline):self.stream,self.deadline,self.bytes=stream,deadline,0
    def write(self,value):
        prep.require(time.monotonic()<=self.deadline and self.bytes+len(value)<=MAX_UPLOAD_BYTES,'Finite sealing time/byte cap exceeded')
        count=self.stream.write(value);self.bytes+=count;return count
    def flush(self):self.stream.flush()


def freeze_and_seal(root,readiness_sha):
    root=prep.no_links(root).resolve(strict=True);evidence=root/prep.EVIDENCE;packet=root/'outputs'/prep.PACKAGE
    prep.require(isinstance(readiness_sha,str) and prep.HEX.fullmatch(readiness_sha),'Exact independent source approval SHA required')
    ready_path=evidence/'source_readiness.json'
    prep.require(prep.sha(ready_path)==readiness_sha,'Source readiness changed')
    ready=prep.read_json(ready_path)
    prep.require(ready.get('complete') is True and ready.get('approved_for_protocol_freeze') is True and
        ready.get('training_authorized') is False,'Independent source approval does not close preparation')
    for name,digest in ready.get('review_evidence_sha256',{}).items():
        prep.require(prep.sha(prep.bounded(root,name))==digest,'Independent approval input changed: '+name)
    stage=prep.read_json(evidence/'stage_inputs.json');p=prep.read_json(evidence/'protocol_template_unsealed.json')
    prep.require(stage['complete'] is True and stage['status']=='STAGED / UNSEALED / UNRUN' and
        not (packet/'protocol.json').exists(),'Preserve previous/partial protocol freeze')
    overlay=stage['files_sha256'];dependencies=stage['dependency_files_sha256']
    prep.require(prep.exact_files(packet)==set(overlay) and p['files_sha256']=={**dependencies,**overlay},'Staged namespace changed')
    prep.require(prep.sha(root/'outputs'/prep.PARENT/'protocol.json')==prep.PARENT_PROTOCOL_SHA,'FP32 parent protocol changed')
    parent=prep.read_json(root/'outputs'/prep.PARENT/'protocol.json');resolve=prep.local_parent_resolver(root,parent)
    prep.require(set(dependencies)==set(prep.dependency_names(parent)),'Frozen dependency allowlist differs')
    for name,digest in p['files_sha256'].items():
        source=resolve(name) if name in dependencies else prep.bounded(packet,name)
        prep.require(prep.sha(source)==digest,'Pre-freeze dependency/overlay file changed: '+name)
    sources=ready.get('source_sha256',{});required={
        'research/codeformer_adain_gradient_isolation_v1/worker.py',
        'research/codeformer_adain_gradient_isolation_v1/gradient_route.py',
        'scripts/prepare_codeformer_adain_gradient_isolation_v1.py',
        'scripts/install_codeformer_adain_gradient_isolation_v1.py',
        'scripts/seal_codeformer_adain_gradient_isolation_v1.py'}
    prep.require(isinstance(sources,dict) and required<=set(sources),'Complete new source approval map required')
    additions={}
    for name,digest in sources.items():
        source=prep.bounded(root,name);prep.require(prep.sha(source)==digest,'Reviewed current source changed')
        if name in p['files_sha256']:prep.require(p['files_sha256'][name]==digest,'Reviewed source differs from staged bytes')
        else:
            prep.require(name not in dependencies and not name.startswith('outputs/'),'Unsafe/colliding reviewed source addition')
            additions[name]=source
    data_path=evidence/'independent_data_resource_v1.json';data=prep.read_json(data_path)
    prep.require(data.get('complete') is True and data.get('approved_for_protocol_freeze') is True and
        data.get('training_authorized') is False and data.get('stage_inputs_sha256')==prep.sha(evidence/'stage_inputs.json') and
        data.get('protocol_template_sha256')==prep.sha(evidence/'protocol_template_unsealed.json'),
        'Complete independent staged data/resource readback required')
    ledger_path=evidence/'final_resource_ledger.json';ledger=prep.read_json(ledger_path);resource_proof(ledger,p)
    additions.update({'source_readiness.json':ready_path,'provenance/resource_ledger.json':ledger_path,
                      'provenance/independent_input_readback.json':data_path})
    for name,source in additions.items():
        prep.require(name not in p['files_sha256'] and not prep.bounded(packet,name,exists=False).exists(),'Never overwrite staged/readiness files')
    for name,source in additions.items():
        prep.put(packet/name,source.read_bytes());digest=prep.sha(source)
        prep.require(prep.sha(packet/name)==digest,'Readiness copy changed');p['files_sha256'][name]=digest;p['export_files'].append(name)
    p['source_readiness']={'path':'source_readiness.json','sha256':readiness_sha}
    p['resource_ledger']={'path':'provenance/resource_ledger.json','sha256':prep.sha(ledger_path)}
    p['export_files']=sorted(set(p['export_files']));p['provenance']['unsealed_stage_only']=False
    from research.codeformer_adain_gradient_isolation_v1 import worker
    worker.validate_protocol(p)
    prep.write_json(packet/'protocol.json',p);protocol_sha=prep.sha(packet/'protocol.json')
    _,static=worker.verify_transfer(packet,protocol_sha,dependency_root=root/'outputs'/prep.PARENT,
                                   legacy_dependency_root=root/'outputs'/prep.V2_PARENT)
    prep.require(static['model_calls']==static['gradient_queries']==static['optimizer_updates']==0,'Static verification performed neural work')
    expected={name:digest for name,digest in p['files_sha256'].items() if name not in dependencies};expected['protocol.json']=protocol_sha
    prep.require(prep.exact_files(packet)==set(expected) and len(expected)<=MAX_FILES,'Final upload overlay namespace differs')
    prep.require(sum((packet/name).stat().st_size for name in expected)<=MAX_UPLOAD_PAYLOAD_BYTES,'Overlay payload cap exceeded')
    archive=root/'outputs'/(prep.PREFIX+'-execution.tar.gz');temporary=Path(str(archive)+'.writing')
    checksum=Path(str(archive)+'.sha256');receipt=evidence/'execution_archive_v1.json'
    for path in (archive,temporary,checksum,receipt):prep.no_links(path);prep.require(not path.exists(),'Retain previous/partial seal artifact')
    started=time.monotonic()
    with temporary.open('xb') as target:
        bounded=FiniteArchive(target,started+MAX_SECONDS)
        with gzip.GzipFile(filename='',mode='wb',fileobj=bounded,mtime=0,compresslevel=6) as gz:
            with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(expected):
                    source=prep.bounded(packet,name);prep.require(prep.sha(source)==expected[name],'Frozen source changed while archiving')
                    info=tar.gettarinfo(str(source),arcname=prep.PACKAGE+'/'+name)
                    prep.require(info.isreg(),'Only regular upload files');info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644;info.pax_headers={}
                    with source.open('rb') as inp:tar.addfile(info,inp)
        target.flush();os.fsync(target.fileno())
    readback=archive_readback(temporary,packet,expected)
    temporary.rename(archive);archive_sha=prep.sha(archive)
    with checksum.open('x',encoding='ascii',newline='\n') as out:
        out.write(archive_sha+'  '+archive.name+'\n');out.flush();os.fsync(out.fileno())
    prep.require(b'\r' not in checksum.read_bytes(),'Exact portable LF sidecar required')
    result={**readback,'complete':True,'status':'VERIFIED / PREPARED / UNRUN','protocol_sha256':protocol_sha,
        'archive':archive.relative_to(root).as_posix(),'archive_sha256':archive_sha,'bytes':archive.stat().st_size,
        'checksum_LF':True,'static_transfer_readback':static,'dependency_files_not_reuploaded':len(dependencies),
        'dependency_copy_bytes':sum(p['dependency_files_bytes'].values()),'source_readiness_sha256':readiness_sha,
        'resource_ledger_sha256':prep.sha(ledger_path),'sealer_source_sha256':prep.sha(__file__),
        'local_neural_gradient_optimizer_VM_calls':0,'training_authorized':False,'model_qualified':False,
        'manual_launch_required':True,'seconds':time.monotonic()-started}
    prep.write_json(receipt,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--readiness-sha',required=True)
    args=parser.parse_args();print(json.dumps(freeze_and_seal(ROOT,args.readiness_sha),indent=2))
