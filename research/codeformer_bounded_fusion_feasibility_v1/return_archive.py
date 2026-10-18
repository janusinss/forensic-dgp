"""Strict stdlib transportation of one frozen bounded-fusion diagnostic return.

No I/O on import; no returned code, arrays, model, shell or network is executed.
Full streaming verification precedes exclusive import; failed/partial evidence
is retained. Completed transportation is never restoration qualification.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time

PROTOCOL_SHA256 = 'c3b0f5f0f3ff83b508625df0259ed0c9acbc2d071da2fdde066a2bda429a15eb'
ARCHIVE_SHA256 = '3e1a0a4bee698b397747ba39d3546cbca3cbd298e9cb450bc1abfc61aa880bb9'
ARCHIVE_BYTES = 668557079
MANIFEST_SHA256 = 'c9286d80ac6586aa281aa91bb00b6b430521b2d797f79308f450559c244548b3'
OUTPUTS = 'outputs/bounded_fusion_feasibility_v1'
MANIFEST_NAME = OUTPUTS + '/export_manifest.json'
PREFIX = 'cctv-codeformer-bounded-fusion-feasibility-v1'
RESULT_ARCHIVE = PREFIX + '-results.tar.gz'
EXPORT_RECEIPT = PREFIX + '-export.json'
CALL_CAPS = {'codeformer_forwards':170,'fusion_block_evaluations':960,'hq_teacher_forwards':0,
    'behavior_teacher_forwards':0,'lpips_forwards':0,'gradient_queries':0,'backward_calls':0,
    'optimizer_attempts':0,'optimizer_updates':0}
LIMITS = {'initialization_seconds':300,'measurement_seconds':600,'worker_seconds':900,
    'export_seconds':300,'initial_free_bytes':8*1024**3,'reserve_free_bytes':4*1024**3,
    'result_bytes':5*1024**3//4,'archive_bytes':1449551462,'metadata_bytes':32*1024**2,
    'regular_files':2500,'temporary_bytes':128*1024**2,'max_reserved_GPU_bytes':22*1024**3,
    'PNG_file_bytes':198656,'array_file_bytes':4198400}
ASSETS = {
    'original':{'parent_dir':'cctv_codeformer_fp32_comparison_v1_vm','path':'weights/codeformer.pth',
        'sha256':'1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7','bytes':376637898,
        'parent_protocol_sha256':'8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'},
    'stopped':{'parent_dir':'cctv_codeformer_fp32_comparison_v1_vm','path':'outputs/fp32_comparison_v1/stopped_checkpoint.pt',
        'sha256':'133c5a1e15e069bdd2abda4ebf818d3f4ff59b966677762753e894ff42e2583d','bytes':855229944,
        'parent_protocol_sha256':'8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'}}
OPERATIONAL = {'outputs/supervision/preflight.log','outputs/supervision/diagnostic.log','outputs/supervision/run_receipt.json'}
MODES = ('original','stopped','rollback','bounded_constant','bounded_fixed')
PATH_FILES = ('actual_tokens.npy','fusion_stats.json','native.png','native512.npy','raw256.npy','record.json')
CONTROL = {'calibration.json','initialization.json','manual_launch.json','run_outputs.json','runtime.json',
    'worker_complete.json','state_checks/initial.json','state_checks/terminal.json'}
FAILED_CONTROL = {'worker_failure.json','neural_failure.json','state_checks/failure.json'}
HEX = re.compile(r'^[a-f0-9]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)',re.I)
METADATA = {'.json','.jsonl','.log','.txt','.md','.py','.sh','.sha256','.csv','.yaml','.yml'}
RESERVE_BYTES = 1024**3
MAX_SECONDS = 900


def require(test, message):
    if not test: raise ValueError(message)


def canonical(name):
    require(isinstance(name,str) and bool(name),'Empty member path')
    p=PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix()==name and all(x not in ('.','..','') for x in p.parts),
            'Noncanonical or escaped path')
    require(not any(ch in name for ch in '\\:"<>|?*') and all(not DEVICE.match(x) and not x.endswith((' ','.')) and
        len(x.encode('utf-16-le'))<=510 and all(ord(ch)>=32 for ch in x) for x in p.parts),'Nonportable path')
    return name


def no_links(path):
    path=Path(path).absolute()
    for p in (path,*path.parents):
        if p.exists() or p.is_symlink():
            require(not p.is_symlink() and not getattr(p.lstat(),'st_file_attributes',0)&1024,'Link/reparse path forbidden')
    return path


def regular(path):
    path=no_links(path); info=path.lstat()
    require(stat.S_ISREG(info.st_mode),'Regular file required: '+str(path)); return info


def identity(info):
    return info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_ino,info.st_dev


def object_identity(info):
    return info.st_size,info.st_mtime_ns,info.st_ino,info.st_dev


def sha(path,budget=lambda:None):
    before=regular(path); digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024**2),b''): budget();digest.update(block)
    require(identity(regular(path))==identity(before),'File changed while hashing');return digest.hexdigest()


def parse(payload):
    def pairs(rows):
        result={}
        for key,value in rows:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    value=json.loads(payload.decode('utf-8-sig'),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    def finite(x):
        if isinstance(x,float):require(math.isfinite(x),'Overflowed/nonfinite JSON number')
        elif isinstance(x,dict):
            for item in x.values():finite(item)
        elif isinstance(x,list):
            for item in x:finite(item)
    finite(value);return value


def read(path,cap=8*1024**2):
    before=regular(path);require(before.st_size<=cap,'Metadata byte cap exceeded')
    payload=Path(path).read_bytes();require(len(payload)==before.st_size and identity(regular(path))==identity(before),'Metadata changed')
    return parse(payload)


def write_new(path,value):
    path=no_links(path)
    with path.open('x',encoding='utf-8',newline='\n') as output:
        json.dump(value,output,indent=2,allow_nan=False);output.write('\n');output.flush();os.fsync(output.fileno())


def exact(value,expected):
    if type(value) is not type(expected):return False
    if isinstance(expected,dict):return value.keys()==expected.keys() and all(exact(value[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(value)==len(expected) and all(exact(a,b) for a,b in zip(value,expected))
    return value==expected


def namespace(names):
    folded=set()
    for name in names:
        canonical(name);require(name.casefold() not in folded,'Duplicate/case-colliding member');folded.add(name.casefold())
    for name in folded:
        parent=PurePosixPath(name).parent
        while parent.parts:
            require(parent.as_posix() not in folded,'File/directory namespace collision');parent=parent.parent


def output_namespace(protocol):
    names={OUTPUTS+'/'+name for name in CONTROL}
    plan=[(row['id'],mode,'paths') for row in protocol['cases'] for mode in MODES]
    plan += [(case,mode,'repeats') for case in protocol['canary_ids'] for mode in ('original','stopped')]
    plan += [(case,mode,'zero_delta') for case in protocol['canary_ids'] for mode in ('bounded_constant','bounded_fixed')]
    plan += [(case,'native_identity','identity') for case in protocol['canary_ids']]
    for case,mode,section in plan:
        directory=OUTPUTS+'/'+section+'/'+case+'/'+mode
        names.update(directory+'/'+file for file in PATH_FILES)
        if section=='paths' and case in protocol['canary_ids'] and mode.startswith('bounded_'):
            names.update(directory+'/features/'+site+'/'+role+'.npy' for site in ('32','64')
                         for role in ('original','stopped','used_delta'))
    names.update(OUTPUTS+'/call_trace/%04d.json'%index for index in range(1,sum(CALL_CAPS.values())+1))
    return names


def status(export,manifest,protocol,records,jsons):
    require(manifest.get('format')=='codeformer-bounded-fusion-feasibility-return-v1' and
        manifest.get('protocol_sha256')==PROTOCOL_SHA256,'Manifest format/protocol differs')
    for flag in ('model_qualified','native_CCTV_qualification','automatic_training'):
        require(manifest.get(flag) is False,'Transport cannot assert qualification/training')
    require(type(manifest.get('optimizer_updates')) is int and manifest['optimizer_updates']==0,'Manifest optimizer boundary differs')
    require(exact(manifest.get('parent_assets'),protocol['assets']),'Manifest parent asset bindings differ')
    expected=manifest.get('files_sha256');sizes=manifest.get('files_bytes')
    require(isinstance(expected,dict) and isinstance(sizes,dict) and set(expected)==set(sizes) and
            MANIFEST_NAME not in expected,'Manifest exact hash/byte namespace required')
    namespace(expected)
    require(set(records)==set(expected)|{MANIFEST_NAME},'Unlisted/missing returned members')
    for name,digest in expected.items():
        require(isinstance(digest,str) and HEX.fullmatch(digest) and type(sizes[name]) is int and sizes[name]>=0 and
                records[name]=={'sha256':digest,'bytes':sizes[name]},'Manifest member bytes/hash differ: '+name)
    mandatory={'protocol.json',*protocol['export_files'],*OPERATIONAL}
    require(mandatory<=set(expected),'Mandatory frozen source/operational companion missing')
    allowed=mandatory|output_namespace(protocol)|{OUTPUTS+'/'+n for n in FAILED_CONTROL}
    require(set(expected)<=allowed,'Unbound output/source namespace')
    for name in mandatory-OPERATIONAL:
        want=PROTOCOL_SHA256 if name=='protocol.json' else protocol['files_sha256'][name]
        require(expected[name]==want,'Returned immutable source/protocol differs')
    retained={name:digest for name,digest in protocol['files_sha256'].items() if name not in expected}
    require(exact(manifest.get('retained_not_exported_sha256'),retained),'Retained immutable input/source namespace differs')
    completed,failed=manifest.get('complete_measurement'),manifest.get('failure_present')
    require(type(completed) is type(failed) is bool and export['measurement_complete']==completed and
            export['failure_present']==failed,'Manifest/export status differs')
    require((OUTPUTS+'/worker_complete.json' in records)==completed and
            (OUTPUTS+'/worker_failure.json' in records)==failed and not(completed and failed),'Worker terminal status differs')
    if completed:
        require(set(expected)==mandatory|output_namespace(protocol),'Completed measurement does not have exact full output census')
    supervision=jsons['outputs/supervision/run_receipt.json']
    require(supervision.get('protocol_sha256')==PROTOCOL_SHA256 and supervision.get('manual_tmux') is True and
            supervision.get('no_automatic_retry') is True and supervision.get('training_success_not_implied') is True and
            supervision.get('model_qualified') is False and supervision.get('automatic_app_promotion') is False and
            type(supervision.get('optimizer_updates')) is int and supervision['optimizer_updates']==0,
            'Closed supervisor receipt boundary differs')
    stages=supervision.get('stages')
    require(isinstance(stages,list) and len(stages)==2,'Two closed preflight/measurement stages required')
    for row,flag,cap,grace,log in zip(stages,('--verify-transfer','--run'),(300,900),(30,60),
            ('outputs/supervision/preflight.log','outputs/supervision/diagnostic.log')):
        require(row.get('stage')==flag and row.get('external_cap_seconds')==cap and row.get('grace_seconds')==grace and
                row.get('log')==log and type(row.get('exit_code')) is int and type(row.get('log_capped')) is bool and
                type(row.get('seconds')) in (int,float) and 0<=row['seconds']<=cap+grace+15,'Finite stage binding differs')
    require(stages[0]['exit_code']==0 and supervision.get('neural_measurement_complete')==(stages[1]['exit_code']==0) and
            completed==(stages[1]['exit_code']==0),'Supervisor and completed-worker status disagree')
    launch=jsons[OUTPUTS+'/manual_launch.json']
    require(launch.get('protocol_sha256')==PROTOCOL_SHA256 and launch.get('manual_tmux') is True and
            type(launch.get('pid')) is int and launch['pid']>0 and launch.get('automatic_training') is False and
            type(launch.get('optimizer_updates')) is int and launch['optimizer_updates']==0,'Manual worker binding differs')
    if completed:
        worker=jsons[OUTPUTS+'/worker_complete.json']
        require(worker.get('complete') is True and worker.get('protocol_sha256')==PROTOCOL_SHA256 and
            worker.get('pid')==launch['pid'] and worker.get('model_qualified') is False and
            worker.get('next_training_authorized') is False and worker.get('automatic_app_promotion') is False and
            exact(worker.get('counts'),CALL_CAPS),'Completed worker declared finite status differs')
    elif failed:
        worker=jsons[OUTPUTS+'/worker_failure.json']
        require(worker.get('complete') is False and worker.get('protocol_sha256')==PROTOCOL_SHA256 and
                worker.get('model_qualified') is False,'Failed worker status differs')
    total=sum(row['bytes'] for row in records.values())
    metadata=sum(row['bytes'] for name,row in records.items() if PurePosixPath(name).suffix.lower() in METADATA or
                 PurePosixPath(name).name in {'LICENSE','NOTICE','COPYING'})
    require(len(records)<=LIMITS['regular_files'] and total<=LIMITS['result_bytes'] and metadata<=LIMITS['metadata_bytes'],
            'Return metadata/file/payload caps exceeded')
    return {'reported_measurement_complete':completed,'failure_present':failed,'payload_bytes':total,
            'metadata_bytes':metadata,'regular_files':len(records),'optimizer_updates':0,
            'model_qualified':False,'independently_recomputed_neural_calls':False}


def inspect_and_import(archive,sidecar,export_path,packet,destination,receipt_path,*,parent_packet,original_asset,stopped_asset):
    paths=map(lambda p:Path(p).absolute(),(archive,sidecar,export_path,packet,destination,receipt_path,parent_packet,original_asset,stopped_asset))
    archive,sidecar,export_path,packet,destination,receipt_path,parent_packet,original_asset,stopped_asset=paths
    journal=destination.parent/(destination.name+'_import_journal.jsonl')
    failure=receipt_path.with_name(receipt_path.stem+'_failure.json');started=time.monotonic();phase='preflight';imported={}
    for path in (destination,receipt_path,journal,failure):
        no_links(path);require(not path.exists(),'Retain prior/partial import evidence: '+str(path))
    require(destination.parent.is_dir() and receipt_path.parent.is_dir(),'Dedicated output parents must already exist')
    for source in (packet,parent_packet,original_asset.parent,stopped_asset.parent):
        no_links(source)
        require(not destination.resolve().is_relative_to(source.resolve()) and not receipt_path.resolve().is_relative_to(source.resolve()),
                'Do not write into an immutable source/asset namespace')
    def budget():require(time.monotonic()-started<=MAX_SECONDS,'Finite local transportation time exceeded')
    try:
        require(archive.name==RESULT_ARCHIVE and sidecar.name==RESULT_ARCHIVE+'.sha256' and export_path.name==EXPORT_RECEIPT,
                'Exact downloaded return filenames required')
        before=regular(archive);require(before.st_size==ARCHIVE_BYTES and 0<before.st_size<=LIMITS['archive_bytes'],'Pinned archive bytes differ')
        export=read(export_path,1024**2)
        require(export.get('complete') is True and export.get('archive_sha256')==ARCHIVE_SHA256 and
            type(export.get('bytes')) is int and export['bytes']==ARCHIVE_BYTES and
            export.get('manifest_member')==MANIFEST_NAME and export.get('manifest_sha256')==MANIFEST_SHA256 and
            export.get('protocol_sha256')==PROTOCOL_SHA256 and type(export.get('measurement_complete')) is bool and
            type(export.get('failure_present')) is bool and type(export.get('optimizer_updates')) is int and
            export['optimizer_updates']==0 and export.get('model_qualified') is False and
            export.get('training_success_not_implied') is True,'Pinned export receipt differs')
        require(regular(sidecar).st_size<=1024 and sidecar.read_bytes()==(ARCHIVE_SHA256+'  '+archive.name+'\n').encode('ascii'),
                'Exact LF checksum sidecar required')
        require(sha(archive,budget)==ARCHIVE_SHA256,'Compressed archive fingerprint differs')
        require(sha(packet/'protocol.json',budget)==PROTOCOL_SHA256,'Frozen local protocol differs')
        protocol=read(packet/'protocol.json')
        require(protocol.get('format')=='codeformer-bounded-fusion-feasibility-finite-manual-v1' and
            protocol.get('outputs')==OUTPUTS and protocol.get('export_prefix')==PREFIX and
            exact(protocol.get('limits'),LIMITS) and exact(protocol.get('assets'),ASSETS) and
            exact(protocol.get('call_caps'),CALL_CAPS),'Frozen finite protocol scope differs')
        files=protocol.get('files_sha256');exports=protocol.get('export_files')
        require(isinstance(files,dict) and isinstance(exports,list) and len(exports)==len(set(exports)) and
            set(exports)<=set(files) and len(files)<1000,'Frozen source/input map differs')
        namespace(files)
        actual=set()
        for folder,dirs,names in os.walk(packet,followlinks=False):
            for name in dirs:no_links(Path(folder)/name)
            for name in names:actual.add((Path(folder)/name).relative_to(packet).as_posix())
        require(actual==set(files)|{'protocol.json'},'Frozen packet namespace changed')
        for name,digest in files.items():
            require(isinstance(digest,str) and HEX.fullmatch(digest) and sha(packet/name,budget)==digest,'Frozen retained source/input changed')
        parent_sha=next(iter(ASSETS.values()))['parent_protocol_sha256']
        require(sha(parent_packet/'protocol.json',budget)==parent_sha,'Retained parent protocol differs')
        parent_assets={}
        for key,path in [('original',original_asset),('stopped',stopped_asset)]:
            require(regular(path).st_size==ASSETS[key]['bytes'] and sha(path,budget)==ASSETS[key]['sha256'],'External parent model bytes differ')
            parent_assets[key]={'path':str(path),'sha256':ASSETS[key]['sha256'],'bytes':ASSETS[key]['bytes']}
        phase='full_stream_verification';records={};jsons={};payload=0
        critical={'protocol.json',MANIFEST_NAME,'outputs/supervision/run_receipt.json',OUTPUTS+'/manual_launch.json',
                  OUTPUTS+'/worker_complete.json',OUTPUTS+'/worker_failure.json'}
        with archive.open('rb') as raw:
            opened=os.fstat(raw.fileno());require(object_identity(opened)==object_identity(before),'Archive object identity changed')
            with tarfile.open(fileobj=raw,mode='r|gz') as tar:
                for member in tar:
                    budget();name=canonical(member.name)
                    require(member.type in (tarfile.REGTYPE,tarfile.AREGTYPE) and member.isfile() and member.sparse is None and
                        not member.linkname and not any(k.startswith('GNU.sparse') for k in member.pax_headers),'Only nonsparse ordinary regular members allowed')
                    require(name not in records and len(records)<LIMITS['regular_files'],'Duplicate/file cap violation')
                    require(type(member.size) is int and 0<=member.size<=LIMITS['result_bytes'],'Invalid/oversized member')
                    payload+=member.size;require(payload<=LIMITS['result_bytes'],'Total payload byte cap exceeded')
                    suffix=PurePosixPath(name).suffix.lower()
                    if name.startswith(OUTPUTS+'/'):
                        require(suffix in {'.json','.npy','.png'},'Executable/model/unexpected output extension forbidden')
                        cap=8*1024**2 if name==MANIFEST_NAME else (LIMITS['array_file_bytes'] if suffix=='.npy' else
                            LIMITS['PNG_file_bytes'] if suffix=='.png' else 1024**2)
                        require(member.size<=cap,'Per-artifact output cap exceeded')
                    if name in OPERATIONAL and name.endswith('.log'):require(member.size<=2*1024**2,'Closed log cap exceeded')
                    if suffix in {'.pt','.pth','.pkl','.pickle','.ckpt','.safetensors'}:raise ValueError('No model/optimizer weight may be returned')
                    h=hashlib.sha256();count=0;chunks=[] if name in critical else None
                    with tar.extractfile(member) as source:
                        for chunk in iter(lambda:source.read(1024**2),b''):
                            budget();h.update(chunk);count+=len(chunk)
                            if chunks is not None:chunks.append(chunk)
                    require(count==member.size,'Truncated TAR member')
                    records[name]={'sha256':h.hexdigest(),'bytes':count}
                    if chunks is not None:
                        require(count<=8*1024**2,'Critical metadata byte cap exceeded');jsons[name]=parse(b''.join(chunks))
            require(identity(regular(archive))==identity(before) and identity(os.fstat(raw.fileno()))==identity(opened),'Archive changed during verification')
        namespace(records)
        require(records[MANIFEST_NAME]['sha256']==MANIFEST_SHA256 and records['protocol.json']['sha256']==PROTOCOL_SHA256 and
                jsons['protocol.json']==protocol,'Manifest/protocol fingerprints differ')
        accounting=status(export,jsons[MANIFEST_NAME],protocol,records,jsons)
        require(shutil.disk_usage(destination.parent).free>=payload+RESERVE_BYTES,'Insufficient local import space plus reserve')
        phase='exclusive_import';destination.mkdir(exist_ok=False)
        with journal.open('x',encoding='utf-8',newline='\n') as log:
            def note(value):log.write(json.dumps(value,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            note({'phase':'all_members_verified_before_import','archive_sha256':ARCHIVE_SHA256,'members':len(records)})
            with archive.open('rb') as raw:
                opened=os.fstat(raw.fileno());require(object_identity(opened)==object_identity(before),'Archive changed before import')
                with tarfile.open(fileobj=raw,mode='r|gz') as tar:
                    for member in tar:
                        budget();name=canonical(member.name);require(name in records and name not in imported,'Import namespace changed')
                        target=no_links(destination/name);require(target.is_relative_to(destination),'Import escape')
                        target.parent.mkdir(parents=True,exist_ok=True);no_links(target.parent)
                        h=hashlib.sha256();count=0
                        with target.open('xb') as output,tar.extractfile(member) as source:
                            for chunk in iter(lambda:source.read(1024**2),b''):
                                budget();count+=len(chunk);require(count<=records[name]['bytes'],'Import member size grew')
                                output.write(chunk);h.update(chunk)
                            output.flush();os.fsync(output.fileno())
                        require({'sha256':h.hexdigest(),'bytes':count}==records[name],'Import hash/bytes changed')
                        imported[name]=records[name];note({'member':name,**records[name]})
                require(identity(regular(archive))==identity(before) and identity(os.fstat(raw.fileno()))==identity(opened),'Archive changed during import')
            require(imported==records,'Partial member import');note({'phase':'import_complete','members':len(imported)})
        phase='closed'
        result={'format':'bounded-fusion-feasibility-strict-return-transport-v1','complete':True,
            'archive_sha256':ARCHIVE_SHA256,'archive_bytes':ARCHIVE_BYTES,'protocol_sha256':PROTOCOL_SHA256,
            'manifest_sha256':MANIFEST_SHA256,'manifest_member':MANIFEST_NAME,'import_directory':str(destination),
            'exact_namespace_verified':True,'whole_archive_and_all_member_hashes_verified_before_import':True,
            'files_sha256':{name:r['sha256'] for name,r in records.items()},
            'files_bytes':{name:r['bytes'] for name,r in records.items()},'member_maps_include_manifest':True,
            'members':len(records),'payload_bytes':payload,'accounting':accounting,
            'frozen_packet_files_freshly_verified':len(files)+1,'parent_asset_files_freshly_verified':parent_assets,
            'closed_operational_companions_sha256':{name:records[name]['sha256'] for name in OPERATIONAL},
            'retained_not_exported_sha256':jsons[MANIFEST_NAME]['retained_not_exported_sha256'],
            'source_sha256':sha(__file__),'import_journal_sha256':sha(journal),'seconds':time.monotonic()-started,
            'returned_code_executed':False,'neural_calls':0,'gradient_queries':0,'optimizer_updates':0,
            'VM_calls':0,'model_qualified':False,'training_authorized':False,
            'classification':'Completed diagnostic transportation only; scientific/visual usefulness remains unreviewed.'}
        write_new(receipt_path,result);return result
    except BaseException as error:
        if not failure.exists():write_new(failure,{'complete':False,'phase':phase,'error_type':type(error).__name__,
            'message':str(error),'imported_members':len(imported),'partial_evidence_retained':True,
            'automatic_retry':False,'model_qualified':False,'returned_code_executed':False})
        raise
