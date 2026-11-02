"""Independent transport admission and exclusive extraction; no model imports."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import gzip,hashlib,json,re,stat,tarfile,time
BASE=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION')
DATA=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
STEM='cctv-completion-codeformer-finetune-v1-r1'
MEMBER='cctv_completion_codeformer_finetune_v1_r1_results'
EXPECTED='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
EXPECTED_ARCHIVE='0281d9230ad9f7e74ec0318d9e5918c836c5475fad544d891cfc3fc8935a638f'
PARENT=BASE/'outputs/completion_codeformer_finetune_v1_r1_return'
CAP=1536*1024**2
started=time.monotonic()
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        need(k not in d,'Duplicate JSON key');d[k]=v
    return d
def loads(b):return json.loads(b,object_pairs_hook=unique_pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def relative(name):
    need(isinstance(name,str) and name and '\\' not in name and ':' not in name and '\x00' not in name,'Unsafe archive name')
    p=PurePosixPath(name);need(not p.is_absolute() and '..' not in p.parts and '.' not in p.parts and p.as_posix()==name,'Noncanonical member name')
    need(p.parts[0]==MEMBER and len(p.parts)>1,'Wrong archive root')
    return p
archive=DATA/(STEM+'-results.tar.gz');sidecar=DATA/(STEM+'-results.tar.gz.sha256');export_path=DATA/(STEM+'-export.json')
original=pin(archive);need(original['bytes']==1319322259 and original['sha256']==EXPECTED_ARCHIVE,'Wrong archive bytes/hash')
line=sidecar.read_bytes();need(b'\r' not in line and line.endswith(b'\n') and line.count(b'\n')==1,'Checksum requires one LF line')
need(line.decode('ascii')==EXPECTED_ARCHIVE+'  '+archive.name+'\n','Checksum identity differs')
export=loads(export_path.read_bytes());need(export['complete'] is True and export['archive_sha256']==original['sha256'] and export['bytes']==original['bytes'] and export['protocol_sha256']==EXPECTED,'Export identity differs')
sealed=(BASE/'outputs/completion_codeformer_finetune_v1_r1_packet/protocol.json').read_bytes();need(hashlib.sha256(sealed).hexdigest()==EXPECTED,'Sealed R1 protocol drift')
PARENT.mkdir(exist_ok=False);ROOT=PARENT/MEMBER;ROOT.mkdir()
seen=set();folded=set();rows=[];total=0;manifest=None;tail_bytes=0
class Limited:
    def __init__(self,stream):self.stream=stream;self.count=0
    def read(self,n=-1):
        need(time.monotonic()-started<600,'Transport600second bound')
        need(n>=0,'Unbounded gzip read refused')
        b=self.stream.read(n);self.count+=len(b);need(self.count<=CAP+16*1024**2,'Decompressed tar bound');return b
with gzip.open(archive,'rb') as gz:
    limited=Limited(gz)
    with tarfile.open(fileobj=limited,mode='r|') as tar:
        for info in tar:
            name=relative(info.name);key=name.as_posix();need(key not in seen and key.casefold() not in folded,'Duplicate/case-colliding member')
            seen.add(key);folded.add(key.casefold());need(info.isreg() and not info.issparse() and info.size>=0,'Nonregular/sparse archive member')
            need(manifest is None,'Manifest must be last member')
            total+=info.size;need(total<=CAP,'Uncompressed file cap')
            destination=PARENT.joinpath(*name.parts);need(destination.resolve().is_relative_to(ROOT.resolve()),'Member escaped result root')
            destination.parent.mkdir(parents=True,exist_ok=True)
            for component in [destination,*destination.parents]:
                need(not component.is_symlink(),'Symlink extraction path')
                if component==ROOT:break
            h=hashlib.sha256();count=0
            source=tar.extractfile(info);need(source is not None,'Missing regular member bytes')
            with source,destination.open('xb') as dst:
                while True:
                    b=source.read(min(8*1024*1024,info.size-count+1))
                    if not b:break
                    count+=len(b);need(count<=info.size,'Member grew beyond declared size');h.update(b);dst.write(b)
            need(count==info.size,'Member truncated')
            row={'path':PurePosixPath(*name.parts[1:]).as_posix(),'bytes':count,'sha256':h.hexdigest()};rows.append(row)
            if row['path']=='protocol.json':need(destination.read_bytes()==sealed,'Returned R1 protocol not exact frozen bytes')
            if row['path']=='export_manifest.json':manifest=loads(destination.read_bytes())
        buffered=getattr(tar.fileobj,'buf',b'');need(not any(buffered),'Nonzero buffered trailing tar data');tail_bytes+=len(buffered)
    while True:
        b=limited.read(1024*1024)
        if not b:break
        need(not any(b),'Nonzero trailing archive data');tail_bytes+=len(b)
need(manifest is not None and manifest['protocol_sha256']==EXPECTED,'Missing/wrong export manifest')
need(manifest['format']=='completion-codeformer-finetune-return-manifest-v1','Unexpected manifest format')
entries=manifest['files'];need(len(entries)==export['payload_files']==444 and len(rows)==445,'Export file count differs')
expected={}
for row in entries:
    relative(MEMBER+'/'+row['path']);need(row['path'] not in expected,'Duplicate manifest entry');expected[row['path']]=row
actual={row['path']:row for row in rows if row['path']!='export_manifest.json'}
need(actual==expected,'Manifest payload membership/hash/size differs')
need({'protocol.json','supervisor.json','trainer.log','outputs/results.json','outputs/terminal_training_state.pth'}<=actual.keys(),'Required terminal evidence absent')
actual_names={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
need(actual_names==set(actual)|{'export_manifest.json'},'Extracted file namespace differs')
need(sha(archive)==original['sha256'] and archive.stat().st_size==original['bytes'],'Original archive changed during extraction')
result=loads((ROOT/'outputs/results.json').read_bytes());supervisor=loads((ROOT/'supervisor.json').read_bytes())
audit={'format':'completion-codeformer-finetune-v1-r1-independent-transport-audit','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'archive':original,'sidecar':pin(sidecar),'export':pin(export_path),'destination':str(ROOT),'regular_members':len(rows),'payload_files':len(entries),'uncompressed_file_bytes':total,'decompressed_tar_bytes':limited.count,'zero_trailing_bytes':tail_bytes,'no_duplicates_links_sparse_devices_traversal_unlisted_files':True,'all_member_sizes_and_SHA256_match_manifest':True,'returned_protocol_equals_frozen_bytes':True,'protocol_sha256':EXPECTED,'worker_exit_reported':supervisor['worker_exit'],'cycles_completed_reported':result.get('cycles_completed'),'reported_counts':result.get('counts'),'reported_numeric_screen':result.get('numeric_screen'),'all_claims_pending_independent_state_pixel_visual_audits':True,'members':rows,'local_neural_calls':0,'model_loads':0,'returned_code_executed':False,'audit_seconds':time.monotonic()-started}
target=PARENT/'transport_audit.json'
with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'complete':True,'destination':str(ROOT),'audit_sha256':sha(target),'files':len(rows),'raw_bytes':total,'reported_cycles':result.get('cycles_completed'),'reported_counts':result.get('counts'),'reported_numeric_screen':result.get('numeric_screen'),'seconds':audit['audit_seconds']},indent=2))