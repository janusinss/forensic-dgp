"""Strict one-stream inert import of the exact failed filter-fusion return.

Never execute returned code or deserialize arrays/models/checkpoints.  Every
physical compressed byte is hashed during canonical extraction; all source and
manifest bindings close before a completed receipt is written. No retry/resume.
"""
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import sys
import time
import traceback
import unicodedata
import zlib

ROOT=Path(__file__).resolve().parents[1]
PREFIX='cctv-codeformer-filter-fusion-training-v1'
VM_NAME='cctv_codeformer_filter_fusion_training_v1_vm'
OUTPUT='outputs/filter_fusion_training_v1'
PROTOCOL='c1f7d8386856bd95ab3af3a8623f33fcecddd4b7e08f0696ea7694a0542c1949'
ARCHIVE_SHA='3d6feaf30d4477fe84bd51c57d560cb10d7b4cd6509db627fc61e03accb35d16'
ARCHIVE_BYTES,MEMBERS,PAYLOAD_BYTES=4408715848,9081,4905597698
PAYLOAD_CAP,MEMBER_CAP,METADATA_CAP=7*1024**3,16000,128*1024**2
RESERVE,JSON_CAP,LOG_CAP,PNG_CAP,ARRAY_CAP=4*1024**3,4*1024**2,2*1024**2,198656,32*1024**2
TAR_CAP=PAYLOAD_CAP+MEMBER_CAP*1023+10240
REVIEW=ROOT/'outputs/codeformer_filter_fusion_return_review_20261019_v1'
DEST=ROOT/'outputs/codeformer_filter_fusion_training_v1_return'
STAGE=ROOT/'outputs'/VM_NAME
MANIFEST=OUTPUT+'/export_manifest.json'
MANIFEST_SHA='8c6463a7fad4501ce55b4be2a07af98ab48b077951533578b77218de68006173'
COUNTS={'candidate_snapshot_forwards':956,'candidate_training_forwards':200,
'original_clear_teacher_forwards':55,'R2_dark_teacher_forwards':55,
'R2_baseline_forwards':478,'original_baseline_forwards':478,'parent_baseline_forwards':478,
'candidate_TRAIN_guard_forwards':48,'parity_candidate_forwards':1,'parity_R2_forwards':1,
'filter_witness_forwards':3,'clear_witness_forwards':2,'dark_witness_forwards':2,
'backward_calls':200,'optimizer_step_attempts':50,'gradient_queries':7}
CHECKPOINTS={OUTPUT+'/snapshot_0000/checkpoint.pt',OUTPUT+'/snapshot_0050/checkpoint.pt',OUTPUT+'/stopped_checkpoint.pt'}
STATE={'path':OUTPUT+'/stopped_checkpoint.pt','sha256':'236eceff57fa315ed776f7e0a973bbfb50fce0b4d3185e92f12545964b5a87de','bytes':461955077}
DEVICE=re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)',re.I)
def need(value, message):
    if not value:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def unlinked(path, regular=False):
    path=Path(path).absolute()
    for item in (path,*path.parents):
        if item.exists() or item.is_symlink():
            st=item.lstat()
            need(not stat.S_ISLNK(st.st_mode) and not getattr(st,'st_file_attributes',0)&1024,'Linked/reparse path forbidden')
    if regular:
        st=path.lstat();need(stat.S_ISREG(st.st_mode) and st.st_nlink==1,'Unlinked regular file required')
    return path


def file_pin(path):
    path=unlinked(path,True);before=path.stat();h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    after=path.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'File changed while hashing')
    return {'sha256':h.hexdigest(),'bytes':before.st_size}


def read(path,cap=JSON_CAP):
    path=unlinked(path,True);need(path.stat().st_size<=cap,'Bounded metadata read exceeded')
    return path.read_bytes()


def parse(payload):
    def unique(items):
        result={}
        for key,value in items:
            need(key not in result,'Duplicate JSON key');result[key]=value
        return result
    return json.loads(payload.decode('utf-8'),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def load(path):return parse(read(path))


def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')


def write_new(path,value):
    path=unlinked(path);payload=encode(value)
    need(shutil.disk_usage(path.parent).free>=RESERVE+len(payload)+65536,'Receipt write reserve unavailable')
    with path.open('xb')as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())


def portable(name):
    p=PurePosixPath(name)
    need(isinstance(name,str) and name and not p.is_absolute() and p.as_posix()==name and
        all(x not in ('','.','..') for x in p.parts),'Unsafe/noncanonical TAR path')
    need(unicodedata.normalize('NFC',name)==name and not any(x in name for x in '\\:"<>|?*') and
        all(not DEVICE.match(x) and not x.endswith((' ','.')) and len(x.encode('utf-16-le'))<=510 and
            all(ord(c)>=32 for c in x)for x in p.parts),'Nonportable TAR path')
    return p


def header(name,size):
    name=name.encode('ascii');prefix=b''
    if len(name)>100:
        choices=[i for i,c in enumerate(name)if c==47 and i<=155 and len(name)-i-1<=100]
        need(choices,'USTAR path cannot be represented');i=choices[0];prefix,name=name[:i],name[i+1:]
    raw=bytearray(512);raw[:len(name)]=name;raw[345:345+len(prefix)]=prefix
    for start,width,value in ((100,8,0o644),(108,8,0),(116,8,0),(124,12,size),(136,12,0)):
        raw[start:start+width]=(('%0*o'%(width-1,value))+'\0').encode()
    raw[148:156]=b' '*8;raw[156]=48;raw[257:263]=b'ustar\0';raw[263:265]=b'00'
    raw[148:156]=('%06o\0 '%sum(raw)).encode()
    return bytes(raw)


class StrictGzip:
    """Bounded zlib wrapper validates exactly one gzip frame, CRC and full EOF."""
    def __init__(self,path):
        self.source=path.open('rb');self.decoder=zlib.decompressobj(31);self.pending=b'';self.buffer=bytearray()
        self.digest=hashlib.sha256();self.compressed=self.expanded=0;self.crc=0;self.ended=False
    def fill(self):
        if self.ended:return
        if self.decoder.eof:
            need(not self.decoder.unused_data and not self.pending and not self.source.read(1),'Trailing gzip bytes or extra frame')
            self.ended=True;return
        payload=self.pending
        if not payload:
            payload=self.source.read(1024**2);need(payload,'Truncated gzip frame')
            self.digest.update(payload);self.compressed+=len(payload)
        block=self.decoder.decompress(payload,4*1024**2);self.pending=self.decoder.unconsumed_tail
        self.expanded+=len(block);self.crc=zlib.crc32(block,self.crc)
        need(self.expanded<=TAR_CAP,'Expanded TAR cap exceeded');self.buffer.extend(block)
    def read(self,size):
        need(type(size)is int and 0<=size<=4*1024**2,'Bounded stream read required')
        while len(self.buffer)<size and not self.ended:self.fill()
        take=min(size,len(self.buffer));block=bytes(self.buffer[:take]);del self.buffer[:take];return block
    def exact(self,size):
        block=self.read(size);need(len(block)==size,'Truncated physical TAR');return block
    def close(self):self.source.close()



def compact(value):
    return (json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def failure_fields(doc):
    need(doc.get('failure_present') is True and doc.get('complete_training_process') is False and
        doc.get('worker_terminal_evidence')==OUTPUT+'/failure.json' and doc.get('completed_training_updates')==50 and
        doc.get('partial_microbatches')==0 and doc.get('optimizer_inflight') is False and
        doc.get('worker_recorded_counts')==COUNTS and doc.get('activity_unknown_without_worker_receipt') is False and
        doc.get('malformed_worker_terminal_preserved') is None and doc.get('checkpoint_members')==sorted(CHECKPOINTS) and
        doc.get('state_checkpoint')==STATE and doc.get('state_checkpoint_complete') is True and
        doc.get('model_qualified') is False and doc.get('training_or_export_completion_not_model_acceptance') is True,
        'Exact failed50update export status differs')


def main():
    started=time.monotonic();outputs=ROOT/'outputs'
    archive=unlinked(outputs/(PREFIX+'-results.tar.gz'),True)
    checksum=outputs/(PREFIX+'-results.tar.gz.sha256');ep=outputs/(PREFIX+'-export.json');fp=outputs/(PREFIX+'-supervisor-final.json')
    receipt_path=REVIEW/'return_import_v1.json';unlinked(DEST);unlinked(REVIEW)
    need(not DEST.exists() and not receipt_path.exists() and not (REVIEW/'transport_v1.json').exists() and
         not (REVIEW/'import_failure_v1.json').exists(),'Preserve every prior/partial import; no replay')
    REVIEW.mkdir(exist_ok=True)
    companion_pins={p.name:file_pin(p) for p in (checksum,ep,fp)}
    exported=load(ep);final=load(fp)
    need(exported.get('format')=='codeformer-filter-fusion-training-export-v1' and exported.get('complete') is True and
         exported.get('archive_sha256')==ARCHIVE_SHA and exported.get('bytes')==ARCHIVE_BYTES and
         exported.get('protocol_sha256')==PROTOCOL and exported.get('manifest_member')==MANIFEST and
         exported.get('manifest_sha256')==MANIFEST_SHA and exported.get('regular_members')==MEMBERS and
         exported.get('payload_bytes')==PAYLOAD_BYTES,'Exact export identity differs')
    failure_fields(exported)
    need(read(checksum,1024)==(ARCHIVE_SHA+'  '+archive.name+'\n').encode('ascii'),'ASCII basename LF checksum differs')
    before=archive.stat();need(before.st_size==ARCHIVE_BYTES,'Actual compressed byte count differs')
    need(shutil.disk_usage(outputs).free>=PAYLOAD_BYTES+RESERVE,'Complete payload plus4GiB reserve required')
    pb=read(STAGE/'protocol.json');need(sha(pb)==PROTOCOL,'Sealed local protocol differs');p=parse(pb)
    static=p['files_sha256'];static_sizes=p['files_bytes']
    need(p['vm_name']==VM_NAME and p['outputs']==OUTPUT and len(static)==1142 and set(static)==set(static_sizes),
         'Exact1143static sealed packet differs')
    allowed=p['allowed_output_names'];need(type(allowed) is list and len(allowed)==len(set(allowed))==9882,
         'Exact prospective output whitelist differs')
    permitted=set(static)|{'protocol.json'}|set(allowed)
    for name in permitted:portable(name)
    need(CHECKPOINTS.issubset(permitted) and MANIFEST in permitted,'Actual checkpoint/manifest whitelist differs')
    need(final.get('format')=='codeformer-filter-fusion-supervisor-final-v1' and final.get('complete') is True and
         final.get('protocol_sha256')==PROTOCOL and final.get('archive')=={'path':archive.name,'sha256':ARCHIVE_SHA,'bytes':ARCHIVE_BYTES} and
         final.get('export_receipt')=={'path':ep.name,**companion_pins[ep.name]} and
         final.get('checksum')=={'path':checksum.name,**companion_pins[checksum.name]} and
         final.get('model_qualified') is False,'Final companion identity differs')
    log_meta=final['export_log'];log=base64.b64decode(log_meta['base64'],validate=True)
    need(log_meta['path']=='outputs/supervision/export.log' and len(log)==log_meta['bytes']<=LOG_CAP and
         sha(log)==log_meta['sha256'] and parse(log.strip())==exported,'External export log closure differs')
    stages=final['stages']
    need([(r['stage'],r['external_cap_seconds'],r['grace_seconds'],r['exit_code']) for r in stages]==
         [('--verify-transfer',300,30,0),('--run',5300,30,1),('--export',600,30,0)] and
         all(r['worker_terminal'] is True and r['child_process_group_terminal'] is True and not r['log_capped'] and
             not r['outer_watchdog_expired'] and r['supervisor_exception'] is None and
             type(r['seconds']) in (int,float) and 0<=r['seconds']<=r['external_cap_seconds']+r['grace_seconds'] for r in stages),
         'Exact finite worker failure supervision differs')
    finished={'complete':True,'protocol_sha256':PROTOCOL,'worker_terminal':True,'stages':stages,'model_qualified':False,
              'training_or_export_completion_not_model_acceptance':True}
    need(final['finished']=={'path':'outputs/supervision/finished.json','sha256':sha(compact(finished)),
                            'bytes':len(compact(finished))},'Separately retained finished bytes differ')
    DEST.mkdir();hashes={};sizes={};folded=set();total=metadata=position=0;previous=''
    strict=StrictGzip(archive)
    try:
        while True:
            need(time.monotonic()-started<=1200,'Finite local inert import1200s cap')
            raw=strict.exact(512)
            if raw==b'\0'*512:
                need(strict.exact(512)==b'\0'*512,'Double physical TAR EOA required')
                tail=((position+1024+10239)//10240)*10240-position-1024
                need(strict.exact(tail)==b'\0'*tail and strict.read(1)==b'','Exact TAR padding/full EOF required')
                break
            def field(start,width):return raw[start:start+width].split(b'\0',1)[0].decode('ascii')
            name=field(0,100);prefix=field(345,155);name=prefix+'/'+name if prefix else name;portable(name)
            sizefield=raw[124:136]
            need(re.fullmatch(b'[0-7]{11}\0',sizefield),'Canonical TAR size required');size=int(sizefield[:-1],8)
            need(raw==header(name,size),'Canonical regular USTAR header differs')
            need(name in permitted and name not in hashes and name.casefold() not in folded and name>previous,
                 'Sorted exact allowed unique namespace differs')
            folded.add(name.casefold());previous=name;need(len(folded)<=MEMBER_CAP,'Physical member cap exceeded')
            suffix=Path(name).suffix.lower();checkpoint=name in CHECKPOINTS;static_file=name in static
            if static_file:need(size==static_sizes[name],'Sealed static physical size differs: '+name)
            elif checkpoint:need(0<size<=(384 if 'snapshot_0000/' in name else 448)*1024**2,'Checkpoint physical cap exceeded')
            else:need(not re.search(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin|npz)(?:$|[.\-])',Path(name).name.lower()),
                      'Unplanned executable model binary forbidden')
            cap=static_sizes[name] if static_file else 448*1024**2 if checkpoint else ARRAY_CAP if suffix=='.npy' else PNG_CAP if suffix=='.png' else JSON_CAP
            need(size<=cap,'Per-member physical cap exceeded: '+name)
            if suffix not in ('.npy','.png','.pt','.pth'):metadata+=size;need(metadata<=METADATA_CAP,'Aggregate metadata cap exceeded')
            total+=size;need(total<=PAYLOAD_CAP,'Aggregate payload cap exceeded')
            target=unlinked(DEST.joinpath(*PurePosixPath(name).parts))
            need(target.resolve(strict=False).is_relative_to(DEST.resolve()),'Destination escaped exact import root')
            need(shutil.disk_usage(outputs).free>=RESERVE+size+65536,'Import disk reserve reached')
            target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();remaining=size
            with target.open('xb') as handle:
                while remaining:
                    block=strict.exact(min(remaining,1024**2));need(handle.write(block)==len(block),'Short inert write')
                    h.update(block);remaining-=len(block)
                handle.flush();os.fsync(handle.fileno())
            padding=(-size)%512;need(strict.exact(padding)==b'\0'*padding,'Nonzero TAR member padding')
            position+=512+size+padding;hashes[name]=h.hexdigest();sizes[name]=size
        need(strict.ended and strict.compressed==ARCHIVE_BYTES and strict.digest.hexdigest()==ARCHIVE_SHA,
             'Full compressed SHA/CRC/frame/full EOF differs')
        with archive.open('rb') as handle:
            need(handle.read(10)==b'\x1f\x8b\x08\0\0\0\0\0\0\xff','Canonical gzip level6 header differs')
            handle.seek(-8,2);crc,isize=struct.unpack('<II',handle.read(8))
        need(crc==strict.crc and isize==strict.expanded%2**32,'Actual gzip CRC/ISIZE trailer differs')
    finally:strict.close()
    after=archive.stat();need((before.st_size,before.st_mtime_ns,before.st_ino)==
        (after.st_size,after.st_mtime_ns,after.st_ino),'Archive changed during import')
    need(len(hashes)==MEMBERS and total==PAYLOAD_BYTES and hashes['protocol.json']==PROTOCOL and hashes[MANIFEST]==MANIFEST_SHA,
         'Full returned physical census/manifest differs')
    manifest=load(DEST/MANIFEST)
    need(manifest['format']=='codeformer-filter-fusion-training-return-v1' and manifest['protocol_sha256']==PROTOCOL and
         manifest['dependencies']==p['dependencies']=={} and manifest['dependency_assets_copied'] is False,
         'Exact immutable manifest/protocol fields differ')
    failure_fields(manifest);bindings=manifest['files']
    need(set(bindings)==set(hashes)-{MANIFEST} and all(set(v)=={'bytes','sha256'} and sizes[n]==v['bytes'] and hashes[n]==v['sha256']
         for n,v in bindings.items()),'Every actual returned member must match full manifest')
    need(set(static).issubset(hashes) and all(hashes[n]==v and sizes[n]==static_sizes[n] for n,v in static.items()),
         'All1142frozen source/data/model payload bytes must match protocol')
    need({n for n in hashes if n.endswith(('.pt','.pth','.ckpt','.safetensors'))}==CHECKPOINTS|set(p['weight_assets']),
         'Exactly3opaque input models and3opaque actual checkpoints required')
    need({k:STATE[k] for k in ('sha256','bytes')}=={'sha256':hashes[STATE['path']],'bytes':sizes[STATE['path']]},
         'Actual stopped checkpoint physical binding differs')
    failure=load(DEST/(OUTPUT+'/failure.json'));supervised=load(DEST/'outputs/supervision/run_receipt.json')
    need(failure.get('complete') is True and failure.get('failure') is True and failure.get('protocol_sha256')==PROTOCOL and
         failure.get('counts')==COUNTS and failure.get('completed_training_updates')==50 and failure.get('partial_microbatches')==0 and
         failure.get('optimizer_inflight') is False and failure.get('state_checkpoint')==STATE and
         failure.get('state_checkpoint_complete') is True and failure.get('model_qualified') is False and
         isinstance(failure.get('error'),str) and failure.get('error'),'Actual failed50worker evidence differs')
    need(all(type(v) is int and 0<=v<=p['call_caps'][k] for k,v in COUNTS.items()) and
         supervised.get('protocol_sha256')==PROTOCOL and supervised.get('worker_terminal') is True and
         supervised.get('child_process_group_terminal') is True and supervised.get('stages')==stages[:2] and
         supervised.get('no_automatic_retry_or_resume') is True and supervised.get('model_qualified') is False,
         'Archived supervisor failure stage closure differs')
    need(all(sizes[r['log']]==r['log_bytes'] for r in stages[:2]) and stages[2]['log_bytes']==len(log),
         'Archived or external log byte counts differ')
    # Returned code stays opaque. Source equality above is a sealed-protocol
    # check, not permission to import that code or accept its scientific claims.
    need(not any(n.split('.')[0] in ('torch','numpy','tensorflow','jax') for n in sys.modules),
         'No neural/array framework execution invariant failed')
    for path in (checksum,ep,fp):need(file_pin(path)==companion_pins[path.name],'Companion changed during import')
    need(sha(read(STAGE/'protocol.json'))==PROTOCOL,'Sealed local protocol changed during import')
    actual_names={q.relative_to(DEST).as_posix() for q in DEST.rglob('*') if q.is_file()}
    need(actual_names==set(hashes),'Exclusive imported file namespace differs')
    for n in hashes:
        q=unlinked(DEST/n,True);need(q.stat().st_size==sizes[n],'Fresh imported file size differs')
    receipt={'format':'codeformer-filter-fusion-training-local-import-v1','complete':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'import_root':str(DEST),'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,
        'protocol_sha256':PROTOCOL,'manifest_sha256':MANIFEST_SHA,'files_sha256':hashes,'files_bytes':sizes,
        'regular_files':len(hashes),'payload_bytes':total,'metadata_bytes':metadata,'expanded_tar_bytes':strict.expanded,
        'companion_pins':companion_pins,'actual_returned_counts':COUNTS,'all1143static_files_exact':True,
        'single_gzip_CRC_manual_USTAR_manifest_companions_pass':True,'source_sha256':file_pin(__file__)['sha256'],
        'returned_code_executed':False,'checkpoint_loaded':False,'actual_local_neural_gradient_optimizer_VM_calls':0,
        'complete_training_process':False,'failure_present':True,'completed_training_updates':50,'model_qualified':False,
        'same_owner_importer_and_transport_auditor':True,'independent_scientific_review_required':True,
        'seconds':time.monotonic()-started}
    write_new(receipt_path,receipt)
    transport={'format':'codeformer-filter-fusion-training-transport-audit-v1','complete':True,'transport_pass':True,
        'import_receipt':file_pin(receipt_path),'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,
        'manifest_sha256':MANIFEST_SHA,'protocol_sha256':PROTOCOL,'regular_members':len(hashes),'payload_bytes':total,
        'expanded_TAR_bytes':strict.expanded,'gzip_CRC32':crc,'gzip_ISIZE':isize,
        'all1143frozen_static_bytes_exact':True,'actual_completed_training_updates':50,'failure_present':True,
        'worker_error':failure['error'],'actual_returned_counts':COUNTS,'checkpoint_members':sorted(CHECKPOINTS),
        'complete_training_process':False,'model_qualified':False,'returned_code_executed':False,'checkpoint_loaded':False,
        'source_sha256':receipt['source_sha256'],'independent_scientific_review_required':True,
        'scope':'Exact inert transport/import closure; scientific state,pixels,visuals remain separate. Failed run retained; no retry/resume.'}
    write_new(REVIEW/'transport_v1.json',transport)
    print(json.dumps({'complete':True,'import_root':str(DEST),'files':len(hashes),'payload_bytes':total,
        'completed_training_updates':50,'worker_error':failure['error'],'model_qualified':False,'seconds':time.monotonic()-started}))


if __name__=='__main__':
    try:main()
    except BaseException:
        REVIEW.mkdir(exist_ok=True);failure=REVIEW/'import_failure_v1.json'
        if not failure.exists():
            write_new(failure,{'complete':False,'traceback':traceback.format_exc(),'source_sha256':file_pin(__file__)['sha256'],
                               'partial_import_preserved':DEST.exists(),'returned_code_executed':False})
        raise
