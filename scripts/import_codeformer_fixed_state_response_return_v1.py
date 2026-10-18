"""Hash-pinned inert return import and same-owner stdlib transport/runtime audit.

Never import returned Python, NumPy, Torch, model or checkpoint contents.
All destinations are exclusive; failed/partial imports remain evidence.
"""
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
import math
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

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'cctv-codeformer-fixed-state-response-v1'
VM_NAME = 'cctv_codeformer_fixed_state_response_v1_vm'
OUTPUT = 'outputs/fixed_state_response_v1'
PROTOCOL = '3ea4f1fe9ce90f3d29d29d6ace108eae7a809e6665cb770811f36773b888c089'
ARCHIVE_SHA = 'e60258356e148ae426dcc8863794ad62377559f858a2b5dbf75c3e0eee0d50ad'
ARCHIVE_BYTES, MEMBERS, PAYLOAD_BYTES = 516081377, 2536, 583143654
PAYLOAD_CAP, MEMBER_CAP, METADATA_CAP = 1024**3, 3000, 128*1024**2
RESERVE, JSON_CAP, LOG_CAP, PNG_CAP, ARRAY_CAP = 4*1024**3, 2*1024**2, 2*1024**2, 198656, 1024**2
TAR_CAP = PAYLOAD_CAP + MEMBER_CAP*1023 + 10240
REVIEW = ROOT/'outputs/codeformer_fixed_state_response_v1_return_review'
DEST = ROOT/'outputs/codeformer_fixed_state_response_v1_return'
STAGE = ROOT/'outputs'/VM_NAME
MANIFEST = OUTPUT+'/export_manifest.json'
COUNTS = {'CodeFormer_new_original':64,'CodeFormer_new_full_coarse':64,
    'controller_fitted_initial':200,'controller_fitted_final':200,
    'controller_heldout_initial':64,'controller_heldout_final':64,'gradient_queries':0,
    'backward_calls':0,'optimizer_constructions':0,'optimizer_steps':0,'committed_training_updates':0}
ZERO = ('head_updates_this_run','restorer_updates_this_run','gradient_queries','backward_calls',
        'optimizer_constructions','optimizer_steps')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)',re.I)


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


def names(protocol):
    fitted=load(STAGE/protocol['fitted_index_file']);heldout=load(STAGE/protocol['new_pairs_file'])
    ids=[r['id']for r in fitted['cases']]+[r['id']for r in heldout['cases']]
    need(len(ids)==len(set(ids))==264 and len(fitted['cases'])==200 and len(heldout['cases'])==64,'Frozen264row namespace differs')
    expected=set(protocol['files_sha256'])|{'protocol.json'}
    for bank in ('initial','final'):
        expected.update(OUTPUT+'/'+bank+'/'+row+'/'+n for row in ids for n in ('raw.npy','native.png','record.json'))
    expected.update(OUTPUT+'/new/'+r['id']+'/'+n for r in heldout['cases']for n in ('original_raw.npy','base_raw.npy'))
    expected.update(OUTPUT+'/trace/%04d.json'%n for n in range(1,657))
    expected.update(OUTPUT+'/'+n for n in ('transfer_verified.json','idle_verified.json','input_index.json','runtime.json',
        'factory.json','initial_state.json','final_state.json','response_summary.json','run_results.json',
        'dependency_verification.json','export_manifest.json'))
    expected.update('outputs/supervision/'+n for n in ('preflight.log','diagnostic.log','run_receipt.json','export_invocation.json'))
    need(len(expected)==MEMBERS,'Exact completed return namespace differs')
    return expected


def main():
    started=time.monotonic();outputs=ROOT/'outputs';archive=unlinked(outputs/(PREFIX+'-results.tar.gz'),True)
    checksum=outputs/(PREFIX+'-results.tar.gz.sha256');ep=outputs/(PREFIX+'-export.json');fp=outputs/(PREFIX+'-supervisor-final.json')
    receipt_path=REVIEW/'return_import_v1.json'
    unlinked(DEST);unlinked(REVIEW)
    need(not DEST.exists() and not receipt_path.exists() and not (REVIEW/'independent_transport_v1.json').exists(),'Preserve previous/partial import and audit')
    REVIEW.mkdir(exist_ok=True)
    companion_pins={p.name:file_pin(p)for p in (checksum,ep,fp)}
    exported=load(ep);final=load(fp)
    need(exported['format']=='codeformer-fixed-state-response-export-v1' and exported['complete']is True and
        exported['archive_sha256']==ARCHIVE_SHA and exported['bytes']==ARCHIVE_BYTES and exported['protocol_sha256']==PROTOCOL and
        exported['manifest_member']==MANIFEST and exported['regular_members']==MEMBERS and exported['payload_bytes']==PAYLOAD_BYTES and
        exported['complete_measurement']is True and exported['failure_present']is False and exported['model_qualified']is False,'Exact export identity differs')
    need(read(checksum,1024)==(ARCHIVE_SHA+'  '+archive.name+'\n').encode('ascii'),'ASCII basename/singleLF checksum differs')
    ap=file_pin(archive);need(ap=={'sha256':ARCHIVE_SHA,'bytes':ARCHIVE_BYTES},'Fresh actual archive hash/size differs')
    need(shutil.disk_usage(outputs).free>=PAYLOAD_CAP+RESERVE,'Import admission needs1GiB plus4GiB reserve')
    pb=read(STAGE/'protocol.json');need(sha(pb)==PROTOCOL,'Frozen local stage protocol differs');p=parse(pb)
    need(p['vm_name']==VM_NAME and p['outputs']==OUTPUT and p['export_prefix']==PREFIX and p['call_caps']==COUNTS and
        len(p['files_sha256'])+1==153,'Frozen source protocol fields differ')
    expected=names(p)
    need(final['format']=='codeformer-fixed-state-response-supervisor-final-v1' and final['complete']is True and
        final['protocol_sha256']==PROTOCOL and final['archive_sha256']==ARCHIVE_SHA and final['archive_bytes']==ARCHIVE_BYTES and
        final['export_receipt_sha256']==companion_pins[ep.name]['sha256'] and final['export_receipt_bytes']==companion_pins[ep.name]['bytes'] and
        final['checksum_sha256']==companion_pins[checksum.name]['sha256'] and final['model_qualified']is False,'External final identity differs')
    log=base64.b64decode(final['export_log_base64'],validate=True)
    need(len(log)==final['export_log_bytes']<=LOG_CAP and sha(log)==final['export_log_sha256'] and parse(log.strip())==exported,'External export log exact bytes/receipt differs')
    finished=final['finished_json'];need(sha(encode(finished))==final['finished_json_sha256'] and finished['stages']==final['stages'] and
        finished['protocol_sha256']==PROTOCOL and finished['complete']is True and finished['worker_terminal']is True,'External finished receipt closure differs')
    need([(s['stage'],s['external_cap_seconds'],s['grace_seconds'])for s in final['stages']]==
        [('--verify-transfer',300,30),('--run',900,30),('--export',300,30)] and
        all(s['exit_code']==0 and s['worker_terminal']is True and s['child_process_group_terminal']is True and
            not s['log_capped'] and not s['outer_watchdog_expired'] and s['supervisor_exception']is None for s in final['stages']),'Finite supervisor stages differ')
    DEST.mkdir();hashes={};sizes={};folded=set();total=metadata=position=0;previous=''
    before=archive.stat();strict=StrictGzip(archive)
    try:
        while True:
            raw=strict.exact(512)
            if raw==b'\0'*512:
                need(strict.exact(512)==b'\0'*512,'Two physical TAR end markers required')
                tail=((position+1024+10239)//10240)*10240-position-1024
                need(strict.exact(tail)==b'\0'*tail and strict.read(1)==b'','Noncanonical TAR EOA/padding/fullEOF')
                break
            def field(start,width):
                value=raw[start:start+width];return value.split(b'\0',1)[0].decode('ascii')
            name=field(0,100);prefix=field(345,155);name=prefix+'/'+name if prefix else name;portable(name)
            sizefield=raw[124:136];need(re.fullmatch(b'[0-7]{11}\0',sizefield),'Canonical octal TAR size required');size=int(sizefield[:-1],8)
            need(raw==header(name,size),'Canonical512byte USTAR header fields/checksum differ')
            need(name in expected and name not in hashes and name.casefold()not in folded and name>previous,'Exact sorted unique return namespace differs')
            previous=name;folded.add(name.casefold());need(len(folded)<=MEMBER_CAP,'File count cap exceeded')
            suffix=Path(name).suffix.lower()
            need(not re.search(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin|npz)(?:$|[.\-])',Path(name).name.lower()),'New checkpoint/model/binary forbidden')
            cap=ARRAY_CAP if suffix=='.npy'else PNG_CAP if suffix=='.png'else JSON_CAP
            need(size<=cap,'Per-file cap exceeded')
            if suffix not in ('.npy','.png'):metadata+=size;need(metadata<=METADATA_CAP,'Metadata cap exceeded')
            total+=size;need(total<=PAYLOAD_CAP,'Payload cap exceeded')
            target=unlinked(DEST.joinpath(*PurePosixPath(name).parts));need(target.resolve(strict=False).is_relative_to(DEST.resolve()),'Destination escapes importroot')
            need(shutil.disk_usage(outputs).free>=RESERVE+size+65536,'Import write reserve reached')
            target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();remaining=size
            with target.open('xb')as stream:
                while remaining:
                    block=strict.exact(min(remaining,1024**2));stream.write(block);h.update(block);remaining-=len(block)
            padding=(-size)%512;need(strict.exact(padding)==b'\0'*padding,'Nonzero TAR member padding')
            position+=512+size+padding;hashes[name]=h.hexdigest();sizes[name]=size
        need(strict.ended and strict.compressed==ARCHIVE_BYTES and strict.digest.hexdigest()==ARCHIVE_SHA,'Stream compressed digest/CRC/fullEOF differs')
        with archive.open('rb')as stream:
            need(stream.read(10)==b'\x1f\x8b\x08\0\0\0\0\0\0\xff','Canonical level6 gzip header differs')
            stream.seek(-8,2);crc,isize=struct.unpack('<II',stream.read(8))
        need(crc==strict.crc and isize==strict.expanded%(2**32),'Actual trailer CRC/ISIZE differs')
    finally:strict.close()
    after=archive.stat();need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Archive changed during import')
    need(set(hashes)==expected and len(hashes)==MEMBERS and total==PAYLOAD_BYTES and hashes['protocol.json']==PROTOCOL and
        hashes[MANIFEST]==exported['manifest_sha256'],'Exact actual return census/fingerprints differ')
    manifest=load(DEST/MANIFEST)
    need(manifest['format']=='codeformer-fixed-state-response-return-v1' and manifest['protocol_sha256']==PROTOCOL and
        manifest['model_qualified']is False and manifest['new_checkpoint_serialization']is False and
        all(manifest[k]==p[k]for k in ('parent_roots','parent_protocol_sha256','dependencies')),'Manifest parent/format/scope differs')
    need(set(manifest['files_sha256'])==set(manifest['files_bytes'])==expected-{MANIFEST} and
        all(hashes[n]==h and sizes[n]==manifest['files_bytes'][n]for n,h in manifest['files_sha256'].items()),'Every actual member must matchmanifest')
    for name,h in p['files_sha256'].items():
        need(hashes[name]==h and sizes[name]==p['files_bytes'][name] and file_pin(STAGE/name)=={'sha256':h,'bytes':sizes[name]},'Frozen static153 bytes differ')
    run=load(DEST/(OUTPUT+'/run_results.json'));runtime=load(DEST/(OUTPUT+'/runtime.json'))
    need(run['complete']is True and run['measurement_complete']is True and run['protocol_sha256']==PROTOCOL and
        run['counts']==COUNTS and all(type(run['counts'][k])is int for k in COUNTS) and run['finished_neural_calls']==656 and
        all(type(run[k])is int and run[k]==0 for k in ZERO+('committed_training_updates',)) and
        run['new_checkpoints_written']is False and run['model_qualified']is False,'Authoritative zero-update completion differs')
    for doc in (exported,manifest):
        need(doc['complete_measurement']is True and doc['failure_present']is False and doc['worker_recorded_counts']==COUNTS and
            doc['checkpoint_members']==[] and doc['counts_unknown_without_worker_receipt']is False and
            all(type(doc[k])is int and doc[k]==0 for k in ZERO),'Transport/runtime counters differ')
        coverage=doc['output_coverage'];need(coverage['complete']is True and coverage['missing_members']==[] and
            coverage['head_RAW_PNG_pairs']==528 and coverage['row_records']==528 and coverage['fresh_parent_RAW_arrays']==128,'Completed coverage differs')
    need(final['completed_measurement']is True and final['checkpoint_members']==[] and final['checkpoint_exported']is False and
        all(type(final[k])is int and final[k]==0 for k in ZERO),'External final actual activity differs')
    supervised=load(DEST/'outputs/supervision/run_receipt.json')
    need(supervised['protocol_sha256']==PROTOCOL and supervised['stages']==final['stages'][:2] and supervised['worker_terminal']is True and
        supervised['child_process_group_terminal']is True,'Archived/final supervisor stages differ')
    for item in final['stages'][:2]:need(sizes[item['log']]==item['log_bytes'],'Archived supervisorlog byte count differs')
    need(final['stages'][2]['log_bytes']==len(log),'Final exportstage log bytes differ')
    resources=run['resources'];need(resources['peak_RSS_bytes']<=8*1024**3 and resources['peak_CUDA_reserved_bytes']<=22*1024**3 and
        resources['minimum_sampled_free_bytes']>=RESERVE and run['seconds']<=900 and
        all(0<=run['phase_seconds'][k]<=cap for k,cap in {'data':300,'initialization':300,'measurement':600}.items()),'Recorded runtimecaps exceeded')
    need(not any(n.split('.')[0]in ('torch','numpy','tensorflow','jax')for n in sys.modules),'No framework execution invariant failed')
    receipt={'format':'codeformer-fixed-state-response-local-import-v1','complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'import_root':str(DEST),'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,'protocol_sha256':PROTOCOL,
        'manifest_sha256':hashes[MANIFEST],'files_sha256':hashes,'files_bytes':sizes,'regular_files':len(hashes),'payload_bytes':total,
        'metadata_bytes':metadata,'expanded_tar_bytes':strict.expanded,'companion_pins':companion_pins,
        'all153static_files_exact':True,'single_gzip_CRC_manual_USTAR_manifest_companions_pass':True,
        'same_owner_importer_and_transport_auditor':True,'returned_code_executed':False,'checkpoint_loaded':False,
        'actual_local_neural_gradient_optimizer_VM_calls':0,'actual_returned_counts':run['counts'],
        'measurement_complete':True,'head_updates_this_run':0,'restorer_updates_this_run':0,'model_qualified':False,
        'source_sha256':file_pin(__file__)['sha256'],'seconds':time.monotonic()-started}
    write_new(receipt_path,receipt)
    transport={'format':'codeformer-fixed-state-response-transport-audit-v1','complete':True,'transport_pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,
        'protocol_sha256':PROTOCOL,'manifest_sha256':hashes[MANIFEST],'regular_members':len(hashes),'payload_bytes':total,
        'expanded_TAR_bytes':strict.expanded,'gzip_CRC32':crc,'gzip_ISIZE':isize,
        'single_gzip_frame_trailer_CRC_full_EOF':True,'manual512USTAR_headers_order_padding_double_EOA_full_EOF':True,
        'all153frozen_static_bytes_exact':True,'all4companions_closed':True,'external_finished_bytes_reconstructed_exactly':True,
        'return_import_receipt':file_pin(receipt_path),'same_owner_importer_and_transport_auditor':True,
        'independent_root_crosscheck_required':True,'actual_counts':run['counts'],'finished_neural_calls':run['finished_neural_calls'],
        'head_updates_this_run':0,'restorer_updates_this_run':0,'checkpoint_members':[],
        'recorded_resources':resources,'recorded_phase_seconds':run['phase_seconds'],'recorded_worker_seconds':run['seconds'],
        'supervisor_stage_seconds':{s['stage']:s['seconds']for s in final['stages']},'runtime':runtime,
        'model_qualified':False,'NN_VM_optimizer_training_calls_by_audit':0,'source_sha256':receipt['source_sha256'],'blockers':[],
        'scope':'Same-owner importer/transport audit. Integrity and zero-update execution only; scientific pixel/state/response audit and model qualification remain separate.'}
    write_new(REVIEW/'independent_transport_v1.json',transport)
    report=('Transport/import PASS: 516,081,377 archive bytes; 2,536 exact regular members; 583,143,654 payload bytes. Fresh hash before import, single gzip CRC/trailer/full EOF, manual canonical USTAR fields/order/padding/double end markers, all manifest/static153 bytes and four companions close.\n\n'
        'The authoritative runtime records 656 neural forwards, zero gradients/optimizer/training updates and no new checkpoints. Measurement completion does not qualify the model. Detailed recorded time/resource fields are in the JSON audit.\n\n'
        'This importer and transport auditor have the same owner; independent root archive/source crosscheck is still required. Returned files were copied as inert bytes; no returned Python, array/model/checkpoint load, NN or VM operation occurred. Pixel/state/response reviews remain separate.\n')
    with (REVIEW/'independent_transport_v1.md').open('x',encoding='utf-8',newline='\n')as stream:stream.write(report)
    print(json.dumps({'complete':True,'import_root':str(DEST),'files':len(hashes),'payload_bytes':total,'seconds':time.monotonic()-started,'model_qualified':False}))


if __name__=='__main__':
    try:main()
    except BaseException:
        REVIEW.mkdir(exist_ok=True)
        failure=REVIEW/'import_failure_v1.json'
        if not failure.exists():
            with failure.open('x',encoding='utf-8',newline='\n')as stream:
                json.dump({'complete':False,'traceback':traceback.format_exc(),'source_sha256':file_pin(__file__)['sha256'],
                    'partial_import_preserved':DEST.exists(),'returned_code_executed':False},stream,indent=2);stream.write('\n')
        raise
