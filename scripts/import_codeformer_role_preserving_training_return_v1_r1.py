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
PREFIX = 'cctv-codeformer-role-preserving-training-v1-r1'
VM_NAME = 'cctv_codeformer_role_preserving_training_v1_r1_vm'
OUTPUT = 'outputs/role_preserving_training_v1_r1'
PROTOCOL = '4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40'
ARCHIVE_SHA = '075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1'
ARCHIVE_BYTES, MEMBERS, PAYLOAD_BYTES = 34174817, 467, 46082287
PAYLOAD_CAP, MEMBER_CAP, METADATA_CAP = 64*1024**2, 500, 16*1024**2
RESERVE, JSON_CAP, LOG_CAP, PNG_CAP, ARRAY_CAP = 4*1024**3, 2*1024**2, 2*1024**2, 198656, 4*1024**2
TAR_CAP = PAYLOAD_CAP + MEMBER_CAP*1023 + 10240
REVIEW = ROOT/'outputs/codeformer_role_preserving_training_v1_r1_return_review'
DEST = ROOT/'outputs/codeformer_role_preserving_training_v1_r1_return'
STAGE = ROOT/'outputs'/VM_NAME
MANIFEST = OUTPUT+'/export_manifest.json'
MANIFEST_SHA = 'ed21a65e43b789c1c50985390541d03a9cdf5c9df29a41b68474504c0934ffcd'
COUNTS = {'optimizer_step_attempts':0,'backward_calls':0,'gradient_queries':0,
 'route_witness_forwards':0,'route_witness_behavior_teacher_forwards':0,
 'route_witness_perceptual_forwards':0,'perceptual_prerequisite_forwards':0,
 'candidate_training_forwards':0,'perceptual_training_forwards':0,
 'original_behavior_teacher_forwards':0,'parent_behavior_teacher_forwards':0,
 'teacher_forwards':100,'snapshot_forwards':0,
 'initializer_parity_candidate_forwards':1,'initializer_parity_parent_forwards':1,
 'initializer_training_candidate_forwards':0,'initializer_parity_perceptual_forwards':0,'training_input_writes':0}
ZERO = ('completed_training_updates','partial_microbatches','new_restorer_updates',
 'gradient_query_reservations','backward_call_reservations','optimizer_constructions_completed',
 'optimizer_steps_completed','optimizer_step_attempts','committed_training_updates','pending_microbatches_not_committed')
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
    ids=protocol['selected_train_reference_ids']
    need(len(ids)==len(set(ids))==100,'Exact100reference namespace differs')
    expected=set(protocol['files_sha256'])|{'protocol.json'}
    expected.update(OUTPUT+'/HQ_cache/'+row+'/'+name for row in ids
        for name in ('indices.npy','features.npy','record.json'))
    expected.update(OUTPUT+'/call_trace/%05d.json'%n for n in range(1,103))
    expected.update(OUTPUT+'/initializer_parity/train_u0003_slot1/'+stem+name
        for stem in ('','parent/') for name in ('native.npy','raw.npy','logits.npy','latent.npy','tokens.npy'))
    expected.update(OUTPUT+'/initializer_parity/train_u0003_slot1/'+name for name in ('record.json','context.json'))
    expected.update(OUTPUT+'/gradient_route_witness/rng_'+label+suffix
        for label in ('initializer_before','initializer_slot1_after')
        for suffix in ('.json','_CPU.npy','_CUDA0.npy'))
    expected.update(OUTPUT+'/'+name for name in ('codebook_compatibility.json',
        'dependency_verification.json','export_manifest.json','factory.json','failure.json',
        'idle_verified.json','initial_frozen_state.json','input_index.json',
        'parameter_census.json','runtime.json','transfer_verified.json'))
    expected.update('outputs/supervision/'+name for name in
        ('diagnostic.log','export_invocation.json','preflight.log','run_receipt.json'))
    need(len(expected)==MEMBERS,'Exact partial467member return namespace differs')
    return expected


def failure_fields(doc):
    need(doc['failure_present']is True and doc['complete_training_process']is False and
        doc['worker_terminal_evidence']==OUTPUT+'/failure.json' and
        doc['completed_training_updates']==0 and doc['partial_microbatches']==0 and
        doc['optimizer_inflight']is False and doc['worker_recorded_counts']==COUNTS and
        doc['gradient_queries']==doc['backward_calls']==doc['optimizer_step_attempts']==0 and
        doc['activity_unknown_without_worker_receipt']is False and doc['checkpoint_members']==[] and
        doc['stopped_checkpoint_alias']is None and doc['model_qualified']is False and
        doc['training_or_export_completion_not_model_acceptance']is True,
        'Exact failed zero-update transport activity differs')
    coverage=doc['output_coverage']
    need(coverage['complete']is False and coverage['expected_members']==4138 and
        len(coverage['missing_members'])==len(set(coverage['missing_members']))==3838 and
        coverage['snapshot_RAW_PNG_pairs']==0,'Exact incomplete planned output coverage differs')


def main():
    started=time.monotonic();outputs=ROOT/'outputs'
    archive=unlinked(outputs/(PREFIX+'-results.tar.gz'),True)
    checksum=outputs/(PREFIX+'-results.tar.gz.sha256');ep=outputs/(PREFIX+'-export.json');fp=outputs/(PREFIX+'-supervisor-final.json')
    receipt_path=REVIEW/'return_import_v1.json';unlinked(DEST);unlinked(REVIEW)
    need(not DEST.exists() and not receipt_path.exists() and not (REVIEW/'transport_v1.json').exists(),
        'Preserve previous/partial import and audit')
    REVIEW.mkdir(exist_ok=True)
    companion_pins={path.name:file_pin(path)for path in (checksum,ep,fp)}
    exported=load(ep);final=load(fp)
    need(exported['format']=='codeformer-role-preserving-training-export-v1-r1' and exported['complete']is True and
        exported['archive_sha256']==ARCHIVE_SHA and exported['bytes']==ARCHIVE_BYTES and
        exported['protocol_sha256']==PROTOCOL and exported['manifest_member']==MANIFEST and
        exported['manifest_sha256']==MANIFEST_SHA and exported['regular_members']==MEMBERS and
        exported['payload_bytes']==PAYLOAD_BYTES,'Exact export identity differs')
    failure_fields(exported)
    need(read(checksum,1024)==(ARCHIVE_SHA+'  '+archive.name+'\n').encode('ascii'),
        'ASCII basename/singleLF checksum differs')
    ap=file_pin(archive);need(ap=={'sha256':ARCHIVE_SHA,'bytes':ARCHIVE_BYTES},'Fresh actual archive hash/size differs')
    need(shutil.disk_usage(outputs).free>=PAYLOAD_CAP+RESERVE,'Import admission needs64MiB plus4GiB reserve')
    pb=read(STAGE/'protocol.json');need(sha(pb)==PROTOCOL,'Frozen local stage protocol differs');p=parse(pb)
    need(p['vm_name']==VM_NAME and p['outputs']==OUTPUT and p['export_prefix']==PREFIX and
        len(p['files_sha256'])+1==32 and p['files_bytes'].keys()==p['files_sha256'].keys(),
        'Frozen32static protocol fields differ')
    expected=names(p)
    need(final['format']=='codeformer-role-preserving-supervisor-final-v1-r1' and final['complete']is True and
        final['protocol_sha256']==PROTOCOL and final['archive_sha256']==ARCHIVE_SHA and final['archive_bytes']==ARCHIVE_BYTES and
        final['export_receipt_sha256']==companion_pins[ep.name]['sha256'] and
        final['export_receipt_bytes']==companion_pins[ep.name]['bytes'] and
        final['checksum_sha256']==companion_pins[checksum.name]['sha256'] and final['model_qualified']is False,
        'External final identity differs')
    log=base64.b64decode(final['export_log_base64'],validate=True)
    need(len(log)==final['export_log_bytes']<=LOG_CAP and sha(log)==final['export_log_sha256'] and
        parse(log.strip())==exported,'External export log exact bytes/receipt differs')
    finished=final['finished_json']
    need(sha(encode(finished))==final['finished_json_sha256'] and finished['stages']==final['stages'] and
        finished['protocol_sha256']==PROTOCOL and finished['complete']is True and
        finished['worker_terminal']is True,'External finished receipt closure differs')
    need([(row['stage'],row['external_cap_seconds'],row['grace_seconds'],row['exit_code'])for row in final['stages']]==
        [('--verify-transfer',300,30,0),('--run',3000,30,1),('--export',600,30,0)] and
        all(row['worker_terminal']is True and row['child_process_group_terminal']is True and
            not row['log_capped'] and not row['outer_watchdog_expired'] and row['supervisor_exception']is None
            for row in final['stages']),'Finite supervisor failure stages differ')
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
            def field(start,width):return raw[start:start+width].split(b'\0',1)[0].decode('ascii')
            name=field(0,100);prefix=field(345,155);name=prefix+'/'+name if prefix else name;portable(name)
            sizefield=raw[124:136]
            need(re.fullmatch(b'[0-7]{11}\0',sizefield),'Canonical octal TAR size required');size=int(sizefield[:-1],8)
            need(raw==header(name,size),'Canonical512byte USTAR header fields/checksum differ')
            need(name in expected and name not in hashes and name.casefold()not in folded and name>previous,
                'Exact sorted unique return namespace differs')
            previous=name;folded.add(name.casefold());need(len(folded)<=MEMBER_CAP,'File count cap exceeded')
            suffix=Path(name).suffix.lower()
            need(not re.search(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin|npz)(?:$|[.\-])',Path(name).name.lower()),
                'Checkpoint/model/binary forbidden in this exact zero-update return')
            cap=ARRAY_CAP if suffix=='.npy'else PNG_CAP if suffix=='.png'else JSON_CAP
            need(size<=cap,'Per-file cap exceeded')
            if suffix not in ('.npy','.png'):metadata+=size;need(metadata<=METADATA_CAP,'Metadata cap exceeded')
            total+=size;need(total<=PAYLOAD_CAP,'Payload cap exceeded')
            target=unlinked(DEST.joinpath(*PurePosixPath(name).parts))
            need(target.resolve(strict=False).is_relative_to(DEST.resolve()),'Destination escapes importroot')
            need(shutil.disk_usage(outputs).free>=RESERVE+size+65536,'Import write reserve reached')
            target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();remaining=size
            with target.open('xb')as stream:
                while remaining:
                    block=strict.exact(min(remaining,1024**2));stream.write(block);h.update(block);remaining-=len(block)
            padding=(-size)%512;need(strict.exact(padding)==b'\0'*padding,'Nonzero TAR member padding')
            position+=512+size+padding;hashes[name]=h.hexdigest();sizes[name]=size
        need(strict.ended and strict.compressed==ARCHIVE_BYTES and strict.digest.hexdigest()==ARCHIVE_SHA,
            'Stream compressed digest/CRC/fullEOF differs')
        with archive.open('rb')as stream:
            need(stream.read(10)==b'\x1f\x8b\x08\0\0\0\0\0\0\xff','Canonical level6 gzip header differs')
            stream.seek(-8,2);crc,isize=struct.unpack('<II',stream.read(8))
        need(crc==strict.crc and isize==strict.expanded%(2**32),'Actual trailer CRC/ISIZE differs')
    finally:strict.close()
    after=archive.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
        'Archive changed during import')
    need(set(hashes)==expected and len(hashes)==MEMBERS and total==PAYLOAD_BYTES and hashes['protocol.json']==PROTOCOL and
        hashes[MANIFEST]==MANIFEST_SHA,'Exact actual return census/fingerprints differ')
    manifest=load(DEST/MANIFEST)
    need(manifest['format']=='codeformer-role-preserving-training-return-v1-r1' and manifest['protocol_sha256']==PROTOCOL and
        all(manifest[key]==p[key]for key in ('parent_dir','parent_protocol_sha256','dependencies',
            'retained_output_dependencies','extra_returned_dependencies')),'Manifest parent/format/scope differs')
    failure_fields(manifest)
    need(manifest['output_coverage']==exported['output_coverage'] and
        set(manifest['files_sha256'])==set(manifest['files_bytes'])==expected-{MANIFEST} and
        all(hashes[name]==h and sizes[name]==manifest['files_bytes'][name]for name,h in manifest['files_sha256'].items()),
        'Every actual member must matchmanifest')
    for name,h in p['files_sha256'].items():
        need(hashes[name]==h and sizes[name]==p['files_bytes'][name] and
            file_pin(STAGE/name)=={'sha256':h,'bytes':sizes[name]},'Frozen static32 bytes differ')
    failure=load(DEST/(OUTPUT+'/failure.json'));runtime=load(DEST/(OUTPUT+'/runtime.json'))
    need(failure['complete']is True and failure['failure']is True and failure['protocol_sha256']==PROTOCOL and
        failure['counts']==COUNTS and all(type(failure['counts'][key])is int for key in COUNTS) and
        all(type(failure[key])is int and failure[key]==0 for key in ZERO) and
        failure['optimizer_inflight']is False and failure['candidate_state_valid']is True and
        failure['historical_parent_updates']==50 and failure['checkpoints']=={} and failure['last_safe_checkpoint']is None and
        failure['optimizer_state_uncertain']is False and failure['stopped_checkpoint_error']is None and
        failure['reason']=='ValueError: Matched initializer inference parity failed; actual arrays and context retained' and
        failure['training_seconds']==0 and failure['model_qualified']is False and failure['app_promoted']is False and
        failure['automatic_retry']is False and failure['automatic_resume']is False,
        'Authoritative pre-optimizer matched-inference initialization failure differs')
    need(final['worker_recorded_counts']==COUNTS and final['checkpoint_members']==[] and final['checkpoint_exported']is False and
        final['optimizer_inflight']is False and all(type(final[key])is int and final[key]==0 for key in
            ('completed_training_updates','partial_microbatches','optimizer_step_attempts','committed_training_updates',
             'gradient_queries','backward_calls')),'External final actual activity differs')
    supervised=load(DEST/'outputs/supervision/run_receipt.json')
    need(supervised['protocol_sha256']==PROTOCOL and supervised['stages']==final['stages'][:2] and
        supervised['worker_terminal']is True and supervised['child_process_group_terminal']is True and
        supervised['training_process_exit_code']==1 and supervised['training_process_completed']is False and
        all(supervised[key]is None and finished[key]is None for key in
            ('completed_training_updates','partial_microbatches','optimizer_inflight')),
        'Archived/final supervisor stages or explicit unknown-before-export differ')
    for row in final['stages'][:2]:need(sizes[row['log']]==row['log_bytes'],'Archived supervisorlog byte count differs')
    need(final['stages'][2]['log_bytes']==len(log),'Final exportstage log bytes differ')
    resources=failure['resources']
    need(resources['peak_RSS_bytes']<=p['limits']['RSS_bytes'] and
        resources['peak_CUDA_reserved_bytes']<=p['limits']['CUDA_reserved_bytes'] and
        resources['minimum_sampled_free_bytes']>=RESERVE and failure['seconds']<=p['limits']['worker_seconds'] and
        set(failure['phase_seconds'])=={'initialization'} and
        0<=failure['phase_seconds']['initialization']<=p['limits']['initialization_seconds'],
        'Recorded runtimecaps exceeded')
    need(not any(name.split('.')[0]in ('torch','numpy','tensorflow','jax')for name in sys.modules),
        'No framework execution invariant failed')
    receipt={'format':'codeformer-role-preserving-training-local-import-v1-r1','complete':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'import_root':str(DEST),'archive_sha256':ARCHIVE_SHA,
        'archive_bytes':ARCHIVE_BYTES,'protocol_sha256':PROTOCOL,'manifest_sha256':hashes[MANIFEST],
        'files_sha256':hashes,'files_bytes':sizes,'regular_files':len(hashes),'payload_bytes':total,
        'metadata_bytes':metadata,'expanded_tar_bytes':strict.expanded,'companion_pins':companion_pins,
        'all32static_files_exact':True,'single_gzip_CRC_manual_USTAR_manifest_companions_pass':True,
        'same_owner_importer_and_transport_auditor':True,'returned_code_executed':False,'checkpoint_loaded':False,
        'actual_local_neural_gradient_optimizer_VM_calls':0,'actual_returned_counts':COUNTS,
        'training_process_complete':False,'failure_present':True,'new_restorer_updates':0,'new_checkpoints':0,
        'model_qualified':False,'source_sha256':file_pin(__file__)['sha256'],'seconds':time.monotonic()-started}
    write_new(receipt_path,receipt)
    transport={'format':'codeformer-role-preserving-training-transport-audit-v1-r1','complete':True,'transport_pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,
        'protocol_sha256':PROTOCOL,'manifest_sha256':hashes[MANIFEST],'regular_members':len(hashes),'payload_bytes':total,
        'expanded_TAR_bytes':strict.expanded,'gzip_CRC32':crc,'gzip_ISIZE':isize,
        'single_gzip_frame_trailer_CRC_full_EOF':True,'manual512USTAR_headers_order_padding_double_EOA_full_EOF':True,
        'all32frozen_static_bytes_exact':True,'all4companions_closed':True,'external_finished_bytes_reconstructed_exactly':True,
        'return_import_receipt':file_pin(receipt_path),'same_owner_importer_and_transport_auditor':True,
        'independent_root_crosscheck_required':True,'actual_counts':COUNTS,'call_reservations':102,
        'reservation_counts_not_completed_profiler_evidence':True,'new_restorer_updates':0,'checkpoint_members':[],
        'recorded_resources':resources,'recorded_phase_seconds':failure['phase_seconds'],'recorded_worker_seconds':failure['seconds'],
        'supervisor_stage_seconds':{row['stage']:row['seconds']for row in final['stages']},'runtime':runtime,
        'model_qualified':False,'NN_VM_optimizer_training_calls_by_audit':0,'source_sha256':receipt['source_sha256'],
        'blockers':['Training failed before optimizer construction at exact initializer forward parity.'],
        'scope':'Same-owner importer/transport audit; safe intact failure evidence, not training success or model qualification. Scientific/source cause review remains separate.'}
    write_new(REVIEW/'transport_v1.json',transport)
    report=('Transport/import PASS: 34,174,817 archive bytes; 467 exact regular members; 46,082,287 payload bytes. '
        'Fresh full hash, single gzip CRC/trailer/full EOF, manual canonical USTAR headers/order/padding/double end markers, '
        'all manifest/static32 bytes and four companions close.\n\n'
        'Training FAILED before optimizer construction: zero new restorer updates, zero Adam constructions, gradients and optimizer steps; '
        'no candidate checkpoint or development restoration outputs were produced. Partial HQ-cache and initialization evidence is preserved. '
        '102 call reservations are not a completed-forward profiler claim.\n\n'
        'Importer/transport auditor have the same owner; independent root archive/source crosscheck remains required. '
        'No returned code, arrays, framework, model or checkpoint was executed/deserialized. No VM operation occurred.\n')
    with (REVIEW/'transport_v1.md').open('x',encoding='utf-8',newline='\n')as stream:stream.write(report)
    print(json.dumps({'complete':True,'import_root':str(DEST),'files':len(hashes),'payload_bytes':total,
        'seconds':time.monotonic()-started,'training_failed':True,'new_restorer_updates':0,'model_qualified':False}))


if __name__=='__main__':
    try:main()
    except BaseException:
        REVIEW.mkdir(exist_ok=True);failure=REVIEW/'import_failure_v1.json'
        if not failure.exists():
            with failure.open('x',encoding='utf-8',newline='\n')as stream:
                json.dump({'complete':False,'traceback':traceback.format_exc(),'source_sha256':file_pin(__file__)['sha256'],
                    'partial_import_preserved':DEST.exists(),'returned_code_executed':False},stream,indent=2);stream.write('\n')
        raise


