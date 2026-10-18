"""Standard-library transfer checks and bounded exclusive role-preserving training export.

No returned code, tensor file, model, optimizer or neural framework is loaded.
Parents remain read-only. Only the exact candidate checkpoint slots and retained partials are admitted.
"""
from datetime import datetime, timezone
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tarfile
import time
import unicodedata

from . import contract as c


OUTPUTS, PREFIX, MODULE = c.OUTPUTS, c.PREFIX, c.MODULE
PAYLOAD_CAP = c.LIMITS['result_payload_bytes']
ARCHIVE_CAP = c.LIMITS['archive_bytes']
METADATA_CAP = c.LIMITS['metadata_bytes']
MEMBER_CAP = c.LIMITS['regular_file_cap']
RESERVE = c.LIMITS['reserve_free_bytes']
EXPORT_SECONDS = c.LIMITS['export_seconds']
HEX = re.compile(r'^[a-f0-9]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)', re.I)
BINARY_SUFFIXES = {'.npy', '.npz', '.png'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.pkl', '.pickle', '.bin'}
SUPER_FILE_NAMES = {'preflight.log', 'diagnostic.log', 'run_receipt.json', 'export_invocation.json',
                    'export.log', 'finished.json'}


def canonical(name):
    c.require(isinstance(name, str) and bool(name), 'Missing portable relative path')
    p = PurePosixPath(name)
    c.require(not p.is_absolute() and p.as_posix() == name and p.parts and
              all(part not in ('', '.', '..') for part in p.parts), 'Noncanonical relative path')
    c.require(unicodedata.normalize('NFC', name) == name and
              not any(char in name for char in '\\:"<>|?*') and
              all(not DEVICE.match(part) and not part.endswith((' ', '.')) and
                  len(part.encode('utf-16-le')) <= 510 and
                  all(ord(char) >= 32 for char in part) for part in p.parts), 'Nonportable relative path')
    return p


def _no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            c.require(not stat.S_ISLNK(info.st_mode) and
                      not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked/reparse path forbidden')
    return path


def _regular(root, name, *, unique_link=True):
    canonical(name)
    root = _no_links(root)
    path = _no_links(c.safe_path(root, name))
    info = path.lstat()
    c.require(stat.S_ISREG(info.st_mode), 'Nonregular file forbidden: ' + name)
    c.require(not unique_link or info.st_nlink == 1, 'Hard-linked packet/output forbidden: ' + name)
    return path


def _walk(root):
    """Inventory every new-root file without traversing links or hidden namespaces."""
    root = _no_links(root)
    c.require(root.is_dir(), 'Exact packet root must be a directory')
    files, portable = {}, {}
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            canonical(relative)
            folded = relative.casefold()
            c.require(folded not in portable, 'Duplicate/colliding portable namespace')
            portable[folded] = relative
            _no_links(path)
            info = path.lstat()
            if name in dirs:
                c.require(stat.S_ISDIR(info.st_mode), 'Non-directory namespace component')
            else:
                c.require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                          'Only unlinked regular new-root files are permitted')
                files[relative] = path
    return files


def _binding_map(mapping, label):
    c.require(isinstance(mapping, dict), 'Missing ' + label + ' map')
    folded = set()
    for name, row in mapping.items():
        canonical(name)
        c.require(name.casefold() not in folded, 'Colliding ' + label + ' paths')
        folded.add(name.casefold())
        c.require(isinstance(row, dict) and set(row) == {'sha256', 'bytes'} and
                  isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']) and
                  type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid ' + label + ' binding')


def verify_data_bundle(root, protocol, parent, *, resolver=None, bundle=None):
    """Strict metadata joins only; numerical arrays are checked by new data.py."""
    maps={**protocol['dependencies'],**protocol['retained_output_dependencies'],**protocol['extra_returned_dependencies']}
    checked={}
    def bound(name):
        c.require(name in maps,'Unbound data dependency: '+name)
        path=_no_links(resolver(name) if resolver is not None else _regular(parent,name))
        pin=maps[name]
        if name not in checked:
            c.require(path.stat().st_size==pin['bytes'] and c.sha256(path)==pin['sha256'],'Data metadata hash differs: '+name)
            checked[name]=pin['sha256']
        return path
    def doc(name):return c.read_json(bound(name))
    if bundle is None:bundle=c.read_json(_regular(root,protocol['data_bundle_file']))
    c.require(type(bundle)is dict and set(bundle)=={'format','training','hq','development'} and
        bundle['format']=='role-preserving-data-bundle-v1','Exact neutral data bundle format required')
    parity=doc('training_parity_v1.json')['cases'];refs={r['id']:r for r in doc('references.json')['references']}
    hq_parity={r['reference_id']:r for r in doc('hq_cache_parity_v1.json')['cases']}
    saved={r['id']:r for r in doc('validation_saved_v1.json')['items']}
    comparisons={'comparison_'+r['id']:r for r in doc('comparison_cases.json')['cases']}
    c.require([(r['update'],r['slot'])for r in parity]==[(u,s)for u in range(1,51)for s in range(1,5)] and
        len(bundle['training'])==200 and len(bundle['hq'])==100 and len(bundle['development'])==406,
        'Exact200TRAIN/100HQ/406DEV census required')
    required={'protocol.json','references.json','comparison_cases.json','training_parity_v1.json','hq_cache_parity_v1.json',
        'validation_saved_v1.json','perceptual/manifest.json','perceptual/LICENSE_LPIPS.txt','perceptual/LICENSE_TORCHVISION.txt',
        'perceptual/NOTICE.md','third_party/codeformer/LICENSE','third_party/codeformer/NOTICE.md',*c.HISTORICAL_RUNTIME_FILES}
    required.update(pin['path']for pin in c.ASSETS.values())
    for row,old in zip(bundle['training'],parity):
        role='normal'if old['slot']in (1,3)else'dark'
        expected={**old,'id':'train_u%04d_slot%d'%(old['update'],old['slot']),'role':role,'sourcebindings':old}
        c.require(c.equal(row,expected),'Exact chronological TRAIN source/role/parity row differs')
        ref=refs[old['reference_id']];record=doc(old['pair_record_path'])
        c.require(ref['historical_role']=='train' and ref['source_sha256']==old['source_sha256'] and
            record['reference_id']==old['reference_id'] and record['source_sha256']==old['source_sha256'] and
            record['update']==old['update'] and record['slot']==old['slot'] and record['member']==role and
            record['input256_rgb8_sha256']==old['input256_rgb8_sha256'] and
            record['input256_float32_sha256']==old['input256_float32_sha256'],'TRAIN pair/reference/source join differs')
        c.require(protocol['dependencies'][old['input_path']]['sha256']==old['input_sha256'] and
            protocol['dependencies'][old['pair_record_path']]['sha256']==old['pair_record_sha256'] and
            protocol['dependencies'][old['clean_source_path']]['sha256']==old['source_sha256'],
            'TRAIN physical PNG/record/clean source differs')
        required.update((old['input_path'],old['pair_record_path'],old['clean_source_path']))
    selected=sorted({r['reference_id']for r in bundle['training']if r['role']=='dark'})
    c.require(selected==protocol['selected_train_reference_ids'] and len(selected)==100 and
        set(selected)=={r['reference_id']for r in bundle['training']} and
        sum(r['role']=='dark' and r['image_eligible']for r in bundle['training'])==74,
        'Selected100 paired TRAIN sources/dark LPIPS eligibility differs')
    c.require([r['reference_id']for r in bundle['hq']]==selected,'HQ order differs from selected100')
    for row in bundle['hq']:
        rid=row['reference_id'];old=hq_parity[rid];source=next(r for r in bundle['training']if r['reference_id']==rid)
        stem='outputs/adain_gradient_isolation_v1/HQ_cache/'+rid+'/'
        record=doc(stem+'record.json')
        expected={**old,'clean_source_path':old['source_path'],'clean512_rgb8_sha256':source['clean512_rgb8_sha256'],
            'indices_path':stem+'indices.npy','features_path':stem+'features.npy','record_path':stem+'record.json','sourcebindings':record}
        c.require(c.equal(row,expected) and record['exact_retained_FP32_target_parity']['pass']is True and
            all(record['exact_retained_FP32_target_parity'][k]==old[k]for k in old) and
            record['indices_path']==row['indices_path'] and record['features_path']==row['features_path'] and
            maps[row['indices_path']]['sha256']==record['indices_sha256'] and
            maps[row['features_path']]['sha256']==record['features_sha256'],'HQ exact retained file/array/teacher parity differs')
    metrics0=doc('outputs/adain_gradient_isolation_v1/snapshot_0000/metrics.json')
    metrics50=doc('outputs/adain_gradient_isolation_v1/snapshot_0050/metrics.json')
    c.require(metrics0['complete']is True and metrics50['complete']is True and
        [r['id']for r in bundle['development']]==[r['id']for r in metrics0['rows']] and
        {r['id']for r in metrics0['rows']}=={r['id']for r in metrics50['rows']}==set(saved)|set(comparisons),
        'Exact historical406 DEV metadata/order differs')
    for row,old in zip(bundle['development'],metrics0['rows']):
        rid=row['id']
        if rid in saved:
            origin=saved[rid];inp,mask,target=origin['input_path'],None,origin['target_path']
            pair=doc(origin['pair_record_path']);required.add(origin['pair_record_path'])
            sourcebindings={'validation_saved':origin,'pair_record':pair}
        else:
            origin=comparisons[rid];inp,mask,target=origin['prepared_path'],origin['observed_path'],origin.get('target_path')
            sourcebindings=origin
        expected={'id':rid,'source':old['source'],'profile':old['profile'],'evidence':old['evidence'],
            'input_path':inp,'observed_path':mask,'target_path':target,
            'original_raw_path':'outputs/adain_gradient_isolation_v1/snapshot_0000/'+rid+'/raw.npy',
            'original_png_path':'outputs/adain_gradient_isolation_v1/snapshot_0000/'+rid+'/native.png',
            'parent_raw_path':'outputs/adain_gradient_isolation_v1/snapshot_0050/'+rid+'/raw.npy',
            'parent_png_path':'outputs/adain_gradient_isolation_v1/snapshot_0050/'+rid+'/native.png','sourcebindings':sourcebindings}
        c.require(c.equal(row,expected),'DEV source/mask/target/benchmark row differs')
        c.require(all(row[k]in protocol['retained_output_dependencies']for k in
            ('original_raw_path','original_png_path','parent_raw_path','parent_png_path')),'DEV retained RAW/PNG pair missing')
        for name in (inp,mask,target):
            if name is not None:required.add(name)
        if old['evidence']=='native_unpaired':c.require(target is None,'Native clean target forbidden')
    c.require(set(protocol['dependencies'])==required,'Minimal physical dependency closure has missing/unused files')
    c.require(len({r['id']for r in bundle['development']})==406 and
        sum(r['evidence']=='native_unpaired'for r in bundle['development'])==48,
        'DEV406/native48 metadata census differs')
    return {'complete':True,'training_observations':200,'normal_observations':100,'dark_observations':100,
        'HQ_selected_references':100,'development_cases':406,'native_unpaired':48,'paired_synthetic':358,
        'minimal_physical_dependencies':len(required),'checked_metadata_sha256':checked,'CPU_arrays_loaded':False,
        'neural_calls':0,'FINAL_used':False,'source_ids_are_not_proven_person_ids':True}


def verify_transfer(root,protocol_or_sha,protocol_sha=None):
    """Fresh full source/minimal parent bytes before any framework import."""
    root=_no_links(root);supplied=protocol_or_sha if protocol_sha is not None else None
    protocol_sha=protocol_or_sha if protocol_sha is None else protocol_sha
    c.require(isinstance(protocol_sha,str) and HEX.fullmatch(protocol_sha),'Exact protocol SHA required')
    pp=_regular(root,'protocol.json');c.require(pp.stat().st_size<=c.LIMITS['source_packet_bytes'] and
        c.sha256(pp)==protocol_sha,'Frozen protocol size/hash differs')
    p=c.read_json(pp);c.verify_protocol(p)
    c.require(supplied is None or c.equal(supplied,p),'Caller protocol differs from exact frozen bytes')
    files,sizes=p['files_sha256'],p['files_bytes'];checked={}
    for name,h in sorted(files.items()):
        canonical(name);c.require(not name.startswith('outputs/') and name!=c.LOCK_NAME and
            Path(name).suffix in ('.py','.json','.md','.txt'),'Unexpected source-only namespace')
        path=_regular(root,name);before=path.stat()
        c.require(before.st_size==sizes[name] and c.sha256(path)==h,'Static source/data hash differs: '+name)
        after=path.stat();c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
            (after.st_size,after.st_mtime_ns,after.st_ino),'Static file changed while hashing')
        checked[name]={'sha256':h,'bytes':sizes[name]}
    c.require(len(files)+1<=c.LIMITS['source_packet_files'] and sum(sizes.values())+pp.stat().st_size<=c.LIMITS['source_packet_bytes'],
        'Protocol-inclusive source packet cap exceeded')
    inventory=_walk(root)
    c.require({n for n in inventory if not n.startswith('outputs/') and n!=c.LOCK_NAME}==set(files)|{'protocol.json'},
        'Unbound source/importable namespace')
    parent=c.resolve_parent(root);counts={}
    for key in ('dependencies','retained_output_dependencies','extra_returned_dependencies'):
        _binding_map(p[key],key)
        for name,pin in sorted(p[key].items()):
            path=_regular(parent,name);before=path.stat()
            c.require(before.st_size==pin['bytes'] and c.sha256(path)==pin['sha256'],'Read-only parent bytes differ: '+name)
            after=path.stat();c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                (after.st_size,after.st_mtime_ns,after.st_ino),'Parent changed while hashing')
        counts[key]=len(p[key])
    manifest=c.read_json(_regular(parent,c.RETURN_MANIFEST['path']))
    c.require(manifest['complete']is True and manifest['protocol_sha256']==c.PARENT_PROTOCOL_SHA and
        all(manifest['files_sha256'].get(n)==pin['sha256']for k in ('retained_output_dependencies','extra_returned_dependencies')
            for n,pin in p[k].items()if n!=c.RETURN_MANIFEST['path']),'Parent returned manifest closure differs')
    bundle=c.read_json(_regular(root,p['data_bundle_file']))
    receipt=verify_data_bundle(root,p,parent,bundle=bundle)
    return {'format':'codeformer-role-preserving-transfer-v1-r2','complete':True,'protocol_sha256':protocol_sha,'protocol':p,
        'static_files':checked,'parent_dir':p['parent_dir'],'parent_protocol_sha256':p['parent_protocol_sha256'],
        **{k:p[k]for k in counts},'dependencies_verified_by_namespace':counts,'dependencies_verified':sum(counts.values()),
        'data_bundle':bundle,'data_receipt':receipt,'readonly':True,'neural_calls':0,'gradient_queries':0,'optimizer_steps':0}



def _process_rows():
    result = subprocess.run(['ps', '-eo', 'pid=,args='], check=True, capture_output=True, text=True, timeout=20)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        c.require(len(fields) == 2 and fields[0].isdigit(), 'Cannot classify full process inventory')
        rows.append((int(fields[0]), fields[1]))
    return rows


def require_worker_terminal(root):
    """A terminal receipt never overrides a live worker or retained PID lock."""
    lock = Path(root) / c.LOCK_NAME
    if lock.exists() or lock.is_symlink():
        _regular(root,c.LOCK_NAME)
        value = c.read_json(lock)
        pid = value.get('pid')
        c.require(type(pid) is int and pid > 0, 'Invalid retained worker lock PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError:
            c.require(False, 'Cannot establish retained worker PID is terminal')
        else:
            c.require(False, 'Worker PID is alive; preserve it and do not export')
    for pid, command in _process_rows():
        if pid == os.getpid():
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            c.require(MODULE not in command, 'Cannot classify potential diagnostic process')
            continue
        c.require(not (MODULE in words and '--run' in words),
                  'Diagnostic worker/wrapper remains live; do not export')



class BoundedArchive:
    def __init__(self, stream, deadline, reserve_path):
        self.stream, self.deadline, self.reserve_path, self.bytes = stream, deadline, reserve_path, 0

    def check(self, size=0):
        c.require(time.monotonic() <= self.deadline, 'Finite export time exceeded')
        c.require(self.bytes + size <= ARCHIVE_CAP, 'Compressed archive cap exceeded')
        c.require(shutil.disk_usage(self.reserve_path).free >= RESERVE + size,
                  'Export disk reserve would be crossed; preserve partial evidence')

    def write(self, payload):
        self.check(len(payload))
        written = self.stream.write(payload)
        c.require(written == len(payload), 'Short archive write')
        self.bytes += written
        return written

    def flush(self):
        self.stream.flush()


class HashReader:
    def __init__(self, stream, bounded):
        self.stream, self.bounded, self.digest, self.bytes = stream, bounded, hashlib.sha256(), 0

    def read(self, size=-1):
        self.bounded.check()
        value = self.stream.read(size)
        self.digest.update(value); self.bytes += len(value)
        return value


@contextmanager
def _finite_export():
    """Bound hashing as well as streaming on the actual Linux export route."""
    if not hasattr(signal, 'setitimer'):
        yield  # CPU fixtures on Windows; actual route is guarded Linux-only.
        return
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise TimeoutError('Finite export600second wall limit exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(EXPORT_SECONDS, previous_timer[0]) if previous_timer[0] > 0 else EXPORT_SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        remaining = max(0.000001, previous_timer[0] - (time.monotonic() - started)) if previous_timer[0] > 0 else 0
        signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def resource_ledger(static_files=None,static_bytes=None):
    """Conservative source projection; checkpoint50 and stopped are exclusive."""
    n=c.LIMITS['source_packet_files']if static_files is None else static_files
    b=c.LIMITS['source_packet_bytes']if static_bytes is None else static_bytes
    c.require(type(n)is int and 0<n<=c.LIMITS['source_packet_files'] and
        type(b)is int and 0<=b<=c.LIMITS['source_packet_bytes'],'Invalid source ledger input')
    rgb256=3*256*256*4+256;rgb512=3*512*512*4+256;logits=256*1024*4+256
    categories={'checkpoints':{'files':2,'bytes':c.LIMITS['initial_checkpoint_file_bytes']+c.LIMITS['trained_checkpoint_file_bytes']},
        'snapshot_RAW':{'files':812,'bytes':812*rgb256},'snapshot_PNG':{'files':812,'bytes':812*c.LIMITS['PNG_file_bytes']},
        'TRAIN_arrays':{'files':1000,'bytes':200*(rgb512+2*rgb256+logits+(256*8+256))},
        'TRAIN_PNG':{'files':200,'bytes':200*c.LIMITS['PNG_file_bytes']},
        'HQ_arrays':{'files':200,'bytes':100*((256*8+256)+(256*16*16*4+256))},
        'witness_arrays':{'files':60,'bytes':60*rgb512},
        'all_metadata_logs':{'files':3200,'bytes':METADATA_CAP},'source_static':{'files':n,'bytes':b}}
    files=sum(v['files']for v in categories.values());payload=sum(v['bytes']for v in categories.values())
    tar=payload+files*1023+1024+10240
    gz=tar+(tar>>12)+(tar>>14)+(tar>>25)+64
    c.require(files<=MEMBER_CAP and payload<=PAYLOAD_CAP and gz<=ARCHIVE_CAP,'Prospective source-derived resource proof exceeds frozencaps')
    simultaneous=PAYLOAD_CAP+ARCHIVE_CAP+c.LIMITS['checkpoint_temporary_bytes']+RESERVE
    c.require(simultaneous<c.LIMITS['initial_free_bytes'],'Initialdisk admission lacks simultaneouspayload/archive/temp/reserve')
    return {'complete':True,'prospective_only':True,'measured_peak_resources':False,'categories':categories,
        'projected_files':files,'projected_payload_bytes':payload,'USTAR_bound_bytes':tar,'single_gzip_bound_bytes':gz,
        'simultaneous_disk_caps_bytes':simultaneous,'admission_margin_bytes':c.LIMITS['initial_free_bytes']-simultaneous,
        'neural_forwards':c.NEURAL_FORWARD_CAP,'reservation_events':c.RESERVATION_CAP,
        'premises':['At most initialcheckpoint plus final50 OR actualstopped; final50/stopped mutuallyexclusive.',
            'Every200TRAIN saves native512,raw256,teacher256,teacherlogits andselectedtokens;812snapshotRAWPNG;100HQindices/features.',
            'Witnessbinary allowance60native512NPY covers separate eval/training captures and RNG;metadata128MiBincludes allrecords/traces/logs;staticmaximum64files16MiB.',
            'Onegzip/USTARarchive;one1GiBcheckpointtempreservation;actualruntimeadmission remainsauthoritative.']}


def checkpoint_kind(name):
    if name in c.CHECKPOINT_PATHS:return 'final'
    for base in c.CHECKPOINT_PATHS:
        if re.fullmatch(re.escape(base)+r'\.tmp-[0-9a-f]{32}',name):return 'temporary'
    return None


def checkpoint_accounting(sizes):
    slots={base:[]for base in c.CHECKPOINT_PATHS};temporary=0
    for name,size in sizes.items():
        kind=checkpoint_kind(name)
        if kind:
            base=name if kind=='final'else next(v for v in slots if name.startswith(v+'.tmp-'))
            slots[base].append((name,size));temporary+=size if kind=='temporary'else 0
    for base,values in slots.items():
        cap=c.LIMITS['initial_checkpoint_file_bytes']if base==c.CHECKPOINT_PATHS[0]else c.LIMITS['trained_checkpoint_file_bytes']
        c.require(len(values)<=1 and sum(size for _,size in values)<=cap,'One final OR retainedpartial per checkpointslot/cap required')
    c.require(not (slots[c.CHECKPOINT_PATHS[1]] and slots[c.CHECKPOINT_PATHS[2]]),'Final50 and stopped checkpoints are mutuallyexclusive')
    c.require(temporary<=c.LIMITS['checkpoint_temporary_bytes'],'Aggregate checkpointtemporary cap exceeded')
    return {'slots':{k:[n for n,_ in v]for k,v in slots.items()},'temporary_bytes':temporary}


def _measure_files(root,files):
    hashes,sizes,metadata={},{},0
    for name in sorted(files):
        path=_regular(root,name);size=path.stat().st_size;kind=checkpoint_kind(name);suffix=path.suffix.lower()
        if re.search(r'\.(?:pt|pth|ckpt|safetensors|pkl|pickle|bin)(?:$|[.\-])',path.name.lower()):
            c.require(kind is not None,'Checkpoint outside exact filename gates forbidden')
        if kind is None:
            if suffix in ('.npy','.npz'):c.require(size<=c.LIMITS['array_file_bytes'],'Array file cap exceeded')
            elif suffix=='.png':c.require(size<=c.LIMITS['PNG_file_bytes'],'PNG256 file cap exceeded')
            else:
                metadata+=size
                if suffix=='.json':c.require(size<=c.LIMITS['one_JSON_bytes'],'IndividualJSON2MiB cap exceeded')
        hashes[name],sizes[name]=c.sha256(path),size
    checkpoint_accounting(sizes)
    c.require(len(sizes)<=MEMBER_CAP and sum(sizes.values())<=PAYLOAD_CAP and metadata<=METADATA_CAP,
        'Whole source/result/file/metadata admission cap exceeded; no omission')
    return hashes,sizes,metadata


def _output_coverage(files,bundle):
    required=set()
    for at in ('0000','0050'):
        required.update(OUTPUTS+'/snapshot_'+at+'/'+r['id']+'/'+n for r in bundle['development']for n in ('raw.npy','native.png','evidence.json'))
        required.add(OUTPUTS+'/snapshot_'+at+'/metrics.json')
    train_names=('input.png','teacher_delivered.npy','teacher_logits.npy','candidate_native.npy','candidate_raw.npy','tokens.npy','record.json')
    required.update(OUTPUTS+'/TRAIN/'+r['id']+'/'+n for r in bundle['training']for n in train_names)
    required.update(OUTPUTS+'/HQ_cache/'+r['reference_id']+'/'+n for r in bundle['hq']for n in ('indices.npy','features.npy','record.json'))
    missing=sorted(required-set(files))
    return {'complete':not missing,'expected_members':len(required),'missing_members':missing,
        'snapshot_RAW_PNG_pairs':sum(OUTPUTS+'/snapshot_'+at+'/'+r['id']+'/'+n in files for at in ('0000','0050')
            for r in bundle['development']for n in ('raw.npy',)),'scope':'Filecoverage only; pixel/state/quality review separate'}


def allowed_output_names(bundle):
    """Exact finite worker artifacts; extra files cannot silently enter export."""
    allowed={OUTPUTS+'/'+n for n in ('transfer_verified.json','idle_verified.json','input_index.json','runtime.json',
        'factory.json','parameter_census.json','codebook_compatibility.json','initial_frozen_state.json','final_frozen_state.json',
        'perceptual_prerequisite.json','gradient_coverage.json','preservation_guards.json','dark_retention.json',
        'retained_original_metrics.json','retained_base_metrics.json','run_results.json','failure.json',
        'stopped_checkpoint_alias.json','dependency_verification.json','export_manifest.json')}
    allowed.update(c.CHECKPOINT_PATHS)
    for at in ('0000','0050'):
        allowed.update(OUTPUTS+'/snapshot_'+at+'/'+r['id']+'/'+n for r in bundle['development']for n in ('raw.npy','native.png','evidence.json'))
        allowed.add(OUTPUTS+'/snapshot_'+at+'/metrics.json')
    allowed.add(OUTPUTS+'/snapshot_0000/preservation_guards.json')
    allowed.update(OUTPUTS+'/TRAIN/'+r['id']+'/'+n for r in bundle['training']for n in
        ('input.png','teacher_delivered.npy','teacher_logits.npy','candidate_native.npy','candidate_raw.npy','tokens.npy','record.json'))
    allowed.update(OUTPUTS+'/HQ_cache/'+r['reference_id']+'/'+n for r in bundle['hq']for n in ('indices.npy','features.npy','record.json'))
    allowed.update(OUTPUTS+'/history/update_%04d_%s.json'%(u,kind)for u in range(1,51)for kind in ('attempt','committed'))
    allowed.update(OUTPUTS+'/call_trace/%05d.json'%n for n in range(1,c.RESERVATION_CAP+1))
    ids=['train_u0003_slot1','train_u0003_slot2'];capture=('native.npy','raw.npy','logits.npy','latent.npy','tokens.npy')
    for row in ids:
        for stem in ('initializer_parity/'+row,'initializer_parity/'+row+'/parent','initializer_training/'+row,'gradient_route_witness/'+row):
            allowed.update(OUTPUTS+'/'+stem+'/'+n for n in capture)
        allowed.update(OUTPUTS+'/initializer_parity/'+row+'/'+n for n in ('record.json','context.json','teacher_delivered.npy','teacher_logits.npy'))
        allowed.update(OUTPUTS+'/initializer_training/'+row+'/'+n for n in ('record.json','context.json'))
        allowed.add(OUTPUTS+'/gradient_route_witness/'+row+'/context.json')
    allowed.add(OUTPUTS+'/gradient_route_witness/report.json')
    for label in ('initializer_before','initializer_slot1_after','initializer_slot2_after','initializer_after','before','after'):
        allowed.update(OUTPUTS+'/gradient_route_witness/rng_'+label+n for n in ('.json','_CPU.npy','_CUDA0.npy'))
    return allowed


def _alias_evidence(root,files,protocol_sha,completed,partial,inflight):
    name=c.STATE_POLICY['stopped_alias_file'];present=name in files;temps=[n for n in files if checkpoint_kind(n)=='temporary']
    if not present:return None
    unknown=completed is None and partial is None and inflight is None
    c.require((unknown or completed==50 and partial==0 and inflight is False) and c.CHECKPOINT_PATHS[1]in files and
        c.CHECKPOINT_PATHS[2]not in files and not temps,'Exact50alias requires safe final50 and no stopped/temp')
    value=c.read_json(_regular(root,name));path=_regular(root,c.CHECKPOINT_PATHS[1])
    c.require(value.get('complete')is True and value.get('protocol_sha256')==protocol_sha and
        value.get('completed_training_updates')==50 and value.get('partial_microbatches')==0 and value.get('optimizer_inflight')is False and
        value.get('source_checkpoint')==c.CHECKPOINT_PATHS[1] and value.get('source_sha256')==c.sha256(path) and
        value.get('source_bytes')==path.stat().st_size and value.get('committed_state_digest_equal')is True and
        isinstance(value.get('committed_state_digest'),str) and HEX.fullmatch(value['committed_state_digest']),
        'Exact committed model/Adam/RNG/sampler/rates/counters alias proof differs')
    return value


def _terminal_evidence(root,files,protocol_sha,bundle):
    run,fail=OUTPUTS+'/run_results.json',OUTPUTS+'/failure.json';sup='outputs/supervision/run_receipt.json'
    c.require(sup in files,'Closed supervisor receipt required')
    supervisor=c.read_json(_regular(root,sup));c.require(supervisor.get('worker_terminal')is True and
        supervisor.get('protocol_sha256')==protocol_sha,'Supervisor terminal protocol differs')
    c.require(not (run in files and fail in files),'Conflicting worker terminal receipts')
    terminal=run if run in files else fail if fail in files else None
    v={}if terminal is None else c.read_json(_regular(root,terminal));counts=v.get('counts')
    completed=v.get('completed_training_updates');partial=v.get('partial_microbatches');inflight=v.get('optimizer_inflight')
    if terminal:
        c.require(v.get('complete')is True and v.get('protocol_sha256')==protocol_sha and type(counts)is dict and set(counts)==set(c.COUNTS),
            'Exact worker terminal schema required')
        c.require(all(type(counts[k])is int and 0<=counts[k]<=cap for k,cap in c.COUNTS.items()) and
            type(completed)is int and 0<=completed<=50 and type(partial)is int and 0<=partial<=4 and type(inflight)is bool,
            'Actual finite activity/state counters differ')
        c.require(completed<=counts['optimizer_step_attempts'] and completed*4+partial<=counts['backward_calls'],
            'Committed/partial progress contradicts actual reservations')
    coverage=_output_coverage(files,bundle);alias=_alias_evidence(root,files,protocol_sha,completed,partial,inflight)
    if terminal==run:
        c.require(completed==50 and partial==0 and inflight is False and counts==c.COUNTS and coverage['complete'] and
            c.CHECKPOINT_PATHS[0]in files and c.CHECKPOINT_PATHS[1]in files and alias is not None,'Complete50/200/alloutputs/checkpoints/alias required')
    elif terminal==fail:c.require(v.get('failure')is True,'Explicit failure status required')
    return {'worker_terminal_evidence':terminal,'failure_present':fail in files,'complete_training_process':terminal==run,
        'completed_training_updates':completed,'partial_microbatches':partial,'optimizer_inflight':inflight,'worker_recorded_counts':counts,
        'gradient_queries':None if counts is None else counts['gradient_queries'],
        'backward_calls':None if counts is None else counts['backward_calls'],
        'optimizer_step_attempts':None if counts is None else counts['optimizer_step_attempts'],
        'activity_unknown_without_worker_receipt':terminal is None,'checkpoint_members':[n for n in files if checkpoint_kind(n)],
        'stopped_checkpoint_alias':alias,'output_coverage':coverage,'model_qualified':False}


def _json_bytes(value):return (json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def _write_reserved_json(path,value):
    payload=_json_bytes(value);c.require(len(payload)<=c.LIMITS['one_JSON_bytes'],'ExportJSON2MiB cap exceeded')
    c.require(shutil.disk_usage(Path(path).parent).free>=RESERVE+len(payload)+65536,'JSON write reserve crossed')
    path=_no_links(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())


def export_results(root,protocol_sha):
    with _finite_export():return _export_results(root,protocol_sha)


def _export_results(root,protocol_sha):
    root=c.manual_root(root);started=time.monotonic();verified=verify_transfer(root,protocol_sha)
    p=verified['protocol'];bundle=verified['data_bundle'];require_worker_terminal(root);inventory=_walk(root)
    status=_terminal_evidence(root,inventory,protocol_sha,bundle)
    allowed=allowed_output_names(bundle)
    for name in inventory:
        if name.startswith('outputs/'):
            c.require(name in allowed or checkpoint_kind(name)=='temporary' or name.startswith('outputs/supervision/')and name[len('outputs/supervision/'):]in SUPER_FILE_NAMES,
                'Unexpected output namespace')
    archive=_no_links(Path.home()/(PREFIX+'-results.tar.gz'));checksum=Path(str(archive)+'.sha256');ep=archive.with_name(PREFIX+'-export.json')
    for path in (archive,checksum,ep):c.require(not path.exists()and not path.is_symlink(),'Preserve prior/partial export')
    depname,mname=OUTPUTS+'/dependency_verification.json',OUTPUTS+'/export_manifest.json'
    c.require(depname not in inventory and mname not in inventory,'Preserve prior dependency/export receipts; no retry')
    dep={k:v for k,v in verified.items()if k not in ('protocol','data_bundle')}
    _write_reserved_json(root/depname,dep);inventory=_walk(root)
    excluded={c.LOCK_NAME,'outputs/supervision/export.log','outputs/supervision/finished.json'}
    hashes,sizes,meta=_measure_files(root,set(inventory)-excluded)
    manifest={'format':'codeformer-role-preserving-training-return-v1-r2','protocol_sha256':protocol_sha,'files_sha256':hashes,'files_bytes':sizes,
        'parent_dir':p['parent_dir'],'parent_protocol_sha256':p['parent_protocol_sha256'],
        **{k:p[k]for k in ('dependencies','retained_output_dependencies','extra_returned_dependencies')},**status,
        'training_or_export_completion_not_model_acceptance':True,'original_assets_modified_or_copied':False,
        'separately_retained':['outputs/supervision/export.log','outputs/supervision/finished.json',ep.name,checksum.name,PREFIX+'-supervisor-final.json'],
        'UTC':datetime.now(timezone.utc).isoformat()}
    payload=_json_bytes(manifest)
    c.require(len(payload)<=c.LIMITS['one_JSON_bytes'] and len(sizes)+1<=MEMBER_CAP and
        sum(sizes.values())+len(payload)<=PAYLOAD_CAP and meta+len(payload)<=METADATA_CAP,'Complete manifest/source/result JSON/payload cap exceeded')
    tar_bytes=sum(512+((v+511)//512)*512 for v in [*sizes.values(),len(payload)])
    tar_upper=((tar_bytes+1024+10239)//10240)*10240
    upper=tar_upper+(tar_upper>>12)+(tar_upper>>14)+(tar_upper>>25)+64
    c.require(upper<=ARCHIVE_CAP and shutil.disk_usage(archive.parent).free>=upper+RESERVE and
        shutil.disk_usage(root).free>=len(payload)+RESERVE,'Archiveenvelope/cap/diskreserve exceeded')
    for name,size in {**sizes,mname:len(payload)}.items():
        info=tarfile.TarInfo(name);info.size,info.mode,info.mtime=size,0o644,0
        info.tobuf(format=tarfile.USTAR_FORMAT,encoding='utf-8',errors='strict')
    _write_reserved_json(root/mname,manifest);hashes[mname]=c.sha256(root/mname);sizes[mname]=len(payload)
    with archive.open('xb')as target:
        bounded=BoundedArchive(target,started+EXPORT_SECONDS,archive.parent)
        with gzip.GzipFile(filename='',mode='wb',fileobj=bounded,mtime=0,compresslevel=6)as gz:
            with tarfile.open(fileobj=gz,mode='w|',format=tarfile.USTAR_FORMAT)as tar:
                for name in sorted(hashes):
                    path=_regular(root,name);before=path.stat();c.require(before.st_size==sizes[name],'Source size changed')
                    info=tarfile.TarInfo(name);info.size,info.mode,info.mtime=sizes[name],0o644,0;info.uid=info.gid=0;info.uname=info.gname=''
                    with path.open('rb')as stream:
                        reader=HashReader(stream,bounded);tar.addfile(info,reader)
                        c.require(reader.bytes==sizes[name]and reader.digest.hexdigest()==hashes[name],'Source bytes changed while exporting')
                    after=path.stat();c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                        (after.st_size,after.st_mtime_ns,after.st_ino),'Sourceidentity changed while exporting')
        target.flush();os.fsync(target.fileno())
    digest=c.sha256(archive)
    c.require(time.monotonic()-started<=EXPORT_SECONDS and shutil.disk_usage(archive.parent).free>=RESERVE+65536,'Finiteexport/checksumreserve exceeded')
    with checksum.open('xb')as stream:stream.write((digest+'  '+archive.name+'\n').encode('ascii'));stream.flush();os.fsync(stream.fileno())
    receipt={'format':'codeformer-role-preserving-training-export-v1-r2','complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,
        'manifest_member':mname,'manifest_sha256':hashes[mname],'protocol_sha256':protocol_sha,**status,
        'regular_members':len(hashes),'payload_bytes':sum(sizes.values()),'seconds':time.monotonic()-started,
        'actual_count_source':'Worker terminal reservations/state; absent staysunknown.','training_or_export_completion_not_model_acceptance':True}
    _write_reserved_json(ep,receipt);return receipt


