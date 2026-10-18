"""CPU-only new overlay staging and manual verified sibling-dependency copies.

No optimizer, model, neural framework, VM connection or automatic launch exists.
Source/data staging is unsealed until separate independent reviews and sealing.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import platform
import re
import shutil
import stat
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PACKAGE='cctv_codeformer_adain_gradient_isolation_v1_vm'
PARENT='cctv_codeformer_fp32_comparison_v1_vm'
PARENT_PROTOCOL_SHA='8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
V2_PARENT='cctv_codeformer_adaptation_v2_vm'
V2_PROTOCOL_SHA='029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
PREFIX='cctv-codeformer-adain-gradient-isolation-v1'
FORMAT='codeformer-adain-gradient-isolation-finite-manual-v1'
DESIGN='CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md'
EVIDENCE='outputs/codeformer_adain_gradient_isolation_v1_preparation'
CORE_DOCS=('SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
HEX=re.compile(r'^[a-f0-9]{64}$')
DEVICE=re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)',re.I)
OLD_METADATA={'source_readiness.json','provenance/resource_ledger.json','provenance/independent_input_readback.json'}


def require(value,message):
    if not value:raise ValueError(message)


def canonical(name):
    require(isinstance(name,str) and name,'Empty relative path');p=PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix()==name and all(x not in ('','.','..') for x in p.parts),
            'Noncanonical/escaped artifact path')
    require(not any(ch in name for ch in '\\:"<>|?*') and all(not DEVICE.match(x) and not x.endswith((' ','.')) and
        len(x.encode('utf-16-le'))<=510 and all(ord(ch)>=32 for ch in x) for x in p.parts),'Nonportable artifact path')
    return name


def no_links(path):
    path=Path(path).absolute()
    for p in (path,*path.parents):
        if p.exists() or p.is_symlink():require(not p.is_symlink() and not getattr(p.lstat(),'st_file_attributes',0)&1024,
                                             'Symlink/reparse artifact forbidden')
    return path


def bounded(root,name,exists=True):
    root=no_links(root);path=no_links(root/canonical(name));require(path.is_relative_to(root),'Artifact escapes root')
    if exists:require(stat.S_ISREG(path.lstat().st_mode),'Regular artifact required: '+name)
    return path


def sha(path):
    path=no_links(path);before=path.stat();require(stat.S_ISREG(before.st_mode),'Regular hash input required')
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024**2),b''):h.update(chunk)
    after=path.stat();require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
                            'Artifact changed while hashing');return h.hexdigest()


def read_json(path):
    file=no_links(path);require(file.stat().st_size<=16*1024**2,'Finite JSON16MiB limit exceeded')
    def pairs(rows):
        out={}
        for key,value in rows:require(key not in out,'Duplicate JSON field');out[key]=value
        return out
    return json.loads(file.read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def put(path,payload):
    path=no_links(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as output:output.write(payload);output.flush();os.fsync(output.fileno())


def write_json(path,value):put(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8'))


def exact_files(root):
    names=set();folded=set()
    for folder,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:no_links(Path(folder)/name)
        for name in files:
            path=Path(folder)/name;relative=path.relative_to(root).as_posix();bounded(root,relative)
            require(relative.casefold() not in folded,'Case-colliding static namespace');folded.add(relative.casefold());names.add(relative)
    for name in folded:
        p=PurePosixPath(name).parent
        while p.parts:require(p.as_posix() not in folded,'File/directory collision');p=p.parent
    return names


def dependency_names(parent):
    """Exact immutable installed-FP32 files; never copy a prior executable run."""
    return sorted(name for name in parent['files_sha256'] if name not in OLD_METADATA and
        name not in {'instructions/'+doc for doc in CORE_DOCS} and
        not (name.startswith('research/') and PurePosixPath(name).name=='worker.py') and
        not name.startswith(('scripts/','tests/','outputs/','research/codeformer_fp32_comparison_v1/')))


def local_parent_resolver(root,parent):
    """Explicit local staging only: immutable FP32 overlay plus its pinned V2 deps."""
    root=Path(root);fp32=root/'outputs'/PARENT;v2=root/'outputs'/V2_PARENT
    require(sha(fp32/'protocol.json')==PARENT_PROTOCOL_SHA and sha(v2/'protocol.json')==V2_PROTOCOL_SHA,
            'Retained FP32/V2 parent protocol differs')
    require(parent['parent_protocol_sha256']==V2_PROTOCOL_SHA,'FP32 dependency lineage differs')
    pp=read_json(v2/'protocol.json');deps=parent['dependency_files_sha256']
    def resolve(name):
        if name in deps:
            require(parent['files_sha256'][name]==deps[name]==pp['files_sha256'].get(name),'Local V2 fallback must be explicitly FP32-pinned')
            return bounded(v2,name)
        return bounded(fp32,name)
    return resolve


def stage_inputs(root,pins_file):
    """Create only a fresh overlay/template; sealing/actual execution is separate."""
    root=no_links(root).resolve(strict=True);pins=read_json(pins_file)
    packet=root/'outputs'/PACKAGE;evidence=root/EVIDENCE
    require(not packet.exists() and not (evidence/'stage_inputs.json').exists() and
            not (evidence/'stage_failure.json').exists(),'Preserve prior/partial stage; no automatic repeat')
    require(sha(root/'outputs'/PARENT/'protocol.json')==PARENT_PROTOCOL_SHA,'Exact FP32 parent required')
    parent=read_json(root/'outputs'/PARENT/'protocol.json');resolve=local_parent_resolver(root,parent)
    source_map=pins.get('source_sha256',{});required={'research/codeformer_adain_gradient_isolation_v1/worker.py',
        'research/codeformer_adain_gradient_isolation_v1/gradient_route.py',
        'scripts/prepare_codeformer_adain_gradient_isolation_v1.py',
        'scripts/install_codeformer_adain_gradient_isolation_v1.py',
        'scripts/seal_codeformer_adain_gradient_isolation_v1.py'}
    require(isinstance(source_map,dict) and required<=set(source_map),'All new executing/preparation source pins required')
    for name,digest in source_map.items():
        require(isinstance(digest,str) and HEX.fullmatch(digest) and sha(bounded(root,name))==digest,'New source pin differs')
    docs=pins.get('instruction_sha256',{});require(set(docs)==set(CORE_DOCS),'Four exact current instruction snapshots required')
    for name,digest in docs.items():require(sha(bounded(root,name))==digest,'Current instruction snapshot changed')
    require(sha(bounded(root,DESIGN))==pins['design_sha256'],'Reviewed formal design changed')
    for key in ('design_readback','hq_cache_parity'):
        row=pins[key];require(sha(bounded(root,row['path']))==row['sha256'],'Reviewed design/cache metadata changed')
    readback=read_json(root/pins['design_readback']['path'])
    require(readback.get('complete') is True and readback.get('approved') is True and
            readback.get('reviewed_files_sha256',{}).get(DESIGN)==pins['design_sha256'] and
            readback.get('training_authorized') is False,'Approved formal design readback required')
    dependencies=dependency_names(parent);dep_hashes={};dep_sizes={}
    for name in dependencies:
        file=resolve(name);require(sha(file)==parent['files_sha256'][name],'Immutable FP32 dependency changed: '+name)
        dep_hashes[name]=parent['files_sha256'][name];dep_sizes[name]=file.stat().st_size
    for name,digest in source_map.items():
        require(not name.startswith('outputs/') and name!='protocol.json','Source pin may not use reserved run/protocol namespace')
        if name in dep_hashes:require(dep_hashes[name]==digest,'Reviewed shared source differs from immutable dependency')
    policy=[]
    for name,digest in docs.items():
        target='instructions/'+name;current=(root/name).read_bytes();old=parent['files_sha256'].get(target)
        row={'path':target,'current_root_sha256':digest,'current_bytes':len(current),'historical_parent_sha256':old,
             'policy':'As-of-stage snapshot only; new protocol/design/manual guide supersede stale historical status. Parent bytes remain unchanged.'}
        if old is not None:
            historical=resolve(target).read_bytes();require(sha(resolve(target))==old and current.endswith(historical),
                'Current instructions must retain exact historical parent bytes as suffix')
            row.update(historical_bytes=len(historical),retained_suffix_offset=len(current)-len(historical),historical_suffix_exact=True)
        policy.append(row)
    packet.mkdir();evidence.mkdir(parents=True,exist_ok=True);files={};sizes={};copies={}
    def copy(source,name,expected):
        require(name not in files and name not in dep_hashes,'Overlay/dependency collision')
        require(sha(source)==expected,'Overlay input changed before copy')
        destination=bounded(packet,name,exists=False);put(destination,source.read_bytes())
        require(sha(destination)==expected,'Overlay copy mismatch')
        files[name]=expected;sizes[name]=destination.stat().st_size
        copies[name]={'source':str(source.relative_to(root)),'sha256':expected,'bytes':sizes[name]}
    try:
        for name,digest in source_map.items():
            if name not in dep_hashes:copy(root/name,name,digest)
        for name,digest in docs.items():
            copy(root/name,'instructions/'+name,digest)
        copy(root/DESIGN,'instructions/'+DESIGN,pins['design_sha256'])
        readback_name='provenance/adain_gradient_isolation_v1/INDEPENDENT_DESIGN_READBACK_V1.json'
        copy(root/pins['design_readback']['path'],readback_name,pins['design_readback']['sha256'])
        copy(root/pins['hq_cache_parity']['path'],'hq_cache_parity_v1.json',pins['hq_cache_parity']['sha256'])
        for name,digest in pins.get('provenance_sha256',{}).items():
            target='provenance/adain_gradient_isolation_v1/'+Path(name).name
            copy(root/name,target,digest)
        instruction_name='provenance/adain_gradient_isolation_v1/instruction_snapshot_policy.json'
        write_json(packet/instruction_name,{'complete':True,'documents':policy,'parent_protocol_sha256':PARENT_PROTOCOL_SHA})
        files[instruction_name]=sha(packet/instruction_name);sizes[instruction_name]=(packet/instruction_name).stat().st_size
        # Only the new reviewed local worker's pure constants are imported;
        # no historical worker from parent trees or staged code is imported.
        if str(root) not in sys.path:sys.path.insert(0,str(root))
        from research.codeformer_adain_gradient_isolation_v1 import worker
        p=deepcopy(parent);p.update(deepcopy(worker.DEFAULT_PROTOCOL))
        require(p['format']==FORMAT and p['parent_dir']==PARENT and p['parent_protocol_sha256']==PARENT_PROTOCOL_SHA,
                'New worker namespace/lineage differs')
        require(p['design']['sha256']==pins['design_sha256'] and p['design']['readback_path']==readback_name,'New worker design pins differ')
        p['files_sha256']={**dep_hashes,**files};p['dependency_files_sha256']=dep_hashes
        p['dependency_files_bytes']=dep_sizes;p['parent_files_sha256']=deepcopy(dep_hashes)
        p['export_files']=sorted(set(name for name in parent['export_files'] if name in dep_hashes and
            Path(name).suffix.lower() not in {'.png','.npy','.pth','.pt'})|set(name for name in files if
            Path(name).suffix.lower() in {'.py','.sh','.md','.json','.txt','.csv','.yaml','.yml','.sha256'}))
        p['hq_cache_parity']={'path':'hq_cache_parity_v1.json','sha256':pins['hq_cache_parity']['sha256'],
                              'cases':391,'source_protocol_sha256':PARENT_PROTOCOL_SHA}
        p.pop('source_readiness',None);p.pop('resource_ledger',None)
        p['provenance'].update(parent_protocol_sha256=PARENT_PROTOCOL_SHA,
            initialized_from_untouched_CodeFormer_not_stopped_state=True,
            same391_HQ_recomputed_and_exact_prior_FP32_cache_parity_required=True,
            instruction_snapshot_policy=instruction_name,unsealed_stage_only=True)
        require(exact_files(packet)==set(files),'Unsealed overlay namespace differs')
        stage={'complete':True,'status':'STAGED / UNSEALED / UNRUN','parent_protocol_sha256':PARENT_PROTOCOL_SHA,
            'dependency_files_sha256':dep_hashes,'dependency_files_bytes':dep_sizes,
            'files_sha256':files,'files_bytes':sizes,'copy_bindings':copies,
            'overlay_files':len(files),'overlay_bytes':sum(sizes.values()),
            'dependencies_not_reuploaded':len(dep_hashes),'dependency_copy_bytes':sum(dep_sizes.values()),
            'instruction_snapshot_policy':policy,'pins_sha256':sha(pins_file),'preparer_sha256':sha(__file__),
            'local_neural_gradient_optimizer_VM_calls':0,'training_authorized':False,'model_qualified':False}
        write_json(evidence/'protocol_template_unsealed.json',p);write_json(evidence/'stage_inputs.json',stage)
        return stage
    except BaseException as error:
        write_json(evidence/'stage_failure.json',{'complete':False,'error_type':type(error).__name__,'message':str(error),
            'partial_stage_retained':True,'automatic_retry':False,'local_neural_gradient_optimizer_VM_calls':0})
        raise


def manual_dependency_roots(root,*,system=None,home=None):
    require((platform.system() if system is None else system)=='Linux','Manual dependency installation requires Linux')
    home=Path.home() if home is None else Path(home);root=no_links(root).resolve(strict=True)
    expected=no_links(home/'forensic-dgp'/PACKAGE)
    require(root==expected.resolve(strict=True),'Exact new VM package root required')
    parent=no_links(root.parent/PARENT).resolve(strict=True);require(parent.parent==root.parent,'Exact installed FP32 sibling required')
    return root,parent


def validate_install(root,parent,protocol_sha):
    require(isinstance(protocol_sha,str) and HEX.fullmatch(protocol_sha) and sha(root/'protocol.json')==protocol_sha,
            'Pinned sealed new protocol required')
    p=read_json(root/'protocol.json');require(p['format']==FORMAT and p['parent_dir']==PARENT and
        p['parent_protocol_sha256']==PARENT_PROTOCOL_SHA and sha(parent/'protocol.json')==PARENT_PROTOCOL_SHA,'Parent namespace/protocol differs')
    pp=read_json(parent/'protocol.json');deps=p['dependency_files_sha256'];sizes=p['dependency_files_bytes']
    require(set(deps)==set(sizes)==set(dependency_names(pp)),'Complete deterministic sibling-copy allowlist differs')
    require(set(deps)<=set(p['files_sha256']),'Dependencies absent from complete new protocol')
    expected_overlay=set(p['files_sha256'])-set(deps)|{'protocol.json'}
    require(exact_files(root)==expected_overlay,'Fresh overlay namespace differs or install was partially attempted')
    for name,digest in p['files_sha256'].items():
        canonical(name);require(isinstance(digest,str) and HEX.fullmatch(digest),'Invalid bound SHA')
        source=bounded(parent,name) if name in deps else bounded(root,name)
        require(sha(source)==digest,'Installed sibling/overlay source differs: '+name)
        if name in deps:
            require(digest==deps[name]==pp['files_sha256'][name] and type(sizes[name]) is int and source.stat().st_size==sizes[name],
                    'Dependency source/byte binding differs')
            require(not bounded(root,name,exists=False).exists(),'Never overwrite an installed/partial dependency')
    require(not any(name.startswith('outputs/') for name in deps),'Historical run outputs are never copied')
    return p


def stage_dependencies(root,protocol_sha,*,system=None,home=None):
    started=time.monotonic();root,parent=manual_dependency_roots(root,system=system,home=home)
    require(not (root/'outputs').exists(),'Preserve existing run evidence')
    receipt=root/'dependency_install_receipt.json';journal=root/'dependency_install_journal.jsonl'
    require(not receipt.exists() and not journal.exists(),'Preserve prior/partial install; no automatic retry')
    p=validate_install(root,parent,protocol_sha);size=sum(p['dependency_files_bytes'].values())
    require(shutil.disk_usage(root).free>=size+p['limits']['initial_free_bytes'],'Require dependency bytes plus declared post-install free space')
    with journal.open('x',encoding='utf-8',newline='\n') as log:
        def event(row):log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
        event({'event':'all_dependencies_verified_before_copy','protocol_sha256':protocol_sha,'files':len(p['dependency_files_sha256']),'bytes':size})
        try:
            for name,digest in sorted(p['dependency_files_sha256'].items()):
                require(time.monotonic()-started<=900,'Finite dependency install900s limit exceeded')
                require(shutil.disk_usage(root).free>=p['dependency_files_bytes'][name]+p['limits']['initial_free_bytes'],
                        'Prospective copy must preserve declared post-install free-space reserve')
                source=bounded(parent,name);target=bounded(root,name,exists=False);target.parent.mkdir(parents=True,exist_ok=True)
                with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2);dst.flush();os.fsync(dst.fileno())
                require(target.stat().st_size==p['dependency_files_bytes'][name] and sha(target)==sha(source)==digest and
                    target.stat().st_ino!=source.stat().st_ino,'Verified regular copy differs or is a hardlink')
                event({'event':'copied_verified_regular_file','path':name,'sha256':digest,'bytes':target.stat().st_size})
            require(shutil.disk_usage(root).free>=p['limits']['initial_free_bytes'],'Post-install free-space requirement failed')
            result={'complete':True,'protocol_sha256':protocol_sha,'parent_protocol_sha256':PARENT_PROTOCOL_SHA,
                'files':len(p['dependency_files_sha256']),'bytes':size,'seconds':time.monotonic()-started,
                'copy_policy':'Exact verified regular sibling copies; no hardlinks, outputs, old workers or automatic launch',
                'neural_calls':0,'gradient_queries':0,'optimizer_updates':0}
            write_json(receipt,result);event({'event':'dependency_install_complete'});return result
        except BaseException as error:
            event({'event':'dependency_install_failed','error_type':type(error).__name__,'message':str(error),'partial_files_retained':True})
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage-inputs',action='store_true',required=True)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--pins',type=Path,required=True)
    args=parser.parse_args();r=stage_inputs(args.root,args.pins)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files_sha256','files_bytes','copy_bindings','dependency_files_sha256','dependency_files_bytes')}))
