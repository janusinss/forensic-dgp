"""Strict standard-library-only fixed-state measurement contract."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

FORMAT='codeformer-fixed-state-response-protocol-v1'
MODULE='research.codeformer_fixed_state_response_v1.worker'
VM_NAME='cctv_codeformer_fixed_state_response_v1_vm'
OUTPUTS='outputs/fixed_state_response_v1'
PREFIX='cctv-codeformer-fixed-state-response-v1'
LOCK_NAME='.fixed_state_response_v1.lock'
HEX=re.compile(r'[0-9a-f]{64}')
ROOT_ALIAS=('tone_training','adain')
PARENT_ROOTS={'tone_training':'cctv_codeformer_tone_control_training_v1_vm','adain':'cctv_codeformer_adain_gradient_isolation_v1_vm'}
PARENT_PROTOCOL_SHA256={'tone_training':'f10ce6d52c2ec259632750af2d3afbdf97c5c925b99849fae2a2c4d9cd04a5b1',
    'adain':'4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'}
DESIGN_SHA256='cac7f360bbc09f9cff283529d85642b228fec0a59125797cefb5231a9790b2eb'


def require(value,message):
    if not value:raise ValueError(message)


def sha256(path,budget=None):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            if budget:budget()
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(rows):
        out={}
        for key,value in rows:
            require(key not in out,'Duplicate JSON key');out[key]=value
        return out
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical_json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def canonical_sha256(value):return hashlib.sha256(canonical_json(value)).hexdigest()
def equal(a,b):return canonical_json(a)==canonical_json(b)


def no_links(path):
    path=Path(path).absolute()
    for item in (path,*path.parents):
        if not item.exists():continue
        st=item.lstat()
        require(not stat.S_ISLNK(st.st_mode)and not getattr(st,'st_file_attributes',0)&1024,'Linked/reparse path forbidden')
        if stat.S_ISREG(st.st_mode):require(st.st_nlink==1,'Hardlinked files forbidden')
    return path


def safe_relative(name):
    require(isinstance(name,str)and name and '\\'not in name and ':'not in name,'Unsafe relative path')
    p=PurePosixPath(name)
    require(not p.is_absolute()and p.as_posix()==name and all(x not in ('','.','..')for x in p.parts),'Unsafe relative path')
    require(all(not x.endswith((' ','.'))for x in p.parts),'Ambiguous path component')
    return name


def safe_path(base,name,must_exist=True):
    safe_relative(name);base=no_links(base).resolve(strict=True)
    path=no_links(base.joinpath(*PurePosixPath(name).parts))
    require(path.resolve(strict=must_exist).is_relative_to(base),'Path escapes declared root')
    if must_exist:require(path.is_file(),'Regular file required')
    return path


def write_new_json(path,value):
    path=no_links(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n')as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def manual_root(root):
    require(platform.system()=='Linux','Manual NVIDIA L4 VM only; local neural calls forbidden')
    root=no_links(root).resolve(strict=True)
    require(root==no_links(Path.home()/'forensic-dgp'/VM_NAME).resolve(strict=False),'Unexpected manual VM root')
    require(bool(os.environ.get('TMUX')),'Manual tmux required before framework import')
    return root


def resolve_parents(root):
    root=no_links(root).resolve(strict=True);out={}
    for alias,name in PARENT_ROOTS.items():
        path=no_links(root.parent/name).resolve(strict=True)
        require(path.is_dir()and path.parent==root.parent and path!=root,'Exact read-only parent sibling required')
        out[alias]=path
    return out


_design_file=Path(__file__).with_name('design.json')
require(sha256(_design_file)==DESIGN_SHA256,'Prospective design changed')
DESIGN=read_json(_design_file)
LIMITS=DESIGN['resource_caps']
COUNTS=DESIGN['counts']
ASSETS={
    'initializer':{'root':'adain','path':'weights/codeformer.pth','sha256':'1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7','bytes':376637898},
    'initial_composite':{'root':'tone_training','path':'outputs/tone_control_training_v1/composite_head0000/checkpoint.pt','sha256':'dc3f161ff94fe7f5d851d7610fdab403211e0bba9a1d4b36ada7328a10130da7','bytes':376665005},
    'stopped_composite':{'root':'tone_training','path':'outputs/tone_control_training_v1/stopped_composite/checkpoint.pt','sha256':'f492d13cd63f9652407ca2eb4b90ae2b9a5fc49a528c4a62d552f47d1aec4a2f','bytes':376734485}}
MODEL_DIGESTS={'original':'6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
    'base':'92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
    'head0000':'9a3f78f85d1bc9b70878cabad61545ec808248566c7b13220052c3ed02918509',
    'head0050':'d6ec3096958fc53c3e067de3c44ff398252a2bd9bd25568dcf89997f68f3b20b'}
HISTORICAL_RUNTIME_FILES=('research/codeformer_adaptation_v2/__init__.py','research/codeformer_adaptation_v2/model.py',
    'research/codeformer_adaptation_v1/__init__.py','research/codeformer_adaptation_v1/common.py',
    'third_party/__init__.py','third_party/codeformer/__init__.py','third_party/codeformer/codeformer_arch.py',
    'third_party/codeformer/vqgan_arch.py','third_party/codeformer/registry.py')
CAMERA_FILES=('research/codeformer_adaptation_v1/data.py','research/codeformer_adaptation_v2/data.py')
REQUIRED_SOURCE_FILES={'research/__init__.py','scripts/supervise_codeformer_fixed_state_response_v1_vm.py',
    *('research/codeformer_fixed_state_response_v1/'+x for x in ('__init__.py','contract.py','data.py','budget.py','design.json','model.py','worker.py','transport.py'))}
REQUIRED_DATA_FILES={'fitted_cases.json','heldout_pairs.json','selection.json','exclusions.json'}


def verify_binding_map(mapping,label):
    require(isinstance(mapping,dict)and mapping,'Nonempty binding map required: '+label)
    require(len({n.casefold()for n in mapping})==len(mapping),'Case-colliding bindings')
    for name,pin in mapping.items():
        safe_relative(name)
        require(type(pin)is dict and set(pin)=={'sha256','bytes'}and HEX.fullmatch(pin['sha256'])and
                type(pin['bytes'])is int and pin['bytes']>=0,'Invalid exact binding: '+label)
    return mapping


def expected_row_ids(fitted_doc,new_pairs_doc):
    require(fitted_doc.get('format')=='codeformer-fixed-state-fitted-cases-v1'and
            new_pairs_doc.get('format')=='codeformer-fixed-state-heldout-pairs-v1','Unknown input index format')
    a=[r['id']for r in fitted_doc['cases']];b=[r['id']for r in new_pairs_doc['cases']]
    require(a==['train_u%04d_slot%d'%(u,s)for u in range(1,51)for s in range(1,5)]and
            len(b)==len(set(b))==64 and not set(a)&set(b),'Exact200fitted/64heldout rows required')
    return {'fitted':a,'heldout':b}


def expected_pair_paths(doc):
    require(doc.get('format')=='codeformer-fixed-state-heldout-pairs-v1'and len(doc['cases'])==64,'Exact64 heldout inputs required')
    out=set()
    for row in doc['cases']:
        rid,role=row['reference_id'],row['role']
        require(re.fullmatch(r'tr_ffhq_[0-9]{5}',rid)and role in ('normal','dark')and
                row['id']=='heldout_'+rid+'_'+role,'Invalid heldout identifier')
        stem='inputs/heldout_'+rid+'/'+role+'/'
        require(row['input_path']==stem+'input.png'and row['pair_record_path']==stem+'pair_record.json','Exact heldout path policy required')
        out.update((row['input_path'],row['pair_record_path']))
    require(len(out)==128,'Duplicate heldout pair artifacts')
    return out


def expected_input_pngs(doc):return {x for x in expected_pair_paths(doc)if x.endswith('.png')}


def verify_protocol(p):
    require(type(p)is dict and p.get('format')==FORMAT and p.get('module')==MODULE and p.get('vm_name')==VM_NAME and
            p.get('outputs')==OUTPUTS and p.get('export_prefix')==PREFIX,'Unknown finite measurement namespace')
    for key,value in {'parent_roots':PARENT_ROOTS,'parent_protocol_sha256':PARENT_PROTOCOL_SHA256,
            'assets':ASSETS,'model_digests':MODEL_DIGESTS,'design':DESIGN,'call_caps':COUNTS,'limits':LIMITS}.items():
        require(equal(p.get(key),value),'Prospective exact protocol field differs: '+key)
    require(p.get('design_sha256')==DESIGN_SHA256 and p.get('fitted_index_file')=='fitted_cases.json'and
            p.get('new_pairs_file')=='heldout_pairs.json'and p.get('selection_file')=='selection.json'and
            p.get('exclusions_file')=='exclusions.json','Bound input document names differ')
    require(p.get('status')=='PREPARED_UNRUN'and p.get('manual_tmux_required')is True and
            p.get('all_states_frozen')is True,'Manual frozen measurement status required')
    for key in ('automatic_VM_launch','automatic_training','automatic_resume','new_training_recipe','model_qualified',
            'automatic_app_promotion','quality_gates_changed','FINAL_used','optimizer_loaded','cached_gradient_reuse'):
        require(p.get(key)is False,'Prohibited protocol flag: '+key)
    deps=p.get('dependencies');require(type(deps)is dict and set(deps)==set(ROOT_ALIAS),'Exact two parent aliases required')
    for alias,mapping in deps.items():
        verify_binding_map(mapping,alias)
        require(mapping.get('protocol.json',{}).get('sha256')==PARENT_PROTOCOL_SHA256[alias],'Exact parent protocol binding required')
    for asset in ASSETS.values():require(deps[asset['root']].get(asset['path'])=={k:asset[k]for k in ('sha256','bytes')},'Asset not exact dependency')
    require(set(HISTORICAL_RUNTIME_FILES+CAMERA_FILES+('references.json','training_parity_v1.json')).issubset(deps['adain']),
            'Required immutable source/data closure missing')
    require({'outputs/tone_control_training_v1/runtime.json','research/codeformer_tone_control_training_v1/model.py',
            'research/codeformer_tone_control_training_v1/data.py','research/codeformer_tone_control_training_v1/design.json'}<=set(deps['tone_training']),
            'Required fitted-state runtime/feature/model-policy provenance missing')
    hashes,sizes=p.get('files_sha256'),p.get('files_bytes')
    require(type(hashes)is dict and type(sizes)is dict and set(hashes)==set(sizes)and
            REQUIRED_SOURCE_FILES|REQUIRED_DATA_FILES<=set(hashes)and len(hashes)+1<=LIMITS['source_packet_files']and
            sum(sizes.values())<=LIMITS['source_packet_bytes']and len({n.casefold()for n in hashes})==len(hashes),
            'Bound exact static packet census differs')
    for name,value in hashes.items():
        safe_relative(name);require(HEX.fullmatch(value)and type(sizes[name])is int and sizes[name]>=0,'Invalid static hash/size')
        require(Path(name).suffix in ('.py','.json','.md','.txt','.png'),'Forbidden static binary')
        if name.endswith('.png'):require(name.startswith('inputs/heldout_')and sizes[name]<=LIMITS['one256PNG_bytes'],'Invalid static PNG')
    require(hashes['research/__init__.py']==hashlib.sha256(b'').hexdigest()and sizes['research/__init__.py']==0 and
            hashes['research/codeformer_fixed_state_response_v1/design.json']==DESIGN_SHA256,'Synthetic namespace/design binding differs')
    return p
