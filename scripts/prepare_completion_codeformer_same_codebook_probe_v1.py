"""Stage compact SAME-codebook zero-optimizer diagnostic sources; archive requires independent/root release.
Standard library preparation only: no checkpoint deserialization, neural/gradient/optimizer/VM operations.
"""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.util,io,json,os,re,shutil,stat,tarfile,time
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'research/completion_codeformer_same_codebook_probe_v1'
DEVELOPMENT=ROOT/'research/completion_codeformer_same_codebook_probe_v1_development'
PACKETS=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet'
TRANSFER=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
VM_NAME='cctv_completion_codeformer_same_codebook_probe_v1_vm'
STEM='cctv-completion-codeformer-same-codebook-probe-v1'
MAX_EXECUTION_BYTES=128*1024**2
V4_STAGE=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm'
V2_RETURN=ROOT/'outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results'
SOURCE_NAMES=['completion.py','pretrained_completion.py','face_workflow.py','third_party/__init__.py','third_party/codeformer/__init__.py','third_party/codeformer/registry.py','third_party/codeformer/vqgan_arch.py','third_party/codeformer/codeformer_arch.py','scripts/prepare_completion_codeformer_same_codebook_probe_v1.py','scripts/export_completion_codeformer_same_codebook_probe_v1.py','scripts/run_completion_codeformer_same_codebook_probe_v1.sh']
SOURCE_MAP={name:ROOT/name for name in SOURCE_NAMES}
SOURCE_MAP['scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py']=DEVELOPMENT/'cctv_completion_codeformer_same_codebook_probe_v1_vm.py'
for name in ('contracts','optimized_budget','probe_forward','app_delivery'):SOURCE_MAP['research/completion_codeformer_same_codebook_probe_v1/'+name+'.py']=DEVELOPMENT/(name+'.py')
CURRENT_INSTRUCTIONS=['SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md','FACE_MASK_COMPLETION_SPEC_V1.md']
DESIGN_PATH=ROOT/'research/completion_codeformer_post_v4_same_codebook_design_v1/root_design.json'
DESIGN_SHA='2ca6fafbaf2a3008459dace2d83bfc0cecb6594ae6e33adc7ac4aaef5bd0effb'
V4_PROTOCOL_SHA='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'
PARENT_PROTOCOL=ROOT/'outputs/completion_codeformer_finetune_v1_r1_packet/protocol.json'
PARENT_PROTOCOL_SHA='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
EVIDENCE={
 'route_design_review':(DESIGN_PATH,DESIGN_SHA),
 'resource_admission':(ROOT/'research/completion_codeformer_same_codebook_probe_v1_audit/independent_resource_revision_admission.json','4c9b2a1da70e508e0482f06235f079ab513dc3910c99073c7c678a67886cef86'),
 'scientific_design_admission':(ROOT/'research/completion_codeformer_same_codebook_probe_v1_audit/independent_design_admission.json','148b5cce56a1ba71b71f5c4e5ad4cfb6e1b78f2df0ef40f6f6e587a89e01821f'),
 'oracle_label_index':(V2_RETURN/'outputs/teacher_labels/index.json','0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69'),
 'readable_design':(ROOT/'COMPLETION_CODEFORMER_POST_V4_REPRESENTATION_DESIGN.md','966fd796e7655eb458740970d7633a549f66033ffd6834d97c40a31b814ba849'),
 'resource_revision':(ROOT/'research/completion_codeformer_post_v4_same_codebook_design_v1/resource_revision_r1.json','b5209566c0529732a4e49b824bde517bc81317046928fa05dbdcbc452265ca10'),
 'source_V4_protocol':(V4_STAGE/'protocol.json',V4_PROTOCOL_SHA),
 'parent_state_audit':(ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_state_audit.json','e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06'),
 'parent_saved_output_audit':(ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_saved_output_audit.json','ab4bd40838e33bce240d28265ca727c4ac52e96e0b78ff8ad5abc67854b5b65e'),
 'V2_saved_output_audit':(ROOT/'outputs/completion_codeformer_coverage_v2_return/independent_saved_output_audit.json','6ac24d0ac4ccbe4d54201e0c907a132b753c2ee84be35ea8adddf6cfde195fbb'),
 'historical_admission':(ROOT/'outputs/completion_codeformer_fusion_only_v3_historical_assets_independent_admission.json','2a93cb5ceb71df7309b9291c7d814fb8d5f422744226663bd13a16c26aff62a0')}
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):
    def reject(x):raise ValueError('NonfiniteJSON '+x)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def regular(path):
    info=path.lstat();need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'Regular unlinked source required');return info
def safe_relative(name):
    need(isinstance(name,str) and name and '\\' not in name and ':' not in name,'Unsafe source member')
    path=PurePosixPath(name);need(not path.is_absolute() and '..' not in path.parts and str(path)==name,'Noncanonical/traversing member');return name

def dependencies():
    for path,digest in EVIDENCE.values():regular(path);need(sha(path)==digest,'Frozen evidence/source drift '+str(path))
    need(sha(PARENT_PROTOCOL)==PARENT_PROTOCOL_SHA,'Full R1 parent protocol differs')
    design=read(DESIGN_PATH);v4=read(V4_STAGE/'protocol.json')
    for name,row in design['evidence'].items():
        path=Path(row['path']);need(pin(path)==row,'Design evidence drift '+name)
    for name,source in SOURCE_MAP.items():safe_relative(name);regular(source)
    for name in CURRENT_INSTRUCTIONS:regular(ROOT/name)
    spec=importlib.util.spec_from_file_location('same_codebook_contract_only',DEVELOPMENT/'contracts.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(module.CONTRACT['forward_slots_planned']==200 and module.CONTRACT['forward_slots_max']==220 and module.CONTRACT['output_bytes']==1536*1024**2 and module.CONTRACT['optimizer_steps_max']==0 and module.CONTRACT['worker_seconds']==900,'Frozen zero-optimizer contract differs')
    return design,v4,module.CONTRACT,module.ARMS

def stage(stage_id):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,48}',stage_id) is not None,'Invalid stage name')
    base=PACKETS/stage_id;destination=base/VM_NAME;need(not base.exists(),'Preserve prior stage; choose a fresh stage id')
    design,v4,contract,arms=dependencies();v4files={r['path']:r for r in v4['files']};planned={};source_derivation=[]
    def add(relative,source,expected=None):
        safe_relative(relative);source=Path(source);regular(source)
        if expected is not None:need(source.stat().st_size==expected['bytes'] and sha(source)==expected['sha256'],'Exact source bytes/hash differ '+relative)
        if relative in planned:need(planned[relative].resolve()==source.resolve(),'Duplicate different packet source');return
        planned[relative]=source
    def frozen(relative):
        need(relative in v4files,'Used data not in frozen V4 protocol '+relative);add(relative,V4_STAGE/relative,v4files[relative])
    monitor=copy.deepcopy(v4['monitor_cases']);challenge=copy.deepcopy(v4['challenge_cases'])
    parity=[copy.deepcopy(r) for r in v4['fit_cases'] if r['role']=='fit_anchor' and r['reference_id'] in sorted({x['reference_id'] for x in v4['fit_cases']})[:2] and r['mask_kind'] in ('lower_face','eye_band')]
    need(len(parity)==4 and len(monitor)==32 and len(challenge)==16,'Fixed48plus4parity source cases')
    for row in parity:row['probe_role']='parity_only_no_optimization'
    need([r['id'] for r in monitor+challenge]==design['data']['case_ids_in_fixed_protocol_order'],'Frozen48case order differs')
    for row in monitor+challenge+parity:
        for field in ('input256','clean256','clean512','mask'):frozen(row[field])
    references=[copy.deepcopy(r) for r in v4['references'] if r['id'] in {x['reference_id'] for x in monitor}]
    need(len(references)==8 and all(r['role']=='monitor' for r in references),'Only8monitor reference metadata')
    for name,source in SOURCE_MAP.items():add(name,source)
    for name in CURRENT_INSTRUCTIONS:add('evidence/current_instructions/'+name,ROOT/name)
    evidence={};evidence_hashes={}
    for key,(path,digest) in EVIDENCE.items():
        relative='evidence/'+key+path.suffix;add(relative,path);evidence[key]=relative;evidence_hashes[key]=digest
    for key,row in design['evidence'].items():
        if key=='local_architecture':continue
        relative='evidence/root_design_evidence/'+key+Path(row['path']).suffix;add(relative,Path(row['path']),row);evidence[key]=relative;evidence_hashes[key]=row['sha256']
    add('evidence/original_R1_protocol.json',PARENT_PROTOCOL)
    # Preserve the initial capacity proposal and reason for its resource-only revision as historical evidence.
    olddesign=ROOT/'research/completion_codeformer_post_v4_same_codebook_design_v1/root_design_initial_1GiB_before_capacity_review.json'
    need(sha(olddesign)=='98ada9f9b56b318dd2aea82931388460ec40fa2bf84631f9ed7c39ffa4061b6a','Initial prospective capacity history changed');add('evidence/historical_initial_1GiB_design_NOT_CURRENT.json',olddesign)
    label_index=read(V2_RETURN/'outputs/teacher_labels/index.json');indexrows={r['id']:r for r in label_index['labels']};oracle=[]
    for declared in design['data']['reuse_only_original16_exact_clean_context_teacher_labels']:
        key=declared['teacher_key'];old=indexrows[key];src=Path(declared['original_label']['path']);need(pin(src)==declared['original_label'],'Original oracle label changed')
        need(old['sha256']==declared['original_label']['sha256'] and old['clean512_sha256']==declared['clean512_sha256'],'Original label index differs')
        relative='oracle_labels/ORACLE_unavailable_clean_information/'+src.name;add(relative,src,declared['original_label']);frozen(old['clean512'])
        oracle.append({'teacher_key':key,'path':relative,'bytes':src.stat().st_size,'sha256':sha(src),'clean512':old['clean512'],'clean512_sha256':declared['clean512_sha256'],'teacher_original_logical_sha256':declared['teacher_original_logical_sha256'],'oracle_unavailable_clean_information':True,'source_pin':declared['original_label'],'unused_prequant_latent_retained':True,'label_semantics':'SAME original inpainting encoder nearest_codes from exact clean context; not independent HQ prior, not available to real covered input.'})
    historical_rows=copy.deepcopy(v4['historical_control']['initial_rows'])
    for rows in historical_rows.values():
        for row in rows:
            for field in ('result256','PNG'):frozen(row[field])
    # Compact derivation retains its original complete provenance even though only48R1raw/PNG payloads are copied here.
    frozen(v4['historical_control']['derivation'])
    for name in ('source_tests.py','source_preparation_tests.json'):add('evidence/development/'+name,DEVELOPMENT/name)
    bridge=ROOT/'research/completion_codeformer_same_codebook_probe_v1_audit/independent_bridge_semantics.json';add('evidence/independent_bridge_semantics.json',bridge);evidence['bridge_semantics']='evidence/independent_bridge_semantics.json';evidence_hashes['bridge_semantics']=sha(bridge)
    candidate=ROOT/'outputs/completion_codeformer_finetune_v1_r1_candidate'
    for name in ('LICENSE','NOTICE.md'):add('legal/'+name,candidate/name)
    add('evidence/original_source_admission.json',ROOT/'outputs/completion_codeformer_manual_v1/source_admission.json')
    records=[{'path':name,'bytes':regular(path).st_size,'sha256':sha(path)} for name,path in sorted(planned.items())]
    need(records and sum(r['bytes'] for r in records)<MAX_EXECUTION_BYTES,'Compact execution payload cap128MiB')
    protocol={'format':'completion-codeformer-same-codebook-probe-v1','status':'staged_source_only_unrun_unreleased','frozen_UTC':datetime.now(timezone.utc).isoformat(),'contract':contract,'arms':arms,'checkpoint':copy.deepcopy(v4['checkpoint']),'parent':copy.deepcopy(v4['parent']),'references':references,'monitor_cases':monitor,'challenge_cases':challenge,'parity_cases':parity,'oracle_labels':oracle,'historical_control':{'initial_rows':historical_rows,'derivation':v4['historical_control']['derivation'],'original_source_archive_sha256':v4['historical_control']['source_archive_sha256'],'payload_subset':'Onlyall48R1initial compactfloat32result256arraysandexactPNGs copied; historicalV2/V3outputs and otherFITassets remain outside this diagnostic.'},'original_parent_protocol':'evidence/original_R1_protocol.json','evidence':evidence,'evidence_sha256':evidence_hashes,'terms':v4['terms'],'data_limits':v4['data_limits'],'source_subset':{'source_V4_protocol_sha256':V4_PROTOCOL_SHA,'used_data_only':True,'all551FITdataset_copied':False,'actual_evaluation_cases':48,'monitor_references':8,'additional_parity_cases':4,'additional_parity_references':2,'oracle_original_label_sets':16,'teacher_network_calls':0,'public_unpaired_or_FINAL_cases':0},'scope':'Zerooptimizer SAME-codebook representation diagnosis only. A/B use observed covered input. C/D explicitly use unavailable clean-information indices, never a deployable result or mathematical upper bound. No model trained, no app promotion.','instruction_snapshot':{'UTC':datetime.now(timezone.utc).isoformat(),'status':'Exact prospective/source-preparation snapshots; root release and current guide govern later transfer readiness.','files':{'evidence/current_instructions/'+n:pin(ROOT/n) for n in CURRENT_INSTRUCTIONS}},'export':{'stem':STEM,'cap_bytes':1536*1024**2,'cap_seconds':900,'minimum_free_bytes':2*1024**3,'maximum_regular_members_including_manifest':1000,'maximum_path_UTF8_bytes':240,'maximum_NPZ':196,'checkpoint_included':False,'failed_partials_retained':True},'files':records,'manual_execution_only':True,'automatic_resume_or_promotion':False,'local_neural_calls':0,'optimizer_updates_locally':0,'diagnostic_only':True,'new_model_created':False,'model_qualified':False,'goal_complete':False,'release_required_before_archive':True}
    body=(json.dumps(protocol,indent=2,allow_nan=False)+'\n').encode();need(len(body)<=1024**2 and len(body)+sum(r['bytes'] for r in records)<MAX_EXECUTION_BYTES,'Protocol/payload compact caps')
    destination.mkdir(parents=True)
    for row in records:
        src=planned[row['path']];dst=destination/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b,8*1024*1024)
        need(dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256'] and sha(src)==row['sha256'],'Stagecopydrift')
    with (destination/'protocol.json').open('xb') as f:f.write(body)
    RESEARCH.mkdir(parents=True,exist_ok=True);research_protocol=RESEARCH/'protocol.json'
    if research_protocol.exists():
        with (base/'previous_research_protocol.json').open('xb') as f:f.write(research_protocol.read_bytes())
    research_protocol.write_bytes(body)
    actual={q.relative_to(destination).as_posix() for q in destination.rglob('*') if q.is_file()};need(actual=={r['path'] for r in records}|{'protocol.json'},'Stage namespace differs')
    receipt={'complete':True,'source_only_stage':True,'stage_id':stage_id,'root':str(destination),'protocol_sha256':sha(destination/'protocol.json'),'files':len(records),'payload_bytes':len(body)+sum(r['bytes'] for r in records),'archive_prepared':False,'manual_release_required':True,'model_loads':0,'neural_calls':0,'VM_calls':0,'optimizer_updates':0}
    write(base/'stage_receipt.json',receipt);print(json.dumps(receipt),flush=True);return receipt

INSTALL_CODE=r'''from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile, uuid

def need(ok, message):
    if not ok:
        raise ValueError(message)
base = Path.home() / 'forensic-dgp'
name = 'cctv_completion_codeformer_same_codebook_probe_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz'
need(base.is_dir() and (not destination.exists()), 'Preserve any existing experiment; do not reinstall')
need(archive.stat().st_size <= 128 * 1024 ** 2, 'Execution archive size cap')

def safe(value):
    p = PurePosixPath(value)
    need(value and (not p.is_absolute()) and ('..' not in p.parts) and (':' not in value) and ('\\' not in value) and (str(p) == value), 'Unsafe member path')
    return p
with tarfile.open(archive, mode='r|gz') as stream:
    iterator = iter(stream)
    first = next(iterator)
    need(first.isreg() and first.name == name + '/protocol.json' and (0 < first.size <= 2 * 1024 ** 2), 'Expected first protocol member')
    body = stream.extractfile(first).read(2 * 1024 ** 2 + 1)
    need(len(body) == first.size and hashlib.sha256(body).hexdigest() == '__PROTOCOL_SHA__', 'Protocol SHA differs')
    p = json.loads(body)
    need(p['format'] == 'completion-codeformer-same-codebook-probe-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 128 * 1024 ** 2 and shutil.disk_usage(base).free >= 7 * 1024 ** 3 + total, 'Need7GiB plus installation payload free')
    stage = base / (name + '.installing-' + uuid.uuid4().hex)
    stage.mkdir()
    with (stage / 'protocol.json').open('xb') as output:
        output.write(body)
    seen = {first.name}
    for member in iterator:
        need(member.isreg() and member.name in expected and (member.name not in seen), 'Unlisted/duplicate/nonregular member')
        parts = safe(member.name).parts
        need(parts[0] == name, 'Wrong member root')
        row = expected[member.name]
        need(member.size == row['bytes'], 'Member size differs')
        path = stage.joinpath(*parts[1:])
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with stream.extractfile(member) as incoming, path.open('xb') as output:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
                count += len(block)
                need(count <= row['bytes'], 'Oversized member')
                digest.update(block)
                output.write(block)
        need(count == row['bytes'] and digest.hexdigest() == row['sha256'], 'Member SHA differs')
        seen.add(member.name)
    need(seen == set(expected) | {name + '/protocol.json'}, 'Missing payload members')
    need(not destination.exists(), 'Experiment appeared during installation; preserve staging')
    stage.rename(destination)
print({'complete': True, 'installed': str(destination), 'protocol_sha256': '__PROTOCOL_SHA__', 'training_launched': False})
'''

def manual_guide(protocol_sha,archive_sha,archive_bytes,file_count):
    install=INSTALL_CODE.replace('__PROTOCOL_SHA__',protocol_sha)
    return '''# SAME-codebook completion probe V1: manual frozen-state diagnostic

RELEASED FOR MANUAL TRANSFER ONLY. No training or VM work was launched by preparation.
This creates no new model. It diagnoses the current R1 code representation while all 497
model tensors remain frozen. Original/R1 and every failed training state stay preserved.

Archive: __ARCHIVE_BYTES__ bytes, __FILE_COUNT__ payload files plus protocol.
SHA256: __ARCHIVE_SHA__.

## 1. Upload from Windows Google Cloud SDK Shell

Run on Windows, where the prompt starts C:, not inside VM SSH.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and safely install in VM SSH

The exact original inpainting checkpoint and complete trained R1 parent must already
exist on this VM. A new VM requires verified copies first. No duplicate weights are
uploaded by this packet. Requires 7 GiB plus installation payload free. The installer
checks every closed regular member and refuses existing experiment directories.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-same-codebook-probe-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test -f ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth &&
python3 - <<'PY'
__INSTALL_CODE__PY
```

Expected complete:true, protocol __PROTOCOL_SHA__, training_launched:false.
If it refuses, retain the error and all existing/staging folders. Do not bypass hashes.

## 3. Open tmux

Use the current session if already inside tmux. Otherwise:

```bash
tmux new-session -A -s completion_same_codebook_probe_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_same_codebook_probe_v1_vm.py --root . --protocol-sha __PROTOCOL_SHA__ --verify-transfer &&
bash scripts/run_completion_codeformer_same_codebook_probe_v1.sh __PROTOCOL_SHA__
```

The first command verifies source/assets without model imports or calls. The shell
starts this distinct frozen-state diagnostic. There are zero optimizer objects,
updates, gradients, backwards or teacher-network calls. No checkpoint is created.

The 48 fixed paired adaptive TRAIN cases include 32 clear and 16 synthetic challenges.
Four ordinary app parity pairs run first (8 calls). All 48 actual-A outputs must match
saved R1 raw256 within 1e-6 and PNG bytes exactly before the other arms proceed.
Each of 4 arms then produces 48 outputs, for 200 planned whole-network calls including
parity (220 hard cap). Arm A: actual predicted codes with R1 fusion w1. Arm B: actual
predicted codes without fusion w0. Arm C: ORACLE clean-context codes with fusion w1.
Arm D: ORACLE clean-context codes without fusion w0. Every C/D filename and receipt
labels unavailable clean information. They cannot be generated from a real covered
input alone and are not a new deployable model or exact hidden identity.

Only 16 already retained labels from the SAME original inpainting codebook are reused.
Official HQ512 labels are incompatible and are not substituted. Observed context,
reviewed mask and copied outside pixels stay the same; clean RGB is never pasted.
All 192 outputs, 4 parity arrays, predicted/used codes and before/after 497 tensor hashes
are retained for independent review. Public/unpaired and FINAL data are not used.

Worker hard cap 15 minutes; cooperative stop 14 minutes. Start disk 7 GiB; floor 2 GiB;
CUDA allocation 20 GiB. Each raw output payload, raw tar stream and gzip archive has
its own 1536 MiB cap. Export cooperative cap 15 minutes. Maximum 196 NPZ and 1000 return
members; failed partial arrays/evidence stay retained. No automatic resume.
Worker exit 0 means the diagnostic finished, regardless of the oracle comparison
scores. A complete export means packaging succeeded. Neither means quality passed.

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/trainer.log
```

Detach safely with Ctrl+B then D. At the end:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_same_codebook_probe_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-same-codebook-probe-v1-export.json
```

The successful archive.partial remains a hard link to the same inode, consuming no
duplicate payload storage. Keep all successful or failed evidence until audited.
No app adoption or further training follows automatically from this diagnosis.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command. Run on Windows, not in VM SSH.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-same-codebook-probe-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-same-codebook-probe-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-same-codebook-probe-v1-export.json" "."
```

Return all three files for transport, saved-array/code/state-invariance and all-case
visual review before choosing a distinct next design. Trained R1 stays unchanged.
'''.replace('__ARCHIVE_BYTES__',str(archive_bytes)).replace('__FILE_COUNT__',str(file_count)).replace('__ARCHIVE_SHA__',archive_sha).replace('__PROTOCOL_SHA__',protocol_sha).replace('__INSTALL_CODE__',install)

def archive_stage(stage_id,release_path):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,48}',stage_id) is not None,'Invalid stage name')
    stagebase=PACKETS/stage_id;stage=stagebase/VM_NAME;ppath=stage/'protocol.json';need(ppath.is_file(),'Stage source first');expected=sha(ppath);p=read(ppath)
    need((RESEARCH/'protocol.json').read_bytes()==ppath.read_bytes(),'Research protocol is a different stage')
    release=read(release_path);need(release.get('format')=='completion-codeformer-same-codebook-probe-v1-root-release-v1' and release.get('released_for_manual_vm_transfer') is True and release.get('automatic_vm_launch') is False,'Root manual-transfer release required')
    need(release.get('protocol_sha256')==expected,'Release protocol differs')
    admission_pin=release['independent_source_admission'];admission_path=Path(admission_pin['path']);need(pin(admission_path)==admission_pin,'Independent admission pin differs')
    admission=read(admission_path);need(admission.get('complete') is True and admission.get('independent') is True and admission.get('protocol_sha256')==expected,'Independent final source admission required')
    need(p['files'] and p['format']=='completion-codeformer-same-codebook-probe-v1' and p['manual_execution_only'] and not p['automatic_resume_or_promotion'],'Manual source closure differs')
    files=[]
    for row in p['files']:
        safe_relative(row['path']);path=stage/row['path'];regular(path);need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Staged source drift '+row['path']);files.append((row,path))
    need(len({r['path'] for r in p['files']})==len(p['files']),'Duplicate final protocol files')
    actual={f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()};need(actual=={r['path'] for r in p['files']}|{'protocol.json'},'Stage extras/missing namespace')
    destination=stagebase/(STEM+'-execution.tar.gz');partial=Path(str(destination)+'.partial');checksum=Path(str(destination)+'.sha256')
    for path in (destination,partial,checksum):need(not path.exists(),'Preserve existing archive/partial')
    total=ppath.stat().st_size+sum(row['bytes'] for row,path in files);need(total<MAX_EXECUTION_BYTES,'Execution payload cap')
    class Limited:
        def __init__(self,stream):self.stream=stream;self.count=0
        def write(self,data):
            need(self.count+len(data)<=MAX_EXECUTION_BYTES,'Execution compressed archive cap');n=self.stream.write(data);need(n==len(data),'Short execution archive write');self.count+=n;return n
        def flush(self):return self.stream.flush()
    with partial.open('xb') as raw:
        counted=Limited(raw)
        with gzip.GzipFile(filename='',mode='wb',fileobj=counted,compresslevel=1,mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|',bufsize=1024*1024,format=tarfile.PAX_FORMAT) as tar:
                ordered=[({'path':'protocol.json','bytes':ppath.stat().st_size,'sha256':expected},ppath)]+files
                for row,path in ordered:
                    need(sha(path)==row['sha256'],'Archive source drift')
                    info=tarfile.TarInfo(VM_NAME+'/'+row['path']);info.size=row['bytes'];info.mode=0o644;info.mtime=0
                    with path.open('rb') as stream:tar.addfile(info,stream)
        raw.flush();os.fsync(raw.fileno())
    digest=sha(partial);os.link(partial,destination)
    with checksum.open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+destination.name+'\n')
    # Publish only verified bytes to the user's established Windows transfer folder.
    published=[]
    for source in (destination,checksum):
        target=TRANSFER/source.name
        with target.open('xb') as stream:
            with source.open('rb') as incoming:shutil.copyfileobj(incoming,stream,8*1024*1024)
        need(sha(source)==sha(target),'Transfer publication differs');published.append(pin(target))
    guide=manual_guide(expected,digest,destination.stat().st_size,len(p['files']))
    for target in (ROOT/'COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_VM.md',TRANSFER/'COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_VM.md'):
        # Earlier source-only placeholder is retained explicitly before replacement.
        if target.exists():
            prior=stagebase/('prior_guide_'+('transfer' if target.parent==TRANSFER else 'worktree')+'.md')
            with prior.open('xb') as stream:stream.write(target.read_bytes())
        target.write_text(guide,encoding='utf-8',newline='\n')
    receipt={'complete':True,'archive':pin(destination),'archive_sha256':digest,'protocol_sha256':expected,'payload_files':len(files)+1,'payload_bytes':total,'root_release':pin(Path(release_path)),'independent_source_admission':admission_pin,'published':published,'manual_VM_guide':pin(TRANSFER/'COMPLETION_CODEFORMER_SAME_CODEBOOK_PROBE_V1_VM.md'),'successful_archive_partial_hardlink_retained':str(partial),'training_launched':False,'model_qualified':False,'VM_calls':0}
    write(stagebase/'archive_publication_receipt.json',receipt);print(json.dumps(receipt),flush=True);return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--archive',action='store_true');parser.add_argument('--stage-id',required=True);parser.add_argument('--release-file',type=Path)
    args=parser.parse_args()
    if args.stage:need(args.release_file is None,'Staging needs no release-file');stage(args.stage_id)
    else:need(args.release_file is not None,'Root release file required for archive');archive_stage(args.stage_id,args.release_file)
if __name__=='__main__':main()