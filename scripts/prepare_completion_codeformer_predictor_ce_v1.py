"""SOURCE-ONLY predictor CE packet preparation; requires formal independent and root release.
Never imports scientific libraries, loads models, uses VM/network, trains or resumes anything.
"""
from __future__ import annotations
import argparse,ast,copy,gzip,hashlib,importlib.util,io,json,os,re,shutil,stat,tarfile,time
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'research/completion_codeformer_predictor_ce_v1'
DEVELOPMENT=ROOT/'research/completion_codeformer_predictor_ce_v1_development'
PACKETS=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet'
TRANSFER=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
VM_NAME='cctv_completion_codeformer_predictor_ce_v1_vm'
STEM='cctv-completion-codeformer-predictor-ce-v1'
MAX_EXECUTION_BYTES=256*1024**2
V4_STAGE=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm'
V2_RESULTS=ROOT/'outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results'
DESIGN_PATH=ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_design.json'
DESIGN_SHA='077ed97ec5dc4ca75f6b6bdc3caf8def09344113c3b7ce927d13b7b68a4f2b78'
FORMAL_ADMISSION_PATH=ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/independent_formal_design_admission.json'
FORMAL_ADMISSION_SHA='691a5ec8151e0102e62f0c54c73770a25277690a7ac848586d33fb47f6ec9fbf'
SUBSET_PATH=ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/balanced128_subset.json'
SUBSET_SHA='8ec2d535cd6fb231af815d0ec833a1cd08f418458cbc4919c0ebd13f638f2a1d'
V4_PROTOCOL_SHA='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d'
PARENT_PROTOCOL=ROOT/'outputs/completion_codeformer_finetune_v1_r1_packet/protocol.json'
PARENT_PROTOCOL_SHA='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
CURRENT_INSTRUCTIONS=['SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md','FACE_MASK_COMPLETION_SPEC_V1.md']
SOURCE_MAP={name:ROOT/name for name in ['completion.py','pretrained_completion.py','face_workflow.py','third_party/__init__.py','third_party/codeformer/__init__.py','third_party/codeformer/registry.py','third_party/codeformer/vqgan_arch.py','third_party/codeformer/codeformer_arch.py','scripts/prepare_completion_codeformer_predictor_ce_v1.py','scripts/export_completion_codeformer_predictor_ce_v1.py','scripts/run_completion_codeformer_predictor_ce_v1.sh']}
SOURCE_MAP['scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py']=DEVELOPMENT/'cctv_completion_codeformer_predictor_ce_v1_vm.py'
for name in ('contracts','optimized_budget','app_delivery','optimizer_topology','terminal_admission'):SOURCE_MAP['research/completion_codeformer_predictor_ce_v1/'+name+'.py']=DEVELOPMENT/(name+'.py')
EVIDENCE={
 'root_design':(DESIGN_PATH,DESIGN_SHA),
 'independent_runtime_semantics':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_source_semantics.json','0eb9cf6ada1554875bf837d01b97a86c7a063887bbfa6608f12f0bffc1d9a0d7'),
 'root_terminal_fixtures':(ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_terminal_admission_fixtures.json','4ce59f5e91e9d68ec52c33d535c1c5acd7daa481fdd277146e7c48099aa41471'),
 'root_installer_fixtures':(ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_installer_fixtures.json','520220a1f7a9e1808e73596f942ad66b7cedaef871ebb5fc37af95a7de0cdbc7'),
 'root_pre_release_clarifications':(ROOT/'research/completion_codeformer_predictor_ce_design_v1/pre_release_clarifications_v1.json','0495ded17bb5d5d90eb59f50608f72c7cf1b82b6f83a7aaef5a76a9cdaae0664'),
 'cached_label_index':(V2_RESULTS/'outputs/teacher_labels/index.json','0567a9ad54e6397390341a8dfc1ffa8ec50ceb9159da299605fee021ec678a69'),
 'source_V4_protocol':(V4_STAGE/'protocol.json',V4_PROTOCOL_SHA),
 'parent_state_audit':(ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_state_audit.json','e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06'),
 'parent_saved_output_audit':(ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_saved_output_audit.json','ab4bd40838e33bce240d28265ca727c4ac52e96e0b78ff8ad5abc67854b5b65e'),
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
    need(len(FORMAL_ADMISSION_SHA)==64,'SOURCE-ONLY: exact independent formal design admission not bound; no packet staging')
    need(sha(FORMAL_ADMISSION_PATH)==FORMAL_ADMISSION_SHA,'Independent formal design admission drift')
    for path,digest in EVIDENCE.values():regular(path);need(sha(path)==digest,'Frozen evidence/source drift '+str(path))
    need(sha(SUBSET_PATH)==SUBSET_SHA and sha(PARENT_PROTOCOL)==PARENT_PROTOCOL_SHA,'Balanced subset or originalR1protocol differs')
    design=read(DESIGN_PATH);v4=read(V4_STAGE/'protocol.json');subset=read(SUBSET_PATH)
    for key,row in design['evidence'].items():need(pin(Path(row['path']))==row,'Root-design evidence drift '+key)
    for name,source in SOURCE_MAP.items():safe_relative(name);regular(source)
    worker_tree=ast.parse(SOURCE_MAP['scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py'].read_text(encoding='utf-8'))
    worker_design=[ast.literal_eval(node.value) for node in worker_tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='DESIGN_SHA' for target in node.targets)]
    need(worker_design==[DESIGN_SHA],'Worker formal design remains unbound or differs; no stage')
    for name in CURRENT_INSTRUCTIONS:regular(ROOT/name)
    spec=importlib.util.spec_from_file_location('predictor_ce_contract_only',DEVELOPMENT/'contracts.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(module.CONTRACT['cycles']==32 and module.CONTRACT['forward_slots_planned']==384 and module.CONTRACT['backward_calls_max']==128 and module.CONTRACT['output_bytes']==2560*1024**2 and module.CONTRACT['terminal_bytes']==1024**3 and module.CONTRACT['minimum_free_bytes']==8*1024**3,'Final predictor contract differs')
    need(subset['complete'] and subset['independent'] and subset['subset_cases']==128,'Balanced source subset not admitted')
    return design,v4,subset,module.CONTRACT

def stage(stage_id):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,48}',stage_id) is not None,'Invalid stage name')
    base=PACKETS/stage_id;destination=base/VM_NAME;need(not base.exists(),'Preserve prior stage; choose a fresh stage id')
    design,v4,subset,contract=dependencies();v4files={row['path']:row for row in v4['files']};planned={}
    def add(relative,source,expected=None):
        safe_relative(relative);source=Path(source);regular(source)
        if expected is not None:need(source.stat().st_size==expected['bytes'] and sha(source)==expected['sha256'],'Exact source asset drift '+relative)
        if relative in planned:need(planned[relative].resolve()==source.resolve(),'Different duplicate member');return
        planned[relative]=source
    def frozen(relative):
        need(relative in v4files,'Selected asset absent from frozenV4source '+relative);add(relative,V4_STAGE/relative,v4files[relative])
    fit=copy.deepcopy(subset['cases_in_reference_role_order']);monitor=copy.deepcopy(v4['monitor_cases']);challenge=copy.deepcopy(v4['challenge_cases']);public=copy.deepcopy(v4['development_cases'])
    parity=[copy.deepcopy(row) for row in v4['fit_cases'] if row['role']=='fit_anchor' and row['reference_id'] in sorted({r['reference_id'] for r in v4['fit_cases']})[:2] and row['mask_kind'] in ('lower_face','eye_band')]
    need(len(fit)==128 and len(monitor)==32 and len(challenge)==16 and len(public)==32 and len(parity)==4,'Exact selectedcase counts')
    for case in fit+monitor+challenge+parity:
        for key in ('input256','clean256','clean512','mask'):frozen(case[key])
    for case in public:
        for key in ('input256','mask'):frozen(case[key])
    targets=copy.deepcopy(v4['teacher_targets']);need(len(targets)==80,'Existing80style target metadata')
    for row in targets:
        frozen(row['clean256']);frozen(row['clean512'])
    labels=[];index=read(V2_RESULTS/'outputs/teacher_labels/index.json')
    for row in index['labels']:
        path=V2_RESULTS/'outputs'/row['path'];relative='cached_teacher_labels/'+path.name;add(relative,path,row)
        labels.append({'teacher_key':row['id'],'path':relative,'bytes':row['bytes'],'sha256':row['sha256'],'clean512':row['clean512'],'clean512_sha256':row['clean512_sha256'],'teacher_original_logical_sha256':row['teacher_original_logical_sha256'],'original_teacher_weights_sha256':row['original_teacher_weights_sha256'],'unused_prequant_latent_retained':True,'monitor_labels_never_optimized':row['id'].split('::')[0] not in {case['reference_id'] for case in fit}})
    need(len(labels)==80,'All80cachelabels required')
    historical=copy.deepcopy(v4['historical_control']['initial_rows'])
    for rows in historical.values():
        for row in rows:
            frozen(row['result256']);frozen(row['PNG'])
    frozen(v4['historical_control']['derivation'])
    for name,source in SOURCE_MAP.items():add(name,source)
    evidence={};evidence_hashes={}
    for key,(path,digest) in EVIDENCE.items():
        relative='evidence/'+key+path.suffix;add(relative,path);evidence[key]=relative;evidence_hashes[key]=digest
    for key,row in design['evidence'].items():
        relative='evidence/root_design_evidence/'+key+Path(row['path']).suffix;add(relative,Path(row['path']),row);evidence[key]=relative;evidence_hashes[key]=row['sha256']
    add('evidence/formal_design_admission.json',FORMAL_ADMISSION_PATH);evidence['formal_design_admission']='evidence/formal_design_admission.json';evidence_hashes['formal_design_admission']=FORMAL_ADMISSION_SHA
    add('evidence/balanced128_subset.json',SUBSET_PATH);add('evidence/original_R1_protocol.json',PARENT_PROTOCOL)
    for name in CURRENT_INSTRUCTIONS:add('evidence/current_instructions/'+name,ROOT/name)
    for name in ('source_tests.py','source_preparation_tests.json','draft_status_v1.json','cpu_clone_alias_fixture.json'):add('evidence/development/'+name,DEVELOPMENT/name)
    candidate=ROOT/'outputs/completion_codeformer_finetune_v1_r1_candidate'
    for name in ('LICENSE','NOTICE.md'):add('legal/'+name,candidate/name)
    add('evidence/original_source_admission.json',ROOT/'outputs/completion_codeformer_manual_v1/source_admission.json')
    records=[{'path':name,'bytes':regular(path).st_size,'sha256':sha(path)} for name,path in sorted(planned.items())]
    need(records and sum(row['bytes'] for row in records)<MAX_EXECUTION_BYTES,'Executionpayload256MiBcap')
    protocol={'format':'completion-codeformer-predictor-ce-v1','status':'staged_source_only_unrun_unreleased','frozen_UTC':datetime.now(timezone.utc).isoformat(),'contract':contract,'checkpoint':copy.deepcopy(v4['checkpoint']),'parent':copy.deepcopy(v4['parent']),'references':copy.deepcopy(v4['references']),'teacher_targets':targets,'cached_labels':labels,'fit_cases':fit,'monitor_cases':monitor,'challenge_cases':challenge,'development_cases':public,'parity_cases':parity,'schedule':copy.deepcopy(subset['schedule']),'balanced_subset':'evidence/balanced128_subset.json','original_parent_protocol':'evidence/original_R1_protocol.json','historical_control':{'initial_rows':historical,'derivation':v4['historical_control']['derivation'],'original_source_archive_sha256':v4['historical_control']['source_archive_sha256']},'evidence':evidence,'evidence_sha256':evidence_hashes,'terms':v4['terms'],'data_limits':v4['data_limits'],'source_subset':{'only128 selectedFITobservations':True,'retained256pool_modified':False,'source_V4_protocol_sha256':V4_PROTOCOL_SHA,'teacher80cached_same_inpainting_codebook':True,'FIT_labels_64_only_optimized':True,'monitor16labels_never_optimized':True,'public_hidden_reference_metrics':False,'FINAL_data_used':False},'instruction_snapshot':{'UTC':datetime.now(timezone.utc).isoformat(),'status':'Exact prospective/source-only snapshots, not live transfer readiness; root release and current guide govern later status.','files':{'evidence/current_instructions/'+name:pin(ROOT/name) for name in CURRENT_INSTRUCTIONS}},'export':{'stem':STEM,'whole_output_cap_bytes':2560*1024**2,'raw_tar_and_gzip_cap_bytes_each':2816*1024**2,'cap_seconds':900,'minimum_free_bytes':2*1024**3,'maximum_regular_members_including_manifest':1000,'maximum_relative_path_UTF8_bytes':240,'NPZ_generated_max':388,'NPZ_cached_teacher_copies_max':80,'NPZ_total_max':468,'terminal_bytes':1024**3,'failed_partials_retained':True,'checkpoint_same_inode_commit_alias':'Export committedpayloadonce only ifsame device/inode/uid/nlink>=2/size/hash; recordandpreserve VMsourcealias. Distinctcheckpointpair refuses.'},'files':records,'manual_execution_only':True,'automatic_resume_or_promotion':False,'optimizer_updates_locally':0,'local_neural_calls':0,'model_qualified':False,'goal_complete':False,'release_required_before_archive':True}
    body=(json.dumps(protocol,indent=2,allow_nan=False)+'\n').encode();need(len(body)<=1024**2 and len(body)+sum(row['bytes'] for row in records)<MAX_EXECUTION_BYTES,'Protocol/payloadcaps')
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
    actual={q.relative_to(destination).as_posix() for q in destination.rglob('*') if q.is_file()};need(actual=={row['path'] for row in records}|{'protocol.json'},'Closedstage namespace')
    result={'complete':True,'source_only_stage':True,'stage_id':stage_id,'root':str(destination),'protocol_sha256':sha(destination/'protocol.json'),'files':len(records),'payload_bytes':len(body)+sum(row['bytes'] for row in records),'archive_prepared':False,'manual_release_required':True,'model_loads':0,'neural_calls':0,'VM_calls':0,'optimizer_updates':0}
    write(base/'stage_receipt.json',result);print(json.dumps(result),flush=True);return result

INSTALL_CODE=r'''from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile, uuid

def need(ok, message):
    if not ok:
        raise ValueError(message)
base = Path.home() / 'forensic-dgp'
name = 'cctv_completion_codeformer_predictor_ce_v1_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz'
need(base.is_dir() and (not destination.exists()), 'Preserve any existing experiment; do not reinstall')
need(archive.stat().st_size <= 256 * 1024 ** 2, 'Execution archive size cap')

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
    need(p['format'] == 'completion-codeformer-predictor-ce-v1' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 256 * 1024 ** 2 and shutil.disk_usage(base).free >= 8 * 1024 ** 3 + total, 'Need8GiB plus installation payload free')
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
    return '''# Predictor-only CodeFormer completion V1: manual finite L4 pilot

RELEASED FOR MANUAL TRANSFER ONLY. Preparation has launched no training or VM action.
R1 remains selected; no candidate is qualified by packet creation or training completion.
Source archive: __ARCHIVE_BYTES__ bytes, __FILE_COUNT__ pinned payloads plus protocol.
Archive SHA256: __ARCHIVE_SHA__.

## 1. Upload from Windows Google Cloud SDK Shell

Run on Windows, not inside VM SSH.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and safely install in VM SSH

Require 8 GiB plus exact installation payload free, the original inpainting checkpoint,
and full trained R1 state already present on this VM. This packet duplicates neither
large checkpoint. A different VM needs verified dependencies first. The fresh
installer refuses existing roots and retains failed staging; do not bypass a refusal.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-predictor-ce-v1-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test -f ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth &&
python3 - <<'PY'
__INSTALL_CODE__PY
```

Expected complete:true, protocol __PROTOCOL_SHA__, training_launched:false.

## 3. Open tmux

Use your current session if already inside tmux. Otherwise:

```bash
tmux new-session -A -s completion_predictor_ce_v1
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_predictor_ce_v1_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py --root . --protocol-sha __PROTOCOL_SHA__ --verify-transfer &&
bash scripts/run_completion_codeformer_predictor_ce_v1.sh __PROTOCOL_SHA__
```

Transfer verification uses no model calls. The shell starts one distinct finite
predictor-only CE test from full R1: maximum 32 Adam updates /128 backwards on exactly
128 selected FIT observations. This is not an epoch over the retained 256-case pool.
Only 114 existing predictor tensors train ;383 encoder/fusion/prior tensors stay exact.
R1's 114 predictor Adam moments map by name from step 32 to 64; frozen 164 encoder and 54 fusion
slots remain unchanged at 32. LR1e-5 and existing optimizer settings remain fixed.
No optimizer reset, latent/image/GAN loss, fusion/encoder learning or oracle inference.

Reuse 80 exact cached original same-codebook label sets: 64 FIT targets and 16 monitor
diagnostics. The 16 monitor sets never optimize. No new teacher call/label generation.
All 32 clear/16 challenge paired cases read out at 0,8,16,32; 32 public unpaired cases at 0/32.
Four fresh ordinary app parity pairs and all 48 initial R1 raw/PNG matches must pass
before updates. Expected 256 evaluation outputs +4 parity arrays +128 FIT logits arrays,
plus 80 retained teacher NPZ copies. Public outputs have no hidden clean-target metrics.
If a quality stop occurs at8/16, its stopped-cycle public readout replaces the planned
final32 readout; no third public readout or extra work after a resource stop is allowed.

Planned 384 network calls =128 code-only FIT  +248 decoded evaluation  +8 parity decoded.
Fullgeneration class cap 256, code-only 128; global hard 420 checked before dispatch.
168 empty bypasses remain exact with no neural call. Empty saved raw/codes are marked
absent. Save every FIT logit/label/case/cycle/microbatch for independent CE replay.
Snapshots 8/16 are safety checks, not selectable winners. Final 32 must pass separate
clear/challenge>=1%PNGholeerror gains, all oldRAW/PNGhole/boundary subgroup nonworse,
exact outside-source preservation, and the new strict-gray macro/fourgeometry channel-
spread guard. Gray color errors, lower CE or higher code agreement do not imply quality.

Finite worker 1800 seconds hard/1680 cooperative with 120 seconds save reserve;20 GiB CUDA.
Require 8 GiB free after installation, 2 GiB floor. Whole output 2.5 GiB, terminal or failed
partial 1 GiB, teacher 24 MiB, metadata 8 MiB including 4 MiB stdout, PNG 256 KiB each.
Streamed rawtar and gzip each 2.75 GiB; export 900 seconds. No second rawtar disk copy.
All source/parent checkpoints and stopping evidence remain preserved. Save once; internal ZIP/member/primitive-pickle/CRC checks run on the closed bounded
partial before commit. The total 1 GiB cap is enforced before every write. A rejected
partial remains evidence and is never automatically retried. Actual failed-prior 165 tensors and partial 114 gradients can be retained within cap.
A hardkill/storagefailure can still prevent recoverability; no automatic resume.

Monitor from a second VM SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_predictor_ce_v1_vm/trainer.log
```

Detach with Ctrl+B then D. At completion:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_predictor_ce_v1_vm/supervisor.json
cat ~/cctv-completion-codeformer-predictor-ce-v1-export.json
```

Worker exit 2 is a quality stop, not a qualified new model. Complete export means
packaging only. Independent weights/Adam/RNG/logit/RAW/PNG audits and all-case visual
review, broader development, user acceptance and final review precede any adoption.
The successful archive.partial is one same-inode hardlink, not duplicate disk data.
Rare same-inode terminal commit aliases are recorded and preserved at source while
one payload is archived; distinct checkpoint files are never hash-only deduplicated.

## 5. Download from Windows Google Cloud SDK Shell

One remote source per command; run on Windows, not in VM SSH.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-predictor-ce-v1-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-predictor-ce-v1-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-predictor-ce-v1-export.json" "."
```

Return all three files for independent admission. Do not automatically repeat or
continue this historical pilot. The working R1 app and restoration remain unchanged.
'''.replace('__ARCHIVE_BYTES__',str(archive_bytes)).replace('__FILE_COUNT__',str(file_count)).replace('__ARCHIVE_SHA__',archive_sha).replace('__PROTOCOL_SHA__',protocol_sha).replace('__INSTALL_CODE__',install)

def archive_stage(stage_id,release_path):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,48}',stage_id) is not None,'Invalid stage name')
    stagebase=PACKETS/stage_id;stage=stagebase/VM_NAME;ppath=stage/'protocol.json';need(ppath.is_file(),'Stage source first');expected=sha(ppath);p=read(ppath)
    need((RESEARCH/'protocol.json').read_bytes()==ppath.read_bytes(),'Research protocol is a different stage')
    release=read(release_path);need(release.get('format')=='completion-codeformer-predictor-ce-v1-root-release-v1' and release.get('released_for_manual_vm_transfer') is True and release.get('automatic_vm_launch') is False,'Root manual-transfer release required')
    need(release.get('protocol_sha256')==expected,'Release protocol differs')
    admission_pin=release['independent_source_admission'];admission_path=Path(admission_pin['path']);need(pin(admission_path)==admission_pin,'Independent admission pin differs')
    admission=read(admission_path);need(admission.get('complete') is True and admission.get('independent') is True and admission.get('protocol_sha256')==expected,'Independent final source admission required')
    need(p['files'] and p['format']=='completion-codeformer-predictor-ce-v1' and p['manual_execution_only'] and not p['automatic_resume_or_promotion'],'Manual source closure differs')
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
    for target in (ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md',TRANSFER/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md'):
        # Earlier source-only placeholder is retained explicitly before replacement.
        if target.exists():
            prior=stagebase/('prior_guide_'+('transfer' if target.parent==TRANSFER else 'worktree')+'.md')
            with prior.open('xb') as stream:stream.write(target.read_bytes())
        target.write_text(guide,encoding='utf-8',newline='\n')
    receipt={'complete':True,'archive':pin(destination),'archive_sha256':digest,'protocol_sha256':expected,'payload_files':len(files)+1,'payload_bytes':total,'root_release':pin(Path(release_path)),'independent_source_admission':admission_pin,'published':published,'manual_VM_guide':pin(TRANSFER/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md'),'successful_archive_partial_hardlink_retained':str(partial),'training_launched':False,'model_qualified':False,'VM_calls':0}
    write(stagebase/'archive_publication_receipt.json',receipt);print(json.dumps(receipt),flush=True);return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--archive',action='store_true');parser.add_argument('--stage-id',required=True);parser.add_argument('--release-file',type=Path)
    args=parser.parse_args()
    if args.stage:need(args.release_file is None,'Staging needs no release-file');stage(args.stage_id)
    else:need(args.release_file is not None,'Root release file required for archive');archive_stage(args.stage_id,args.release_file)
if __name__=='__main__':main()