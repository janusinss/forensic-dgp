"""Stage source-only coverage-v2 files; archive only after independent/root release.
No Torch imports, models, VM/network operations, training, cleanup or overwrite.
"""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.util,io,json,os,re,shutil,stat,tarfile,time
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/completion_codeformer_coverage_v2_data'
OLD=ROOT/'outputs/completion_codeformer_finetune_v1_packet/cctv_completion_codeformer_finetune_v1_vm'
PARENT_RETURN=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return'
RESEARCH=ROOT/'research/completion_codeformer_coverage_v2'
PACKETS=ROOT/'outputs/completion_codeformer_coverage_v2_packet'
TRANSFER=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
VM_NAME='cctv_completion_codeformer_coverage_v2_vm'
STEM='cctv-completion-codeformer-coverage-v2'
MAX_EXECUTION_BYTES=256*1024**2
SOURCE_LIST=['scripts/cctv_completion_codeformer_coverage_v2_vm.py','research/completion_codeformer_coverage_v2/contracts.py','scripts/export_completion_codeformer_coverage_v2.py','scripts/run_completion_codeformer_coverage_v2.sh','scripts/prepare_completion_codeformer_coverage_v2.py','completion.py','pretrained_completion.py','face_workflow.py','third_party/__init__.py','third_party/codeformer/__init__.py','third_party/codeformer/registry.py','third_party/codeformer/vqgan_arch.py','third_party/codeformer/codeformer_arch.py']
PINS={'data_manifest':'393bea14e4ba8a350644bc51d65c68a0eb57ba919b1c9d0954245c1f40935ea7','data_admission':'7c072ec5a916468275e4e5ba3e3124e9b472030014f1e7afed803f5beca3d63a','root_visual':'8488e9354a87ce186931c2770a9ce6a8de0da06b92baddba2c6d593aea54db72','design':'9566e77f7fb33b4116f744d043ccf5af1ff133222fb69d6ad0b047db324a4682','design_md':'09d11298cf5a23e91f26b46a2a9c0b4af8477d8c44530664ac06b759086728f3','parent_state_audit':'e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06','parent_pixels_audit':'ab4bd40838e33bce240d28265ca727c4ac52e96e0b78ff8ad5abc67854b5b65e','parent_protocol':'8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'}

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
    need(sha(DATA/'manifest.json')==PINS['data_manifest'],'Frozen data manifest differs')
    m=read(DATA/'manifest.json');need(m['complete'] and len(m['files'])==551,'Data sourcefreeze differs')
    for row in m['files']:
        file=Path(row['path']);regular(file);need(file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],'Frozen data/source drift '+row['path'])
        need(file.resolve().is_relative_to(DATA.resolve()),'Data source escaped freeze')
    locations={'data_admission':DATA/'independent_data_admission_v1.json','root_visual':ROOT/'outputs/completion_codeformer_coverage_v2_root_data_visual_review.json','design':DATA/'evidence/design_review_final.json','design_md':DATA/'evidence/design_final.md','parent_state_audit':PARENT_RETURN/'independent_state_audit.json','parent_pixels_audit':PARENT_RETURN/'independent_saved_output_audit.json','parent_protocol':ROOT/'outputs/completion_codeformer_finetune_v1_r1_packet/protocol.json'}
    for key,path in locations.items():need(sha(path)==PINS[key],'Admission/design/evidence changed '+key)
    for name in SOURCE_LIST:regular(ROOT/name)
    spec=importlib.util.spec_from_file_location('coverage_v2_contract_only',ROOT/'research/completion_codeformer_coverage_v2/contracts.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    need(module.CONTRACT['cycles']==64 and module.CONTRACT['output_bytes']==2304*1024**2 and module.CONTRACT['teacher_bytes']==24*1024**2,'Final coverage contract differs')
    return m,locations,module.CONTRACT

def normalized_case(row):
    def target(key):return 'coverage_data/'+safe_relative(row[key]['relative_path'])
    return {'id':row['id'],'reference_id':row['reference_id'],'teacher_key':row['teacher_key'],'role':row['role'],'original_role':row['original_role'],'legacy_completion_split':row['legacy_completion_split'],'condition':row['condition'],'style':row['style'],'mask_kind':row['mask_kind'],'input256':target('input256'),'clean256':target('target256'),'clean512':target('target512'),'mask':target('mask256'),'mask_pixels':row['mask_pixels'],'prepared_bounds':row['prepared_bounds'],'case_seed':row['case_seed'],'irregular_mask':row['mask_transform']['enabled'],'metadata_case_id':row['id'],'appearance_matched_clean_target':True}

def stage(stage_id):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,48}',stage_id) is not None,'Invalid stage name')
    m,locations,contract=dependencies();stagebase=PACKETS/stage_id;destination=stagebase/VM_NAME;need(not stagebase.exists(),'Preserve prior stage; choose a fresh stage id')
    # All requested dependencies are checked before this stage writes anything.
    planned={}
    def add(relative,source):
        safe_relative(relative);need(relative not in planned,'Duplicate packet member');regular(source);planned[relative]=Path(source)
    for row in m['files']:add('coverage_data/'+safe_relative(row['relative_path']),Path(row['path']))
    add('coverage_data/manifest.json',DATA/'manifest.json')
    for name in SOURCE_LIST:add(name,ROOT/name)
    mapping={'data_admission':'evidence/independent_data_admission_v1.json','root_visual':'evidence/root_data_visual_review.json','parent_state_audit':'evidence/parent_state_audit.json','parent_pixels_audit':'evidence/parent_saved_output_audit.json'}
    for key,name in mapping.items():add(name,locations[key])
    parent_protocol=read(locations['parent_protocol'])
    original_summary=PARENT_RETURN/'cctv_completion_codeformer_finetune_v1_r1_results/outputs/u0000/monitor/results.json';regular(original_summary);add('evidence/original_codeformer_monitor_summary.json',original_summary)
    add('evidence/source_admission_original_codeformer.json',ROOT/'outputs/completion_codeformer_manual_v1/source_admission.json')
    candidate=ROOT/'outputs/completion_codeformer_finetune_v1_r1_candidate'
    for name in ('LICENSE','NOTICE.md'):add('legal/'+name,candidate/name)
    dev=[]
    for case in parent_protocol['development_cases']:
        row=copy.deepcopy(case)
        for field,prefix in [('input256','data/public/input256'),('mask','data/public/masks256')]:
            relative=prefix+'/'+case['id']+'.png';add(relative,OLD/case[field]);row[field]=relative
        dev.append(row)
    target_rows=[]
    references=[]
    for reference in m['references']:
        references.append({'id':reference['id'],'role':reference['role'],'original_role':reference['original_role'],'legacy_completion_split':reference['legacy_completion_split'],'attribution':reference['attribution'],'source1024_pin':reference['original_source1024']})
    roles={r['id']:r['role'] for r in references}
    for row in m['teacher_target_sets']:
        target_rows.append({'id':row['teacher_key'],'cache_id':row['reference_id']+'_'+row['style'].lower(),'reference_id':row['reference_id'],'style':row['style'],'role':roles[row['reference_id']],'clean256':'coverage_data/'+row['target256']['relative_path'],'clean512':'coverage_data/'+row['target512']['relative_path'],'teacher_semantics':row['teacher_semantics']})
    sampler=read(DATA/'sampler.json');schedule=[row['case_ids'] for row in sampler['cycles']]
    need(len(schedule)==64 and all(row['case_ids']==row['code_case_ids']==row['fusion_case_ids'] for row in sampler['cycles']),'Sampler routes differ')
    files=[{'path':relative,'bytes':regular(source).st_size,'sha256':sha(source)} for relative,source in sorted(planned.items())]
    need(files and sum(row['bytes'] for row in files)<MAX_EXECUTION_BYTES,'Source archive payload cap')
    protocol={'format':'completion-codeformer-coverage-v2','status':'staged_source_only_unrun_unreleased','frozen_UTC':datetime.now(timezone.utc).isoformat(),'contract':contract,'checkpoint':{'path':'../cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth','bytes':370786611,'sha256':'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'},'parent':{'path':'../cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth','bytes':701952165,'sha256':'0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd','protocol_sha256':PINS['parent_protocol'],'logical_sha256':'55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6','completed_paired_cycles':32,'optimizer_steps_per_route':32},'references':references,'teacher_targets':target_rows,'fit_cases':[normalized_case(row) for row in m['fit_anchor_cases']+m['fit_augmented_cases']],'monitor_cases':[normalized_case(row) for row in m['monitor_original_cases']],'challenge_cases':[normalized_case(row) for row in m['monitor_challenge_cases']],'development_cases':dev,'development_excluded_before_generation':parent_protocol['development_excluded_before_generation'],'schedule':schedule,'sampler_source':'coverage_data/sampler.json','data_manifest':'coverage_data/manifest.json','original_parent_protocol':'coverage_data/evidence/R1_protocol.json','original_monitor_summary':'evidence/original_codeformer_monitor_summary.json','evidence':{'design_review':'coverage_data/evidence/design_review_final.json','data_admission':mapping['data_admission'],'root_visual_review':mapping['root_visual'],'parent_state_audit':mapping['parent_state_audit'],'parent_saved_output_audit':mapping['parent_pixels_audit']},'evidence_sha256':{'design_review':PINS['design'],'data_admission':PINS['data_admission'],'root_visual_review':PINS['root_visual'],'parent_state_audit':PINS['parent_state_audit'],'parent_saved_output_audit':PINS['parent_pixels_audit']},'teacher_semantics':'Original frozen inpainting encoder prequantlatent andnearest quantizer codes, 40RGB+40GRAY sets; same original definition, not official quantizedHQteacher reproduction','teacher_retention':'Retain all80 latent/code NPZ arrays under24MiB cap; filenames use safe cache_id, not colon teacherid','terms':parent_protocol['terms'],'development_family_counts':parent_protocol['development_family_counts'],'data_limits':m['data_limits'],'export':{'stem':STEM,'cap_bytes':2304*1024**2,'cap_seconds':900,'minimum_free_bytes':2*1024**3,'original_or_parent_weights_included':False,'failed_partials_retained':True},'files':files,'manual_execution_only':True,'automatic_resume_or_promotion':False,'local_neural_calls':0,'optimizer_updates_locally':0,'model_qualified':False,'goal_complete':False,'release_required_before_archive':True}
    body=(json.dumps(protocol,indent=2,allow_nan=False)+'\n').encode();need(sum(row['bytes'] for row in files)+len(body)<MAX_EXECUTION_BYTES,'Protocol plus payload cap')
    destination.mkdir(parents=True)
    for row in files:
        source=planned[row['path']];dst=destination/row['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as incoming,dst.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing,8*1024*1024)
        need(dst.stat().st_size==row['bytes'] and sha(dst)==row['sha256'] and sha(source)==row['sha256'],'Staging drift '+row['path'])
    with (destination/'protocol.json').open('xb') as stream:stream.write(body)
    research_protocol=RESEARCH/'protocol.json'
    if research_protocol.exists():
        with (stagebase/'previous_research_protocol.json').open('xb') as stream:stream.write(research_protocol.read_bytes())
    research_protocol.write_bytes(body)
    actual={p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    need(actual=={r['path'] for r in files}|{'protocol.json'},'Stage namespace differs')
    receipt={'complete':True,'source_only_stage':True,'stage_id':stage_id,'root':str(destination),'protocol_sha256':sha(destination/'protocol.json'),'files':len(files),'payload_bytes':sum(r['bytes'] for r in files)+len(body),'archive_prepared':False,'manual_release_required':True,'model_loads':0,'neural_calls':0,'VM_calls':0}
    write(stagebase/'stage_receipt.json',receipt);print(json.dumps(receipt),flush=True);return receipt

INSTALL_CODE=r'''from pathlib import Path, PurePosixPath
import hashlib, json, shutil, tarfile, uuid

def need(ok, message):
    if not ok:
        raise ValueError(message)
base = Path.home() / 'forensic-dgp'
name = 'cctv_completion_codeformer_coverage_v2_vm'
destination = base / name
archive = Path.home() / 'cctv-completion-codeformer-coverage-v2-execution.tar.gz'
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
    need(p['format'] == 'completion-codeformer-coverage-v2' and p['files'], 'Wrong/empty protocol')
    expected = {name + '/' + safe(r['path']).as_posix(): r for r in p['files']}
    need(len(expected) == len(p['files']), 'Duplicate protocol member')
    total = len(body) + sum((r['bytes'] for r in p['files']))
    need(total <= 256 * 1024 ** 2 and shutil.disk_usage(base).free >= 7 * 1024 ** 3 + total, 'Need7GiB plus installation payload free')
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
    return '''# CodeFormer completion coverage V2: manual VM continuation

RELEASED FOR MANUAL TRANSFER ONLY. No training has been launched by preparation.
Start from the retained trained R1 state and its complete Adam/RNG/sampler evidence.
Original CodeFormer and every prior state/failure remain preserved.

The source upload is __ARCHIVE_BYTES__ bytes and contains __FILE_COUNT__ pinned
payload files plus its protocol. It reuses original weights and full R1 terminal
state already on the same L4 VM. No duplicate large checkpoint is uploaded.

## 1. Upload from Windows Google Cloud SDK Shell

Run these on Windows, where the prompt starts with C:, not inside VM SSH.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-coverage-v2-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "cctv-completion-codeformer-coverage-v2-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

Archive SHA256: __ARCHIVE_SHA__.

## 2. Verify and safely install in VM SSH

This checks every regular member and its hash while extracting to a fresh staging
folder, then publishes the new directory. It preserves prior experiments and any
failed installation staging. Requires 7 GiB plus the small installation payload.

```bash
cd ~ &&
sha256sum -c cctv-completion-codeformer-coverage-v2-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth &&
test -f ~/forensic-dgp/cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth &&
python3 - <<'PY'
__INSTALL_CODE__PY
```

Expected complete:true, protocol __PROTOCOL_SHA__, training_launched:false.
If installation refuses, retain the error and existing folders. Do not repeat an
old pilot or bypass hashes. A different VM requires the verified parent state,
original checkpoint and environment before these commands apply.

## 3. Open tmux

If already inside tmux, use the current session. Otherwise:

```bash
tmux new-session -A -s completion_coverage_v2
```

## 4. Verify and launch inside tmux

```bash
cd ~/forensic-dgp/cctv_completion_codeformer_coverage_v2_vm &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_completion_codeformer_coverage_v2_vm.py --root . --protocol-sha __PROTOCOL_SHA__ --verify-transfer &&
bash scripts/run_completion_codeformer_coverage_v2.sh __PROTOCOL_SHA__
```

Transfer verification performs no neural or training work. The next command starts
the new finite continuation. It restores both R1 Adam states at step 32, retains
R1 RNG and a declared new sampler phase, and keeps the generator/codebook frozen.
64 additional paired cycles means 128 new Adam updates and 512 new backwards;
terminal step is at most 96 per route (192 cumulative updates across both phases).

256 FIT pairs include all 128 original clear anchors and 128 reviewed appearance/
context/mask augmentations. The original 32-case TRAIN monitor remains fixed.
16 new challenges use the same eight adaptive TRAIN monitor sources and are
reported separately; all 32 public cases remain unpaired, without clean targets.
Both input and clean targets receive the declared gray appearance transformation.
Synthetic mild degradation changes only observed input, never the clean target.
These cases are not measured native CCTV, independent identities or FINAL data.

80 original-frozen RGB/gray label sets are retained for audit. Planned work is
896 typed slots (736 whole-model forwards plus 160 teacher components), capped
at 1000. Four fresh frozen-R1 app comparisons must pass the unchanged 1e-6 gate
before updates. Existing losses, LR 1e-5 and clipping remain unchanged.

Worker hard cap: 30 minutes, with a 28-minute cooperative stop for state retention.
Start free disk: 7 GiB; stop floor: 2 GiB. CUDA allocation cap: 20 GiB. Returned
output and both raw/compressed export cap: 2304 MiB. Export cooperative cap:
15 minutes. Failed partial states remain evidence; no automatic resume or earlier
snapshot rescue. Limits are enforced stops, not promises of training completion.

Monitor from a second SSH terminal:

```bash
tail -F ~/forensic-dgp/cctv_completion_codeformer_coverage_v2_vm/trainer.log
```

Detach safely with Ctrl+B then D. At the end:

```bash
cat ~/forensic-dgp/cctv_completion_codeformer_coverage_v2_vm/supervisor.json
cat ~/cctv-completion-codeformer-coverage-v2-export.json
```

A complete export means packaging succeeded. Quality requires the predeclared
incremental gain versus frozen R1 on both separately reported monitor cohorts,
nonworsening declared RAW/PNG groups, independent pixel/state audit, all planned
visual reviews and user acceptance. No hidden-identity or full-family qualification
follows from numeric screening. Original/source visible pixels remain exact.

Successful export keeps its .partial pathname as a hard link to the same archive
inode; the receipt explicitly records it, and it consumes no duplicate payload
storage. Failed archive/state partials also remain retained; do not delete them.

## 5. Download from Windows Google Cloud SDK Shell

Use one remote source per command. Run on Windows, not inside VM SSH.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-coverage-v2-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-coverage-v2-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-completion-codeformer-coverage-v2-export.json" "."
```

Tell this chat when all three files are downloaded. Keep original R1 and CodeFormer
available while the returned candidate is independently reviewed before adoption.
'''.replace('__ARCHIVE_BYTES__',str(archive_bytes)).replace('__FILE_COUNT__',str(file_count)).replace('__ARCHIVE_SHA__',archive_sha).replace('__PROTOCOL_SHA__',protocol_sha).replace('__INSTALL_CODE__',install)


def archive_stage(stage_id,release_path):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,48}',stage_id) is not None,'Invalid stage name')
    stagebase=PACKETS/stage_id;stage=stagebase/VM_NAME;ppath=stage/'protocol.json';need(ppath.is_file(),'Stage source first');expected=sha(ppath);p=read(ppath)
    need((RESEARCH/'protocol.json').read_bytes()==ppath.read_bytes(),'Research protocol is a different stage')
    release=read(release_path);need(release.get('format')=='completion-codeformer-coverage-v2-root-release-v1' and release.get('released_for_manual_vm_transfer') is True and release.get('automatic_vm_launch') is False,'Root manual-transfer release required')
    need(release.get('protocol_sha256')==expected,'Release protocol differs')
    admission_pin=release['independent_source_admission'];admission_path=Path(admission_pin['path']);need(pin(admission_path)==admission_pin,'Independent admission pin differs')
    admission=read(admission_path);need(admission.get('complete') is True and admission.get('independent') is True and admission.get('protocol_sha256')==expected,'Independent final source admission required')
    need(p['files'] and p['format']=='completion-codeformer-coverage-v2' and p['manual_execution_only'] and not p['automatic_resume_or_promotion'],'Manual source closure differs')
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
    for target in (ROOT/'COMPLETION_CODEFORMER_COVERAGE_V2_VM.md',TRANSFER/'COMPLETION_CODEFORMER_COVERAGE_V2_VM.md'):
        # Earlier source-only placeholder is retained explicitly before replacement.
        if target.exists():
            prior=stagebase/('prior_guide_'+('transfer' if target.parent==TRANSFER else 'worktree')+'.md')
            with prior.open('xb') as stream:stream.write(target.read_bytes())
        target.write_text(guide,encoding='utf-8',newline='\n')
    receipt={'complete':True,'archive':pin(destination),'archive_sha256':digest,'protocol_sha256':expected,'payload_files':len(files)+1,'payload_bytes':total,'root_release':pin(Path(release_path)),'independent_source_admission':admission_pin,'published':published,'manual_VM_guide':pin(TRANSFER/'COMPLETION_CODEFORMER_COVERAGE_V2_VM.md'),'successful_archive_partial_hardlink_retained':str(partial),'training_launched':False,'model_qualified':False,'VM_calls':0}
    write(stagebase/'archive_publication_receipt.json',receipt);print(json.dumps(receipt),flush=True);return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--stage',action='store_true');mode.add_argument('--archive',action='store_true');parser.add_argument('--stage-id',required=True);parser.add_argument('--release-file',type=Path)
    args=parser.parse_args()
    if args.stage:need(args.release_file is None,'Staging needs no release-file');stage(args.stage_id)
    else:need(args.release_file is not None,'Root release file required for archive');archive_stage(args.stage_id,args.release_file)
if __name__=='__main__':main()