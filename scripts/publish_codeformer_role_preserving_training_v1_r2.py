"""Publish only sealed/manual R2 training commands; preserve six prior documents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/codeformer_role_preserving_training_v1_r2_preparation'
PREFIX='cctv-codeformer-role-preserving-training-v1-r2'
VM='cctv_codeformer_role_preserving_training_v1_r2_vm'
PARENT='cctv_codeformer_adain_gradient_isolation_v1_vm'
OUTPUT='outputs/role_preserving_training_v1_r2'
GUIDE='CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_VM.md'
PREPARATION='CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_PREPARATION.md'
RETURN_REPORT='CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_RETURN_REVIEW.md'
REVIEW=ROOT/'outputs/codeformer_role_preserving_training_v1_r1_return_review'
OLD_PROTOCOL='4edacae4f1e61f06b21fb1308423398d5b9f842def160c02f74f8edc0975fc40'
OLD_ARCHIVE='075d0fcc9b6dbd35e1ba65050c7837c00f1bd64afb44164a2a45ae991f1ab2e1'
CURRENT_DOCS=('SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
              'PROJECT_HANDOFF.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
HISTORICAL_DOCS=('CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_VM.md',
                 'CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R1_PREPARATION.md')


def need(value,message):
    if not value:raise ValueError(message)


def safe(path,regular=False):
    path=Path(path).absolute()
    need(path.resolve(strict=False).is_relative_to(ROOT),'Publication path outside workspace')
    for item in (path,*path.parents):
        if item.exists()or item.is_symlink():
            s=item.lstat()
            need(not stat.S_ISLNK(s.st_mode)and not getattr(s,'st_file_attributes',0)&1024,
                 'Linked/reparse publication path forbidden')
    if regular:
        s=path.lstat();need(stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Unlinked regular publication source required')
    return path


def sha(path):
    path=safe(path,True);before=path.stat();h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    after=path.stat()
    need((before.st_size,before.st_mtime_ns,before.st_ino)==
         (after.st_size,after.st_mtime_ns,after.st_ino),'Publication source changed during hash')
    return h.hexdigest()


def load(path,cap=2*1024**2):
    path=safe(path,True);need(path.stat().st_size<=cap,'Publication metadata exceeds bounded cap')
    return json.loads(path.read_bytes())


def new(path,payload):
    path=safe(path);need(not path.exists()and not path.is_symlink(),'Preserve prior/partial publication: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb')as stream:stream.write(payload)


def main():
    seal_path=PREP/'seal_v1.json';physical_path=PREP/'independent_sealed_transport_v1.json'
    stage_path=PREP/'stage_v1.json';seal=load(seal_path);physical=load(physical_path);stage=load(stage_path,8*1024**2)
    need(seal['complete']is True and physical['complete']is True and physical['unresolved_findings']==[],
         'Complete independent sealed transport review required')
    need(physical['archive_sha256']==seal['archive_sha256']and physical['protocol_sha256']==seal['protocol_sha256']and
         physical['archive_bytes']==seal['archive_bytes']and physical['seal_receipt_sha256']==sha(seal_path)and
         physical['stage_receipt_sha256']==sha(stage_path)==seal['stage_receipt_sha256']and
         physical['member_sha256']==seal['members_sha256']and physical['member_bytes']==seal['members_bytes']and
         physical['members']==seal['static_members']and physical['payload_bytes']==seal['payload_bytes'],
         'Seal/independent physical stage identity differs')
    need(all(physical[key]is True for key in ('single_gzip_frame','gzip_header_filename_mtime_level_exact',
         'manual512_USTAR_headers_order_payload_hash_padding_exact','double_EOA_minimalrecordpadding_fullEOF',
         'all_current_stage_bytes_equal','all_current_origins_equal'))and
         physical['packet_code_executed']is False and physical['actual_local_NN_autograd_optimizer_VM_calls']==0,
         'Independent physical checks or no-execution evidence incomplete')
    archive=safe(Path(seal['archive']),True)
    need(archive==(ROOT/'outputs'/(PREFIX+'-execution.tar.gz'))and
         archive.stat().st_size==seal['archive_bytes']and sha(archive)==seal['archive_sha256'],
         'Exact new R2 archive required')
    checksum=safe(Path(str(archive)+'.sha256'),True)
    need(checksum.read_bytes()==(seal['archive_sha256']+'  '+archive.name+'\n').encode('ascii'),
         'Exact ASCII basename/LF checksum required')
    protocol=seal['protocol_sha256'];folder=safe(ROOT/'outputs'/VM)
    need(sha(folder/'protocol.json')==protocol==stage['protocol_sha256'],'Exact R2 stage protocol required')
    p=load(folder/'protocol.json')
    need(p['vm_name']==VM and p['parent_dir']==PARENT and p['outputs']==OUTPUT and
         p['export_prefix']==PREFIX,'Exact R2 namespaces required')
    numerical=p['initializer_parity']
    need(numerical['numerical_initializer_gate_changed']is True and
         numerical['restoration_quality_gates_changed']is False and
         numerical['exact_arrays']==['native','raw','latent','tokens']and
         numerical['logits_atol']==1e-5 and numerical['logits_rtol']==1.3e-6 and
         numerical['token_argmax_candidate_parent_saved_all_exact']is True,
         'Explicit logits-only R2 numerical policy must match the published guide')
    forward_cap=sum(value for key,value in p['call_caps'].items()if key.endswith('_forwards'))
    reservation_cap=sum(p['call_caps'].values())
    need(forward_cap==1399 and reservation_cap==1854 and p['call_caps']['optimizer_step_attempts']==50 and
         p['call_caps']['backward_calls']==200 and p['call_caps']['gradient_queries']==5,
         'R2 finite operation census differs')
    dependency_counts={key:len(p[key])for key in
        ('dependencies','retained_output_dependencies','extra_returned_dependencies')}
    need(sum(dependency_counts.values())==3362 and len(p['selected_train_reference_ids'])==100,
         'Selected data/dependency census differs')
    imported=load(REVIEW/'return_import_v1.json');transport=load(REVIEW/'transport_v1.json');hq=load(REVIEW/'hq_v1/audit.json')
    need(imported['complete']is True and imported['archive_sha256']==OLD_ARCHIVE and imported['protocol_sha256']==OLD_PROTOCOL and
         imported['regular_files']==467 and imported['all32static_files_exact']is True and
         imported['new_restorer_updates']==0 and imported['new_checkpoints']==0 and imported['failure_present']is True and
         transport['transport_pass']is True and hq['pass']is True and hq['selected_TRAIN_references']==100 and
         hq['exact_parent_file_bytes']==200 and hq['canonical_array_payload_hashes']==200,
         'Actual preserved R1 failure/HQ audit must precede R2 publication')
    cause=load(REVIEW/'cause_v1/audit.json');report_receipt=load(REVIEW/'report_readback_v1.json')
    need(cause['complete']is True and cause['recorded_worker_difference_metrics_independently_reproduced']is True and
         cause['archive_sha256']==OLD_ARCHIVE and report_receipt['complete']is True and
         report_receipt['report_sha256']==sha(ROOT/RETURN_REPORT) and
         report_receipt['new_training_updates']==report_receipt['new_checkpoints']==0,
         'Actual R1 saved-array cause/report required before R2 publication')
    safe(ROOT/RETURN_REPORT,True)
    backups=PREP/'publication_v1'
    need(not backups.exists()and all(not(ROOT/name).exists()for name in(GUIDE,PREPARATION)),
         'Preserve previous/partial R2 guides and publication')
    originals={name:safe(ROOT/name,True).read_bytes()for name in CURRENT_DOCS+HISTORICAL_DOCS}
    guide=f'''# CodeFormer restoration training R2: five manual steps

**VERIFIED TRANSFER PREPARED / NOT RUN — 18 October 2026.**
[R2 preparation and limits]({PREPARATION}). [Latest R1 return review]({RETURN_REPORT}).

Both V1 and R1 stopped before Adam: **zero new updates and no new checkpoint**.
R1 saved exactly equal reconstructed pixels and dictionary selections; only
intermediate logits differed by at most 0.00000762939453125. Their actual files
and failed source packages remain intact. The low-level cause remains unknown.
R2 explicitly changes the initializer's float32 logit numerical test to the
declared PyTorch 2.9 tolerances (atol0.00001/rtol0.0000013), with exact argmax/token
and margin-stability checks. Other arrays/state/RNG and restoration-quality gates
remain exact/unchanged. CPU checks do not prove this prerequisite passes on L4.

This is an actual **restoration-network training pilot**, with at most **50 new
Adam updates / 200 microbatches**. It starts from the retained full-coarse
CodeFormer research checkpoint, uses fresh Adam, and trains no brightness head.
Untouched pretrained CodeFormer remains the benchmark. This is our fine-tuned
adaptation of CodeFormer; restoration improvement remains unproven.

Require **16 GiB free after installation**, retaining a **4 GiB reserve**. Parent
`~/forensic-dgp/{PARENT}` and the existing venv must remain available. Preflight
verifies **3,362 existing file bindings**. No automatic cleanup, retry or resume.
Preparation did not contact the VM or measure current free space.

Upload: **{seal['archive_bytes']:,} bytes**. SHA-256:
`{seal['archive_sha256']}`. Protocol: `{protocol}`.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{PREFIX}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{PREFIX}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {PREFIX}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/{PARENT} &&
test ! -e ~/forensic-dgp/{VM} &&
tar -xzf {PREFIX}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s codeformer_role_preserving_v1_r2
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{VM} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
TASK_ROOT="$(pwd -P)" &&
PYTHON_VM="$(python -c 'import sys; print(sys.executable)')" &&
"$PYTHON_VM" -B -u "$TASK_ROOT/scripts/supervise_codeformer_role_preserving_training_v1_r2_vm.py" --root "$TASK_ROOT" --protocol-sha {protocol}
```

The supervisor runs preflight, training and export once. Estimated duration:
**15–35 minutes**, unmeasured for this repaired recipe. Enforced limits: worker
**3,000 seconds**, initialization **600 seconds**, gradient witness **180 seconds**,
optimization **900 seconds**, and each 406-case snapshot **600 seconds**; phases
share the worker cap. Preflight/export caps are **300/600 seconds**, with **30
seconds** termination grace. Maximum **1,399 neural forwards / 1,854 reserved
operations**, including **five gradient queries / 200 backward calls / 50 updates**.
Detach with **Ctrl+B, then D**. A competing workload stops this launch untouched.

5. Download the **four return files** from **Windows Google Cloud SDK Shell** after the supervisor finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-export.json" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-supervisor-final.json" "."
```

One remote source per SCP command avoids the Windows PuTTY multiple-source error.
The checksum is ASCII/LF. Export completion means packaging finished and can
include a failed run. Preserve errors, partial outputs and checkpoints; do not
erase or relaunch an existing folder unchanged. Failed preflight prevents training.

After 50 updates and the final comparison bank finish, the genuine checkpoint is:
`~/forensic-dgp/{VM}/{OUTPUT}/snapshot_0050/checkpoint.pt`.
An earlier training failure retains its actual stopped/partial evidence. An
initializer failure before Adam can correctly produce **no new checkpoint**.
A final-state alias can reference an existing verified checkpoint without copying it.

Optional monitoring from another **VM SSH** terminal:

```bash
tail -F ~/forensic-dgp/{VM}/outputs/supervision/diagnostic.log ~/forensic-dgp/{VM}/outputs/supervision/export.log
```

Returned optimizer/state/RNG/sampler/counters, every RAW/PNG and planned visual
sheet require independent audit. Existing normal preservation and dark-retention
gates remain unchanged. User image-quality acceptance, independent FINAL review
and the existing app Auto/override/PNG flow remain pending. No app promotion.
'''
    preparation=f'''# CodeFormer restoration training R2 preparation — 18 October 2026

**Sealed and independently verified source transfer; manual L4 run pending.**
[Five upload/tmux/download steps]({GUIDE}). [Latest R1 return review]({RETURN_REPORT}).

Both original V1 and R1 stopped before Adam construction: **zero new restorer
updates, zero gradients/backwards and no new checkpoint**. V1 mismatched
evaluation/training comparison contexts. R1 corrected the contexts and exported
467 regular files / 46,082,287 payload bytes, preserving both actual initializer
captures. Independent auditing found native512, RAW256, encoder latent and all
256 selected tokens exactly equal. Only logits differ: 223,123/262,144 values,
maximum absolute difference 0.00000762939453125. All 200 fresh HQ target arrays
match their historical parent bytes and canonical payload hashes. Neither run
reached the restoration objective or a development quality bank. Original V1/R1
sources, returned arrays/contexts/RNG and failures remain preserved.

R2 changes only the prospective initializer **numerical test policy**. Float32
logits must satisfy declared PyTorch 2.9 defaults, atol0.00001 and rtol0.0000013;
all 256 argmax indices and saved tokens must be identical, and every parent
top-two margin must exceed twice the corresponding maximum logit error.
Native/RAW/latent/token arrays, model state, contexts and RNG stay exact.
The initializer gate therefore changes explicitly; restoration-quality gates
do not change. This does not establish the unknown low-level numerical cause.
[PyTorch 2.9 assert_close](https://docs.pytorch.org/docs/2.9/testing.html#torch.testing.assert_close).

The candidate/parent comparison retains the same input, eval/no-grad context,
RNG and helper. Training-mode unscoped/scoped route checks remain separate;
their existing exact loss/array requirements are retained. Compared arrays,
actual contexts, RNG and differences are saved before rejection. No additional
neural forward, standalone diagnostic, rate search or automatic retry is added.
The source/CPU-tested policy and actual neural gradient behavior remain
unverified until the user's manual L4 training run.

The pilot trains CodeFormer's actual restoration network: 515 parameter tensors,
350 trainable and 165 frozen. It starts from the retained full-coarse research
checkpoint with fresh Adam; it does not append to either failed run's zero updates or
load historical optimizer/RNG/sampler state. Normal slots use untouched-model full
RGB L1/token KL behavior at weights 2/0.1. Dark slots retain clean token CE,
latent, full-coarse, eligible pixel/LPIPS terms at 0.5/1/4/1/1, plus frozen-parent
RGB/token retention at 2/0.1. Roles select TRAIN losses; inference uses only the
face crop. There is no tone head, normal brightness-centering, runtime brightness
switch or weight search. These choices are prospective, not a promise of improvement.

Fresh Adam retains base rates 0.00001 for encoder/token parameters and 0.00002
for fusion, with a new 50-update linear warmup. Maximum 50 committed updates,
200 microbatches/backward calls, five gradient queries, 1,399 neural forwards
and 1,854 reservation events. Evaluation parity and real normal/dark gradient
connectivity must pass before Adam construction. Initial/final or stopped
checkpoints, optimizer/RNG/sampler/counters and actual training records are retained.

Unchanged acceptance gates compare all 61 clear and 53 normal-blur development
cases with untouched CodeFormer in RAW and delivered PNG, at mean/worst
regression limits 0.01/0.03. Initial parent failures stay recorded. At update50,
both 53-case dark validation cohorts must retain or improve parent mean PNG
MAE and SSIM. All 406 DEV outputs require independent review. Native CCTV is
unpaired and receives no clean-reference score. No quality threshold is relaxed.

CPU preparation closes 3,362 immutable physical file bindings: 1,434 selected
sources/assets/data, 1,624 retained RAW/PNG outputs and 304 HQ/runtime/metric
witnesses. Data scope remains 200 training observations from 100 selected TRAIN
references, 100 direct clean512/HQ target sets and 406 DEV inputs/targets/support
masks. Historical metadata retains the 391 TRAIN/53 validation split. Source and
photograph identifiers do not prove independent human identities. FINAL pixels
remain unused. Source-bound CPU and independent stage/transport reviews are
recorded in the preparation folder; no local neural training or VM run occurred.

Upload **{seal['archive_bytes']:,} bytes**, SHA-256 `{seal['archive_sha256']}`.
Protocol `{protocol}`. All **{seal['static_members']} staged regular members**
passed independent physical single-frame gzip/USTAR readback and source binding.
Require 16 GiB free after installation and retain a 4 GiB reserve. Worker cap:
3,000 seconds; initialization600, witness180, optimization900 and each snapshot600.
Preflight300/export600, plus30 termination grace. Source-derived storage bounds
remain prospective; measured VM timing/resources and actual state are audited
after return. Manual launch remains the user's next action.

The full goal remains useful 256x256 restoration, visible-appearance preservation,
insufficient-input refusal, user acceptance, independent FINAL review and the
existing app Auto/override/PNG/Playwright flow. Credit CodeFormer accurately;
preserve untouched CodeFormer, original DGP, trained states, caches, splits,
provenance, terms and failures. Keep paired synthetic and source-separated native
unpaired evidence distinct; no ethnicity or Zamboanga performance claim. Separate
covering detection/mask review/seven-family plausible completion remains deferred
until restoration acceptance. Storage offload is complete; this preparation
does not complete model qualification or the research goal.
'''
    prefix=f'''# CodeFormer restoration R2 prepared — 18 October 2026

**CURRENT: VERIFIED REPAIRED TRAINING TRANSFER / MANUAL L4 LAUNCH PENDING / NOT QUALIFIED.**
[Five R2 upload/tmux/download steps]({GUIDE}). [R2 preparation and limits]({PREPARATION}).
[Actual R1 failure and saved-array review]({RETURN_REPORT}).

Both failed V1/R1 runs stopped before Adam: zero new updates/gradients/checkpoints.
R1 saved exact native/RAW/latent/tokens but slightly different float32 logits.
Its 467-file return and all 200 selected HQ arrays passed audit. The exact
low-level cause remains unknown; there is no new restoration-quality result.
R2 explicitly changes the initializer logit test to atol0.00001/rtol0.0000013,
requiring identical argmax/tokens and top-two margins greater than twice observed
error. Other arrays/state/RNG remain exact; restoration-quality gates remain
unchanged. Actual R2 neural behavior remains unrun. Original sources/failures
and all trained model states remain preserved.

R2 will train the actual CodeFormer restoration network from the retained
full-coarse parent, using fresh Adam and the same role-preserving objective.
Maximum 50 new updates / 200 microbatches / five gradient queries / 1,399 forwards;
1,854 reservation events. Both normal slots preserve untouched RGB/token
behavior; dark slots retain clean supervision and frozen-parent behavior.
There is no tone head or historical optimizer continuation.

Upload {seal['archive_bytes']:,} bytes; SHA-256 `{seal['archive_sha256']}`;
protocol `{protocol}`. All {seal['static_members']} staged source members and
3,362 physical parent bindings are bound to the independent preparation reviews.
Require 16 GiB free after installation and retain 4 GiB reserve. No local neural
work, VM connection or automatic launch occurred. Only manual L4/tmux training.

The full restoration-first goal stays active: useful256 output, visible structure
and appearance, clearer-input refusal, user acceptance, independent FINAL and
existing-app Auto/override/PNG/Playwright verification. Retain original
CodeFormer/DGP, checkpoints/caches/splits/provenance/terms and failures. Native
CCTV remains unpaired and source-separated; no ethnicity/Zamboanga/hidden-identity
claim follows. Covering detection and seven-family completion stay deferred
until restoration acceptance. Storage offload is complete.

Earlier status and commands below are historical. Use the new R2 guide above.

---

'''.encode('utf-8')
    historical=f'''# Historical R1 commands: stopped before training — 18 October 2026

**R1 FAILED INITIALIZER LOGIT PARITY: ZERO NEW UPDATES / NO NEW CHECKPOINT.**
This document is retained as historical evidence. Do not rerun the failed folder
unchanged. [Actual return review]({RETURN_REPORT}).
Use the separately prepared [R2 five-step guide]({GUIDE}); R2 is unrun and
keeps the restoration objective and quality requirements. Its initializer
numerical logit test changes explicitly; see the new guide before uploading.

---

'''.encode('utf-8')
    guides={GUIDE:guide,PREPARATION:preparation}
    # All admission checks precede any document write. Retain all six original
    # byte streams before prefixing a single existing document.
    for name,before in originals.items():new(backups/name,before)
    for name,value in guides.items():new(ROOT/name,value.encode('utf-8'))
    publications=[]
    for name,before in originals.items():
        path=safe(ROOT/name,True);need(path.read_bytes()==before,'Document changed after publication admission')
        addition=prefix if name in CURRENT_DOCS else historical
        path.write_bytes(addition+before)
        need(path.read_bytes()[len(addition):]==before,'Original document exact suffix lost')
        publications.append({'path':name,'old_sha256':hashlib.sha256(before).hexdigest(),
            'prefix_bytes':len(addition),'new_sha256':sha(path),'old_bytes_exact_suffix':True})
    result={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'seal_sha256':sha(seal_path),'physical_review_sha256':sha(physical_path),'stage_sha256':sha(stage_path),
        'archive_sha256':seal['archive_sha256'],'protocol_sha256':protocol,
        'guides_sha256':{name:sha(ROOT/name)for name in guides},'publications':publications,
        'actual_R1_return_review_sha256':sha(ROOT/RETURN_REPORT),
        'actual_R1_import_sha256':sha(REVIEW/'return_import_v1.json'),
        'actual_R1_transport_sha256':sha(REVIEW/'transport_v1.json'),
        'actual_R1_HQ_audit_sha256':sha(REVIEW/'hq_v1/audit.json'),
        'actual_R1_saved_array_cause_audit_sha256':sha(REVIEW/'cause_v1/audit.json'),
        'initializer_numerical_gate_changed':True,'restoration_quality_gates_changed':False,
        'publisher_sha256':sha(__file__),'manual_training_not_started':True,
        'model_created_by_publication':False,'model_qualified':False,'goal_complete':False}
    new(backups/'publication.json',(json.dumps(result,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'complete':True,'guides':list(guides),'prior_documents_preserved':6,
        'manual_training_not_started':True,'model_created':False}))


if __name__=='__main__':main()
