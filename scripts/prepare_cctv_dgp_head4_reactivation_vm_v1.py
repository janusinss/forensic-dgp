"""Prospective packet freeze. No VM connection, neural calls or training."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import shutil
import tarfile
import time
from cctv_dgp_head4_reactivation_contract_v1 import NAME,STEM,BUDGETS,PARTS,COMPONENTS,validate,read,write,sha

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
PARITY=ROOT/'outputs/cctv_dgp_head4_reactivation_parity_v1'
OUT=ROOT/'outputs'/NAME
PREP=ROOT/'outputs/cctv_dgp_head4_reactivation_v1_preparation'


def guide(pin,archive_sha,size):
    return f'''# Connected DGP branch check: exact manual commands

This packet is prepared for the existing NVIDIA L4/g2-standard-4 VM, not launched.
It compares original and repaired gradients, with **zero optimizer updates/epochs**.
Our deepest original branch is numerically collapsed. A separate fixed initializer
preserves all100 current CPU outputs exactly. VM gradient usefulness is unverified.
This experiment decides whether the connected path can learn before more epochs.

Expected **4–10 minutes** plus **1–3 minutes** export; worker600s/external630s,
export180s/external210s,30s kill grace. Require **3GiB free after installation**;
20GiB allocated VRAM,768MiB uncompressed return,512MiB disk reserve.
Only160 autograd queries are allowed; no fitting, solver, epoch, automatic follow-on
or app promotion. Every unchanged structure/preservation requirement remains.

Execution archive: {size:,} bytes; SHA256 `{archive_sha}`.
Protocol SHA256: `{pin}`.
Use only after the independent packet audit in outputs/cctv_dgp_head4_reactivation_v1_preparation/
passes. Source, data roles and original failures are preserved. Do not rerun this
packet over a partial or completed run.

If the existing VM is stopped, start it manually in Google Cloud. Equivalent
**Windows Google Cloud SDK Shell** command (starts billed VM runtime):

```bat
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

Stop if free space is below3GiB. No deletion is bundled. Maintenance requires
a fresh inventory and exact retained-backup hashes, preserving all research assets.

3. Open tmux:

```bash
tmux new-session -A -s dgp_head4_reactivation_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_head4_reactivation_vm_v1.py --root . --protocol-sha {pin} --verify-transfer &&
python -B -u scripts/supervise_cctv_dgp_head4_reactivation_v1.py --root . --protocol-sha {pin}
```

Detach: Ctrl+B, release, D. Reattach:
`tmux attach-session -t dgp_head4_reactivation_v1`.
The supervisor prints worker output and exports a retained failure when possible.
It does not refuse merely because you are inside the intended tmux session.
`connected_route_pass: true` only means the repaired pieces have finite nonzero
improvement gradients. It is not a structure/preservation/quality pass.
`complete: true` in the export JSON means archive completion only.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Return allthree for independent saved-vector/initial-output audit. There is no
autograd or optimizer replay locally. Keep original checkpoints, splits, caches
and gate failures. Additional epochs1/2/5 are a later finite study after the new
path's training capacity and native/paired development preservation justify it.
No final identities or completion families are qualified by this experiment.
The full DGP-led application goal remains active/incomplete.
'''


def main():
    start=time.monotonic();assert not OUT.exists() and not PREP.exists()
    parent=read(PARENT/'protocol.json');parity=read(PARITY/'results.json');audit=read(PARITY/'independent_audit.json')
    assert audit['complete'] and parity['initial_CPU_parity_cases']==100
    assert parity['maximum_original_CPU_candidate_error']==0 and parity['normal_scale_seed_response_cases']==100
    assert all(audit['phase3_dead_branch_value_equality'].values())
    assert read(ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json')['complete']
    for name,digest in parent['assets_sha256'].items():assert sha(PARENT/name)==digest,name
    for name,digest in parent['local_sources_sha256'].items():assert sha(ROOT/name)==digest,name
    OUT.mkdir();PREP.mkdir();mapping={}
    root_files={'cctv_dgp_pilot.py','dgp_face_restoration.py','dgp_frozen_inference_v2.py','cctv_dgp_frozen_norm.py',
        'frozen_capacity_contract.py','frozen_raw_metrics.py','frozen_definitions.py','cctv_dgp_group_conflicts_v1_losses.py'}
    def copy(origin,name):
        destination=OUT/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,destination)
        mapping[name]=Path(origin).relative_to(ROOT).as_posix()
    for name in parent['assets_sha256']:
        if name.startswith(('data/','models/','weights/')) or name in root_files:copy(PARENT/name,name)
    source_map={'cctv_dgp_head4_reactivation_v1.py':'cctv_dgp_head4_reactivation_v1.py',
        'cctv_dgp_head4_reactivation_contract_v1.py':'cctv_dgp_head4_reactivation_contract_v1.py',
        'cctv_dgp_head4_reactivation_vm_v1.py':'scripts/cctv_dgp_head4_reactivation_vm_v1.py',
        'supervise_cctv_dgp_head4_reactivation_v1.py':'scripts/supervise_cctv_dgp_head4_reactivation_v1.py'}
    for source,name in source_map.items():
        ast.parse((ROOT/'scripts'/source).read_text(),feature_version=(3,10));copy(ROOT/'scripts'/source,name)
    for folder,name in [(PARITY,'plan.json'),(PARITY,'results.json'),(PARITY,'independent_audit.json')]:
        copy(folder/name,'evidence/parity_'+name)
    copy(PARENT/'protocol.json','evidence/parent_protocol.json')
    for name in ['CCTV_DGP_HEAD4_REACTIVATION_V1_DESIGN.md','CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md']:
        copy(ROOT/name,'evidence/'+name)
    p={key:parent[key] for key in ['original_state','original_checkpoint_sha256','recognizer_state','cases','references','cohorts']}
    p.update({'format':'own-DGP-connected-head4-gradient-ablation-v1','UTC':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Restore numerical scale of the original deepest head and its fusion, while anchoring exact current output; measure real connected gradients before epochs.',
        'initial_repaired_state':parity['states_before']['candidate'],'initialization_only':True,
        'seed':'current retained head3 kernels and adjacent fusion slice; fixed before any new forward',
        'frozen_anchors':6,'parameter_parts':[{'name':name,'elements':n} for name,n in PARTS],
        'three_requires_grad_tensors':True,'full_fusion_requires_grad_elements_including_unselected_slices':147456,
        'selected_connected_vector_elements':147456,'components':COMPONENTS,'gradient_shape':[2,20,4,147456],
        'all_individual_gradient_vectors_saved':True,'optimizer_updates':0,'epochs':0,'budgets':BUDGETS,
        'route_gate':'Each repaired piece in each existing TRAIN cohort has finite nonzero improvement norm over MSE/ArcFace/degraded structure; no quality claim.',
        'original_and_repaired_gradient_inputs_identical':True,'cohort_exposure_history':'Both are previously exposed TRAIN, never independent evaluation',
        'initial_GPU_original_repaired_max_abs':0,'raw_CPU_replay_max_abs':3e-6,'embedding_CPU_replay_max_abs':5e-5,
        'independent_replay_case_ids':read(PARITY/'plan.json')['independent_replay_case_ids'],
        'unchanged_quality_requirements':parent['scientific_thresholds'],'quality_requirements_not_tested_by_gradient_success':True,
        'manual_tmux_required':True,'native_DEV_or_reserved_final_used':False,'app_promotion':False,
        'automatic_follow_on':False,'resume_permitted':False,'goal_complete':False,
        'source_protocol_sha256':sha(PARENT/'protocol.json'),'provenance_terms_overlap_from_parent_unchanged':True,'source_mapping':mapping})
    p['assets_sha256']={q.relative_to(OUT).as_posix():sha(q) for q in sorted(OUT.rglob('*')) if q.is_file()}
    extra=['scripts/prepare_cctv_dgp_head4_reactivation_vm_v1.py','scripts/audit_cctv_dgp_head4_reactivation_return_v1.py',
        'scripts/verify_cctv_dgp_head4_reactivation_packet_v1.py','CCTV_DGP_HEAD4_REACTIVATION_V1_DESIGN.md',
        'CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md','outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json',
        'outputs/cctv_dgp_original_head4_trace_v1/results.json','outputs/cctv_dgp_original_head4_weight_audit_v1/statistics.json',
        'outputs/cctv_dgp_head4_reactivation_parity_v1/plan.json','outputs/cctv_dgp_head4_reactivation_parity_v1/results.json',
        'outputs/cctv_dgp_head4_reactivation_parity_v1/independent_audit.json']+['scripts/'+name for name in source_map]
    for name in extra:
        if name.endswith('.py'):ast.parse((ROOT/name).read_text(),feature_version=(3,10))
    p['local_source_bindings']={**parent['local_sources_sha256'],**{name:sha(ROOT/name) for name in extra}}
    validate(p);write(OUT/'protocol.json',p);pin=sha(OUT/'protocol.json')
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for q in sorted(OUT.rglob('*')):
            if q.is_file():tar.add(q,arcname=NAME+'/'+q.relative_to(OUT).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    path=ROOT/'CCTV_DGP_HEAD4_REACTIVATION_V1_VM.md'
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(guide(pin,digest,archive.stat().st_size))
    write(PREP/'preparation.json',{'complete':True,'packet_sha256':digest,'packet_bytes':archive.stat().st_size,
        'protocol_sha256':pin,'assets':len(p['assets_sha256']),'old_sources_preserved':len(parent['local_sources_sha256']),
        'local_source_bindings':len(p['local_source_bindings']),'manual_guide_sha256':sha(path),
        'all_neural_or_gradient_calls':0,'optimizer_updates':0,'VM_launched':False,'independent_packet_audit_pending':True,
        'goal_complete':False,'seconds':time.monotonic()-start})
    print({'prepared':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size,
        'assets':len(p['assets_sha256']),'VM_launched':False},flush=True)


if __name__=='__main__':main()
