"""Prepare a distinct finite manual PCGrad treatment after audited V40 diagnostics."""
from datetime import datetime, timezone
from pathlib import Path
import ast
import shutil
import tarfile
import time
from cctv_dgp_spatial_fit_v40_contract import read, write, sha, validate_schedule
from render_cctv_dgp_pcgrad_v41 import NAME, STEM, FORMAT, worker, decoder, auditor
from cctv_dgp_pcgrad_v41 import task_orders

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_preparation'
ANALYSIS = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis'


def text(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as output: output.write(value)


def guide(pin, digest, size):
    return f'''# V41 spatial DGP PCGrad pilot: five manual steps

The audited V40 endpoint diagnostic finds conflicting improvement/preservation
gradients. V41 tests one learning change: PCGrad combines the same seven weighted
component gradients before the same AdamW update. No loss weights or preservation
thresholds change. Fixed random order seed20261008 is declared before execution.
The same original spatial decoder seed,781 TRAIN references/3,905 cases and
paired800-batch schedule remain. Original DGP, initial reference and recognizer
stay frozen. No pretrained restoration targets or primary substitution are used.

Maximum800 updates (one epoch plus19 batches), with the unchanged1% structure
requirement at50 and10% at800. All17 preservation groups, both-source nonregression
and20% mean-only fraction remain. Failure at50 stops and exports every failed gate;
there is no resume, rerun unchanged, sweep, automatic launch or app promotion.
Each update retains seven raw gradient arrays, projected/clipped gradients,
before/after parameters, AdamW moments, scalar losses and input IDs for independent
readback. Gradients and actual training occur only on the existing L4 VM.

Require the existing **NVIDIA L4/g2-standard-4**, existing venv and **6GiB free
after installation**. Packet:{size:,} bytes. Estimate **20–50 minutes** training
and **3–15 minutes** export; these are V41 estimates, not measured results.
Cache900s, fit3600s, worker4500s/external4800s+30s grace, export900s/external930s+30s,
allocated VRAM20GiB and uncompressed return3GiB are enforced. Update20 timing
projection stops before exceeding the fit budget. The pilot performs no cleanup.

Protocol SHA256: `{pin}`
Execution archive SHA256: `{digest}`

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
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_pcgrad_fit_v41
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_pcgrad_fit_v41_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v41.sh {pin}
```

Detach safely with **Ctrl+B**, release, then **D**. Keep the VM running while
training/export is active. Trainer exit1 means retain the stop; complete:true
on the export receipt means packaging completed, not restoration qualification.

5. Download in **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

The prospective local auditor uses `venv\\Scripts\\python.exe`, frozen inference
and saved-array arithmetic only. Return arrays are never independently
differentiated locally. Parameter arithmetic tolerance3e-7 is for float32
CPU/CUDA reduction readback. Combined-gradient arithmetic allows two float32
ULPs plus1e-12 for reduction rounding; parameter chains and snapshot tensors
stay exact. All image/gate thresholds remain unchanged.
Useful native output, covering-family quality and independent final review
remain required. No real Zamboanga samples exist; source labels do not establish
ethnicity. Reserved final pixels remain unopened. Goal active/incomplete.
'''


def main():
    start=time.monotonic(); assert not OUT.exists() and not PREP.exists()
    old=read(BASE/'protocol.json'); diagnostic=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json')
    analysis=read(ANALYSIS/'analysis.json'); checked=read(ANALYSIS/'independent_analysis_audit.json'); visual=read(ANALYSIS/'visual_review.json')
    assert diagnostic['complete'] and diagnostic['diagnostic_complete'] and diagnostic['gradient_readback']['component_queries_readback']==280
    assert checked['complete'] and checked['analysis_sha256']==sha(ANALYSIS/'analysis.json') and checked['saved_batches']==40
    assert analysis['all40_saved_batches_first_order_nonincrease'] and not visual['convincing_incremental_structure_gain']
    assert visual['all50_optimized_TRAIN_cases_and200_cells_actually_viewed'] and not visual['app_promotion']
    for name,digest in old['assets_sha256'].items(): assert sha(BASE/name)==digest,name
    validate_schedule(old['cases'],read(BASE/'schedule.json')['batches'])
    OUT.mkdir(); PREP.mkdir(); mapping={}
    def copy(path,name):
        assert time.monotonic()-start < 300
        dest=OUT/name; assert dest.resolve().is_relative_to(OUT); dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest); assert sha(dest)==sha(path); mapping[name]=path.relative_to(ROOT).as_posix()
    skip={'scripts/cctv_dgp_spatial_fit_v40_vm.py','scripts/run_v40.sh','cctv_dgp_spatial_decoder_v40.py'}
    for name in old['assets_sha256']:
        if name not in skip: copy(BASE/name,name)
    rendered_worker=worker((ROOT/'scripts/cctv_dgp_spatial_fit_v40_vm.py').read_text())
    rendered_auditor=auditor((ROOT/'scripts/audit_cctv_dgp_spatial_fit_v40_return.py').read_text())
    local_worker=ROOT/'scripts/cctv_dgp_pcgrad_fit_v41_vm.py'; local_auditor=ROOT/'scripts/audit_cctv_dgp_pcgrad_fit_v41_return.py'
    text(local_worker,rendered_worker); text(local_auditor,rendered_auditor)
    copy(local_worker,'scripts/cctv_dgp_pcgrad_fit_v41_vm.py')
    for name in ['cctv_dgp_pcgrad_v41.py','cctv_dgp_pcgrad_v41_evidence.py']: copy(ROOT/'scripts'/name,name)
    text(OUT/'cctv_dgp_spatial_decoder_v41.py',decoder((BASE/'cctv_dgp_spatial_decoder_v40.py').read_text()))
    shell=(BASE/'scripts/run_v40.sh').read_text().replace('cctv_dgp_spatial_fit_vm_v40',NAME).replace('cctv_dgp_spatial_fit_v40_vm.py','cctv_dgp_pcgrad_fit_v41_vm.py').replace('V40','V41')
    text(OUT/'scripts/run_v41.sh',shell)
    assets={path.relative_to(OUT).as_posix():sha(path) for path in OUT.rglob('*') if path.is_file()}
    local_names=['scripts/prepare_cctv_dgp_pcgrad_fit_v41.py','scripts/render_cctv_dgp_pcgrad_v41.py','scripts/cctv_dgp_pcgrad_v41.py',
        'scripts/cctv_dgp_pcgrad_v41_evidence.py','scripts/verify_cctv_dgp_pcgrad_fit_v41_packet.py',local_worker.relative_to(ROOT).as_posix(),
        local_auditor.relative_to(ROOT).as_posix(),'scripts/cctv_dgp_spatial_fit_v40_vm.py','scripts/audit_cctv_dgp_spatial_fit_v40_return.py',
        'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json','outputs/cctv_dgp_spatial_fit_v40_independent_audit.json',
        'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json','CCTV_DGP_V40_LEARNING_SIGNAL_V1_RESULTS.md',
        'outputs/cctv_dgp_v40_learning_signal_v1_analysis/analysis.json','outputs/cctv_dgp_v40_learning_signal_v1_analysis/independent_analysis_audit.json',
        'outputs/cctv_dgp_v40_learning_signal_v1_analysis/visual_review.json']
    p=dict(old); p.update({'format':FORMAT,'UTC':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Removing measured component-gradient conflict in the same spatial DGP may enable useful finite structure learning while retaining every delivered-image gate. Endpoint tangents are not a guarantee.',
        'gradient_combination':{'method':'PCGrad; each component projected against unmodified other gradients in a seeded order; zero norm skipped',
            'seed':20261008,'ordering':'NumPy default_rng(seed+update-1), one permutation of seven per component, self excluded',
            'projection_precision':'float64 arithmetic, merged gradient cast to float32 before retained global clipping',
            'loss_weights_changed':False,'queries_per_update':7,'queries_bound':5600,'local_gradient_calls':0},
        'gradient_orders':[task_orders(i) for i in range(1,801)],
        'step_evidence':{'all_updates_retained':True,'fields':['components','combined','applied','before','after','exp_avg','exp_avg_sq'],
            'prospective_parameter_arithmetic_tolerance':3e-7,'clip_and_moment_relative_tolerance':3e-6,
            'combined_gradient_reduction_tolerance':'Two float32 ULPs of independent expected result plus1e-12; exact parameter chain and snapshots',
            'saved_gradients_not_independently_differentiated':True},
        'research_sources':['https://arxiv.org/abs/2001.06782','https://github.com/tianheyu927/PCGrad'],
        'V40_diagnostic_audit_sha256':sha(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json'),
        'sources_sha256':{name:sha(ROOT/name) for name in set(local_names)|set(mapping.values())},
        'copied_source_mapping':mapping,'assets_sha256':assets,'prepared_only':True})
    assert p['cases']==old['cases'] and p['optimizer']==old['optimizer'] and p['retained_capacity_gates']==old['retained_capacity_gates']
    for path in OUT.rglob('*.py'): ast.parse(path.read_text(),feature_version=(3,10))
    write(OUT/'protocol.json',p); pin=sha(OUT/'protocol.json'); archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for path in sorted(OUT.rglob('*')):
            if path.is_file(): tar.add(path,arcname=NAME+'/'+path.relative_to(OUT).as_posix(),recursive=False)
    digest=sha(archive); text(Path(str(archive)+'.sha256'),digest+'  '+archive.name+'\n')
    assert time.monotonic()-start < 300
    text(ROOT/'CCTV_DGP_PCGRAD_FIT_V41_VM.md',guide(pin,digest,archive.stat().st_size))
    write(PREP/'prepared.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
        'packet_files':len(assets)+1,'sources_verified':len(p['sources_sha256']),'updates_bound':800,'gradient_queries_bound':5600,
        'local_neural_calls':0,'local_gradient_calls':0,'local_optimizer_updates':0,'VM_calls':0,'prepared_only':True,
        'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'prepared':True,'files':len(assets)+1,'bytes':archive.stat().st_size,'protocol_sha256':pin,'archive_sha256':digest},flush=True)


if __name__=='__main__':main()
