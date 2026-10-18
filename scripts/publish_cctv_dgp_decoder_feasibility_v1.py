"""Publish a verified unrun manual packet while preserving instruction history."""
from datetime import datetime,timezone
from pathlib import Path

from cctv_dgp_decoder_feasibility_v1_contract import NAME,STEM,CHECKPOINT,read,write,sha
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend,retained_new

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
PREP=OUT/'cctv_dgp_decoder_feasibility_v1_preparation'


def link(label,path):return f'[{label}](<{(ROOT/path).as_posix()}>)'


def main():
    stage=PREP/'publication';assert not (stage/'publication.json').exists()
    prepared=read(PREP/'preparation.json');audit=read(PREP/'independent_preparation_audit.json')
    tests=read(PREP/'cpu_tests.json');syntax=read(PREP/'bash_syntax.json')
    assert all(x['complete'] for x in [prepared,audit,tests,syntax])
    pin,digest=prepared['protocol_sha256'],prepared['archive_sha256']
    assert sha(OUT/NAME/'protocol.json')==pin and sha(OUT/(STEM+'-execution.tar.gz'))==digest
    assert sha(OUT/'cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth')==CHECKPOINT
    previous=read(OUT/'cctv_dgp_decoder_activation_v1_analysis/publication/publication.json')
    names=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
           'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_NEXT_PILOT.md','CCTV_DGP_POST_ACTIVATION_V1_REVIEW.md']
    expected={name:previous['documents'][name]['after_sha256'] if name in previous['documents']
              else previous['new_document_sha256'][name] for name in names}
    assert all(sha(ROOT/name)==value for name,value in expected.items()),'Preserve concurrent document edits'
    guide_name='CCTV_DGP_DECODER_FEASIBILITY_V1_VM.md'
    scope_name='CCTV_DGP_DECODER_FEASIBILITY_V1_PREPARATION.md'
    guide=f'''# Decoder feasibility V1: five manual steps

**VERIFIED / PREPARED / UNRUN.** This is one new direction search followed,
only if independently certified, by three fixed diagnostic placements. It
performs zero optimizer updates, zero new gradient queries and zero epochs.
Do not resume or rerun the failed historical training recipes.

Require **4 GiB free**, **8 GiB available RAM**, an idle existing L4 and all
retained parent directories. Estimated runtime: **10–25 minutes**, with finite
stops rather than a promise. Worker maximum:40minutes; saved audit:15minutes;
export:5minutes. External limits add30seconds per stage, with30second kill
grace. The solver has60seconds/10,000iterations, exact certificates600seconds,
cache240seconds and each snapshot300seconds. Output1280MiB, disk reserve1GiB,
VRAM20GiB and process RSS12GiB are capped. Model loading counts toward limits.

1. Upload in **Windows Google Cloud SDK Shell (cmd.exe)**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_decoder_feasibility_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_decoder_feasibility_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_feasibility.sh {pin}
```

5. Download in **Windows Google Cloud SDK Shell**, after export:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each download uses one remote source for Windows PuTTY compatibility. Detach
with `Ctrl+B`, then `D`. Keep all failure/partial archives. Export completion
means evidence was packaged; it does not mean a direction or model qualified.

Expected terminal paths: `no_certified_direction` retains the search/certificate
evidence with zero neural calls, or `three_fixed_placements_complete` retains
400 output cases, all RAW/PNG gates and exact model/RNG restoration. A technical
stop may leave partial evidence; do not rerun the immutable directory.

Archive:{prepared['archive_bytes']:,}bytes; SHA256:`{digest}`.
Protocol SHA256:`{pin}`.
{link('Scope, test evidence and limits',scope_name)}.
{link('Local returned-evidence importer','scripts/import_cctv_dgp_decoder_feasibility_v1_return.py')}.
No VM connection, solve on research gradients, gradient or optimizer execution
was performed by the agent during preparation.
'''
    scope=f'''# Decoder feasibility V1: verified preparation, no model qualification

The preceding activation audit established a concrete preservation conflict:
the retained direction improves sampled landmark terms while worsening
low-light/compound RGB terms. Tiny finite regression allowances activate before
the smallest tested displacement. The mean actual objective rises from5.200000
to5.218366 at1/16; all nonzero RAW/PNG gates remain failed. This distinct
experiment asks whether another direction in the exposed decoder span avoids
those conflicts. It does not add epochs or resume that failed direction.

The search reuses the 70 saved underlying gradients from two fixed TRAIN
references and the original 15-tensor/609,219-parameter A decoder. One bounded
HiGHS dual-simplex call maximizes the worst normalized degraded structure slope
over a homogeneous ray. All30 per-profile preservation slopes must be
nonpositive; all8 degraded structure terms must improve. Exact sign-dependent
clipped-mean coefficients supply64 brightness-branch inequalities. The fixed
1e-6 interior margin is a stricter search condition, not a relaxed quality gate.
The coefficient box normalizes a ray; radial scaling and exact certificates
retain the unchanged original-relative L2 radius0.04018470287178282.

This is a **strict sufficient subset**, not a solve of all original heterogeneous
finite-slack constraints. No certified ray is a valid stopping result, not a
proof of whole-span, decoder or restoration impossibility. Default solver
tolerances cannot admit a proposal. Independent exact dyadic endpoint products
and exact rational prepared-pixel clip coefficients check the actual rounded
float32 quarter, half and full placements before any learned forward. Full
placement must predict at least1% improvement in each of the8 sampled structure
terms; partial placements must predict positive improvement. These are selected
branch/idealized clip certificates, not certified neural or PNG behavior.

A proposed direction and all constructed vectors are retained before exact
certification, including a rejected/partial proposal. An uncertified direction
receives no learned inference. One admitted direction receives exactly the
three prospectively fixed placements in one finite comparison. Each is tested
on all20references/100cases with separate RAW/floor-PNG metrics, all17 groups,
the original global/each-source1% structure requirement, RGB/SSIM/recognizer
preservation, brightness fraction≤0.2 and exact exterior pixels. The other18
references were not used for gradients and are essential finite checks. The
quality gates do not authorize automatic checkpoint selection or follow-on.

The run retains the initial controls, per-placement full model states, original
before/after state and RNG, solver/certificate/code/environment/count/timing
records and all failures. Every planned whole-face visual comparison must be
generated, bound and inspected after return; numeric certificates alone do not
establish useful detail. TRAIN and DEV evidence may justify finite continuation;
freeze the candidate and processing before independent final review. Reserved
final results never select further epochs.

The local preparation passed **{tests['tests']} CPU tests** and independent
packet/source/parent review. Tests use small synthetic solver/array fixtures,
opaque transfer/export data and strict corruption/guard/schema checks. No
actual research-gradient solve, local new parameter proposal, learned forward,
gradient query, optimizer update or VM connection occurred. CPU syntax and
fixture success do not certify CUDA behavior or useful restoration.

RouteA remains the corrected DeblurGAN-v2-compatible conditional restorer.
Frozen external GAN features are cached for source parity and are not injected
into this A path. It is not a newly trained GAN or qualified generative face
prior. The justified replacement-design comparison remains part of the goal.
Permanent checkpoint:`{CHECKPOINT}`.
Native CCTV/DEV, reserved final identities and covering pixels are absent from
this experiment; dataset labels do not establish ethnicity or Zamboanga quality.

{link('Independent packet audit','outputs/cctv_dgp_decoder_feasibility_v1_preparation/independent_preparation_audit.json')}.
{link('CPU regression evidence','outputs/cctv_dgp_decoder_feasibility_v1_preparation/cpu_tests.json')}.
{link('Exact manual steps',guide_name)}.
{link('Causal design decision','CCTV_DGP_POST_ACTIVATION_V1_REVIEW.md')}.
All five milestones and all seven separately reported automatic/assisted
covering families remain required and unqualified. App model/design stay fixed.
'''
    for name,text in [(guide_name,guide),(scope_name,scope)]:retained_new(ROOT/name,text)
    prefix=f'''# New decoder feasibility comparison verified, UNRUN — 15 October 2026

**One new manual direction experiment is VERIFIED / PREPARED / UNRUN.** It
reuses the audited activation gradients; zero new gradients, optimizer updates
or epochs are planned. One homogeneous ray search must earn exact certificates
for all three fixed rounded placements before any learned forward. The search
is a stricter sufficient subset, not a test of all decoder directions.

{link('Five upload/install/tmux/run/download steps',guide_name)}.
{link('Scientific scope, test evidence and finite limits',scope_name)}.

The original checkpoint and previous failed gates remain unchanged. No local
research-gradient solve or VM execution occurred; no model or incumbent is
qualified. TRAIN/DEV selection must precede candidate/processing freeze and
independent final review. All five milestones and seven automatic/assisted
covering families remain active. Earlier USED/PREPARED status below is retained
history; only this named new packet is the current manual experiment.

'''
    (stage/'before').mkdir(parents=True,exist_ok=True)
    docs={name:finish_prepend(ROOT/name,prefix,stage/'before'/name,value) for name,value in expected.items()}
    evidence=[PREP/name for name in ['preparation.json','independent_preparation_audit.json','cpu_tests.json','bash_syntax.json']]
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),documents=docs,
        new_document_sha256={name:sha(ROOT/name) for name in [guide_name,scope_name]},
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in evidence},
        previous_goal_turn_classification='progress: independently closed activation return audit and isolated finite preservation conflict',
        current_goal_turn_classification='progress: prepared and independently verified one distinct finite manual feasibility experiment',
        protocol_sha256=pin,archive_sha256=digest,archive_bytes=prepared['archive_bytes'],
        new_local_solver_calls_on_research_gradients=0,new_local_parameter_proposals=0,new_local_gradient_queries=0,
        new_local_optimizer_updates=0,agent_VM_calls=0,model_qualified=False,goal_complete=False,status='PREPARED_UNRUN'))
    print(dict(complete=True,new_documents=2,preserved_documents=6,status='PREPARED_UNRUN',goal_complete=False))


if __name__=='__main__':main()
