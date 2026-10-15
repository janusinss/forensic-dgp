"""Publish a verified manual packet; publishing does not launch computation."""
from datetime import datetime, timezone
from pathlib import Path

from cctv_dgp_decoder_hinge_repair_v1_contract import NAME, STEM, read, write, sha, CHECKPOINT
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new

ROOT=Path(__file__).resolve().parent.parent
DESIGN=ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_design'


def link(label,name):return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def main():
    prep=read(DESIGN/'preparation.json');audit=read(DESIGN/'preparation_audit.json')
    assert prep['complete'] and audit['complete'] and audit['status']=='VERIFIED_PREPARED_UNRUN'
    importer=read(DESIGN/'importer_preparation.json')
    assert importer['complete'] and importer['tests_run']==12
    assert importer['importer_sha256']==sha(ROOT/'scripts/import_cctv_dgp_decoder_hinge_repair_v1_return.py')
    pin=prep['protocol_sha256'];packet=ROOT/'outputs'/NAME
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert sha(packet/'protocol.json')==pin==audit['protocol_sha256']
    assert sha(archive)==prep['archive_sha256']==audit['archive_sha256']
    assert not (packet/'outputs').exists() and not (packet/'run.log').exists()
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth')==CHECKPOINT
    publication=DESIGN/'publication'
    assert not (publication/'publication.json').exists()
    publication.mkdir(exist_ok=True);(publication/'before').mkdir(exist_ok=True)
    guide_name='CCTV_DGP_DECODER_HINGE_REPAIR_V1_VM.md'
    report_name='CCTV_DGP_DECODER_HINGE_REPAIR_V1_PREPARATION.md'
    return_name='CCTV_DGP_DECODER_HINGE_REPAIR_V1_LOCAL_RETURN.md'
    guide=f'''# Decoder brightness-hinge correction — manual L4 commands

**VERIFIED / PREPARED / UNRUN.** This fresh packet corrects the demonstrated
brightness-penalty import regression. It keeps Adam at0.0001 so the intended
loss change is isolated. Actual correction effectiveness and model improvement
remain unverified. The original app checkpoint stays unchanged.
{link('Completed replay and discovered cause', 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_RESULTS.md')}.

Require **8 GiB free after installation** and **8 GiB available RAM** before
models. Reuse the existing three research roots: spatial_readout_v1,
bank_photometric_guard_v1 and spatial_path_replay_v1. Do not remove their assets.
Upload only **{prep['archive_bytes']:,} bytes**. Estimate10–30minutes including
saved audit/export; the new CUDA runtime is unmeasured. Enforced maxima including
kill grace total63minutes plus hash-only verification: worker2400/external2430s,
saved audit600/external630s, export600/external630s,30s kill grace per stage.
VRAM20GiB/RSS12GiB/output3GiB/free reserve1GiB; every limit is enforced.

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
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_spatial_readout_v1_vm &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test -d ~/forensic-dgp/cctv_dgp_spatial_path_replay_v1_vm &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

Expected checksum:`OK`. The sidecar is verified LF UTF-8 without BOM/CRLF.
If a required root is missing or the new destination already exists, stop and
preserve it; do not remove evidence or bypass the guard.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_decoder_hinge_repair_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_decoder_hinge.sh {pin}
```

The hash-only preflight checks18 packet assets,410 source assets,203 parent
assets and136 saved-replay assets, with zero neural/gradient/optimizer calls.
The supervised phase then verifies exact original/A0/legacy-A1 outputs and all20
corrected inactive penalty gradients before any optimizer. It saves one corrected
update plus two smaller same-direction diagnostic controls. Those controls never
select a model. Continuation permits **at most50updates**, with preservation and
positive source/global structure checks at1,10,20,30,40,50. The final≥1% requirement
stays unchanged. These are small TRAIN-sample updates, not full-corpus epochs.

`Ctrl+B`, then `D`, detaches tmux safely. A retained quality stop returns exit2;
it is not a reason to rerun or add epochs. A technical failure retains its
traceback/state. Saved audit and export still run. Archive `complete:true` means
packaging, never training or model-quality success. Download all three files.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Each SCP has one remote source, compatible with Windows PuTTY. Preserve any
partial export, failed logs and all original checkpoints/caches/splits.
{link('Independent local return workflow',return_name)}.
'''
    report=f'''# Decoder hinge correction: preparation evidence — 15 October 2026

**VERIFIED / PREPARED / UNRUN.** No upload, agent VM connection, model creation,
local gradient, optimizer update or actual VM training was performed by this
preparation. No useful model or new incumbent is claimed.

The spatial A/B path review confirms a legacy brightness helper was imported
after its derivative had been corrected in an earlier diagnostic. This new
experiment isolates that scalar-equivalent derivative repair on a fresh copy
of decoder A, keeping Adam1e-4 and all other intended objective arithmetic.
It does not assert this defect alone caused all previous failures.

The independent source AST comparison finds only the declared guard selection
in the objective; the helper changes only three clamp-to-ReLU zero-boundary
operators. Penalty values, coefficients, anchors, normalizers, schedule,15
tensors/609,219elements, gradient clipping, original radius and acceptance
thresholds stay fixed. The corrected file is loaded explicitly by exact SHA.
The original app model and old failed workers remain immutable.

All20 initial blocks must retain old/new scalar and metadata equality and
exactly zero corrected guard derivatives. Two parameter-gradient controls must
connect all15 tensors before the first Adam update. Full corrected first-step
RAW/PNG preservation and positive global/each-source structure are required
before updates2–50. Two scaled diagnostic controls cannot authorize continuation.
Every scheduled state is saved; first regression/resource stop ends training.
The≥1% global/each-source requirement is still the numerical qualification gate,
followed by direct useful-output review and separate native development work.
Passing small positive continuation checks is not model acceptance.

Preparation passes **14 CPU tests**, including rejection of the old import,
exact projection/stop/boundary cases, six fixed-objective rows across both
sources and read-only arithmetic checks against all50 retained Adam updates.
Those Adam proposal/moment errors were at most2.98023224e-8/1.39698386e-9;
these check implementation rounding, not softened scientific image tolerances.
An additional200 saved compositions match all RAW/PNG scorer values. The archive
has19 regular members/18 bound assets. All410source/203parent/136replay bindings,
actual hash-only CLI, Windows pre-neural rejection, six forbidden protocol
mutations, Python3.10 AST and read-only Bash syntax pass independently.
The safe local importer additionally passes12archive-boundary tests, including
resolution of the actual frozen preparation proof and rejection of missing
downloads before extraction. Returned VM code is never imported or executed.

The first unsealed draft and its13-test receipts remain hash-bound in
`outputs/cctv_dgp_decoder_hinge_repair_v1_design/preseal_revision_r0`. Pre-seal
review added saved-Adam calibration, complete probe RNG restoration evidence,
one combined180-second preflight and runtime versions. No historical sealed
protocol or pilot was changed. The sole sealed packet is the current one.

Protocol SHA256:`{pin}`.
Execution archive SHA256:`{prep['archive_sha256']}`.
Execution archive bytes:{prep['archive_bytes']:,}.
Design SHA256:`{prep['design_sha256']}`.
Original checkpoint SHA256:`{CHECKPOINT}`.

Manual caps:50updates/42gradient queries/50backwards,20original and20prior-feature
batches, at most252decoder/492recognizer batches, at most1,100saved compositions.
Up to20planned comparison sheets require direct return review. Worker2400s,
combined guard/parameter preflight180s, gradient-fit time600s excluding snapshots,
each snapshot300s, cache240s, audit/export600s each; external caps2430/630/630s
and30s kill grace. Disk/RAM8GiB before models, reserve1GiB, output3GiB,
VRAM20GiB/RSS12GiB. Runtime projections and export space are checked on the VM.
New CUDA parity, derivatives, timing, preservation and actual gain remain unrun.

Full model/optimizer/shadow/RNG and per-update gradient/proposal/projected
journals are retained; exact future resume is not guaranteed or authorized.
Diagnostic probe model/shadow/RNG restoration is required byte-exact. No final
identity, native CCTV or completion pixels enter this paired photographic TRAIN
experiment. No source/ethnicity/local-performance or hidden-identity claim follows.
The five milestones and seven automatic/assisted covering families remain.

{link('Independent prospective preparation audit','outputs/cctv_dgp_decoder_hinge_repair_v1_design/preparation_audit.json')}.
{link('Frozen design','outputs/cctv_dgp_decoder_hinge_repair_v1_design/design.json')}.
{link('Exact five manual steps',guide_name)}.
'''
    return_text=f'''# Decoder hinge repair — independent local return workflow

This workflow is used **after the human's manual L4 run and three downloads**.
The new experiment is currently PREPARED / UNRUN. Import/audit commands do not
start neural inference or training. Original/final identities and old failures
remain retained. Do not overwrite or rerun an existing returned analysis.

1. Import the three downloaded files from the project root in **PowerShell**:

```powershell
Set-Location -LiteralPath 'C:\\xampp\\htdocs\\YEAR 4\\Testing'
& '.\\venv\\Scripts\\python.exe' -B scripts/import_cctv_dgp_decoder_hinge_repair_v1_return.py
```

2. Audit the imported saved arrays and states in **PowerShell**:

```powershell
& '.\\venv\\Scripts\\python.exe' -B scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py --root outputs/cctv_dgp_decoder_hinge_repair_v1_return --source-root outputs/cctv_dgp_spatial_readout_v1_return --parent-root outputs/cctv_dgp_bank_photometric_guard_v1_return --replay-root outputs/cctv_dgp_spatial_path_replay_v1_return --protocol-sha {pin} --receipt outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/independent_audit.json
```

Import verifies the archive SHA, exact safe regular-file namespace, full byte/
SHA manifest and immutable packet bindings before extraction. Numerical audit
recomputes saved RAW/PNG metrics and17+17groups, controls, guard-gradient arrays,
every projection, Adam arithmetic, frozen/full states and first-stop logic.
Saved recognizer/derivative evidence is not a fresh local neural/autograd replay.

Every generated comparison sheet and any partial failure outputs still require
direct visual review. Numerical/export completion alone never qualifies a new
model, extra epochs, native CCTV results, the app or covering-family scope.
'''
    for name,text in [(guide_name,guide),(report_name,report),(return_name,return_text)]:retained_new(ROOT/name,text)
    prefix=f'''# Hinge correction isolated; new manual packet verified, UNRUN — 15 October 2026

The saved-path replay is audited and rejected; best sampled PNG gain0.392688%
still has14 preservation failures. Its design review found a reintroduced old
brightness-gradient defect. A fresh scalar-equivalent hinge repair is now
**VERIFIED / PREPARED / UNRUN**, retaining Adam1e-4, all15 A decoder tensors,
objective weights, original-relative radius, input/split roles and RAW/PNG gates.
This is a demonstrated processing fix to test, not proven model improvement.

The finite manual comparison first verifies20inactive-gradient controls and
original/A0/old-A1parity. It permits at most50newupdates, stopping at the first
scheduled preservation regression or nonpositive source/global structure.
Two smaller diagnostic controls cannot select a model. The full≥1% requirement
and useful-image/native/final reviews stay unchanged.14CPUtests,200saved scorer
compositions,50retained Adam arithmetic updates, file/archive/source AST,
Windows guard and Bash checks pass. No new training or agent VM call occurred.

{link('Exact five upload/install/tmux/run/download steps',guide_name)}.
{link('Independent preparation and finite bounds',report_name)}.
{link('Independent local return workflow',return_name)}.
The122,323-byte archive is ready. Actual CUDA correction/quality remains unrun.
Original checkpoints, research caches, splits and failures stay retained; no
app promotion, extra epoch or goal completion. All five milestones and seven
automatic/assisted covering families remain required. Exact prior status follows.

'''
    previous=read(ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_analysis/publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_NEXT_PILOT.md',
        'CCTV_DGP_POST_PATH_REPLAY_REVIEW_V1.md','CCTV_DGP_SPATIAL_PATH_REPLAY_V1_RESULTS.md']
    expected={name:previous['documents'][name]['after_sha256'] if name in previous['documents'] else previous['new_document_sha256'][name] for name in docs}
    changes={name:finish_prepend(ROOT/name,prefix,publication/'before'/name,expected[name]) for name in docs}
    write(publication/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        documents=changes,new_document_sha256={name:sha(ROOT/name) for name in [guide_name,report_name,return_name]},
        preparation_sha256=sha(DESIGN/'preparation.json'),independent_preparation_audit_sha256=sha(DESIGN/'preparation_audit.json'),
        importer_preparation_sha256=sha(DESIGN/'importer_preparation.json'),
        protocol_sha256=pin,archive_sha256=sha(archive),status='VERIFIED_PREPARED_UNRUN',
        model_qualified=False,new_optimizer_updates=0,agent_VM_calls=0))
    print(dict(complete=True,manual_guide=guide_name,protocol_sha256=pin,status='VERIFIED_PREPARED_UNRUN'))


if __name__=='__main__':main()
