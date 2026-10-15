"""Publish a verified UNRUN training guide; keep historical documents byte-exact."""
from datetime import datetime,timezone

from import_cctv_dgp_nearby_gradient_learning_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend,retained_new


def main():
    stage=OUT/'cctv_dgp_nearby_gradient_learning_v1_publication'
    (stage/'before').mkdir(parents=True,exist_ok=True);assert not (stage/'publication.json').exists()
    proofs=[OUT/('cctv_dgp_nearby_gradient_learning_v1_'+s+'.json') for s in ['preparation','transfer_audit','bash_syntax','saved_fixture_r1','source_paths_r1','preparation_order_review']]
    prepared,audit,bash,fixture,paths,order=[read(f) for f in proofs];assert all(read(f)['complete'] for f in proofs)
    packet=OUT/NAME;p=read(packet/'protocol.json');archive=OUT/(STEM+'-execution.tar.gz')
    assert sha(packet/'protocol.json')==PIN==prepared['protocol_sha256']==audit['protocol_sha256']
    assert sha(archive)==prepared['archive_sha256']==audit['archive_sha256']
    assert bash['checked_shell_sha256']==sha(packet/'scripts/run_learning.sh') and bash['exit_code']==0
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    prior=read(OUT/'cctv_dgp_recognizer_precision_v1_analysis/publication/publication.json')
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
          'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md',
          'CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md','CCTV_DGP_RECOGNIZER_PRECISION_REVIEW_V1.md']
    expected={n:prior['documents'][n]['after_sha256'] for n in docs}
    expected['CCTV_DGP_NEARBY_GRADIENT_LEARNING_REVIEW_V1.md']=prior['new_documents_sha256']['CCTV_DGP_NEARBY_GRADIENT_LEARNING_REVIEW_V1.md']
    assert all(sha(ROOT/n)==h for n,h in expected.items())
    protected=read(OUT/'cctv_dgp_recognizer_precision_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    guide_name='CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_VM.md';prep_name='CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_PREPARATION.md'
    upload=f'''cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"'''
    install=f'''cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp'''
    tmux='tmux new-session -A -s dgp_nearby_gradient_learning_v1'
    launch=f'''cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_nearby_gradient_learning_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_learning.sh {PIN}'''
    download=f'''cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."'''
    guide=f'''# Nearby-gradient learning V1 — manual VM commands

**Verified and UNRUN. This is a new finite training comparison.** It samples
gradients at the original state and two already measured nearby states, fits
one direction using all 96 RAW group constraints, and tests at most three
steps. At most one preserving research update can be accepted. The earlier
failed states are derivative samples only; their failures remain binding.

Require **9 GiB free after installation**. Estimate **8–20 minutes measurement**,
**10–25 minutes including export**, with an enforced **34-minute external
ceiling**. The estimate uses prior timings; this recipe's L4 runtime and current
VM space have not been measured. Timing, memory, storage, output and export
limits are enforced. Keep the brightness-guard and conditioning-response VM
directories, virtual environment, checkpoints, caches and prior failures intact.
If a stop occurs, retain it and download the exported evidence. A failure does
not authorize a rerun or a longer run. No cleanup or automatic training launch
is part of this guide.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
{upload}
```

2. Install in **VM SSH**:

```bash
{install}
```

3. Open **tmux**:

```bash
{tmux}
```

4. Launch **inside tmux**:

```bash
{launch}
```

5. Download after the export finishes, from **Windows Google Cloud SDK Shell**:

```bat
{download}
```

Protocol SHA256: `{PIN}`. Execution archive SHA256:
`{prepared['archive_sha256']}`. The checksum is ASCII with LF line endings.
Every PuTTY download uses one remote source. The wrapper preserves the training
exit code and exports success or failure evidence. Export `complete: true`
means the archive was created; it does not establish a successful training
comparison or useful restoration.

Every retained step must pass the original 17 RAW and 17 delivered-PNG group
checks and positive structure progress globally and in both sources. The
endpoint still requires at least 1% structure gain globally and in both
sources in both representations. This percentage measures reduction in a
specific paired TRAIN feature error; it is not recovered identity or native
CCTV performance. Below 1%, a safely accepted research state remains saved
but the pilot stops. The starting state and RNG are restored in every normal
outcome. Independent saved-evidence auditing and direct review of all 20 new
visual sheets follow the download. No epoch extension, app adoption or final
review qualification follows automatically.
{link('Preparation, limits and prospective audit',prep_name)}.
'''
    report=f'''# Nearby-gradient learning V1 preparation — 14 October 2026

**The new finite training comparison is verified and UNRUN. No new model or
structure gain is established.** The recognizer precision diagnostic passed
30 original parity checks, but FP64 reduced finite prediction misses only from
six to five of ten. Signed states crossed measured PReLU regions in both
precisions; repeats did not. This supports testing nearby derivatives while
retaining actual output acceptance. It does not identify a unique cause or
guarantee an improving step.
{link('Audited precision and activation evidence','CCTV_DGP_RECOGNIZER_PRECISION_V1_RESULTS.md')}.

## Changed learning recipe and frozen data

Use the same 100 exposed paired TRAIN controls: 20 references, ten from each
audited photographic source, with clear, blur, lowlight, motion and compound
profiles. Preserve presentation order separately from the original reference-
batch order. No native CCTV or reserved-final identities enter this comparison.
Dataset source labels do not imply ethnicity or Zamboanga performance.

The permanent current checkpoint remains SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Begin from the retained anchored comparison state that reproduces its RAW
output exactly. Only 12 conditioning tensors / 2,580,864 parameters may change
in a disposable copy. The original decoder, readouts, recognizer, buffers and
externally pretrained StyleGAN2 bank remain frozen. The original restorer is
DeblurGAN-v2-compatible; neither the frozen external bank nor a prepared pilot
is described as our trained generative prior.

Collect three fresh derivative rounds at initial and the exact saved plus02 /
minus02 vectors. These previously failed states cannot be accepted updates.
Full RAW arrays must match their retained arrays before gradients are used;
the scientific scalars, case/mask/grid/target roles and frozen hashes remain
bound. Initial-gradient comparisons retain the original parity bounds and
report any misses rather than erase previous failures. There is one solve,
at most three new actually evaluated proposals and at most one accepted step.

Each location retains all 32 RAW source/profile losses, giving 96 inequalities.
Normalize each gradient by its norm; retain feature target fraction 1 and
other-loss fraction 0.1. Form the float64 normalized Gram matrix in blocks.
Represent its numerical span using eigenvalues above 96×epsilon×max(1, largest
eigenvalue), retaining all 96 constraint rows. Solve the minimum-norm primal
with SLSQP, then obtain a nonnegative NNLS dual certificate. Finite iterations,
60s total solve stop, PSD/reconstruction, feasibility, complementarity and
primal/dual checks are fixed in the protocol. Recheck every original dot
inequality against the actual float32 direction. No ridge or loss removal is
permitted. An infeasible common direction stops with evidence retained.

This is a deterministic constrained experiment inspired by nearby-gradient
sampling, not the published random-sampling algorithm or a convergence claim.
[Burke, Lewis and Overton, author-hosted paper](https://cims.nyu.edu/~overton/papers/pdffiles/gradsamp.pdf).
PReLU sign changes demonstrate region crossings, not that the entire recognizer
is piecewise affine or every mismatch comes from those activations.
[PyTorch 2.9 PReLU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.PReLU.html).

## Actual acceptance and retained stop rules

The proposal uses the original 0.02 rate ceiling, factors 1, 0.5, 0.25, and
0.001 cumulative distance relative to the initial all-37 parameter norm.
PNG surrogate derivatives do not determine this proposal. Exact RAW/PNG
measurements still decide it. All 17 groups in both stages retain MSE ≤1e-12
regression, SSIM/appearance ≤1e-6 regression, and brightness-only gain fraction
≤0.2. Global and both-source structure gain must be positive before accepting
any step; the one-step endpoint must still reach at least 1% in both stages.
The later full-TRAIN 10% requirement, useful DEV review and untouched final
identities remain. A provisional accepted step below the endpoint requirement
stays archived as research evidence and the pilot stops.

Save the actual proposed conditioning before evaluating it. Retain complete
accepted component tensors, Torch CPU/CUDA RNG, frozen/original/recognizer
hashes, code/environment/data bindings and stateless learning metadata. Restore
the initial state and RNG after the finite comparison, with fresh failure
restoration receipts where possible. Exact continuation requires its own
declared packet; weights-only transfer is not an exact resume.

| Limit | Value |
| --- | ---: |
| Derivative queries / restoration forwards / recognizer forwards | 960 / 200 / 340 |
| Measurement / worker / gradient round / solve | 1,350s / 1,500s / 600s / 60s |
| Export / external worker / external export / each kill grace | 420s / 1,530s / 450s / 30s |
| Free after install / reserve / output payload | 9 GiB / 1 GiB / 3.5 GiB |
| Peak VRAM / process RSS | 20 GiB / 12 GiB |

After the first two batches of each derivative round, project the slowest
observed batch×20×1.5 and stop above 600s. Actual forwards are counted by hooks.
The upper counts derive from seven evaluations and three derivative rounds
over 20 reference batches: 200 restoration / 340 recognizer forwards. Three
rounds×20 batches×16 losses gives 960 queries. External termination ceilings
including both kill graces total 2,040s (34 minutes). Estimates 8–20 minutes
measurement / 10–25 with export are planning estimates, not new CUDA timings.

## Verified transfer and prospective independent audit

All **{prepared['packet_bindings']} packet bindings**, **{len(p['parent_required_sha256']):,} bank**
and **{len(p['response_required_sha256']):,} response** dependencies pass. The 17 original
RAW R1 source bindings also pass. The one explicitly volatile response
`export.log` remains retained; exact protocol/manifests/weights/caches and
outputs stay bound. Archive SHA256: `{prepared['archive_sha256']}`.
{link('Independent transfer and actual hash-only CLI audit','outputs/cctv_dgp_nearby_gradient_learning_v1_transfer_audit.json')}.

Meaningful checks reproduce an unsafe initial-only direction on an analytical
three-region fixture, solve all 96 rank-deficient constraints, reject an
infeasible direction, nonfinite data, a forged certificate and a dropped row.
These are array fixtures, not neural or trained outputs. The saved-output
fixture independently replays 600 RAW/PNG rows from 100 actual cases across
initial, signed and rollback states; all five existing PNG failures and the
original brightness stop remain rejected. Erasing failures is rejected.
The original and updated fixtures remain retained.
{link('Actual saved-output regression','outputs/cctv_dgp_nearby_gradient_learning_v1_saved_fixture_r1.json')}.

The preparation checker initially conflated frozen preview order with frozen
reference-batch order. It now checks each against its original definition,
with identical membership; worker, sealed protocol and quality bounds were
unchanged. The failed assertion and correction are retained. All 32 actual
retained gradient paths and their source/profile/metric query mappings pass.
Fourteen unsafe archive sets are rejected. The actual Windows execution guard
stops before neural loading. Hash-only CLI and Python 3.10 parsing pass. ASCII/LF
checksum and wrapper pass read-only Bash syntax; the sandbox signal-pipe error
was resolved by external syntax-only parsing, without executing the wrapper.

The prospective importer checks safe member paths, exact sizes, hashes,
membership, original protected files and transfer receipt without executing
returned code. The independent checker replays saved RAW/PNG pixels, metrics,
embeddings, conditioning states, all 96 gradient aggregates and certificates,
actual rounded proposals, gates, accepted full state and rollback/RNG evidence.
Saved neural derivative vectors are audited as retained VM evidence, not
recomputed locally. Every new proposal's output must receive direct visual
review across all 20 sheets before any quality claim. New full TRAIN, native
DEV and final evaluation remain separate prerequisites.

No actual training was launched by preparation. More epochs and app adoption
remain unqualified. Preserve original checkpoints, splits, caches, provenance,
full states and failed recipes. Actual training/gradients stay human-manual on
the existing L4 in tmux; direct agent VM access remains maintenance only. No
automatic historical run or follow-on exists. Detector-confidence recording
remains prepared only and UNRUN. All five milestones and all seven automatic/
assisted covering families remain required. Full goal incomplete.
{link('Exact five manual steps',guide_name)}.
'''
    retained_new(ROOT/guide_name,guide);retained_new(ROOT/prep_name,report)
    prefix=f'''# Nearby-gradient training comparison verified, UNRUN — 14 October 2026

The completed precision diagnostic left five of ten sampled finite prediction
misses in FP64. The next changed recipe collects fresh derivatives at three
measured states, retains all 96 RAW group constraints and requires actual
original RAW/PNG acceptance. Same 100 exposed paired TRAIN controls; original
decoder, readouts, pretrained bank and recognizer frozen. Only 12 conditioning
tensors may change. At most three new proposals and one accepted research
update; no epoch extension. No new improvement or model qualification exists.
{link('Exact Windows upload, tmux and single-source downloads',guide_name)}.
{link('Frozen design, finite limits and verification',prep_name)}.

All 25 archive / 5,852 bank / 1,246 response bindings, actual hash-only CLI,
meaningful solver/certificate/saved-output regressions, Windows manual guard
and read-only Bash syntax pass. Require 9 GiB free after install; estimate
10–25 minutes including export, external ceiling 34 minutes. Fresh VM capacity
and this recipe's timing remain unknown. No cleanup or automatic launch occurred.

Original 17 RAW and 17 PNG groups, preservation/appearance/brightness bounds,
1% global and both-source endpoint gate, 0.001 distance cap and later 10%
full-TRAIN requirement remain. Failed sample points cannot be accepted.
Safely accepted research state is retained even if the endpoint fails;
the initial state/RNG are restored. Independent return audit and all 20 visual
sheets remain mandatory. No new incumbent, app promotion or more epochs qualify.

Training/gradients/parameter changes stay human-manual L4 tmux. Agent VM access
stays inventory-first hash-bound maintenance only. Preserve checkpoints,
splits, caches, provenance, full states and gate failures. Native unpaired,
paired TRAIN and reserved-final roles remain separate. Existing app design,
all five milestones and seven automatic/assisted covering families remain.
Detector-confidence recording is prepared only and UNRUN. Goal incomplete.
Exact previous document follows.

'''
    results={}
    for n,h in expected.items():results[n]=finish_prepend(ROOT/n,prefix,stage/'before'/n,h)
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=sha(archive),documents=results,
        new_documents_sha256={n:sha(ROOT/n) for n in [guide_name,prep_name]},
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in proofs},
        exact_command_blocks=dict(upload=upload,install=install,tmux=tmux,launch=launch,download=download),
        manual_UNRUN=True,local_neural_or_training_calls=0,automatic_VM_launch=False,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=len(results),new_documents=2,command_blocks=5,manual_UNRUN=True))


if __name__=='__main__':main()
