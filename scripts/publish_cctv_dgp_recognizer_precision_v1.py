"""Publish verified unrun manual commands and preserve prior scientific history."""
from datetime import datetime, timezone

from import_cctv_dgp_recognizer_precision_v1_return import ROOT, OUT, NAME, STEM, PIN, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import finish_prepend, retained_new


def main():
    stage = OUT / 'cctv_dgp_recognizer_precision_v1_publication'
    (stage / 'before').mkdir(parents=True, exist_ok=True); assert not (stage / 'publication.json').exists()
    evidence = [OUT / ('cctv_dgp_recognizer_precision_v1_' + s + '.json') for s in ['preparation', 'transfer_audit', 'bash_syntax', 'fixture']]
    prepared, audited, bash, fixture = [read(f) for f in evidence]
    assert all(v['complete'] for v in [prepared, audited, bash, fixture])
    packet = OUT / NAME; p = read(packet / 'protocol.json')
    assert PIN == sha(packet / 'protocol.json') == prepared['protocol_sha256'] == audited['protocol_sha256']
    assert sha(OUT / (STEM + '-execution.tar.gz')) == prepared['archive_sha256'] == audited['archive_sha256']
    assert bash['exit_code'] == 0 and bash['checked_shell_sha256'] == sha(packet / 'scripts/run_precision.sh')
    assert not (packet / 'outputs').exists()
    previous = read(OUT / 'cctv_dgp_appearance_input_response_v1_analysis/publication/publication.json')
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md',
            'CCTV_DGP_APPEARANCE_RESPONSE_REVIEW_V1.md']
    expected = {n: previous['documents'][n]['after_sha256'] for n in docs}
    expected['CCTV_DGP_RECOGNIZER_PRECISION_REVIEW_V1.md'] = previous['new_documents_sha256']['CCTV_DGP_RECOGNIZER_PRECISION_REVIEW_V1.md']
    assert all(sha(ROOT / n) == h for n, h in expected.items())
    guide_name = 'CCTV_DGP_RECOGNIZER_PRECISION_V1_VM.md'; prep_name = 'CCTV_DGP_RECOGNIZER_PRECISION_V1_PREPARATION.md'
    upload = f'''cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"'''
    install = f'''cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp'''
    tmux = 'tmux new-session -A -s dgp_recognizer_precision_v1'
    launch = f'''cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_recognizer_precision_v1_vm.py --root . --input-root ../cctv_dgp_appearance_input_response_v1_vm --response-root ../cctv_dgp_conditioning_response_v1_vm --bank-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&
bash scripts/run_precision.sh {PIN}'''
    download = f'''cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."'''
    guide = f'''# Recognizer precision V1 — manual VM commands

**Verified and UNRUN. This is a computation diagnostic, not more restoration
training.** Appearance-input V1 completed and its gradients repeated, but finite
appearance predictions still disagreed in 9/6/3 groups. This check compares the
entire frozen recognizer in FP32 and FP64 on two problematic references and all
five profiles: ten already exposed TRAIN cases.

Require **3 GiB free after installation**. Estimate **2–6 minutes measurement**,
**3–8 minutes including export**; the external ceilings total **12 minutes
30 seconds**. FP64 speed has not been measured here; an early projection and
finite stops protect the run. Keep the existing input-response, conditioning-
response and brightness-guard VM directories intact. No cleanup is included.
An insufficient-space or competing-workload stop must remain retained.

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

The wrapper retains `review.log`, exit code and an export after success or
failure. Detach with Ctrl+B, then D. This does not stop the run. Preserve the
logs and downloaded archive even if a stop occurs; do not edit or rerun it.
`complete: true` describes export and does not imply trained-model improvement.

5. Download to **Windows Google Cloud SDK Shell** after export finishes:

```bat
{download}
```

Each download has one remote source, as required by Windows PuTTY. Return all
three files in the local `outputs` directory for an independent audit.

Protocol SHA256: `{PIN}`.
Execution archive: **{prepared['archive_bytes']:,} bytes**, SHA256
`{prepared['archive_sha256']}`. Checksum and shell script are ASCII/LF-compatible.
{link('Scope, limits and preparation proof', prep_name)}.

Maximum 16 recognizer forward attempts and 40 RGB-input gradient queries;
zero restoration forwards, learned-weight/optimizer/accepted updates or epochs.
The original recognizer and checkpoint remain intact. FP64 uses a disposable
precision copy, not new learned weights. Official FP32/PNG scientific scores
and every quality gate remain unchanged. No new restored image, native CCTV
or reserved-final identity is used. No automatic next training run follows.
The full goal and seven automatic/assisted covering families remain incomplete.
'''
    report = f'''# Recognizer precision V1 preparation — 14 October 2026

**The finite diagnostic is verified and UNRUN; no CUDA result or improved model
is established.** It follows the audited appearance-input boundary result and
does not repeat the failed training direction. The same smallest saved plus02
and minus02 arrays are measurement inputs, not fresh restoration proposals.
{link('Audited appearance-input result', 'CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_RESULTS.md')}.

## Frozen case selection and measurement

Selection was sealed before execution: one source reference with the largest
absolute smallest-step saved-double residual among failed degraded TRAIN cases,
with deterministic id tie-breaks. Selected references are `tr_asian_06782`
(lowlight failure) and `tr_ffhq_59043` (compound failure). Each retains clear,
blur_lr24, lowlight_lr32, motion_lr48 and compound_lr24: ten cases, two existing
five-case batches. Their references, masks, fixed 2×3 grids and targets stay
unchanged. Dataset labels do not imply ethnicity or Zamboanga performance.

Both FP32 and FP64 use initial, plus02, minus02 and an initial repeat. Preserve
the original initial DGP RGB as the fixed baseline in every phase. The original
float32 embeddings must agree with the already saved arrays. FP64 promotes
canonical saved RGB, the float32 grid and float32 neutral fill exactly and
retains double crop, encoder output, normalization and cosine dot. All floating
encoder outputs are checked for the declared dtype. A disposable double copy's
weights must round-trip to the original frozen FP32 state, while both precision
states and buffers remain frozen. Unsupported kernels or mixed precision stop;
there is no fallback or quality-tolerance change.

Save 40 per-case RGB-input gradients, 16 embedding batches and all scalar,
state/RNG/precision receipts. Packed positive/zero maps from each PReLU/ReLU
module preserve signs for the five candidate images. The independent checker
can recount crossed activation regions and case counts without replaying a
neural model. The ONNX graph was inspected without executing it: 237 float32
initializers and 25 PRelu nodes. Actual converted-module/kernel support remains
a VM measurement question. These are directional, selected-case diagnostics;
no universal derivative proof or new restoration quality is claimed.

FP32 repeat and prior-gradient checks retain elementwise 1e-4/1e-8 and norm
1e-4/1e-10 bounds. FP64 repeats use those same declared diagnostic bounds. Full
network precision responses use the original 5% diagnostic comparison and noise
rule; official FP32/PNG scientific metrics and all preservation/structure gates
stay separate and unchanged. Original previous parity misses and all failed
model/image decisions remain binding.

## Finite limits

| Limit | Value |
| --- | ---: |
| Measurement / worker / export | 360s / 390s / 180s |
| External measurement / export / each kill grace | 480s / 210s / 30s |
| Free space after install / reserve / output payload | 3 GiB / 1 GiB / 256 MiB |
| Peak VRAM / process RSS | 12 GiB / 8 GiB |
| Recognizer forwards / RGB-input derivative queries | 16 / 40 |

After the first FP64 initial phase, project its time ×8×1.5; stop above 300s.
Total external ceiling is 750s. Restore saved Torch CPU/CUDA RNG, export success
or failure evidence and allow no used-packet rerun or automatic follow-on.
Zero restoration forwards, learned-weight changes, optimizer/accepted updates
and full epochs. Estimate 2–6 minutes measurement / 3–8 with export, based on
the smaller case set and the previous 77.61s diagnostic. FP64 speed is unknown
until the finite manual run; estimates are not fresh CUDA evidence.

## Verified preparation and boundaries

All {prepared['packet_bindings']} archive bindings and all {p['input_required_sha256'].__len__()}
stable input / {len(p['response_required_sha256']):,} response / {len(p['bank_required_sha256']):,}
bank bindings pass. Archived empty `export.log` files grow after VM export;
those live dependency paths alone are explicitly excluded and retained. The
scientific manifests, protocols, source files, gradients and saved outputs stay
exactly bound. Archive SHA256: `{prepared['archive_sha256']}`.
{link('Independent transfer and actual hash-only CLI audit', 'outputs/cctv_dgp_recognizer_precision_v1_transfer_audit.json')}.

Meaningful saved-array checks cover both gradient dtypes, wrong signed
projections, unsafe data, frozen TRAIN selection, 14 unsafe tar-member sets,
packed sign consistency and a real Windows manual-execution rejection before
neural loading. A five-case saved-data fixture rejects incorrect projections,
a moving control baseline and contradictory activation maps, restoring its
scratch bytes. Its promoted FP64 coefficients and activation maps are analytical
fixtures, not new neural outputs. The original fixture's unnormalized promoted
FP32 vectors failed the stricter FP64 norm check; R1 normalizes that analytical
fixture without changing the VM worker, auditor or any bound. The original
fixture/stop and RED checks remain retained.
{link('Saved-data corruption fixture', 'outputs/cctv_dgp_recognizer_precision_v1_fixture.json')}.

The actual hash-only CLI passes with zero neural calls. Its free-space receipt
describes Windows, not the VM. ASCII/LF checksum and wrapper pass. Bash -n
syntax-only parsing passed after the Windows sandbox signal-pipe limitation;
no shell-script body or VM command was executed.
{link('Read-only Bash syntax receipt', 'outputs/cctv_dgp_recognizer_precision_v1_bash_syntax.json')}.

The immutable return importer checks safe paths, regular files, membership,
sizes, protocol, source bindings, exact hashes and protected files before import;
it executes no returned source. The prospective auditor replays full saved
gradients, scalars, actual pixel projections, packed activations, repeat
comparisons and precision responses without local neural execution. State and
off-case-zero receipts stay distinguished from independently replayed arrays.
No new face image or visual sheet is planned; the earlier 20 sheets/1,000 cells
have already been reviewed. Actual new candidate images in future training
still require their own independent raw/PNG and every planned-sheet review.

The original app checkpoint and design, checkpoints, splits, caches, provenance,
full states and failures remain. Original restorer: DeblurGAN-v2-compatible;
comparison StyleGAN2 bank: externally pretrained and frozen. No model promotion
or extra epochs qualify from this preparation or a successful diagnostic.
Native unpaired / paired TRAIN / reserved final roles remain separate. All five
milestones and all seven automatic/assisted covering families remain required.
Derivatives and precision-copy execution stay human-manual on the existing L4
in tmux; agent VM access is maintenance only. Detector-confidence work stays
prepared only and UNRUN. Full goal incomplete.
{link('Exact five manual steps', guide_name)}.
'''
    retained_new(ROOT / guide_name, guide); retained_new(ROOT / prep_name, report)
    prefix = f'''# Recognizer precision diagnostic verified, UNRUN — 14 October 2026

Appearance-input V1 completed and passed saved-evidence auditing: all 100
original gradient-repeat bounds pass, but finite appearance-input predictions
still disagree in 9/6/3 groups. Saved double-dot reporting does not repair them;
case cancellation and the earlier path also contribute. No new model or gain.
{link('Completed audited result', 'CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_RESULTS.md')}.

Next verified manual diagnostic compares full FP32 and disposable FP64
recognizer calculations plus packed activation transitions on ten exposed TRAIN
cases: two problematic references, all five profiles. Maximum 16 forwards /
40 RGB-input derivatives; zero restoration forwards, learned-weight/optimizer/
accepted updates or epochs. Original baseline remains fixed in every phase.
{link('Exact Windows upload, tmux and single-source downloads', guide_name)}.
{link('Scope, finite limits and preparation proof', prep_name)}.

Require 3 GiB free after installation, 1 GiB reserve, 256 MiB output, finite
timing/projection/VRAM/RSS/export stops; external ceiling 12min30s. All 14 archive,
192 stable input, 1,246 response and 5,852 bank bindings, hash-only CLI, meaningful
saved corruption checks, Windows guard and read-only Bash syntax pass. UNRUN
means no new CUDA result or improved restored image. Actual precision-copy and
derivative execution stays human-manual L4 tmux; direct agent VM access remains
maintenance only. No automatic follow-on exists.

All original scientific FP32/PNG metrics, gates, failure records, original
checkpoints, splits, caches, provenance, full states and app design remain.
No new incumbent or more epochs qualified. Native unpaired / paired TRAIN /
reserved final roles, all five milestones and seven automatic/assisted covering
families remain required. Detector-confidence work remains prepared only and
UNRUN. Goal incomplete. Exact previous document follows.

'''
    records = {n: finish_prepend(ROOT / n, prefix, stage / 'before' / n, h) for n, h in expected.items()}
    protected = read(OUT / 'cctv_dgp_appearance_input_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(stage / 'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__), protocol_sha256=PIN, archive_sha256=prepared['archive_sha256'],
        documents=records, new_documents_sha256={n: sha(ROOT / n) for n in [guide_name, prep_name]},
        exact_command_blocks=dict(upload=upload, install=install, tmux=tmux, launch=launch, download=download),
        evidence_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in evidence},
        manual_UNRUN=True, automatic_VM_launch=False, local_neural_or_training_calls=0,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, history_documents=len(records), new_documents=2, manual_UNRUN=True))


if __name__ == '__main__':
    main()
