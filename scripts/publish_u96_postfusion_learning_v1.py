"""Publish a sealed but unrun manual pilot and preserve all old document bytes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_packet'
DATA = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_preparation_r1'
REVIEWS = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_design_review'
GUIDE = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_VM.md'
REPORT = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_PREPARATION.md'


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def doc(path):
    return json.loads(path.read_bytes())


def write_new(path, raw):
    assert path.resolve().is_relative_to(ROOT)
    with path.open('xb') as stream:
        stream.write(raw)


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def main():
    build_path = OUT / 'build_receipt_v1.json'
    packet_review_path = OUT / 'independent_packet_audit_v1.json'
    cli_path = OUT / 'local_cli_verification_v1.json'
    build = doc(build_path)
    assert build['complete'] and build['neural_gradient_optimizer_training_VM_calls'] == 0
    assert doc(packet_review_path)['independent_packet_audit_pass']
    assert doc(cli_path)['complete'] and doc(cli_path)['transfer_verification_pass']
    assert doc(DATA / 'independent_preparation_audit_v1.json')['independent_preparation_audit_pass']
    for name in ('packet_helper_review_v1_r2.json', 'worker_source_review_v1.json'):
        review = doc(REVIEWS / name)
        assert review['complete'] and review['source_review_pass']
        for binding in review['sources']:
            assert pin(ROOT / binding['path']) == {key: binding[key] for key in ('path', 'bytes', 'sha256')}
    protocol_sha = build['protocol']['sha256']
    archive_sha = build['archive']['sha256']
    archive_bytes = build['archive']['bytes']
    prefix = 'cctv-codeformer-u96-postfusion-learning-v1'
    stage = 'cctv_codeformer_u96_postfusion_learning_v1_vm'
    guide = f'''# U96 post-fusion learning V1: five manual VM steps

Prepared and unrun. This is a distinct image-learning trial from complete U96. It trains four new correction heads containing36 tensors, with the539 incumbent tensors frozen. Their final output projections start at zero, preserving the initial U96 output. U96 remains the selected model until returned evidence and visible outputs justify a replacement.

Upload size: **{archive_bytes:,} bytes**. Execution SHA256: `{archive_sha}`. Protocol SHA256: `{protocol_sha}`.

Require **8GiB free after installation**; exclusive weight copying needs about8.36GiB before copying. Estimated training/runtime **15–35minutes**, unbenchmarked for this new worker. Worker hard cap46minutes plus up to20seconds termination grace; separate export cap5minutes45seconds. Input verification and transfers add time. Training stops at64updates or an earlier preservation/resource failure.256 role-case exposures equal four passes over64 optimization cases; this is not64epochs.

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prefix}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Install in VM SSH

```bash
cd ~ &&
sha256sum -c {prefix}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm &&
test ! -e ~/forensic-dgp/{stage} &&
tar -xzf {prefix}-execution.tar.gz -C ~/forensic-dgp &&
cd ~/forensic-dgp/{stage} &&
python3 -B research/codeformer_u96_postfusion_learning_v1/install_weights_from_r1.py --root . --protocol-sha {protocol_sha}
```

The installer exclusively copies the two exact retained R1 initializer files after checking its successful preflight receipt. It preserves the old stage/checkpoints. It creates no optimizer or model. Bytecode is disabled in the command and source. If a required source or storage check fails, preserve the output/error; do not bypass guards or overwrite the stage.

## 3. Open tmux from the normal VM SSH prompt

```bash
tmux new-session -A -s u96_postfusion_learning_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/{stage} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_learning_v1.py --root . --protocol-sha {protocol_sha} --verify-transfer &&
bash scripts/run_learning.sh {protocol_sha}
```

The supervisor prints the child PID and `run.log` path. To detach safely, press **Ctrl+B**, then **D**. Reattach with `tmux attach -t u96_postfusion_learning_v1`. In another VM SSH window, view progress with:

```bash
tail -F ~/forensic-dgp/{stage}/run.log
```

Exit0 means the finite TRAIN screen passed. Exit2 means a preservation/quality stop; retain it. Other nonzero codes indicate an infrastructure/integrity stop. Successful export still does not establish an improved or qualified model. Every committed update retains head/Adam/RNG/sampler state. A timeout can leave an explicitly unknown optimizer outcome; the export preserves that distinction. Never append updates or rerun this stage automatically.

## 5. Download from Windows Google Cloud SDK Shell

After export completes, use one remote source per command for Windows PuTTY compatibility:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-export.json" "."
```

The next local action is independent archive/state/RAW/PNG/gate auditing and review of all planned sheets. A side-by-side experimental app comparison follows the returned evidence; no automatic default-model replacement is enabled. Keep the existing U96/R2/original benchmarks and all failures.
'''
    write_new(GUIDE, guide.encode('utf-8'))
    report = f'''# Complete-U96 image-learning pilot V1: prepared, unrun

## Material Passport

Experiment plan and reproducibility preparation,21October2026. Source/data/transfer verification only. No new model, gradient, optimizer, training, browser or VM call occurred in preparation. Actual execution remains the user's manual L4/tmux workflow. Full model qualification is pending.

## Why this trial

The real R1 preflight passed independently audited same-device full-U96 parity and image-gradient connectivity, with zero optimizer updates. Root reviewed both sheets/all20 generated cells; no quality gain existed because no weight changed. The next question is whether a connected image-loss update yields useful correction while preserving appearance. The feature-only U96 recipe is retained unchanged as the incumbent.

This new B-only candidate freezes the exact515 R2 base tensors plus24 U96 adapters and trains36 new spatial post-fusion head tensors(168,960parameters). It starts from exact complete-U96 output parity. It is a route/capacity/objective package comparison against fixed U96, not a pure causal comparison of architectural placement. It cannot change wrong frozen code indices and its residual bound can limit correction. A passing preflight or nonzero gradient does not guarantee useful learning.

## Data and finite learning

The fixed cohort contains16 optimization photographs and8 separate TRAIN-monitor photographs, each with clear/globalGaussian1.2/localGaussian1.2/linear-sRGB exposure0.35 roles.72 degraded inputs received isolated input-only readable labels;24 clear controls retain the root-admitted genuine1024 clean sources and exact direct256 targets. The independent data audit passed4,488 checks,409 current file pins and all96 exact input reconstructions with zero differing components.

The initial input-only review retained67 readable,2 borderline and3 out-of-scope labels. Two monitor photographs were deferred as whole groups; the next two entries in the prospectively frozen reserve order replaced them before any new model output. All six replacement degraded inputs passed a fresh isolated review. All16 optimization photographs stayed unchanged. Original candidate manifests, five exclusions, eight deferred role cases and one readable-but-deferred observation remain exact. The remaining pool is38 unused plus2 input-deferred photographs; the four preflight witnesses remain outside learning.

These are photographs, not verified independent subjects. All new cases are paired synthetic FFHQ TRAIN evidence; exact-photo/pixel/URL alias checks do not establish person-level or pretraining nonoverlap. TRAIN monitoring is held out from this pilot's optimizer, not an independent DEV/FINAL set. Source-specific attribution/terms, including NC restrictions, remain retained. No native CCTV, wild input or FINAL image enters optimization or monitor selection. No ethnicity or Zamboanga-performance inference is supported.

Fresh Adam uses a prospective5e-5 maximum LR, four-update linear warmup, betas(.9,.99), eps1e-8, no decay/AMP/foreach/fused path and clip norm1. Each update accumulates four singleton roles from four different photographs.64updates mean256 exposures/four full role-table passes, not64epochs. The LR/coefficient choices are finite safety hypotheses, not proven optimal settings.

Local blur balances central/outside RGB and adjacent-residual losses; global blur uses paired RGB plus adjacent residual. Clear controls anchor complete-U96 output. Synthetic dark inputs use actual clean targets plus a block16 luminance anchor to preserve useful U96 illumination. These known synthetic regions supervise loss; they are not deployment masks or automatic surroundings preparation. No GAN, completion, new perceptual network or arbitrary added-noise preprocessing is introduced. Exact formulas are in the reviewed [contract](<C:/xampp/htdocs/YEAR 4/Testing/research/codeformer_u96_postfusion_learning_v1/CONTRACT.md>) and [design recommendation](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_learning_v1_design_review/FINITE_LEARNING_RECOMMENDATION_V1.md>).

## Stops, state and evidence

Snapshots0/1/4/16/32/64 evaluate32 monitor cases plus the24 unchanged historical TRAIN guards. Historical original-CodeFormer clear teachers, R2 dark teachers, paired filter targets and original thresholds remain exact; none are reset to U96. Additional monitor checks retain RAW/PNG paired fidelity and clear/dark behavior. At64, local-center/global-whole adjacent-residual mean gain must each reach1% in both RAW and PNG, with preservation/dark-retention checks. This is a new small TRAIN-monitor screen; it does not replace the older DGP feature gate, full DEV406 preservation, R2 dark-retention or visible-quality acceptance.

Ceilings are712 real CodeFormer forwards,256 backward attempts, one Adam construction and64 commits. One separately counted controlled hook-exception wrapper probe invokes a raising stub without an original CodeFormer forward. All backward warning scopes, strict-mode restoration, frozen-state receipts and failed/partial state remain retained. Caller device gradient replay is not guaranteed. Failure gates retain their actual phase/update and stop; there is no automatic retry/resume.

Every committed update saves portable head arrays plus CPU-serialized Adam/RNG/sampler continuation state; all training/readout RAWs and floored PNGs remain separate. Frozen539-tensor payloads are omitted from return and their unchanged claims remain runtime/source-bound receipts; original files remain retained. No model is promoted automatically. Useful visible DEV outputs, separately reported native/unpaired evidence, user acceptance, independent FINAL and the full app flow are still required.

Resources: initial8GiB free after installation, continuing4GiB reserve,10GiB hostRSS,18GiB CUDA reserved and1536MiB runtime output ceiling. Worker2700seconds internal/2760external plus up to20seconds termination grace; export330internal/345external seconds and1536MiB payload/archive ceiling. Runtime estimate15–35minutes is unbenchmarked for this worker. Incremental exclusive-write accounting avoids a quadratic growing-tree scan. Full namespace/size readback occurs at checkpoints, snapshots and final integrity checks. Failure handling retains partial files, durable status and a failure receipt without forcing a full namespace scan.

## Transfer and next action

[Five-step manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_VM.md>). Thin upload{archive_bytes:,}bytes, archiveSHA`{archive_sha}`, protocolSHA`{protocol_sha}`. The installer reuses two exact retained successful-R1 initializer files after verifying their original protocol/result; all predecessors remain untouched. Every Python command disables bytecode. All three downloads use one remote source per command.

The independently reviewed source, explicit input admission, byte-verified archive and inert CLI establish preparation only. Actual Linux/L4 learning, endpoint success, learned quality and app integration remain unrun. Optional protected-surroundings preparation stays separate, with exact reviewed protected pixels, immutable original, correction/confirmation/bypass; paired results do not justify always-on blur. Automatic proposals/Protect-Apply painting/adaptive matching require their own validation. Restoration remains first; later completion/removal-mask and covering-family requirements remain in the full goal.

Documentary correction: an earlier R1-return milestone used the phrase 'R1 missing-dependency'. The missing-psutil failure belonged to originalV1; R1's earlier installation stop was bytecode-cache creation. All original records remain preserved and the successful real R1 preflight supersedes both as the latest feasibility evidence.
'''
    write_new(REPORT, report.encode('utf-8'))
    note = f'''# 21 October 2026: complete-U96 finite image-learning pilot prepared; manual launch only

[Preparation](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_PREPARATION.md>) and [five-step guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_VM.md>). The real R1 preflight passed parity/gradient feasibility with0updates. This distinct B-only trial preserves all539 complete-U96 tensors and trains36 new tensors across four image-supervised correction heads, capped at64updates/256exposures(four role-table passes, not64epochs). Adam maximumLR5e-5 with four-update warmup, clear U96 anchors and paired dark/illumination supervision are prospective choices. No new training or VM process was launched here.

The independently audited cohort is16 optimization plus8 TRAIN-monitor photographs, four roles each:96 exact inputs/24 genuine direct256 targets,72 degraded input-only readable labels and4,488 saved-data checks. Two monitor photographs were deferred after5 nonreadable/out-of-scope labels, then replaced from the fixed reserve order before outputs. All exclusions, prior manifests/labels, source terms/overlap limits and preflight witnesses remain retained. These photograph partitions are not independent identities or FINAL evaluation; native/unpaired evidence remains separate.

Snapshots0/1/4/16/32/64 retain all24 historical TRAIN guards with original teachers/thresholds plus additional RAW/PNG monitor checks. A1% terminal blur-structure screen is new small-TRAIN evidence, not a replacement for existing DEV/preservation or R2-dark qualification.712 real forwards,256 backwards, one separately declared non-neural-base hook probe, one Adam and64commits are the ceilings. Every committed head/Adam/RNG/sampler state and failures remain saved. Worker cap46minutes plus20seconds grace; export5minutes45seconds; require8GiB free after copying exact R1 weights. Independent source/data/archive/CLI checks establish preparation only.

Thin upload{archive_bytes:,}bytes; protocolSHA`{protocol_sha}`. The user manually uploads/installs/launches in tmux; download each of three result files with a separate gcloud command. No automatic retry, extra epochs, baseline replacement or promotion follows. First audit returned states/RAW/PNG/gates and all planned visuals, then assess a side-by-side experimental comparison. U96 remains current.

Optional protected-surroundings preparation remains separate with correction/confirmation/bypass; always-on blur remains unjustified. Preserve all original models/splits/failures and the existing app design. Restoration-first improvement, user acceptance, native evidence, independent FINAL, full app flow and later covering-family scope remain required. The full goal is incomplete. Clarification: originalV1 had the missing-psutil failure; R1 had the earlier installer-cache stop. Earlier dated records below remain exact history, with next actions superseded only by this entry.

---

'''.encode('utf-8')
    documents = []
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        path = ROOT / name
        before = pin(path)
        old = path.read_bytes()
        pending = ROOT / (name + '.u96-learning-v1.pending')
        write_new(pending, note + old)
        assert path.read_bytes() == old
        os.replace(pending, path)
        assert path.read_bytes()[len(note):] == old
        documents.append({'before': before, 'after': pin(path),
                          'preserved_suffix_offset_bytes': len(note), 'old_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
               'guide': pin(GUIDE), 'preparation_report': pin(REPORT), 'documents': documents,
               'build_receipt': pin(build_path), 'packet_review': pin(packet_review_path), 'CLI': pin(cli_path),
               'new_model_optimizer_training_VM_calls': 0, 'actual_pilot_run': False, 'model_qualified': False}
    write_new(OUT / 'milestone_publication_v1.json', encode(receipt))
    print(json.dumps({'complete': True, 'guide': str(GUIDE), 'archive_bytes': archive_bytes,
                      'protocol_sha256': protocol_sha, 'actual_training_run': False}))


if __name__ == '__main__':
    main()
