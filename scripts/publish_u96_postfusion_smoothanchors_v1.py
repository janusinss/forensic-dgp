"""Publish an independently reviewed, unrun manual comparison and exact history."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_packet'
REV = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_design_review'


def doc(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, raw):
    with path.open('xb') as stream:
        stream.write(raw)


def main():
    build_path = OUT / 'build_receipt_v1.json'
    audit_path = OUT / 'independent_packet_audit_v1.json'
    cli_path = OUT / 'local_cli_verification_v1.json'
    build = doc(build_path)
    assert build['complete'] and build['neural_gradient_optimizer_training_VM_calls'] == 0
    assert doc(audit_path)['independent_packet_audit_pass'] is True
    assert doc(cli_path)['complete'] and doc(cli_path)['transfer_verification_pass']
    for name in ['packet_helper_review_v1.json', 'worker_source_review_v1.json']:
        review = doc(REV / name)
        assert review['complete'] and review['source_review_pass']
        for row in review['sources']:
            assert pin(ROOT / row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')}
    visual_path = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_return_audit/root_visual_review_v1.json'
    assert doc(visual_path)['all_planned_sheets_reviewed'] is True
    assert doc(visual_path)['generated_PNG_cells_reviewed'] == 712
    prefix = 'cctv-codeformer-u96-postfusion-smoothanchors-v1'
    stage = 'cctv_codeformer_u96_postfusion_smoothanchors_v1_vm'
    package = 'codeformer_u96_postfusion_smoothanchors_v1'
    protocol = build['protocol']['sha256']
    archive_sha = build['archive']['sha256']
    archive_bytes = build['archive']['bytes']
    guide_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_VM.md'
    guide = f'''# U96 smoother-anchor comparison V1: five manual VM steps

Prepared and **UNRUN**. The previous B64 trial completed64 real updates but failed the structure endpoint. Preserve it. This separate trial starts fresh from complete U96, changing only the clear/dark anchor loss to SmoothL1(beta0.01). It does not resume B64 or add epochs. U96 remains current.

Upload **{archive_bytes:,} bytes**. Archive SHA256 `{archive_sha}`. Protocol SHA256 `{protocol}`.

Require **8GiB free after installation**, about8.36GiB before copying weights. Previous B64 worker took about7minutes; plan10–20minutes for this new unbenchmarked criterion, excluding transfer/verification. Hard worker cap46minutes plus20seconds termination grace; separate export cap5minutes45seconds. Maximum64 updates/256 role exposures, four passes over64 optimization cases, not64 epochs. Preservation failures stop earlier.

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
python3 -B research/{package}/install_weights_from_r1.py --root . --protocol-sha {protocol}
```

The installer validates the successful R1 preflight and exclusively copies its exactR2/U96 initializer files. No new model or optimizer is constructed. It preserves all previous sources and checkpoints. Keep `-B`; it prevents the earlier bytecode-directory installation failure. If a source/storage guard fails, preserve the message and partial stage; do not bypass or overwrite it.

## 3. Open tmux from the normal VM SSH prompt

```bash
tmux new-session -A -s u96_smoothanchors_v1
```

## 4. Launch inside tmux

```bash
cd ~/forensic-dgp/{stage} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_u96_postfusion_smoothanchors_v1.py --root . --protocol-sha {protocol} --verify-transfer &&
bash scripts/run_learning.sh {protocol}
```

The supervisor prints a child PID and writes training output into `run.log`. In another VM SSH window, monitor with:

```bash
tail -F ~/forensic-dgp/{stage}/run.log
```

Detach using **Ctrl+B**, then **D**. Reattach with `tmux attach -t u96_smoothanchors_v1`. Exit0 passes only the finite TRAIN screen. Exit2 retains a quality stop; other nonzero exits indicate integrity/infrastructure/resource failure. Export success is separate from model success. Each committed head/Adam/RNG/sampler state remains saved; an interruption can retain an explicitly unknown optimizer outcome. Do not append updates, rerun the started stage or promote automatically.

## 5. Download from Windows Google Cloud SDK Shell

After export completes, download each remote file separately for PuTTY compatibility:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prefix}-export.json" "."
```

Independent archive/RAW/PNG/optimizer/gate auditing and all planned visual sheets precede any app/model decision. The same8 monitor photographs are reused adaptive TRAIN evidence after B64 feedback, not unseen validation. No native, FINAL or new app qualification is claimed. New actual training remains your manual workflow.
'''
    write(guide_path, guide.encode('utf-8'))
    report_path = ROOT / 'CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_PREPARATION.md'
    report = f'''# Complete-U96 smoother-anchor comparison V1: prepared, unrun

## Evidence and narrow hypothesis

[B64 returned review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_RETURN_REVIEW.md>) confirms64 real Adam commits, all36 new tensors changed and no activated norm clipping. All finite TRAIN preservation/dark checks passed, but terminal structure gains0.00436%–0.01615% missed1%. Root reviewed all30 sheets/712 generated images and found no visible useful improvement. Preserve the failed checkpoint; U96 stays current. Successful audit/export is not model qualification.

This distinct fresh comparison changes the **common clear/dark anchor criterion only**, from L1 to SmoothL1 beta0.01. Paired clean-target RGB/adjacent losses, coefficients, completeU96 initialization, four new head architecture/seed,539 frozen/36 trainable tensors, sampler/data/order, Adam/LR/warmup,64-update ceiling and all quality gates remain unchanged. It never resumes failed B64. This is not a pure architecture comparison or evidence that more epochs will help.

For a small nonzero residual, L1 has a sign-sized derivative; SmoothL1's derivative decreases toward zero inside its transition. That provides a mathematical reason to test whether anchoring suppresses tiny useful correction. It does not prove actual opposing head gradients or rule out bounded correction/frozen-token limitations. Beta is a prospective loss transition scale, not a new MAE allowance or a flat dead zone. Both clear and dark anchors change together, so the trial cannot isolate one anchor's contribution. [Official PyTorch2.9 SmoothL1](https://docs.pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html) and [exact source/design rationale](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_design_review/DESIGN_RATIONALE_V1.md>).

Clear loss becomes2*(SmoothL1 RGB+.25 SmoothL1 adjacent), against detached same-inputU96. Dark retains paired RGB/adjacent L1 and adds2*SmoothL1 of block16 luma difference from same-inputU96. Original L1 anchors remain saved as diagnostics. Objective totals across changed criteria have different scales; compare identical RAW/PNG target errors, gates and images. The ordered scalar-sum portability fix exactly matches all12 actualB64 VM gate dictionaries, without tolerance or changed thresholds. This implementation parity is not a second learning factor.

## Data reuse and immutable limits

A new explicit [conditional reuse admission](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_u96_postfusion_smoothanchors_v1_data_review/conditional_data_reuse_admission_v1.json>) passed3,053 checks. All96 syntheticTRAIN inputs,24 genuine direct256 targets,16 optimization/8 monitor photograph orders,72 original readable labels,24 clean controls, exclusions and source terms are exact. No new image selection occurred. Four earlier preflight witnesses remain excluded. Monitor has now informed design and is **reused adaptive TRAIN evidence**, without optimizer exposure; it is not unseen DEV/FINAL or verified independent people. Pretraining/person-level overlap remains unknown. Source NC/attribution limits stay retained. No native/wild/FINAL data enters training or this monitor.

Fresh Adam: maximum5e-5, four-update warmup, betas(.9,.99), eps1e-8, no weight decay/AMP/foreach/fused optimization and clip norm1. Four singleton roles from distinct photographs form each update.64updates equal256 exposures/four passes over64 optimization cases. Snapshots0/1/4/16/32/64 retain all24 original guards/teachers/thresholds plus32 monitor cases. Terminal local-center/global-whole blur adjacent-residual gain must reach1% in both RAW and PNG with all preservation/dark checks. No gate is rebased to U96 or relaxed; the small TRAIN screen never replaces fullDEV406 preservation/R2dark qualification, native evidence, user acceptance or independent FINAL.

Caps:712 full forwards,256 backwards, one separately declared non-neural-base hook exception probe, one Adam construction/64commits. Original539 tensors stay frozen; new heads start at exact output parity. Every commit retains head/Adam/RNG/sampler/status; failed and partial evidence remain. Frozen payloads are omitted from results and unchanged claims remain source/runtime-bound. Strict forwards stay active; only declared backward scopes may warn for inherited CUDA nondeterminism, then restore strict mode. Bitwise gradient replay is not claimed.

Require8GiB free after installation, continuing4GiB reserve,10GiB hostRSS,18GiB CUDAreserved and1536MiB runtime outputs. Internal2700/external2760seconds plus20seconds grace; export330/internal345/external seconds,1536MiB payload/archive ceiling. Prior B64 worker was about7minutes; new criterion runtime is unbenchmarked. Full namespace scans occur at checkpoints/snapshots/final checks; failure handling retains partials/status without forcing a full scan. No automatic retry/resume or extra epoch follows a stop.

## Verification and manual execution

Worker/contract peer review, helpers/builder review, explicit data reuse, full archive readback and actual inert CLI verification passed. Preparation used saved arrays/JSON/bytes and pure numerical fixtures; no local model, optimizer construction/step, forward/backward, VM connection or training launch occurred. These checks do not establish a Linux/L4 runtime pass or learning success. EarlierB64 safe CPU saved-checkpoint decoding is an audit, not training.

[Five-step manual guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_VM.md>). Thin upload{archive_bytes:,}bytes; archiveSHA`{archive_sha}`, protocolSHA`{protocol}`. Exact R1 initializer files are reused through an exclusive-copy installer with bytecode disabled. Each download uses one remote source. All predecessors stay immutable.

After return: independent state/pixel/gate audits and every visual sheet, followed by broader DEV/native and app testing only when evidence justifies progression. No default-model change or qualified successor exists yet. Optional protected-surroundings preparation stays separate with exact reviewed protected input pixels, original/derived preview, correction/confirmation/bypass. Prior paired studies do not justify always-on blur or copying arbitrary noise/JPEG/mosaic. Restoration remains first. The full goal, independent FINAL/user acceptance/app verification and later completion/covering-family requirements remain incomplete.
'''
    write(report_path, report.encode('utf-8'))
    note = f'''# 21 October2026: B64 learning audited; smoother-anchor comparison prepared, manual only

[Returned B64 review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_LEARNING_V1_RETURN_REVIEW.md>): all64 Adam commits verified, all36 new tensors changed, clipping inactive. All finite TRAIN preservation/illumination/dark endpoint checks passed; four blur structure checks failed at0.00436%–0.01615% against1%. Worker QUALITY_STOP/exit2 is authoritative. All712 saved RAW/PNG pairs,65 checkpoint/resume states and all30 planned sheets/712 generated cells were independently audited/root reviewed. No visible useful gain over U96; no promotion or appended epochs. Preserve the failed state, archive and all original records.

[New comparison preparation](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_PREPARATION.md>) and [five-step guide](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_V1_VM.md>): fresh completeU96 plus same initial new heads/Adam, changing only clear/dark anchor L1 to SmoothL1 beta0.01. This tests whether tiny corrections face excessive anchoring; the cause is unproven. Same data/order/seed/LR/64updates/539frozen/36trainable/qualitygates/caps. Explicit ordered scalar sums preserve actualVM3.10 gate arithmetic, with no tolerance or threshold change. Old L1 diagnostics stay saved. No training or VM action was launched here.

New explicit data-only reuse admission passed3,053 checks, preserving all96inputs/24targets,16opt/8monitor photograph splits and labels/terms/exclusions. Monitor is now reused adaptiveTRAIN evidence afterB64 feedback, never unseen DEV/FINAL or verified separate identities. No native/wild/FINAL optimization. Source/packet/CLI preparation checks pass but actual new L4 execution remains UNRUN. Thin upload{archive_bytes:,}bytes; protocolSHA`{protocol}`. User manually uploads/installs/launches in tmux; require8GiB free after installation, hard worker46minutes plus20seconds grace, separate export5minutes45seconds. New returns again require independent audits/all planned visuals. No automatic old rerun/resume or extra epochs.

U96 remains current. Native unpaired versus synthetic paired evidence, honest CodeFormer adaptation attribution, insufficient-information refusal, original app design, independent FINAL/user acceptance/fullapp checks remain required. Optional protected-surroundings preparation stays separately evaluated with exact protected input pixels, original/derived previews/correction/confirmation/bypass; always-on blur remains unjustified. Restoration remains first; later completion/covering-family scope remains required. The full goal remains ACTIVE and incomplete. This entry supersedes older next actions only; all previous bytes remain exact history below.

---

'''.encode('utf-8')
    documents = []
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md']:
        path = ROOT / name
        old = path.read_bytes()
        before = pin(path)
        pending = ROOT / (name + '.smoothanchors-v1.pending')
        write(pending, note + old)
        assert path.read_bytes() == old
        os.replace(pending, path)
        assert path.read_bytes()[len(note):] == old
        documents.append({'before': before, 'after': pin(path),
                          'preserved_suffix_offset_bytes': len(note), 'old_bytes_preserved_exactly': True})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
               'guide': pin(guide_path), 'preparation_report': pin(report_path), 'documents': documents,
               'build': pin(build_path), 'independent_packet_audit': pin(audit_path), 'inert_CLI': pin(cli_path),
               'new_model_optimizer_training_VM_calls': 0, 'actual_pilot_run': False, 'model_qualified': False}
    write(OUT / 'milestone_publication_v1.json', (json.dumps(receipt, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'complete': True, 'guide': str(guide_path), 'protocol_sha256': protocol,
                      'upload_bytes': archive_bytes, 'manual_training_UNRUN': True}))


if __name__ == '__main__':
    main()
