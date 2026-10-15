"""Publish audited R3 loss stop, completed visual review and unrun R4 commands."""
from datetime import datetime, timezone
import re

from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, RETURN, ANALYSIS, PIN, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage = ANALYSIS / 'publication'
    assert not (stage / 'publication.json').exists()
    imported = read(ANALYSIS / 'import.json')
    audited = read(ANALYSIS / 'failure_audit.json')
    review = read(ANALYSIS / 'loss_alignment_review.json')
    visual = read(ANALYSIS / 'visual_review.json')
    green = read(ANALYSIS / 'loss_forward_green.json')
    prep = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_preparation.json')
    transfer = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json')
    syntax = read(OUT / 'cctv_dgp_conditioning_learning_v1_r4_bash_syntax.json')
    assert all(v['complete'] for v in [imported, audited, review, visual, green, prep, transfer, syntax])
    assert visual['all20_sheets_and300_cells_directly_viewed'] and not any(review['reported_counts'][n] for n in ['accepted_updates', 'parameter_proposals', 'backwards'])
    packet = OUT / 'cctv_dgp_conditioning_learning_v1_r4_vm'
    new_pin = sha(packet / 'protocol.json')
    assert new_pin == prep['protocol_sha256'] == transfer['protocol_sha256']
    assert transfer['loss_forward_and_unchanged_recipe_audit']['complete']
    assert syntax['exit_code'] == 0 and syntax['checked_shell_sha256'] == sha(packet / 'scripts/run_learning.sh')
    assert sha(OUT / 'cctv-dgp-conditioning-learning-v1-r4-execution.tar.gz') == prep['archive_sha256'] == transfer['archive_sha256']
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    previous = read(OUT / 'cctv_dgp_conditioning_learning_v1_r2_analysis/publication/publication.json')
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    used = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R3_VM.md'
    expected = {n: previous['documents'][n]['after_sha256'] for n in docs}
    expected[used] = previous['new_documents_sha256'][used]
    for n, h in expected.items():
        assert sha(ROOT / n) == h or ((stage / 'before' / n).exists() and sha(stage / 'before' / n) == h), 'Preserve concurrent edits'
    report_name = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R3_RESULTS.md'
    guide_name = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R4_VM.md'
    row = review['failed_CPU_row']
    report = f'''# Conditioning learning R3 loss-alignment stop — 14 October 2026

**R3 produced no learned checkpoint or image-quality gain.** It completed the
100 starting controls and420 gradient queries, then stopped before a parameter
proposal because the differentiable RAW SSIM value did not agree with the saved
scientific score. It did not reach the1% structure/preservation test. The earlier
256×256 pixel-fixture correction passed on the L4.

The reported manual VM duration is **{review['reported_VM_seconds']:.2f}s**:
34 candidate forwards,67 recognizer forwards,420 gradient queries,zero backward
calls,zero parameter proposals and **zero accepted updates/epochs**. The100
initial raw controls match the retained original DGP exactly. Saved raw/PNG
compositions,protected pixels,200 metric rows and reconstructed initial selected
component/frozen-state schemas passed the independent audit. No fresh
post-failure RAM-state hash is claimed; the failure handler reports no rollback
error. Original app/checkpoint files remain unchanged.

All166 return members and5,852 required parent assets pass hash checks. The
{imported['archive_bytes']:,}-byte result archive SHA256 is
`{imported['archive_sha256']}`. Four retained single-clear-MSE derivative samples
are finite and have the declared2,580,864-element layout. The gradient round is
incomplete; full aggregates,direction and step decisions do not exist. All420
neural queries are not independently replayed locally.
{link('Independent returned audit', 'outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/failure_audit.json')}.
{link('Bound failure and partial-gradient review', 'outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/loss_alignment_review.json')}.

## Demonstrated cause and correction

Location: returned `scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py:284`.
The worker originally returned an analytic float64 pooling expression as its
RAW SSIM forward,while the scientific evaluator uses retained float32
scikit-image operations. The13 completed32-query batches plus four extra
queries identify the next check as RAW query4,blur-profile SSIM for
`v9_tr_asian_00180_blur_lr24`. This location is inferred from the counts and
fixed ordering; the original failed GPU scalar was not exported.

A CPU utility replay checks all100 real saved controls and both raw/PNG MSE,
SSIM and landmark-detail losses:600 values. It reproduces one failing RAW SSIM
row on that same case. The analytic loss is{row['utility_value']:.17g},while
the saved scientific loss is{row['scientific_saved_value']:.17g}; difference
**{row['absolute_difference']:.17g}** exceeds the unchanged1e-6 alignment limit.
This is a CPU reproduction of the definitions,not an independently measured
GPU error magnitude.
{link('Retained failed regression', 'outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/loss_forward_red.json')}.

Fresh R4 uses the retained scientific RAW SSIM forward and the original
float64 surrogate's explicitly approximate backward,following the convention
already used for PNG SSIM. Removing this canonical-value correction restores
the exact original RAW derivative source. MSE,feature loss,mask support,
architecture,readouts,initial weights,cohort,quality/alignment bounds and finite
learning limits remain fixed. After correction **all600 values pass**; maximum
RAW SSIM difference is{green['maximum_RAW_SSIM_difference']:.17g}. No local neural
model,gradient query,optimizer or parameter displacement was used for the replay.
{link('Passing all-case regression', 'outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/loss_forward_green.json')}.

R4 also records case,stage,query,actual/expected loss,bound and snapshot if a
future alignment check fails. The used R3 directory,packet,helper,return and
checkers are retained. R4 is a separate immutable,**verified and UNRUN** manual
packet. There have been two processing stops in this new finite-learning
sequence,not two completed improvement experiments. Any third unsuccessful
attempt invokes the existing stop-and-diagnostic-question rule.

## Visual review and pending training

All20 full-resolution sheets and300 cells were directly inspected. Their only
model column is the exact original DGP starting output. Clear eye,lip,hair,
teeth and skin detail is softened in many cases; degraded features remain
diffuse. There is no new trained output to compare or promote. The review covers
20 exposed photographic TRAIN references and100 synthetic paired cases; it is
not native CCTV,independent final review,hidden identity,ethnicity or Zamboanga
evidence. Source labels identify archives only.
{link('Every sheet and per-sheet observations', 'outputs/cctv_dgp_conditioning_learning_v1_r3_analysis/visual_review.json')}.

The R4 pilot still trains only12 active conditioning tensors,freezing the
original decoder,readouts,bank,reference and recognizer. At most two accepted
full-cohort updates and three actual trials each; refresh all64 current-state
RAW/approximate-PNG derivatives after acceptance. Every accepted step needs
actual preservation across17 RAW and17 PNG metric groups,positive degraded
detail globally/per-source,brightness<=0.2 and cumulative parameter distance
<=0.1%. The endpoint requires1% structure gain globally and per-source on both
raw and delivered PNG. All partial accepted states and rejected trials are
exported. No longer-epoch continuation or app adoption is automatic.
{link('Current manual upload,tmux and download commands', guide_name)}.

Preparation verifies all14 archive files,ASCII/LF checksum,5,852 parent assets,
initial saved-state schema,100 scientific baseline cases,the600-value forward
regression,nonfinite rejection,retained gate boundaries and the four previously
rejected proposals. Windows execution is rejected before neural import; Bash
syntax and all command namespaces are checked. These are preparation checks,
not a new CUDA training result. Require9GiB free at launch; the original finite
time,VRAM,RSS,storage and export stops remain.
{link('Transfer and narrow repair audit', 'outputs/cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json')}.

The permanent baseline checkpoint SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Original app design,Auto/override,input-only review,checkpoints/splits/caches/
provenance/full states/failures,all five milestones and all seven automatic/
assisted completion families remain required. Full TRAIN's10% structure gate,
useful native DEV and independent final/app qualification remain pending.
Actual training stays human-manual on the L4; this return audit performs no VM
maintenance or training launch. The full goal remains active and incomplete.
'''
    old_path = stage / 'before' / used if (stage / 'before' / used).exists() else ROOT / used
    old_text = old_path.read_text(encoding='utf-8')
    blocks = re.findall(r'```[^\n]*\n(.*?)```', old_text, re.S)
    assert len(blocks) == 5
    blocks = [b.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r4').replace(
        'cctv-dgp-conditioning-learning-v1-r3', 'cctv-dgp-conditioning-learning-v1-r4').replace(PIN, new_pin) for b in blocks]
    guide = f'''# Conditioning learning V1 R4 — finite manual L4 pilot

**Verified and UNRUN. At most two accepted training updates.** R3 stopped at a
RAW SSIM loss-definition mismatch after420 gradient queries and zero updates.
R4 corrects the scientific forward value; its original approximate derivative,
model settings and quality gates remain. All600 saved RAW/PNG loss checks pass.
{link('Audited result and correction', report_name)}.

It learns only12 conditioning tensors on20 photographic TRAIN references/100
cases. Maximum three trials per update; all rejected trials are retained.
Each accepted change requires actual raw/PNG preservation,brightness<=0.2,
positive per-source detail and cumulative0.1% parameter distance. The endpoint
still needs1% structure gain globally and per-source on raw/PNG. Full epochs,
native DEV,final review and seven covering families remain later requirements.

Require **9GiB free after installation**. Estimate **8–15 minutes plus export**.
Launch checks current space. Learning1350s/internal worker1500s/outer1530s;
export420s/outer450s,each outer timeout has30s kill grace. Total outer allowance
34minutes. Gradient round600s,solver60s;VRAM20GiB/RSS12GiB/output3.5GiB/reserve1GiB.
Maximum180 candidate/360 recognizer forwards and1,280 gradient queries. Upload
size is{prep['archive_bytes']:,}bytes. Export includes any failure and accepted research
components with true update counts. No automatic follow-on or app promotion.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
{blocks[0]}```

2. Install in **VM SSH**:
```bash
{blocks[1]}```

3. Open **tmux** in VM SSH:
```bash
{blocks[2]}```
If an older tmux session is attached,detach **Ctrl+B,D** first.

4. Launch **inside tmux**:
```bash
{blocks[3]}```
Detach **Ctrl+B,D**; reattach with `tmux attach -t dgp_conditioning_learning_v1_r4`.
Progress is retained in `review.log`; the wrapper prints it when the finite
worker ends. Export also runs after a retained stop. `complete: true` means
archive creation; `accepted_updates` records actual learning. Keep the used
directory and original checkpoints/splits/failure records.

5. Download from **Windows Google Cloud SDK Shell** after export:
```bat
{blocks[4]}```
Use one remote source per PuTTY download. Return all3 files for independent
raw/PNG,state,derivative,preservation and every-sheet review. Do not rerun the
used R3 pilot. This R4 experiment is not a full epoch or a qualified app model.

Protocol SHA256: `{new_pin}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
{link('Transfer/state/forward/gate audit', 'outputs/cctv_dgp_conditioning_learning_v1_r4_transfer_audit.json')}.
'''
    retained_new(ROOT / report_name, report)
    retained_new(ROOT / guide_name, guide)
    stage.mkdir(exist_ok=True)
    backup = stage / 'before'
    backup.mkdir(exist_ok=True)
    prefix = f'''# R3 loss stop audited; canonical RAW-SSIM R4 prepared — 14 October 2026

R3 completed100 exact starting controls and420 gradient queries in115.99s,
then stopped before any proposal/update because RAW SSIM's float64 surrogate
forward differed from the saved float32 scientific score. No new trained output
or structure-gate result exists. All166 return files and5,852 parent bindings
pass;200 raw/PNG rows and all20 sheets/300 cells are reviewed. Starting outputs
retain the original DGP's softness. Actual failed GPU scalar was not exported.
{link('Audited result and narrow correction', report_name)}.

The CPU replay reproduces one mismatch on the inferred same case among600
checks:1.0115103e-6 versus the unchanged1e-6 alignment limit. Canonical scientific
RAW SSIM forward plus the existing approximate backward passes all600 values.
R4 changes this forward/import,immutable namespace and failure-scalar logging.
Architecture,initial weights,12 selected tensors,100-case cohort,64 losses,
quality/alignment bounds and all finite stops remain. R4 is verified and UNRUN:
at most2 accepted updates/3 trials each,actual preservation before acceptance,
and1% RAW/PNG structure globally/per-source at its endpoint. No longer epochs
or app adoption follow automatically. Two processing stops are retained; after
any third unsuccessful attempt,stop changes and ask one diagnostic question.
{link('Current manual upload,tmux and single-source downloads', guide_name)}.

Original checkpoints/app design,splits/caches/provenance/full states/failures,
paired-versus-native-unpaired separation,reserved final identities,all five
milestones and all seven covering families remain. Useful full TRAIN/native
DEV/final outputs and app qualification are pending. Actual training stays
human-manual on the L4. No agent VM action or training occurred. The full goal
remains active/incomplete. Exact previous content follows.

'''
    published = {n: finish_prepend(ROOT / n, prefix, backup / n, expected[n]) for n in docs}
    banner = f'''# Returned loss-alignment stop — retain R3; use corrected R4

R3 reached100 starting outputs and420 gradient queries,but no proposal or update.
It stopped before the1% gate because RAW SSIM's forward definition disagreed
with the scientific evaluator. {link('Audited findings', report_name)}.
{link('Current verified,unrun manual R4 commands', guide_name)}.
The used R3 instructions below are retained historical evidence; do not rerun.

'''
    published[used] = finish_prepend(ROOT / used, banner, backup / used, expected[used])
    evidence = [ANALYSIS / n for n in ['import.json', 'failure_audit.json', 'loss_alignment_review.json',
                 'visual_review.json', 'loss_forward_red.json', 'loss_forward_green.json']]
    evidence += [OUT / ('cctv_dgp_conditioning_learning_v1_r4_' + n + '.json') for n in
                 ['preparation', 'checker_preparation', 'transfer_audit', 'pixel_fixture_green', 'bash_syntax', 'RAW_forward_preparation']]
    write(stage / 'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          publisher_sha256=sha(__file__), documents=published,
          new_documents_sha256={n: sha(ROOT / n) for n in [report_name, guide_name]},
          evidence_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in evidence},
          returned_protocol_sha256=PIN, returned_archive_sha256=imported['archive_sha256'],
          returned_training_updates=0, loss_alignment_stop_before_actual_structure_gate=True,
          all20_visual_sheets_and300_cells_reviewed=True, all600_corrected_loss_forward_checks_passed=True,
          next_manual_protocol_sha256=new_pin, next_manual_archive_sha256=prep['archive_sha256'],
          next_training_pilot_unrun=True, actual_learning_worker_processing_stops_since_activation=2,
          no_agent_VM_training_or_maintenance=True, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, returned_updates=0, reviewed_sheets=20, forward_checks=600,
               next_manual_guide=guide_name, history_documents_preserved=6, training_unrun=True))


if __name__ == '__main__':
    main()
