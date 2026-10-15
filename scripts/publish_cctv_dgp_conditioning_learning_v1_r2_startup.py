"""Publish the audited startup failure and verified, unrun R3 manual commands."""
from datetime import datetime, timezone
from pathlib import Path

from import_cctv_dgp_conditioning_learning_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage = ANALYSIS / 'publication'
    assert not (stage / 'publication.json').exists()
    imported = read(ANALYSIS / 'import.json')
    failure = read(ANALYSIS / 'failure_audit.json')
    startup = read(ANALYSIS / 'startup_review.json')
    prep = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_preparation.json')
    transfer = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_transfer_audit.json')
    green = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_pixel_fixture_green.json')
    syntax = read(OUT / 'cctv_dgp_conditioning_learning_v1_r3_bash_syntax.json')
    assert all(r['complete'] for r in [imported, failure, startup, prep, transfer, green, syntax])
    assert not any(startup['reported_counts'].values()) and not startup['finite_structure_gate_reached']
    assert transfer['startup_shape_and_exact_unchanged_recipe_audit']['complete']
    packet = OUT / 'cctv_dgp_conditioning_learning_v1_r3_vm'
    pin = sha(packet / 'protocol.json')
    assert pin == prep['protocol_sha256'] == transfer['protocol_sha256'] == green['protocol_sha256']
    assert syntax['exit_code'] == 0 and syntax['checked_shell_sha256'] == sha(packet / 'scripts/run_learning.sh')
    assert sha(OUT / 'cctv-dgp-conditioning-learning-v1-r3-execution.tar.gz') == prep['archive_sha256'] == transfer['archive_sha256']
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    previous = read(OUT / 'cctv_dgp_live_conditioning_v1_r1_analysis/publication/publication.json')
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected = {n: previous['documents'][n]['after_sha256'] for n in docs}
    used = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R2_VM.md'
    activation = 'CCTV_DGP_LIVE_CONDITIONING_V1_R1_RESULTS.md'
    expected.update({n: previous['new_documents_sha256'][n] for n in [used, activation]})
    for n, h in expected.items():
        assert sha(ROOT / n) == h or ((stage / 'before' / n).exists() and sha(stage / 'before' / n) == h), 'Preserve concurrent edits'
    report_name = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R2_RESULTS.md'
    guide_name = 'CCTV_DGP_CONDITIONING_LEARNING_V1_R3_VM.md'
    report = f'''# Conditioning learning R2 startup failure and R3 repair — 13 October 2026

**R2 stopped before model construction, inference or training. It did not reach
the structure/preservation gate.** The new worker's canonical-pixel fixture was
16×16 while its unchanged delivery helper requires 256×256. This is an
implementation bug in the prepared startup check. Earlier preparation verified
transfer, state schemas, metrics and gates, but missed exercising this fixture
against the real helper.

Location: returned `scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py:152`, calling
`conditioning_losses.py:10`. The fixture's actual shape is `[1,3,16,16]`; the
helper rejects it before the first model constructor. The traceback, failure
record and all original source bytes remain retained.

The VM failure reports **{startup['reported_worker_seconds']:.6f} seconds**, zero
candidate/recognizer forwards, zero gradient/backward calls, zero proposals and
**zero accepted updates**. Only `failure.json` exists in the run-output stage.
There is no initial model-state receipt, improved raw/PNG image, new checkpoint,
quality measurement or visual sheet. No fresh post-failure memory-state hash is
claimed. The export's `complete: true` certifies archive creation only.

All **{imported['returned_file_bindings']} returned bindings** and **5,852 parent
assets** passed the independent import checks. The {imported['archive_bytes']:,}-byte
archive SHA256 is `{imported['archive_sha256']}`. Protected app/checkpoint files
remain unchanged. {link('Independent failure audit', 'outputs/cctv_dgp_conditioning_learning_v1_r2_analysis/failure_audit.json')}.
{link('Source-bound startup review', 'outputs/cctv_dgp_conditioning_learning_v1_r2_analysis/startup_review.json')}.

## Demonstrated repair

A CPU utility regression extracts only the sealed worker's actual NumPy fixture
expression and calls the unchanged delivery helper. It reproduces R2's exact
shape assertion. The same regression passes the fresh R3 fixture at
`[1,3,256,256]`, checking all 256 byte levels and exact float32 RGB / float64
feature values. It constructs no neural model and performs no gradients or
parameter changes. This establishes the shape correction locally; it does not
establish a new CUDA run or useful restoration.
{link('Retained failing regression', 'outputs/cctv_dgp_conditioning_learning_v1_r2_analysis/pixel_fixture_red.json')}.
{link('Passing regression', 'outputs/cctv_dgp_conditioning_learning_v1_r3_pixel_fixture_green.json')}.

R3 uses a separate immutable directory and filenames. The worker differs only
in that fixture and the namespace. The model's namespace and wrapper paths are
updated; architecture, readout initialization, helper/loss/optimizer code,
initial state, cohort, scientific thresholds and resource limits are unchanged.
The failed R2 packet, return, logs, receipts and checkers remain intact.

R3 is **verified and UNRUN**. It trains only the 12 active conditioning tensors,
with at most two accepted full-cohort updates and three actual trials per update.
Each accepted change needs actual raw/PNG preservation, brightness <=0.2 and
positive detail progress globally and in both source groups. All 64 current-state
loss derivatives are refreshed after acceptance. The endpoint still needs 1%
structure gain globally and per source, on both raw and delivered PNG; cumulative
parameter distance remains 0.1%. All partial accepted components and rejected
trials are exported with true counts. No longer epochs or app adoption follow
automatically. {link('Verified manual commands', guide_name)}.

Preparation checked all 12 archive members, exact ASCII/LF checksum, all 5,852
parent hashes, initial saved-state reconstruction, 100 real saved baseline cases,
scientific gate boundaries, rejection of nonfinite metrics, all four previously
rejected proposal groups, Windows pre-neural rejection and read-only Bash syntax.
{link('Transfer and repair audit', 'outputs/cctv_dgp_conditioning_learning_v1_r3_transfer_audit.json')}.
These checks are preparation evidence; the actual new manual L4 run is pending.

The previously audited activation result remains valid. Neither that result nor
this startup repair demonstrates improved native CCTV faces or more-epoch
qualification. The original app checkpoint/design, final identity separation,
paired/unpaired boundaries, five milestones and seven automatic/assisted
completion families remain. Full TRAIN's 10% structure requirement and useful
native DEV / independent final review remain separate. All actual training is
human-manual on the existing L4. No VM cleanup, direct training launch or app
change was performed during this return audit. The full goal remains active
and incomplete.
'''
    # Retain the previously audited five-step format; exact command namespaces,
    # PIN, size and archive hash are changed to the independently sealed R3.
    old_guide_path = stage / 'before' / used if (stage / 'before' / used).exists() else ROOT / used
    original_guide = old_guide_path.read_text(encoding='utf-8')
    guide = original_guide.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
    guide = guide.replace('cctv-dgp-conditioning-learning-v1-r2', 'cctv-dgp-conditioning-learning-v1-r3')
    guide = guide.replace('Conditioning learning V1 R2', 'Conditioning learning V1 R3')
    guide = guide.replace('Use this R2 guide;', 'Use this R3 guide;')
    guide = guide.replace(read(RETURN / 'protocol.json')['name'], packet.name)
    guide = guide.replace(sha(RETURN / 'protocol.json'), pin)
    old_prep = read(OUT / 'cctv_dgp_conditioning_learning_v1_r2_preparation.json')
    guide = guide.replace(old_prep['archive_sha256'], prep['archive_sha256'])
    guide = guide.replace(f"{old_prep['archive_bytes']:,}bytes", f"{prep['archive_bytes']:,}bytes")
    guide = guide.replace('The fresh maintenance snapshot found99.93GiB free; launch checks current space.',
                          'The worker checks current free space on launch.')
    marker = '**Verified and UNRUN. This is actual training, capped at2 accepted updates.**\n'
    assert guide.count(marker) == 1
    guide = guide.replace(marker, marker + '\nR2 stopped at its 16×16 startup fixture with zero updates. R3 corrects that\nfixture to 256×256; all model settings and scientific gates are unchanged.\n' +
                          link('Audited startup failure and repair', report_name) + '.\n')
    assert 'conditioning-learning-v1-r2' not in guide and sha(RETURN / 'protocol.json') not in guide
    retained_new(ROOT / report_name, report)
    retained_new(ROOT / guide_name, guide)
    stage.mkdir(exist_ok=True)
    backup = stage / 'before'
    backup.mkdir(exist_ok=True)
    prefix = f'''# Conditioning learning R2 startup audited; corrected R3 prepared — 13 October 2026

R2 stopped before model construction/inference or training: its pixel fixture
was16×16 while the unchanged helper requires256×256. All counts are zero; no
new image, checkpoint, structure measurement or quality-gate decision exists.
The16-file return and5,852 parent assets passed hash checks. This is a startup
implementation failure, separate from earlier structure/preservation failures.
{link('Audited failure and demonstrated repair', report_name)}.

Fresh R3 changes only the fixture and immutable namespace. The real helper
regression reproduced R2's failure and passed all256 byte levels at256×256 on
CPU, without models/gradients. The architecture, exact initial state,12 selected
tensors,100 paired TRAIN cases,64 losses,1% endpoint, scientific preservation
gates and timing/resource stops are unchanged. R3 permits at most2 accepted
updates and3 trials each. It is verified and UNRUN; CUDA learning remains pending.
{link('Current manual upload, tmux and download commands', guide_name)}.

The prior activation audit remains valid. Useful restoration, longer epochs,
full TRAIN/native DEV/final qualification and all seven covering families remain
pending. Original checkpoints/app design, splits/provenance/failures and paired
versus unpaired boundaries remain. Actual training is human-manual on the L4.
No agent VM training, cleanup or app promotion occurred. All five milestones
and the full goal remain active and incomplete. Exact previous content follows.

'''
    published = {n: finish_prepend(ROOT / n, prefix, backup / n, expected[n]) for n in docs}
    banner = f'''# Returned startup failure — retain R2; use corrected R3

R2 stopped before model inference or training because its pixel fixture was16×16.
Zero updates; no structure/preservation decision or improved output was produced.
{link('Audited result', report_name)}. {link('Current manual R3 commands', guide_name)}.
The used R2 instructions below remain historical evidence; do not rerun them.

'''
    published[used] = finish_prepend(ROOT / used, banner, backup / used, expected[used])
    update = f'''# Activation remains valid; learning startup repaired — 13 October 2026

The subsequent R2 learning run stopped at a16×16 pixel fixture before any model
construction/inference, gradient or update. It did not test image quality.
{link('Audited startup failure and repair', report_name)}.
{link('Current verified, unrun R3 manual commands', guide_name)}.
The activation finding and exact previous report are retained below.

'''
    published[activation] = finish_prepend(ROOT / activation, update, backup / activation, expected[activation])
    evidence = [ANALYSIS / n for n in ['import.json', 'failure_audit.json', 'startup_review.json', 'pixel_fixture_red.json']]
    evidence += [OUT / ('cctv_dgp_conditioning_learning_v1_r3_' + n + '.json') for n in
                 ['preparation', 'checker_preparation', 'transfer_audit', 'pixel_fixture_green', 'bash_syntax']]
    write(stage / 'publication.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          publisher_sha256=sha(__file__), documents=published,
          new_documents_sha256={n: sha(ROOT / n) for n in [report_name, guide_name]},
          evidence_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in evidence},
          returned_protocol_sha256=sha(RETURN / 'protocol.json'), returned_archive_sha256=imported['archive_sha256'],
          returned_training_updates=0, finite_structure_gate_reached=False,
          next_manual_protocol_sha256=pin, next_manual_archive_sha256=prep['archive_sha256'],
          next_training_pilot_unrun=True, exact_startup_repair_only=True, no_agent_VM_training_or_cleanup=True,
          model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, startup_cause='16x16 fixture', actual_updates=0,
               current_manual_guide=guide_name, exact_histories_preserved=len(published), training_unrun=True))


if __name__ == '__main__':
    main()
