"""Publish audited quality stop, user DEV feedback and an unrun frozen probe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from publish_cctv_dgp_bank_AB_and_guard_v1_review import prepend_locked

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
ANALYSIS = OUT/'cctv_dgp_bank_photometric_guard_v1_analysis'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def new(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def link(label, path):
    return '['+label+'](<'+(ROOT/path).as_posix()+'>)'


def main():
    publication = ANALYSIS/'publication.json'
    assert not publication.exists()
    required = [ANALYSIS/name for name in ['import.json', 'route_import.json', 'execution_audit.json',
        'independent_audit.json', 'matched_guard_comparison.json', 'visual_review.json',
        'preview_integrity_audit.json', 'saved_bank_attribution.json', 'user_development_feedback.json']]
    required += [OUT/'cctv_dgp_bank_path_ablation_v1_preparation.json', OUT/'cctv_dgp_bank_path_ablation_v1_transfer_audit.json']
    assert all(read(path)['complete'] for path in required)
    audit = read(ANALYSIS/'independent_audit.json')
    execution = read(ANALYSIS/'execution_audit.json')
    matched = read(ANALYSIS/'matched_guard_comparison.json')
    attribution = read(ANALYSIS/'saved_bank_attribution.json')
    visual = read(ANALYSIS/'visual_review.json')
    prep, transfer = [read(path) for path in required[-2:]]
    assert visual['viewed_sheets'] == 26 and visual['sheet_cells_independently_verified'] == 844
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7910 and audit['native_compositions_checked'] == 72
    assert execution['actual_optimizer_updates'] == 50 and matched['unchanged_baseline_RGB_embedding_packs'] == 781
    assert transfer['preparation_sha256'] == sha(required[-2]) and transfer['protocol_sha256'] == prep['protocol_sha256']
    assert transfer['archive_members_checked'] == 3 and transfer['parent_assets_checked'] == 5852
    assert sha(OUT/(prep['stem']+'-execution.tar.gz')) == prep['archive_sha256']
    assert not (OUT/prep['name']/'outputs').exists()
    raw, png = [audit['metric_and_gate_decisions']['B_'+stage] for stage in ['raw', 'png']]
    assert not raw['pass'] and not png['pass']
    assert len(raw['failures']) == len(png['failures']) == 4
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == expected for name,expected in protected.items())
    report_name = 'CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_RESULTS.md'
    guide_name = 'CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md'
    pin, stem = prep['protocol_sha256'], prep['stem']
    group = attribution['groups']['all|degraded']['stages']['raw']
    enabled_gain, disabled_gain = [100*group['relative_detail_gain_from_original'][label] for label in ['B50', 'B50_bank_disabled']]
    report = f"""# Brightness-guard return and next diagnostic — 11 October 2026

The downloaded experiment trained correctly and stopped on the retained quality
requirements after **50 updates**. It is not a qualified replacement. The user
judged the fifth column on the first fixed native sheet **much better than the
original DGP** and requested the best next approach. That development feedback
is retained separately from numerical gates and independent final review.

| Candidate | Raw / PNG landmark-detail error reduction | Raw / PNG overlapping group-metric failures |
|---|---:|---:|
| Earlier B r4 | 0.625591% / 0.627191% | 2 / 3 |
| Brightness guard | {100*raw['gain']:.6f}% / {100*png['gain']:.6f}% | 4 / 4 |

The early detail requirement is **1%**. Brightness-only contribution is
**{100*raw['brightness']:.6f}% raw / {100*png['brightness']:.6f}% PNG**, above
the retained **20%** maximum. Both source-label blur groups regress in MSE;
the Asian-source-label compound group regresses in observed ArcFace similarity.
The older low-light MSE failures are absent, but the new tradeoffs prevent a pass.
Failure counts overlap groups and metrics; they are not counts of failed people.
This is an intentional supervisor quality exit **2**, not a loading exception.

The structure score is **landmark_high_frequency_MSE**, a paired photographic
TRAIN proxy. It is not recognizer feature MSE or a percentage of recovered
identity. Recognition similarity is a separate preservation measure. The 24
native ChokePoint DEV crops have no aligned clean targets and no native PSNR/SSIM.
Source labels do not establish ethnicity or Zamboanga performance.

## Execution and independent checks

The L4 made 50 optimizer updates using 250 distinct TRAIN references, 125 per
source, and 1,250 profile exposures: **zero complete epochs**, 0.3201 of the
781-reference pool. Actual fitting took **{execution['actual_fit_seconds']/60:.2f} minutes**;
the **{execution['worker_seconds']/60:.2f}-minute worker** also included baseline,
preflight and snapshot work. Archive export took about **4.20 minutes**.
This does not establish the original checkpoint's complete ancestral epoch count
or authorize the conditional additional-epoch proposal.

The {5_217_289_325:,}-byte returned archive has SHA256
`f2ff55481a68d1690a3f7463384cd337c4dad238fcf8b8a600362bed54e2d356`.
Import verified 7,702 regular members. Independent checks covered **7,910 paired
raw/PNG compositions**, **72 native compositions**, strict full state, six
CPU/CUDA prior fixtures and two limited fresh CPU replay batches. This is not
a full fresh neural replay of every saved case. All **781 baseline RGB/embedding
packs** and 24 native baselines match B r4. All **37 trainable tensors changed**;
755 frozen state tensors stayed exact. Full model/optimizer/scheduler/RNG state
is retained; failed-state training continuation stays prohibited.

The new guard contribution was positive in **48 of 50 updates**. Independently
recomputed final per-image active hinges fell from **2,184 to 1,717 of 3,124**
degraded cases. It changed learning pressure, but did not enforce every final
preservation and brightness requirement. Changing-batch objectives do not prove
plateau, convergence or a unique cause of failure. Exposed and unexposed members
of this same TRAIN pool remain TRAIN evidence, not independent DEV identities.

## Visible outputs and the selected next approach

All **26 sheets** were displayed at their original resolution and all **844
source cells** were independently checked. They cover 100 paired TRAIN previews
and all 24 fixed native DEV inputs. Clearer native faces retain hair, glasses and
visible facial texture better than the original baseline's smoothing. Darker or
turned cases remain soft around eyes, nose and mouth. Differences from B r4 are
mostly subtle; the user's statement compares the guard against the original,
not against B r4. The review was by the implementing assistant and the user on
the first native sheet; independent final review remains pending. Input-only
usable labels are retained. Model softness cannot relabel a usable input as
insufficient information.

On the fixed 100-case bank ablation, **80 degraded TRAIN cases** have raw detail
gains of **{enabled_gain:.6f}% enabled / {disabled_gain:.6f}% disabled**. Disabling
the bank also reduces pooled degraded MSE, but worsens clear MSE, SSIM, similarity
and detail relative to the enabled model. Some individual profile effects differ
from the pooled result. Only nine of the twenty preview references were exposed
in the 50-update schedule. This is the same rejected jointly trained B50 decoder
with its bank switched off, not an independently trained decoder control or a
full 3,905-case bank-disabled gate. Global bank disabling is not selected.

**Next: a frozen feature-path diagnostic before another training recipe.** Keep
the saved weights fixed; compare enabled, disabled, and dropping one of the five
16/32/64/128/256 readouts. Each uses the same 100 TRAIN cases and 24 native DEV
crops. It must prove zero-readout parity and exact same-L4 enabled/disabled saved
output and embedding replay. Seven variants mean 700 paired and 168 native
compositions. No gradient, backward, optimizer update, new checkpoint, epoch,
inference strength selection or app promotion occurs. One-path removal measures
finite responses of this checkpoint; it cannot prove the capacity of a separately
trained architecture or select a deployable model by itself.

GLEAN studies how multiresolution encoder, bank and decoder features affect
restoration. That supports inspecting the contribution of each bank path rather
than assuming that a frozen generative prior automatically helps every input.
This is a design inference from its ablations, not a reproduction or evidence
of our CCTV performance. [Primary GLEAN paper, section 5](https://arxiv.org/html/2012.00739).
The prior DC-centering probe remains rejected; this diagnostic does not repeat
global feature centering or alter a scientific gate. No new clean-target
acquisition is justified by a 128-source-size metadata field: the existing 391
HQ FFHQ counterpart targets remain bound in the audited packet.

After the three unsuccessful A/B/guard comparisons, the AGENTS.md diagnostic
question was asked. The user chose the visibly better fifth column and the best
next approach. That resolves the design discussion for this bounded diagnostic;
it does not waive preservation or authorize unchanged failed training.

## Prepared manual diagnostic and remaining goal

The independently checked transfer is **{prep['archive_bytes']:,} bytes**, with
three members and **5,852 hash-bound parent assets**. No large weight/data copy
is needed. Source/data/archive checks, Python 3.10-compatible hashing, readonly
Bash syntax and actual Windows rejection before neural imports passed. The first
Bash check was blocked by the Windows sandbox signal pipe; its log is retained
and the same readonly check passed outside the sandbox. This was an audit-runtime
issue, not a training or model-source change.

Actual CUDA diagnostic parity, timing, outputs and visual review are **unrun**.
Require **4 GiB free after installation**, an idle L4 and manual tmux. Estimate
**5–15 minutes inference plus 1–5 minutes export**; this new timing is unmeasured.
Finite limits: worker 20 minutes/external 20.5, inference 15, export 5/external
5.5, 30-second kill grace, VRAM 20 GiB, output 1 GiB, disk reserve 1 GiB,
312 B forwards and 140 recognizer forwards. Preserve any stop. No agent VM
connection, training, gradient work, cleanup or VM power change occurred in
this return audit/preparation turn. Old free-space snapshots are not current.

{link('Five exact manual upload/tmux/download steps', guide_name)}.
{link('Numerical and state audit', 'outputs/cctv_dgp_bank_photometric_guard_v1_analysis/independent_audit.json')};
{link('Matched trial comparison', 'outputs/cctv_dgp_bank_photometric_guard_v1_analysis/matched_guard_comparison.json')};
{link('Visual closure', 'outputs/cctv_dgp_bank_photometric_guard_v1_analysis/visual_review.json')};
{link('Saved bank attribution and user feedback', 'outputs/cctv_dgp_bank_photometric_guard_v1_analysis/saved_bank_attribution.json')}.

The permanent app checkpoint remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The custom replacement trains our reconstruction and conditioning while declaring
its external frozen StyleGAN2 bank; it is not an own-trained StyleGAN generator
or an exact published-DGP implementation. Preserve app design, source/split
roles, terms, caches, all historical failures and original states. Pretrained
restorers remain comparators. No real Zamboanga samples exist; final identities
remain outside tuning. All five milestones stay required, including identical
resize/Phase3/pretrained comparisons, useful DEV and independent final review,
qualified primary DGP Auto/override integration, meaningful regressions and
bundled inline Playwright app checks.

Completion remains separate and must cover masks, sunglasses, strong glare,
hands, obstructing hair, scarves and objects, with automatic removal-area
preview/correction, preserved clear glasses/ordinary hair/visible appearance,
original/mask/one plausible estimate, PNG and optional bundle downloads, clearer
or less-covered requests and separate automatic/assisted review. This experiment
qualifies none of those covering families. Goal **ACTIVE and incomplete**.
"""
    guide = f"""# Frozen bank-path diagnostic: manual VM commands

**Prepared and independently transfer-verified; actual VM inference is unrun.**
This measures the existing rejected B50 state with fixed weights. It performs
zero training/gradient updates and cannot promote or resume that failed recipe.
{link('Audited result and diagnostic rationale', report_name)}.

Upload only **{prep['archive_bytes']:,} bytes**. The diagnostic reads **5,852
hash-bound files** from `~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm`;
preserve that root, its state and saved outputs. It copies no large research asset.
Require **4 GiB free after installation** and an idle L4. No current VM storage
or power state is claimed. Estimate **5–15 minutes inference + 1–5 minutes export**;
new timing is unmeasured. Worker/inference/export stops are 20/15/5 minutes,
external worker/export caps 20.5/5.5, kill grace 30 seconds, VRAM 20 GiB,
output 1 GiB and disk reserve 1 GiB. No automatic follow-on training.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\\r' < {stem}-execution.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{prep['name']} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

Expected archive SHA256: `{prep['archive_sha256']}`.
Expected protocol SHA256: `{pin}`.
Installation creates one small new folder; it deletes nothing. A launch preflight
checks free space, idle GPU and exact retained-parent hashes before diagnostic logs.
Preserve any failure; a used diagnostic root cannot run again unchanged.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_path_ablation_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_path_ablation_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_paths.sh {pin}
```

Expected transfer receipt: `complete: true`, `packet_assets: 2`,
`parent_assets: 5852`, `neural_calls: 0`, `optimizer_updates: 0`.
The run prints one line for each of seven variants. Detach safely with
**Ctrl+B, then D**. Reattach using `tmux attach -t dgp_bank_path_ablation_v1`.
An export receipt with `complete: true` verifies packaging only; quality and
independent return review remain separate. Preserve logs if any control fails.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "."
```

Keep the three files together. Returned outputs will be hash checked, independently
recomputed and visually reviewed before a new training recipe or app decision.
Native CCTV remains unpaired; paired photographic TRAIN measurements, reserved
final identities and all seven automatic/assisted completion families stay separate.
"""
    new(ROOT/report_name, report); new(ROOT/guide_name, guide)
    prefix = f"""# Latest verified status — 11 October 2026: brightness-guard quality stop audited; frozen path diagnostic prepared

The returned brightness-guard B model trained **50 updates**, 250 distinct TRAIN
references and 1,250 profiles, **zero complete epochs** (0.3201 of this pool).
Raw/PNG landmark-detail gains are **{100*raw['gain']:.6f}% / {100*png['gain']:.6f}%**,
below 1%, with **4 / 4 overlapping group-metric failures**. Brightness-only
contribution remains **{100*raw['brightness']:.6f}% / {100*png['brightness']:.6f}%**,
above 20%. Blur MSE and one compound recognizer group regress. All stops remain;
no new incumbent, extra epochs or failed-state training continuation is qualified.

Independent checks cover 7,910 paired raw/PNG compositions, 72 native compositions,
full state, six prior fixtures and two limited CPU replays. All 781 baseline packs
match B r4. All 37 trainable tensors changed; 755 frozen tensors stayed exact.
All 26 sheets were viewed and 844 cells checked. The user judged the fifth column
on the first native sheet much better than original DGP and requested the best
next approach. This is DEV feedback, not a B-r4 win or independent final review.
Input-only usability labels remain; model softness cannot exclude usable inputs.

The three-attempt diagnostic discussion is answered. Select a **frozen bank-path
ablation before further training**: seven fixed variants on the same 100 TRAIN
and 24 native DEV cases. It has zero gradients, updates and epochs, no weight or
gate changes, and no global DC centering or disabled-model adoption. The bank
helps clear preservation but shows mixed degraded effects in the limited saved
ablation. That motivates measuring each readout, not repeating an unchanged pilot.

The **{prep['archive_bytes']:,}-byte transfer** and 5,852 retained-parent hashes
passed source/data/archive, Python310 hashing, readonly Bash syntax and Windows
pre-neural rejection checks. Actual CUDA parity, runtime and diagnostic quality
are **unrun**. Manual tmux only; require 4 GiB free, worker/inference/export
20/15/5-minute caps, VRAM 20 GiB, output/reserve 1 GiB each. No agent VM model,
training, gradient, cleanup or power action occurred in this turn.
{link('Audited result and selected diagnosis', report_name)}.
{link('Five exact manual transfer/tmux/download steps', guide_name)}.

Goal ACTIVE; permanent app checkpoint `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`,
existing app design, all five milestones and seven covering families remain.
Native CCTV is unpaired; paired TRAIN evidence is separate. Final identities
remain unused; no ethnicity, Zamboanga performance or hidden-identity claim.
Conditional extra epochs require preserved useful development evidence.
Historical dated sections below retain exact bytes and authorize no reruns.

---

"""
    backup = ANALYSIS/'publication_before'; backup.mkdir()
    changes = {}
    for name in ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        changes[name] = prepend_locked(ROOT/name, prefix, backup/name)
    historical_name = 'CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_VM.md'
    historical_prefix = f"""# Executed and audited — 11 October 2026: do not rerun this historical recipe

This transfer has now been executed. Its 50-update candidate failed the retained
quality requirements and is not eligible for extra epochs or unchanged retry.
The prepared/unrun wording and launch commands below are historical.
{link('Audited return and user feedback', report_name)}.
The current next action is the **unrun frozen-inference diagnostic**:
{link('Exact new manual commands', guide_name)}.
Original guide bytes are preserved as the suffix below and in the publication backup.

---

"""
    changes[historical_name] = prepend_locked(ROOT/historical_name, historical_prefix, backup/historical_name)
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), publisher_sha256=sha(Path(__file__)),
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in required},
        created_documents={name:sha(ROOT/name) for name in [report_name, guide_name]}, canonical_prefix_changes=changes,
        protected_app_and_checkpoint_sha256=protected, model_qualified=False, app_promotion=False,
        manual_VM_frozen_diagnostic_pending=True, local_training_or_gradient_calls=0,
        all_five_milestones_and_seven_covering_families_retained=True, goal_complete=False)
    new(publication, json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, report=report_name, guide=guide_name, canonical_documents=5, goal_complete=False))


if __name__ == '__main__':
    main()
