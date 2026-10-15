"""Guarded publication of audited A/B stop, negative DC probe and unrun pilot."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
ANALYSIS = OUT/'cctv_dgp_bank_comparison_v1_r4_B_analysis'
MAINT = OUT/'cctv_dgp_bank_B4_output_cleanup_v2'
ORIGINAL = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def new(path,text):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)


def link(label,name): return '['+label+'](<'+(ROOT/name).as_posix()+'>)'


def prepend_locked(path,prefix,backup):
    # A Windows share-none handle prevents another writer changing this document
    # while its exact original bytes are saved and a status prefix is applied.
    import ctypes
    import msvcrt
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    handle = api.CreateFileW(str(path),0xC0000000,0,None,3,0x80,None)
    assert handle != ctypes.c_void_p(-1).value, 'Document busy; retain staged publication: '+str(path)
    fd = msvcrt.open_osfhandle(handle,os.O_RDWR | os.O_BINARY)
    with os.fdopen(fd,'r+b') as stream:
        old = stream.read()
        with backup.open('xb') as saved: saved.write(old)
        before = hashlib.sha256(old).hexdigest()
        combined = prefix.encode('utf-8')+old
        stream.seek(0); stream.write(combined); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
    assert path.read_bytes() == combined and backup.read_bytes() == old
    return dict(before_sha256=before,after_sha256=sha(path),backup=str(backup),prefix_bytes=len(prefix.encode('utf-8')),
                old_bytes_preserved_as_exact_suffix=True)


def main():
    publication = ANALYSIS/'publication.json'; assert not publication.exists()
    evidence = {}
    paths = [ANALYSIS/name for name in ['import.json','route_import.json','execution_audit.json','independent_audit.json',
        'matched_AB_comparison.json','saved_learning_diagnostic.json','visual_review.json','preview_integrity_audit.json']]
    paths += [OUT/'cctv_dgp_bank_feature_DC_probe_v1_analysis'/name for name in ['independent_audit.json','visual_review.json']]
    paths += [OUT/name for name in ['cctv_dgp_bank_photometric_guard_v1_preparation.json',
        'cctv_dgp_bank_photometric_guard_v1_transfer_audit.json','cctv_dgp_bank_photometric_guard_v1_math_audit.json']]
    paths += [MAINT/name for name in ['independent_audit.json','inventory_after.json','post_execution.json']]
    for path in paths:
        assert read(path)['complete'], str(path)
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    audit, execution, matched = [read(ANALYSIS/name) for name in ['independent_audit.json','execution_audit.json','matched_AB_comparison.json']]
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7910 and audit['native_compositions_checked'] == 72
    assert execution['actual_optimizer_updates'] == 50 and matched['optimizer_updates_each'] == 50
    visuals = read(ANALYSIS/'visual_review.json')
    dc_visuals = read(OUT/'cctv_dgp_bank_feature_DC_probe_v1_analysis/visual_review.json')
    assert visuals['viewed_sheets'] == 26 and dc_visuals['viewed_sheets'] == 5
    assert visuals['all_planned_sheets_viewed'] and dc_visuals['all_planned_sheets_viewed']
    assert not visuals['independent_final_review'] and not dc_visuals['independent_final_review']
    for visual in [visuals,dc_visuals]:
        for name,expected in visual['evidence_files_sha256'].items(): assert sha(ROOT/name) == expected, name
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == value for name,value in protected.items())
    assert sha(OUT/'cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == ORIGINAL
    prep_path = OUT/'cctv_dgp_bank_photometric_guard_v1_preparation.json'; prep = read(prep_path)
    transfer = read(OUT/'cctv_dgp_bank_photometric_guard_v1_transfer_audit.json')
    assert transfer['preparation_sha256'] == sha(prep_path) and transfer['protocol_sha256'] == prep['protocol_sha256']
    assert transfer['archive_members_checked'] == 5644 and transfer['transfer_members_checked'] == 21
    assert transfer['local_neural_gradient_or_optimizer_calls'] == 0 and prep['manual_VM_run_pending']
    assert sha(OUT/(prep['stem']+'-code-transfer.tar.gz')) == prep['transfer_archive_sha256']
    cleanup, space = read(MAINT/'independent_audit.json'), read(MAINT/'inventory_after.json')
    assert cleanup['retained_state_exactly_equal'] and cleanup['exact_deletions_checked'] > 0
    assert cleanup['complete_local_backup_hashes_rechecked'] == cleanup['exact_deletions_checked']
    assert space['manual_guard_space_requirement_met'] and space['current_snapshot_not_future_launch_guarantee']
    gates = matched['unchanged_gate_decisions']
    lines = []
    for branch in ['A','B']:
        raw,png = [gates[branch][branch+'_'+stage] for stage in ['raw','png']]
        assert not raw['pass'] and not png['pass']
        lines.append('| '+branch+' | '+f"{100*raw['gain']:.6f}% / {100*png['gain']:.6f}%"+' | '+
            str(len(raw['failures']))+' / '+str(len(png['failures']))+' | Rejected |')
    report_name = 'CCTV_DGP_BANK_AB_COMPARISON_V1_RESULTS.md'
    guide_name = 'CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_VM.md'
    free = space['free_bytes']/1024**3
    post = space['conservative_post_install_free_bytes']/1024**3
    reclaimed = cleanup['observed_free_bytes_change']/1024**3
    report = f"""# Audited A/B restoration comparison — 11 October 2026

Both trained routes failed the retained quality requirements. B reduced the
detail proxy error and the number of group/metric failures relative to A, but
neither is a qualified replacement. The permanent app checkpoint remains
`{ORIGINAL}`. Training completion, export completion and numerical improvement
do not establish useful native CCTV restoration or hidden identity recovery.

| Route | Raw / delivered PNG detail gain | Raw / PNG overlapping failures | Decision |
|---|---:|---:|---|
{chr(10).join(lines)}

The early requirement remains **1%** detail improvement, nonnegative detail
gains in each source, all retained MSE/SSIM/recognizer preservation limits and
at most **20%** brightness-only gain contribution. B's contribution was
**40.516852% raw / 38.623819% PNG**. Its raw low-light FFHQ-label MSE worsened;
delivered PNG also worsened the Asian-source-label low-light MSE. Counts above
overlap groups and metrics; they are not counts of failed faces.

**Metric correction:** the A/B detail gate is `landmark_high_frequency_MSE`,
grayscale high-frequency error in landmark supports. Earlier dated A status
text calling this frozen-recognizer feature MSE is incorrect. The recognizer
similarity is a separate preservation measure, not evidence of recovered identity.
Historical records retain their bytes; this definition supersedes that wording.

## What actually ran and was checked

Each route ran **50 updates**, **250 distinct TRAIN references**, **125 per
source**, **1,250 profile exposures**, **zero complete epochs** (250/781 =
0.3201 epoch). B worker time was **{execution['worker_seconds']/60:.2f} minutes**;
this includes preflight/baseline/snapshot work, and fit time was not separately
recorded in that historical run. Supervisor exit **2** was its intentional
quality stop; there was no model-loading exception. Failed states are retained
for audit/migration and cannot be resumed as a qualified recipe.

The full return archive `{read(ANALYSIS/'import.json')['archive_sha256']}`
contained **7,687 regular members**. Independent checks covered **7,910 paired
raw/PNG compositions**, **72 native unpaired compositions**, strict full state,
six CPU/CUDA prior fixtures and two limited fresh CPU replay batches. All
**781 baseline RGB/embedding packs** and **24 native baselines** match A/B.
This is not a full fresh neural replay of all 3,905 cases.

All **26 A/B sheets** were displayed and observations retained: 100 paired TRAIN
previews and 24 native DEV crops. Clear and higher-information faces retain
more input texture than the original baseline's smoothing. Blur/compound
cases remain soft; dark/downcast/turned native crops remain limited. B versus
A differences are mostly subtle. No convincing whole-face clarity improvement
supports adoption. The reviewer was the implementing assistant; independent
final visual review remains required. Review evidence:
{link('A/B audit and sheet closure','outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/visual_review.json')}.

All **37 trainable tensors changed**, all received nonzero finite gradients
after update 1, and **755 frozen state tensors remained exact**. Changing-batch
objective reductions do not prove convergence. The 100-case bank-disabled
subset preserves clear detail less well but reduces some dark-image errors;
it is not a full-cohort causal bank ablation. The existing **391 high-quality
FFHQ counterpart targets** are hash bound. `native_size=128` describes the
observed source, not those prepared 256 targets. New target acquisition is not
justified by that field alone.

## Why global bank-feature centering was rejected

A separate **CPU inference-only** probe tested 100 paired TRAIN and 24 unpaired
native cases, using the frozen rejected B50 state. Removing every projected
bank-feature spatial mean reduced some low-light pixel errors, but worsened
clear-face detail error by about **26.7%**, lowered clear SSIM/recognizer
similarity and visibly softened clearer faces. All **five planned probe sheets**
were displayed. The independent counterfactual check covered 200 paired and
48 native compositions plus limited fresh model/recognizer spots. No local
gradient, optimizer update or failed-state training continuation occurred.
{link('Counterfactual audit','outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis/independent_audit.json')}
and {link('viewed probe sheets','outputs/cctv_dgp_bank_feature_DC_probe_v1_analysis/visual_review.json')}.

Constant feature planes influence spatial detail through the nonlinear decoder;
they are not a proven pure brightness channel or the sole cause of failure.
Global feature centering and global bank disabling are rejected for adoption.
The encoder–generative-bank–decoder design is informed by the primary
[GLEAN abstract](https://arxiv.org/abs/2012.00739); that paper does not validate
this custom guard, this implementation or CCTV performance.

## One finite next hypothesis, prepared but unrun

The original B training objective has no direct brightness-only-gain constraint,
although its return gate requires one. Add a **TRAIN-only observed-pixel hinge**
on the four degraded profiles, keeping bank spatial features and all original
losses. Let Gmean be positive MSE improvement from the candidate's mean-only
control, and Gtotal be positive candidate MSE improvement. The added term is
`10 × mean(max(Gmean − 0.2 × Gtotal, 0)) / retained_MSE_normalizer`.
This per-image constraint is stronger than the aggregate return criterion; it
avoids division by small/negative improvements. Coefficient 10 is a fixed finite
hypothesis at twice the existing per-metric regression-barrier scale, not a
claim of optimality. Clear profiles stay under the existing preservation losses.
The added term operates on raw float outputs; delivered PNG preservation and
the detail requirement must still pass their independent checks.

This changes **one training objective term**. Architecture, bank fusion,
initializer file, all frozen weights, optimizer, training data/exposures,
normalizers, schedule, scientific thresholds and resource caps remain exact.
Start from the **unchanged initial B checkpoint at the permanent original DGP**,
not rejected B50. No display correction, oracle inference or global DC deletion.
Local forward-only math checks covered 100 saved cases, exact baseline zero
penalty, pure brightness improvement and exclusion of unobserved pixels.
All 5,644 full-packet members and the 21-member reuse transfer passed independent
hash/source/namespace/syntax/CLI checks; eight unsafe protocol changes were
rejected. Actual CUDA gradients, runtime and useful quality remain unverified.

The experiment is **50 updates**, not extra epochs or a completed replacement.
Estimated **25–35 minutes** if runtime resembles B r4; new timing is unmeasured.
Unchanged caps: worker 90 minutes/external 90.5, fit 10, preflight 5,
snapshot 20, export 10/external 10.5, VRAM 20 GiB, output 6 GiB,
reserve 1 GiB and **16 GiB free after installation**. Record actual fit time
and the new objective contribution. Preserve any stop and full state.
After a third unsuccessful trained comparison, stop modification and ask one
diagnostic question before proposing another recipe. Extra epochs remain
conditional on independently audited preservation and useful DEV improvement.
{link('Exact manual commands',guide_name)}.

## Maintenance and remaining goal

Authorized, backup-verified maintenance removed **{cleanup['exact_deletions_checked']:,} exact
completed output/archive copies**. Observed free-space increase: **{reclaimed:.3f} GiB**;
fresh free: **{free:.3f} GiB**; conservative copied-install free: **{post:.3f} GiB**.
The launch must check again. Checkpoints, full states, raw V42 arrays, training
caches, splits, logs, instructions, provenance and gate/failure records remain
on the VM. Complete local returned evidence remains available. Independent
closure compared {cleanup['protected_metadata_entries']:,} metadata entries and
{cleanup['retained_bank_hashes']:,} retained research byte hashes. All scientific
cache bytes were not rehashed. No agent training or VM power change occurred.

Native evidence is **24 ChokePoint P1E_S1_C1 development crops**, unpaired and
without aligned clean targets. Synthetic paired TRAIN capacity and its metrics
remain separate. Source labels do not establish ethnicity or Zamboanga
performance; no real Zamboanga CCTV samples exist. Final identities remain
outside tuning. Resize/Phase3/pretrained comparison completion, useful DEV and
independent final review, primary DGP Auto/override integration, meaningful
regressions and bundled inline Playwright app verification remain required.
All seven completion families also remain required: masks, sunglasses, strong
glare, hands, obstructing hair, scarves and objects, with automatic mask preview,
optional correction, preservation of visible appearance/clear glasses/ordinary
hair, original/mask/one plausible estimate, PNG and optional bundle downloads,
and separate automatic/assisted review. This experiment tests none of those
completion families. Goal **ACTIVE**; no app adoption or completed milestone
claim. Preserve the existing app design and request clearer/less-covered crops
when usable information is insufficient.
"""
    stem,pin = prep['stem'],prep['protocol_sha256']
    guide = f"""# Brightness-guard comparison: manual L4 VM commands

**Prepared and independently transfer-verified; training is unrun.** The prior
B run loaded correctly, trained 50 updates and failed quality. This new run adds
one degraded TRAIN brightness-gain loss; it preserves bank features and starts
from the unchanged initializer. It cannot resume rejected B50. No extra epochs
are authorized by archive completion. {link('Evidence and rationale',report_name)}.

Upload only the **{prep['transfer_archive_bytes']/1024**2:.2f} MiB code transfer**.
The data-only installer copies **{prep['reused_assets']:,} hash-verified assets**
from the retained B r4 root into a distinct folder with **{prep['overlay_assets']} overlays**.
It imports no model code, runs no training, deletes no source file and has a
300-second limit. The **{prep['archive_bytes']:,}-byte full execution archive**
stays locally for recovery; these commands use the small transfer.

Current verified free: **{free:.3f} GiB**; conservative post-install: **{post:.3f} GiB**.
Require **16 GiB after installation**; installer and launch recheck. Estimate
25–35 minutes based on B r4's 23.90-minute worker plus export, not a measured
new-run ETA. Finite caps: worker 90 minutes/external 90.5, fit 10, preflight 5,
snapshot 20, export 10/external 10.5, VRAM 20 GiB, output 6 GiB, reserve 1 GiB.
50 updates, 250 distinct TRAIN references, 1,250 profiles, **zero complete epochs**.
Existing 1% structure and preservation/brightness requirements remain unchanged.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-code-transfer.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-code-transfer.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
tr -d '\\r' < {stem}-code-transfer.tar.gz.sha256 | sha256sum -c - &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/{prep['transfer_name']} &&
test ! -e ~/forensic-dgp/{prep['name']} &&
tar -xzf {stem}-code-transfer.tar.gz -C ~ &&
python3 -B ~/{prep['transfer_name']}/install_packet.py --protocol-sha {pin}
```

Expected transfer SHA256: `{prep['transfer_archive_sha256']}`.
Expected protocol SHA256: `{pin}`.
The source B r4 assets and quality stop must remain intact. The installer
does not need its removed completed RGB outputs. Preserve a partial install
for diagnosis; do not rerun a used root or historical recipe.

3. Open **tmux from the normal VM SSH shell**:

```bash
tmux new-session -A -s dgp_bank_photometric_guard_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_comparison_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_bank_comparison.sh {pin}
```

Transfer verification must print `"loader_hash_runtime_checked": true`.
The actual trainer separately checks L4/VM/tmux, source hashes, space, timing,
fixed assets, output parity and gradients before/after update 1. Detach with
**Ctrl+B**, then **D**; reattach using step 3. Wait for supervised export.
Exit 2 means quality stop; `"complete": true` in export means the archive
was produced, not that the model passed. Preserve results from any stop.

5. Download after export, from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "."
```

Each download has one remote source for Windows/PuTTY. Return all three for
independent import, full state/exposure/fit-time audits, separate raw/PNG checks,
matched B r4 comparison and every planned visual sheet. Final identities stay
outside tuning. No automatic extra epochs, model adoption or completion claim.
The app's restoration/covering-family scope remains required in the active goal.
"""
    commands = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all('--zone=us-central1-a' in line for line in commands)
    assert all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert guide.count(pin) == 4 and '\r' not in guide
    assert [int(value) for value in re.findall(r'^(\d)\. ',guide,re.M)] == [1,2,3,4,5]
    for name in ['scripts/import_cctv_dgp_bank_photometric_guard_v1_return.py','scripts/audit_cctv_dgp_bank_photometric_guard_v1_execution.py']:
        ast.parse((ROOT/name).read_text(),feature_version=(3,10)); evidence[name] = sha(ROOT/name)
    assert all(sha(ROOT/name) == value for name,value in evidence.items())
    new(ROOT/report_name,report); new(ROOT/guide_name,guide)
    header = f"""# Latest verified status — 11 October 2026: trained A/B quality stops audited; one new loss trial prepared

B r4 loaded successfully and made **50 updates**, **250 distinct TRAIN
references**, **1,250 profiles**, **zero complete epochs** (0.3201 epoch).
Raw/PNG detail gains: **0.625591% / 0.627191%**, below 1%, with **2 / 3**
overlapping group/metric failures including low-light preservation and
brightness-only contribution. A's gains were **0.471905% / 0.473663%**, with
14 / 15 failures. Neither is a qualified incumbent; both quality stops remain.
The metric is **landmark high-frequency MSE**, not frozen-recognizer feature
MSE as earlier dated A text says. ArcFace similarity is a separate measure.

Independent B return checks covered 7,910 paired raw/PNG compositions, 72
native compositions, strict full state, six prior fixtures and two limited CPU
replays. All 781 baseline packs and 24 native baselines match A/B. All 26 A/B
review sheets were viewed. All 37 trainable tensors changed and 755 frozen
state tensors stayed exact. A CPU inference-only counterfactual also passed
its saved-output audit; all five planned sheets were viewed. Global bank-feature
centering worsened clear detail by about 26.7% and is rejected for adoption.
Independent final visual review, useful DEV qualification and all seven
automatic/assisted completion families remain pending; final identities unused.

Next: one TRAIN-only brightness-gain hinge, preserving bank features,
architecture, unchanged initializer, optimizer, data/exposures, schedule, gates
and finite caps. This is a new **50-update manual VM trial**, not extra epochs
or continuation of failed B50. 100 saved-array math checks and both full/reuse
transfer audits passed. Actual CUDA gradients, timing and quality are **unrun**.
{link('Review and evidence',report_name)}.
{link('Five exact manual upload/tmux/download steps',guide_name)}.

Authorized hash-bound maintenance removed {cleanup['exact_deletions_checked']:,} backed-up
completed image/archive copies, preserving checkpoints, full states, training
caches, raw V42 arrays, splits, instructions, provenance, logs and stop/failure
records. Observed reclaim **{reclaimed:.3f} GiB**; free **{free:.3f} GiB**;
conservative copied-install free **{post:.3f} GiB**. Independently checked
{cleanup['protected_metadata_entries']:,} metadata entries and {cleanup['retained_bank_hashes']:,} retained
byte hashes; all scientific cache bytes were not rehashed. Launch rechecks
storage/workloads. No agent training, app change or VM power change occurred.

All five milestones and seven covering families remain required. Goal ACTIVE;
permanent app checkpoint `{ORIGINAL}`, existing design, scientific boundaries
and manual L4 training rule remain. Native CCTV is unpaired; synthetic TRAIN
capacity is separate. No ethnicity, Zamboanga performance or hidden-identity
claim. Earlier dated sections retain their history and do not authorize reruns.

---

"""
    backups = ANALYSIS/'publication_before'; assert not backups.exists(); backups.mkdir()
    docs = ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    mutations = {}
    for name in docs: mutations[name] = prepend_locked(ROOT/name,header,backups/name)
    old_guide = 'CCTV_DGP_BANK_COMPARISON_B_V1_R4_VM.md'
    notice = '# Executed historical B r4 recipe — do not rerun\n\nThe returned 50-update run was independently audited and rejected on quality.\nThis preserved guide records that executed recipe. Current evidence: '+link('A/B review',report_name)+'. New unrun trial: '+link('brightness-guard manual guide',guide_name)+'. Historical commands below do not authorize repeating the rejected run.\n\n---\n\n'
    mutations[old_guide] = prepend_locked(ROOT/old_guide,notice,backups/old_guide)
    assert all(sha(ROOT/name) == value for name,value in evidence.items())
    assert all(sha(ROOT/name) == value for name,value in protected.items())
    record = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256=evidence,document_mutations=mutations,new_documents_sha256={name:sha(ROOT/name) for name in [report_name,guide_name]},
        protected_local_sha256=protected,all_prior_document_bytes_retained=True,
        paired_metric_definition_corrected_without_altering_historical_records=True,
        A_B_and_DC_probe_rejected_for_adoption=True,new_guard_VM_training_pending=True,
        local_gradients_or_optimizer_updates=0,app_promotion=False,independent_final_review=False,goal_complete=False)
    new(publication,json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,report=report_name,guide=guide_name,updated_documents=len(mutations),VM_training_pending=True),flush=True)


if __name__ == '__main__': main()
