"""Publish evidence-bound B2 failure, B3 preparation and completed maintenance status."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis'
MAINT = ROOT / 'outputs/cctv_dgp_bank_B_return_offload_v1'
PIN = 'f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35'
STEM = 'cctv-dgp-bank-comparison-v1-r3-B'
ORIGINAL = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def link(label, path):
    return '[' + label + '](<' + (ROOT / path).as_posix() + '>)'


def new_text(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    receipt = ANALYSIS / 'publication.json'
    assert not receipt.exists()
    backup_dir = ANALYSIS / 'documents_before'
    before = read(backup_dir / 'manifest.json')['files_sha256']
    for name, expected in before.items():
        assert sha(ROOT / name) == sha(backup_dir / name) == expected, name
    evidence = {}
    names = ['import.json', 'route_import.json', 'independent_audit.json',
             'failure_execution_audit.json', 'CPU_initializer_inspection.json',
             'pinned_loader_regression.json', 'visual_review.json']
    records = {}
    for name in names:
        path = ANALYSIS / name
        records[name] = read(path)
        assert records[name]['complete']
        evidence[str(path.relative_to(ROOT).as_posix())] = sha(path)
    quality = records['independent_audit.json']
    execution = records['failure_execution_audit.json']
    visual = records['visual_review.json']
    assert quality['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 0
    assert quality['saved_paired_raw_PNG_cases_checked'] == 3905
    assert quality['native_compositions_checked'] == 24
    assert execution['failure_message'] == 'Fresh initializer differs'
    assert execution['B_initializer_tensor_differences_not_saved'] and execution['initial_load_occurs_after_failed_check']
    assert visual['viewed_sheets'] == 26 and not visual['B_quality_tested']
    assert visual['baseline_rgb_files_exactly_equal_to_A'] == 929
    assert records['CPU_initializer_inspection.json']['fresh_CPU_equals_pinned']
    assert records['pinned_loader_regression.json']['cases_checked'] == 5
    preparation_path = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_preparation.json'
    transfer_path = ROOT / 'outputs/cctv_dgp_bank_comparison_B3_transfer_v1_audit.json'
    preparation, transfer = read(preparation_path), read(transfer_path)
    assert preparation['complete'] and transfer['complete']
    assert preparation['protocol_sha256'] == transfer['protocol_sha256'] == PIN
    assert transfer['preparation_sha256'] == sha(preparation_path)
    assert transfer['archive_members_checked'] == 5623
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == transfer['archive_sha256'] == preparation['archive_sha256']
    assert archive.stat().st_size == preparation['archive_bytes'] == 811541289
    assert sha(ROOT / 'outputs' / preparation['name'] / 'protocol.json') == PIN
    cleanup, space = read(MAINT / 'independent_audit.json'), read(MAINT / 'inventory_after.json')
    assert cleanup['complete'] and cleanup['exact_deletions_checked'] == 932
    assert cleanup['complete_local_backup_hashes_rechecked'] == 932
    assert cleanup['retained_state_exactly_equal'] and not cleanup['all_scientific_cache_bytes_rehashed']
    assert read(MAINT / 'post_execution.json')['complete']
    assert space['manual_B_space_requirement_met']
    assert space['current_snapshot_not_future_launch_guarantee']
    assert cleanup['model_gradient_or_training_calls'] == 0
    for path in [preparation_path, transfer_path, MAINT / 'independent_audit.json',
                 MAINT / 'inventory_after.json', MAINT / 'post_execution.json']:
        evidence[str(path.relative_to(ROOT).as_posix())] = sha(path)
    protected = read(MAINT / 'local_protected_sha256.json')
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == ORIGINAL
    a_analysis = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis'
    a_execution, a_quality = read(a_analysis / 'execution_audit.json'), read(a_analysis / 'independent_audit.json')
    assert a_execution['complete'] and a_execution['actual_optimizer_updates'] == 50
    assert a_quality['complete'] and not a_quality['model_qualified']
    for name in ['execution_audit.json', 'independent_audit.json', 'visual_review.json']:
        path = a_analysis / name
        evidence[str(path.relative_to(ROOT).as_posix())] = sha(path)
    guide = ROOT / 'CCTV_DGP_BANK_COMPARISON_B_V1_R3_VM.md'
    guide_text = guide.read_text(encoding='utf-8')
    commands = [line for line in guide_text.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all('--zone=us-central1-a' in line for line in commands)
    assert all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert [int(n) for n in re.findall(r'(?m)^([1-5])\. ', guide_text)] == [1, 2, 3, 4, 5]
    assert '--protocol-sha ' + PIN + ' --verify-transfer' in guide_text
    assert 'bash scripts/run_bank_comparison.sh ' + PIN in guide_text
    assert "tr -d '\\r' < " + STEM + '-execution.tar.gz.sha256 | sha256sum -c -' in guide_text
    assert preparation['archive_sha256'] in guide_text
    wrapper = ROOT / 'scripts/import_cctv_dgp_bank_B3_return_v1.py'
    tree = ast.parse(wrapper.read_text(encoding='utf-8'))
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)}
    assert values['PIN'] == PIN and values['STEM'] == STEM
    assert values['ARCHIVE_SHA'] == preparation['archive_sha256']
    assert values['NAME'] == preparation['name']
    old_bindings = records['route_import.json']['source_bindings_unchanged']
    assert all(sha(Path(path)) == expected for path, expected in old_bindings.items())
    evidence[str(guide.relative_to(ROOT).as_posix())] = sha(guide)
    evidence[str(wrapper.relative_to(ROOT).as_posix())] = sha(wrapper)
    report = ROOT / 'CCTV_DGP_BANK_B_R2_RETURN_REVIEW.md'
    assert not report.exists()
    recovered = cleanup['measured_recovered_bytes'] / 1024**3
    allocated = cleanup['allocated_reclaim_bytes'] / 1024**3
    fresh = space['free_GiB']
    installed = space['conservative_B_post_install_GiB']
    utc = datetime.now(timezone.utc).isoformat()
    guide_link = link('five manual transfer, tmux and download steps', guide.name)
    report_text = f'''# Bank B r2 returned evidence and B r3 loading repair - 11 October 2026

Status: B r2 is an independently audited **initialization execution failure**,
not a trained B quality rejection. B r3 is a new verified transfer packet;
its manual VM execution, gradients, learning and image quality remain pending.
The complete comparative restoration/completion goal remains ACTIVE.

## What the B r2 return proves

The downloaded archive is 2,925,496,812 bytes, SHA256
`b811a9433f26aa23bf0579dd078c96c5410c21f0d3bb1d77daa23920b30425e8`.
The safe importer checked all 6,554 regular members and the prepared assets.
Independent audit covered 6,553 manifest bindings, all 3,905 paired baseline
raw/PNG records and 24 native unpaired compositions. No B capacity gate,
trained output, bank-disabled output, CUDA prior fixture, B gradient or B
full-state check was possible: those artifacts were never produced.

The worker stopped after {execution['worker_seconds']:.3f} seconds
({execution['worker_seconds']/60:.2f} minutes). Counts were original restorer
805, fixed recognizer 781, B forwards 0, prior fixture 0, gradient queries 0,
backwards 0 and optimizer updates 0. Successful export packaged that failure;
it did not make the run a training success.

All 26 planned sheets were viewed at original resolution: 20 paired TRAIN
sheets covering 100 previews, and six sheets covering 24 ChokePoint DEV crops.
The 372 missing candidate tiles reflect the pre-training stop and A outputs
being stored in A's separate return. Clear facial texture, eye/teeth edges,
ordinary hair and clear glasses often soften in the retained baseline.
Blurred, dark and compound inputs retain indistinct central features.
No completion estimate or new B improvement is demonstrated.

All 929 baseline RGB files were physically rehashed and are byte-identical
to the independently audited A return. Protocol inputs, selection, schedule,
normalizers and thresholds match. Training exposures do not match: A performed
50 updates; B performed zero. A's retained raw/PNG structure gains remain
0.471905% / 0.473663%, below 1%, with 14 / 15 overlapping group/metric
preservation failures. This is not a completed trained A/B comparison.
No reserved final identity was used. Implementing-assistant sheet review is
distinct from the required independent final review.

## Failure location, cause and repair

Location: the `create()` initialization helper, before its strict initializer
load, raised `Fresh initializer differs`. The old ordering compared a freshly
constructed model to a saved initializer before loading that authoritative
state. B r2 did not save per-tensor differences, so the exact changed tensors
and the runtime-specific cause are unknown.

Local CPU reconstruction exactly reproduces the saved B initializer, state
SHA256 `98675d10be6f3405dd4bd0950d9251a61b26b9b161ae62a9ce48f775bc43e2bd`.
Local Torch is 2.13.0+cpu; the VM recorded 2.9.1+cu129. PyTorch documents that
seeded reproducibility is not guaranteed across releases/platforms or CPU/GPU.
This supports inspecting constructor differences; it does **not** prove why
the specific VM tensors differed. See
[official reproducibility documentation](https://docs.pytorch.org/docs/2.9/notes/randomness.html).

B r3 verifies the same initializer file hash, exact keys, shapes, dtypes and
finite values; records constructor tensor hashes; rejects any difference in
retained/frozen, mean-style or zero-fusion tensors; loads the unchanged saved
initializer strictly; and requires every loaded tensor and whole state to
match exactly. Only freshly generated new style-conditioning tensors can
differ before the load. The saved initializer remains identical. See
[official strict state loading documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html).

The loading check now runs before the long baseline snapshot. Each attempt
saves `initializer_load_B_*.json`, including differences if it stops. Five
CPU checks passed: exact load; restoration of a deliberately perturbed new
style tensor; rejection of a frozen-prior change; rejection of a zero-fusion
change; rejection of a wrong initializer hash. The synthetic perturbation
does not diagnose B r2's missing VM tensor evidence. No local neural forward,
gradient, backward or optimizer update was used for these loading checks.

## Verified B r3 transfer and scientific limits

Protocol SHA256: `{PIN}`.
Execution archive: `{STEM}-execution.tar.gz`, 811,541,289 bytes.
Archive SHA256: `{preparation['archive_sha256']}`.

Independent transfer audit checked all 5,623 tar members, the exact LF checksum,
the actual transfer-only CLI, unchanged model-computation AST, 11 unchanged
worker functions, unchanged initializer/data/schedule/objective/scientific
gates/resource caps, and rejection of six unsafe protocol changes. The repair
changes loading, diagnostic ordering and the immutable root, not the recipe.
The incomplete first local packaging attempt is retained separately; it ran
no VM/model/training code. The old B r2 protocol and failure remain preserved.

The intended matched run is 50 updates, 250 distinct TRAIN references
(125 per source), 1,250 profile exposures, zero complete epochs / 0.3201 epoch.
The bank remains an external frozen StyleGAN2 component. Our input conditioning
and reconstruction must learn and demonstrate useful preserved 256 restoration;
the bank itself is not an own-trained generator. CUDA parity, all 37 learning
tensors, bank ablation, saved full state, delivered PNG preservation, native
usefulness and final generalization remain unverified for B r3.

All finite limits remain: 16 GiB free after installation, 1 GiB reserve,
6 GiB output, 20 GiB VRAM, 90-minute worker, 90.5-minute external worker,
10-minute fit, five-minute preflight, 20-minute snapshot, 10-minute export
and 10.5-minute external export. No immutable historical run or rejected state
is automatically resumed. Follow the {guide_link}.

## Authorized storage maintenance, independently closed

Exactly 932 inactive files were removed after full local hash-bound backups:
929 B baseline RGB output files and three completed/superseded home archive
copies. Allocated reclaim: {allocated:.3f} GiB; measured recovered space:
{recovered:.3f} GiB. Fresh VM free space: {fresh:.3f} GiB; conservative
B r3 post-upload/installation estimate: {installed:.3f} GiB. The unchanged
16 GiB requirement is met at this snapshot and must be checked again at launch.

The independent deletion ledger checked all 932 paths, regular/single-link
ownership and local backup hashes. Before/after records match for
{cleanup['retained_metadata_entries']:,} retained research metadata entries and
{cleanup['retained_bank_byte_hashes']:,} retained B byte hashes. Checkpoints,
full states, training/reference caches, inputs/splits, logs, measurements,
instructions, provenance and gate/failure evidence remain. All scientific
cache bytes were **not** rehashed; do not turn metadata preservation into a
claim of complete byte-level cache verification. Two post-export log tails
were also saved locally without replacing the original returned archive.

No VM was started/stopped/switched and no training/gradient task was launched
by the agent. Local application bindings and the permanent checkpoint SHA256
`{ORIGINAL}` are unchanged. No new model is adopted.

## Evidence and next decision

Evidence is bound in
{link('B r2 analysis', 'outputs/cctv_dgp_bank_comparison_v1_r2_B_analysis/publication.json')},
{link('B r3 transfer audit', 'outputs/cctv_dgp_bank_comparison_B3_transfer_v1_audit.json')} and
{link('maintenance closure', 'outputs/cctv_dgp_bank_B_return_offload_v1/independent_audit.json')}.
The future B r3 importer is
{link('import_cctv_dgp_bank_B3_return_v1.py', 'scripts/import_cctv_dgp_bank_B3_return_v1.py')}.
It has a checked CLI/contract and will run only when its three result files
are returned. Existing B r2 import/checker sources remain unchanged.

After manual execution, independently audit actual initializer differences,
counts/stops, full states where present, all saved raw/PNG records, native
unpaired compositions, bank ablation and every planned visual sheet. Compare
trained A/B routes only when B completes its intended exposures. A loading
repair or successful export cannot approve extra epochs or app promotion.
The five milestones, separate seven-family automatic/assisted completion,
independent final review and bundled inline Playwright app flow remain required.
'''
    prefix = f'''# Latest verified status - 11 October 2026: B initialization failure audited; B r3 manual run prepared

B r2 stopped with `Fresh initializer differs` before its first B forward,
gradient or optimizer update. Independent return checks covered 3,905 paired
baseline raw/PNG records and 24 native unpaired compositions. All 26 sheets
were viewed; 929 baseline files are byte-identical to A. A made 50 updates
and failed quality; B made zero, so a trained matched comparison is pending.

The B r3 loading repair uses the unchanged hash-pinned initializer and records
constructor differences before the long baseline pass. Five CPU loader checks
and the independent 5,623-member transfer audit passed. The specific B r2 VM
tensor differences are unknown. B r3 CUDA/gradient/training/quality is unrun;
model computation, data, schedule, preservation gates and finite caps remain.

Authorized backup-verified cleanup removed 929 inactive B RGB files and three
home archive copies. Reclaimed allocation: {allocated:.3f} GiB; measured:
{recovered:.3f} GiB. Fresh VM free: {fresh:.3f} GiB; conservative B r3
post-install free: {installed:.3f} GiB, meeting 16 GiB at this snapshot.
Independent closure matched {cleanup['retained_metadata_entries']:,} metadata
entries and {cleanup['retained_bank_byte_hashes']:,} retained B byte hashes.
All scientific cache bytes were not rehashed. Checkpoints, states, caches,
splits, instructions, provenance, measurements and failure/gate records remain.

Next: {guide_link}.
Review: {link('B r2 evidence and B r3 limits', report.name)}.
The permanent app checkpoint and design remain unchanged. Training stays
manual in tmux on the existing L4. All five milestones and seven completion
families remain required; goal ACTIVE, no candidate adoption or extra epochs.
Earlier dated sections below are preserved history and do not override this
status or authorize rerunning B r2.

---

'''
    new_text(report, report_text)
    changed = {}
    for name in ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md',
                 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
        old = (backup_dir / name).read_bytes()
        assert sha(ROOT / name) == before[name]
        (ROOT / name).write_bytes(prefix.encode('utf-8') + old)
        assert (ROOT / name).read_bytes().endswith(old)
        changed[name] = dict(before_sha256=before[name], after_sha256=sha(ROOT / name), original_body_retained=True)
    name = 'CCTV_DGP_BANK_COMPARISON_B_V1_VM.md'
    old = (backup_dir / name).read_bytes()
    replacement = f'''# Superseded B r2 commands - 11 October 2026

B r2 has run and stopped before training with `Fresh initializer differs`.
Its downloaded failure is independently audited and retained. Use the
{guide_link} for the new B r3 root. Do not rerun the historical commands
below or resume its failed state. Latest evidence: {link('B r2 return review', report.name)}.

---

'''
    assert sha(ROOT / name) == before[name]
    (ROOT / name).write_bytes(replacement.encode('utf-8') + old)
    assert (ROOT / name).read_bytes().endswith(old)
    changed[name] = dict(before_sha256=before[name], after_sha256=sha(ROOT / name), original_body_retained=True)
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    assert all(sha(ROOT / name) == expected for name, expected in evidence.items())
    result = dict(complete=True, UTC=utc, publisher_sha256=sha(__file__),
                  evidence_files_sha256=evidence, report_sha256=sha(report),
                  before_manifest_sha256=sha(backup_dir / 'manifest.json'), documents=changed,
                  guide_exact_commands_and_pins_checked=True, B3_return_wrapper_AST_checked=True,
                  B3_return_import_execution_pending=True, baseline_optimizer_updates=0,
                  B3_VM_execution_pending=True, local_protected_bindings_unchanged=True,
                  original_checkpoint_sha256=ORIGINAL, model_gradient_or_optimizer_calls=0,
                  app_changed=False, model_qualified=False, goal_complete=False)
    new_text(receipt, json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, documents_updated=len(changed), B3_manual_pending=True,
               free_GiB=fresh, conservative_B3_post_install_GiB=installed, goal_complete=False), flush=True)


if __name__ == '__main__':
    main()
