"""Publish the independently audited B3 failure and pending B4 manual transfer."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis'
MAINT = ROOT / 'outputs/cctv_dgp_bank_B3_archive_cleanup_v1'
PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'
STEM = 'cctv-dgp-bank-comparison-v1-r4-B'
ORIGINAL = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_text(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def link(label, name):
    return '[' + label + '](<' + (ROOT / name).as_posix() + '>)'


def main():
    publication = ANALYSIS / 'publication.json'
    assert not publication.exists()
    records, evidence = {}, {}
    for name in ['import.json', 'route_import.json', 'independent_audit.json',
                 'failure_execution_audit.json', 'hash_compat_loader_regression.json']:
        path = ANALYSIS / name
        records[name] = read(path)
        assert records[name]['complete']
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    audit, execution = records['independent_audit.json'], records['failure_execution_audit.json']
    regression = records['hash_compat_loader_regression.json']
    assert records['import.json']['returned_file_bindings'] == 5631
    assert audit['saved_paired_raw_PNG_cases_checked'] == audit['native_compositions_checked'] == 0
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 0
    assert not any(execution['counts'].values())
    assert execution['recorded_VM_python'] == '3.10.12'
    assert execution['failure_message'] == "module 'hashlib' has no attribute 'file_digest'"
    assert execution['missing_hash_API_reproduced'] and execution['original_B2_constructor_difference_still_untested']
    assert regression['cases_checked'] == 5 and regression['every_loaded_tensor_bit_exact']
    assert regression['hash_API_absence_simulation'] and not regression['actual_local_Python310_interpreter_used']
    assert regression['local_neural_forwards'] == regression['local_gradient_queries'] == regression['local_optimizer_updates'] == 0

    preparation_path = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r4_B_preparation.json'
    transfer_path = ROOT / 'outputs/cctv_dgp_bank_comparison_B4_transfer_v1_audit.json'
    preparation, transfer = read(preparation_path), read(transfer_path)
    assert preparation['complete'] and transfer['complete']
    assert preparation['protocol_sha256'] == transfer['protocol_sha256'] == PIN
    assert transfer['preparation_sha256'] == sha(preparation_path)
    assert transfer['archive_members_checked'] == 5630 and transfer['transfer_members_checked'] == 15
    assert transfer['exactly_reconstructed_reused_assets'] == 5618 and transfer['overlay_assets'] == 11
    assert transfer['full_and_reuse_transfers_match_exact_packet']
    assert transfer['real_transfer_CLI_passed_with_loader_file_digest_absent']
    assert transfer['exact_initializer_weights_data_schedule_scientific_gates_and_resource_caps']
    for suffix, field in [('-execution.tar.gz', 'archive_sha256'),
                          ('-code-transfer.tar.gz', 'transfer_archive_sha256')]:
        path = ROOT / 'outputs' / (STEM + suffix)
        assert sha(path) == preparation[field] == transfer[field]
    assert sha(ROOT / 'outputs' / preparation['name'] / 'protocol.json') == PIN

    cleanup, space = read(MAINT / 'independent_audit.json'), read(MAINT / 'inventory_after.json')
    assert cleanup['complete'] and cleanup['exact_deletions_checked'] == cleanup['complete_local_backup_hashes_rechecked'] == 2
    assert cleanup['retained_state_exactly_equal'] and not cleanup['all_scientific_cache_bytes_rehashed']
    assert read(MAINT / 'post_execution.json')['complete']
    assert space['manual_B4_space_requirement_met'] and space['archive_reuse_upload_required']
    assert space['current_snapshot_not_future_launch_guarantee']
    for path in [preparation_path, transfer_path, MAINT / 'independent_audit.json',
                 MAINT / 'inventory_after.json', MAINT / 'post_execution.json']:
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    protected = read(MAINT / 'local_protected_sha256.json')
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == ORIGINAL
    old_bindings = records['route_import.json']['source_bindings_unchanged']
    assert all(sha(Path(path)) == expected for path, expected in old_bindings.items())
    a_analysis = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis'
    a_execution = read(a_analysis / 'execution_audit.json')
    a_quality = read(a_analysis / 'independent_audit.json')
    assert a_execution['complete'] and a_execution['actual_optimizer_updates'] == 50
    assert a_quality['complete'] and not a_quality['model_qualified']
    a_gates = list(a_quality['metric_and_gate_decisions'].values())
    assert len(a_gates) == 2 and [len(gate['failures']) for gate in a_gates] == [14, 15]
    assert abs(a_gates[0]['gain'] * 100 - 0.471905) < 0.000001
    assert abs(a_gates[1]['gain'] * 100 - 0.473663) < 0.000001
    for name in ['execution_audit.json', 'independent_audit.json']:
        path = a_analysis / name
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)

    guide = ROOT / 'CCTV_DGP_BANK_COMPARISON_B_V1_R4_VM.md'
    guide_text = guide.read_text(encoding='utf-8')
    commands = [line for line in guide_text.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all('--zone=us-central1-a' in line for line in commands)
    assert all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert [int(n) for n in re.findall(r'(?m)^([1-5])\. ', guide_text)] == [1, 2, 3, 4, 5]
    assert '--protocol-sha ' + PIN + ' --verify-transfer' in guide_text
    assert 'bash scripts/run_bank_comparison.sh ' + PIN in guide_text
    assert 'python3 -B ~/cctv_dgp_bank_comparison_B4_transfer/install_packet.py --protocol-sha ' + PIN in guide_text
    assert "tr -d '\\r' < " + STEM + '-code-transfer.tar.gz.sha256 | sha256sum -c -' in guide_text
    assert preparation['transfer_archive_sha256'] in guide_text
    assert all(STEM + suffix in guide_text for suffix in ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json'])
    wrapper = ROOT / 'scripts/import_cctv_dgp_bank_B4_return_v1.py'
    tree = ast.parse(wrapper.read_text(encoding='utf-8'))
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)}
    assert values['PIN'] == PIN and values['STEM'] == STEM
    assert values['ARCHIVE_SHA'] == preparation['archive_sha256'] and values['NAME'] == preparation['name']
    cli = subprocess.run([sys.executable, '-B', str(wrapper), '--help'], check=True,
                         capture_output=True, text=True, timeout=30)
    assert 'usage:' in cli.stdout
    for path in [guide, wrapper]:
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)

    report = ROOT / 'CCTV_DGP_BANK_B_R3_RETURN_REVIEW.md'
    assert not report.exists()
    documents = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md',
                 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md',
                 'CCTV_DGP_BANK_COMPARISON_B_V1_R3_VM.md']
    backup_dir = ANALYSIS / 'documents_before'
    assert not backup_dir.exists()
    backup_dir.mkdir()
    before = {}
    for name in documents:
        original = (ROOT / name).read_bytes()
        (backup_dir / name).write_bytes(original)
        before[name] = sha(ROOT / name)
        assert sha(backup_dir / name) == before[name]
    new_text(backup_dir / 'manifest.json', json.dumps(dict(complete=True,
        UTC=datetime.now(timezone.utc).isoformat(), files_sha256=before), indent=2) + '\n')

    recovered = cleanup['measured_recovered_bytes'] / 1024**3
    allocated = cleanup['allocated_reclaim_bytes'] / 1024**3
    fresh = space['free_GiB']
    installed = space['conservative_B4_reuse_post_install_GiB']
    guide_link = link('five manual upload, installation, tmux and download steps', guide.name)
    report_text = f'''# Bank B r3 returned evidence and B r4 hash compatibility repair - 11 October 2026

Status: B r3 is a confirmed initialization **execution failure**, with zero
model forwards, gradients or optimizer updates. B r4 is a new independently
verified transfer; its manual VM loading, training and quality remain pending.
The complete restoration/completion goal remains ACTIVE.

## Returned evidence

Archive: `cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz`.
Bytes: 811,929,274. SHA256:
`52ea8aca05f5d74403ef707915c51d4b800718ddd2f7795443255c56b4180d3a`.
The safe importer checked all 5,631 regular members and 5,623 prepared bindings.
The independent saved-artifact audit checked all 5,630 manifest bindings.
The only run-output records are environment, failure and initializer-loading
receipts. There are no new paired/native candidate or baseline image outputs,
model states, gradient results or quality-gate records to review. No new visual
sheets were generated. The previously reviewed A and B r2 sheets remain separate.

Worker stop: {execution['worker_seconds']:.3f} seconds; supervisor:
{execution['supervisor_seconds']:.3f} seconds. All eight counters are zero:
original/A/B/recognizer/prior forwards, gradient queries, backwards and updates.
The failure occurred during `pinned_initializer_before_baseline`, before the
initializer hash or state loading completed. Export completion packaged the
failure; it is not training completion or image-quality evidence.

## Location, cause and correction

Location: `cctv_dgp_bank_pinned_initializer_v1.py`, `sha()` called from
`load_pinned_initializer()` before the baseline pass.
Cause: `hashlib.file_digest` is unavailable in recorded VM Python 3.10.12.
Python documents it as added in 3.11; see
[official Python hashlib documentation](https://docs.python.org/3.11/library/hashlib.html#hashlib.file_digest).
The new loader used an API unavailable in the known VM environment. This was
an implementation compatibility mistake, not evidence that the B design
cannot learn. It does not establish why B r2's constructor tensors differed.

Correction: new immutable loader `cctv_dgp_bank_pinned_initializer_v2.py`
uses streaming SHA256 over one-MiB blocks. Only its `sha()` function changes.
Hash-pinned initializer bytes, exact keys/shapes/dtypes/finite checks, permitted
new style-constructor differences, rejection of frozen/zero-fusion differences,
strict state loading and exact whole-state requirements remain unchanged.
The original loader, failed protocols, returns and receipts are preserved.

The missing API was reproduced without model import by extracting the old hash
helper and giving it a hashlib namespace without `file_digest`. The corrected
loader was tested on the real B CPU model/initializer with the API absent:
exact loading; exact restoration of a deliberately perturbed permitted style
tensor; rejection without overwrite of a frozen-prior change; rejection of a
zero-fusion change; rejection of a wrong initializer hash. Five cases passed;
three real file hashes also matched direct full-byte SHA256. State SHA256:
`98675d10be6f3405dd4bd0950d9251a61b26b9b161ae62a9ce48f775bc43e2bd`.
These are API-absence simulations on local Python 3.13, not execution in a
local Python 3.10 interpreter. They perform zero forwards/gradients/updates.
Actual VM loading and B r2's missing tensor diagnosis remain pending.

## Verified B r4 transfer

Protocol SHA256: `{PIN}`.
Full archive: `{STEM}-execution.tar.gz`, {preparation['archive_bytes']:,} bytes.
Full archive SHA256: `{preparation['archive_sha256']}`.
Small transfer: `{STEM}-code-transfer.tar.gz`, {preparation['transfer_archive_bytes']:,} bytes
({preparation['transfer_archive_bytes']/1024**2:.2f} MiB).
Small transfer SHA256: `{preparation['transfer_archive_sha256']}`.

Independent audit checked every one of the full packet's 5,630 regular tar
members and the transfer's 15 members, both LF checksum files, all 31 Python
files against Python 3.10 syntax, and absence of runtime `file_digest` calls
in that packet. It reconstructed all 5,618 reused assets plus 11 overlays and
proved that the resulting packet exactly matches the full archive. Model
computation AST and 11 scientific worker functions remain identical. Data,
initializer, schedule, quality thresholds and finite resource caps match the
old packet; six unsafe protocol changes were rejected. The actual transfer
CLI hashes the real initializer with the repaired API-absent loader. The
data-only installer verification CLI also passed; its live VM copy is unrun.

Manual installation copies assets from the retained B r3 directory into a
new B r4 root; it never deletes or changes the source. It validates source
protocol/assets/failure/terminal supervisor, idle GPU and every copied byte,
has a 300-second stop, and writes `packet_install_receipt.json`. No model code
or training is imported. The full archive remains locally for recovery or
migration. Use the small transfer on this VM to retain the required disk margin.
Follow the {guide_link}.

Scientific run remains 50 updates, 250 distinct TRAIN references, 1,250 profile
exposures and zero complete epochs / 0.3201 epoch, matching the intended A
exposures. B learns our conditioning/reconstruction with a frozen external
StyleGAN2 bank; the bank is not an own-trained generator. A's rejected trained
state is not the B initializer. A's retained raw/PNG gains remain 0.471905% /
0.473663%, below 1%, with 14 / 15 overlapping group/metric preservation failures.
A trained matched comparison is pending because neither B r2 nor B r3 updated.
No extra epochs, plateau bypass or app promotion are authorized by packaging.

Caps are unchanged: 16 GiB free after install, 1 GiB reserve, 6 GiB output,
20 GiB VRAM, worker 90 minutes, external 90.5 minutes, fitting 10 minutes,
preflight five minutes, snapshot 20 minutes, export 10 minutes and external
export 10.5 minutes. Runtime for a complete B r4 run is unmeasured.

## Authorized two-archive cleanup

Exactly two inactive VM home files were removed after full hash verification
against retained local backups: B r3 execution and B r3 result gzip archives.
Allocated reclaim: {allocated:.3f} GiB; measured recovery: {recovered:.3f} GiB.
Fresh free space: {fresh:.3f} GiB. Conservative space after the small upload
and complete B r4 copy: {installed:.3f} GiB, meeting 16 GiB at this snapshot.
The installer and launch must check current resources again.

Independent closure checked exact paths, UID 1001, single links, inode/mtime/
size/hash bindings, idle workloads and the deletion ledger. Before/after
state matches for {cleanup['retained_metadata_entries']:,} research metadata
entries and {cleanup['retained_bank_byte_hashes']:,} retained B r3 byte hashes.
Checkpoints, full states, data/splits, training caches, instructions, logs,
measurements, provenance and gate/failure records remain. Complete scientific
cache bytes were not rehashed; their namespace/metadata preservation is not
whole-cache byte verification. Two final post-export log tails are saved
locally without substituting any original returned archive bytes.

The separate Windows local cleanup by the other chat remains recorded in
PROJECT_HANDOFF.md and its exact previous bytes are retained. No VM start,
stop or switch, automatic pilot, gradient or training task was performed.
App source bindings and permanent checkpoint SHA256 `{ORIGINAL}` match.

## Remaining goal and evidence

After manual B r4 execution, independently audit constructor differences,
actual counters/stops, full states and frozen tensors where present, bank
ablations, all raw/PNG records and every planned sheet. Native CCTV remains
unpaired without clean alignment; paired synthetic metrics stay separate.
Reserved final identities remain unused. No Zamboanga or ethnicity claim
follows from these sources. All five milestones, seven covering families with
separate automatic/assisted review, independent final review and bundled
inline Playwright app qualification remain pending. App unchanged this turn;
no browser test was rerun and no quality candidate was adopted.

Evidence: {link('B r3 failure audit', 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis/failure_execution_audit.json')},
{link('compatibility regression', 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis/hash_compat_loader_regression.json')},
{link('B r4 transfer audit', 'outputs/cctv_dgp_bank_comparison_B4_transfer_v1_audit.json')} and
{link('cleanup closure', 'outputs/cctv_dgp_bank_B3_archive_cleanup_v1/independent_audit.json')}.
Future return importer: {link('import_cctv_dgp_bank_B4_return_v1.py', 'scripts/import_cctv_dgp_bank_B4_return_v1.py')}.
Its contract/CLI are checked; execution awaits the three returned result files.
Publication bindings: {link('publication.json', 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis/publication.json')}.
'''
    prefix = f'''# Latest verified status - 11 October 2026: B r3 compatibility failure audited; B r4 manual transfer ready

B r3 stopped after {execution['worker_seconds']:.3f} seconds, before any model
forward, gradient or optimizer update. Its loader used `hashlib.file_digest`,
which is unavailable in recorded VM Python 3.10.12. All 5,631 returned files
were imported and independently checked. No new image outputs exist. A's
failed 50-update quality result and B r2's constructor failure remain retained;
B r2's original tensor differences and a trained matched A/B result are pending.

B r4 changes only the hash helper to streaming SHA256. Five real-initializer
checks and the independent transfer audit passed with the unavailable API
removed. Actual Python 3.10/CUDA loading, training and quality are unverified.
Initializer, model computation, data, schedule, gates and finite caps remain.
Use the 3.55 MiB code transfer: 5,618 retained assets are copied into a distinct
B r4 root with 11 overlays. The complete packet is hash-verified; B r3 remains.
No agent training or automatic historical run was launched.

Authorized cleanup removed exactly two backed-up B r3 home archive copies.
Measured reclaim: {recovered:.3f} GiB; fresh free: {fresh:.3f} GiB; conservative
post-install free with the small transfer: {installed:.3f} GiB. The unchanged
16 GiB gate is met at this snapshot. Independent closure preserved
{cleanup['retained_metadata_entries']:,} research metadata entries and
{cleanup['retained_bank_byte_hashes']:,} B r3 byte hashes. Full scientific cache
bytes were not rehashed. The separate local-cleanup record below is retained.

Next: {guide_link}.
Review: {link('B r3 return, compatibility correction and limits', report.name)}.
The permanent checkpoint, app design and scientific gates remain unchanged.
The 50-update B experiment remains manual in tmux on the existing L4. All five
milestones and seven covering families remain required; goal ACTIVE, no model
adoption, automatic extra epochs or completion-quality claim. Historical
sections below retain their dated meaning and do not authorize reruns.

---

'''
    new_text(report, report_text)
    changed = {}
    for name in documents:
        old = (backup_dir / name).read_bytes()
        assert sha(ROOT / name) == before[name], 'Concurrent document edit; retain exact backup'
        header = prefix
        if name == 'CCTV_DGP_BANK_COMPARISON_B_V1_R3_VM.md':
            header = f'''# Superseded B r3 commands - 11 October 2026

B r3 has run and stopped before training because its loader requires a hash
API unavailable in VM Python 3.10. Its return and failure remain preserved.
Use the {guide_link} for the repaired B r4 packet. Do not rerun the historical
commands below. Evidence: {link('B r3 returned evidence', report.name)}.

---

'''
        (ROOT / name).write_bytes(header.encode('utf-8') + old)
        assert (ROOT / name).read_bytes().endswith(old)
        changed[name] = dict(before_sha256=before[name], after_sha256=sha(ROOT / name), original_body_retained=True)
    assert all(sha(ROOT / name) == expected for name, expected in protected.items())
    assert all(sha(ROOT / name) == expected for name, expected in evidence.items())
    assert all(sha(Path(path)) == expected for path, expected in old_bindings.items())
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__), evidence_files_sha256=evidence,
        report_sha256=sha(report), before_manifest_sha256=sha(backup_dir / 'manifest.json'),
        documents=changed, guide_exact_commands_and_pins_checked=True,
        B4_return_wrapper_AST_and_CLI_checked=True, B4_return_import_execution_pending=True,
        original_B3_wrapper_and_sources_unchanged=True, actual_B3_optimizer_updates=0,
        actual_local_Python310_interpreter_used=False, B4_VM_execution_pending=True,
        local_protected_bindings_unchanged=True, original_checkpoint_sha256=ORIGINAL,
        model_gradient_or_optimizer_calls=0, app_changed=False,
        browser_reverification_not_run=True, model_qualified=False, goal_complete=False)
    new_text(publication, json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, documents_updated=len(changed), B4_manual_pending=True,
        free_GiB=fresh, conservative_B4_post_install_GiB=installed, goal_complete=False), flush=True)


if __name__ == '__main__':
    main()
