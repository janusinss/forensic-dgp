"""Preserve history and bind the V22 audit/V23 transfer milestone. No neural/VM work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_cctv_dgp_detail_prior_v22_r1 import read, require, sha, write

EVIDENCE = ROOT / 'outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone'
PREVIOUS = ROOT / 'outputs/dgp_structure_and_detail_pilot_milestone_v21_v22/milestone.json'
PREVIOUS_SHA = '1dce521bf7e446729d1018299c080c2f2001401f5a8350a3b722cb2c2faeff79'
APP = ROOT / 'outputs/dgp_app_v3_integration_record.json'
APP_SHA = 'e744f3708f248012469a111e204222f66261af4ca28bbaf9521ee4c5464b5e7a'
PREPARATION = ROOT / 'outputs/cctv_dgp_detail_skip_v23_preparation'
BUNDLE = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23'
PIN = '4cddb98fb5e6a215cb84f98132cfa5a2146ee157c1cb939919ffbf8c5740e863'
ARCHIVE_SHA = '21e5ea32887e99f35e8196f46e862ab102db0ecdeef2285d793ca3432c35fa53'
DOCS = ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_DGP_DETAIL_PRIOR_V22_VM.md')

PREFIXES = {
    'PROJECT_HANDOFF.md': '''## Current restoration milestone — 6 October 2026: V22 R1 failure audited; different V23 packet ready

The downloaded V22 R1 archive matches76,931,848 bytes/SHA256
`4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad`.
All366 returned files,196 original assets,100 raw/PNG pairs/metric rows, both
complete50-case snapshots and exact stopped-head state pass the independent
68.30s audit. Fixed CPU replay uses100 head and110 recognizer forwards; no
original-DGP forward, backward or optimization. Ten return-auditor checks pass.
Initial torch-version rounding/Windows temporary-directory failures and their
bounded auditor-only corrections remain preserved; quality tolerances are intact.

The original L4 stop is verified:50 updates/51 backwards,0.0000322991% feature
gain versus the frozen1% early condition. All ten original-resolution sheets/
50 cases are reviewed; no visible structure gain.4,486 pixels change by at most
one byte. A50-case known-layer CPU trace finds a too-small spatial correction
through the deep route/projection. Independent saved evidence checks375 sources
and200 exact review cells. Nine original preservation checks diagnostically fail
at update50; the unexecuted800-update final gate is not called evaluated.
Close V22 without unchanged rerun, relaxed stop or app adoption.

V23 prepares a different4613parameter full256-resolution input-detail bypass
plus shallow branch, preserving data, objective,800 updates/80epochs/batch5,
original preservation/brightness gates and timing stops.209 assets/211 members,
Python3.10 grammar, Bash syntax and Windows training rejection pass. All50 zero
raw/PNG outputs are exact; fixed sensitivity and nonempty-padding contracts pass.
Preparation uses56 new-head/four closed-V22-head fixed CPU forwards; no new local
optimizer/backward or VM execution. Complete timing/VRAM/gradient/count receipts
are added. Learning, capacity and useful outputs remain pending on the manual L4.

The user feedback “its useful but needed clearer structure” remains a useful-case
finding with visible-structure work required. No usable input is relabeled
insufficient to hide model softness. Previous560 milestone and22 current app
bindings pass; prior workflow/runbook bytes are preserved before this update.
App/design, checkpoints, splits and original failed gates remain intact; historical
34 regressions/Playwright are not rerun for this evidence/transfer-only change.

No native/reserved/covering/COFW-test pixels enter V23. Native evidence remains
unpaired, photographic training metrics separate; no inferred ethnicity or local
Zamboanga performance. Useful native DGP restoration, all seven automatic/assisted
covering families, canonical app parity and independent final review remain open.
Goal active. Training execution is verified transfers/pasteable commands only.
The separate maintenance entry below records a stopped VM; it is preserved and
availability is not refreshed here. No assistant SSH/upload/start/cleanup/training.

Commands: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
Plan: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_PLAN.md>).
Result: [V22 R1](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Evidence: `outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/`.
''',
    'SYSTEM_WORKFLOW_AND_GOAL.md': '''**Latest 6 October 2026 — V22 R1 stopped run independently closed; different V23 transfer prepared:**
The returned archive/source hashes,196 assets/366 returned members,100 raw/PNG
pairs/metrics and both50-case snapshots pass the independent local audit. The
original L4 run stops after50 updates/51 backwards:0.0000322991% delivered feature
gain versus its unchanged1% early gate. Ten original-resolution sheets/all50 cases
show no visible structure improvement. The too-small correction through the deep
route/detail projection is traced and independently saved-evidence checked.
Nine original preservation checks diagnostically fail at update50; the unexecuted
800-update final gate is not evaluated. V22 is closed without adoption or rerun.
Failed auditor fixtures/source and explicit initializer-only compatibility
correction are preserved; replay/quality tolerances remain unchanged.

V23 is a new4613parameter full-resolution detail bypass plus shallow branch.
Same ten exposed training photographs/50 cases, objective,800 updates/80 epochs,
batch5, appearance/brightness gates, budgets and early stops; no new data or
CodeFormer RGB/features.209 assets/211 members, Python3.10/Bash/Windows guard,
50 exact initial raw/PNG cases and fixed sensitivity/padding contracts pass.
These verify implementation/transfer, not gradients, learned structure or quality.
Actual one-batch gradient check and finite fitting remain manual on the existing
NVIDIA L4/g2-standard-4 under ~/forensic-dgp. The packet exports fuller timing,
step/VRAM/component counts and supervisor receipts; failed partial outputs remain.
No assistant VM/cloud action or local training occurs. User execution preference
remains verified transfer files and exact pasteable upload/tmux/download commands.

Prior560 milestone and22 app bindings pass; original workflow/runbook bytes are
preserved. The DGP-led app/design, original checkpoints, splits and historical
failures remain unchanged; historical app regressions/Playwright are not rerun.
The shown native crop remains useful and input-usable with clearer structure
required. Full native usefulness, canonical app parity, all seven automatic/
assisted covering families and independent final review remain unqualified.
Native remains unpaired; paired exposed-training metrics are separate. No real
Zamboanga CCTV samples, inferred ethnicity or local-performance claims. Final
reserves/COFW test pixels remain unopened, reviewers/cohort unassigned. Goal active.
Commands: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
Result: [V22 R1](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Evidence: `outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/`.
''',
    'PRACTICAL_OUTPUT_SCOPE.md': '''Latest 6 October 2026: the downloaded V22 R1 failure is independently audited
and all50 cases reviewed. Its unchanged1% early structure condition fails after
50 updates; actual delivered feature improvement is0.0000322991%, with no visible
structure gain. All returned hashes/source/snapshots/metrics pass the integrity
and replay audit. A separate known-layer trace finds too-small spatial corrective
signal; original failures, stopped state and auditor-only corrections remain.
Close that exact recipe without relaxed gates, unchanged rerun or app adoption.

Different V23 prepares a full256-resolution input-detail bypass and shallow own
head, with the same exposed training data/objective/schedule/quality safeguards.
All209 assets/211 members, Python3.10/Bash/Windows guard,50 exact initial raw/PNG
cases and fixed no-target sensitivity/nonempty-padding checks pass. These do not
prove gradients, learned capacity or clearer restoration. Manual L4 execution is
finite800 updates/80epochs, with unchanged timing/early stops and fuller exported
receipts. Verified transfers and exact upload/tmux/separate-download commands only;
no assistant VM action or local backward/optimization occurs.

The user's useful-case feedback still requires clearer visible facial structure;
the shown usable crop remains usable. No full native or covering acceptance is
claimed. All seven families remain required: masks, sunglasses, strong lens glare,
hands, obstructing hair, scarves and other objects. Preserve clear glasses,
ordinary hair and visible appearance, with the documented small removal margin.
Show the proposed removal area for optional correction before generation, request
a less-covered image when too little face remains, and return one plausible
estimate with original/mask plus PNG and optional bundle. No exact hidden identity.
Automatic and assisted results remain separate.

V23 uses no native/final/covering or COFW-test pixels. Native remains unpaired;
paired exposed-training metrics are separate, with no ethnicity or local Zamboanga
claims. Original checkpoints/splits/gate failures and the DGP-led Auto/On/Off app
remain intact; prior560 milestone/22 app bindings pass, original documents saved.
Historical app regressions/Playwright are not rerun. Useful development outputs,
canonical app parity, full-family review and independent final review remain
required. A packet, capacity pass or training completion does not complete the Goal.
Commands: [V23](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
Result: [V22 R1](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Evidence: `outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/`.
''',
    'CCTV_DGP_DETAIL_PRIOR_V22_VM.md': '''**Latest status — 6 October 2026: V22 R1 is independently audited and closed.**
The downloaded76,931,848-byte archive matches its checksum. All366 files,
100 raw/PNG pairs/metrics, both50-case snapshots and stopped checkpoint pass
integrity/replay checks. All50 cases are reviewed: no visible structure gain.
The original update50 early stop remains failed; no final800 gate was executed.
The retained failure and diagnostic report remain available.

The commands below are historical V22 procedure. Do not rerun steps1–4 or restart
this unchanged failed recipe. A different verified V23 detail path is now prepared:
[V23 upload/tmux/download commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_SKIP_V23_VM.md>).
[V22 R1 independent result](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md>).
Original pre-update runbook bytes are preserved in
`outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/before_docs/`.

Historical status and procedure follow.
''',
}


def relative(path):
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(ROOT), 'Path exits workspace')
    return resolved.relative_to(ROOT).as_posix()


def previous_bindings(after=False):
    require(sha(PREVIOUS) == PREVIOUS_SHA, 'Previous milestone differs')
    previous = read(PREVIOUS)
    require(len(previous['new_evidence_sha256']) == 560, 'Previous scope differs')
    locations = {}
    for name, expected in previous['new_evidence_sha256'].items():
        path = ROOT / name
        if name in ('PROJECT_HANDOFF.md', 'CCTV_DGP_DETAIL_PRIOR_V22_VM.md'):
            path = ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_putty_download_fix_v1/before_docs' / name
        elif after and name in DOCS:
            path = EVIDENCE / 'before_docs' / name
        require(sha(path) == expected, 'Previous binding differs: ' + name)
        if path != ROOT / name:
            locations[name] = relative(path)
    return len(previous['new_evidence_sha256']), locations


def app_bindings():
    require(sha(APP) == APP_SHA, 'App record differs')
    app = read(APP)
    bindings = {**app['sources_sha256'], **app['evidence_sha256']}
    require(len(bindings) == 22, 'App scope differs')
    for name, expected in bindings.items():
        require(sha(ROOT / name) == expected, 'App binding differs: ' + name)
    return len(bindings)


def packet():
    require(sha(BUNDLE / 'protocol.json') == PIN, 'Protocol differs')
    prep = read(PREPARATION / 'preparation.json')
    check = read(PREPARATION / 'independent_preparation_audit.json')
    padding = read(PREPARATION / 'nonempty_padding_contract_v1.json')
    require(prep['complete'] and check['complete'] and padding['complete'], 'Incomplete preparation')
    require(prep['archive_sha256'] == ARCHIVE_SHA and
            sha(ROOT / 'outputs/cctv-dgp-detail-skip-v23-execution.tar.gz') == ARCHIVE_SHA, 'Archive differs')
    require(check['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_detail_skip_v23_preparation.py'), 'Checker differs')
    require(padding['checker_sha256'] == sha(ROOT / 'scripts/verify_cctv_dgp_detail_skip_v23_padding_v1.py'), 'Padding checker differs')
    p = read(BUNDLE / 'protocol.json')
    require(len(p['assets_sha256']) == 209 and len(p['cases']) == 50, 'Packet scope differs')
    for name, expected in p['assets_sha256'].items():
        require(sha(BUNDLE / name) == expected, 'Packet asset differs: ' + name)
    require(sha(ROOT / 'scripts/cctv_dgp_detail_skip_v23.py') == p['assets_sha256']['scripts/cctv_dgp_detail_skip_v23.py'], 'Root runtime differs')
    return p


def prepare():
    started = time.monotonic()
    require(not EVIDENCE.exists(), 'Preserve existing milestone preparation')
    count, locations = previous_bindings()
    app_count = app_bindings()
    packet()
    EVIDENCE.mkdir()
    (EVIDENCE / 'before_docs').mkdir()
    snapshots = {}
    for name in DOCS:
        source = ROOT / name
        copy = EVIDENCE / 'before_docs' / name
        shutil.copy2(source, copy)
        require(sha(copy) == sha(source), 'Document copy differs')
        snapshots[name] = sha(copy)
    write(EVIDENCE / 'plan.json', {
        'date': '2026-10-06', 'scope': 'V22 independently audited failure; V23 fixed-forward/transfer milestone, no app acceptance',
        'checker_sha256': sha(Path(__file__)), 'previous_milestone_sha256': PREVIOUS_SHA, 'app_record_sha256': APP_SHA,
        'before_docs_sha256': snapshots, 'preserve_existing_maintenance_entry': True,
        'VM_execution_preference': 'Verified transfer files and pasteable commands only',
        'time_cap_seconds': 120, 'new_neural_calls': 0, 'local_training_calls': 0, 'VM_actions': False,
        'goal_status': 'active', 'goal_complete': False,
    })
    write(EVIDENCE / 'before_update_audit.json', {
        'complete': True, 'previous_bindings': count, 'previous_document_locations': locations,
        'app_bindings': app_count, 'documents_preserved': snapshots,
        'packet_protocol_sha256': PIN, 'archive_sha256': ARCHIVE_SHA, 'packet_assets': 209,
        'seconds': time.monotonic() - started, 'neural_calls': 0, 'VM_actions': False,
    })
    print(json.dumps({'prepared': True, 'previous_bindings': count, 'app_bindings': app_count, 'before_docs': len(snapshots)}))


def publish():
    started = time.monotonic()
    plan = read(EVIDENCE / 'plan.json')
    require(plan['checker_sha256'] == sha(Path(__file__)), 'Milestone source changed')
    require(not (EVIDENCE / 'milestone.json').exists(), 'Preserve frozen milestone')
    packet()
    for name, expected in plan['before_docs_sha256'].items():
        require(sha(ROOT / name) == expected and sha(EVIDENCE / 'before_docs' / name) == expected, 'Concurrent document edit: ' + name)
    for name, prefix in PREFIXES.items():
        path = ROOT / name
        original = path.read_bytes()
        split = original.index(b'\n') + 1
        require(original.startswith(b'# '), 'Unexpected document title')
        path.write_bytes(original[:split] + ('\n' + prefix + '\n').encode('utf-8') + original[split:])
    count, locations = previous_bindings(after=True)
    app_count = app_bindings()
    names = set(DOCS) | {'CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md', 'CCTV_DGP_DETAIL_SKIP_V23_PLAN.md', 'CCTV_DGP_DETAIL_SKIP_V23_VM.md',
        'scripts/import_cctv_dgp_detail_prior_v22_r1.py', 'scripts/audit_cctv_dgp_detail_prior_v22_r1.py',
        'scripts/diagnose_cctv_dgp_detail_prior_v22_r1.py', 'scripts/verify_cctv_dgp_detail_prior_v22_r1_diagnostic.py',
        'scripts/prepare_cctv_dgp_detail_skip_v23.py', 'scripts/cctv_dgp_detail_skip_v23.py',
        'scripts/verify_cctv_dgp_detail_skip_v23_preparation.py', 'scripts/verify_cctv_dgp_detail_skip_v23_padding_v1.py',
        'scripts/finalize_cctv_dgp_detail_v22_v23_milestone.py', 'scripts/verify_cctv_dgp_detail_v22_v23_milestone.py',
        'tests/test_cctv_dgp_detail_prior_v22_r1_return_audit.py',
        'outputs/cctv-dgp-detail-prior-v22-r1-results.tar.gz', 'outputs/cctv-dgp-detail-prior-v22-r1-results.tar.gz.sha256',
        'outputs/cctv-dgp-detail-prior-v22-r1-export.json', 'outputs/cctv_dgp_detail_prior_v22_r1_return_import.json',
        'outputs/cctv_dgp_detail_prior_v22_r1_independent_audit.json',
        'outputs/cctv-dgp-detail-skip-v23-execution.tar.gz', 'outputs/cctv-dgp-detail-skip-v23-execution.tar.gz.sha256'}
    for directory in (ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_return', ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_diagnostic',
                      ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_return_audit_preparation', PREPARATION, BUNDLE, EVIDENCE):
        names.update(relative(path) for path in directory.rglob('*') if path.is_file())
    names.discard(relative(EVIDENCE / 'publish.log'))
    bindings = {name: sha(ROOT / name) for name in sorted(names)}
    seconds = time.monotonic() - started
    require(seconds <= plan['time_cap_seconds'], 'Milestone publication exceeds cap')
    milestone = {
        'date': '2026-10-06', 'complete': True, 'scope': plan['scope'],
        'new_evidence_sha256': bindings, 'new_bindings': len(bindings),
        'previous_milestone_sha256': PREVIOUS_SHA, 'previous560_bindings_verified': count,
        'previous_document_locations': locations, 'app_record_sha256': APP_SHA, 'app22_bindings_verified': app_count,
        'user_feedback': 'its useful but needed clearer structure.', 'input_usable_label_preserved': True,
        'v22_return_sha256': '4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad',
        'v22_archive_members': 366, 'v22_snapshots': [0, 50], 'v22_updates': 50, 'v22_backwards_VM': 51,
        'v22_feature_improvement_percent': 0.00003229907770130325, 'v22_original_early_gate_pass': False,
        'v22_unexecuted_final_gate_evaluated': False, 'v22_cases_reviewed': 50, 'v22_visible_structure_gain': False,
        'v22_trace_fixed_CPU_head_forwards': 50, 'v22_independent_CPU_replay_head_recognizer_forwards': [100, 110],
        'v22_diagnostic_preservation_failures': 9, 'original_failures_and_auditor_corrections_preserved': True,
        'v23_protocol_sha256': PIN, 'v23_archive_sha256': ARCHIVE_SHA, 'v23_assets': 209, 'v23_archive_members': 211,
        'v23_trainable_parameters': 4613, 'v23_finite_updates_epochs': [800, 80],
        'v23_transfer_and_fixed_forward_contracts_pass': True, 'v23_new_head_CPU_contract_forwards': 56,
        'v23_closed_v22_CPU_sensitivity_forwards': 4, 'v23_DGP_recognizer_forwards': 0,
        'v23_VM_gradient_training_capacity_quality_verified': False, 'model_changed_in_current_app': False,
        'local_optimizer_updates': 0, 'local_backward_calls': 0, 'assistant_VM_cloud_actions': 0,
        'native_reserved_covering_COFW_test_pixels_enter_new_pilot': False, 'app_design_source_changed': False,
        'historical_app_regressions_or_Playwright_rerun': False, 'historical_gates_waived': False,
        'native_usefulness_qualified': False, 'covering_families_qualified': False,
        'canonical_app_parity_complete_for_new_head': False, 'independent_final_review_complete': False,
        'execution_preference': plan['VM_execution_preference'], 'goal_status': 'active', 'goal_complete': False,
        'next': 'Manual existing-L4 V23 finite pilot; independently audit returned receipts/states/outputs and review all50 cases before any separately frozen broader stage',
        'seconds': seconds,
    }
    write(EVIDENCE / 'milestone.json', milestone)
    print(json.dumps({k: milestone[k] for k in ('complete', 'scope', 'new_bindings', 'previous560_bindings_verified', 'app22_bindings_verified', 'goal_complete', 'seconds')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'publish'))
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else publish)()
