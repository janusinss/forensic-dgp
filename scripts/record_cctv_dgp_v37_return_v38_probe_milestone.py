"""Preserve the full handoff and bind audited V37 and unrun V38 evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v37_return_v38_probe_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    started = time.monotonic(); assert not OUT.exists()
    old = read(PREVIOUS / 'milestone.json'); prior = read(PREVIOUS / 'independent_closure_audit.json')
    assert old['complete'] and prior['complete'] and prior['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    for name, digest in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
        assert time.monotonic() - started < 300
    audit = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json')
    analysis = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1/analysis.json')
    math = read(ROOT / 'outputs/cctv_dgp_v37_delivered_margins_v1/analysis.json')
    packet = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_preparation/independent_packet_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and not audit['failure_retained']
    assert analysis['all100_original_raw_and_PNG_files_byte_exact'] and analysis['V36_failures_flagged_by_new_coarse_prediction'] == 0
    assert math['geometry_qualified_for_disposable_probe'] and packet['complete'] and packet['all210_projection_rows_verified']
    addition = '''
**DGP milestone — 8 October 2026: V37 diagnostic independently audited; V38 finite PNG-margin probe prepared.**

The human returned V37. Its 43.838-second L4 diagnostic made 300 saved coarse
gradient queries, zero optimizer/parameter updates and no new checkpoint.
The independent local audit verifies all 431 regular files, 100 PNG guard
measurements, 102 group aggregates and 20 frozen CPU outputs. The audit takes
164.784 seconds inside a 165.947-second bounded supervisor. No returned code,
local autograd or optimizer is executed. Packaging completion does not imply
restoration quality, training success or preservation qualification.

All 100 original raw files and 100 original PNG files are byte-identical to V36.
There are no new restoration outputs to visually qualify. The independent saved
array analysis finds the PNG-valued coarse directions predict zero of V36's
11 measured PNG failures. An identity surrogate is not the true derivative of
discrete PNG conversion. V37 changes no adopted loss, decoder or application.

A new finite hypothesis retains every original raw/restoration check and all
65 V36 clearances. It adds 102 PNG coarse checks with 97 positive empirical
margins from all 408 group/metric/scale comparisons of V36's four trials. The
margin is twice the maximum positive scale-normalized finite-minus-coarse loss
discrepancy. This is calibration on TRAIN design data, not a validated error
bound or independent evaluation. All 210 rows and original clearances are
independently reassembled, with scaling/order/infeasibility/invalid-input checks.
The direction's magnitude is 1.08019 times the original proposal, within the
unchanged limit of 2. Its certificate permits a disposable finite probe only.

V38 resets the original DGP separately for scales 1, 0.5, 0.25 and 0.125 and
measures 100 original plus 400 trial outputs. It makes zero new gradients,
optimizer updates, epochs, committed trajectory updates or new checkpoints.
The image/metric/CPU replay definitions are verified unchanged by AST. All 17
PNG groups, source gains, brightness limit and prior capacity/development gates
remain. A future actual training pilot needs separate justification and manual
execution; the finite probe cannot satisfy 1% at50 or 10% at800 training gates.

The nine-file 12,234,380-byte transfer packet passes the independent 32.988-second
audit: 210-row geometry, 230 V34/28 V37/16 V36 saved bindings, original108 rows
and65 margins, 408 calibration observations, Python3.10/Bash syntax, three scope
rejections, eight archive rejection fixtures and five single-source gcloud
commands. The direction differs from failed V36. Protocol SHA256:
caea39469005b7066d7cdbd74e073ba674b9b9b6fb9324f3110fe827a06948ab.
Execution archive SHA256:
6096168f41f4be1fbaae9490340c7bdab70140d744b1db3de7194bab90ea38b0.

V38 is PREPARED, NOT RUN. The five-step guide uses the human's existing L4/
g2-standard-4 VM and tmux. Require 6 GiB free; estimate 3–6 minutes for the
probe plus 1–3 minutes for export. Worker/external caps are900/930 seconds,
export300/330 seconds, with30-second kill grace, VRAM20GiB and uncompressed
return768MiB. Missing or mismatched historical dependencies stop; neither
historical recipes nor new training are automatically launched. Return all
three files, including failure evidence, for independent audit and all100
case visual review. No VM connection or cleanup was performed in this turn;
historical free-space measurements are not a fresh inventory.

The14 current app/checkpoint hashes are unchanged. The original own-trained
DGP remains primary with existing Auto/On/Off selection, mask review and PNG/
bundle downloads. The prior inline Playwright functional checks are preserved;
there was no UI change here. Completion conditioning defects remain separately
recorded, with both automatic and assisted covering quality unqualified.
Original checkpoints, source/splits, caches/local backup and all failures remain.

The two50-case diagnostic cohorts are photographic TRAIN design data; the
historical 'unexposed' name is not a held-out claim. Source labels describe
acquisition, not ethnicity. Native24-crop evidence remains unpaired, paired
development520 cases/104 references remains separate, and reserved45 identities/
58 crops remain unopened. No real Zamboanga CCTV samples exist. Useful native
structure, all seven covering families, independent final review and the
complete reviewed DGP-led workflow remain required. Preserve visible appearance,
clear glasses and non-obstructing hair; request clearer/less-covered crops when
usable information is insufficient. Goal active/incomplete.

[Audited V37 findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_GUARD_GRAD_V37_RESULTS.md>)
[V38 five exact manual upload/install/tmux/launch/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_VM.md>)

The complete preceding handoff follows. Its V37-prepared-only language describes
the earlier milestone and is superseded by this audited human return.

'''
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    saved = OUT / 'before_docs/PROJECT_HANDOFF.md'; saved.write_bytes(before)
    at = before.index(b'\n') + 1; added = addition.encode('utf-8')
    handoff.write_bytes(before[:at] + added + before[at:])
    p = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_vm/protocol.json')
    bindings = dict(p['local_basis_sha256'])
    imported = read(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return_import.json')
    for name, digest in imported['files_sha256'].items(): bindings['outputs/cctv_dgp_delivered_guard_grad_v37_return/' + name] = digest
    bindings.update(analysis['bindings_sha256']); bindings.update(math['basis_sha256'])
    folders = ['outputs/cctv_dgp_delivered_guard_grad_v37_audit_run_v1', 'outputs/cctv_dgp_delivered_guard_grad_v37_analysis_v1',
               'outputs/cctv_dgp_v37_delivered_margins_v1', 'outputs/cctv_dgp_delivered_margin_probe_v38_preparation',
               'outputs/cctv_dgp_delivered_margin_probe_v38_vm']
    for folder in folders:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file(): bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    names = ['PROJECT_HANDOFF.md', saved.relative_to(ROOT).as_posix(), 'CCTV_DGP_DELIVERED_GUARD_GRAD_V37_RESULTS.md', 'CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_VM.md',
             'outputs/cctv_dgp_delivered_guard_grad_v37_return_import.json', 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json',
             'outputs/cctv-dgp-delivered-guard-grad-v37-results.tar.gz', 'outputs/cctv-dgp-delivered-guard-grad-v37-results.tar.gz.sha256',
             'outputs/cctv-dgp-delivered-guard-grad-v37-export.json', 'outputs/cctv-dgp-delivered-margin-probe-v38-execution.tar.gz',
             'outputs/cctv-dgp-delivered-margin-probe-v38-execution.tar.gz.sha256', 'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone/milestone.json',
             'outputs/cctv_dgp_v36_return_v37_diagnostic_milestone/independent_closure_audit.json', 'scripts/supervise_cctv_dgp_v37_return_audit_v1.py',
             'scripts/analyze_cctv_dgp_delivered_guard_grad_v37_v1.py', 'scripts/verify_cctv_dgp_delivered_guard_grad_v37_analysis_v1.py',
             'scripts/cctv_dgp_delivered_margin_geometry_v38.py', 'scripts/analyze_cctv_dgp_v37_delivered_margins_v1.py',
             'scripts/verify_cctv_dgp_v37_delivered_margins_v1.py', 'scripts/cctv_dgp_delivered_margin_readback_v38.py',
             'scripts/prepare_cctv_dgp_delivered_margin_probe_v38.py', 'scripts/cctv_dgp_delivered_margin_probe_v38_vm.py',
             'scripts/audit_cctv_dgp_delivered_margin_probe_v38_return.py', 'scripts/verify_cctv_dgp_delivered_margin_probe_v38_packet.py',
             'scripts/record_cctv_dgp_v37_return_v38_probe_milestone.py', 'scripts/verify_cctv_dgp_v37_return_v38_probe_milestone.py']
    for name in names: bindings[name] = sha(ROOT / name)
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items(): assert sha(ROOT / name) == digest, name
    bindings.update(app)
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    m = {'complete':True, 'UTC':datetime.now(timezone.utc).isoformat(), 'previous_milestone_sha256':sha(PREVIOUS / 'milestone.json'),
         'previous_handoff_path':saved.relative_to(ROOT).as_posix(), 'new_evidence_sha256':bindings, 'previous_bindings':len(old['new_evidence_sha256']),
         'document':{'before_sha256':hashlib.sha256(before).hexdigest(), 'after_sha256':sha(handoff), 'addition_bytes':len(added)},
         'V37_diagnostic_independently_audited':True, 'V37_optimizer_updates':0, 'V37_coarse_predictions_flagged_V36_failures':0,
         'V36_11_PNG_failures_retained':True, 'V38_prepared_only':True, 'V38_VM_launches':0,
         'local_gradient_calls':0, 'local_optimizer_updates':0, 'VM_calls_here':0, 'app_changes':False, 'app_adoption':False,
         'automatic_quality_qualification':False, 'assisted_quality_qualification':False, 'independent_final_review':False,
         'goal_complete':False, 'seconds':time.monotonic() - started, 'cap_seconds':300}
    assert m['seconds'] < 300
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(m, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete':True, 'bindings':len(bindings), 'seconds':m['seconds'], 'milestone_sha256':sha(OUT / 'milestone.json')}), flush=True)


if __name__ == '__main__': main()
