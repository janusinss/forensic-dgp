"""Derive finite V26 import/replay audit from frozen V25 sources; no neural work."""
import ast
from pathlib import Path
import time

from import_cctv_dgp_spatial_features_v25 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return_audit_preparation'
PIN = 'f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994'
BASIS = {
    'scripts/import_cctv_dgp_spatial_features_v25.py': '8c1c14f02fd97251d305753a12419fe4ba902fbf9bce6dec0fee34acb9e94711',
    'scripts/audit_cctv_dgp_spatial_features_v25.py': '0285f2c92b7c1b296c81b7af9588083dfd789bd85690777e64a1be9328190bc4',
    'scripts/audit_cctv_dgp_spatial_features_v25_execution.py': 'eda8c316e45131cb242208bbf02ce4d0aff47d6e785f3518972b5bc32671ba80',
    'scripts/audit_cctv_dgp_spatial_features_v25_features.py': 'f597446c24a241a629cd435879b936205cda28bc3bfa7c6ea0ac60ae4d9bd187',
    'scripts/audit_cctv_dgp_degraded_detail_v24_cohort.py': 'df856ec341c1c78d6ed139b08dee8c61850b4456af7faf379a8856de335c95aa',
}


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Frozen migration anchor changed: ' + old[:80])
    return text.replace(old, new, 1)


def derive_importer(text):
    text = text.replace('V25', 'V26').replace('spatial-features-v25', 'batchmatched-identity-v26')
    text = text.replace('cctv_dgp_spatial_features_v25_return', 'cctv_dgp_batchmatched_identity_v26_return')
    text = replace_once(text, "PIN = 'ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175'", "PIN = '" + PIN + "'")
    text = replace_once(text, 'MAX_MEMBERS = 1024', 'MAX_MEMBERS = 1600')
    text = replace_once(text, 'import time\n', 'import time\n\nfrom cctv_dgp_batchmatched_identity_v26_return_rules import protocol, source_hashes, allowed_return_name, REQUIRED_RECEIPTS\n')
    text = replace_once(text, '    fingerprints = {}\n', "    p = protocol(); expected_sources = source_hashes()\n    require(expected_pin == PIN, 'Only the frozen V26 protocol is permitted')\n    fingerprints = {}\n")
    text = replace_once(text, "            name = '/'.join(part.parts[1:])\n", "            name = '/'.join(part.parts[1:])\n            require(allowed_return_name(name, p), 'Unexpected returned file role: ' + name)\n")
    old = "    for name in ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_spatial_features_v25_vm.py', 'scripts/run_v25.sh', 'cctv_dgp_spatial_features_v25.py', 'cctv_dgp_spatial_features_v25_prepare.py', 'cctv_dgp_degraded_objective_v24.py'):\n        require(name in declared, 'Missing returned frozen source: ' + name)\n"
    new = "    for name, digest in expected_sources.items():\n        require(fingerprints.get(name) == digest, 'Missing or changed returned frozen source: ' + name)\n    for name in REQUIRED_RECEIPTS:\n        require(name in declared, 'Missing returned execution/install receipt: ' + name)\n"
    text = replace_once(text, old, new)
    text = replace_once(text, '    manifest, fingerprints, total = inspect_archive(archive)\n',
        "    import math\n    require(type(receipt.get('seconds')) in (int, float) and math.isfinite(receipt['seconds']) and\n            0 < receipt['seconds'] <= 180, 'Export receipt outside finite external bound')\n    manifest, fingerprints, total = inspect_archive(archive)\n")
    return text


def derive_auditor(text):
    text = text.replace('Independent V25 saved-output', 'Independent V26 saved-output')
    text = replace_once(text, 'from import_cctv_dgp_spatial_features_v25 import PIN, ROOT, read, relative_name, require, safe, sha, write',
        'from import_cctv_dgp_batchmatched_identity_v26 import PIN, ROOT, read, relative_name, require, safe, sha, write\nfrom cctv_dgp_batchmatched_identity_v26_return_rules import REQUIRED_SOURCES, allowed_return_name, check_installation, check_identity_preflights')
    text = text.replace("BUNDLE = ROOT / 'outputs/cctv_dgp_spatial_features_vm_v25'", "BUNDLE = ROOT / 'outputs/cctv_dgp_batchmatched_identity_vm_v26'")
    text = text.replace('Frozen local V25 protocol differs', 'Frozen local V26 protocol differs')
    text = replace_once(text, "p['format'] == 'dgp-spatial-feature-capacity-v25' and len(p['assets_sha256']) == 235", "p['format'] == 'dgp-spatial-batchmatched-identity-capacity-v26' and len(p['assets_sha256']) == 240")
    text = replace_once(text, '    started = time.monotonic()\n', "    started = time.monotonic()\n    require(cap_seconds == 600, 'Frozen local CPU audit cap differs')\n")
    text = replace_once(text, "    for name in ('protocol.json', 'schedule.json', 'scripts/cctv_dgp_spatial_features_v25_vm.py', 'scripts/run_v25.sh', 'cctv_dgp_spatial_features_v25.py', 'cctv_dgp_spatial_features_v25_prepare.py', 'cctv_dgp_degraded_objective_v24.py'):", '    for name in REQUIRED_SOURCES:')
    text = replace_once(text, '    head = make_head(bundle)\n', "    require(all(allowed_return_name(name, p) for name in files), 'Returned file role differs')\n    installation_audit = check_installation(returned, p, bundle)\n    identity_preflight_audit = check_identity_preflights(returned, p)\n    head = make_head(bundle)\n")
    text = replace_once(text, "{'detail_head': 50, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 100}", "{'detail_head': 60, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 120}")
    text = text.replace('from audit_cctv_dgp_spatial_features_v25_execution import check_execution', 'from audit_cctv_dgp_batchmatched_identity_v26_execution import check_execution')
    text = text.replace("with_name('audit_cctv_dgp_spatial_features_v25_execution.py')", "with_name('audit_cctv_dgp_batchmatched_identity_v26_execution.py')")
    text = replace_once(text, "        'cohort_loss_audit': cohort_audit,", "        'cohort_loss_audit': cohort_audit,\n        'installation_audit': installation_audit, 'identity_preflight_audit': identity_preflight_audit,\n        'identity_receipt_checker_sha256': sha(Path(__file__).with_name('cctv_dgp_batchmatched_identity_v26_return_rules.py')),\n        'actual_local_derivative_replay': False,")
    text = text.replace('outputs/cctv_dgp_spatial_features_v25_return', 'outputs/cctv_dgp_batchmatched_identity_v26_return')
    text = text.replace('outputs/cctv_dgp_spatial_features_v25_independent_audit.json', 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json')
    return text


def derive_execution(text):
    text = text.replace('Independent V25 gradient/count/timing', 'Independent V26 gradient/count/timing')
    text = text.replace('from import_cctv_dgp_spatial_features_v25 import PIN, read, require', 'from import_cctv_dgp_batchmatched_identity_v26 import PIN, read, require')
    text = replace_once(text, "    if execution is None:\n", "    require(supervisor['trainer_exit_code'] != 0 or result is not None, 'Successful supervised exit lacks completed800 result')\n    if result is not None:\n        require(supervisor['within_external_bound'] is True, 'Complete result exceeds external supervisor bound')\n    if execution is None:\n")
    text = replace_once(text, "{'detail_head': 101, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 151}", "{'detail_head': 111, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 171}")
    text = replace_once(text, "{'detail_head': 50 + 50 * len(snapshots) + execution['backwards'],", "{'detail_head': 60 + 50 * len(snapshots) + execution['backwards'],")
    text = replace_once(text, "'fixed_recognizer': 100 + 50 * len(snapshots) + execution['backwards']}", "'fixed_recognizer': 120 + 50 * len(snapshots) + execution['backwards']}")
    return text


def main():
    started = time.monotonic()
    require(not OUT.exists(), 'Preserve prior audit preparation')
    milestone = ROOT / 'outputs/dgp_v25_gradient_audit_and_batchmatched_identity_v26_milestone/milestone.json'
    require(sha(milestone) == 'e7dc3ae9a6d7501fc9fc3e048e0d0b50a4a0f16543e35d5bcba9b9943617957a', 'Current milestone changed')
    for name, digest in read(milestone)['new_evidence_sha256'].items():
        require(sha(ROOT / name) == digest, 'Milestone evidence changed: ' + name)
    for name, digest in BASIS.items():
        require(sha(ROOT / name) == digest, 'Frozen auditor basis changed: ' + name)
    jobs = [
        ('scripts/import_cctv_dgp_spatial_features_v25.py', 'scripts/import_cctv_dgp_batchmatched_identity_v26.py', derive_importer),
        ('scripts/audit_cctv_dgp_spatial_features_v25.py', 'scripts/audit_cctv_dgp_batchmatched_identity_v26.py', derive_auditor),
        ('scripts/audit_cctv_dgp_spatial_features_v25_execution.py', 'scripts/audit_cctv_dgp_batchmatched_identity_v26_execution.py', derive_execution),
    ]
    generated = {}
    for source, destination, derive in jobs:
        require(not (ROOT / destination).exists(), 'Preserve generated source: ' + destination)
        generated[destination] = derive((ROOT / source).read_text(encoding='utf-8'))
        ast.parse(generated[destination], feature_version=(3, 10))
    OUT.mkdir()
    for destination, text in generated.items():
        with (ROOT / destination).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    write(OUT / 'preparation.json', {
        'complete': True, 'date': '2026-10-06', 'protocol_sha256': PIN,
        'scope': 'Prospective independent return import/inference audit; actual V26 run pending',
        'basis_sha256': BASIS, 'generated_source_sha256': {name: sha(ROOT / name) for name in generated},
        'identity_receipt_rules_sha256': sha(ROOT / 'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py'),
        'archive_member_cap': 1600, 'CPU_audit_seconds_cap': 600, 'original_quality_and_replay_bounds_retained': True,
        'all50_whole_face_review_required': True, 'returned_VM_code_executed': False,
        'local_model_calls': 0, 'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'VM_actions': False, 'actual_V26_return_present': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - started,
    })
    print('V26 prospective return importer and independent replay audit prepared; no model execution.')


if __name__ == '__main__':
    main()
