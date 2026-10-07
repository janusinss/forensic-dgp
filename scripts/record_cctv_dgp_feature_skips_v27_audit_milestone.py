"""Bind closed V27 evidence and completed maintenance without a new model recipe."""
import json
from pathlib import Path
import time

from import_cctv_dgp_feature_skips_v27 import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_feature_skips_v27_audit_milestone'
CLEANUP = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json'
PREVIOUS = ROOT / 'outputs/dgp_v26_gradient_return_and_feature_skips_v27_milestone/milestone.json'
DOCS = ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_FEATURE_SKIPS_V27_PLAN.md', 'CCTV_DGP_FEATURE_SKIPS_V27_VM.md')


def verify(bindings, mapping=None):
    for name, digest in bindings.items():
        path = (ROOT / (mapping or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and sha(path) == digest, name
    return len(bindings)


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Preserve previous milestone'
    audit_path = ROOT / 'outputs/cctv_dgp_feature_skips_v27_independent_audit.json'
    audit = read(audit_path)
    folder = ROOT / 'outputs/cctv_dgp_feature_skips_v27_saved_output_review'
    review = read(folder / 'visual_review.json')
    arithmetic = read(folder / 'stopped_capacity_arithmetic.json')
    assert audit['complete'] and audit['complete_snapshot_updates'] == [0, 50]
    assert not audit['early_structure_stop']['pass'] and audit['failure_cause'] == 'No one-percent early structural gain; retain stop'
    assert review['complete'] and review['cases_reviewed'] == 50 and not review['independent_final_review']
    assert arithmetic['capacity_arithmetic_at_stopped50']['preservation_failures'] == []
    cleanup = read(CLEANUP)
    assert sha(CLEANUP) == 'c6834df1627d2a93467391dd9c883a7df7a612659fa907888aead9866bd8d567'
    assert cleanup['complete'] and cleanup['archives_removed'] == 14 and not cleanup['goal_complete']
    verify(cleanup['new_evidence_sha256'])
    assert sha(PREVIOUS) == cleanup['previous_milestone_sha256']
    previous = read(PREVIOUS)
    verify(previous['new_evidence_sha256'], cleanup['previous309_original_locations'])
    app_path = ROOT / 'outputs/dgp_app_v3_integration_record.json'
    app = read(app_path)
    assert sha(app_path) == previous['app_record_sha256']
    assert verify({**app['sources_sha256'], **app['evidence_sha256']}) == 22
    OUT.mkdir()
    before = OUT / 'before_docs'
    before.mkdir()
    originals, mapping = {}, {}
    for name in DOCS:
        original = (ROOT / name).read_bytes()
        with (before / name).open('xb') as stream:
            stream.write(original)
        originals[name] = original
        mapping[name] = (before / name).relative_to(ROOT).as_posix()
    prefix = '''**Latest research milestone — 6 October 2026: V27 return independently audited/reviewed; structure failure closed.**

The325,777,145-byte archive matches SHA256
1bd4aec2bdc10cd6aca4455f24875b006bf4bca2d1a4f1cae8f4932400b386ab.
Safe import verifies631 regular files. Independent CPU replay checks246 original
assets,100 raw/PNG pairs and metric rows,100 head/110 recognizer forwards and250
own-DGP feature arrays. All original source/state/numerical/cohort/timing checks
remain. No local gradients/backwards/optimizer updates or model changes occur.

V27's delivered degraded structure gain is0.2210885780%, below the unchanged1%
requirement. It stops at50/800 updates with51 backwards; final800 never runs.
All five new feature-readout gradients are active before optimization and the
original five projection gradients are active at update2. Initial corrected
identity value/all36 gradients are exactly zero for all ten batches. Original
DGP and recognizer states remain frozen. Worker30.127s, fit9.615s, supervisor
36.006s and allocated VRAM1,843,821,568 bytes pass original timing/memory bounds.
This is a quality stop, not a transfer, CUDA, timing or absent-gradient failure.

All ten original-detail sheets/50 paired photographic TRAIN cases are reviewed;
all200 source cells are exact. All50 PNGs change, with median degraded correction
1.0231445 byte units/max8. Small tonal changes do not convincingly add eye, nose,
mouth, outline and overall visible-appearance structure together. At stopped50,
saved arithmetic has no original17-group pixel/SSIM/identity violation. This does
not establish final capacity, native generalization or independent human acceptance.

V25/V26/V27 gains are0.0282235%/0.0335636%/0.2210886%. Stop model modifications for
the architecture discussion required by the frozen plan and workspace circuit
breaker. The invalid assumption is that these finite small-head changes on frozen
own-DGP pixels/features would provide sufficient whole-face structure. No unique
optimization cause or impossibility of all heads is proved. The user selects
review of a separate candidate of the original DGP reconstruction decoder;
the direct reply is preserved. Review its source/initial parity before a distinct
finite protocol. No next
pilot/protocol is created, no failed recipe reruns and no gate is relaxed.

The user-authorized direct VM cleanup is complete and independently verified:
14 backed-up archive duplicates/1.64GiB removed;8,036,728,832 bytes(7.48GiB) free;
207,967 protected hashes,4,817 scientific tensor stamps and490 current research
bindings preserved. The39,448,585,279-byte scientific caches/current V27 archives
remain. CUDA available/L4 idle; VM left running. Direct access was maintenance
only; actual training remains manual on the existing L4/g2-standard-4 with verified
transfers and pasteable commands. The exact restated user goal remains recorded.

Completed cleanup60 bindings, previous309 and deeper66/692/299/697/513 histories,
five complete document bodies, app22 bindings and every original checkpoint,
split and failed gate remain preserved. Own-trained DGP remains primary and
pretrained restoration models comparisons. Native CCTV stays unpaired; paired
photographic TRAIN evidence stays separate. Reserved-final identities remain
unviewed. No new native/covering pixels, ethnicity or Zamboanga/hidden-identity
claim enters this milestone. Previously useful inputs remain usable. Useful
native outputs, canonical app flow/regressions, all seven automatic/assisted
covering families and independent final review remain required. Goal active/incomplete.

[Audited V27 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_SKIPS_V27_RESULTS.md>) ·
[Architecture discussion](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V27_ARCHITECTURE_REVIEW.md>) ·
[Verified VM cleanup](<C:/xampp/htdocs/YEAR 4/Testing/VM_STORAGE_CLEANUP_20261006_V2.md>)

Earlier bodies below are preserved history. Their V27 install/launch steps and
pilot-pending statements are historical. V27 is closed; do not rerun those steps.

'''
    for name, original in originals.items():
        split = original.index(b'\n') + 1
        (ROOT / name).write_bytes(original[:split] + b'\n' + prefix.encode('utf-8') + original[split:])
    original309 = dict(cleanup['previous309_original_locations'])
    for name in ('CCTV_DGP_FEATURE_SKIPS_V27_PLAN.md', 'CCTV_DGP_FEATURE_SKIPS_V27_VM.md'):
        original309[name] = mapping[name]
    verify(previous['new_evidence_sha256'], original309)
    verify(cleanup['new_evidence_sha256'], mapping)
    evidence = {}
    def bind(path):
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in DOCS + ('CCTV_DGP_FEATURE_SKIPS_V27_RESULTS.md', 'CCTV_DGP_POST_V27_ARCHITECTURE_REVIEW.md',
                        'scripts/review_cctv_dgp_feature_skips_v27_saved_outputs.py',
                        'scripts/record_cctv_dgp_feature_skips_v27_review.py',
                        'scripts/record_cctv_dgp_feature_skips_v27_audit_milestone.py',
                        'scripts/verify_cctv_dgp_feature_skips_v27_audit_milestone.py',
                        'outputs/cctv-dgp-feature-skips-v27-results.tar.gz',
                        'outputs/cctv-dgp-feature-skips-v27-results.tar.gz.sha256',
                        'outputs/cctv-dgp-feature-skips-v27-export.json',
                        'outputs/cctv_dgp_feature_skips_v27_return_import.json',
                        'outputs/cctv_dgp_feature_skips_v27_independent_audit.json',
                        'outputs/cctv_dgp_post_v27_architecture_review_v1/user_architecture_decision.json',
                        'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/closure_manifest.json',
                        'outputs/cctv_dgp_vm_storage_cleanup_20261006_v2/independent_closure_readback.json'):
        bind(ROOT / name)
    for directory in (ROOT / 'outputs/cctv_dgp_feature_skips_v27_return', folder, before):
        for path in directory.rglob('*'):
            if path.is_file():
                bind(path)
    milestone = {'complete': True, 'date': '2026-10-06', 'scope': 'Verified VM maintenance and closed actual V27 TRAIN capacity failure; full Goal active',
                 'new_evidence_sha256': evidence, 'cleanup_closure_sha256': sha(CLEANUP),
                 'cleanup60_original_locations': {name: mapping[name] for name in DOCS[:3]},
                 'previous_milestone_sha256': sha(PREVIOUS), 'previous309_original_locations': original309,
                 'original_document_bodies_preserved': 5, 'app_record_sha256': sha(app_path), 'app22_bindings_preserved': True,
                 'actual_V27_return_audit_complete': True, 'V27_whole_cohort_assistant_review_complete': True,
                 'V27_structure_gain_percent': 100 * audit['early_structure_stop']['relative_feature_error_gain'],
                 'V27_early_stop_failed': True, 'V27_updates': 50, 'unchanged_rerun_permitted': False,
                 'model_modifications_stopped_for_architecture_discussion': True,
                 'architecture_decision_pending': False, 'architecture_decision': 'Review a separate copy of our original DGP reconstruction decoder',
                 'next_training_recipe_created': False,
                 'VM_training_launched_by_assistant': False, 'VM_storage_maintenance_performed': True,
                 'reserved_final_viewed': False, 'independent_final_review_complete': False, 'app_promotion': False,
                 'goal_status': 'active', 'goal_complete': False, 'seconds': time.monotonic() - start}
    write(OUT / 'milestone.json', milestone)
    print(json.dumps({key: milestone[key] for key in ('complete', 'V27_structure_gain_percent',
        'V27_early_stop_failed', 'architecture_decision_pending', 'goal_status', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
