"""Bind the audited V41 failure and design review without changing any model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import time
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v41_return_review_milestone'
PRIOR = ROOT/'outputs/cctv_dgp_v40_signal_return_v41_prepared_milestone'


def main():
    start = time.monotonic(); assert not OUT.exists()
    parent = read(PRIOR/'milestone.json'); closed = read(PRIOR/'independent_closure_audit.json')
    assert closed['complete'] and closed['milestone_sha256'] == sha(PRIOR/'milestone.json')
    for name, digest in parent['new_evidence_sha256'].items(): assert sha(ROOT/name) == digest, name
    audit = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json')
    analysis = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/analysis.json')
    checked = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/independent_analysis_audit.json')
    visual = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis/visual_review.json')
    assert audit['complete'] and audit['failure_retained'] and not audit['necessary_capacity_pass']
    assert audit['per_update_learning_evidence']['all_logged_updates_verified'] == 50
    assert checked['complete'] and checked['saved_updates'] == 50 and checked['exact_unscaled_sheet_cells'] == 500
    assert visual['complete'] and visual['all100_TRAIN_cases_and500_cells_actually_viewed']
    assert not visual['quality_qualification'] and not visual['app_promotion']
    gain = audit['gates'][0]['relative_feature_gain']; failed = audit['gates'][0]['preservation_failures']
    assert len(failed) == 1 and failed[0]['metric'] == 'ArcFace_observed_fixed'
    conflicts = analysis['direction_summary'][0]['actual_ascent_despite_projected_nonincrease']
    assert len(conflicts) == 9
    OUT.mkdir(); (OUT/'before_docs').mkdir()
    before = (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section = f'''
**Research milestone — 9 October 2026: V41 failure audited; applied-step design review retained.**

The human returned V41,1,178,179,150bytes, SHA256
01a6e9198e731d81895c0b64cde21f45ed7eb01be2bb48b4c78a9186726c94dc.
Independent audit verifies27,551 files, all50 parameter/moment chains and350
saved component-gradient queries,7,810 delivered PNGs plus7,810 mean-only PNGs
and100 frozen CPU
preview replays. The VM stops at50 of800 updates; its17.37-minute run and
successful export do not imply restoration qualification.

Delivered paired photographic TRAIN structure gain is{100*gain:.10f}%, below
the unchanged1% requirement. One FFHQ compound-group ArcFace preservation
failure also remains. All3,905 initial PNGs match V40 exactly. Both preselected
TRAIN cohorts,100 cases/500 unchanged256-pixel cells, are actually reviewed:
useful added eye/nose/mouth/outline definition is not established. No native,
development or reserved-final evaluation is opened or relabeled from outputs.

Saved-step arithmetic shows nine actual AdamW updates opposing the local
landmark-improvement gradient despite its projected direction being nonascending.
This remains after removing weight-decay displacement. It is first-order evidence,
not a unique cause or finite-image guarantee. An independently KKT-checked,
fixed21-iteration cone calculation on all50 saved arrays is a diagnostic proposal
only; no proposed parameter is assigned to a network or trained checkpoint.

The original prospective auditor rejects the exporter's retained V40 archive
prefix before extraction. Its source/log/receipt remain. Distinct R1 changes
only that exact prefix and its receipt filename, imports into the separate V41
folder and passes11 archive-boundary checks. Training scripts, protocol, weights,
image/gate thresholds and all historical failures are unchanged.

The post-V38/V40/V41 circuit-breaker question asks which design diagnostic to
investigate before another training pilot. No new training recipe is modified,
resumed or automatically run. The reviewed next direction is testing actual
optimizer-step preservation before further learning; finite neural evidence is
still required. All actual training and gradient diagnostics stay manual on the
existing L4/g2-standard-4 at ~/forensic-dgp.

All14 DGP-primary app bindings, original/stopped checkpoints, caches/local backup,
data/splits/provenance and failure records remain. Public CCTV is unpaired; paired
synthetic metrics remain separate. Source names do not infer ethnicity or
Zamboanga performance. Reserved45 identities/58 crops remain unopened. Useful
native restoration, seven automatic/assisted covering families, independent
final review and qualified full app flow remain outstanding. Goal incomplete.
The complete previous handoff is archived and preserved below.

[V41 audited result and design review](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V41_OPTIMIZER_REVIEW.md>)

'''
    addition = section.encode('utf-8'); at = before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    names = [p.relative_to(ROOT).as_posix() for directory in [
        'outputs/cctv_dgp_pcgrad_fit_v41_return', 'outputs/cctv_dgp_pcgrad_fit_v41_audit_recovery_r1',
        'outputs/cctv_dgp_pcgrad_fit_v41_analysis'] for p in (ROOT/directory).rglob('*') if p.is_file()]
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'outputs').iterdir() if p.is_file() and
                 (p.name.startswith('cctv-dgp-pcgrad-fit-v41-results.') or p.name == 'cctv-dgp-pcgrad-fit-v41-export.json' or
                  p.name.startswith('cctv_dgp_pcgrad_fit_v41_return_') or p.name in [
                      'cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json', 'cctv_dgp_pcgrad_fit_v41_archive_scope_diagnostic.json']))
    names.extend(['CCTV_DGP_POST_V41_OPTIMIZER_REVIEW.md',
                  'scripts/audit_cctv_dgp_pcgrad_fit_v41_return_r1.py',
                  'scripts/verify_cctv_dgp_pcgrad_fit_v41_archive_scope_r1.py',
                  'scripts/analyze_cctv_dgp_pcgrad_fit_v41_return.py',
                  'scripts/verify_cctv_dgp_pcgrad_fit_v41_analysis.py',
                  'scripts/record_cctv_dgp_pcgrad_fit_v41_visual_review.py',
                  'scripts/record_cctv_dgp_v41_return_review_milestone.py',
                  'scripts/verify_cctv_dgp_v41_return_review_milestone.py',
                  (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
                  (PRIOR/'milestone.json').relative_to(ROOT).as_posix(),
                  (PRIOR/'independent_closure_audit.json').relative_to(ROOT).as_posix()])
    bindings = {name: sha(ROOT/name) for name in set(names)}
    assert time.monotonic()-start < 600
    write(OUT/'milestone.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'previous_milestone_sha256': sha(PRIOR/'milestone.json'),
          'previous_handoff_path': (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
          'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'addition_bytes': len(addition)},
          'new_evidence_sha256': bindings, 'V41_human_run_audited': True, 'V41_updates': 50,
          'all100_TRAIN_comparison_cases_reviewed': True, 'failed_capacity_and_preservation_retained': True,
          'new_training_recipe_prepared': False, 'no_local_proposals_assigned_to_models': True,
          'all14_DGP_primary_app_bindings_unchanged': True, 'VM_calls_here': 0,
          'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'app_promotion': False,
          'independent_final_review': False, 'goal_complete': False,
          'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'new_bindings': len(bindings), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
