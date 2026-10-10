"""Retain the prospective visual selection while excluding an incomplete triad."""
from pathlib import Path
import ast
import time
from cctv_dgp_actual_step_review_v1_contract import NAME, read, write, sha

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'scripts/analyze_cctv_dgp_actual_step_review_v1_return.py'
GENERATED = ROOT / 'scripts/analyze_cctv_dgp_actual_step_review_v1_partial_return.py'
OUT = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis'


def main():
    start = time.monotonic()
    source = ORIGINAL.read_text(encoding='utf-8')
    prior_plan_path = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json'
    prior_plan = read(prior_plan_path)
    partial_scope_path = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit/prospective_partial_scope.json'
    partial_scope = read(partial_scope_path)
    p = read(ROOT / 'outputs' / NAME / 'protocol.json')
    assert partial_scope['protocol_sha256'] == prior_plan['protocol_sha256'] == sha(ROOT / 'outputs' / NAME / 'protocol.json')
    assert not OUT.exists() and not GENERATED.exists()
    complete = {(row['update'], row['proposal']) for row in partial_scope['conditions']}
    triads = [probe['update'] for probe in p['probes'] if all((probe['update'], proposal) in complete for proposal in ['zero', 'recorded', 'cone'])]
    assert triads == [10, 12, 22, 27, 37, 38, 39, 42, 44]
    visual_rows = [row for row in prior_plan['visual_rows'] if row['update'] in triads]
    assert len(visual_rows) == 225 and [row for row in prior_plan['visual_rows'] if row['update'] not in triads] == prior_plan['visual_rows'][-25:]
    changes = []
    def change(before, after):
        nonlocal source
        assert source.count(before) == 1, before
        source = source.replace(before, after, 1)
        changes.append({'before': before, 'after': after})
    change("OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_analysis'", "OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_partial_analysis'")
    change("audit_path=ROOT/'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json'", "audit_path=ROOT/'outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json'")
    change("assert audit['complete'] and audit['full_3150_review_complete'] and audit['all3150_raw_and_PNG_and_mean_only_outputs_checked']", "assert audit['complete'] and not audit['full_3150_review_complete'] and audit['all3045_raw_PNG_and_mean_only_outputs_checked']")
    change("audit['CPU_replay']['cases']==150", "audit['CPU_replay']['cases']==145")
    change("    for probe in p['probes']:\n        receipt={}", "    completed={(row['update'],row['proposal']) for row in audit['conditions']}\n    for probe in p['probes']:\n        receipt={}")
    change("        for proposal in PROPOSALS:\n            path=", "        for proposal in PROPOSALS:\n            if (probe['update'],proposal) not in completed:continue\n            path=")
    change("        for proposal in PROPOSALS[1:]:\n            terms=", "        for proposal in PROPOSALS[1:]:\n            if proposal not in receipt:continue\n            terms=")
    change('assert len(rows)==120 and len(term_changes)==20', 'assert len(rows)==114 and len(term_changes)==19')
    change('assert len(subset)==10', "assert len(subset)==(10 if proposal=='recorded' else 9)")
    change("'states':10", "'states':len(subset)")
    change('for begin in range(0,250,5):', 'for begin in range(0,len(plan[\'visual_rows\']),5):')
    change('assert len(sheet_records)==50 and sum(len(s[\'cells\']) for s in sheet_records)==1500', 'assert len(sheet_records)==45 and sum(len(s[\'cells\']) for s in sheet_records)==1350')
    change("'all120_role_stage_state_proposal_rows':rows", "'all114_role_stage_state_proposal_rows':rows")
    change("'all20_current_batch_raw_term_changes':term_changes", "'all19_current_batch_raw_term_changes':term_changes")
    change("'paired_TRAIN_diagnostic_only':True", "'full_3150_review_complete':False,'original_storage_failure_retained':True,'uncompleted_cone_update45_excluded':True,'paired_TRAIN_diagnostic_only':True")
    change("'conditions':30,'summary_rows':120,'sheets':50,'cells':1500", "'conditions':29,'summary_rows':114,'sheets':45,'cells':1350")
    inverse = source
    for row in reversed(changes):
        assert inverse.count(row['after']) == 1
        inverse = inverse.replace(row['after'], row['before'], 1)
    assert inverse == ORIGINAL.read_text(encoding='utf-8')
    ast.parse(source, feature_version=(3, 10))
    OUT.mkdir()
    GENERATED.write_text(source, encoding='utf-8', newline='\n')
    plan = {**prior_plan, 'visual_rows': visual_rows, 'sheets': 45, 'exact256_pixel_cells': 1350,
        'selection': prior_plan['selection'] + '; only nine completely returned triads, unchanged case selection',
        'scientific_summary_scope': 'All29 completed conditions/3045 slots;10 recorded and9 cone states;raw and PNG separate',
        'original_full_visual_plan_sha256': sha(prior_plan_path),
        'prospective_partial_scope_sha256': sha(partial_scope_path),
        'incomplete_triad_update45_excluded_visually': True,
        'completed_update45_zero_and_recorded_in_numeric_summary_only': True,
        'full_3150_review_complete': False, 'original_storage_failure_retained': True,
        'source_changes': changes, 'inverse_source_matches_original': True,
        'source_bindings': {path.relative_to(ROOT).as_posix(): sha(path) for path in [ORIGINAL, GENERATED, Path(__file__), prior_plan_path, partial_scope_path]},
        'seconds': time.monotonic() - start,
    }
    write(OUT / 'prospective_review.json', plan)
    print({'complete': True, 'visual_rows': 225, 'sheets_planned': 45, 'exact_cells': 1350, 'model_forwards': 0}, flush=True)


if __name__ == '__main__':
    main()
