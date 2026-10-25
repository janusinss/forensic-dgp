"""Audit conservative candidate holds and mixed-role historical comparison metadata.

Standard library only; no images are judged, models loaded or training performed.
Earlier root labels/candidate arrays and the164-update design remain immutable.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_filter_train_v2_preparation'
HELPER = ROOT / 'scripts/audit_codeformer_filter_train_v2_admission.py'
HELPER_SHA = '3d19203dbe137d7a5646c9e31147b0fcb7c32aa50259abc4754e230597ea5a5b'
HOLD_SHA = '5c59dbf7826e976eb4e1605183b5a3255eaa518745e1ee085b2ba9062cbde24a'
OBSERVER_SHA = 'baa4ddb514ebe046f9b3b1b1d7eeccf8e8d9349d706474cfbf4081893886b8ac'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
spec = importlib.util.spec_from_file_location('_immutable_admission_metadata_helpers', HELPER)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def audit():
    M.require(not any(x in sys.modules for x in ('numpy', 'PIL', 'scipy', 'torch', 'tensorflow', 'jax')),
              'Standard-library-only metadata audit')
    raw = M.bound(PREP / 'candidate_holds_v1.json', HOLD_SHA, 240143)
    holds = M.strict(raw)
    root_raw = M.from_pin(holds['root_review']); root = M.strict(root_raw)
    obs_raw = M.from_pin(holds['second_observer_review']); obs = M.strict(obs_raw)
    M.require(hashlib.sha256(root_raw).hexdigest() == M.REVIEW_SHA and
              hashlib.sha256(obs_raw).hexdigest() == OBSERVER_SHA, 'Immutable reviewer inputs differ')
    for pin in obs['page_and_sheet_pins']: M.from_pin(pin)
    M.require(obs['complete'] is True and obs['scope']['sheets_personally_viewed'] == 4 and
              obs['scope']['exact256_input_panels_personally_viewed'] == 64 and
              obs['scope']['root_per_case_labels_first_read_only_after_all4_page_locks'] is True and
              obs['scope']['root_aggregate19R44B1I_exposed_in_task_before_viewing'] is True,
              'Second observer scope/declaration differs')
    M.require(obs['data_manifest'] == root['data_manifest'] and obs['criteria'] == root['criteria'],
              'Both observers must use identical data/criteria')
    root_map = {x['id']: x for x in root['cases']}
    obs_map = {x['id']: x for x in obs['per_case']}
    M.require(len(root_map) == 512 and len(obs_map) == 64 and all(
        x['profile'] == 'global_mosaic_8' for x in obs_map.values()), 'Exact512/64 scope required')
    confusion = Counter()
    for key, case in obs_map.items():
        r = root_map[key]
        M.require(case['root_decision'] == r['decision'] and
                  case['second_observer_decision'] == case['decision'] and
                  case['sheet_cell_RGB8_sha256'] == r['sheet_cell_RGB8_sha256'] and
                  case['reference_id'] == r['reference_id'] and case['pilot_admission_by_review'] is False,
                  'Observer/root/exact input hash join differs')
        confusion[r['decision'], case['second_observer_decision']] += 1
    agreement = sum(v for (r, o), v in confusion.items() if r == o)
    M.require(agreement == obs['exact_agreement'] == 46 and obs['disagreements'] == 18 and
              confusion['readable', 'readable'] == obs['both_readable'] == 13,
              'Recomputed64 agreement/confusion differs')
    M.require(Counter(x['second_observer_decision'] for x in obs_map.values()) == obs['observer_counts'] and
              Counter(root_map[k]['decision'] for k in obs_map) == obs['root_counts'], 'Review counts differ')
    pose = {'tr_ffhq_28856', 'tr_ffhq_62653'}
    six = {'tr_ffhq_' + x + '__global_mosaic_8' for x in ('32648', '32828', '39377', '41369', '51547', '60124')}
    M.require({k for k, x in obs_map.items() if x['root_decision'] == 'readable' and
              x['second_observer_decision'] == 'borderline'} == six and
              set(holds['held_pose_source_ids']) == pose, 'Six per-case/two source holds differ')
    candidates, held_former = [], []
    M.require(len(holds['all512_cases']) == 512, 'Every512 case must remain in hold ledger')
    for saved, original in zip(holds['all512_cases'], root['cases']):
        key = original['id']; initial = original['candidate_for_explicit_finite_pilot']
        reasons = []
        if not initial: reasons.append('not_in_root_readable_source_and_input_candidate_list')
        if original['reference_id'] in pose:
            reasons.append('held_source_pose_scope_disagreement_pending_input_only_adjudication')
        observer_label = obs_map[key]['second_observer_decision'] if key in obs_map else None
        # Root non-readable is already recorded by the first reason. The
        # additional global reason records absence of observer-readable status.
        if original['profile'] == 'global_mosaic_8' and observer_label != 'readable':
            reasons.append('global_mosaic_not_jointly_readable_pending_adjudication')
        candidate = not reasons
        M.require(saved['id'] == key and saved['reference_id'] == original['reference_id'] and
                  saved['profile'] == original['profile'] and saved['root_input_decision'] == original['decision'] and
                  saved['root_source_clear_decision'] == original['source_clear_input_decision'] and
                  saved['initial_root_candidate'] is initial and saved['second_observer_global_decision'] == observer_label and
                  saved['hold_reasons'] == reasons and saved['candidate_for_finite_packet_preparation'] is candidate and
                  saved['optimization_permitted_now'] is False, 'Conservative per-case calculation or original label differs')
        if candidate: candidates.append(saved)
        if initial and not candidate: held_former.append(key)
    M.require(held_former == holds['held_former_root_candidate_ids'] and len(held_former) ==
              holds['held_former_root_candidates'] == 16, 'Exact16 previous candidates held')
    cprofile = Counter(x['profile'] for x in candidates)
    sources = {x['reference_id'] for x in candidates}
    M.require(len(candidates) == holds['candidate_total'] == 370 and cprofile['clear'] == holds['candidate_clear'] == 57 and
              len(candidates) - cprofile['clear'] == holds['candidate_nonclear'] == 313 and
              len(sources) == holds['candidate_sources'] == 57 and dict(cprofile) == holds['by_profile'],
              'Exact370/313/57 candidate census differs')
    selected = M.strict(M.bound(PREP / 'cohort_selection_v1.json'))
    cohort = {x['reference_id']: x for x in selected['cohort']}
    strata = Counter((cohort[rid]['attribution']['license_url'], cohort[rid]['official_category']) for rid in sources)
    expected_strata = [{'license_url': k[0], 'official_category': k[1], 'sources': v} for k, v in sorted(strata.items())]
    M.require(expected_strata == holds['source_strata'] and len(strata) == 9, 'Current57-source nine-stratum census differs')
    M.require(holds['all_original_labels_and_splits_retained'] is True and holds['all_optimization_still_false'] is True and
              holds['provisional_candidates_not_final_training_admission'] is True and
              holds['full391_53_pool_old72_406_refused24_and_reserved_FINAL_unchanged'] is True and
              holds['strong_rows_require_explicit_new_pilot_admission'] is True and
              holds['policy_frozen_before_new_training_or_model_output_selection'] is True and
              holds['neural_calls'] == holds['optimizer_updates'] == 0 and holds['goal_complete'] is False,
              'Retained scope/no-training boundaries differ')
    M.require(holds['planned_updates_if_later_admitted'] == 157 and
              holds['planned_filter_slots_if_later_admitted'] == 314 and
              holds['planned_microbatches_if_later_admitted'] == 628 and
              holds['padding_duplicate_count_if_later_admitted'] == 1, 'Conditional finite census differs')
    old_path = ROOT / 'outputs/codeformer_filter_comparison_v1/manifest.json'
    old_raw = M.bound(old_path, '7b476134e4f0f7975eb4dd41385e1197b2b08f7094b970a25165b196c0dd7be8')
    old = M.strict(old_raw)
    planned = Counter(x['split'] for x in old['cases'] if x['inference_planned'])
    refused = Counter(x['split'] for x in old['cases'] if not x['inference_planned'])
    M.require(len(old['cases']) == 96 and planned == {'TRAIN': 36, 'DEV': 36} and
              refused == {'TRAIN': 12, 'DEV': 12} and old['actual_training_updates'] == 0,
              'Old72 mixed TRAIN/DEV role census differs')
    snapshot = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation/design_before_second_observer_v1'
    snapshot_pins = []
    for name, expected in [('DESIGN_DRAFT.md', 'f89b656b4d56cbe373f8faafed30a521d04ba971ec4458d1361a31fdef1093bf'),
        ('proposed_training_design_v1.json', 'e2b03a7761b257e9f594a5220f533204461cab2333614214ec277f886e2dc25f')]:
        M.bound(snapshot / name, expected); snapshot_pins.append(M.fingerprint(snapshot / name))
    # Initial root review must remain byte-exact after the new conservative layer.
    M.bound(PREP / 'root_input_only_review_v1.json', M.REVIEW_SHA)
    receipt = {'format': 'codeformer-filter-train-v2-candidate-holds-independent-metadata-audit-v1',
        'complete': True, 'pass': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'auditor': M.fingerprint(Path(__file__)), 'immutable_helper_sha256': HELPER_SHA,
        'hold_ledger_sha256': HOLD_SHA, 'root_review_sha256': M.REVIEW_SHA,
        'second_observer_sha256': OBSERVER_SHA, 'reviewed_metadata_case_joins': 512,
        'observer_case_joins': 64, 'observer_exact_agreement': agreement, 'observer_disagreements': 18,
        'both_global_readable_before_source_scope': 13, 'both_global_readable_after_source_scope': 12,
        'six_root_R_observer_B_holds': sorted(six), 'whole_source_pose_holds': sorted(pose),
        'held_former_candidates': 16, 'former_candidates_from_two_pose_sources': 10,
        'all16_pose_source_variants_held': True, 'original_per_input_labels_and_initial_candidates_unchanged': True,
        'candidate_total': 370, 'candidate_clear': 57, 'candidate_nonclear': 313, 'candidate_sources': 57,
        'candidate_by_profile': dict(cprofile), 'current_source_strata': expected_strata,
        'all9_current_strata_independently_recomputed': True,
        'old_filter_comparison_manifest': M.fingerprint(old_path),
        'old_filter96_split': {'TRAIN': 48, 'DEV': 48},
        'old_filter72_inferred_split': dict(planned), 'old24_refused_split': dict(refused),
        'planned_snapshot_output_slots': 478, 'existing_historical_DEV_slots': 406,
        'old_filter_mixed_comparison_slots': 72, 'old_filter_TRAIN_slots': 36, 'old_filter_DEV_slots': 36,
        'snapshot_slots_are_not_independent_DEV_identities': True,
        'prior164_design_snapshot': snapshot_pins, 'conditional_updates': 157,
        'conditional_microbatches': 628, 'conditional_filter_slots': 314,
        'clear_dark_joins_or_sampler_sealed': False, 'optimization_authorized': False,
        'independent_visual_review_or_adjudication_performed': False,
        'observer_role_and_aggregate_count_exposure_retained': True,
        'no_clean_target_resolution_of_disagreement': True,
        'new_neural_gradient_optimizer_VM_browser_calls': 0, 'model_or_FINAL_qualification': False, 'goal_complete': False}
    out = PREP / 'candidate_holds_metadata_audit_v1.json'
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    return {'complete': True, 'pass': True, 'candidate_nonclear': 313, 'candidate_sources': 57,
            'old72_split': dict(planned), 'receipt': str(out)}


if __name__ == '__main__': print(json.dumps(audit()), flush=True)
