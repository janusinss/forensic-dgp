"""Metadata-only FINAL readiness/draft. Never reads images, arrays or archives."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_restoration_FINAL_protocol_v1_preparation'
OPENED = {}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def metadata(path):
    path = ROOT / path
    need(path.resolve().is_relative_to(ROOT), 'Metadata must remain within workspace')
    need(path.suffix.lower() in {'.json', '.csv', '.md', '.html', '.txt', '.py'} or path.name == 'LICENSE',
         'Pixel/array/archive access prohibited')
    raw = path.read_bytes()
    name = path.relative_to(ROOT).as_posix()
    OPENED[name] = {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return raw.decode('utf-8-sig')


def read(path):
    return json.loads(metadata(path))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    need(not OUT.exists(), 'Preserve prior FINAL preparation namespace')
    qbase = 'outputs/cctv_native_development_v2/'
    cbase = 'outputs/cctv_chokepoint_native_development_v1/'
    q = read(qbase + 'frozen_subset.json'); qp = read(qbase + 'selection_policy.json')
    qr = read(qbase + 'input_review.json')
    c = read(cbase + 'frozen_subset.json'); cp = read(cbase + 'selection_plan.json')
    cr = read(cbase + 'input_review.json')
    extension = read('outputs/cctv_native_structure_extension_v1/frozen_subset.json')
    terms = read('outputs/cctv_dgp_native_contract_audit_20261009_v1/source_terms_review.json')
    historical_audit = read('outputs/cctv_dgp_native_contract_audit_20261009_v1/audit.json')
    terms_paths = [cbase + 'LICENSE_SOURCE.html', cbase + 'DERIVATIVE_NOTICE.txt',
        'outputs/cctv_dgp_checkpoint_comparison_v1/source_terms/QMUL_request.json',
        'outputs/cctv_dgp_checkpoint_comparison_v1/source_terms/QMUL_acquisition.json',
        'third_party/codeformer/LICENSE']
    for path in terms_paths:
        metadata(path)
    oldbase = 'outputs/dgp_independent_review_protocol_v1/'
    old = read(oldbase + 'coordinator_assignment.json')
    oldprep = read(oldbase + 'preparation_receipt.json')
    oldaudit = read(oldbase + 'independent_preparation_audit.json')
    for name in ['INDEPENDENT_REVIEW_PROTOCOL_V1.md', 'INDEPENDENT_REVIEW_PROTOCOL_V1_R1.md',
                 'INDEPENDENT_REVIEW_READINESS_V1.md', oldbase + 'REVIEWER_INSTRUCTIONS.md',
                 oldbase + 'covering_review_template.csv', oldbase + 'readiness_and_evidence.csv']:
        metadata(name)
    input_forms = list(csv.DictReader(io.StringIO(metadata(oldbase + 'native_input_review.csv'))))
    output_forms = list(csv.DictReader(io.StringIO(metadata(oldbase + 'native_output_review.csv'))))
    need(len(input_forms) == 116 and len(output_forms) == 580, 'Retained review form counts')
    need(all(not r[k] for r in input_forms for k in ('reviewer_id', 'review_date', 'input_decision', 'input_reason')),
         'Existing input forms must remain unfilled')
    need(all(not r[k] for r in output_forms for k in ('reviewer_id', 'review_date',
         'visible_structure_and_appearance_preserved', 'useful_improvement_over_resize', 'recommendation')),
         'Existing final verdicts must remain unfilled')
    need(not old['reviewers_assigned'] and old['final_candidate_execution_binding'] is None and not old['final_execution_ready'],
         'Historical FINAL binding remains not ready')
    need(OPENED['INDEPENDENT_REVIEW_PROTOCOL_V1.md']['sha256'] == oldaudit['protocol_original_sha256'] and
         OPENED['INDEPENDENT_REVIEW_PROTOCOL_V1_R1.md']['sha256'] == oldaudit['protocol_r1_sha256'], 'Old protocols unchanged')
    for name, digest in oldprep['artifacts_sha256'].items():
        need(OPENED[oldbase + name]['sha256'] == digest, 'Old forms/coordinator remain exact')

    reserved = {}; development = {}; overlaps = {}
    for name, cohort, field, namespace in [('QMUL-SurvFace', q, 'global_person_id', 'QMUL-SurvFace/global'),
                                          ('ChokePoint', c, 'source_person_id', 'ChokePoint/P1')]:
        rs = [r for r in cohort['cases'] if r['role'] == 'reserved_evaluation']
        ds = [r for r in cohort['cases'] if r['role'] == 'development']
        ri, di = {str(r[field]) for r in rs}, {str(r[field]) for r in ds}
        overlaps[name] = sorted(ri & di)
        need(not overlaps[name], 'Within-source DEV/reserved label overlap')
        reserved[name] = {'case_count': len(rs), 'release_identity_count': len(ri),
            'identity_namespace': namespace, 'release_identity_ids': sorted(ri),
            'native_short_side_range': [min(min(r['native_width'], r['native_height']) for r in rs),
                                       max(min(r['native_width'], r['native_height']) for r in rs)],
            'cases': rs, 'input_eligibility_review': 'NOT_STARTED', 'case_to_identity': {r['id']: str(r[field]) for r in rs}}
        development[name] = {'case_count': len(ds), 'release_identity_count': len(di), 'release_identity_ids': sorted(di)}
    need(reserved['QMUL-SurvFace']['case_count'] == reserved['QMUL-SurvFace']['release_identity_count'] == 32 and
         reserved['ChokePoint']['case_count'] == 26 and reserved['ChokePoint']['release_identity_count'] == 13,
         'Exact reserved58cases/45namespacedlabels')
    need(set(reserved['ChokePoint']['release_identity_ids']) == set(cp['reserved_person_ids']) and
         all(not any(k in r for k in ('native_crop', 'input', 'observed')) for r in reserved['ChokePoint']['cases']),
         'ChokePoint reserve remains26metadata-only cases')
    need(q['reserved_inputs_rendered'] == c['reserved_crops_rendered'] == c['reserved_image_pixels_decoded'] == 0,
         'Creation-time reserved exposure metadata')
    for path, digest in historical_audit['metadata_bindings_sha256'].items():
        if path in OPENED:
            need(OPENED[path]['sha256'] == digest, 'Retained native contract metadata binding')
    extension_ids = {r['global_person_id'] for r in extension['cases']}
    q_all_ids = {r['global_person_id'] for r in q['cases']}
    need(not extension_ids & q_all_ids, 'Excluded extension3 labels are disjoint from originalQMUL56')

    split_path = 'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json'
    split = read(split_path)
    need(set(split) == {'train', 'validation'} and len(split['train']) == 76000 and len(split['validation']) == 4000,
         'Historical split has train/validation only, not a frozen FINAL role')
    need(not set(split['train']) & set(split['validation']), 'Historical exact-path split separation')
    refs = read('outputs/cctv_codeformer_adaptation_v2_vm/references.json')
    eval_bundle = read('outputs/codeformer_filter_fusion_training_v1_preparation/evaluation_bundle_v1.json')
    paired_dev = [r for r in eval_bundle['cases'] if r['cohort'] == 'historical_DEV406' and r['evidence'] == 'paired_synthetic']
    old40 = read('outputs/cctv_paired_regression_v1/frozen_protocol.json')
    old520 = read('outputs/cctv_dgp_v29_paired_development_v1/plan.json')
    cofw = read('outputs/cofw_final_candidate_metadata_v1_r1/frozen_candidate_queue.json')
    need(len(paired_dev) == 358 and all(r['split'] == 'DEV' and not r['reserved_FINAL'] for r in paired_dev),
         'Historical358 are DEV, not FINAL')
    need(len(old40['cases']) == 40 and len(old520['cases']) == 520 and not old520['independent_final_review'], 'Earlier paired evidence is development')
    model_receipt = read('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/inference_export_v1.json')
    fixed_dev_plan = read('outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_preparation/fixed_DEV_plan_v1.json')
    metadata('PRACTICAL_OUTPUT_SCOPE.md')
    metadata(Path(__file__).relative_to(ROOT))
    unknowns = [
        {'id': 'FINAL_CANDIDATE_AND_SCOPE', 'blocking': True, 'detail': 'u64 terminal artifact is frozen for DEV inference; FINAL challenger/candidate selection, exact arm set, learned contribution and relation to legacy own-DGP/five-milestone scope are not finalized. Broader preservation/source-separated DEV review must be closed first.'},
        {'id': 'INDEPENDENT_REVIEWERS_AND_DECISION_RULE', 'blocking': True, 'detail': 'No assigned independent human reviewers/adjudicator or prospectively agreed scope-specific acceptance/disagreement rule. Implementing assistant/root DEV review and informal user preference are separate evidence.'},
        {'id': 'PAIRED_FINAL_HOLDOUT', 'blocking': True, 'detail': 'No untouched paired FINAL cohort found in inspected frozen plans. Historical train/validation,358/520pairedDEV and53FFHQvalidation sources cannot be renamed FINAL. A new metadata-bound holdout/exposure/terms decision is needed before any paired FINAL claim.'},
        {'id': 'FINAL_INPUT_REVIEW_AND_EXECUTION_BINDING', 'blocking': True, 'detail': 'Reserved58 input-only decisions remain unstarted; exact candidate sources/components, masks/preparation, output names, device/threads, presentation mapping, finite call/wall/storage caps and partial-stop audit plan are unbound. No reserved opening or execution is authorized.'},
        {'id': 'ASSISTED_AUTOMATIC_COMPLETION_AND_TERMS', 'blocking': True, 'detail': 'Optional Protect/Apply and covering-family FINAL cohorts, automatic versus assisted rules and final app flow require separate frozen bindings. COFW64 is only a metadata candidate queue. Retained source terms are historical snapshots, not a new permission or live legal check.'},
    ]
    draft = {'format': 'codeformer-restoration-prospective-FINAL-protocol-draft-v1', 'complete': True,
        'status': 'UNFROZEN', 'protocol_frozen': False, 'execution_authorized': False,
        'reserved_pixel_opening_authorized': False, 'FINAL_results_exist_from_this_preparation': False,
        'candidate_artifact_metadata_available': True, 'FINAL_candidate_selected': False,
        'FINAL_candidate_binding': None, 'FINAL_arm_binding': None, 'independent_reviewers': None,
        'independent_adjudicator': None, 'paired_FINAL_cohort': None, 'covering_FINAL_cohort': None,
        'execution_limits': {'total_model_forward_cap': None, 'wall_seconds': None, 'per_forward_seconds': None,
                             'bytes_cap': None, 'automatic_retry_or_resume': False},
        'native_cohort': {'QMUL_cases': 32, 'QMUL_release_identities': 32, 'ChokePoint_cases': 26,
                          'ChokePoint_release_identities': 13, 'total_cases': 58, 'namespaced_labels': 45,
                          'native_clean_targets': False, 'favorable_replacements': False},
        'proposed_current_practical_contrast': {'arms_to_decide': ['same prepared resize input', 'unchanged U96', 'frozen u64'],
            'is_final_arm_selection': False, 'two_learned_arms_only_example_ceiling': 116,
            'ceiling_is_not_execution_budget_or_authorization': True,
            'legacy_five_arm_forms_cannot_be_silently_reused_for_different_models': True,
            'other_required_baselines_and_own_trained_contribution_scope': 'Explicit decision before FINAL; retain all original scope/failed evidence and accurately declare CodeFormer/R2 pretrained515 + U96adaptation24 + postfusion36head contribution.'},
        'stages': [
            {'stage': 'A_PREREQUISITES', 'rule': 'Close broader DEV review. Record exact model/canonical state/source/terms and all preprocessing, same display and refusal rules. Assign two independent human reviewers plus adjudicator and prospective source-specific decision rule. Keep existing protocols/forms unchanged; new versioned bindings required.'},
            {'stage': 'B_INPUT_ONLY', 'rule': 'Only after prior bindings authorize opening: both independent reviewers inspect native/prepared inputs alone, before outputs, arm labels, metrics or each other. Use readable/usable, needs_clearer/insufficient, out_of_scope, or unresolved pending adjudication. Preserve every selected ID and reason; no size-only/blur-only/dark-only rejection or output-based relabeling. Adjudicate before generation; no replacements.'},
            {'stage': 'C_EXECUTION_SEAL', 'rule': 'After input-only eligibility is fixed, bind exact eligible/refused58rows, arm labels and ordered finite calls, wall/storage caps, source/image/mask/model pins and failure accounting. Refused or unresolved cases have zero calls and explicit NOT_GENERATED_INPUT_EXCLUDED/PENDING rows. No hidden retry; retain reservations/returns/saves/unknown activity and partial evidence.'},
            {'stage': 'D_INDEPENDENT_SAVED_EVIDENCE', 'rule': 'Audit every original/prepared/support/geometry and model pin; FP32 RAW, clamp/native512-to256 policy and floorPNG separately; outside-support bytes exact. Audit all planned outputs/partial counters before any successful completion claim. Native has no paired PSNR/SSIM/identity truth. Paired FINAL, if separately frozen, uses only genuine aligned targets/support and prospectively pinned metric definitions.'},
            {'stage': 'E_BLIND_AND_SAME_DISPLAY', 'rule': 'Coordinator keeps per-case anonymous arm mapping, model names and DEV conclusions private until signed independent verdicts. Present all planned cases at identical256pixels with identical common display; no arm-specific enhancement. Keep RAW, native floorPNG and any prospectively fixed display views separate. Judge useful clarity over resize/incumbent, coherent visible eyes/nose/mouth/contour, expression/glasses/hair preservation, artifacts and acceptable softness; unverifiable fine detail stays uncertain.'},
            {'stage': 'F_USER_AND_SOURCE_SEPARATE_DECISION', 'rule': 'Record user acceptance separately with exact input/output/config hashes and date; positive earlier unbound preference is not an independent FINAL verdict. Preserve both reviewers and adjudication before unblinding. Report all selected/refused/uncertain cases, each source/size stratum and ChokePoint13person clusters; never count26frames as26independent people. No tuning on FINAL, silent omission, automatic pooled score or promotion from audit completion.'},
        ],
        'automatic_assisted_separation': {'baseline_restoration': 'Bypass optional blur/preparation unless a separately justified policy is prospectively frozen; no always-on blur.',
            'Protect_Apply': 'Original plus native Protect/Apply masks, exact protected/outsideApply input pixels, approximate contextual blur declaration, original/derived preview, per-edit invalidation, renewed confirmation and bypass. This is assisted preprocessing, not automatic restoration/identity preservation proof.',
            'covering_completion': 'Separate seven families and clear-glasses/uncovered/insufficient controls; freeze final cases/margins/automatic proposals and assistance. Preserve visible face outside intended removal; score proposal/removal/plausibility/preservation/joins separately; never count corrected masks as automatic success.',
            'full_goal_scope_unchanged': True, 'restoration_alone_closes_completion': False},
        'fixed_input_scope': cp['input_review_criteria'], 'blocking_decisions': unknowns,
        'no_exact_hidden_identity_country_ethnicity_or_Zamboanga_claim': True,
        'model_qualified': False, 'main_default_changed': False, 'goal_complete': False}
    readiness = {'format': 'codeformer-restoration-FINAL-metadata-readiness-v1', 'complete': True,
        'metadata_audit_pass': True, 'status': 'UNFROZEN', 'FINAL_execution_ready': False,
        'UTC': datetime.now(timezone.utc).isoformat(), 'sources': list(OPENED.values()),
        'source_separated_reserved': reserved, 'source_separated_development': development,
        'within_source_DEV_reserved_identity_intersections': overlaps,
        'QMUL_excluded_extension_identities': sorted(extension_ids),
        'reserved_identity_exclusion_from_future_training_required': True,
        'TRAIN_overlap_limits': {'known_native_selections': 'Native DEV and reserve roles are evaluation metadata, not optimization admission. Selected within-source release identities do not overlap.',
            'current_U96_u64': 'Frozen training manifests declare no native/FINAL optimization; this metadata audit does not establish globally unique people or exhaustive pretraining/historical exposure.',
            'cross_source_historical_pretrained_person_overlap': 'UNKNOWN; different namespace/hash/source does not prove identity separation',
            'country_ethnicity_capture_location': 'No per-case inference; no Zamboanga sample or performance claim'},
        'paired_holdout_readiness': {'untouched_paired_FINAL_identified': False,
            'historical_split': {'source': split_path, 'keys': list(split), 'train_paths': len(split['train']),
                'validation_paths': len(split['validation']), 'exact_path_intersection': 0,
                'train_source_counts': dict(Counter('/'.join(x.replace('\\', '/').split('/')[:2]) for x in split['train'])),
                'validation_source_counts': dict(Counter('/'.join(x.replace('\\', '/').split('/')[:2]) for x in split['validation']))},
            'adaptation_reference_roles': dict(Counter(r['historical_role'] for r in refs['references'])),
            'historical358': '53FFHQvalidation x6 +40olderpaired; allDEV repeated across development',
            'earlier520': {'cases': len(old520['cases']), 'references': len(old520['references']), 'split_limit': old520['split_limit'],
                'independent_FINAL': old520['independent_final_review']},
            'unused_paths_are_not_automatically_FINAL': True,
            'search_scope': 'Exact existing native selections, old independent protocols/forms, historicalPhase4 split, adaptation references, historicalpaired40/358/520 and COFW candidate metadata; no exhaustive unknown-history nonexposure claim'},
        'earlier_review_process': {'native_input_forms': 116, 'native_output_forms': 580, 'arms': old['arms'],
            'all_reviewers_and_verdicts_unfilled': True, 'coordinator_model_binding': None,
            'original_V1_R1_exact_unchanged': True, 'not_reusable_as_completed_current_FINAL': True},
        'source_terms': {'last_retained_primary_page_check': terms['checked_date'], 'fresh_live_check_performed': False,
            'records': terms['sources'], 'notice_files': [OPENED[p] for p in terms_paths],
            'CodeFormer_retained_notice': 'S-Lab1.0 noncommercial use; retain copyright/conditions/disclaimer and no endorsement; not new commercial permission',
            'copy_or_distribution_authorization': 'No reviewers contacted or copies distributed by this preparation; retain research restrictions and original/derivative notices.'},
        'other_FINAL_scope': {'COFW_metadata_candidate_count': cofw['candidate_count'],
            'COFW_covering_cohort_frozen': cofw['final_covering_cohort_frozen'], 'COFW_family_labels': cofw['family_coverage'],
            'COFW_is_paired_hidden_face_target': False, 'seven_family_completion_acceptance_not_established': True},
        'candidate_available_only_as_prior_DEV_artifact_metadata': {'inference_export_source': OPENED['outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/inference_export_v1.json'],
            'head_file_pin_as_recorded_not_reopened': model_receipt.get('head_file'),
            'fixed_DEV_arms_as_recorded': fixed_dev_plan['arms'], 'FINAL_model_decision': 'PENDING'},
        'blockers': unknowns, 'this_audit': {'metadata_files_opened': len(OPENED), 'reserved_image_files_opened': 0,
            'image_decodes': 0, 'generated_output_files_opened': 0, 'array_or_archive_reads': 0,
            'model_framework_imports_or_calls': 0, 'training_optimizer_VM_browser_network_calls': 0,
            'old_files_modified': 0, 'reserved_selection_changed': False},
        'model_qualified': False, 'main_app_default_changed': False, 'goal_complete': False}
    OUT.mkdir(parents=True)
    write(OUT / 'metadata_readiness_v1.json', readiness)
    write(OUT / 'draft_protocol_v1.json', draft)
    print(json.dumps({'complete': True, 'status': 'UNFROZEN', 'metadata_sources': len(OPENED),
                      'reserved_cases': 58, 'namespaced_labels': 45, 'blockers': len(unknowns), 'image_reads': 0}))


if __name__ == '__main__':
    main()
