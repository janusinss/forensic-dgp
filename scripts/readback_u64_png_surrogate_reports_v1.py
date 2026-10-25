"""Read-only stdlib report/design evidence and status review, without arrays.

Rehash retained receipt inputs, compare published numbers, independently check
the small16x16 certificate arithmetic, and audit prospective selection rules.
No worker, neural framework, parameter application, VM, or browser execution.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit'
REPORT=ROOT/'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md'
DESIGN=ROOT/'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md'
SPEC=importlib.util.spec_from_file_location('pure_json_pins',ROOT/'scripts/audit_u64_role_gradients_return_transport_v1.py')
F=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(F)
need,strict_json,pin=F.need,F.strict_json,F.pin
ROLES=('local_gaussian_1p2','global_gaussian_1p2','clear','lowlight_linear_0p35')
QUERIES=('raw_mae','raw_adjacent','png_surrogate_mae','png_surrogate_adjacent')


def main():
    consumed={};identities={};checks=[]
    def checked(path,expected=None):
        path=Path(path);key=path.relative_to(ROOT).as_posix()
        if key not in consumed:
            consumed[key]=pin(path);s=path.stat();identities[key]=(s.st_ino,s.st_size,s.st_mtime_ns)
        actual=consumed[key]
        if expected is not None:
            need({k:actual[k] for k in ('bytes','sha256')}=={k:expected[k] for k in ('bytes','sha256')},'Current consumed receipt input differs: '+key)
        return actual
    def doc(path):checked(path);return strict_json(path.read_bytes())
    def check(name,ok,evidence=None):checks.append({'name':name,'pass':bool(ok),'evidence':evidence})
    def text(path):checked(path);return path.read_text(encoding='utf-8')
    raw_report=text(REPORT);raw_design=text(DESIGN)
    flat=lambda t:' '.join(t.replace('**','').replace('`','').split())
    report,design=flat(raw_report),flat(raw_design)
    compact=lambda t:re.sub(r'\s+','',t)
    dr=compact(design)
    T=doc(ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_transport/receipt_v1.json')
    P=doc(ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels/receipt_v1.json')
    G=doc(ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_gradients/receipt_v1.json')
    V=doc(OUT/'visual_readback_v1.json');R=doc(OUT/'root_visual_review_v1.json');E=doc(OUT/'feasibility_v1.json')
    PX=doc(ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels/consumed_files_v1.json')
    protocol_path=ROOT/'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm/protocol.json'
    Q=doc(protocol_path)
    old_protocol=doc(ROOT/'outputs/cctv_codeformer_u64_gate_aligned_pilot_v1_vm/protocol.json')
    # Every pin consumed by the scientific and visual receipts is verified
    # against current bytes, once per unique path; arrays are hashed only.
    bound_count=0
    for evidence in (T,G,V,E):
        bindings=evidence['consumed_files']
        rows=bindings if isinstance(bindings,list) else [dict(path=k,**v) for k,v in bindings.items()]
        for value in rows:checked(ROOT/value['path'],value);bound_count+=1
    for value in PX['files']:checked(ROOT/value['path'],value);bound_count+=1
    checked(ROOT/V['root_visual_review_bound']['path'],V['root_visual_review_bound'])
    checked(ROOT/V['visual_bindings_bound']['path'],V['visual_bindings_bound'])
    for page in R['reviewed_pages']:checked(ROOT/page['source']['path'],page['source'])
    for field in ('archive','export','sidecar','protocol'):checked(ROOT/T[field]['path'],T[field])
    check('all_current_receipt_input_file_pins_exact',True,{'references':bound_count,'unique_paths':len(consumed),'arrays_decoded':0})
    sha='0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'
    check('all_scientific_receipts_complete_and_same_seal',all(v['complete'] is True for v in (T,P,G,V,R,E)) and
        T['transport_audit_pass'] and P['saved_pixel_scalar_consistency_pass'] and G['saved_state_gradient_geometry_audit_pass'] and
        V['visual_readback_pass'] and all(v['protocol_sha256']==sha for v in (P,G,V,E)) and
        T['protocol']['sha256']==sha and not T['unresolved_transport_blockers'] and not P['failed_checks'] and not V['unresolved_blockers'])
    check('report_archive_protocol_and673_payload_numbers',T['archive']['bytes']==979202030 and
        T['framing']['regular_members']==673 and T['framing']['payload_bytes']==1093522436 and
        T['sealed_static_sources_verified']==21 and T['parent_static_bindings_rehashed']==469 and len(T['local_model_aliases'])==2 and
        all(s in report for s in ('979,202,030','1,093,522,436','673 unique regular USTAR','21 static bindings','Twelve approved','469 parent files','two explicitly declared model aliases',sha,T['archive']['sha256'])))
    J=T['journal'];W=T['runtime']['worker'];S=T['runtime']['status']
    check('report992events328pairs72forward256query_zero_unknown',J['journal_events_replayed']==992 and
        J['completed_operation_pairs']==J['maximum_operation_pairs']==328 and J['full_exact_plan_completed'] and
        J['counters']['full_forward_returns']==72 and J['counters']['gradient_query_returns']==256 and
        not J['unknown_reservations'] and not J['operation_outcome_unknown'] and
        all(s in report for s in ('992 events and 328 completed','72 forwards, 256 gradient queries','zero parameter applications','zero optimizer constructions','no pending unknown operation')))
    check('report_actual_worker_exit_cleanup_timers_and_no_export_exit_invention',W['exit_code']==0 and W['timeout'] is False and
        W['child_pid']==4497 and W['cleanup']['child_reaped'] and W['cleanup']['process_group_absent'] and
        f"{S['elapsed_seconds']:.3f}"=='128.140' and f"{W['elapsed_seconds']:.3f}"=='130.919' and
        T['runtime']['empty_run_log_valid'] and len(T['runtime']['post_export_receipts_absent_by_design'])==8 and
        T['runtime']['final_export_exit_code_available'] is False and all(s in report for s in
        ('Worker exit 0','PID 4497','128.140 seconds','130.919 seconds','empty worker log is legal','Eight final export-process receipts','final export exit is unavailable')))
    counts=P['counts']
    check('report1642_pixel_checks_and72_64_256_4_4',P['checks']==1642 and counts['RAW_floor_PNG_pairs']==72 and
        counts['actual_PNG_decoded_FP32_surrogate_forward_references']==64 and counts['photographic_query_scalars']==256 and
        counts['historical_u64_RAW_PNG_parities']==4 and counts['full_six_output_repeats']==4 and
        P['quality_tolerance_added']==0 and all(s in report for s in
        ('1,642 checks','72 exact RAW/floor-PNG pairs','64 bitwise-exact','256 target-bound scalars','four historical original-u64 parities','four complete repeats','not a changed quality tolerance')))
    check('report575_state36head539frozen_and_canonical_pins',G['all575_initial_closing_exact_original_u64'] and
        G['tensor_census']=={'heads':36,'base':515,'adapters':24} and len(G['full575_tensor_rows'])==575 and
        G['full575_canonical_sha256'] in report and G['original_full_NPZ_sha256'] in report and
        all(s in report for s in ('All 575 tensors remain exactly original trained u64','36 heads and 539 R2/U96 tensors')))
    X=G['independent_geometry_crosscheck']
    check('report256_gradient36tensor168960_coordinates_and18306_crosscheck',G['gradient_NPZ_files']==256 and
        G['head_tensors_each']==36 and G['coordinates_each']==168960 and G['zero_tensor_receipts']==G['unused_tensor_receipts']==G['complete_zero_vectors']==0 and
        X['pass'] and X['numeric_leaves']==18306 and f"{X['maximum_absolute_error']:.2e}"=='3.55e-15' and
        G['RNG_and_fixed_sampler_null_optimizer_receipts_pass'] and all(s in report for s in
        ('256 saved FP32 gradient witnesses','168,960 coordinates','No missing, unused, all-zero tensor or complete zero vector','18,306 numerical values','3.55e-15','does not independently rerun the VM')))
    check('report12pages204cells68outputs4repeat_aliases',V['planned_pages']==V['readback_pages']==V['root_review_pages_bound']==12 and
        V['exact_RGB256_cells']==204 and V['distinct_output_cells']==68 and len(V['repeated_witness_aliases'])==4 and
        V['all72_outputs_covered_once_or_exact_alias'] and R['root_personally_inspected_all_planned_sheets'] and len(R['reviewed_pages'])==12 and
        all(s in report for s in ('all 12 exact RGB256 sheets','204 cells','68 distinct output cells plus four byte-identical repeat aliases','no newly trained comparison')))
    rows=re.findall(r'^\| (Local Gaussian, central128|Global Gaussian, whole256|Clear, whole256|Low light, whole256) \| ([0-9.]+) \| ([0-9.]+) \|$',raw_report,re.M)
    aligns=G['pooled_role_mean16']['RAW_vs_PNG_surrogate_by_role']
    check('all8_displayed_pooled_RAW_PNG_cosines_correctly_rounded',len(rows)==4 and all(
        row[1]==f"{aligns[role]['mae']['cosine']:.6f}" and row[2]==f"{aligns[role]['adjacent']['cosine']:.6f}"
        for row,role in zip(rows,ROLES)),{r:{q:aligns[r][q]['cosine'] for q in ('mae','adjacent')} for r in ROLES})
    per=G['RAW_PNG_surrogate_alignment_per_photo']
    positive=sum(per[r][q]['positive'] for r in ROLES for q in ('mae','adjacent'))
    pair=G['pooled_role_mean16']['pairwise16'];conflicts=[pair[14][i]['cosine'] for i in (2,6,10)]
    check('report128positive_per_photo_and_lowlight_conflicts',positive==128 and all(per[r][q]['count']==16 and
        per[r][q]['negative']==per[r][q]['zero']==per[r][q]['null']==0 for r in ROLES for q in ('mae','adjacent')) and
        '-0.7370/-0.9158/-0.8627' in report and all(f'{v:.4f}' in report for v in conflicts),{'all_positive':positive,'PNG_MAE_conflicts':conflicts})
    solver=E['solver'];syn=E['synthetic_checks'];historical=E['comparison_authenticated_historical8RAW_u1_direction'];new8=E['comparison_new8RAW_only_direction']
    check('offline16solver_all65535faces_synthetic_KKT_bound_and_no_model_calls',E['outside_returned_measurement_protocol'] and
        E['actual_gradient_files_reconstructed']==256 and E['all16_mandatory_role_domain_gate_vectors_retained'] and
        solver['faces_enumerated']==65535 and solver['accepted_faces']+solver['rejected_faces']==65535 and
        solver['elapsed_seconds']<=solver['deadline_seconds']==120 and syn['pass'] and solver['independent_actual_KKT']['pass'] and
        E['new_model_or_optimizer_or_actual_gradient_or_parameter_VM_calls']==0 and not E['direction_or_parameter_NPZ_written'] and
        all(s in report for s in ('all 16 role/domain/metric vectors','normalized after pooling','65,535 nonempty faces','KKT certificate')))
    # Independently recompute the retained small Gram/simplex certificate only.
    gram=solver['unit_gram'];alpha=solver['simplex_coefficients'];n=16
    need(len(gram)==len(alpha)==n and all(len(row)==n for row in gram),'Exact16x16 certificate required')
    products=[math.fsum(gram[i][j]*alpha[j] for j in range(n)) for i in range(n)]
    squared=math.fsum(alpha[i]*products[i] for i in range(n));norm=math.sqrt(squared)
    predicted=[-x/norm for x in products];actual=solver['actual_unit_gradient_dot_direction']
    error=max(abs(a-b) for a,b in zip(predicted,actual));active=[i for i,a in enumerate(alpha) if a>1e-8]
    kkt=max(max(abs(products[i]-squared) for i in active),max(0,squared-min(products)),abs(math.fsum(alpha)-1))
    check('independent_stdlib16x16_simplex_and_dot_certificate',min(alpha)>=-1e-12 and error<3e-12 and kkt<1e-8 and
        abs(norm-solver['minimum_norm_combination_norm'])<3e-12 and
        abs(-max(actual)-solver['minimum_normalized_descent_margin'])<3e-12,
        {'dot_max_error':error,'KKT_max_error':kkt,'certificate_only_no_direction_or_arrays':True})
    check('report_prior_direction_all16negative_and_nearidentical_margin',historical['all16_negative'] and len(historical['all16_unit_gradient_dots'])==16 and
        all(x<0 for x in historical['all16_unit_gradient_dots']) and abs(historical['minimum16_normalized_margin']-0.17981636)<5e-9 and
        abs(solver['minimum_normalized_descent_margin']-0.18201706)<5e-9 and
        abs(historical['cosine_with_full16_direction']-0.99999338198)<5e-12 and
        abs(new8['cosine_with_full16_direction']-historical['cosine_with_full16_direction'])<1e-12 and
        all(s in report for s in ('negative dots against all 16 pooled','0.17981636','0.18201706','0.99999338198')))
    dots=[v for row in E['all64_rolecase_four_query_dots'] for v in row['four_query_gradient_dots_hypothetical16_direction'].values()]
    decreases=sum(v<0 for v in dots);increases=sum(v>0 for v in dots)
    check('report233negative23positive_individual_dots',len(dots)==256 and decreases==233 and increases==23 and
        '233 individual query dots are negative and 23 positive' in report,
        {'negative':decreases,'positive':increases,'zero':sum(v==0 for v in dots)})
    check('report_no_true_PNG_derivative_or_finite_improvement_claim',P['true_PNG_derivative_claimed'] is False and
        E['finite_step_or_actual_PNG_gain_certified'] is False and all(s in report for s in
        ('biased surrogate','not the true derivative','no new finite displacement at all','not image quality scores','not a demonstrated cause','not test a materially different corrective mechanism')))
    check('report_no16pilot_no_oldstate_selection_no_newguide',not E['new_training_recipe_selected'] and all(s in report for s in
        ('No such pilot is prepared or authorized to auto-run','old update-one state remains unselected','failed update-two state remains failed',
         'there is no next launch guide yet','Do not rerun this completed diagnostic')))
    check('report_domain_incumbent_no_promotion_goal_boundaries',all(v['model_qualified'] is False for v in (T,P,G,V,R,E)) and
        all(s in report for s in ('paired TRAIN cases','U96 stays the experimental app incumbent','original u64 remains the retained challenger',
        'Broader DEV failures remain binding','Native/wild remain unpaired','FINAL remains unopened and execution-unfrozen',
        'Automatic blur matching and completion development remain deferred','No model promotion','ACTIVE/incomplete')))
    oldlimits=old_protocol['pilot']['limits']
    check('design_fixed_four_scales_preserve_original_step_and_cumulative_caps',
        oldlimits['maximum_absolute_coordinate_step']==2.5e-5 and oldlimits['maximum_cumulative_absolute_coordinate_delta']==1e-4 and
        all(s in design for s in ('2e-5, 1.5e-5, 1e-5, 5e-6','fixed 0.99 numerical headroom','original 2.5e-5 per-application bound',
        '1e-4 original-relative cumulative bound','Each restore and final selected application needs the same actual rounded bounds')),
        {'scales':[2e-5,1.5e-5,1e-5,5e-6],'per_application_cap':2.5e-5,'original_relative_cap':1e-4,'headroom':.99})
    check('design_distinct_actual_PNG_selection_not_nearidentical16gradient_pilot',all(s in design for s in
        ('same direction as the rejected RAW-only pilot','actual TRAIN PNG outcomes','original-u64 eight-RAW pooled direction',
         'already audited 256-query measurement','No Adam/momentum object, new autograd query or unbounded line search')))
    check('design_independent_original_initialization_and_8RAW_rounded_descent',all(s in design for s in
        ('Initialize exact original trained u64','Never initialize from failed update two','never from the preceding trial',
        'all eight strictly negative RAW pooled gradient dots after rounding','restore is explicitly labeled as a reversal, not a descent step',
        'destination must equal original heads bitwise','all 539 base and adapter tensors bitwise')))
    check('design_all16_actual_RAW_PNG_zero_tolerance_and_strict_blur',all(s in design for s in
        ('all 64 optimization cases','16 actual RAW/PNG role-mean MAE/adjacent endpoints','local central128 and other whole256 regions',
        'all 16 means to be no greater than original u64','both blur PNG adjacent means strictly smaller','Quality comparisons keep zero tolerance')))
    check('design_complete_fixed_trial_banks_then_first_eligible_or_stop',all(s in design for s in
        ('all four complete banks with exactly the same 64 IDs, target bindings and ROIs',
        'first eligible candidate in the predeclared order only after retaining all four trials',
        'If none qualifies, retain the stop; do not append scales','quality-ineligible trial does not cancel the registered remaining trials')))
    check('design_selector_no_monitor_guard_inputs_and_terminal_monitor_fail',all(s in design for s in
        ('Its API accepts no monitor or guard records','monitor/guard bank is excluded from candidate selection',
        'execute its 56 cases once','monitor/guard failure rejects the entire test','do not try the next candidate',
        'No monitor output may leak into candidate selection or scale choice','There is no monitor-driven fallback')))
    check('design_all_inherited_U96_and_original_quality_screens_retained',all(s in design for s in
        ('Preserve all original inherited gates','1% blur improvement against frozen U96','clear/low-light requirements',
        'additional 16 endpoint nonincrease checks against original u64','relax a threshold or resume training')))
    check('design_all9_actual_parameter_writes_restores_and_final_commit_journaled',4+4+1==9 and all(s in design for s in
        ('one selected training update','four candidate trials, four explicit restores to original','nine parameter applications total',
        'Trial writes and restores must be journaled','Restore original heads after every trial, including the fourth',
        'Any execution, nonfinite, bound, frozen tensor, RNG, copy or resource invariant failure immediately stops',
        'partially applied write remains unknown; no restore may erase that outcome')))
    check('design436forward_budget_and380_no_candidate_prefix',56+4+64+4*64+56==436 and 56+4+64+4*64==380 and
        all(s in dr for s in ('completeforwardbudgetis436','initial56qualitycases,foursame-staterepeats,initial64optimizationcases,four64-casetrialbanks',
        'oneselected56-casefinalbank','Ifnocandidateiseligible,thelast56andfinalapplicationarenotattempted')),
        {'complete_forwards':436,'no_eligible_candidate_forwards':380,'complete_parameter_applications':9,'no_eligible_candidate_applications':8})
    check('design_prospective_finite_resources_and_manual_no_autolaunch',all(s in dr for s in
        ('worker1200seconds,external1260plus20cleanup','eachphase180seconds','Export300/330seconds','Require8GiBfree',
        'retain4GiB','RAM10GiB','reservedCUDA18GiB','outputs/archive2GiBeach','Exactruntimeisunmeasured',
        'designrequirements,notimplementedlimits')) and all(s in design for s in
        ('Prospective design only: no runnable worker, sealed transfer or launch guide exists','separate pure-array selector prototype may support source tests; it does not implement or execute the VM test',
        'No developer or agent auto-launches the test',
        "user's manual L4 workflow")))
    check('design_before_seal_pure_tests_independent_review_and_all_returned_sheets',all(s in design for s in
        ('Before sealing, pure tests must cover FP32 candidate and restore accounting','fixed four-scale order','all16 eligibility endpoints',
        'quantization-code crossing fixtures','no monitor selection leakage','zero-candidate stop',
        'Independent review must establish actual trial/restore/commit counters','full timers and failure capture before any runnable guide',
        'Independently audit returns and all planned sheets','Preserve all historical immutable packets and their failed states')))
    check('design_no_quality_generalization_incumbent_FINAL_promotion_claim',all(s in design for s in
        ('successful one-step screen is not a useful-restoration, native CCTV, FINAL or app-promotion result',
        'does not establish that a different scalar will fix discrete PNG behavior or generalize','stop this recipe',
        'U96 and original u64 remain retained','Native/wild stay unpaired','ACTIVE/incomplete')))
    checked(Path(__file__));checked(Path(F.__file__))
    for name,identity in identities.items():
        s=(ROOT/name).stat();need((s.st_ino,s.st_size,s.st_mtime_ns)==identity,'Consumed file changed after readback: '+name)
    # Fresh second hashes of the small current reports and authoritative receipts.
    for path in (REPORT,DESIGN,ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_transport/receipt_v1.json',
        ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels/receipt_v1.json',
        ROOT/'outputs/codeformer_u64_png_surrogate_measurement_v1_return_gradients/receipt_v1.json',
        OUT/'visual_readback_v1.json',OUT/'root_visual_review_v1.json',OUT/'feasibility_v1.json'):
        need(pin(path)==consumed[path.relative_to(ROOT).as_posix()],'Small report/evidence bytes changed after review')
    blockers=[r['name'] for r in checks if not r['pass']]
    value={'complete':True,'report_design_review_pass':not blockers,'return_report_review_pass':not any('report' in n for n in blockers),
        'prospective_design_review_pass':not any('design' in n for n in blockers),'unresolved_blockers':blockers,
        'created_UTC':datetime.now(timezone.utc).isoformat(),'report':consumed[REPORT.name],'design':consumed[DESIGN.name],
        'checks_pass':sum(r['pass'] for r in checks),'checks':checks,'consumed_files':list(consumed.values()),
        'all_current_bound_input_hashes_exact':True,'all_consumed_file_identities_unchanged_after':True,
        'current_reports_and_authoritative_receipts_rehashed_after':True,
        'resolved_initial_design_finding':{'finding':'First reviewed scales1e-4/7.5e-5/5e-5/1.25e-5 exceeded inherited2.5e-5per-application bound.',
            'root_repair':'Current reviewed design uses2e-5/1.5e-5/1e-5/5e-6 and retains actual rounded step/cumulative bounds for trials/restores/final.',
            'status':'resolved_at_exact_current_design_pin'},
        'scope':'Independent report numerical/status readback and prospective scientific/control-flow design review against authenticated current receipts. Scientific audits are joined, not rerun; only small retained certificate arithmetic is independently recomputed.',
        'external_paper_guarantees_imported':False,'scientific_array_decodes_new_gradients_worker_model_parameter_VM_browser_calls':0,
        'new_worker_implementation_or_packet_or_guide_approved':False,'model_qualified':False}
    target=OUT/'report_design_readback_v1.json'
    with target.open('x',encoding='utf-8',newline='\n') as stream:
        import json
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'receipt':pin(target),'checks_pass':value['checks_pass'],'checks_total':len(checks),
        'unresolved_blockers':blockers,'report':value['report'],'design':value['design']},indent=2))


if __name__=='__main__':main()
