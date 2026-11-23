"""Saved-only per-case/coverage/gradient-receipt tradeoff diagnosis. No model/VM."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/completion_codeformer_coverage_v2_return'
RETURN=PARENT/'cctv_completion_codeformer_coverage_v2_results'
AUDIT=PARENT/'independent_saved_output_audit.json'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
METRICS=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
TOL=1e-12
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def mean(values):return sum(values)/len(values)
def details(rows):
    return {metric:{'baseline_mean':mean([r['metrics'][metric]['baseline'] for r in rows]),'candidate_mean':mean([r['metrics'][metric]['candidate'] for r in rows]),'mean_delta_candidate_minus_baseline':mean([r['metrics'][metric]['delta_candidate_minus_baseline'] for r in rows]),'better':sum(r['metrics'][metric]['direction']=='better' for r in rows),'equal':sum(r['metrics'][metric]['direction']=='equal' for r in rows),'worse':sum(r['metrics'][metric]['direction']=='worse' for r in rows)} for metric in METRICS}
def main():
    started=time.monotonic();destination=PARENT/'independent_learning_tradeoff_diagnosis_v1.json';need(not destination.exists(),'Preserve prior diagnosis')
    need(sha(AUDIT)=='6ac24d0ac4ccbe4d54201e0c907a132b753c2ee84be35ea8adddf6cfde195fbb','Independentpixelbinding')
    audit=read(AUDIT);p=read(RETURN/'protocol.json');need(sha(RETURN/'protocol.json')=='c838f4472cf87c138b2d518ef4f543bfcec3e8149ea7a5411b863397bf66859d','Protocolbinding')
    original={r['path']:r for r in read(RETURN/'export_manifest.json')['files']}
    def evidence(path):
        row=original[path.relative_to(RETURN).as_posix()];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Evidencechanged');return path
    index={(r['cycle'],r['kind'],r['id']):r for r in audit['all_outputs']}
    comparisons={}
    for kind in ('monitor','challenge'):
        cases=p['monitor_cases' if kind=='monitor' else 'challenge_cases'];rows=[]
        for case in cases:
            ident=case['id'];baseline=index[(0,kind,ident)];candidate=index[(64,kind,ident)];metrics={}
            for metric in METRICS:
                before=baseline[metric];after=candidate[metric];delta=after-before;direction='worse' if delta>TOL else 'better' if delta < -TOL else 'equal'
                metrics[metric]={'baseline':before,'candidate':after,'delta_candidate_minus_baseline':delta,'relative_improvement_percent':100*(before-after)/max(before,1e-12),'direction':direction}
            rows.append({'id':ident,'reference_id':case['reference_id'],'mask_kind':case['mask_kind'],'condition':case['condition'],'style':case['style'],'mask_pixels':case['mask_pixels'],'all_four_endpoints_nonworse':all(v['direction']!='worse' for v in metrics.values()),'metrics':metrics})
        comparison={'cases':rows,'case_count':len(rows),'per_case_endpoint_direction_counts':details(rows),'cases_all_four_nonworse':sum(r['all_four_endpoints_nonworse'] for r in rows),'by_geometry':{kind:{'cases':len([r for r in rows if r['mask_kind']==kind]),'metrics':details([r for r in rows if r['mask_kind']==kind])} for kind in KINDS}}
        if kind=='challenge':comparison['by_condition']={condition:{'cases':len([r for r in rows if r['condition']==condition]),'metrics':details([r for r in rows if r['condition']==condition])} for condition in sorted({r['condition'] for r in rows})}
        comparisons[kind]=comparison
    gray=[]
    for case in p['challenge_cases']:
        if case['style']!='GRAY':continue
        ident=case['id'];before=index[(0,'challenge',ident)];after=index[(64,'challenge',ident)]
        gray.append({'id':ident,'mask_kind':case['mask_kind'],'target_style':'strict replicated GRAY','raw_channel_spread_before':before['GRAY_raw_mean_channel_spread'],'raw_channel_spread_after':after['GRAY_raw_mean_channel_spread'],'raw_channel_spread_delta':after['GRAY_raw_mean_channel_spread']-before['GRAY_raw_mean_channel_spread'],'PNG_channel_spread_before_0to255':before['GRAY_PNG_mean_channel_spread'],'PNG_channel_spread_after_0to255':after['GRAY_PNG_mean_channel_spread'],'PNG_channel_spread_delta':after['GRAY_PNG_mean_channel_spread']-before['GRAY_PNG_mean_channel_spread'],'candidate_PNG_hole_gain_percent':100*(before['PNG_hole_L1']-after['PNG_hole_L1'])/max(before['PNG_hole_L1'],1e-12),'chroma_is_diagnostic_not_new_gate':True})
    need(len(gray)==8,'EightstrictGRAYchallenges')
    # Public BW native case and its separately declared degraded counterpart.
    public_gray=[]
    for ident in ('val_25_hand_mouth_native','val_25_hand_mouth_degraded'):
        record={'id':ident,'reference':'unpaired public DEV photograph; no aligned hidden clean reference','condition':next(c['condition'] for c in p['development_cases'] if c['id']==ident)}
        for cycle,label in ((0,'R1'),(64,'V2')):
            row=index[(cycle,'public_development',ident)];path=Path(row['NPZ']['path']);need(sha(evidence(path))==row['NPZ']['sha256'],'PublicgrayNPZpin')
            with np.load(path,allow_pickle=False) as z:source=z['input256'].copy();mask=z['mask256'][0,0].astype(bool);result=z['result256'].copy()
            finite=np.isfinite(result).all() and result.dtype==np.float32;need(finite,'Finitepublicgray')
            png_path=Path(row['PNG']['path']);need(sha(evidence(png_path))==row['PNG']['sha256'],'PublicgrayPNGpin')
            from PIL import Image
            delivered=np.array(Image.open(png_path),copy=True)
            sourceRGB=source[0].transpose(1,2,0);resultRGB=result[0].transpose(1,2,0);spread=np.ptp(sourceRGB,axis=-1);rawspread=np.ptp(resultRGB,axis=-1);qspread=np.ptp(delivered.astype(np.int16),axis=-1)
            record[label]={'source_globally_strict_gray':bool((spread==0).all()),'source_mean_channel_spread_0to1':float(spread.mean()),'inside_raw_mean_channel_spread_0to1':float(rawspread[mask].mean()),'inside_PNG_mean_channel_spread_0to255':float(qspread[mask].mean()),'inside_PNG_max_channel_spread_0to255':int(qspread[mask].max()),'mask_pixels':int(mask.sum()),'outside_exact_already_audited':True}
        record['inside_PNG_mean_spread_delta']=record['V2']['inside_PNG_mean_channel_spread_0to255']-record['R1']['inside_PNG_mean_channel_spread_0to255'];record['interpretation']='Lower spread reduces false color in the strict-gray native input; the degraded counterpart already has colored input noise and is not a strict-gray input. No hidden-region fidelity claim or automatic desaturation rule follows.';public_gray.append(record)
    fit={r['id']:r for r in p['fit_cases']};cycles=[];losses=[];clip={'code':[],'fusion':[]};group_norms={};group_expected={'code':{'encoder':164,'position':1,'feature':2,'transformer':108,'prediction':3},'fusion':{'32':18,'64':18,'128':18}}
    for cycle in range(1,65):
        row=read(evidence(RETURN/'outputs'/('training_cycle_%04d.json'%cycle)));need(row['cycle']==cycle and row['microbatch_ids']==p['schedule'][cycle-1],'Actualschedule')
        ids=row['microbatch_ids'];need([r['id'] for r in row['code_losses']]==ids==[r['id'] for r in row['fusion_losses']],'Routesamesampler')
        expected_forward=244+cycle*8+32*sum(s<cycle for s in (16,32))
        need(row['counts']['forward_slots']==expected_forward and row['counts']['backward_calls']==cycle*8 and row['counts']['code_optimizer_steps']==row['counts']['fusion_optimizer_steps']==cycle,'Phasecounterreceipt')
        for route in ('code','fusion'):
            groups=row[route+'_gradient_groups'];need(set(groups)==set(group_expected[route]),'Gradientgroupnames')
            for name,g in groups.items():
                need(g['tensors']==group_expected[route][name] and g['finite'] and g['nonzero'] and math.isfinite(g['L2']) and g['L2']>0,'Finiteconnectedgradientgroup')
                group_norms.setdefault(route+'/'+name,[]).append(g['L2'])
            norm=row[route+'_preclip_L2'];need(math.isfinite(norm) and norm>0,'Finitepreclipnorm');combined=math.sqrt(sum(g['L2']**2 for g in groups.values()));need(abs(norm-combined)<=max(1e-5,1e-5*combined),'Globalgroupnormconsistency');clip[route].append(norm)
        for c,f in zip(row['code_losses'],row['fusion_losses']):
            case=fit[c['id']];values={k:v for k,v in c.items() if k!='id'}|{k:v for k,v in f.items() if k!='id'};need(all(math.isfinite(v) and v>=0 for v in values.values()),'Alllossesfinite')
            losses.append({'cycle':cycle,'id':c['id'],'role':case['role'],'condition':case['condition'],'mask_kind':case['mask_kind'],'style':case['style'],**values})
        cycles.append({'cycle':cycle,'microbatch_ids':ids,'code_preclip_L2':row['code_preclip_L2'],'fusion_preclip_L2':row['fusion_preclip_L2'],'counts':row['counts'],'receipt_seconds':row['seconds']})
    need(len(losses)==256 and len({r['id'] for r in losses})==256,'Everycaseonceperroute')
    components=('CE','latent_MSE','mask_RGB512_L1','inner_boundary_RGB512_L1')
    def loss_summary(rows):return {'cases':len(rows),'components':{name:{'mean':mean([r[name] for r in rows]),'minimum':min(r[name] for r in rows),'maximum':max(r[name] for r in rows)} for name in components}}
    by_condition={name:loss_summary([r for r in losses if r['condition']==name]) for name in sorted({r['condition'] for r in losses})}
    receipt_gradient_summary={route:{'cycles':len(values),'minimum_preclip_L2':min(values),'maximum_preclip_L2':max(values),'mean_preclip_L2':mean(values),'cycles_above_clip1':sum(x>1 for x in values),'cycles_at_or_below_clip1':sum(x<=1 for x in values),'clipping_threshold':1.,'all_receipt_norms_finite_positive':True} for route,values in clip.items()}
    group_gradient_summary={name:{'receipts':len(v),'minimum':min(v),'maximum':max(v),'all_finite_nonzero':True} for name,v in group_norms.items()}
    first_ids=set(cycles[0]['microbatch_ids']);last_ids=set(cycles[-1]['microbatch_ids'])
    need(not first_ids&last_ids,'Differentfirstlastpairsdocumented')
    gray_summary={'cases':8,'mean_R1_PNG_channel_spread_0to255':mean([r['PNG_channel_spread_before_0to255'] for r in gray]),'mean_V2_PNG_channel_spread_0to255':mean([r['PNG_channel_spread_after_0to255'] for r in gray]),'mean_delta':mean([r['PNG_channel_spread_delta'] for r in gray]),'cases_reduced_false_chroma':sum(r['PNG_channel_spread_delta'] < -TOL for r in gray),'cases_increased_chroma':sum(r['PNG_channel_spread_delta'] > TOL for r in gray),'mean_R1_raw_channel_spread_0to1':mean([r['raw_channel_spread_before'] for r in gray]),'mean_V2_raw_channel_spread_0to1':mean([r['raw_channel_spread_after'] for r in gray]),'selection_gate_added':False}
    options=[
      {'priority':1,'option':'Controlled route-isolation review from retained R1: freeze the predictor/encoder/code path and compare fusion-only continuation against the already observed jointly changed route.','why_evidence_supports_review':'All64 code updates exceeded the clipping threshold if receipts show64/64, and both prediction+fusion changed together. Clear regressions concentrate in eye-band/lens hole error while all boundaries improved; this does not identify which route caused them. Frozen R1 code prediction would isolate one factor without discarding the current incumbent.','limits':'Unproven direction, not a promise that fusion-only will work. V2 terminal partial is not a usable initialization. Any actual future pilot needs a fresh finite prospective protocol, identical fixed clear/challenge monitors, original comparisons and manual L4 launch. Do not change teacher/loss/data simultaneously if the purpose is route attribution.','prepared_or_launched':False},
      {'priority':2,'option':'Review anchor coverage and weighting only after route attribution, with original clear controls represented explicitly and the same nonworse subgroup checks.','why_evidence_supports_review':'The64cycles contained exactly128clearanchors+128augmentations. That coverage helped mean challenge loss, yet original clear hole quality regressed. Equal case counts do not establish equal gradients or preservation.','limits':'There is no saved matched gradient-per-case before/after comparison and no proof that simply adding more anchors will solve it. Avoid new DEV targets, retrospectively chosen identity splits or relaxed thresholds.','prepared_or_launched':False},
      {'priority':3,'option':'Source/label teacher-semantics review or isolated frozen-teacher ablation if predictor drift remains a demonstrated cause.','why_evidence_supports_review':'Current code loss uses the original frozen inpainting encoder prequantization latent plus nearest code indices. That is an inherited target definition, not an independently learned clean-face prior or a new reproduction of official HQ quantized teacher training.','limits':'The saved labels are finite and pinned, but no neural counterfactual/teacher replacement was run. A different teacher changes the learning objective and must be justified with audited terms/splits and matched finite evidence; never use public DEV or FINAL hidden regions as labels.','prepared_or_launched':False}
    ]
    findings=[
      {'status':'measured','finding':'The data-only broader-coverage continuation improved the16case adaptive synthetic TRAIN challenge average, but not every geometry. It worsened the fixed32case clear TRAIN monitor average. Both prospective cohorts therefore failed their required incremental R1 quality screen.'},
      {'status':'measured','finding':'Clear failures are six raw/PNG hole endpoints: macro, eye-band and image-left-lens. Boundary endpoints are nonworsening. Challenge failures are two raw/PNG cheek-object hole endpoints; condition means pass but geometry protection still fails.'},
      {'status':'interpretation_not_causal_proof','finding':'Exposure to matched gray, mildly degraded and irregular-mask inputs is consistent with the challenge improvement. Because both predictor and fusion were updated and the challenge includes the same eight adaptive TRAIN source identities, the return cannot establish which change caused improvement, broad real-covering generalization or an independent test gain.'},
      {'status':'measured','finding':'All saved outside-mask float/PNG pixels and eight saved empty bypasses remain exact. This compositing property is distinct from plausible anatomy and likeness inside the removal region. Passing exact outside preservation cannot excuse inside-hole endpoint regression.'},
      {'status':'measured','finding':'The new model remains better than original CodeFormer on the cumulative clear PNG hole endpoint, but the current acceptance target is incremental improvement over retained trained R1, which already outperformed original. Positive cumulative gain does not make a worse replacement acceptable.'},
      {'status':'operational_failure','finding':'All64pairedcycle receipts and all224output comparisons survived. The supervisor hard-killed at1800.592s while only an uncommitted terminal partial existed. Export completed successfully, but no deployable new checkpoint or final results receipt exists. Writer/checkpoint engineering is a separate issue from the measured quality failure.'},
      {'status':'receipt_scope','finding':'All64route receipts have finite positive intended-group norms, with512recorded backwards and128Adam updates. These receipts and source checks support the reported work; they are not an optimizer replay proof. Different cases are used once perroute, so first/last training loss values are not matched learning curves.'}
    ]
    prior=read(ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/independent_saved_output_audit.json')
    report={'format':'completion-codeformer-coverage-v2-independent-learning-tradeoff-diagnosis-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'source_pixel_audit':pin(AUDIT),'protocol':pin(RETURN/'protocol.json'),'findings':findings,'retained_incumbent':'Trained R1 remains the user-selected development model; original CodeFormer remains comparison. V2 is not promoted.','prospective_quality':{'clear_incremental_vs_R1':audit['clear_incremental_vs_R1'],'challenge_incremental_vs_R1':audit['challenge_incremental_vs_R1'],'clear_cumulative_vs_original':audit['clear_cumulative_vs_original'],'retained_R1_clear_cumulative_gain_percent':100*prior['terminal_independent_numeric_screen']['PNG_macro_hole_gain'],'all_prospective_screens_pass':False,'cohorts_never_pooled':True,'thresholds_unchanged':True,'case_equal_means_not_pixel_weighted':True},'per_case_group_decomposition':comparisons,'strict_gray_challenge_cases':gray,'strict_gray_challenge_summary':gray_summary,'public_gray_diagnostics':public_gray,'chroma_scope':'Strict gray targets make false chroma measurable on8paired TRAIN challenges. Public nativeBW is unpaired: channel consistency alone does not establish correct hidden anatomy/identity. Its noisy colored degraded counterpart is reported separately, without assuming it qualifies for global grayscale processing. No new chroma selection gate was added.','training_receipt_audit':{'completed_paired_cycles':64,'recorded_backward_calls':512,'recorded_code_steps':64,'recorded_fusion_steps':64,'total_recorded_new_Adam_updates':128,'code_trainable_tensors':278,'fusion_trainable_tensors':54,'all256microcase_losses_finite_nonnegative':True,'same256case_ids_once_each_route':True,'same_cycle_fourrefs_twoanchors_twoaugmented':True,'gradient_and_clipping_by_route':receipt_gradient_summary,'gradient_groups':group_gradient_summary,'all_losses_aggregate':loss_summary(losses),'by_condition_loss_levels_not_training_trends':by_condition,'first_cycle_loss_levels':loss_summary(losses[:4]),'last_cycle_loss_levels':loss_summary(losses[-4:]),'first_last_case_ids_disjoint':True,'first_cycle_ids':sorted(first_ids),'last_cycle_ids':sorted(last_ids),'first_last_loss_causal_comparison_valid':False,'reason':'Eachcase appears onlyonce perroute. The first and last four pairs are different references/targets/conditions; changes in levels cannot be interpreted as convergence, improvement or deterioration on matched data.','code_vs_fusion_causal_attribution_possible':False,'reason_no_attribution':'Both routes changed in every cycle; no frozen-route matched counterfactual or complete V2 checkpoint exists for retrospective inference.','cycles':cycles,'loss_records':losses,'optimizer_replayed':False,'norms_are_VM_receipts_not_independent_gradient_computation':True},'next_design_options_unproven_unprepared':options,'prohibited_shortcuts':['Do not extend epochs or rerun the unchanged failed jointly trained recipe.','Do not lower the1%incremental gate or drop failing geometries, boundary metrics or cohort cases.','Do not treat positive original-baseline cumulative gains as evidence that V2 surpasses R1.','Do not add DEV/FINAL targets or use independent final identities to tune the design.','Do not promote/export the incomplete terminal partial as a usable trained checkpoint.','Do not infer mask/sunglass/hair/scarf/object family readiness from synthetic challenge means or empty controls.'],'execution':{'model_loads':0,'neural_calls':0,'gradients':0,'optimizer_steps':0,'VM_calls':0,'app_source_weight_protocol_edits':0,'new_training_packet':False,'reserved_FINAL_access':False},'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
    need(sha(AUDIT)==report['source_pixel_audit']['sha256'],'Auditsourcestable')
    with destination.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'receipt':pin(destination),'clear_cases_all_four_nonworse':comparisons['monitor']['cases_all_four_nonworse'],'challenge_cases_all_four_nonworse':comparisons['challenge']['cases_all_four_nonworse'],'gradient_clipping':receipt_gradient_summary,'gray_summary':gray_summary,'public_gray':public_gray,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()
