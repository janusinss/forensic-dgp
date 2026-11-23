"""Freeze a distinct zero-update spatial-code diagnostic before packet sealing."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design'
OLD_STAGE=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm'
OLD_RETURN=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results'
CURRENT=ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'

def need(ok,label):
    if not ok:raise ValueError(label)
def physical(path):
    p=Path(path).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(physical(path).read_text(encoding='utf-8-sig'))
def new(path,data):
    with physical(path).open('xb') as f:f.write(data)

def main():
    need(not DEST.exists(),'Preserve earlier frozen design')
    old=read(OLD_STAGE/'protocol.json')
    need(sha(OLD_STAGE/'protocol.json')==sha(OLD_RETURN/'protocol.json')=='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec','Exact previous diagnostic protocol')
    cases=old['monitor_cases']+old['challenge_cases']
    need(len(cases)==48 and len({r['id'] for r in cases})==48,'Fixed48 cases')
    bridge=['tr_ffhq_05179__lower_face','tr_ffhq_05302__eye_band','v2m_05179_g0_gd','v2m_06781_g2_gd']
    need(set(bridge)<={r['id'] for r in cases},'Four declared RGB/GRAY bridge cases')
    evidence={
      'actual_current_return_milestone':pin(CURRENT/'return_milestone_v1.json'),
      'current_actual_results_report':pin(ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_RESULTS.md'),
      'current_independent_pixels':pin(CURRENT/'independent_saved_output_audit.json'),
      'current_root_all20_visual':pin(CURRENT/'root_visual_review.json'),
      'current_root_code_geometry':pin(CURRENT/'root_saved_code_geometry_diagnosis.json'),
      'current_state_Adam_CE':pin(ROOT/'research/completion_codeformer_predictor_ce_v1_state_review/independent_state_Adam_logits_audit.json'),
      'current_independent_return_closure':pin(ROOT/'research/completion_codeformer_predictor_ce_v1_state_review/independent_return_milestone_closure.json'),
      'FIT_CE_allocation':pin(ROOT/'research/completion_codeformer_post_predictor_ce_v1_review/FIT_loss_geometric_decomposition.json'),
      'independent_scientific_review':pin(ROOT/'research/completion_codeformer_post_predictor_ce_v1_review/independent_scientific_review.md'),
      'independent_scientific_admission':pin(ROOT/'research/completion_codeformer_post_predictor_ce_v1_review/scientific_review_admission.json'),
      'retained_oracle_actual_pixel_audit':pin(OLD_RETURN.parent/'independent_saved_output_audit.json'),
      'retained_oracle_actual_state_audit':pin(OLD_RETURN.parent/'independent_state_code_audit.json'),
      'retained_oracle_actual_transport':pin(OLD_RETURN.parent/'transport_audit.json'),
      'retained_oracle_actual_protocol':pin(OLD_RETURN/'protocol.json'),
      'R1_source_admission':pin(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json'),
    }
    expected={'actual_current_return_milestone':'181b00064b9e3c4862095d651c8298a2e3f00b75d6c8f8c2071c8806a0c0c1d9','current_actual_results_report':'85f1aef910241b3d91a3f8ca29740d37dabb4585ce6acf949ab4d231452d0aa0','current_independent_return_closure':'f28bfdd6b9f9ba66bba80e10e92d7321709bcd05fb05333e0b5c855f02398af9'}
    for k,h in expected.items():need(evidence[k]['sha256']==h,'Current actual evidence pin '+k)
    controlfiles=[];controlreports={}
    for arm in old['arms']:
        if arm['id'] not in ('A','C'):continue
        reports={}
        for kind in ('monitor','challenge'):
            folder=OLD_RETURN/'outputs'/arm['filename_tag']/kind
            reports[kind]=read(folder/'results.json')
            for path in sorted(folder.iterdir()):
                need(physical(path).is_file() and path.suffix in ('.json','.png','.npz'),'Closed retained control directory')
                actual=pin(path);actual['relative_output_path']=path.relative_to(OLD_RETURN/'outputs').as_posix();controlfiles.append(actual)
        controlreports[arm['id']]=reports
    need(len(controlfiles)==196 and len({r['relative_output_path'] for r in controlfiles})==196,'Exactly96PNG96NPZ4result controls')
    # Plain integer serialization bounds: no model import or forward.
    parity_data=786432+262144+4*3145728+2*2048+2*786432
    new_data=786432+262144+786432+3145728+2*2048+256+2
    maximum_array=(4*(parity_data+65536))+(100*(new_data+65536))
    need(maximum_array<560*1024**2,'Worst numeric payload/NPYZIP reserve within560MiB')
    budgets={'arrays_bytes':560*1024**2,'PNG_bytes':24*1024**2,'metadata_bytes':8*1024**2,'failure_actual_state_bytes':384*1024**2}
    need(sum(budgets.values())==976*1024**2<1024**3,'976MiB bounded categories inside1GiB output')
    contract={
      'forward_slots_planned':108,'forward_slots_max':120,'typed_forward_slots_exact':{'parity':8,'fresh_C_bridge':4,'H':48,'V':48},
      'optimizer_objects':0,'optimizer_updates':0,'gradient_queries':0,'backward_calls':0,'teacher_neural_calls':0,
      'state_tensors_frozen':497,'fusion_w':1,'codebook_size':512,'precision':'FP32_no_autocast_no_TF32',
      'worker_hard_seconds':900,'worker_cooperative_seconds':840,'finish_reserve_seconds':60,'export_hard_seconds':900,
      'minimum_free_bytes':6*1024**3,'stop_free_bytes':2*1024**3,'CUDA_allocated_bytes_cap':20*1024**3,
      'output_bytes':1024**3,'raw_tar_bytes_cap':1280*1024**2,'gzip_bytes_cap':1280*1024**2,'output_category_caps':budgets,
      'NPZ_count':104,'new_HV_PNG_count':96,'parity_NPZ_count':4,'fresh_C_bridge_NPZ_count':4,'normal_checkpoint_files':0,
      'failure_checkpoint_files_at_most':1,'normal_return_regular_member_cap':400,'relative_path_UTF8_bytes':240,
      'metadata_file_bytes_cap':1024**2,'trainer_log_bytes_cap':4*1024**2,'NPZ_numeric_members_max':10,'NPY_header_bytes_cap':512,
      'single_new_HV_or_bridge_NPZ_bytes_cap':6*1024**2,'single_parity_NPZ_bytes_cap':16*1024**2,'single_PNG_bytes_cap':256*1024,
      'no_quality_based_early_stop':True,'automatic_launch':False,'automatic_resume':False,'automatic_promotion':False,
    }
    design={
      'format':'completion-codeformer-spatial-code-intervention-v1-root-design-v1','status':'PROSPECTIVE_SOURCE_PREPARATION_NOT_READY_NO_ACTUAL_RUN','created_UTC':datetime.now(timezone.utc).isoformat(),
      'purpose':'Distinguish geometric removal-region clean-code effects from surrounding-context clean-code effects before changing completion training again.',
      'working_model':'R1 remains selected','actual_failed_trial':'Predictor CE V1:32updates,validstate,failed1percent/preservation/style requirements; no unchanged repeat or appended epochs.',
      'models':{'original_checkpoint':old['checkpoint'],'exact_full_R1_parent':old['parent'],'original_parent_protocol_sha256':'8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d','all497_frozen_R1_logical':'55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6','no_predictor_CE_terminal_initialization':True},
      'data':{'case_ids_in_fixed_protocol_order':[r['id'] for r in cases],'monitor_cases':32,'challenge_cases':16,'monitor_reference_ids':[r['id'] for r in old['references']],
          'bridge_case_ids':bridge,'bridge_readouts':'Four new full-C oracle NPZs; compare raw512/result256/codes and in-memoryPNG to corresponding oldC. Do not save extra bridge PNGs.',
          'parity_cases_exact':old['parity_cases'],'oracle_label_sets':16,'original16labels_exact_only':old['oracle_labels'],
          'fixed_masks':True,'new_data':0,'new_teacher_labels':0,'clean_RGB_pasted':False,'native_unpaired_cases_new':0,'reserved_FINAL_used':False,
          'scope':'Eight already adaptive TRAIN photograph references,32clear+16syntheticchallenge cases. Not independent people/pretraining separation, FINAL or nativeCCTV truth.'},
      'grid_rule':{'source':'Original reviewed binary256x256 mask','shape':[16,16],'row_major':True,'rule':'Any nonzero reviewed removal pixel in each corresponding16x16 original-image block marks that code cell touched. Complement is zero removal pixels.','fixed_threshold':'any','no_dilation_or_threshold_search':True,'not_a_receptive_field_map':True,'delivered256_mask_unchanged':True},
      'conditions':[
        {'id':'A','type':'retained_control','codes':'All ordinary R1 predicted indices','fusion_w':1,'new_forwards':0,'deployable_control':True},
        {'id':'H','type':'new_ORACLE_unavailable_clean_information','codes':'Touched cells use exact retained same-codebook clean-context indices; complement keeps ordinary predicted indices.','fusion_w':1,'new_forwards':48,'deployable':False},
        {'id':'V','type':'new_ORACLE_unavailable_clean_information','codes':'Touched cells keep ordinary predicted indices; complement uses exact retained same-codebook clean-context indices.','fusion_w':1,'new_forwards':48,'deployable':False},
        {'id':'C','type':'retained_ORACLE_control_unavailable_clean_information','codes':'All cells use exact retained same-codebook clean-context indices.','fusion_w':1,'new_forwards':4,'deployable':False},
      ],
      'external_controls':{'VM_root_relative':'../cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs','source_protocol_sha256':'51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec','local_admitted_root':str(OLD_RETURN),'files':controlfiles,'reports':controlreports,'historical_not_fresh48_A_or_C_calls':True,'exact_hashes_required_no_fallback_recompute':True},
      'pre_dispatch_checks':['Four fresh owned ordinary/app parity pairs8calls before any intervention.','Four freshC bridge calls match historicalsame-codebookC raw/result/predicted/used indices and PNG; before96H/V.','For everyH/V case predictedcodes equal exacthistoricalordinaryA; usedcodes equal declaredwhere(grid,oracle,predicted) or complement.','Check all497actualstate tensors finite/hashidentical before/after and both original/parent files exact.','Allgeometry/input/conditioning and raw512→float256→PNG delivery remain exact; noAdaIN/sharpening/resampling changes.'],
      'diagnostic_analysis':{'metrics':['raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1','GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread'],
        'cohorts_separate':True,'groups':['clear32macro/fourgeometry','challenge16macro/fourgeometry/twoconditions','eightGRAYmacro/fourgeometry/percase'],
        'effects':'For each lower-is-better endpoint/case compute H-A, V-A, C-A and C-H-V+A interaction; report all signs/individual regressions, without calling oracle changes trained improvement.',
        'existing_quality_requirements_unchanged':True,'no_diagnostic_arm_adopted':True,
        'next_learning_decision':'If H gives broad useful changes with acceptable boundaries/style whileV contributeslittle, review support-aware code learning with context retention. IfV/coordinatedC effects are important, review global representation and frozenfusion compatibility. If evenoracleH ispoor, do notpromiseCEonly canfixanatomy/style. Allnexttraining needsa separatejustifiedfiniteprotocol,notautomatic selection.'},
      'resources':contract,'resource_arithmetic':{'parity_data_bytes_each':parity_data,'HV_or_bridge_data_bytes_each':new_data,'all104_numeric_data_plus64KiB_each':maximum_array,'arrays_560MiB_reserve_pass':True,'whole_category_reservation_bytes':sum(budgets.values()),'output_1GiBcap_pass':True},
      'failure_retention':{'normal':'No learned checkpoint: all497state unchanged; save fullstatehash/finite/invariance receipts and counters/progress.',
        'unexpected_mutation_or_nonfinite':'One bounded384MiB save of actual497CPU tensors with RNG/progress/counters, no optimizer, before rejection. Record actualnonfinite hashes rather than replacing with original. Serialize toexclusivepartial, safeinternalZIP/CRC checks beforeexclusivecommit. Allpartialsretained;IO/hardkill maypreventrecoverability;noautoretry.',
        'no_integrity_failure_veto_of_resource_bounded_failure_save':True,'output_failure_paths':['outputs/failed_actual_model_state.pth','outputs/failed_actual_model_state.pth.partial'],'do_not_export_original_or_full_parent_weights':True},
      'required_return_review':['Fullboundedtransport/protocol/sourcecontrolmanifest pins and failure truth.','All497stateinvariance,all108typedcalls,code/gridwhereexact,16labelprovenance,all4freshbridges andparity.','All96newRAW/PNG outputs and outsidepixels,plusretained96A/Ccontrol joins.','All48cases/fourconditions in12original256cellpairedsheets,explicitoraclelabels.','Rootallcasereview andindependentQA beforeanysuccessortraining/appdecision.'],
      'evidence':evidence,'runtime_ready':False,'transfer_ready':False,'actual_executions':0,'local_neural_calls':0,'local_training_calls':0,'new_model_created':False,'model_qualified':False,'full_goal_complete':False,
    }
    physical(DEST).mkdir(parents=True)
    path=DEST/'root_design.json';new(path,(json.dumps(design,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    readable='''# Spatial-code intervention V1: a finite completion diagnostic

27 October 2026. **Prospective design; source/packet not ready and no actual run.**
R1 stays selected. The failed predictor trial trained correctly but produced
only0.307561%clear/0.446027%challenge hole-error reductions, with boundary and
gray-style regressions. All20actual sheets were reviewed. More uniform-CE
epochs or a larger LR has not been justified by this evidence.

The old clean-code oracle improved the full-grid result, using information
unavailable in a real covered input. We still do not know which code positions
provided that benefit. The next distinct test isolates this before training.

|Condition|Codes in cells touched by reviewed removal|Codes in the visible complement|Execution|
|---|---|---|---|
|A|Ordinary R1 predictions|Ordinary R1 predictions|Reuse exact audited control|
|H — ORACLE|Retained clean-context codes|Ordinary R1 predictions|48new forwards|
|V — ORACLE|Ordinary R1 predictions|Retained clean-context codes|48new forwards|
|C — ORACLE|Retained clean-context codes|Retained clean-context codes|Reuse audited control;4fresh bridge checks|

All497R1 tensors stay frozen, including encoder/fusion/originalprior. Preserve
w1fusion, hard integer lookup, input geometry, reviewed mask and raw/PNG delivery.
No oracle is deployable or a recovery of hidden identity. No clean RGB is pasted,
no new targets/models/codebooks are generated, and no mask is enlarged.

The fixed grid rule is any reviewed removal pixel within each16x16original256
image block. No dilation/threshold search. This is geometric, not a map of the
neural receptive field. Reuse the same48adaptive paired TRAIN cases/16existing
teacher-label sets. No new public/unpaired or FINAL cases are consumed.

Four ordinary/app parity pairs(8calls), then4fixed fresh-C bridge checks against
the retained raw/code/PNG evidence, precede96newH/V calls:108forwards planned,
120global hard maximum and exacttyped8/4/48/48 ceilings. AllH/Vpredicted codes
must equal retainedA. This does not pretend48freshA/fullC runs occurred.

Save96H/VPNGs plus104NPZs(96H/V+4parity+4bridge). Retain external196A/Ccontrol
files byexacthash:96PNGs/96NPZs/fourresults; no unnecessary reupload or silent
recomputation. Failif they are unavailable. Everynew raw/PNG,outsidepixel,
code-selection and all48/four-condition comparison must be independently audited;
all12planned exact256cell sheets must be viewed, with oracle labels visible.

Keep both cohorts separate and measure hole,boundary,graystyle and per-case
changes. CompareH-A,V-A,C-A and interaction C-H-V+A. Any beneficial oracle
score is diagnostic only. If spatialH benefits broadly with littleV contribution,
review support-aware learning with context retention; otherwise review global
context/fusion compatibility or a different appearance-aware path. Strict gray
appearance remains a separate blocker even when oracle code agreement improves.
No next learning run follows automatically or relaxes the old quality gates.

Prospective execution:15minhard/14minwork/60sfinish reserve,6GiBfree after
install plus payload beforeextract,2GiBfloor,20GiBCUDA. Output1GiB,tar/gzip
1280MiBeach,export15min. Category caps:560MiBarrays,24MiBPNGs,8MiBmetadata,
384MiBunexpected-statefailure =976MiB. Normal completion has no checkpoint;
unexpectedactualstatechange/nonfinite retains onebounded actual497CPUstate plus
RNG/progress, not a substituted original. Preserve failed partials and truth
about IO/hardkill; no automatic rerun/resume. Zerooptimizer/gradient/backward/
teacher calls. Source/interface, resource, controlbridge and transfer checks
must pass before the exact manual VM guide is released.

The immutable machine-readable design is research/completion_codeformer_spatial_code_intervention_v1_design/root_design.json.
The full goal remains ACTIVE: all-family usefulness, automatic-versus-assisted
mask evidence, user acceptance, blindedhumanFINAL, WindowsCPUminimum/freshinstall
and acceptedselective IMPLEMENTATION port remain incomplete. No RESTORATION or
current R1 app change is authorized by a diagnosis alone.
'''
    md=ROOT/'COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_DESIGN.md';new(md,readable.encode('utf-8'))
    receipt={'format':'completion-spatial-code-intervention-v1-root-freeze-v1','complete':True,'root_design':pin(path),'readable_design':pin(md),'author':pin(Path(__file__)),'new_actual_model_training_or_inference':False,'evidence_count':len(evidence),'external_controls':len(controlfiles),'runtime_ready':False,'transfer_ready':False}
    new(DEST/'freeze_receipt.json',(json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps(receipt))

if __name__=='__main__':main()
