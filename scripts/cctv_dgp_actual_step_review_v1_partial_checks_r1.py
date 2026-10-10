"""Separate partial-scope checks; original prospective checks remain unchanged."""
from audit_cctv_dgp_actual_step_review_v1_return import *


roundoff_receipts=[]

def check_comparison_roundoff(actual,reported,zero_condition,grouped,receipt,tolerance):
    # Retained saved-row comparisons must still reproduce exactly under the
    # original allowance. There is no exception for categories or gate values.
    canonical_groups={role:{stage:review_groups(receipt['rows'][role],stage) for stage in ['raw','PNG']} for role in ROLES}
    canonical=None if zero_condition is None else {role:{stage:finite_comparison(zero_condition['groups'][role][stage],canonical_groups[role][stage]) for stage in ['raw','PNG']} for role in ROLES}
    derived_equal(canonical,reported,tolerance)
    if actual is None:
        derived_equal(actual,reported,tolerance);return
    for role in ROLES:
        for stage in ['raw','PNG']:
            a,s=actual[role][stage],reported[role][stage]
            derived_equal({k:v for k,v in a.items() if k!='mean_only_fraction'},
                          {k:v for k,v in s.items() if k!='mean_only_fraction'},tolerance)
            error=abs(a['mean_only_fraction']-s['mean_only_fraction'])
            if error<=tolerance:continue
            baseline=zero_condition['groups'][role][stage]['degraded']['MSE']
            after=grouped[role][stage]['degraded']
            saved_after=receipt['groups'][role][stage]['degraded']
            assert after['MSE']>baseline+1e-12 and saved_after['MSE']>baseline+1e-12
            assert max(baseline-after['MSE'],1e-12)==max(baseline-saved_after['MSE'],1e-12)==1e-12
            assert not a['finite_preservation_pass'] and not s['finite_preservation_pass']
            assert any(r['group']=='degraded' and r['metric']=='MSE' for r in a['preservation_failures'])
            assert any(r['group']=='degraded' and r['metric']=='MSE' for r in s['preservation_failures'])
            scale=max(abs(baseline),abs(after['constant_mean_shift_only_MSE']),abs(saved_after['constant_mean_shift_only_MSE']))
            bound=8*np.finfo(np.float64).eps*scale/1e-12+8*max(abs(np.spacing(a['mean_only_fraction'])),abs(np.spacing(s['mean_only_fraction'])))
            assert np.isfinite(bound) and error<=bound
            roundoff_receipts.append({'update':receipt['update'],'proposal':receipt['proposal'],'role':role,'stage':stage,
                'computed_ratio':a['mean_only_fraction'],'reported_ratio':s['mean_only_fraction'],'absolute_ratio_error':error,
                'roundoff_bound':float(bound),'denominator':1e-12,'group_mean_control_MSE_error':abs(after['constant_mean_shift_only_MSE']-saved_after['constant_mean_shift_only_MSE']),
                'original_stored_row_arithmetic_pass':True,'both_degraded_MSE_gates_failed':True,'all_categorical_decisions_exact':True})

def scientific_partial_r1(p,failure,start,completed):
    sys.path.insert(0,str(BUNDLE))
    from cctv_dgp_spatial_fit_v40_contract import feature_support,exported_pixel_metrics,detail_metric
    cases={c['id']:c for c in p['cases']};checked=0;term_errors=np.zeros(7);condition_list=[]
    for probe in p['probes']:
        zero_condition=None
        with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as data:values=data['values'].copy()
        for index,proposal in enumerate(PROPOSALS):
            if (probe["update"],proposal) not in completed:continue
            folder=RETURN/output_prefix(probe['update'],proposal);receipt=read(folder/'metrics.json')
            assert receipt['complete'] and receipt['decoder_state']==parameter_state_hash(values[index],p['parameter_layout'])
            assert receipt['parameter_vector_sha256']==hashlib.sha256(values[index].tobytes()).hexdigest()
            assert receipt['update']==probe['update'] and receipt['proposal']==proposal
            assert receipt['optimizer_updates']==receipt['gradient_queries']==0 and not receipt['new_trained_checkpoint'] and not receipt['app_promotion']
            assert receipt['first_order_proposal_arithmetic']==probe['arithmetic']
            rows_by_role={}
            for role in ROLES:
                ids=role_ids(p,probe,role);assert [r['id'] for r in receipt['rows'][role]]==ids;rows=[]
                for cid,reported in zip(ids,receipt['rows'][role]):
                    assert time.monotonic()-start<CAP
                    c=cases[cid];camera=pixels(BUNDLE/c['input']);target=pixels(BUNDLE/c['target'])
                    with Image.open(BUNDLE/c['observed']) as image:mask=np.asarray(image.convert('L'))>0
                    a=arrays(folder/role/(cid+'.npz'),proposal=='zero')
                    zero=arrays(RETURN/output_prefix(probe['update'],'zero')/role/(cid+'.npz'),True)
                    assert np.array_equal(a['rgb'][~mask],camera[~mask].astype(np.float32)/np.float32(255))
                    png=pixels(folder/role/(cid+'.png'));expected=np.floor(a['rgb']*np.float32(255)).astype(np.uint8);expected[~mask]=camera[~mask]
                    assert np.array_equal(png,expected) and np.array_equal(png[~mask],camera[~mask])
                    assert reported['raw_float32_sha256']==hashlib.sha256(a['rgb'].tobytes()).hexdigest()
                    mean_raw,mean_png,shift=mean_only(a['rgb'],zero['rgb'],camera,mask)
                    assert np.array_equal(mean_png,pixels(folder/role/(cid+'_mean_only.png')))
                    assert np.allclose(shift,reported['postclip_mean_RGB_shift'],rtol=0,atol=1e-12)
                    support=feature_support(mask,c['landmarks5_canvas_xy'])
                    raw=pixel_metrics(a['rgb'],target,mask);raw.update({'ArcFace_observed_fixed':float(a['raw_vector']@a['truth']),
                        'landmark_high_frequency_MSE':detail_float(a['rgb'],target,support),'constant_mean_shift_only_MSE':pixel_metrics(mean_raw,target,mask)['MSE']})
                    displayed=exported_pixel_metrics(png,target,mask);displayed.update({'ArcFace_observed_fixed':float(a['PNG_vector']@a['truth']),
                        'landmark_high_frequency_MSE':detail_metric(png,target,support),'constant_mean_shift_only_MSE':exported_pixel_metrics(mean_png,target,mask)['MSE']})
                    for stage,actual in [('raw',raw),('PNG',displayed)]:
                        for key,value in actual.items():
                            saved=reported[stage][key]
                            if value is None or isinstance(value,bool):assert value==saved
                            else:assert abs(value-saved)<=1e-10,(cid,stage,key)
                    assert np.array_equal(a['terms'].astype(np.float64),reported['objective_terms'])
                    independent,anchor=raw_terms(a,zero['original_rgb'],target,mask,support,p)
                    if c['profile']=='clear':independent[:3]=0;independent[3]=anchor
                    error=np.abs(independent-a['terms']);term_errors=np.maximum(term_errors,error)
                    assert np.all(error<=p['prospective_audit']['raw_objective_term_absolute_error']),(cid,error.tolist())
                    rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'raw':raw,'PNG':displayed,'objective_terms':reported['objective_terms']});checked+=1
                rows_by_role[role]=rows
            grouped={role:{stage:review_groups(rows_by_role[role],stage) for stage in ['raw','PNG']} for role in ROLES}
            derived_equal(grouped,receipt['groups'],p['prospective_audit']['derived_group_and_comparison_absolute_error'])
            for role in ROLES:assert np.array_equal(np.mean([r['objective_terms'] for r in rows_by_role[role]],axis=0),receipt['objective_term_means'][role])
            comparison=None if zero_condition is None else {role:{stage:finite_comparison(zero_condition['groups'][role][stage],grouped[role][stage]) for stage in ['raw','PNG']} for role in ROLES}
            check_comparison_roundoff(comparison,receipt['comparison_to_same_before_state'],zero_condition,grouped,receipt,p['prospective_audit']['derived_group_and_comparison_absolute_error'])
            condition_list.append({'update':probe['update'],'proposal':proposal,'metrics_sha256':sha(folder/'metrics.json'),'comparison_to_same_before_state':receipt['comparison_to_same_before_state']})
            if proposal=='zero':zero_condition=receipt
            print({'partial_condition_checked':len(condition_list),'of':29,'slots':checked},flush=True)
    assert checked==3045
    derived_equal(condition_list,failure['completed_conditions'],p['prospective_audit']['derived_group_and_comparison_absolute_error'])
    return term_errors,condition_list,checked

def CPU_partial_replay(p,start,completed):
    import torch
    sys.path.insert(0,str(BUNDLE))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_decoder_v41 import SpatialDGPCandidateV41
    from cctv_dgp_pilot import FixedObservedIdentity,state_hash,grid112
    torch.set_num_threads(4)
    original,_=load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    seed=torch.load(BUNDLE/'untrained_initial_decoder.pth',map_location='cpu',weights_only=True)
    candidate=SpatialDGPCandidateV41(original,seed).eval().requires_grad_(False)
    identity=FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx','cpu')
    states=lambda:{'original':state_hash(original.net),'decoder':state_hash(candidate.decoder),'reference_decoder':state_hash(candidate.reference_decoder),'recognizer':state_hash(identity)}
    initial=states();assert initial==p['initial_states'];cases={c['id']:c for c in p['cases']};refs={r['id']:r for r in p['references']}
    canonical=lambda a:torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
    result={'cases':0,'raw_maximum_error':0.,'original_raw_maximum_error':0.,'PNG_maximum_byte_difference':0,'embedding_maximum_error':0.}
    with torch.inference_mode():
        for probe in p['probes']:
            with np.load(BUNDLE/probe['proposal_arrays'],allow_pickle=False) as data:values=data['values'].copy()
            for index,proposal in enumerate(PROPOSALS):
                if (probe["update"],proposal) not in completed:continue
                assert time.monotonic()-start<CAP
                state={r['name']:torch.from_numpy(values[index,r['start']:r['end']].copy()).reshape(r['shape']) for r in p['parameter_layout']}
                candidate.decoder.load_state_dict(state,strict=True)
                assert state_hash(candidate.decoder)==parameter_state_hash(values[index],p['parameter_layout'])
                ids=probe['case_ids'];images=[pixels(BUNDLE/cases[cid]['input']) for cid in ids]
                masks=[]
                for cid in ids:
                    with Image.open(BUNDLE/cases[cid]['observed']) as image:masks.append(np.asarray(image.convert('L'))>0)
                x=torch.cat([canonical(a) for a in images]);mask=torch.from_numpy(np.stack(masks))[:,None]
                actual=candidate.forward_components(x,mask);raw=actual['result'].permute(0,2,3,1).numpy().copy()
                initial_raw=actual['original_raw'].permute(0,2,3,1).numpy().copy()
                folder=RETURN/output_prefix(probe['update'],proposal)/'current_batch';saved=[arrays(folder/(cid+'.npz'),proposal=='zero') for cid in ids]
                zero=[arrays(RETURN/output_prefix(probe['update'],'zero')/'current_batch'/(cid+'.npz'),True) for cid in ids]
                pngs=[pixels(folder/(cid+'.png')) for cid in ids]
                grids=torch.from_numpy(np.stack([grid112(refs[cases[cid]['reference_id']]['matrix112']) for cid in ids]))
                raw_x=torch.cat([torch.from_numpy(a['rgb']).permute(2,0,1)[None] for a in saved])
                original_x=torch.cat([torch.from_numpy(a['original_rgb']).permute(2,0,1)[None] for a in zero])
                target_x=torch.cat([canonical(pixels(BUNDLE/cases[cid]['target'])) for cid in ids])
                vectors={'raw_vector':identity.embedding(raw_x,mask.float(),grids).numpy(),
                         'PNG_vector':identity.embedding(torch.cat([canonical(a) for a in pngs]),mask.float(),grids).numpy(),
                         'truth':identity.embedding(target_x,mask.float(),grids).numpy(),
                         'raw_reference_vector':identity.embedding(original_x,mask.float(),grids).numpy()}
                for slot,(cid,camera,support) in enumerate(zip(ids,images,masks)):
                    error=float(np.abs(raw[slot]-saved[slot]['rgb']).max());assert error<=p['prospective_audit']['raw_replay_maximum_error']
                    result['raw_maximum_error']=max(result['raw_maximum_error'],error)
                    error=float(np.abs(initial_raw[slot]-zero[slot]['original_rgb']).max());assert error<=p['prospective_audit']['raw_replay_maximum_error']
                    result['original_raw_maximum_error']=max(result['original_raw_maximum_error'],error)
                    q=np.floor(raw[slot]*np.float32(255)).astype(np.uint8);q[~support]=camera[~support]
                    error=int(np.abs(q.astype(np.int16)-pngs[slot].astype(np.int16)).max());assert error<=1
                    result['PNG_maximum_byte_difference']=max(result['PNG_maximum_byte_difference'],error)
                    for key in vectors:
                        error=float(np.abs(vectors[key][slot]-saved[slot][key]).max());assert error<=p['prospective_audit']['embedding_replay_maximum_error']
                        result['embedding_maximum_error']=max(result['embedding_maximum_error'],error)
                    result['cases']+=1
                print({'CPU_partial_replay_cases':result['cases'],'of':145,'update':probe['update'],'proposal':proposal},flush=True)
    candidate.decoder.load_state_dict(seed,strict=True);assert states()==initial and result['cases']==145
    assert all(not v.requires_grad and v.grad is None for model in [candidate,identity] for v in model.parameters())
    return {**result,'all_persistent_model_states_unchanged':True,'optimizer_updates':0,'gradient_queries':0}
