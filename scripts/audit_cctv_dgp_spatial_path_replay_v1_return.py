"""Independent saved-array/state-byte audit. Never builds or mutates a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from cctv_dgp_spatial_path_replay_v1_contract import (
    read,write,sha,array_sha,verify_transfer,label,selected_vector,expected_state_hash,UPDATES,
)


def recompute_metrics(raw,png,camera,target,mask,support,baseline,retained_embeddings,er,ep,index,initial):
    """Fixed saved-array math, also calibrated against actual audited controls."""
    import numpy as np
    from frozen_raw_metrics import pixel_metrics,detail_float,deliver,mean_only
    from frozen_capacity_contract import exported_pixel_metrics,detail_metric
    assert np.array_equal(png,deliver(raw,camera,mask))
    assert np.array_equal(raw[~mask],camera[~mask].astype(np.float32)/np.float32(255))
    assert np.array_equal(png[~mask],camera[~mask])
    mean,mean_png,_=mean_only(raw,baseline,camera,mask)
    rm,pm=pixel_metrics(raw,target,mask),exported_pixel_metrics(png,target,mask)
    for metric,vectors,slot in [(rm,er,0),(pm,ep,1)]:
        v=retained_embeddings[index,slot] if initial else vectors[index]
        t=retained_embeddings[index,2] if initial else vectors[10+index]
        metric['ArcFace_observed_fixed']=float(v@t)
    rm.update(landmark_high_frequency_MSE=detail_float(raw,target,support),constant_mean_shift_only_MSE=pixel_metrics(mean,target,mask)['MSE'])
    pm.update(landmark_high_frequency_MSE=detail_metric(png,target,support),constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png,target,mask)['MSE'])
    return rm,pm


def audit_saved(root,source,parent,p,old,data,receipt):
    import numpy as np
    from PIL import Image,ImageDraw
    import torch  # read-only source tensor deserialization only
    sys.path.insert(1,str(source));sys.path.insert(1,str(parent))
    from audit_cctv_dgp_spatial_readout_v1_return import grouped,independent_gate,state_hash
    from frozen_raw_metrics import pixel_metrics,detail_float,deliver,mean_only
    from frozen_capacity_contract import feature_support,exported_pixel_metrics,detail_metric
    from cctv_dgp_head4_lossless_v1 import unpack
    start=time.monotonic();out=root/'outputs/spatial_path_replay_v1'
    def clock():
        assert time.monotonic()-start<=p['budgets']['saved_audit_seconds'], 'Finite saved-audit stop'
    assert not receipt.exists(), 'Retain prior audit'
    if not (out/'run_results.json').is_file():
        assert (out/'failure.json').is_file()
        failure=read(out/'failure.json')
        assert failure['counts']['optimizer_updates']==failure['counts']['gradient_queries']==failure['counts']['backwards']==0
        write(receipt,dict(complete=True,partial_failure_evidence_only=True,failure=failure,
            model_qualified=False,visual_review_pending=True,neural_calls=0,optimizer_updates=0))
        return
    result=read(out/'run_results.json')
    assert result['complete'] and not result['model_qualified'] and result['full_corpus_epochs']==0
    assert result['counts']==dict(optimizer_updates=0,gradient_queries=0,backwards=0,
        original_batch_calls=20,prior_feature_batch_calls=20,decoder_batch_calls=280,recognizer_batch_calls=600)
    assert result['snapshot_compositions']==1400 and result['fresh_original_controls']==100
    env=read(out/'environment.json')
    assert env['host']=='forensic-dgp-thesis' and env['GPU']=='NVIDIA L4' and env['torch']=='2.9.1+cu129'
    assert env['original_state']==old['original_state'] and env['recognizer_state']==old['recognizer_state']
    assert env['provider_state']==old['feature_provider_initial_state'] and not env['historical_training_called'] and not env['gradient_mode']
    cache=read(out/'cache_receipt.json')
    assert cache['complete'] and cache['all_source_feature_hashes_exact']
    original_cache=read(source/'outputs/spatial_readout_v1/cache_receipt.json')['rows']
    assert len(cache['rows'])==20
    for row,original in zip(cache['rows'],original_cache):
        assert all(row[k]==original[k] for k in ['reference','maps','features','source'])
    refs,cases={r['id']:r for r in data['references']},{c['id']:c for c in data['cases']}
    def pixels(path,mode='RGB'):
        with Image.open(path) as image:
            assert image.size==(256,256)
            return np.asarray(image.convert(mode)).copy()
    reductions,cells={},{}
    labels=['baseline']+[label(s) for s in p['states']]
    states={label(s):s for s in p['states']}
    checked=0;control_count=0
    for key in labels:
        assert time.monotonic()-start<=p['budgets']['saved_audit_seconds'], 'Finite saved-audit stop'
        saved=read(out/key/'metrics.json');saved_rows={r['case']:r for r in saved['rows']}
        assert set(saved_rows)==set(p['case_ids'])
        rows=[]
        state=states.get(key)
        initial=key=='baseline' or state['update']==0
        source_label='baseline' if initial else state['route']+'50' if state['update']==50 else None
        for index,block in enumerate(p['reference_blocks']):
            clock()
            ref=refs[block['reference']]
            retained,retained_embeddings=unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
            with np.load(out/key/'packs'/f'r{index:02d}.npz',allow_pickle=False) as pack:
                assert set(pack.files)=={'ids','raw','embeddings_raw','embeddings_png','target_embeddings'} and pack['ids'].tolist()==block['case_ids']
                raw,er,ep,tv=[pack[k].copy() for k in ['raw','embeddings_raw','embeddings_png','target_embeddings']]
            assert raw.shape==(5,256,256,3) and raw.dtype==np.float32 and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
            assert np.array_equal(tv,retained_embeddings[:,2])
            for vectors,slot in [(er,0),(ep,1)]:
                assert vectors.shape==(15,512) and vectors.dtype==np.float32 and np.isfinite(vectors).all()
                np.testing.assert_allclose(vectors[5:10],retained_embeddings[:,slot],rtol=1e-6,atol=1e-6)
                np.testing.assert_allclose(vectors[10:15],tv,rtol=1e-6,atol=1e-6)
            if source_label:
                with np.load(source/'outputs/spatial_readout_v1'/source_label/'packs'/f'r{index:02d}.npz',allow_pickle=False) as controls:
                    assert np.array_equal(raw,controls['raw'])
            for i,cid in enumerate(block['case_ids']):
                c=cases[cid];camera=pixels(parent/c['input']);target=pixels(parent/ref['target'])
                mask=pixels(parent/ref['observed'],'L')>0;support=feature_support(mask,c['landmarks5_canvas_xy'])
                png=pixels(out/key/'png'/(cid+'.png'))
                rm,pm=recompute_metrics(raw[i],png,camera,target,mask,support,retained[i],retained_embeddings,er,ep,i,initial)
                if source_label:
                    assert np.array_equal(png,pixels(source/'outputs/spatial_readout_v1'/source_label/'png'/(cid+'.png')))
                    control_count+=1
                row=dict(case=cid,source=c['source'],profile=c['profile'],raw=rm,png=pm)
                assert all(row[k]==saved_rows[cid][k] for k in ['case','source','profile'])
                for mode in ['raw','png']:
                    for metric,value in row[mode].items():
                        other=saved_rows[cid][mode][metric]
                        if value is None or isinstance(value,bool):assert value==other
                        else:assert abs(value-other)<=1e-13,(key,cid,mode,metric)
                rows.append(row);cells[(key,cid)]=png;checked+=1
        reductions[key]={mode:grouped(rows,mode) for mode in ['raw','png']}
        for mode in ['raw','png']:
            for group,values in reductions[key][mode].items():
                for metric,value in values.items():assert abs(value-saved['groups'][mode][group][metric])<=1e-13
    assert checked==1500 and control_count==500  # baseline + two initial + two endpoint sets
    assert result['outcomes']==[read(out/(label(s)+'_outcome.json')) for s in p['states']]
    verified_states=[];outcomes=[]
    for route in ['A','B']:
        original=torch.load(source/('outputs/spatial_readout_v1/state_'+route+'0/full_state.pth'),map_location='cpu',weights_only=True)['model']
        names=old['trainable_parameter_names'][route]
        fixed=state_hash({k:v for k,v in original.items() if k not in names})
        buffers=read(source/('outputs/spatial_readout_v1/state_'+route+'0/receipt.json'))['buffer_state']
        layout=read(source/('outputs/spatial_readout_v1/'+route+'_parameter_layout.json'))
        initial=np.load(source/('outputs/spatial_readout_v1/'+route+'_initial_vector.npy'),allow_pickle=False)
        for state in [s for s in p['states'] if s['route']==route]:
            clock()
            key=label(state);expected=selected_vector(source,state,initial)
            vector=np.load(out/(key+'_vector.npy'),allow_pickle=False)
            assert np.array_equal(vector,expected)
            receipt_state=read(out/(key+'_state.json'))
            assert receipt_state['model_state']==expected_state_hash(original,names,layout,vector)
            assert receipt_state['fixed_state']==fixed and receipt_state['buffer_state']==buffers
            assert receipt_state['vector_sha256']==array_sha(vector) and receipt_state['installed_vector_exact']
            assert not receipt_state['gradients_enabled'] and receipt_state['optimizer_updates']==0
            stored=read(out/(key+'_outcome.json'))
            gates={mode:independent_gate(reductions['baseline'][mode],reductions[key][mode]) for mode in ['raw','png']}
            for mode,g in gates.items():
                v=stored['gates'][mode]
                assert v['pass']==g['pass']
                assert [(f['group'],f['metric']) for f in v['preservation_failures']]==[(f['group'],f['metric']) for f in g['preservation_failures']]
                assert abs(v['global_structure_gain']-g['global_structure_gain'])<1e-12
                assert all(abs(v['source_structure_gains'][s]-g['source_structure_gains'][s])<1e-12 for s in g['source_structure_gains'])
                # Same recorded/replayed reductions permit only arithmetic propagation.
                bg=read(out/'baseline/metrics.json')['groups'][mode]['degraded']
                ag=read(out/key/'metrics.json')['groups'][mode]['degraded']
                b,a=reductions['baseline'][mode]['degraded'],reductions[key][mode]['degraded']
                nr=max(0.,bg['MSE']-ag['constant_mean_shift_only_MSE']);dr=max(bg['MSE']-ag['MSE'],1e-12)
                nd=max(0.,b['MSE']-a['constant_mean_shift_only_MSE']);dd=max(b['MSE']-a['MSE'],1e-12)
                propagation=abs(nr-nd)/dr+abs(nd)*abs(dr-dd)/(dr*dd)+1e-12
                assert abs(v['brightness_gain_fraction']-g['brightness_gain_fraction'])<=propagation
            assert not stored['incumbent_selection'] and not stored['model_qualified'] and stored['new_optimizer_updates']==0
            outcomes.append(dict(route=route,source_update=state['update'],snapshot=key,gates=gates))
            verified_states.append(dict(snapshot=key,source_vector_exact=True,expected_complete_model_state_hash=receipt_state['model_state'],fixed_bytes_unchanged=True))
        del original
    gallery=receipt.parent/'visual_sheets'
    assert not gallery.exists(), 'Retain previous visual sheets'
    gallery.mkdir();sheets=[];cell_bindings=[]
    for route in ['A','B']:
        for block in p['reference_blocks']:
            clock()
            sheet=Image.new('RGB',(2304,1320),'#fafafa');draw=ImageDraw.Draw(sheet)
            titles=p['visualization']['columns']
            for x,title in enumerate(titles):draw.text((256*x+8,8),route+' '+title,fill='#111111')
            for y,cid in enumerate(block['case_ids']):
                ref=refs[block['reference']]
                images=[pixels(parent/cases[cid]['input']),pixels(parent/ref['target']),cells[('baseline',cid)]]
                images += [cells[(route+'_u'+str(u).zfill(3),cid)] for u in UPDATES if u]
                assert len(images)==9
                for x,value in enumerate(images):
                    sheet.paste(Image.fromarray(value),(256*x,40+256*y))
                    cell_bindings.append(dict(sheet=route+'_'+block['reference']+'.png',case=cid,column=x,pixels_sha256=array_sha(value)))
            path=gallery/(route+'_'+block['reference']+'.png');sheet.save(path)
            cells_written=np.asarray(Image.open(path).convert('RGB'))
            for y,cid in enumerate(block['case_ids']):
                for x in range(9):
                    row=cell_bindings[-45+y*9+x]
                    assert array_sha(cells_written[40+256*y:40+256*(y+1),256*x:256*(x+1)])==row['pixels_sha256']
            sheets.append(dict(path=str(path),sha256=sha(path),cells=45,route=route,reference=block['reference']))
    assert len(sheets)==40 and len(cell_bindings)==1800
    curves=[]
    for route in ['A','B']:
        rows=[o for o in outcomes if o['route']==route]
        bad=[r['source_update'] for r in rows if r['source_update'] and any(g['preservation_failures'] for g in r['gates'].values())]
        passed=[r['source_update'] for r in rows if all(g['pass'] for g in r['gates'].values())]
        first={}
        for row in rows:
            if row['source_update']:
                for mode,g in row['gates'].items():
                    for f in g['preservation_failures']:first.setdefault(mode+'/'+f['group']+'/'+f['metric'],row['source_update'])
        curves.append(dict(route=route,first_sampled_preservation_failure_update=min(bad) if bad else None,
            first_sampled_failed_check_updates=first,numerical_exposed_TRAIN_pass_updates=passed,
            unsampled_updates_not_measured=True,no_retrospective_incumbent_selection=True))
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),compositions_checked=1500,
        original_initial_endpoint_control_cases_exact=500,all17_RAW_all17_PNG_source_groups_recomputed=True,
        states_verified=verified_states,outcomes=outcomes,curves=curves,sheets=sheets,cell_bindings=cell_bindings,
        every_gallery_cell_exact=True,visual_review_pending=True,independent_final_reviewer=False,
        model_qualified=False,full_corpus_epochs=0,neural_calls=0,gradient_queries=0,optimizer_updates=0,
        read_only_tensor_hash_streaming=True,no_model_construction_or_weight_assignment=True,seconds=time.monotonic()-start)
    clock()
    write(receipt,record)
    print(json.dumps(dict(complete=True,compositions=1500,saved_states=14,visual_sheets=40,optimizer_updates=0,model_qualified=False)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--parent-root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    a=parser.parse_args();p,old,data=verify_transfer(a.root,a.source_root,a.parent_root,a.protocol_sha)
    audit_saved(a.root.resolve(),a.source_root.resolve(),a.parent_root.resolve(),p,old,data,a.receipt.resolve())


if __name__=='__main__':main()
