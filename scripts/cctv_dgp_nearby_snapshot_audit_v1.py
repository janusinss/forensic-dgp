"""Saved-image replay; initial and rollback both use exact cached control scores."""
import sys
import numpy as np
from import_cctv_dgp_nearby_gradient_learning_v1_return import OUT,PARENT,read,sha
from audit_cctv_dgp_joint_raw_png_forward_v1_return import image,support,metrics,delivered,grouped
from audit_cctv_dgp_raw_conditioning_learning_v1_r1_return import state_with_component


def audit_snapshot(folder,p,old,virtual,names,edges):
    snapshot=read(folder/'snapshot.json');refs={r['id']:r for r in old['references']}
    rows=[];cells={}
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    stored={r['id']:r for r in snapshot['rows']};assert set(stored)==set(p['case_ids'])
    for block in p['reference_blocks']:
        begin=block['begin'];cs=old['cases'][begin:begin+5];stem=f'b{begin//5:03d}'
        raw=np.load(folder/(stem+'.npy'),allow_pickle=False)
        assert raw.dtype==np.float32 and raw.shape==(5,256,256,3) and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
        baseline,expected=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/{stem}.npz',block['case_ids'])
        with np.load(folder/(stem+'_embeddings.npz'),allow_pickle=False) as archive:
            assert set(archive.files)=={'raw','png'};emb={k:archive[k] for k in archive.files}
        proof=next(r for r in snapshot['blocks'] if r['begin']==begin)
        assert sha(folder/(stem+'.npy'))==proof['raw_sha256'] and sha(folder/(stem+'_embeddings.npz'))==proof['embeddings_sha256']
        for stage,index in [('raw',0),('png',1)]:
            assert emb[stage].dtype==np.float32 and emb[stage].shape==(15,512) and np.isfinite(emb[stage]).all()
            np.testing.assert_allclose(emb[stage][5:10],expected[:,index],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(emb[stage][10:15],expected[:,2],rtol=1e-6,atol=1e-6)
        for i,c in enumerate(cs):
            ref=refs[c['source_person_or_reference']]
            camera,target,mask=image(PARENT/c['input']),image(PARENT/ref['target']),image(PARENT/ref['observed'],'L')>0
            patch=support(mask,c['landmarks5_canvas_xy']);png=image(folder/'png'/(c['id']+'.png'))
            assert np.array_equal(delivered(raw[i],camera,mask),png)
            assert np.array_equal(raw[i][~mask],(camera.astype(np.float32)/np.float32(255))[~mask]) and np.array_equal(png[~mask],camera[~mask])
            if folder.name in ['initial','rollback']:assert np.array_equal(raw[i],baseline[i])
            shift=(raw[i]-baseline[i])[mask].astype(np.float64).mean(0)
            mean=np.clip(baseline[i].astype(np.float64)+shift,0,1).astype(np.float32);mean[~mask]=(camera.astype(np.float32)/np.float32(255))[~mask]
            mean_png=delivered(mean,camera,mask)
            row=dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'])
            assert all(row[k]==stored[c['id']][k] for k in row)
            for stage,actual,mean_actual in [('raw',raw[i],mean),('png',png.astype(np.float32)/np.float32(255),mean_png.astype(np.float32)/np.float32(255))]:
                v=emb[stage]
                j=0 if stage=='raw' else 1
                score_vector=expected[i,j] if folder.name in ['initial','rollback'] else v[i]
                target_vector=expected[i,2] if folder.name in ['initial','rollback'] else v[10+i]
                row[stage]=metrics(actual,target,mask,patch,score_vector,target_vector,mean_actual,prediction8=png if stage=='png' else None)
                for key,value in row[stage].items():
                    other=stored[c['id']][stage][key]
                    if value is None or isinstance(value,bool):assert value==other
                    else:assert np.isclose(value,other,rtol=0,atol=1e-13),(folder.name,c['id'],stage,key)
            rows.append(row);cells[c['id']]=png
    group={stage:grouped(rows,stage) for stage in ['raw','png']}
    for stage,groups in group.items():
        for key,values in groups.items():
            for metric,value in values.items():assert np.isclose(value,snapshot['groups'][stage][key][metric],rtol=0,atol=1e-13)
    if folder.name in ['initial','rollback']:flat=np.load(folder.parent/'initial_conditioning.npy',allow_pickle=False)
    else:flat=np.load(folder/'conditioning.npy',allow_pickle=False)
    assert state_with_component(virtual,names,edges,flat)==snapshot['model_state']
    return snapshot,group,cells
