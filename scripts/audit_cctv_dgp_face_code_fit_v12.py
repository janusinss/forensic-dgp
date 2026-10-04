"""Independent pixels, logits, latent arithmetic and trace audit; no backward."""
import argparse
import math
from pathlib import Path
import sys
import time


def audit(root, expected_sha, out, receipt):
    sys.path.insert(0, str(root))
    from cctv_dgp_face_code_fit_v12 import verify, read, write, sha, require, safe
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.special import logsumexp
    from skimage.metrics import structural_similarity
    import torch
    import ast
    start = time.monotonic(); p = verify(root, expected_sha); design = p['design']
    require(not receipt.exists(), 'Preserve completed audit')
    r = read(out / 'results.json'); terminal = read(out / 'neural_execution_receipt.json'); preflight = read(out / 'cuda_preflight.json')
    require(r['complete'] and r['protocol_sha256'] == expected_sha and terminal['complete'] and preflight['complete'], 'Completion binding differs')
    require(r['optimizer_updates'] == terminal['optimizer_updates'] == 300 and r['training_exposures'] == 600 and
            r['backward_calls'] == terminal['backward_calls'] == 301 and preflight['optimizer_updates'] == 0, 'Training/preflight counts differ')
    require(r['seconds'] <= 600 and r['peak_vram_bytes'] <= design['peak_vram_cap_bytes'], 'Budget exceeded')
    expected_counts = {'dgp': 752, 'conditioner': 752, 'prior_encoder': 753, 'prior_transformer_head': 753,
                       'prior_generator': 302, 'teacher_encoder': 10, 'teacher_quantizer': 10, 'recognizer': 310}
    require(r['counts'] == terminal['counts'] == terminal['expected_counts'] == expected_counts, 'Neural count mismatch')
    require(terminal['frozen_before'] == terminal['frozen_after'] and terminal['initial_conditioner_state_hash'] !=
            terminal['final_conditioner_state_hash'], 'Frozen/trainable state contract differs')
    for key in ['native_used','validation_used','native_reserved_used','production_promoted','checkpoint_selected','best_checkpoint_created']:
        require(r[key] is False, 'Unsupported use/selection/promotion')
    require(not list(out.rglob('best.pth')), 'Training-only diagnostic created a best checkpoint')
    for name,pin in r['artifacts_sha256'].items(): require(sha(safe(out,name))==pin,'Changed artifact: '+name)
    steps = read(out/'update_trace.json')['steps']; expected=read(root/'schedule_v12.json')['steps']
    import json
    require(steps == [json.loads(line) for line in (out/'update_trace.jsonl').read_text().splitlines()], 'Incremental trace differs')
    require(len(steps)==len(expected)==300, 'Missing trace step')
    for i,(saved,step) in enumerate(zip(steps,expected),1):
        require(saved['update']==i and saved['epoch']==step['epoch'] and saved['case_ids']==step['case_ids'],'Step/exposure differs')
        require(all(math.isfinite(saved[key]) for key in ['loss','code_ce','feature_mse','code_accuracy','gradient_norm_before_clip','seconds']), 'Nonfinite trace')
        require(0<=saved['code_accuracy']<=1 and saved['code_ce']>=0 and saved['feature_mse']>=0,'Invalid trace objective')
        require(abs(saved['loss']-(.5*saved['code_ce']+saved['feature_mse'])) <= 2e-5,'Trace loss arithmetic differs')
    source=root/'cctv_dgp_pilot.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exported_pixel_metrics']
    require(len(nodes)==1,'Independent numeric function missing')
    ns={'np':np,'cv2':cv2,'structural_similarity':structural_similarity}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    prior_state=torch.load(root/p['weights']['prior'],map_location='cpu',weights_only=True)['params_ema']
    codebook=prior_state['quantize.embedding.weight'].numpy();del prior_state
    def rgb(path):
        with Image.open(path) as im:
            require(im.mode=='RGB' and im.size==(256,256),'Bad delivered RGB geometry')
            return np.asarray(im).copy()
    def vector(path):
        value=np.load(path,allow_pickle=False)
        require(value.dtype==np.float32 and value.shape==(512,) and np.isfinite(value).all() and np.isclose(np.linalg.norm(value),1,atol=1e-5),'Invalid embedding')
        return value
    def close(a,b,tolerance=1e-6):
        if a is None or isinstance(a,bool): require(a==b,'Recorded scalar differs')
        else: require(np.isclose(a,b,rtol=tolerance,atol=tolerance),'Recorded numeric value differs')
    targets={};codes={};features={};supports={};embeddings={};labels=0
    for ref in p['references']:
        rid=ref['id'];targets[rid]=rgb(root/ref['target']);supports[rid]=np.asarray(Image.open(root/ref['observed']))>0
        codes[rid]=np.load(out/'teacher'/f'{rid}_codes.npy',allow_pickle=False)
        require(codes[rid].dtype==np.int64 and codes[rid].shape==(1,256) and 0<=codes[rid].min()<=codes[rid].max()<1024,'Bad teacher labels')
        features[rid]=np.load(out/'teacher'/f'{rid}_features.npy',allow_pickle=False)
        expected_q=codebook[codes[rid]].reshape(1,16,16,256).transpose(0,3,1,2)
        np.testing.assert_array_equal(features[rid],expected_q);labels+=256
        embeddings[rid]=vector(out/'teacher'/f'{rid}_embedding.npy')
    require([s['update'] for s in r['snapshots']]==[0,100,300],'Snapshots differ')
    pngs=cosines=code_probes=grid_cells=0;stage_rows={};summaries={}
    case_map={c['id']:c for c in p['cases']}
    from collections import defaultdict
    for snap in r['snapshots']:
        update=snap['update'];data=read(out/snap['metrics']);require(data['update']==update,'Snapshot update differs')
        rows=data['rows'];require(len(rows)==100,'Snapshot rows missing')
        expected_order=[(c['id'],w) for c in p['cases'] for w in [0.0,1.0]]
        require([(row['id'],row['fidelity']) for row in rows]==expected_order,'Snapshot case/fidelity order differs')
        stage_rows[update]={(row['id'],row['fidelity']):row for row in rows};seen=set();grouped=defaultdict(list)
        for row in rows:
            c=case_map[row['id']];rid=c['reference_id'];support=supports[rid]
            require(all(row[k]==c[k] for k in ['id','reference_id','source','profile']),'Role/source/profile changed')
            raw=np.load(out/row['raw'],allow_pickle=False)
            require(raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1,'Invalid raw render')
            delivered=rgb(out/row['prediction']);expected_png=(raw*255).astype(np.uint8);expected_png[~support]=rgb(root/c['input'])[~support]
            np.testing.assert_array_equal(delivered,expected_png)
            for key,value in ns['exported_pixel_metrics'](delivered,targets[rid],support).items():close(row[key],value,1e-7)
            close(row['ArcFace_observed_fixed'],float(np.clip(vector(out/row['embedding'])@embeddings[rid],-1,1)),1e-7)
            pngs+=1;cosines+=1
            if row['id'] not in seen:
                seen.add(row['id']);logits=np.load(out/row['code_logits'],allow_pickle=False);latent=np.load(out/row['code_features'],allow_pickle=False)
                require(logits.dtype==np.float32 and logits.shape==(1,256,1024) and np.isfinite(logits).all(),'Invalid code logits')
                require(latent.dtype==np.float32 and latent.shape==(1,256,16,16) and np.isfinite(latent).all(),'Invalid conditioned features')
                # Torch nearest downsampling uses floor(output_index * scale),
                # unlike Pillow's center-index nearest sampling.
                token_support=support[np.arange(16)*16][:,np.arange(16)*16].reshape(1,256)
                values=logits.astype(np.float64);ce=logsumexp(values,axis=2)-np.take_along_axis(values,codes[rid][...,None],axis=2)[...,0]
                fmse=np.square(latent.astype(np.float64)-features[rid].astype(np.float64)).mean(1).reshape(1,256)
                correct=logits.argmax(2)==codes[rid]
                close(row['code_ce'],float(ce[token_support].mean()),2e-6)
                close(row['feature_mse'],float(fmse[token_support].mean()),2e-6)
                close(row['code_accuracy'],float(correct[token_support].mean()),2e-6);code_probes+=1
            else:
                first=stage_rows[update][(row['id'],0.0)]
                require(row['code_logits']==first['code_logits'] and row['code_features']==first['code_features'],'Fidelity arms use different code probes')
                for key in ['code_ce','feature_mse','code_accuracy']:close(row[key],first[key],1e-7)
            for group in [row['source']+'/'+row['profile'],row['source']+'/all','all']:
                grouped['w'+str(int(row['fidelity']))+'/'+group].append(row)
        summaries[str(update)]={}
        for group,items in grouped.items():
            mse=float(np.mean([i['MSE'] for i in items]));summaries[str(update)][group]={
                'cases':len(items),'PSNR':float(-10*np.log10(mse)) if mse else None,'MSE':mse,
                **{k:float(np.mean([i[k] for i in items])) for k in ['SSIM','ArcFace_observed_fixed','code_ce','feature_mse','code_accuracy']}}
    for w in [0.0,1.0]:
        for profile in p['profiles']:
            name='grids/w'+str(int(w))+'_'+profile+'_10_rows.png'
            with Image.open(out/name) as im:sheet=np.asarray(im)
            require(sheet.shape==(2904,1300,3),'Grid geometry differs')
            for i,ref in enumerate(p['references']):
                case=next(c for c in p['cases'] if c['reference_id']==ref['id'] and c['profile']==profile)
                images=[rgb(root/case['input'])]+[rgb(out/stage_rows[u][(case['id'],w)]['prediction']) for u in [0,100,300]]+[targets[ref['id']]]
                for j,image in enumerate(images):
                    y=52+i*288;np.testing.assert_array_equal(sheet[y:y+256,2+j*260:258+j*260],image);grid_cells+=1
    changed=[];initial=torch.load(out/'update0/conditioner.pth',map_location='cpu',weights_only=True)
    for update in [100,300]:
        state=torch.load(out/f'update{update}/conditioner.pth',map_location='cpu',weights_only=True)
        require(state.keys()==initial.keys() and all(torch.isfinite(t).all() for t in state.values()),'Conditioner state schema/nonfinite')
        keys=[k for k in state if not torch.equal(state[k],initial[k])]
        require(any(k.startswith('features.') for k in keys) and any(k.startswith('projection.') for k in keys),'No changes in our conditioner trunk/projection')
        changed.append({'update':update,'changed_parameter_tensors':len(keys)})
    require((pngs,cosines,code_probes,labels,grid_cells)==(300,300,150,2560,500),'Audit counts differ')
    result={'complete':True,'protocol_sha256':expected_sha,'results_sha256':sha(out/'results.json'),'auditor_sha256':sha(Path(__file__)),
            'pngs_checked':pngs,'raw_renders_checked':pngs,'embedding_cosines_rebuilt':cosines,'code_loss_probes_rebuilt':code_probes,
            'teacher_code_labels_checked':labels,'grid_cells_checked':grid_cells,'trace_updates_checked':300,'exposures_checked':600,
            'checkpoint_changes':changed,'source_profile_summaries':summaries,'seconds':time.monotonic()-start,
            'neural_forwards_in_audit':0,'local_backward_calls':0,'local_optimizer_updates':0,
            'native_used':False,'validation_used':False,'production_promoted':False,
            'limitation':'Rebuilds pixel/logit/latent/codebook arithmetic and recorded state/counts. Does not replay CUDA gradients, teacher encoder or recognizer; training-only fitting is not generalization or restoration usefulness.'}
    write(receipt,result);print({k:v for k,v in result.items() if k!='source_profile_summaries'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--expected-protocol-sha',required=True)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();audit(args.root.resolve(),args.expected_protocol_sha,args.results.resolve(),args.receipt.resolve())
