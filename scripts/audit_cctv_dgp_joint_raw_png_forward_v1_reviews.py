"""Independent saved-array review binding check; no model construction/forwards."""
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256,256)
        return np.asarray(im.convert(mode)).copy()


def main():
    receipt = ANALYSIS/'review_binding_audit.json'
    assert not receipt.exists()
    audit = read(ANALYSIS/'independent_audit.json')
    finite = read(ANALYSIS/'finite_metrics.json')
    learning = read(ANALYSIS/'finite_learning_review.json')
    spatial = read(ANALYSIS/'spatial_patch_review.json')
    quantization = read(ANALYSIS/'quantization_review.json')
    visual = read(ANALYSIS/'visual_review.json')
    assert all(d['complete'] for d in [audit,finite,learning,spatial,quantization,visual])
    assert sha(RETURN/'protocol.json') == PIN == audit['protocol_sha256']
    assert sha(ROOT/'scripts/audit_cctv_dgp_joint_raw_png_forward_v1_return.py') == audit['checker_sha256']
    assert sha(ANALYSIS/'finite_metrics.json') == audit['finite_metrics_sha256'] == learning['finite_metrics_sha256']
    assert all(d['independent_audit_sha256']==sha(ANALYSIS/'independent_audit.json') for d in [learning,spatial,quantization])
    assert visual['raw_and_PNG_audit_sha256'] == sha(ANALYSIS/'independent_audit.json')
    assert visual['all20_sheets_and800_cells_inspected'] and not visual['independent_final_review']
    reviewer=ROOT/'scripts/review_cctv_dgp_joint_raw_png_forward_v1_evidence.py'
    assert sha(reviewer)==learning['reviewer_sha256']==spatial['reviewer_sha256']==visual['record_script_sha256']
    stopped=ANALYSIS/'finite_learning_review_preparation_stop'
    assert sha(stopped/'failed_reviewer.py') == read(stopped/'stop.json')['failed_reviewer_sha256'] == learning['retained_strict_preparation_stop_sha256']
    p=read(RETURN/'protocol.json'); old=read(PARENT/'protocol.json')
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['additional_source_degraded_structure_gain_required']==.01
    torch.set_grad_enabled(False)
    initial=torch.load(PARENT/'initializers/initial_B.pth',map_location='cpu',weights_only=True)
    edges=np.cumsum([0]+[initial[n].numel() for n in p['learning_names']])
    with np.load(RETURN/'directions.npz',allow_pickle=False) as saved:
        directions=[saved[f'balanced_p{i:02d}'] for i in range(37)]
    deltas={prop['label']:np.concatenate([((initial[n].numpy()+np.float32(prop['rate'])*directions[i]).astype(np.float64)
                                        -initial[n].numpy().astype(np.float64)).reshape(-1)
                                        for i,n in enumerate(p['learning_names'])]) for prop in p['proposals']}
    closed=[i for i,n in enumerate(p['learning_names']) if n.startswith(('style_conditioner.','spatial_conditioners.'))]
    assert len(closed)==12 and sum(initial[p['learning_names'][i]].numel() for i in closed)==2580864
    assert all(not np.count_nonzero(directions[i]) for i in closed)
    dots=0
    for row in learning['groups']:
        bindings=read(ROOT/'outputs/cctv_dgp_png_group_gradients_v1_r2_analysis/joint_raw_png_direction_v1_r2/direction_review.json')['aggregate_gradient_bindings']
        path=ROOT/next(b['file'] for b in bindings if b['key']==row['key'])
        assert sha(path)==row['saved_gradient_sha256']
        g=np.load(path,allow_pickle=False,mmap_mode='r')
        assert all(not np.count_nonzero(g[edges[i]:edges[i+1]]) for i in closed)
        for label, delta in deltas.items():
            value=math.fsum(float(np.dot(g[i:i+131072],delta[i:i+131072])) for i in range(0,len(g),131072))
            entry=row['steps'][label]
            assert math.isclose(value,entry['actual_parameter_bytes_first_order_change'],rel_tol=1e-12,abs_tol=1e-16)
            assert math.isclose(entry['actual_loss']-row['baseline_loss']-value,
                                entry['remainder_vs_saved_derivative'],rel_tol=1e-12,abs_tol=1e-16)
            dots+=1
    assert dots==256
    summary=[]
    for label, trial in finite['proposals'].items():
        assert not trial['all_preview_requirements_pass'] and not trial['source_structure_requirement_pass']
        summary.append(dict(proposal=label,raw_structure_gain_percent=100*trial['gates']['raw']['relative_feature_gain'],
            PNG_structure_gain_percent=100*trial['gates']['png']['relative_feature_gain'],
            raw_preservation_failure_count=len(trial['gates']['raw']['preservation_failures']),
            PNG_preservation_failure_count=len(trial['gates']['png']['preservation_failures']),
            source_structure_gains=trial['source_structure_gains'],all_preview_requirements_pass=False))
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    z=np.arange(-6,7,dtype=np.float64);k=np.exp(-z*z/8);k/=k.sum()
    def hp(a):
        v=np.sum(a.astype(np.float64)*np.array([.299,.587,.114]),axis=2)
        return v-cv2.sepFilter2D(v,cv2.CV_64F,k,k,borderType=cv2.BORDER_REFLECT)
    refs={r['id']:r for r in old['references']}
    records={(r['proposal'],r['id'],r['stage'],r['region']):r for r in spatial['rows']}
    qrecords={(r['proposal'],r['id']):r for r in quantization['rows']}
    assert len(records)==4000 and len(qrecords)==400
    regions=['eye_left_image','eye_right_image','nose','mouth_left_image','mouth_right_image']
    checked=0;quant_checked=0
    controls=read(RETURN/'outputs/joint_raw_png_forward_v1/run_results.json')['controls']
    for control in controls:
        begin=control['begin'];cases=old['cases'][begin:begin+5]
        baseline,_=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',[c['id'] for c in cases])
        after={prop['label']:np.load(RETURN/f"outputs/joint_raw_png_forward_v1/{prop['label']}/b{begin//5:03d}.npy",allow_pickle=False) for prop in p['proposals']}
        for i,c in enumerate(cases):
            ref=refs[c['source_person_or_reference']]
            observed=pixels(PARENT/ref['observed'],'L')>0
            inside=cv2.erode(observed.astype(np.uint8),np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            target=hp(pixels(PARENT/ref['target']).astype(np.float64)/255)
            base8=pixels(PARENT/'outputs/bank_comparison_v1/baseline/previews'/(c['id']+'.png'))
            base={'raw':hp(baseline[i]),'png':hp(base8.astype(np.float64)/255)}
            for prop in p['proposals']:
                label=prop['label'];arr=after[label][i]
                png8=pixels(RETURN/'outputs/joint_raw_png_forward_v1'/label/'png'/(c['id']+'.png'))
                q=qrecords[(label,c['id'])]
                assert q['observed_RGB_values']==int(observed.sum()*3)
                assert q['raw_changed_RGB_values']==np.count_nonzero(arr[observed]!=baseline[i][observed])
                assert q['PNG_changed_RGB_values']==np.count_nonzero(png8[observed]!=base8[observed])
                assert q['PNG_maximum_absolute_byte_change']==int(abs(png8[observed].astype(int)-base8[observed].astype(int)).max())
                quant_checked+=1
                for stage, candidate in [('raw',hp(arr)),('png',hp(png8.astype(np.float64)/255))]:
                    for name,(x,y) in zip(regions,np.floor(c['landmarks5_canvas_xy']).astype(int),strict=True):
                        mask=np.zeros((256,256),bool);mask[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)]=True
                        mask &= inside
                        r=records[(label,c['id'],stage,name)]
                        assert mask.sum()==r['pixels'] and r['profile']==c['profile'] and r['source']==c['source']
                        assert math.isclose(float(np.square(base[stage]-target)[mask].mean()),r['baseline_MSE'],rel_tol=1e-12,abs_tol=1e-16)
                        assert math.isclose(float(np.square(candidate-target)[mask].mean()),r['candidate_MSE'],rel_tol=1e-12,abs_tol=1e-16)
                        checked+=1
    assert checked==4000 and quant_checked==400
    cells=0
    assert len(visual['sheets'])==len(audit['comparison_sheets'])==20
    for seen,bound in zip(visual['sheets'],audit['comparison_sheets'],strict=True):
        path=ROOT/seen['file'];assert seen['file']==bound['file'] and sha(path)==seen['sha256']==bound['sha256']
        assert seen['all40_cells_inspected'] and not seen['convincing_useful_gain_over_retained_DGP']
        assert seen['requested_display_detail']=='original' and seen['observations']
        with Image.open(path) as im: image=np.asarray(im.convert('RGB')).copy();assert im.size==(2048,1460)
        for i,row in enumerate(bound['cells']):
            patch=image[40+(i//8)*284:296+(i//8)*284,(i%8)*256:(i%8+1)*256]
            assert hashlib.sha256(patch.tobytes()).hexdigest()==row['RGB_sha256'];cells+=1
    assert cells==800
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    model=PARENT/'cctv_dgp_bank_comparison_v1_model.py'
    source=model.read_text(encoding='utf-8')
    assert sha(model)==p['parent_required_sha256']['cctv_dgp_bank_comparison_v1_model.py']
    assert 'nn.init.zeros_(layer.weight)' in source and 'styles = self.mean_style[None, None] + self.style_conditioner(map4)' in source
    assert 'self.net.fpn(image * 2 - 1)' in source and 'result = self.current.reconstruct(image, maps, extras)' in source
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        independent_return_audit_sha256=sha(ANALYSIS/'independent_audit.json'),
        evidence_sha256={n:sha(ANALYSIS/n) for n in ['finite_metrics.json','finite_learning_review.json','spatial_patch_review.json','quantization_review.json','visual_review.json']},
        summary=summary,actual_parameter_step_derivative_dots_rechecked=256,
        exact_unchanged_conditioning_tensors=12,exact_unchanged_conditioning_values=2580864,
        spatial_patch_rows_rechecked_with_independent_OpenCV_kernel=4000,quantization_cases_recounted=400,
        all20_visual_sheets_and800_source_cells_bound=True,all_four_proposals_rejected_for_adoption=True,
        model_source_sha256=sha(model),original_checkpoint_and_app_examples_preserved=True,
        no_new_model_neural_gradient_or_optimizer_calls=True,no_agent_VM_connection_or_launch=True,
        independent_final_review=False,model_qualified=False,more_epochs_qualified=False,app_adoption=False,goal_complete=False))
    print(dict(complete=True,spatial_patch_rows=checked,quantization_cases=quant_checked,source_cells=cells,
               derivative_dots=dots,all_proposals_unqualified=True,neural_calls=0))


if __name__=='__main__': main()
