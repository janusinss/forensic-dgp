"""Audit saved detector arrays and proposals without constructing a neural model."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_detector_confidence_v1_r1'
PARENT=ROOT/'outputs/dgp_app_covering_review_v3'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pixels(path,mode='RGB'):
    with Image.open(path) as im:
        assert im.size==(256,256)
        a=np.asarray(im.convert(mode)).copy()
    if mode=='L':assert np.isin(a,[0,255]).all()
    return a


def measure(probability,raw,proposal,operator,protected):
    assert probability.dtype==np.float32 and probability.shape==(256,256)
    assert np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1
    assert raw.shape==proposal.shape==operator.shape==protected.shape==(256,256)
    assert np.isin(raw,[0,1]).all() and np.isin(proposal,[0,1]).all()
    assert operator.dtype==protected.dtype==np.bool_
    np.testing.assert_array_equal(raw,probability>=np.float32(.5))
    np.testing.assert_array_equal(proposal,maximum_filter(raw,size=7,mode='constant',cval=0))
    proposal=proposal.astype(bool);union=proposal|operator
    def stats(area):
        v=probability[area]
        return dict(pixels=int(v.size),minimum=float(v.min()) if v.size else None,
            median=float(np.median(v)) if v.size else None,p95=float(np.quantile(v,.95)) if v.size else None,
            maximum=float(v.max()) if v.size else None,selected_fraction=float((v>=.5).mean()) if v.size else None)
    return dict(raw_pixels=int(raw.sum()),proposal_pixels=int(proposal.sum()),operator_pixels=int(operator.sum()),
        missed_operator_pixels=int((operator&~proposal).sum()),extra_pixels=int((~operator&proposal).sum()),
        iou_against_operator_footprint=float((operator&proposal).sum()/union.sum()) if union.any() else 1.,
        scores=dict(crop=stats(np.ones_like(operator)),operator=stats(operator),outside_operator=stats(~operator),protected=stats(protected)))


def compare_measurement(actual,reported):
    for key,value in actual.items():assert reported[key]==value,key
    return True


def checkpoint_schema_hash(state):
    h=hashlib.sha256()
    for name,v in sorted(state.items()):
        a=v.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        h.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return h.hexdigest()


def main():
    p=read(OUT/'protocol.json');binding=read(OUT/'binding.json')
    assert binding['complete'] and binding['protocol_sha256']==sha(OUT/'protocol.json')
    assert p['format']=='current-dgp-detector-missing-confidence-diagnostic-v1' and len(p['cases'])==36
    assert p['threshold']==.5 and p['native_margin_pixels']==3 and p['crop_geometry']==[256,256]
    assert p['budget']==dict(detector_forward_images=36,completion_forward_images=0,restorer_forward_images=0,
        neural_gradient_queries=0,optimizer_updates=0,seconds=180,output_byte_cap=32*1024**2,RSS_byte_cap=4*1024**3,threads=4,device='cpu')
    assert not p['actual_training_permitted'] and not p['family_qualified'] and p['automatic_generation_outputs']==p['assisted_completion_outputs']==0
    assert all(sha(ROOT/n)==h for n,h in p['assets_sha256'].items())
    prior_plan=read(PARENT/'plan.json');prior_result=read(PARENT/'results.json')
    assert p['cases']==prior_plan['cases'] and prior_result['complete']
    assert sum(c['input_review']=='usable' for c in p['cases'])==32 and sum(c['synthetically_degraded'] for c in p['cases'])==18
    assert all(c['hidden_ground_truth'] is None for c in p['cases'])
    if not (OUT/'results.json').exists():
        fail=read(OUT/'failure.json') if (OUT/'failure.json').exists() else None
        stop=read(OUT/'supervisor.json') if (OUT/'supervisor.json').exists() else None
        assert fail or (stop and (stop['timeout'] or stop['child_exit_code']!=0))
        write(OUT/'failure_audit.json',dict(complete=True,checker_sha256=sha(__file__),scope='Local inference stopped; not a model-quality result',failure=fail,supervisor=stop,
            no_completed_confidence_diagnostic=True,original_source_bindings_unchanged=True,model_qualified=False,goal_complete=False));return
    result=read(OUT/'results.json');assert result['complete'] and result['protocol_sha256']==sha(OUT/'protocol.json')
    assert result['counts']==dict(detector=36,completion=0,restorer=0,gradients=0,optimizer_updates=0)
    assert result['device']=='cpu' and result['gradients_disabled'] and result['seconds']<180
    assert result['exact_all36_prior_app_proposals_reproduced'] and result['no_threshold_or_app_or_model_change']
    assert result['automatic_generation_outputs']==result['assisted_completion_outputs']==0 and not result['automatic_or_assisted_family_qualified']
    expected={f'probabilities/{c["id"]}.npy' for c in p['cases']}|{f'masks/{c["id"]}_{s}.png' for c in p['cases'] for s in ['raw','proposal']}|{f'sheets/page{i:02d}.png' for i in range(1,7)}
    assert set(result['artifacts_sha256'])==expected and len(expected)==114
    for n,h in result['artifacts_sha256'].items():
        f=OUT/n;assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(OUT.resolve()) and sha(f)==h
    assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=32*1024**2
    import torch
    torch.set_grad_enabled(False)
    checkpoint=ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['format']=='dgp-completion-v1' and state['size']==256 and state['width']==24 and state['epoch']==2 and not state['dry_run']
    assert result['detector_checkpoint_sha256']==sha(checkpoint)==p['assets_sha256'][checkpoint.relative_to(ROOT).as_posix()]
    assert checkpoint_schema_hash(state['model'])==result['detector_model_state_before']==result['detector_model_state_after']
    assert result['detector_checkpoint_metadata']==dict(format=state['format'],size=256,width=24,epoch=2)
    assert any(n.startswith('generator.') for n in state['model']) and any(n.startswith('segmenter.') for n in state['model'])
    del state
    rows=[];arrays={};by_id={r['id']:r for r in result['rows']};assert set(by_id)=={c['id'] for c in p['cases']}
    for c in p['cases']:
        row=by_id[c['id']];assert all(row[k]==c[k] for k in ['id','family','condition','input_review'])
        assert row['source_review_exclusion_retained']==(c['input_review']!='usable')
        probability=np.load(OUT/row['probability'],allow_pickle=False)
        rgb=pixels(ROOT/c['input']);raw=(pixels(OUT/row['raw'],'L')==255).astype(np.uint8)
        proposal=(pixels(OUT/row['proposal'],'L')==255).astype(np.uint8)
        operator=pixels(ROOT/c['reviewed'],'L')==255
        protected=pixels(ROOT/c['protected'],'L')==255 if c['protected'] else np.zeros_like(operator)
        measured=measure(probability,raw,proposal,operator,protected);compare_measurement(measured,row)
        np.testing.assert_array_equal(proposal,pixels(PARENT/'images'/(c['id']+'_automatic.png'),'L')//255)
        old=next(r for r in prior_result['rows'] if r['id']==c['id'])
        for key in ['proposal_pixels','operator_pixels','missed_operator_pixels','extra_pixels','iou_against_operator_footprint']:
            assert measured[key]==old['automatic'][key]
        rows.append(dict(id=c['id'],family=c['family'],condition=c['condition'],input_review=c['input_review'],clear_control=c['clear_control'],
            **measured,protected_proposal_pixels=int((proposal.astype(bool)&protected).sum()),operator_protected_overlap=int((operator&protected).sum()),
            exact_prior_app_proposal_reproduction=True,confidence_is_uncalibrated_score=True))
        arrays[c['id']]=(rgb,probability,raw,proposal,operator)
    sheets=[]
    for page in range(6):
        path=OUT/f'sheets/page{page+1:02d}.png'
        with Image.open(path) as im:assert im.size==(1280,1752);sheet=np.asarray(im.convert('RGB'))
        for index,c in enumerate(p['cases'][page*6:(page+1)*6]):
            rgb,probability,raw,proposal,operator=arrays[c['id']]
            heat=cv2.cvtColor(cv2.applyColorMap(np.rint(probability*255).astype(np.uint8),cv2.COLORMAP_MAGMA),cv2.COLOR_BGR2RGB)
            overlay=rgb.copy();inside=proposal.astype(bool)
            overlay[inside]=(rgb[inside].astype(np.float32)*.35+np.array([255,30,30])*.65).astype(np.uint8)
            columns=[rgb,heat,np.repeat((raw*255)[...,None],3,2),overlay,np.repeat((operator.astype(np.uint8)*255)[...,None],3,2)]
            for column,a in enumerate(columns):np.testing.assert_array_equal(sheet[index*292+36:(index+1)*292,column*256:(column+1)*256],a)
        sheets.append(dict(file=path.relative_to(OUT).as_posix(),sha256=sha(path),cells=30,image_size=[1280,1752],case_ids=[c['id'] for c in p['cases'][page*6:(page+1)*6]],direct_review_pending=True))
    assert all(sha(ROOT/n)==h for n,h in p['assets_sha256'].items())
    write(OUT/'audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=sha(OUT/'protocol.json'),results_sha256=sha(OUT/'results.json'),source_bindings=len(p['assets_sha256']),output_bindings=114,
        rows=rows,planned_visual_sheets=sheets,all36_maps_and_raw_and_margin_masks_replayed=True,
        independently_reconstructed_dilation_with_SciPy=True,all36_prior_app_proposal_pixels_and_statistics_exact=True,
        all6_sheet_cells_match_bound_inputs_outputs=True,detector_checkpoint_tensor_schema_and_frozen_receipts_match=True,
        detector_container_includes_unused_generator_weights=True,completion_generator_forward_calls=0,
        local_neural_inference_not_repeated_by_auditor=True,source_review_exclusions_not_promoted=True,
        operator_footprints_approximate_not_segmentation_or_hidden_ground_truth=True,confidence_scores_not_calibrated_probabilities=True,
        threshold_selection_or_sweep_not_performed=True,no_app_model_mask_or_checkpoint_change=True,
        native_CCTV_or_reserved_final_pixels_used=False,automatic_generation_outputs=0,assisted_generation_outputs=0,
        automatic_or_assisted_family_qualified=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,maps=36,prior_masks_exact=36,sheets=6,neural_inference_repeated_by_auditor=0))


if __name__=='__main__':main()
