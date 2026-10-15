"""CPU saved-value loss/byte fixtures only; no model inference or autograd."""
from datetime import datetime, timezone
import ast
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from prepare_cctv_dgp_png_group_gradients_v1 import ROOT, OUT, PARENT, BASE, build_worker, build_contract
from import_cctv_dgp_bank_group_step_check_v1_return import read, sha, write
from cctv_dgp_png_group_gradients_v1_losses import delivered_surrogates, pixel_and_structure_losses


def rejected(fn,*args):
    try: fn(*args)
    except (AssertionError, ValueError): return True
    raise AssertionError('Invalid declared fixture accepted')


def main():
    start=time.monotonic(); torch.set_num_threads(4);torch.set_grad_enabled(False)
    receipt=OUT/(BASE+'_forward_audit.json');assert not receipt.exists()
    worker,_=build_worker();contract,_=build_contract()
    scope={};exec(compile(contract,'pure PNG contract','exec'),scope)
    p=read(PARENT/'protocol.json');blocks,groups=scope['scopes'](p['cases'],p['preview_case_ids'])
    assert len(groups)==32 and all(group['key'].startswith('PNG/') and group['stage']=='PNG' for group in groups)
    contributions={group['key']:[] for group in groups}
    for block in blocks:
        for query in range(16): contributions[scope['query_key'](block['source'],query)].append(block['reference'])
    assert sum(map(len,contributions.values()))==320 and all(len(set(v))==len(v)==10 for v in contributions.values())
    assert scope['batch_values'](dict(MSE=np.arange(5.),SSIM_loss=np.arange(5.)+10,landmark_structure=np.arange(5.)+20),np.arange(5.)+30)[-1]==22.5
    tree=ast.parse(worker,feature_version=(3,10))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['step','backward','copy_','add_','sub_','Adam','AdamW','SGD'] for node in ast.walk(tree))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='grad' for n in ast.walk(tree))==1
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert isinstance(run.body[0],ast.Expr) and run.body[0].value.func.id=='require_vm'
    assert worker.index('require_vm(root,parent)') < worker.index('import torch')
    assert 'assert maximum_error == 0' in worker
    assert worker.index('exact_RGB32_and_feature64_PNG_forward') > worker.index('delivered_surrogates(pred,delivered_u8)')
    assert worker.index('Initial delivered loss definition differs') < worker.index('torch.autograd.grad')
    with torch.inference_mode():
        bytes_all=torch.arange(256,dtype=torch.uint8).repeat(768).reshape(1,3,256,256)
        raw=bytes_all.float()/255.
        exact32,exact64=delivered_surrogates(raw,bytes_all)
        assert torch.equal(exact32,raw) and torch.equal(exact64,bytes_all.double()/255.)
        assert not exact32.requires_grad and not exact64.requires_grad
        bad=[('raw_dtype',raw.half(),bytes_all),('byte_dtype',raw,bytes_all.int()),('geometry',raw[:,:,:128,:128],bytes_all[:,:,:128,:128])]
        invalid=[]
        for label,a,b in bad:
            assert rejected(delivered_surrogates,a,b);invalid.append(label)
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support
    refs={r['id']:r for r in p['references']}
    metrics={r['id']:r for r in read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows']}
    def image(path,mode='RGB'):
        with Image.open(path) as im:
            assert im.size==(256,256)
            return np.asarray(im.convert(mode)).copy()
    def tensor(a): return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous()
    limits=dict(MSE=1e-13,SSIM_loss=1e-6,feature_MSE=1e-12)
    errors={key:0. for key in limits};bindings=[];records=[]
    for block in blocks:
        begin=block['begin'];cs=p['cases'][begin:begin+5];raw,emb=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',block['case_ids'])
        cameras=np.stack([image(PARENT/c['input']) for c in cs]); targets=np.stack([image(PARENT/refs[c['source_person_or_reference']]['target']) for c in cs])
        masks=np.stack([image(PARENT/refs[c['source_person_or_reference']]['observed'],'L')>0 for c in cs])
        features=np.stack([feature_support(mask,c['landmarks5_canvas_xy']) for mask,c in zip(masks,cs)])
        png=np.floor(raw*np.float32(255)).astype(np.uint8)
        for i,c in enumerate(cs):
            png[i][~masks[i]]=cameras[i][~masks[i]]
            path=PARENT/'outputs/bank_comparison_v1/baseline/previews'/(c['id']+'.png')
            assert np.array_equal(png[i],image(path));bindings.append(dict(case=c['id'],PNG_sha256=sha(path)))
        with torch.inference_mode():
            raw_t=torch.from_numpy(raw).permute(0,3,1,2).contiguous()
            byte_t=torch.from_numpy(png).permute(0,3,1,2).contiguous()
            rgb32,feature64=delivered_surrogates(raw_t,byte_t)
            assert np.array_equal(rgb32.permute(0,2,3,1).numpy(),png.astype(np.float32)/np.float32(255))
            assert np.array_equal(feature64.permute(0,2,3,1).numpy(),png.astype(np.float64)/255.)
            terms=pixel_and_structure_losses(rgb32,tensor(targets),torch.from_numpy(masks.astype(np.float32))[:,None],torch.from_numpy(features.astype(np.float32))[:,None],feature64)
        for i,c in enumerate(cs):
            expected=metrics[c['id']]['png'];fields=dict(MSE=expected['MSE'],SSIM_loss=1-expected['SSIM'],feature_MSE=expected['landmark_high_frequency_MSE'])
            values=dict(MSE=float(terms['MSE'][i]),SSIM_loss=float(terms['SSIM_loss'][i]),feature_MSE=float(terms['landmark_structure'][i]))
            for key,got in values.items():
                error=abs(got-fields[key]);errors[key]=max(errors[key],error)
                assert error<=limits[key],(c['id'],key,got,fields[key],error)
            records.append(dict(case=c['id'],values=values,retained_PNG_metrics=fields))
        assert time.monotonic()-start<180, 'Saved-value preparation cap'
    assert len(records)==len(bindings)==100
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        prepared_worker_draft_sha256=hashlib.sha256(worker.encode()).hexdigest(),loss_helper_sha256=sha(ROOT/'scripts/cctv_dgp_png_group_gradients_v1_losses.py'),
        saved_forward_cases=100,exact_delivered_RGB32_and_feature64_values=True,records=records,PNG_bindings=bindings,
        maximum_forward_loss_errors=errors,forward_loss_numeric_tolerances=limits,
        numeric_loss_reproduction_not_quality_gates=True,all320_query_bindings_and32_PNG_groups_checked=True,
        all256_byte_levels_checked=True,invalid_surrogate_fixtures_rejected=invalid,
        zero_local_model_inference_autograd_or_optimizer_calls=True,backward_surrogate_not_locally_executed=True,
        hypothesis_requires_manual_L4_measurement=True,no_checkpoint_or_VM_action=True,
        seconds=time.monotonic()-start,model_qualified=False,goal_complete=False))
    print(dict(complete=True,cases=100,groups=32,maximum_forward_loss_errors=errors,seconds=time.monotonic()-start))


if __name__=='__main__':main()
