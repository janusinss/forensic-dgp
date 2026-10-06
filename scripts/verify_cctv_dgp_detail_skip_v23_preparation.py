"""Source, transfer and fixed CPU forward contracts for V23. No backward/fitting."""
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_detail_prior_v22_r1 as a
from import_cctv_dgp_detail_prior_v22_r1 import relative_name


def head_from(source):
    import torch
    from torch.nn import functional as F
    tree = ast.parse(source.read_text(encoding='utf-8'))
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    node = next(n for n in prepare.body if isinstance(n, ast.ClassDef) and n.name == 'DetailHead')
    scope = {'torch': torch, 'nn': torch.nn, 'F': F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-V23-class-only>', 'exec'), scope)
    torch.manual_seed(20261005)
    return scope['DetailHead']().cpu().eval().requires_grad_(False)


def verify():
    import numpy as np
    import torch
    started = time.monotonic(); torch.set_num_threads(4)
    bundle = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23'
    evidence = ROOT / 'outputs/cctv_dgp_detail_skip_v23_preparation'
    prep = a.read(evidence / 'preparation.json');p = a.read(bundle / 'protocol.json');old = a.read(a.BUNDLE / 'protocol.json')
    source = bundle / 'scripts/cctv_dgp_detail_skip_v23.py'
    a.require(a.sha(bundle / 'protocol.json') == prep['protocol_sha256'], 'New protocol differs')
    a.require(p['cases'] == old['cases'] and p['references'] == old['references'] and p['budgets'] == old['budgets'], 'Case/data/budget alteration')
    a.require({k:v for k,v in p['prospective_gates'].items() if k != 'next_if_pass'} ==
              {k:v for k,v in old['prospective_gates'].items() if k != 'next_if_pass'}, 'Quality gate alteration')
    for key in ('seed','updates','epochs','batch_size','learning_rate','weight_decay','gradient_clip_norm','AMP','EMA','snapshot_updates'):
        a.require(p['design'][key] == old['design'][key], 'Training recipe setting differs: ' + key)
    a.require(a.sha(bundle / 'schedule.json') == a.sha(a.BUNDLE / 'schedule.json'), 'Schedule changed')
    schedule = a.read(bundle / 'schedule.json')['batches']
    for epoch in range(80):
        a.require(sorted(i for batch in schedule[epoch*10:(epoch+1)*10] for i in batch) == list(range(50)), 'Epoch exposure differs')
    python_files = 0
    for name,digest in p['assets_sha256'].items():
        path = a.safe(bundle,name);a.require(a.sha(path) == digest,'Asset differs: ' + name)
        if name.endswith('.py'):
            ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10));python_files += 1
    original = ast.parse((a.BUNDLE / 'scripts/cctv_dgp_detail_prior_v22_r1.py').read_text())
    revised = ast.parse(source.read_text())
    def nested(tree, outer, name):
        f = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==outer)
        return next(n for n in f.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('loss','ssim','feature_errors','detail_metric'):
        a.require(ast.dump(nested(original,'run',name),include_attributes=False) ==
                  ast.dump(nested(revised,'run',name),include_attributes=False),'Loss function changed: ' + name)
    archive = ROOT / 'outputs/cctv-dgp-detail-skip-v23-execution.tar.gz'
    a.require(a.sha(archive)==prep['archive_sha256'], 'Archive differs')
    expected = {f.relative_to(bundle).as_posix():a.sha(f) for f in bundle.rglob('*') if f.is_file()}
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers()
        a.require(len(members)==len(expected) and len({m.name.casefold() for m in members})==len(members), 'Archive member count differs')
        for member in members:
            part=relative_name(member.name)
            a.require(member.isfile() and not member.issparse() and part.parts[0]==bundle.name,'Unsafe archive member')
            name='/'.join(part.parts[1:]);payload=stream.extractfile(member).read()
            a.require(name in expected and __import__('hashlib').sha256(payload).hexdigest()==expected[name],'Archive payload differs')
            a.require(member.mode==(0o755 if name=='scripts/run_v23.sh' else 0o644),'Archive mode differs')
    sidecar=Path(str(archive)+'.sha256').read_bytes()
    a.require(sidecar==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii'),'Checksum format differs')
    modes={}
    for mode in ('--verify-transfer','--run'):
        result=subprocess.run([sys.executable,'-B',str(source),'--root',str(bundle),'--protocol-sha',prep['protocol_sha256'],mode],capture_output=True,text=True,timeout=30)
        (evidence/(mode[2:]+'_local.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        a.require(result.returncode==0 if mode=='--verify-transfer' else
                  result.returncode!=0 and 'Existing Linux VM only' in result.stderr,'Platform/transfer guard differs')
        modes[mode]={'exit_code':result.returncode,'expected':True}
    a.require(not (bundle/'outputs').exists(),'Local worker created training outputs')
    head=head_from(source);a.require(sum(v.numel() for v in head.parameters())==4613,'Parameter count differs')
    before=a.state_hash(head.state_dict());calls=0
    def tensors(c):
        camera,base=a.rgb(bundle/c['input']),a.raw_rgb(bundle/c['raw_dgp'])
        mask=a.observed(bundle/c['observed'])
        x=torch.from_numpy(camera.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
        b=torch.from_numpy(base.copy()).permute(2,0,1)[None]
        m=torch.from_numpy(mask.astype(np.float32))[None,None]
        return camera,base,mask,x,b,m
    with torch.inference_mode():
        for c in p['cases']:
            camera,base,mask,x,b,m=tensors(c)
            raw=head(x,b,m)[0].permute(1,2,0).numpy().copy();calls+=1
            a.require(np.array_equal(raw,base) and np.array_equal(a.png(raw,camera,mask),a.rgb(bundle/c['png_dgp'])),'Zero-head50-case raw/PNG baseline differs')
    old_head=a.make_head(a.BUNDLE)
    old_head.load_state_dict(torch.load(ROOT/'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/update50/head.pth',map_location='cpu',weights_only=True),strict=True)
    old_before=a.state_hash(old_head.state_dict());old_calls=0;probes=[]
    probe_ids=('v9_tr_ffhq_00084_clear','v9_tr_asian_00048_motion_lr48')
    plan={'date':'2026-10-06','scope':'Fixed no-target forward sensitivity/implementation diagnostic only; not training or quality',
        'protocol_sha256':prep['protocol_sha256'],'delta_parameter':.001,'new_parameter':'direct.weight[0,6,1,1] camera-red high-pass',
        'old_parameter':'saved update50 tail.weight[0,0,0,0] first latent channel',
        'cases':list(probe_ids),'minimum_signal_ratio':10,'comparison_limit':'Different feature coordinates; demonstrates available spatial signal, not capacity equivalence, trainability or restoration benefit',
        'source_sha256':a.sha(source),'optimizer_updates':0,'backward_calls':0,'coefficient_search':False}
    a.write(evidence/'sensitivity_plan.json',plan)
    for cid in probe_ids:
        c=next(c for c in p['cases'] if c['id']==cid);camera,base,mask,x,b,m=tensors(c)
        with torch.inference_mode():
            n0=head(x,b,m).clone();calls+=1;o0=old_head(x,b,m).clone();old_calls+=1
            head.direct.weight[0,6,1,1]+=.001;old_head.tail.weight[0,0,0,0]+=.001
            n1=head(x,b,m).clone();calls+=1;o1=old_head(x,b,m).clone();old_calls+=1
            head.direct.weight[0,6,1,1]=0
            # Restore the original float value exactly, avoiding arithmetic roundoff.
            old_head.load_state_dict(torch.load(ROOT/'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/update50/head.pth',map_location='cpu',weights_only=True),strict=True)
        selected=torch.from_numpy(a.feature_mask(c,mask))[None,None].expand_as(n0)
        new_rms=float(torch.sqrt((n1-n0).double()[selected].square().mean()))
        old_rms=float(torch.sqrt((o1-o0).double()[selected].square().mean()))
        ratio=new_rms/max(old_rms,1e-12)
        a.require(new_rms>0 and ratio>=10,'Fixed bypass sensitivity insufficient; do not issue a VM capacity claim')
        a.require(torch.equal(n1[~m.expand_as(n1).bool()],b[~m.expand_as(b).bool()]),'Fixed sensitivity changes padding')
        probes.append({'id':cid,'new_raw_response_RMS':new_rms,'old_raw_response_RMS':old_rms,'ratio':ratio})
    a.require(a.state_hash(head.state_dict())==before and a.state_hash(old_head.state_dict())==old_before and
              not any(v.grad is not None or v.requires_grad for v in list(head.parameters())+list(old_head.parameters())),'Contract changed model state or gradients')
    receipt={'complete':True,'date':'2026-10-06','scope':'Independent source/archive and fixed CPU forward contracts; VM gradients/capacity/quality pending',
        'checker_sha256':a.sha(Path(__file__)),'protocol_sha256':prep['protocol_sha256'],'archive_sha256':prep['archive_sha256'],
        'assets':len(p['assets_sha256']),'regular_archive_members':len(expected),'Python310_files':python_files,'same50_cases_and_original_schedule':True,
        'losses_and_quality_gates_unchanged':True,'cases_zero_raw_PNG_parity':50,'trainable_parameters':4613,
        'new_head_CPU_forwards':calls,'closed_v22_CPU_forwards':old_calls,'sensitivity':probes,'sensitivity_plan_sha256':a.sha(evidence/'sensitivity_plan.json'),
        'platform_checks':modes,'states_unchanged':True,'seconds':time.monotonic()-started,
        'recognizer_DGP_forwards':0,'optimizer_updates':0,'backward_calls':0,'VM_actions':False,'app_promotion':False,'goal_complete':False}
    a.write(evidence/'independent_preparation_audit.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':verify()
