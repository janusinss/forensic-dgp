"""Frozen-state mixed gradient measurements on the VM; no optimizer exists."""
import copy
import itertools
import json
from pathlib import Path
import sys
import tarfile
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coverage_gradient_audit import objective_terms, cosine
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks
from scripts.train_coverage_vm import require_vm_gpu, sha, PROTOCOL_SHA, PARENT_SHA, safe_path


def main():
    require_vm_gpu()
    root=Path.cwd().resolve()
    out=root/'outputs/coverage_gradient_audit'
    archive=root/'coverage-gradient-results.tar.gz'
    if out.exists() or archive.exists():raise ValueError('Preserve existing diagnostic output')
    train=root/'outputs/coverage_training_vm'
    run=json.loads((train/'run.json').read_text())
    inventory=root/'outputs/coverage_protocol_v1/vm_inventory.json'
    assert sha(inventory)==run['inventory_sha256']
    for name,digest in json.loads(inventory.read_text()).items():assert sha(safe_path(root,name))==digest
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    assert sha(protocol_path)==PROTOCOL_SHA==run['protocol_sha256']
    assert sha(train/'replay_pixels.pth')==run['replay_sha256']
    cache=torch.load(train/'replay_pixels.pth',weights_only=True,map_location='cpu')
    protocol=json.loads(protocol_path.read_text())
    paths={'initial':root/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth',
           'control':train/'control/epoch_10.pth','extended':train/'extended/epoch_10.pth'}
    expected={'initial':PARENT_SHA,'control':'771c4347ee738d0be4d9a9ec146a8f6940c7339fcad9d530409770dc4f6f17d8',
              'extended':'9c251c89574156bfdb9e8594b2ba4355191f79a0d0bc89b20209fcaf75feddff'}
    for k,p in paths.items():assert sha(p)==expected[k]
    rows=[r for r in load_manifest('dataset/detector_training_extension_v2/manifest.json') if r['split']=='train']
    real=ReviewedMasks(rows,256)
    original={r['image_sha256'] for r in load_manifest('dataset/detector_glare_review_v3/manifest.json')}
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    parent,_=load_completion(paths['initial'],'cuda')
    teacher=copy.deepcopy(parent.segmenter).eval().requires_grad_(False)
    del parent
    # Same first six extended-schedule batches at all three frozen states.
    batches=protocol['schedules']['extended']['batches'][0][:6]
    result={'optimizer_updates':0,'selection':'first six epoch1 extended batches, fixed before gradient measurements',
            'checkpoints':expected,'protocol_sha256':PROTOCOL_SHA,'torch':str(torch.__version__),
            'gpu':torch.cuda.get_device_name(),'states':{},'script_sha256':sha(__file__),
            'helper_sha256':sha('coverage_gradient_audit.py'),'decomposition_tolerance':1e-4,
            'limitations':'Frozen-state local directions, not AdamW updates or proof of causality. No held-out gradients.'}
    for name,path in paths.items():
        model,_=load_completion(path,'cuda');model.eval();model.generator.requires_grad_(False)
        before={k:v.cpu().clone() for k,v in model.state_dict().items()}
        params=list(model.segmenter.parameters());measurements=[]
        for batch_number,batch in enumerate(batches):
            data=[real[i] if i<len(real) else (cache[i-len(real)][0].float()/255,cache[i-len(real)][1].float()) for i in batch]
            x,m=(torch.stack([r[j] for r in data]).cuda() for j in (0,1))
            synthetic=torch.tensor([i>=len(real) for i in batch],device='cuda')
            added=torch.tensor([i<len(real) and rows[i]['image_sha256'] not in original for i in batch],device='cuda')
            logits=model.detect(x)
            with torch.no_grad():reference=teacher(x[synthetic])
            terms,full=objective_terms(logits,m,synthetic,added,reference)
            def gradient(loss):
                values=torch.autograd.grad(loss,params,retain_graph=True)
                return torch.cat([v.detach().flatten().cpu().double() for v in values])
            gradients={k:gradient(v) for k,v in terms.items()}
            total=gradient(full)
            relative=float((sum(gradients.values())-total).norm()/total.norm().clamp_min(1e-12))
            if relative>1e-4:raise ValueError('Gradient decomposition does not reproduce objective')
            real_grad=sum(v for k,v in gradients.items() if k.startswith('real_'))
            replay_grad=gradients['replay_supervised']+gradients['teacher']
            measurements.append({'batch':batch_number,'indices':batch,
                'real_images':[rows[i]['image'] for i in batch if i<len(real)],
                'added_images':[rows[i]['image'] for i in batch if i<len(real) and rows[i]['image_sha256'] not in original],
                'losses':{k:float(v.detach()) for k,v in terms.items()},
                'norms':{k:float(v.norm()) for k,v in gradients.items()},
                'cosines':{a+' vs '+b:cosine(gradients[a],gradients[b]) for a,b in itertools.combinations(gradients,2)},
                'real_vs_replay_cosine':cosine(real_grad,replay_grad),'total_norm':float(total.norm()),
                'clip1_scale':min(1.,1./max(float(total.norm()),1e-12)),
                'decomposition_relative_error':relative})
            print(name,batch_number,'cosine',measurements[-1]['real_vs_replay_cosine'],flush=True)
        assert all(torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in params)
        result['states'][name]={'state_unchanged':True,'batches':measurements}
        del model
    out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out/'results.json',arcname='results.json')
        tar.add(__file__,arcname='audit_coverage_gradients_vm.py')
        tar.add('coverage_gradient_audit.py',arcname='coverage_gradient_audit.py')
    print('Complete; zero optimizer updates. Download',archive,flush=True)


if __name__=='__main__':main()
