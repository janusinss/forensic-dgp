"""Extract exact reviewed adapters for CPU inference; no neural/training calls."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
import torch
from research_restoration_web import state_sha


def pin(path):
    b=path.read_bytes();return {'path':path.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    review=ROOT/'outputs/codeformer_spatial_continuation_return_review_v1'
    expected={'import':('import_v1.json',None),'state':('independent_checkpoint_contents_v1.json','8bd1e0fdcb671792d83a2f904b01fa1940330117824aa69f375b14fb15fdc2b5'),
      'pixels':('pure_return_audit_v1.json','bffe1b4673a939067467838bd4deee6016280346a05e80d96861f1af94c1e057'),
      'visual':('root_visual_review_v1.json',None)}
    reviews={}
    for key,(name,digest) in expected.items():
        path=review/name;binding=pin(path);proof=json.loads(path.read_bytes())
        assert proof['complete'] is True
        if digest:assert binding['sha256']==digest
        if key in ('state','pixels'):assert proof.get('audit_pass') is True or proof.get('pass') is True
        if key=='visual':assert proof['root_personal_review_complete'] is True and proof['sheets_reviewed']==18
        reviews[key]=binding
    folder=ROOT/'outputs/codeformer_spatial_continuation_v1_models';assert not folder.exists();folder.mkdir()
    status='Experimental only: both reached96 and passed repeated TRAIN stops; inherited R2 preservation failures and full DEV/native/final usefulness remain unresolved.'
    result={'format':'codeformer-spatial-adapters-research-export-v1','complete':True,
        'model_qualified':False,'app_default_changed':False,'neural_gradient_optimizer_VM_calls':0,
        'original515_base_values_unchanged':True,'adapter_tensors':24,'adapter_scalar_parameters':51328,
        'base_tensor_sha256':'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f',
        'protocol_sha256':'c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed',
        'quality_status':status,'review_pins':reviews,'arms':{},
        'format_note':'Adapter-only pth files require the exact retained R2 base; not a standalone pretrained CodeFormer weight.',
        'raw_note':'Saved float arrays represent clipped-and-resized delivery pixels before RGB8 quantization, not unclipped network activations.'}
    canonical={'U':'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a',
               'S':'eb2e443e52ed4b3eace824afdfa6eff0f59a53613b7435304173b4f335e2ec42'}
    for arm,modelkey,source_sha in [('U','spatial_u','9a71153b1d3d127d00f193a93ea15ebb0b66c102033f6f16517ef42e7f41165b'),
                                  ('S','spatial_s','9cf9f414595ea4c48a051cfd89e124d22600fad2fd29e8a66dfa9a075f08f54f')]:
        source=ROOT/f'outputs/cctv_codeformer_spatial_continuation_v1_return/outputs/spatial_continuation_v1/checkpoints/{arm}/u0096.pt'
        assert pin(source)['sha256']==source_sha
        saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
        state={k:v.detach().cpu().clone().contiguous() for k,v in saved['adapter_state'].items()}
        assert len(state)==24 and sum(v.numel() for v in state.values())==51328
        assert all(v.dtype==torch.float32 and torch.isfinite(v).all() and not v.requires_grad for v in state.values())
        assert state_sha(state,torch)==canonical[arm]==saved['adapter_sha256']
        path=folder/f'codeformer_spatial_{arm.lower()}_u96.pth'
        with path.open('xb') as f:torch.save({'format':'codeformer-spatial-adapter-inference-v1','adapter_state':state},f)
        exported=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert set(exported)=={'format','adapter_state'} and state_sha(exported['adapter_state'],torch)==canonical[arm]
        assert all(torch.equal(state[k],exported['adapter_state'][k]) for k in state)
        result['arms'][modelkey]={'source_arm':arm,'absolute_updates':96,'additional_continuation_updates':72,
          'model_qualified':False,'terminal_status':'complete','quality_status':status,
          'label':f'Our spatial {arm}96 · experimental','adapter_tensor_sha256':canonical[arm],
          'source_checkpoint_sha256':source_sha,'source_checkpoint':pin(source),'export':pin(path),
          'all24_tensor_values_exact':True,'optimizer_RNG_removed_for_inference':True}
    manifest=folder/'model_exports_v1.json'
    with manifest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'complete':True,'manifest':pin(manifest),'exports':{k:v['export'] for k,v in result['arms'].items()},
       'all48_exported_tensor_values_exact':True,'neural_optimizer_training_VM_calls':0,'model_qualified':False}))


if __name__=='__main__':main()
