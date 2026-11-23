"""Pinned opt-in R1 completion loader; original CodeFormer wrapper is unchanged.

Manifest validation uses only the standard library. Loading is CPU weights_only;
no optimizer is constructed, and all inference parameters remain frozen.
"""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent
CANDIDATE_ROOT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_candidate'
CANDIDATE_PATH=CANDIDATE_ROOT/'codeformer_completion_finetune_r1.pth'
CANDIDATE_MANIFEST=CANDIDATE_ROOT/'candidate_manifest.json'
STATE_AUDIT_ROOT=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return'
PROTOCOL_SHA='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
ORIGINAL_SHA='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
MANIFEST_FORMAT='completion-codeformer-finetune-r1-candidate-v1'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def regular(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink(),'Missing or linked candidate evidence: '+str(path))
    for ancestor in path.parents:
        need(not ancestor.is_symlink(),'Linked candidate evidence directory')
        if ancestor==ROOT:break
    return path

def validate_candidate_manifest(expected_sha,*,checkpoint=None,check_checkpoint=True):
    need(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha),'Explicit candidate manifest SHA256 required')
    manifest_path=regular(CANDIDATE_MANIFEST)
    need(manifest_path.stat().st_size<=1024*1024 and sha(manifest_path)==expected_sha,'Candidate manifest SHA256 differs')
    body=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    need(body.get('format')==MANIFEST_FORMAT and body.get('complete') is True,'Candidate manifest is not admitted')
    need(body.get('protocol_sha256')==PROTOCOL_SHA and body.get('original_checkpoint_sha256')==ORIGINAL_SHA,'Candidate training/base provenance differs')
    need(body.get('model_qualified') is False,'This loader admits an experimental candidate only')
    need(isinstance(body.get('reconstructed_logical_sha256'),str) and re.fullmatch('[0-9a-f]{64}',body['reconstructed_logical_sha256']),'Missing candidate logical state fingerprint')
    pin=body['checkpoint'];candidate=Path(pin['path'])
    need(candidate.resolve()==CANDIDATE_PATH.resolve() and (checkpoint is None or Path(checkpoint).resolve()==CANDIDATE_PATH.resolve()),'Only the fixed reviewed R1 candidate checkpoint is permitted')
    need(type(pin['bytes']) is int and 0<pin['bytes']<=512*1024**2 and re.fullmatch('[0-9a-f]{64}',pin['sha256']),'Invalid candidate size/hash pin')
    audit_pin=body['independent_state_audit'];audit=regular(Path(audit_pin['path']))
    need(audit.resolve().is_relative_to(STATE_AUDIT_ROOT.resolve()) and audit.stat().st_size<=2*1024**2,'State admission escaped retained R1 return')
    need(sha(audit)==audit_pin['sha256'] and json.loads(audit.read_text(encoding='utf-8-sig')).get('complete') is True,'Independent state admission differs or is incomplete')
    if check_checkpoint:
        regular(candidate)
        need(candidate.stat().st_size==pin['bytes'] and sha(candidate)==pin['sha256'],'R1 checkpoint bytes/SHA256 differ')
    return body

def load_candidate(*,manifest_sha,checkpoint=None,device='cpu'):
    need(device=='cpu','The opt-in R1 prototype uses CPU inference only')
    admission=validate_candidate_manifest(manifest_sha,checkpoint=checkpoint)
    import torch
    from third_party.codeformer.codeformer_arch import CodeFormer
    from pretrained_completion import CodeFormerCompletion,CODEFORMER_REVISION
    path=Path(admission['checkpoint']['path'])
    state=torch.load(path,map_location='cpu',weights_only=True)
    need(isinstance(state,dict) and set(state)=={'params_ema'} and isinstance(state['params_ema'],dict),'Candidate must contain only the full params_ema state')
    params=state['params_ema'];need(len(params)==497,'Expected the complete 497-tensor candidate')
    need(all(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==torch.float32 and torch.isfinite(value).all().item() for value in params.values()),'Candidate has invalid/nonfinite/non-FP32 tensors')
    logical=hashlib.sha256()
    for name,value in sorted(params.items()):
        array=value.detach().contiguous();logical.update(name.encode());logical.update(str(tuple(array.shape)).encode());logical.update(str(array.dtype).encode());logical.update(array.numpy().tobytes())
    need(logical.hexdigest()==admission['reconstructed_logical_sha256'],'Candidate logical state differs')
    net=CodeFormer(dim_embd=512,codebook_size=512,n_head=8,n_layers=9,connect_list=['32','64','128'])
    net.load_state_dict(params,strict=True)
    model=CodeFormerCompletion(net).cpu().float().eval().requires_grad_(False)
    need(not model.training and not any(p.requires_grad for p in model.parameters()),'Candidate must stay frozen for local inference')
    validate_candidate_manifest(manifest_sha,checkpoint=path)
    provenance={'backend':'codeformer-inpainting-finetune-r1','source_revision':CODEFORMER_REVISION,'weights_sha256':admission['checkpoint']['sha256'],'original_checkpoint_sha256':ORIGINAL_SHA,'protocol_sha256':PROTOCOL_SHA,'candidate_manifest_sha256':manifest_sha,'logical_state_sha256':logical.hexdigest(),'compositing_policy':'mask-only-v1','input_policy':'visible-normalized-resize-v1','model_label':'Our trained completion R1 — experimental','model_qualified':False,'hidden_identity_verified':False,'original_CodeFormer_retained_as_comparison_baseline':True}
    return model,provenance
