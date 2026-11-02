"""Serialize an experimental inference-only R1 completion checkpoint after audit.
CPU tensor deserialization/serialization only; no model, gradients or optimizer.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,time
from audit_completion_codeformer_finetune_return_v1 import pin,bound,read,passive,logical,need
BASE=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION')
RETURN=BASE/'outputs/completion_codeformer_finetune_v1_r1_return'
ROOT=RETURN/'cctv_completion_codeformer_finetune_v1_r1_results'
DEST=BASE/'outputs/completion_codeformer_finetune_v1_r1_candidate'
AUDIT_SHA='e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06'
PROTOCOL='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
def main():
    started=time.monotonic();audit_path=RETURN/'independent_state_audit.json';audit_pin=pin(audit_path);need(audit_pin['sha256']==AUDIT_SHA,'Independent audit identity differs')
    audit=read(audit_path);need(audit['complete'] is True and audit['protocol']['sha256']==PROTOCOL and audit['trainable']['changed_tensors']==332,'State audit not fully passed')
    original=Path(audit['base_checkpoint']['path']);terminal=Path(audit['terminal_checkpoint']['path']);bound(original,audit['base_checkpoint']);bound(terminal,audit['terminal_checkpoint'])
    # Repeat passive review before these fresh CPU-only weights_only loads.
    before_passive={'original':passive(original),'terminal':passive(terminal)}
    import torch
    torch.set_num_threads(1)
    base=torch.load(original,map_location='cpu',weights_only=True)['params_ema']
    saved=torch.load(terminal,map_location='cpu',weights_only=True)
    trained=saved['trainable_state'];need(saved['protocol_sha256']==PROTOCOL,'Terminal protocol differs')
    assembled=dict(base);assembled.update(trained)
    target_logical=audit['frozen_reconstruction']['trained_logical_sha256']
    need(len(assembled)==497 and logical(assembled)==target_logical,'Assembled logical state differs')
    need(logical({n:v for n,v in assembled.items() if n.startswith(('quantize.','generator.'))})==audit['frozen_reconstruction']['frozen_hash'],'Candidate frozen hash differs')
    del saved
    need(not DEST.exists(),'Never overwrite a candidate namespace');DEST.mkdir()
    target=DEST/'codeformer_completion_finetune_r1.pth';partial=DEST/'codeformer_completion_finetune_r1.pth.partial'
    with partial.open('xb') as stream:torch.save({'params_ema':assembled},stream);stream.flush();os.fsync(stream.fileno())
    need(not target.exists(),'Candidate appeared before commit');partial.rename(target)
    candidate_passive=passive(target)
    reloaded=torch.load(target,map_location='cpu',weights_only=True)
    need(set(reloaded)=={'params_ema'} and len(reloaded['params_ema'])==497 and logical(reloaded['params_ema'])==target_logical,'Candidate safe readback differs')
    checkpoint_pin=pin(target)
    source_admission=BASE/'outputs/completion_codeformer_manual_v1/source_admission.json';source_pin=pin(source_admission);source=read(source_admission)
    need(source_pin['sha256']=='188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab','Original source admission changed')
    licenses=[]
    for name in ['LICENSE','NOTICE.md']:
        source_path=BASE/'third_party/codeformer'/name
        admission=next(row for row in source['source_files'] if Path(row['path'])==source_path)
        bound(source_path,admission)
        destination=DEST/name
        with destination.open('xb') as stream:stream.write(source_path.read_bytes())
        licenses.append(pin(destination))
    protocol=read(ROOT/'protocol.json');attribution={'format':'completion-codeformer-r1-training-attribution-v1','protocol_sha256':PROTOCOL,'terms':protocol['terms'],'reference_attributions':[{'id':row['id'],'original_role':row['original_role'],'attribution':row['attribution'],'source1024':row['original_source1024']} for row in protocol['references']],'limits':protocol['data_limits'],'model_pretraining_overlap':'Unknown; no independent identity or blanket license clearance'}
    attribution_path=DEST/'TRAINING_ATTRIBUTION.json'
    with attribution_path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(attribution,stream,indent=2,allow_nan=False);stream.write('\n')
    bound(original,audit['base_checkpoint']);bound(terminal,audit['terminal_checkpoint']);bound(audit_path,audit_pin)
    manifest={'format':'completion-codeformer-finetune-r1-candidate-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'checkpoint':checkpoint_pin,'protocol_sha256':PROTOCOL,'original_checkpoint_sha256':audit['base_checkpoint']['sha256'],'returned_terminal_checkpoint':audit['terminal_checkpoint'],'reconstructed_logical_sha256':target_logical,'frozen_logical_sha256':audit['frozen_reconstruction']['frozen_hash'],'independent_state_audit':audit_pin,'source_admission':source_pin,'original_source_revision':source['source_revision'],'architecture':'CodeFormer inpainting, dim512/codebook512/heads8/layers9/fusion32,64,128; code and fusion paths fine-tuned; generator/codebook retained','source_repository':'https://github.com/sczhou/CodeFormer','license_files':licenses,'training_attribution':pin(attribution_path),'terms':protocol['terms'],'data_limits':protocol['data_limits'],'tensor_partition':{'full':497,'retained_frozen':165,'trained':332},'inference_payload':{'only_root_key':'params_ema','optimizer_or_RNG_payloads':False,'safe_weights_only_readback':True,'ZIP_passive_review':candidate_passive},'baseline_preserved':True,'terminal_recovery_state_preserved':True,'experimental':True,'model_qualified':False,'automatic_app_adoption':False,'independent_visual_and_pixel_admission_pending':True,'generation_claim':'One plausible estimate in reviewed removal area; no exact hidden identity claim','local_actions':{'CPU_weights_only_loads':3,'model_instantiations':0,'neural_forwards':0,'gradients':0,'backwards':0,'optimizer_updates':0,'VM_calls':0},'seconds':time.monotonic()-started}
    manifest_path=DEST/'candidate_manifest.json'
    with manifest_path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'manifest':pin(manifest_path),'checkpoint':checkpoint_pin,'logical_sha256':target_logical,'model_qualified':False,'seconds':manifest['seconds']},indent=2))
if __name__=='__main__':main()