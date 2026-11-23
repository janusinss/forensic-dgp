"""Stage a prospective manual-L4 gradient packet; no archive, Torch or neural calls.

Preparation copies exact selected source/architecture/checkpoint bytes into a new
namespace, then freezes protocol SHA. It must be separately reviewed before use.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
READINESS=ROOT/"outputs/completion_codeformer_quality_triage_v1/training_data_readiness.json"
READINESS_SHA="87dabe0b2c81350d91f9c0a00666c44dd8b514d1ac23d83c2b571f4cef80f0d6"
PROBE=ROOT/"outputs/completion_codeformer_hq_train_probe_v1/protocol.json"
CHECKPOINT=Path(r"C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth")
CHECKPOINT_SHA="b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797"
CHECKPOINT_BYTES=370786611
ADMISSION_SHA="188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab"
VM_SCRIPT="scripts/cctv_completion_codeformer_gradient_v1_vm.py"
DEFAULT_STAGE=ROOT/"outputs/completion_codeformer_gradient_v1_packet"/"cctv_completion_codeformer_gradient_v1_vm"
NEW_RESEARCH_FILES={VM_SCRIPT,"scripts/export_completion_codeformer_gradient_v1.py","scripts/run_completion_codeformer_gradient_v1.sh"}
SOURCE_FILES=[VM_SCRIPT,"scripts/export_completion_codeformer_gradient_v1.py","scripts/run_completion_codeformer_gradient_v1.sh","third_party/__init__.py","third_party/codeformer/__init__.py",
              "third_party/codeformer/registry.py","third_party/codeformer/vqgan_arch.py",
              "third_party/codeformer/codeformer_arch.py","third_party/codeformer/LICENSE","third_party/codeformer/NOTICE.md"]
LIMITS={"seconds":600,"forward_slots":24,"gradient_queries":16,"optimizer_updates":0,"references":4,"cases":8,"output_bytes":128*1024**2}

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}
def verify_pin(row):
    file=Path(row['path']);need(file.is_file() and not file.is_symlink() and pin(file)=={k:row[k] for k in ('bytes','sha256')},'Source pin drift: '+str(file))
def stage(destination,expected_probe_sha):
    need(not destination.exists(),'Retain existing packet namespace; staging never overwrites')
    need(sha(READINESS)==READINESS_SHA and sha(PROBE)==expected_probe_sha,'Readiness/probe receipt drift')
    ready=read(READINESS);probe=read(PROBE)
    need(ready['complete'] and ready['independent'],'Readiness incomplete')
    rows=ready['genuine_HQ_candidate_sources']['training_sources']
    need(len(rows)==391 and all(r['original_role']==r['legacy_completion_split']=='train' for r in rows),'TRAIN registry changed')
    chosen=sorted(rows,key=lambda r:r['reference_id'])[:4]
    need([r['reference_id'] for r in chosen]==[r['reference_id'] for r in probe['selected_sources'][:4]],'Probe source selection changed')
    need(pin(CHECKPOINT)=={'bytes':CHECKPOINT_BYTES,'sha256':CHECKPOINT_SHA},'Original inpainting weights differ')
    admission_path=ROOT/'outputs/completion_codeformer_manual_v1/source_admission.json'
    need(sha(admission_path)==ADMISSION_SHA,'Architecture source admission drift')
    admission=read(admission_path)
    need(admission['complete'] and admission['independent'] and admission['source_revision']=='b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b','Architecture admission/revision differs')
    admitted={Path(r['path']).resolve():r for r in admission['source_files']}
    for relative in SOURCE_FILES:
        source=ROOT/relative
        need(source.is_file() and not source.is_symlink(),'Missing runtime source')
        if relative not in NEW_RESEARCH_FILES:verify_pin(admitted[source.resolve()])
    references=[];cases=[]
    for source in chosen:
        need(source['source_size']==[1024,1024] and source['source_review_decision']=='accept_clean_restoration','Clean HQ source not admitted')
        verify_pin(source['source']);verify_pin(source['target256'])
        selected=[c for c in probe['cases'] if c['reference_id']==source['reference_id'] and c['mask_kind'] in ('lower_face','eye_band')]
        need(len(selected)==2,'Missing prospective masks')
        for case in selected:verify_pin(case['mask256'])
    destination.mkdir(parents=True)
    def copy(source,relative):
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with Path(source).open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        need(pin(target)==pin(Path(source)),'Copy changed bytes')
    for relative in SOURCE_FILES:copy(ROOT/relative,relative)
    copy(CHECKPOINT,'weights/codeformer_inpainting.pth')
    copy(READINESS,'evidence/training_data_readiness.json')
    copy(PROBE,'evidence/hq_train_probe_protocol.json')
    copy(admission_path,'evidence/source_admission.json')
    for source in chosen:
        reference=source['reference_id'];relative='assets/source1024/'+reference+'.png'
        copy(source['source']['path'],relative)
        references.append({'reference_id':reference,'original_role':'train','legacy_completion_split':'train','source1024':relative,'original_source_pin':source['source'],'attribution':source['attribution'],'clean_review_decision':source['source_review_decision']})
        for kind in ('lower_face','eye_band'):
            case=next(c for c in probe['cases'] if c['reference_id']==reference and c['mask_kind']==kind)
            mask_path='assets/masks256/'+case['id']+'.png';copy(case['mask256']['path'],mask_path)
            cases.append({'id':case['id'],'reference_id':reference,'mask_kind':kind,'mask256':mask_path,'mask_pixels':case['mask_pixels'],'role':'TRAIN_gradient_diagnostic'})
    provenance={'format':'completion-gradient-source-terms-v1','source_revision':admission['source_revision'],'codeformer_license':'third_party/codeformer/LICENSE','codeformer_notice':'third_party/codeformer/NOTICE.md','checkpoint_URL':'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth','checkpoint_SHA256':CHECKPOINT_SHA,'photo_attributions':[{'reference_id':r['reference_id'],**r['attribution']} for r in references],'source_role':'TRAIN only; no reserved FINAL or original VAL examples','pretraining_overlap_unknown':True,'subject_identity_disjointness_unestablished':True,'dataset_terms':'Per-image Flickr licenses retained from audited registry. FFHQ distribution research provenance remains in readiness evidence; no broad deployment permission asserted.'}
    (destination/'evidence/provenance_and_terms.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    files=[{'path':file.relative_to(destination).as_posix(),**pin(file)} for file in sorted(destination.rglob('*')) if file.is_file()]
    protocol={'format':'completion-codeformer-gradient-diagnostic-v1','status':'prepared_not_executed_unqualified','frozen_UTC':datetime.now(timezone.utc).isoformat(),'selection':'First four numerically sorted references from pinned391 TRAIN registry and pinned six-reference HQ probe, independent of model output/error ranking','readiness_sha256':READINESS_SHA,'probe_protocol_sha256':expected_probe_sha,'files':files,'references':references,'cases':cases,'budgets':LIMITS,'optimizer_updates':0,'reserved_FINAL_used':False,'teacher_limitation':'Clean targets are nearest-code/latent targets from the SAME frozen original inpainting encoder and codebook, not an independent clean generative prior','targets':'Genuine1024 originals downsampled directly by PIL LANCZOS to512; fixed256 masks nearest-upsampled512; hole fill white1; normalized[-1,1]','diagnostic_routes':{'code':'CE over all256 nearest-clean-code tokens + latent MSE; encoder/position/feature/transformer/logit groups only','fusion':'mask-only unclampedRGB L1 + seven-pixel-kernel inner-boundaryRGB L1; existing fusion32/64/128 groups only; selected codes and encoder fusion features detached by unmodified architecture'},'precision':'float32 CUDA no autocast/TF32','no_update_policy':'autograd.grad only; no optimizer, scheduler, backward accumulation, parameter writes or new checkpoint saved','gradient_pass':'Every intended group finite, nonzero and connected in all8 cases; zero tensor names retained rather than hidden','quality_claim':'None. Gradient connectivity is not useful learning, improved anatomy, visible preservation, native CCTV evidence or app qualification','unknown_pretraining_overlap':True,'subject_identity_disjointness_unestablished':True}
    (destination/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    digest=sha(destination/'protocol.json')
    (destination.parent/'protocol.sha256').write_text(digest+'\n',encoding='ascii')
    print(json.dumps({'complete':True,'stage':str(destination),'protocol_sha256':digest,'files':len(files),'bytes':sum(r['bytes'] for r in files),'neural_calls':0,'gradient_queries':0,'optimizer_updates':0,'archive_created':False,'manual_VM_review_required':True},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',action='store_true',required=True)
    p.add_argument('--probe-protocol-sha',required=True)
    p.add_argument('--destination',type=Path,default=DEFAULT_STAGE)
    a=p.parse_args();stage(a.destination.resolve(),a.probe_protocol_sha)
if __name__=='__main__':main()
