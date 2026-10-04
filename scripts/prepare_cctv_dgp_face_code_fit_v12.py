"""Create and independently verify a thin local execution package; no training."""
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cctv_dgp_face_code_fit_v12 import FORMAT,DESIGN,PROFILES,SEED,schedule,sha,read,write,verify,require
OUT=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12'


def prepare():
    require(not OUT.exists(),'Preserve existing frozen execution package')
    parent=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2';p=read(parent/'mixed_protocol_v9.json')
    require(sha(parent/'mixed_protocol_v9.json')=='6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70','Audited parent changed')
    checks=ROOT/'outputs/dgp_face_code_prior_checks_v11_r2';cp=read(checks/'frozen_plan.json')
    require(sha(checks/'frozen_plan.json')=='865cef25d09029615a62d097d5051386bb1981cdd50c252a02bb0a3591761d92','Prior-capacity plan changed')
    audit=read(checks/'run/independent_audit.json');review=read(checks/'run/assistant_visual_review.json')
    require(audit['complete'] and review['complete'] and not review['production_promoted'],'Require audited/reviewed capacity check')
    require(read(ROOT/'outputs/cctv_dgp_face_code_fit_v12_boundary_tests_r2.json')['complete'],'Require passing VM-boundary tests')
    OUT.mkdir();assets={};members=[];refs=[];cases=[]
    def copy(path,name):
        destination=OUT/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination);assets[name]=sha(path);require(sha(destination)==assets[name],'Copied asset differs');members.append(name)
    ids={r['id'] for r in cp['references']}
    require(ids=={r['id'] for r in p['references'] if r['id'] in ids and r['role']=='train'},'Inherited training role changed')
    for r in cp['references']:
        target='data/targets/'+r['id']+'.png';observed='data/observed/'+r['id']+'.png'
        for name,key in [(target,'target'),(observed,'observed')]:
            path=parent/r[key];require(sha(path)==p['assets_sha256'][r[key]],'Target/observation changed');copy(path,name)
        refs.append({**r,'target':target,'observed':observed,'files':None})
    for c in p['training_cases']:
        if c['reference_id'] not in ids:continue
        path=parent/c['input'];require(sha(path)==p['assets_sha256'][c['input']],'Fixed camera input changed')
        name='data/inputs/'+c['id']+'.png';copy(path,name);cases.append({**c,'input':name})
    for name in ['cctv_dgp_face_code_fit_v12.py','dgp_face_code_conditioner_v11.py','dgp_frozen_inference_v2.py',
        'dgp_face_restoration.py','cctv_dgp_frozen_norm.py','cctv_dgp_pilot.py','cctv_dgp_targets_v6.py',
        'pretrained_face_restoration.py','face_prior_grid_v10.py',
        'scripts/train_cctv_dgp_face_code_fit_v12.py','scripts/audit_cctv_dgp_face_code_fit_v12.py',
        'scripts/run_cctv_dgp_face_code_fit_v12_supervised.py','scripts/prepare_cctv_dgp_face_code_fit_v12.py',
        'tests/test_cctv_dgp_face_code_fit_v12.py','tests/test_dgp_face_code_conditioner_v11.py']:
        copy(ROOT/name,name)
    for folder in ['models','third_party/codeformer']:
        for path in (ROOT/folder).glob('*'):
            if path.is_file() and (path.suffix=='.py' or path.name in ['LICENSE','NOTICE.md']):copy(path,path.relative_to(ROOT).as_posix())
    for key,path in [('v11_audit',checks/'run/independent_audit.json'),('v11_review',checks/'run/assistant_visual_review.json'),
                     ('v11_plan',checks/'frozen_plan.json'),('v11_execution',checks/'run/neural_execution_receipt.json'),
                     ('teacher_acquisition',ROOT/'outputs/codeformer_teacher_pretrained_v1/acquisition.json'),
                     ('teacher_compatibility',ROOT/'outputs/codeformer_teacher_component_diagnostic_v11.json'),
                     ('boundary_tests',ROOT/'outputs/cctv_dgp_face_code_fit_v12_boundary_tests_r2.json')]:copy(path,'lineage/'+key+'.json')
    weight_sources={'dgp':('weights/dgp_v2.pth','outputs/cctv_dgp_mixed_return_v9/outputs/cctv_dgp_mixed_v9/checkpoints/baseline.pth','646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'),
        'prior':('weights/codeformer.pth','outputs/codeformer_restoration_pretrained_v1/codeformer.pth','1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7'),
        'teacher':('weights/vqgan_code1024.pth','outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth','4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f'),
        'arcface':('weights/w600k_r50.onnx','outputs/cctv_dgp_vm_bundle_v1/weights/w600k_r50.onnx','4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43')}
    weights={};external={}
    for key,(name,source,pin) in weight_sources.items():
        src=ROOT/source;require(sha(src)==pin,'Changed pretrained/retained weight')
        destination=OUT/name;destination.parent.mkdir(exist_ok=True)
        os.link(src,destination)  # Same-volume read-only weight reuse; never modify these files.
        assets[name]=pin;weights[key]=name;external[name]={'sha256':pin,'bytes':src.stat().st_size,'local_source':source}
    protocol={'format':FORMAT,'date':'2026-10-04','design':DESIGN,'profiles':PROFILES,'references':refs,'cases':cases,
        'weights':weights,'external_weights':external,'assets_sha256':assets,'frozen_before_VM_execution':True,
        'native_used':False,'validation_used':False,'native_reserved_used':False,'production_promotion':False,
        'selection':'Same deterministic ten training references as V11; all five frozen camera profiles. No output-driven sampling.',
        'contribution':'Train only our 455072-parameter code conditioner; original DGP and official prior/teacher/recognizer frozen.',
        'loss_rule':'Official code/feature supervision: observed-token CE weight0.5 plus feature MSE weight1.0; labels from clean VQGAN teacher, feature targets from frozen prior codebook lookup.',
        'input_policy':'Exact inherited RGB256 and observations; original RGB to prior encoder, frozen V2 DGP RGB as extra conditioning; no new warp/CLAHE/sharpening.',
        'fitting_decision':'Check gradient reachability, code accuracy/loss, trained conditioning changes and original-cell output structure. No automatic best selection; training-only fit never establishes CCTV generalization.',
        'pretrained_license':'S-Lab License1.0 retained; observed acquisition SHA, not author-published checksum.',
        'target_limit':'Asian sources remain below256 native resolution. Clean targets are photographic proxies, not actual Zamboanga CCTV truth.'}
    write(OUT/'schedule_v12.json',{'steps':schedule(protocol)})
    assets['schedule_v12.json']=sha(OUT/'schedule_v12.json');members.append('schedule_v12.json')
    write(OUT/'face_code_fit_protocol_v12.json',protocol);pin=sha(OUT/'face_code_fit_protocol_v12.json')
    (OUT/'face_code_fit_protocol_v12.sha256').write_text(pin+'\n',encoding='ascii',newline='\n')
    members+=['face_code_fit_protocol_v12.json','face_code_fit_protocol_v12.sha256']
    verify(OUT,pin)
    archive=ROOT/'outputs/cctv-dgp-face-code-v12-execution.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=3) as stream:
        for name in sorted(members):stream.add(OUT/name,arcname=OUT.name+'/'+name,recursive=False)
    # Independently verify every archive payload, rather than trusting tar creation.
    with tarfile.open(archive,'r:gz') as stream:
        contents=stream.getmembers();require(len(contents)==len(members),'Archive member count differs')
        for member in contents:
            require(member.isfile() and member.name.startswith(OUT.name+'/') and '..' not in Path(member.name).parts,'Unsafe member')
            name=member.name[len(OUT.name)+1:];require(name in members,'Unexpected member')
            data=stream.extractfile(member).read();require(hashlib.sha256(data).hexdigest()==sha(OUT/name),'Archive member differs')
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    write(ROOT/'outputs/cctv_dgp_face_code_v12_preparation.json',{'complete':True,'protocol_sha256':pin,
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'archive_members':len(members),
        'external_weight_files':external,'updates_budget':300,'exposures_budget':600,
        'local_backward_calls':0,'local_optimizer_updates':0,'native_or_validation_used':False,'production_promoted':False})
    print({'prepared':True,'protocol_sha256':pin,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'archive_members':len(members),'external_weights':len(external)})


if __name__=='__main__':prepare()
