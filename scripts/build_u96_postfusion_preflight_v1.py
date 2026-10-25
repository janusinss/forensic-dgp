"""Build and byte-verify a manual zero-optimizer L4 packet; never execute it."""
import ast
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'outputs/cctv_codeformer_u96_postfusion_preflight_v1_vm'
PACKAGE=ROOT/'research/codeformer_u96_postfusion_preflight_v1'
DATA=ROOT/'outputs/codeformer_u96_postfusion_v1_preparation_r1'
PREFIX='cctv-codeformer-u96-postfusion-preflight-v1'


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()
def pin(p,root=ROOT):return {'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def write(p,d):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def doc(p):return json.loads(p.read_bytes())


def main():
    assert not STAGE.exists(),'Keep any earlier stage immutable'
    archive=ROOT/'outputs'/(PREFIX+'-execution.tar.gz');checksum=Path(str(archive)+'.sha256')
    assert not archive.exists() and not checksum.exists()
    assert shutil.disk_usage(ROOT).free>=6*1024**3
    spec=importlib.util.spec_from_file_location('inert_packet',PACKAGE/'packet.py')
    packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
    # All reviewers must have passed the current source, not a stale pre-repair binding.
    worker_review=doc(PACKAGE/'worker_source_review_v1.json')
    assert worker_review['complete'] and worker_review['source_review_pass']
    assert worker_review['worker']==pin(PACKAGE/'worker.py')
    assert worker_review['contract']==pin(PACKAGE/'CONTRACT.md')
    protocol_review=doc(PACKAGE/'packet_helper_review_v1_r3.json')
    assert protocol_review['complete'] and protocol_review['source_review_pass']
    reviewed={row['path']:{key:row[key] for key in ('path','bytes','sha256')} for row in protocol_review['sources']}
    for name in ('packet.py','entry.py','stage_entry.py'):
        actual=pin(PACKAGE/name);assert reviewed[actual['path']]==actual
    assert protocol_review['contract']==pin(PACKAGE/'CONTRACT.md')
    assert {key:protocol_review['shell'][key] for key in ('path','bytes','sha256')}==pin(PACKAGE/'run_preflight.sh')
    admission=doc(DATA/'input_admission_v1.json');candidate=doc(DATA/'candidate_manifest_v1.json')
    assert admission['counts']=={'readable':8} and admission['optimization_permitted'] is False
    assert admission['readable_reference_ids']==list(packet.REFS)
    assert doc(DATA/'independent_preparation_audit_v1.json')['independent_preparation_audit_pass']
    assert pin(DATA/'candidate_manifest_v1.json')==admission['candidate_manifest']
    STAGE.mkdir();sources=[]
    def copy(source,relative):
        source=Path(source);target=STAGE/relative;target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();original=pin(source)
        shutil.copyfile(source,target)
        assert pin(source)==original and target.stat().st_size==original['bytes'] and sha(target)==original['sha256']
        sources.append({'workspace_source':original,'stage':pin(target,STAGE)})
        return pin(target,STAGE)
    for name in ['__init__.py','packet.py','worker.py','entry.py','CONTRACT.md']:
        copy(PACKAGE/name,'research/codeformer_u96_postfusion_preflight_v1/'+name)
    copy(PACKAGE/'stage_entry.py','scripts/run_u96_postfusion_preflight_v1.py')
    copy(PACKAGE/'run_preflight.sh','scripts/run_preflight.sh')
    assert b'\r' not in (STAGE/'scripts/run_preflight.sh').read_bytes()
    (STAGE/'research/__init__.py').write_bytes(b'"""Inert stage research modules."""\n')
    (STAGE/'third_party/__init__.py').parent.mkdir(exist_ok=True)
    (STAGE/'third_party/__init__.py').write_bytes(b'"""Retained third-party architecture."""\n')
    (STAGE/'third_party/codeformer').mkdir()
    (STAGE/'third_party/codeformer/__init__.py').write_bytes(b'"""Retained CodeFormer modules; attribution in license."""\n')
    for name in ['codeformer_arch.py','vqgan_arch.py','registry.py']:
        copy(ROOT/'third_party/codeformer'/name,'third_party/codeformer/'+name)
    # Retain upstream redistribution terms from the project copy, without network downloads.
    licenses=[p for p in (ROOT/'third_party/codeformer').iterdir() if p.is_file() and ('license' in p.name.lower() or 'notice' in p.name.lower())]
    assert licenses,'Retained CodeFormer license is required in transfer'
    for p in licenses:copy(p,'third_party/codeformer/'+p.name)
    prototype={}
    for key,name in [('source','model.py'),('design','architecture_design_v1_r1.json'),('review','independent_source_review_v1_r1.json')]:
        prototype[key]=copy(ROOT/'research/codeformer_u96_postfusion_v1'/name,'research/codeformer_u96_postfusion_v1/'+name)
    copy(ROOT/'research/codeformer_u96_postfusion_v1/__init__.py','research/codeformer_u96_postfusion_v1/__init__.py')
    for p in [PACKAGE/'worker_source_review_v1.json',PACKAGE/'packet_helper_review_v1_r3.json']:
        copy(p,'evidence/'+p.name)
    data={}
    for key,name in [('provenance','candidate_manifest_v1.json'),('admission','input_admission_v1.json'),('review','isolated_input_review_v1.json'),('audit','independent_preparation_audit_v1.json')]:
        data[key]=copy(DATA/name,'evidence/data/'+name)
    copy(ROOT/'outputs/codeformer_original_spatial_correction_v1_preparation/data_audit_v1/root_clean_target_review_v1.json','evidence/retained_clean_source_review_v1.json')
    copy(ROOT/'CCTV_CODEFORMER_U96_LEARNING_PATH_REVIEW_V1.md','evidence/learning_path_review_v1.md')
    copy(ROOT/'CCTV_CODEFORMER_U96_MILD_CONTEXT_V1_REVIEW.md','evidence/readable_context_study_v1.md')
    for name in ['SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md']:
        copy(ROOT/name,'evidence/current_instructions/'+name)
    model={
        'r2':copy(ROOT/'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth','weights/codeformer_r2.pth'),
        'u96':copy(ROOT/'outputs/codeformer_spatial_continuation_v1_models/codeformer_spatial_u_u96.pth','weights/codeformer_u96_adapter.pth')}
    lookup={(row['reference_id'],row['role']):row for row in candidate['cases']};witnesses=[]
    for ref in packet.REFS:
        target_source=ROOT/lookup[(ref,'clear')]['target']['path']
        target=copy(target_source,'assets/targets/'+ref+'.png')
        for role in ['clear','local_gaussian_1p2']:
            row=lookup[(ref,role)];source=ROOT/row['input']['path']
            assert pin(source)==row['input']
            inp=target if role=='clear' else copy(source,'assets/inputs/'+row['id']+'.png')
            from PIL import Image
            with Image.open(source) as im:rgb=im.tobytes()
            with Image.open(target_source) as im:target_rgb=im.tobytes()
            witnesses.append({'id':row['id'],'reference_id':ref,'role':role,'input':inp,'target':target,
                'input_RGB8_sha256':hashlib.sha256(rgb).hexdigest(),'target_RGB8_sha256':hashlib.sha256(target_rgb).hexdigest(),
                'gradient_witness':role=='local_gaussian_1p2'})
    files={p.relative_to(STAGE).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(STAGE.rglob('*')) if p.is_file()}
    for p in STAGE.rglob('*.py'):ast.parse(p.read_bytes(),feature_version=(3,10))
    protocol={'format':packet.FORMAT,'stage_name':packet.STAGE_NAME,'created_UTC':datetime.now(timezone.utc).isoformat(),
        'optimizer_constructions':0,'optimizer_updates':0,'files':files,'model':model,'prototype':prototype,'data':data,
        'witnesses':witnesses,'limits':packet.LIMITS,'manual_launch_only':True,'automatic_retry_resume':False,
        'training_or_model_quality_implied':False,'return_policy':'All static sources,data,evidence/protocol plus runtime output/logs; baseline weights omitted from return.',
        'source_status':'Prepared exact full-U96 zero-optimizer feasibility diagnostic, not new trained weights.'}
    write(STAGE/'protocol.json',protocol);protocol_sha=sha(STAGE/'protocol.json')
    packet.verify(STAGE,protocol_sha)
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for p in sorted(STAGE.rglob('*')):
                    if p.is_file():tar.add(p,arcname=STAGE.name+'/'+p.relative_to(STAGE).as_posix(),recursive=False)
    digest=sha(archive)
    with checksum.open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    expected={name:row for name,row in files.items()};expected['protocol.json']={'bytes':(STAGE/'protocol.json').stat().st_size,'sha256':protocol_sha}
    seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            assert member.isfile() and not member.issym() and member.name.startswith(STAGE.name+'/')
            name=member.name[len(STAGE.name)+1:];assert name in expected and name not in seen;seen.add(name)
            stream=tar.extractfile(member);h=hashlib.sha256();count=0
            for block in iter(lambda:stream.read(1024**2),b''):h.update(block);count+=len(block)
            assert {'bytes':count,'sha256':h.hexdigest()}==expected[name]
    assert seen==set(expected) and 'torch' not in sys.modules
    receipt={'complete':True,'status':'VERIFIED_MANUAL_ZERO_OPTIMIZER_PACKET_UNRUN','builder':pin(Path(__file__).resolve()),
        'stage':str(STAGE),'protocol':pin(STAGE/'protocol.json'),'archive':pin(archive),'checksum':pin(checksum),
        'static_file_count':len(files),'archive_members':len(seen),'sources':sources,'full_stream_archive_readback':True,
        'shell_LF_only':True,'AST_python3p10_syntax_pass':True,'actual_Python3p10_L4_runtime_unverified':True,
        'neural_gradient_optimizer_training_VM_calls':0,'no_model_quality_claim':True}
    out=ROOT/'outputs/codeformer_u96_postfusion_preflight_v1_packet';out.mkdir(exist_ok=False)
    write(out/'build_receipt_v1.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['complete','status','protocol','archive','static_file_count','archive_members','neural_gradient_optimizer_training_VM_calls']}))


if __name__=='__main__':main()
