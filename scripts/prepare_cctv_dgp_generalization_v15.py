"""Freeze a thin inference package from existing audited data; no neural calls."""
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cctv_dgp_generalization_v15 as v


def prepare():
    out=ROOT/'outputs/cctv_dgp_generalization_vm_v15'
    v.require(not out.exists(),'Preserve frozen V15 package')
    mixed=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2';mp=v.read(mixed/'mixed_protocol_v9.json')
    v.require(v.sha(mixed/'mixed_protocol_v9.json')==v.MIXED_PIN,'Mixed parent changed')
    parent=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2';pp=v.read(parent/'face_code_fit_protocol_v12.json')
    v.require(v.sha(parent/'face_code_fit_protocol_v12.json')==v.PARENT_PIN,'Frozen parent changed')
    capacity=ROOT/'outputs/cctv_dgp_direct_codes_return_v14_r2/outputs/cctv_dgp_direct_codes_v14_r2'
    cr=v.read(capacity/'results.json');v.require(v.sha(capacity/'results.json')==v.CAPACITY_PIN,'Capacity return changed')
    audit=v.read(ROOT/'outputs/cctv_dgp_direct_codes_return_v14_r2/local_independent_audit.json')
    review=v.read(ROOT/'outputs/cctv_dgp_direct_codes_review_v14_r2.json')
    v.require(audit['complete'] and review['complete'] and review['capacity_demonstrated'] and
              not review['generalization_established'],'Require completed capacity audit/review')
    boundary=ROOT/'outputs/cctv_dgp_generalization_v15_tests.json'
    v.require(v.read(boundary)['complete'],'Require passing arithmetic/split tests')
    out.mkdir();assets={}
    def copy(src,name,pin=None):
        pin=pin or v.sha(src);v.require(v.sha(src)==pin,'Source asset differs')
        dst=v.safe(out,name);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assets[name]=pin;v.require(v.sha(dst)==pin,'Copied payload differs')
    refs=[dict(r) for r in mp['references'] if r['role']=='validation']
    cases=[dict(c) for c in mp['validation_cases']]
    for r in refs:
        for key in ['target','observed']:copy(mixed/r[key],r[key],mp['assets_sha256'][r[key]])
    for c in cases:copy(mixed/c['input'],c['input'],mp['assets_sha256'][c['input']])
    copy(capacity/'update1000/direct_conditioner.pth','trained_conditioner.pth',cr['artifacts_sha256']['update1000/direct_conditioner.pth'])
    for name in ['cctv_dgp_generalization_v15.py','dgp_direct_face_code_v14.py',
            'scripts/run_cctv_dgp_generalization_v15.py','scripts/audit_cctv_dgp_generalization_v15.py',
            'scripts/prepare_cctv_dgp_generalization_v15.py','scripts/supervise_cctv_dgp_generalization_v15.py',
            'scripts/launch_cctv_dgp_generalization_v15.py','scripts/import_cctv_dgp_generalization_v15.py',
            'tests/test_cctv_dgp_generalization_v15.py']:copy(ROOT/name,name)
    for key,path in [('capacity_audit',ROOT/'outputs/cctv_dgp_direct_codes_return_v14_r2/local_independent_audit.json'),
                     ('capacity_review',ROOT/'outputs/cctv_dgp_direct_codes_review_v14_r2.json'),('boundary_tests',boundary)]:copy(path,'lineage/'+key+'.json')
    capassets={name:pin for name,pin in cr['artifacts_sha256'].items() if name.startswith('update1000/none/raw/') or name.startswith('update1000/none/images/')}
    inventory=[{k:r[k] for k in ['id','source_sha256','target_rgb_sha256']} for r in mp['references'] if r['role']=='train']
    plan={'format':v.FORMAT,'date':'2026-10-04','design':v.DESIGN,'assets_sha256':assets,
        'parent_protocol_sha256':v.PARENT_PIN,'parent_assets_sha256':pp['assets_sha256'],
        'capacity_results_sha256':v.CAPACITY_PIN,'capacity_artifacts_sha256':capassets,
        'mixed_protocol_sha256':v.MIXED_PIN,'references':refs,'cases':cases,
        'parity_references':pp['references'],'parity_cases':pp['cases'],
        'training_identity_inventory':inventory,'trained_head_state_hash':v.read(capacity/'update1000/metrics.json')['state_hash'],
        'preview_reference_ids':[r['id'] for s in v.SOURCES for r in sorted((r for r in refs if r['source']==s),key=lambda r:r['id'])[:5]],
        'native_used':False,'native_reserved_used':False,'training':False,'production_promoted':False,
        'rendering_decision':'Freeze no AdaIN/statistic transfer from V14 training-only review before any validation output.',
        'split_limit':'Own-conditioning-train-disjoint development cohort; exact source/target content overlap checked. Near duplicates and pretrained exposure remain unproven. Existing validation has informed earlier development.',
        'selection':'No best.pth or checkpoint selection. Strict source/profile MSE/SSIM/ArcFace and0.1dB guards are diagnostic reports only; no relaxation after results.'}
    v.write(out/v.PLAN,plan);pin=v.sha(out/v.PLAN)
    (out/'protocol.sha256').write_text(pin+'\n',encoding='ascii',newline='\n')
    v.verify(out,parent,pin)
    archive=ROOT/'outputs/cctv-dgp-generalization-v15-execution.tar.gz'
    names=sorted(assets)+[v.PLAN,'protocol.sha256']
    with tarfile.open(archive,'x:gz',compresslevel=3) as t:
        for name in names:t.add(out/name,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers();v.require(len(members)==len(set(names)),'Archive coverage differs')
        for m in members:
            v.safe(out,m.name);v.require(m.isfile() and m.name in names,'Unsafe/unexpected member')
            v.require(hashlib.sha256(t.extractfile(m).read()).hexdigest()==v.sha(out/m.name),'Archive payload differs')
    ap=v.sha(archive);Path(str(archive)+'.sha256').write_text(ap+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    receipt={'complete':True,'protocol_sha256':pin,'archive_sha256':ap,'archive_bytes':archive.stat().st_size,
        'archive_members':len(names),'validation_references':104,'validation_cases':520,'parity_cases':50,
        'neural_forwards':0,'backward_calls':0,'optimizer_updates':0,'native_reserved_used':False,'production_promoted':False}
    v.write(ROOT/'outputs/cctv_dgp_generalization_v15_preparation.json',receipt);print(receipt,flush=True)


if __name__=='__main__':prepare()
