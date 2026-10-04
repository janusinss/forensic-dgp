"""Freeze audited lineage and a thin V14 package; no neural/training calls."""
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cctv_dgp_direct_codes_v14 import (DESIGN, FORMAT, PLAN, PARENT_PROTOCOL,
    PARENT_RESULTS, CONTROL_PROTOCOL, CONTROL_RESULTS, read, write, sha, require, verify, schedule)


def prepare():
    out=ROOT/'outputs/cctv_dgp_direct_codes_vm_v14'
    require(not out.exists(),'Preserve frozen/partial V14 preparation')
    parent=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'
    returned=ROOT/'outputs/cctv_dgp_face_code_fit_return_v12_r2_verified'
    control=ROOT/'outputs/cctv_dgp_face_code_render_controls_return_v13'
    require(sha(parent/'face_code_fit_protocol_v12.json')==PARENT_PROTOCOL,'V12 protocol differs')
    require(sha(returned/'outputs/cctv_dgp_face_code_fit_v12/results.json')==PARENT_RESULTS,'V12 results differ')
    require(sha(control/'render_controls_protocol_v13.json')==CONTROL_PROTOCOL and
            sha(control/'outputs/cctv_dgp_face_code_render_controls_v13/results.json')==CONTROL_RESULTS,'V13 control differs')
    for path in [returned/'local_independent_audit.json',control/'local_independent_audit.json',
                 ROOT/'outputs/cctv_dgp_face_code_controls_review_v13.json']:
        require(read(path)['complete'],'Require completed audit/review')
    tests=read(ROOT/'outputs/cctv_dgp_direct_codes_v14_boundary_tests.json')
    require(tests['complete'] and tests['backward_calls']==tests['optimizer_updates']==0,'Require local guard/inference checks')
    pp=read(parent/'face_code_fit_protocol_v12.json')
    names=['dgp_direct_face_code_v14.py','cctv_dgp_direct_codes_v14.py',
        'scripts/train_cctv_dgp_direct_codes_v14.py','scripts/audit_cctv_dgp_direct_codes_v14.py',
        'scripts/run_cctv_dgp_direct_codes_v14_supervised.py','scripts/import_cctv_dgp_direct_codes_v14.py',
        'scripts/launch_direct_codes_v14_vm.py','scripts/prepare_cctv_dgp_direct_codes_v14.py',
        'tests/test_dgp_direct_face_code_v14.py']
    out.mkdir();pins={}
    for name in names:
        destination=out/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination);pins[name]=sha(ROOT/name)
        require(sha(destination)==pins[name],'Copied source differs')
    protocol={'format':FORMAT,'date':'2026-10-04','design':DESIGN,
        'parent_protocol_sha256':PARENT_PROTOCOL,'parent_results_sha256':PARENT_RESULTS,
        'control_protocol_sha256':CONTROL_PROTOCOL,'control_results_sha256':CONTROL_RESULTS,
        'parent_assets_sha256':pp['assets_sha256'],'references':pp['references'],
        'cases':pp['cases'],'profiles':pp['profiles'],'sources_sha256':pins,
        'native_used':False,'validation_used':False,'native_reserved_used':False,'production_promotion':False,
        'contribution':'Our 2619808-parameter direct residual logit + mean/logstd conditioner; retained DGP, original prior/teacher labels and recognizer frozen.',
        'input_policy':'Same50 fixed RGB256 training cases; cached original encoder features/base logits; frozen DGP RGB calculated once per input. No new warp/CLAHE/sharpening or output-selected sources.',
        'targets':'Clean VQGAN teacher labels from audited V12; rendering-stat targets are channel mean and logstd of corresponding frozen prior codebook vectors (unbiased spatial variance plus1e-5).',
        'loss':'Observed-token CE weight1 plus channel mean MSE weight1 plus channel logstd MSE weight1; no feature MSE or recognizer gradient.',
        'starting_state':'Both output projections zero; cached w0 observed baseline unchanged. Startup observed/predicted raw and PNG parity checked on all50 cases.',
        'rendering':'Same learned top1 codes, frozen w0 decoder, with observed/none/predicted statistics. Outside observed support copy original input. No decoder/skip tuning.',
        'controls':'Two train-reference oracle pairs verify target codebook stats equal no-stat teacher-code rendering; deliberately clean-target informed, not learned outputs.',
        'counts':{'dgp':50,'direct_conditioner':1151,'prior_encoder':0,'prior_transformer_head':0,'prior_generator':454,'unused_v11_conditioner':0,'recognizer':450},
        'decision':'A capacity diagnostic, not a causal architecture ablation: head, objective, LR and finite exposures differ from V12. Review all five 10-row grids, code accuracy, clear preservation and source/profile identity before proposing broader training. No held-out use, best.pth or adoption.',
        'limits':'Ten photographic training identities, Asian targets below256 native resolution, unknown overlap with pretrained FFHQ; not real CCTV truth or Zamboanga evidence.',
        'lineage_audits_sha256':{str(path.relative_to(ROOT)):sha(path) for path in
            [returned/'local_independent_audit.json',control/'local_independent_audit.json',
             ROOT/'outputs/cctv_dgp_face_code_controls_review_v13.json',ROOT/'outputs/cctv_dgp_direct_codes_v14_boundary_tests.json']}}
    write(out/'schedule_v14.json',{'steps':schedule(protocol)})
    protocol['schedule_sha256']=sha(out/'schedule_v14.json')
    write(out/PLAN,protocol);pin=sha(out/PLAN)
    (out/'direct_code_protocol_v14.sha256').write_text(pin+'\n',encoding='ascii',newline='\n')
    verify(out,parent,pin)
    archive=ROOT/'outputs/cctv-dgp-direct-codes-v14-execution.tar.gz'
    names+=['schedule_v14.json',PLAN,'direct_code_protocol_v14.sha256']
    with tarfile.open(archive,'x:gz',compresslevel=3) as stream:
        for name in sorted(names):stream.add(out/name,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();require(len(members)==len(names),'Archive member count differs')
        for member in members:
            require(member.isfile() and member.name in names,'Unexpected archive member')
            require(hashlib.sha256(stream.extractfile(member).read()).hexdigest()==sha(out/member.name),'Archived payload differs')
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii',newline='\n')
    result={'complete':True,'protocol_sha256':pin,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'archive_members':len(names),'optimizer_updates_budget':1000,'exposures_budget':2000,
        'local_neural_forwards':0,'local_backward_calls':0,'local_optimizer_updates':0,'native_or_validation_used':False}
    write(ROOT/'outputs/cctv_dgp_direct_codes_v14_preparation.json',result);print(result,flush=True)


if __name__=='__main__':prepare()
