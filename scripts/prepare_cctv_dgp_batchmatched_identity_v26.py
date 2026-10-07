"""Prepare one processing correction and a finite manually launched L4 pilot."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from import_cctv_dgp_v25_gradient_diagnostic_v1 import read,write,sha,require

OLD=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25'
NEW=ROOT/'outputs/cctv_dgp_batchmatched_identity_vm_v26'
OUT=ROOT/'outputs/cctv_dgp_batchmatched_identity_v26_preparation'
NAME='cctv-dgp-batchmatched-identity-v26'


def runtime_source(text):
    for old,new in [('dgp-spatial-feature-capacity-v25','dgp-spatial-batchmatched-identity-capacity-v26'),
        ('scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/cctv_dgp_batchmatched_identity_v26_vm.py'),
        ('scripts/run_v25.sh','scripts/run_v26.sh'),('cctv_dgp_spatial_features_v25_return','cctv_dgp_batchmatched_identity_v26_return'),
        ('cctv-dgp-spatial-features-v25','cctv-dgp-batchmatched-identity-v26'),('V25 update','V26 update')]:text=text.replace(old,new)
    tree=ast.parse(text);prepare=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    lines=text.splitlines(keepends=True)
    revised='''def prepare(root, p):
    require_vm(root, idle=True)
    from cctv_dgp_spatial_features_v25_prepare import prepare_features
    from cctv_dgp_batchmatched_identity_v26_preflight import prove
    head, identity, items, preflight = prepare_features(root, p, require_vm)
    preflight['batchmatched_identity_proof'] = prove(root, p, head, identity, items, require_vm)
    preflight['neural_forward_counts'] = dict(head.audit_forward_counts)
    return head, identity, items, preflight
'''
    text=''.join(lines[:prepare.lineno-1])+revised+''.join(lines[prepare.end_lineno:])
    old='from cctv_dgp_degraded_objective_v24 import cohort_normalizers, objective_terms'
    require(text.count(old)==1,'One original objective import');text=text.replace(old,'from cctv_dgp_batchmatched_identity_v26 import cohort_normalizers, objective_terms')
    anchor="    files+=sorted(root.glob('feature_preflight_*.json'))"
    addition="""
    files+=sorted(root.glob('batchmatched_identity_preflight_*.json'))
    files+=[root/name for name in ['cctv_dgp_batchmatched_identity_v26.py', 'cctv_dgp_batchmatched_identity_v26_preflight.py', 'scripts/install_v26.py', 'installation_receipt.json']]
"""
    require(text.count(anchor)==1,'One export anchor');text=text.replace(anchor,anchor+addition)
    ast.parse(text,feature_version=(3,10));return text


def main():
    started=time.monotonic();require(not NEW.exists() and not OUT.exists(),'Preserve earlier V26 preparation')
    prior=read(ROOT/'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone/milestone.json')
    for name,digest in prior['new_evidence_sha256'].items():require(sha(ROOT/name)==digest,'Previous697 evidence changed: '+name)
    gradient_path=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_independent_audit.json';gradient=read(gradient_path)
    require(gradient['complete'] and gradient['identity_baseline_numerical_discrepancy_confirmed'] and
        gradient['initial_raw_equals_cached_baseline_cases']==50 and gradient['VM_optimizer_updates_verified']==0,'Audited numerical defect required')
    require(sha(OLD/'protocol.json')=='ed43355b1fa0bd768d78e4e2b9ed3bd32cc046fe6221629605899a6a9b2e3175','Original V25 changed')
    original=read(OLD/'protocol.json');NEW.mkdir();inherited={};assets={}
    for name,digest in original['assets_sha256'].items():
        require(sha(OLD/name)==digest,'Original asset changed: '+name)
        revised='lineage/v25_original_'+Path(name).name if name in ['scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/run_v25.sh'] else name
        destination=NEW/revised;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,destination)
        require(sha(destination)==digest,'Local inherited copy changed');assets[revised]=digest;inherited[revised]={'source':name,'sha256':digest}
    source=runtime_source((OLD/'scripts/cctv_dgp_spatial_features_v25_vm.py').read_text())
    runtime=ROOT/'scripts/cctv_dgp_batchmatched_identity_v26_vm.py'
    with runtime.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    transfers={}
    for name,file in [('scripts/cctv_dgp_batchmatched_identity_v26_vm.py',runtime),
        ('cctv_dgp_batchmatched_identity_v26.py',ROOT/'scripts/cctv_dgp_batchmatched_identity_v26.py'),
        ('cctv_dgp_batchmatched_identity_v26_preflight.py',ROOT/'scripts/cctv_dgp_batchmatched_identity_v26_preflight.py'),
        ('scripts/install_v26.py',ROOT/'scripts/install_cctv_dgp_batchmatched_identity_v26_vm.py')]:
        (NEW/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,NEW/name);transfers[name]=sha(NEW/name)
    shell=(OLD/'scripts/run_v25.sh').read_text().replace('scripts/cctv_dgp_spatial_features_v25_vm.py','scripts/cctv_dgp_batchmatched_identity_v26_vm.py')
    with (NEW/'scripts/run_v26.sh').open('x',encoding='utf-8',newline='\n') as f:f.write(shell)
    transfers['scripts/run_v26.sh']=sha(NEW/'scripts/run_v26.sh');assets.update(transfers)
    require(len(inherited)==235 and len(transfers)==5 and len(assets)==240,'Exact inheritance and transfer counts')
    p=copy.deepcopy(original);p['format']='dgp-spatial-batchmatched-identity-capacity-v26';p['assets_sha256']=assets
    p['inherited_assets']=inherited;p['transfer_assets_sha256']=transfers
    p['closed_V25_protocol_sha256']=sha(OLD/'protocol.json')
    p['closed_V25_early_failure_sha256']=sha(ROOT/'outputs/cctv_dgp_spatial_features_v25_return/outputs/early_structure_stop.json')
    p['purpose']='Remove demonstrated same-image identity-reference numerical gradient before testing useful own-DGP spatial training capacity; no app/native/final acceptance'
    p['hypothesis']='A per-case baseline identity reference compared with a different batch/grad context activates a zero-margin penalty on identical outputs. A single shared recognizer call and detached baseline reference must remove that artifact before a fresh otherwise identical finite capacity run.'
    p['difference_from_closed_recipes']='Only identity reference processing changes: baseline+prediction in one frozen recognizer call, detach baseline embedding, original target/geometry/coefficient5/margin0. No longer/unchanged retry, new head, loosened gate or pretrained restorer replacement.'
    p['loss_alignment_evidence']='The independent V25 diagnostic verifies identical baseline/raw outputs for all50 yet initial ArcFace penalty1.3932586e-7 with gradient norm0.0102489 versus landmark0.00706145. It does not establish the entire AdamW failure cause. Same-call reference is an explicit minimal processing hypothesis.'
    p['identity_preflight_policy']={'all50_initial_outputs_exact_baseline':True,'ten_batches_of_five':True,'gradient_calls':20,
        'legacy_discrepancy_must_reproduce':True,'matched_identity_penalty_exactly_zero':True,
        'all26_matched_parameter_gradients_exactly_zero':True,'head_and_recognizer_states_unchanged':True,
        'optimizer_constructed':False,'component_weight':5,'margin':0,'seconds_cap':180,
        'run_preflight_total_seconds_cap':300,'same_degraded_objective_and_original_capacity_gates':True}
    p['thin_install_policy']={'existing_parent':'~/forensic-dgp/cctv_dgp_spatial_features_vm_v25',
        'original_assets_verified_and_copied':235,'seconds_cap':60,'minimum_free_after_copy_bytes':3*1024**3,
        'original_V25_failure_must_remain':True,'original_files_changed':False,'model_or_gradient_calls':0}
    p['prospective_return_audit']={'max_archive_members':1600,'local_CPU_seconds_cap':600,'complete_snapshot_updates':[0,50,400,800],
        'exact_original_quality_and_CPU_replay_tolerances':True,'initial_identity_preflight_required':True,
        'expected_preflight_forward_counts':{'detail_head':60,'DGP':50,'DGP_CPU':4,'fixed_recognizer':120},
        'additional_preflight_gradient_calls_per_process':20,'original_training_backward_counts_unchanged':True,
        'independent_installer/source/feature/count/timing/matrix/PNG/state_audit':True,
        'all50_whole_face_visual_review_required':True,'local_backward_calls':0,'local_optimizer_updates':0}
    p['actual_VM_processing_correction_and_learning_pending']=True
    p['local_basis_sha256']={name:sha(ROOT/name) for name in ['outputs/cctv_dgp_v25_gradient_diagnostic_v1_independent_audit.json',
        'outputs/cctv_dgp_v25_gradient_diagnostic_v1_return_import.json','outputs/cctv_dgp_spatial_features_v25_independent_audit.json',
        'outputs/cctv_dgp_spatial_features_v25_diagnostic/visual_review.json',
        'outputs/cctv_dgp_post_v24_architecture_review_v1/user_architecture_decision.json']}
    write(NEW/'protocol.json',p);pin=sha(NEW/'protocol.json')
    with (NEW/'protocol.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(pin+'  protocol.json\n')
    archive=ROOT/'outputs'/(NAME+'-execution.tar.gz')
    transfer_paths=['protocol.json','protocol.sha256']+sorted(transfers)
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for name in transfer_paths:
            file=NEW/name;info=tar.gettarinfo(str(file),NEW.name+'/'+name)
            info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o755 if file.suffix=='.sh' else 0o644
            with file.open('rb') as f:tar.addfile(info,f)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    OUT.mkdir();receipt={'complete':True,'date':'2026-10-06','scope':'Single demonstrated processing correction; actual L4 proof and useful capacity pending',
        'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'archive_members':7,
        'inherited_assets':235,'assets':240,'transfer_assets':5,'data_or_weights_reuploaded':False,
        'original_head_and_encoder_unchanged':True,'same50_cases_same800_updates_same80_epochs':True,
        'all_original_quality_timing_and_failed_gates_retained':True,'changed_identity_reference_processing_only':True,
        'zero_baseline_loss_and_gradients_must_pass_L4_before_optimizer':True,
        'actual_V26_run_pending':True,'local_model_calls':0,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'VM_actions':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(OUT/'preparation.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
