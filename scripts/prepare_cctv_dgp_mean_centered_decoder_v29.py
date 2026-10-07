"""One changed finite L4 pilot; no local training or historical source edits."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'
V28 = ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return'
DIAG = ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return'
NAME = 'cctv_dgp_mean_centered_decoder_vm_v29'
BUNDLE = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_preparation'
STEM = 'cctv-dgp-mean-centered-decoder-v29'
WORKER = 'scripts/cctv_dgp_mean_centered_decoder_v29_vm.py'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)


CLOSED_CHECK = '''def closed_v28_and_diagnostic_check(parent,p):
    v28=parent.parent/'cctv_dgp_active_original_decoder_vm_v28'
    diagnostic=parent.parent/'cctv_dgp_v28_preservation_diagnostic_v1_vm'
    for folder,key in [(v28,'closed_V28_evidence_sha256'),(diagnostic,'closed_diagnostic_evidence_sha256')]:
        for name,digest in p[key].items():
            path=(folder/name).resolve()
            assert path.is_relative_to(folder) and path.is_file() and not path.is_symlink() and sha(path)==digest,name
    original=read(v28/'outputs/results.json');measurement=read(diagnostic/'outputs/results.json')
    assert original['complete'] and original['optimizer_updates']==800 and not original['necessary_capacity_pass']
    assert len(original['preservation_failures'])==3
    assert measurement['complete'] and measurement['component_gradient_calls']==100
    assert measurement['optimizer_updates']==measurement['backwards']==measurement['epochs']==0
    assert measurement['candidate_state_before_after']==original['candidate_DGP_state']
    assert not measurement['new_checkpoint_created'] and not measurement['app_promotion']
    return v28


'''


def worker():
    s=(OLD/'scripts/cctv_dgp_active_original_decoder_v28_vm.py').read_text(encoding='utf-8')
    s=s.replace('cctv_dgp_active_original_decoder_vm_v28','cctv_dgp_mean_centered_decoder_vm_v29')
    s=s.replace('own-DGP-active-original-decoder-capacity-v28','own-DGP-mean-centered-original-decoder-capacity-v29')
    s=s.replace('cctv-dgp-active-original-decoder-v28','cctv-dgp-mean-centered-decoder-v29')
    s=s.replace('cctv_dgp_active_original_decoder_v28_return/','cctv_dgp_mean_centered_decoder_v29_return/')
    s=s.replace('V28 preflight300s/worker1800s cap','V29 preflight300s/worker1800s cap')
    s=s.replace('V28 worker cap1800 seconds','V29 worker cap1800 seconds')
    s=replace(s,'def run(root, parent, p, pin):',CLOSED_CHECK+'def run(root, parent, p, pin):')
    s=replace(s,'        guard(root, idle=True)','        v28_dependency=closed_v28_and_diagnostic_check(parent,p)\n        guard(root, idle=True)')
    s=replace(s,'        sys.path.insert(0, str(parent))','        sys.path.insert(0, str(parent))\n        sys.path.insert(0,str(v28_dependency))')
    s=replace(s,'from cctv_dgp_active_original_decoder_v28 import ActiveOriginalDecoderV28 as OriginalDecoderCandidate','from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29 as OriginalDecoderCandidate')
    s=replace(s,"candidate(b['x'],b['mask'])","candidate(b['x'],b['mask'],b['base'])")
    s=replace(s,'from cctv_dgp_active_decoder_v28_training import train','from cctv_dgp_mean_centered_v29_training import train')
    s=replace(s,'lambda:parent_check(parent,p))','lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p)))')
    s=replace(s,'        base = parent_check(parent,p)','        base = parent_check(parent,p)\n        closed_v28_and_diagnostic_check(parent,p)')
    s=replace(s,'Finite active-original-decoder L4 pilot. Historical all14 failure retained.','Finite mean-centered-original-decoder L4 pilot. Historical failures retained.')
    ast.parse(s,feature_version=(3,10));return s


def training():
    s=(OLD/'cctv_dgp_active_decoder_v28_training.py').read_text(encoding='utf-8')
    s=s.replace("candidate(b['x'],b['mask'])","candidate(b['x'],b['mask'],b['base'])")
    assert s.count("candidate(b['x'],b['mask'],b['base'])")==2
    s=s.replace('dgp_candidate_v28.pth','dgp_candidate_v29.pth').replace('V28 fitting cap1500 seconds','V29 fitting cap1500 seconds').replace('V28 update','V29 update')
    s=replace(s,"'frozen_encoder_head4_and_all_buffers_unchanged':True,","'mean_centered_path_used':True,'mean_centering_after_training_only':False,\n                'original_seven_loss_weights_unchanged':True,'clipping_can_reintroduce_mean_shift':True,\n                'frozen_encoder_head4_and_all_buffers_unchanged':True,")
    ast.parse(s,feature_version=(3,10));return s


def prospective_auditor(pin):
    s=(ROOT/'scripts/audit_cctv_dgp_active_original_decoder_v28_return.py').read_text(encoding='utf-8')
    s=s.replace('cctv_dgp_active_original_decoder_vm_v28','cctv_dgp_mean_centered_decoder_vm_v29')
    s=s.replace('cctv_dgp_active_original_decoder_v28_return','cctv_dgp_mean_centered_decoder_v29_return')
    s=s.replace('cctv-dgp-active-original-decoder-v28','cctv-dgp-mean-centered-decoder-v29')
    s=s.replace('cctv_dgp_active_original_decoder_v28_independent_audit','cctv_dgp_mean_centered_decoder_v29_independent_audit')
    s=s.replace('dgp_candidate_v28.pth','dgp_candidate_v29.pth')
    s=replace(s,"PIN='27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8'","PIN="+repr(pin))
    s=replace(s,"from cctv_dgp_active_decoder_v28_training import frozen_partition","from cctv_dgp_mean_centered_v29_training import frozen_partition\n    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta")
    s=replace(s,"    sys.path.insert(0,str(PARENT));sys.path.insert(0,str(BUNDLE))","    sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'))\n    sys.path.insert(0,str(PARENT));sys.path.insert(0,str(BUNDLE))")
    s=replace(s,"                expected=torch.where(support,net(x),x)[0].permute(1,2,0).numpy().copy();CPU_candidate+=1", "                baseline_raw=np.load(OUT/'outputs/initial_baseline'/(c['id']+'.npy'),allow_pickle=False)\n                baseline_tensor=torch.from_numpy(baseline_raw.copy()).permute(2,0,1)[None]\n                expected=center_observed_delta(torch.where(support,net(x),x),baseline_tensor,x,support)[0].permute(1,2,0).numpy().copy();CPU_candidate+=1")
    s=replace(s,"    assert p['retained_capacity_gates']==base['prospective_gates'] and p['case_rows']==base['cases']", "    assert p['retained_capacity_gates']==base['prospective_gates'] and p['case_rows']==base['cases']\n    for folder,key in [(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return','closed_V28_evidence_sha256'),(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return','closed_diagnostic_evidence_sha256')]:\n        for name,digest in p[key].items():assert sha(folder/name)==digest,name\n    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name")
    s=replace(s,"    assert not terminal['app_promotion'] and not terminal['goal_complete']", "    assert not terminal['app_promotion'] and not terminal['goal_complete']\n    if result_path.exists():\n        assert terminal['mean_centered_path_used'] and terminal['original_seven_loss_weights_unchanged']\n        assert terminal['mean_centering_after_training_only'] is False and terminal['clipping_can_reintroduce_mean_shift']")
    ast.parse(s,feature_version=(3,10));return s


def main():
    started=time.monotonic();assert not BUNDLE.exists() and not PREP.exists()
    previous=ROOT/'outputs/dgp_v28_return_and_preservation_diagnostic_v1_milestone/milestone.json'
    assert sha(previous)=='5e163e129b438081214b0914845a3204431abc70751457b101dcd368ce87ac86'
    protected=read(previous)
    for name,digest in protected['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    diagnostic=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit_r1.json')
    analysis=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis/results.json')
    assert diagnostic['complete'] and diagnostic['diagnostic_complete'] and diagnostic['optimizer_updates']==0
    assert analysis['mean_anchor_increases_along_plain_SGD_direction_at_final_state'] and analysis['clear_SSIM_and_ArcFace_hinges_decrease_along_same_direction']
    assert analysis['V28_failed_gates_retained']==3 and analysis['mean_control_failed_gates_retained']==5
    BUNDLE.mkdir();(BUNDLE/'scripts').mkdir();PREP.mkdir()
    (BUNDLE/'cctv_dgp_mean_centered_decoder_v29.py').write_bytes((ROOT/'scripts/cctv_dgp_mean_centered_decoder_v29.py').read_bytes())
    (BUNDLE/'cctv_dgp_mean_centered_v29_training.py').write_text(training(),encoding='utf-8',newline='\n')
    (BUNDLE/'cctv_dgp_app_input_v28.py').write_bytes((OLD/'cctv_dgp_app_input_v28.py').read_bytes())
    (BUNDLE/WORKER).write_text(worker(),encoding='utf-8',newline='\n')
    (BUNDLE/'schedule.json').write_bytes((OLD/'schedule.json').read_bytes())
    shell=(OLD/'scripts/run_v28.sh').read_bytes().replace(b'cctv_dgp_active_original_decoder_v28_vm.py',b'cctv_dgp_mean_centered_decoder_v29_vm.py')
    (BUNDLE/'scripts/run_v29.sh').write_bytes(shell)
    p=copy.deepcopy(read(OLD/'protocol.json'))
    p.update({'format':'own-DGP-mean-centered-original-decoder-capacity-v29','date':'2026-10-06',
        'purpose':'One fixed in-optimization mean-centering path to test the demonstrated unconstrained photometric direction; all prior preservation failures remain rejected',
        'design':'Separate copy of original DGP selected12 decoder tensors. Frozen same-input original-DGP baseline. Subtract observed per-channel mean of candidate-minus-baseline BEFORE unchanged losses and before final clamping. No target/profile/source/identity conditions in forward. No loss-weight, optimizer, schedule or acceptance change.',
        'difference_from_failed_recipes':'V28 trained an unconstrained raw output and only a later post-training mean control was measured. V29 applies one fixed differentiable centering path during every training and inference forward, allowing spatial weights to adapt under it. The post-training V28 mean-control failures are not accepted.',
        'mean_centering':'Delta=unprojectedCandidate-baseline; RGBmean=sum(delta*observed)/sum(observed); output=where(observed,clamp(baseline+delta-RGBmean,0,1),input). Float32 throughout. No strength parameter. Clipping can reintroduce a mean shift; the original brightness gate remains decisive.',
        'baseline_policy':'Fresh retained original DGP on the identical prepared input and five-case cohort; cached only within this fixed pilot after parity. No target or clean-label inference access. Deployment would require the original baseline plus learned candidate and this fixed projection, not a standalone snapshot.',
        'initial_parity':'All50 zero-delta candidate projected raw/PNG outputs equal the fresh unchanged DGP before any optimizer; all12 gradients must pass again under the changed path.',
        'gradient_policy':'Selected12 mean-centered-path improvement gradients connected, finite and aggregate nonzero; four initial preservation values/all12 gradients exactly zero in all ten batches. No local gradients. No surrogate gradients or quantization trick.',
        'failure_policy':'Stop on first source/state/gradient/time or early structure failure; retain partials/checkpoints/logs. Final800 only with all original17-group and brightness gates. No overwrite/resume/unchanged rerun or automatic next recipe.',
        'next':'Independently audit all checkpoints, projected raw arrays, delivered PNGs, embeddings, all17 groups and timing; review all50 final faces. Native/broader/app qualification is a later separately frozen step only after this training-capacity qualification.',
        'execution':'Human gcloud transfer/SSH/tmux on existing NVIDIA L4/g2-standard-4 only; no assistant cloud action',
        'projection_learned_parameters':0,'diagnostic_extra_terms_used_as_losses':False,
        'new_checkpoint_created':False})
    p['closed_V28_evidence_sha256']={name:sha(V28/name) for name in ['protocol.json',*read(OLD/'protocol.json')['assets_sha256'],'outputs/results.json','outputs/update800/metrics.json','outputs/update800/dgp_candidate_v28.pth']}
    p['closed_diagnostic_evidence_sha256']={name:sha(DIAG/name) for name in ['protocol.json',*read(DIAG/'protocol.json')['assets_sha256'],'outputs/results.json','outputs/gradient_summary.json','outputs/gradient_components.npy','outputs/parameter_displacement.npy','supervisor_receipt.json','trainer_exit_code.txt']}
    basis=['scripts/prepare_cctv_dgp_mean_centered_decoder_v29.py','scripts/cctv_dgp_mean_centered_decoder_v29.py',
        'scripts/analyse_cctv_dgp_v28_preservation_diagnostic_v1.py',
        'outputs/cctv_dgp_v28_preservation_diagnostic_v1_independent_audit_r1.json',
        'outputs/cctv_dgp_v28_preservation_diagnostic_v1_audit_repair_r1/original_failure.json',
        'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis/results.json',
        'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/results.json',
        'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/independent_readback.json',
        'outputs/cctv_dgp_active_original_decoder_v28_visual_review/visual_review.json',
        'tests/test_cctv_dgp_mean_centered_decoder_v29.py',previous.relative_to(ROOT).as_posix()]
    p['local_basis_sha256']={name:sha(ROOT/name) for name in basis}
    p['assets_sha256']={path.relative_to(BUNDLE).as_posix():sha(path) for path in sorted(BUNDLE.rglob('*')) if path.is_file()}
    write(BUNDLE/'protocol.json',p);pin=sha(BUNDLE/'protocol.json')
    audit=ROOT/'scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py'
    with audit.open('x',encoding='utf-8',newline='\n') as f:f.write(prospective_auditor(pin))
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file():tar.add(path,arcname=NAME+'/'+path.relative_to(BUNDLE).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    receipt={'complete':True,'protocol_sha256':pin,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'packet_regular_files':len(p['assets_sha256'])+1,'closed_V28_files':len(p['closed_V28_evidence_sha256']),
        'closed_diagnostic_files':len(p['closed_diagnostic_evidence_sha256']),'original_data_or_weights_uploaded':False,
        'new_finite_training_protocol_created':True,'optimizer_updates_bound':800,'epochs_bound':80,
        'loss_weights_schedule_optimizer_and_gates_unchanged':True,'prospective_auditor_sha256':sha(audit),
        'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'actual_VM_training_started':False,
        'human_manual_VM_execution_required':True,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(PREP/'preparation.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
