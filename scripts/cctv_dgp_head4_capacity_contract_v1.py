"""Prospective 50-update necessary-capacity stage. No neural imports."""
import hashlib
import json
from pathlib import Path

NAME='cctv_dgp_head4_capacity_vm_v1';STEM='cctv-dgp-head4-capacity-v1'
STATE='d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
CHECKPOINT='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
WEIGHTS=[.2,1.,1.,1.]
NORMALIZERS=[0.022182490369457405,0.30685945008702126,0.5253697948823282,0.0017099954417771745]
BUDGETS={'worker_seconds':2400,'external_seconds':2430,'kill_grace_seconds':30,'fit_seconds':300,
    'snapshot_seconds':900,'export_seconds':600,'export_external_seconds':630,'peak_vram_bytes':20*1024**3,
    'minimum_free_disk_bytes':14*1024**3,'disk_reserve_bytes':1024**3,'return_uncompressed_bytes':6*1024**3,
    'original_forward_calls':781,'candidate_forward_calls':1614,'recognizer_forward_calls':1614,
    'maximum_gradient_queries':6,'maximum_optimizer_updates':50}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def validate(p):
    assert p['format']=='own-DGP-repaired-head4-capacity-stage-v1'
    assert p['original_state']==STATE and p['original_checkpoint_sha256']==CHECKPOINT and p['budgets']==BUDGETS
    assert p['reconstruction_weights']==WEIGHTS and p['normalizers']==NORMALIZERS
    assert p['optimizer']=={'name':'Adam','lr':1e-5,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-5,'gradient_clip_L2':1.}
    assert p['parameter_tensors']==3 and p['parameter_elements']==147456
    assert p['maximum_updates']==50 and p['snapshots']==[0,50] and p['updates_per_full_epoch']==781
    assert len(p['cases'])==3905 and len({c['id'] for c in p['cases']})==3905 and len(p['references'])==781
    assert all(c['role']=='train' for c in p['cases']) and all(r['role']=='train' for r in p['references'])
    assert len(p['preview_case_ids'])==100 and len(p['schedule'])==50
    for ids in p['schedule']:
        assert len(set(ids))==len(ids)==5 and all(type(i) is int and 0<=i<3905 for i in ids)
        batch=[p['cases'][i] for i in ids]
        assert [c['profile'] for c in batch]==['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
        assert len({c['source_person_or_reference'] for c in batch})==1
    assert p['scientific_thresholds']=={'early_structure_gain':.01,'final_structure_gain':.1,
        'MSE_regression_tolerance':1e-12,'SSIM_ArcFace_regression_tolerance':1e-6,'brightness_fraction_maximum':.2}
    for k in ['native_DEV_or_reserved_final_used','automatic_follow_on','app_promotion','goal_complete','failed_run_resume_allowed']:
        assert p[k] is False
    assert p['manual_tmux_required'] and p['stage50_does_not_qualify_epochs_or_model']
    assert p['raw_storage']=='lossless float32 byte planes; candidate XOR to original; all3905 retained in both snapshots'


def verify(root,pin):
    root=Path(root).resolve();assert sha(root/'protocol.json')==pin;p=read(root/'protocol.json');validate(p)
    for n,d in p['assets_sha256'].items():
        f=root/n;assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(root) and sha(f)==d,n
    allowed={'protocol.json','trainer.log','trainer_exit_code.txt','supervisor_receipt.json','export_manifest.json','export.log','export_exit_code.txt'}
    for f in root.rglob('*'):
        assert not f.is_symlink()
        if f.is_file():
            n=f.relative_to(root).as_posix();assert n in p['assets_sha256'] or n in allowed or n.startswith('outputs/'),n
    return p
