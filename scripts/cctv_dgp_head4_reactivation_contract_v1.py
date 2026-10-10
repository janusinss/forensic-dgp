"""Finite connected-branch gradient ablation; no neural imports."""
import hashlib
import json
from pathlib import Path

NAME = 'cctv_dgp_head4_reactivation_vm_v1'
STEM = 'cctv-dgp-head4-reactivation-v1'
STATE = 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
PARTS = [('head4.block0.weight',73728),('head4.block1.weight',36864),('smooth.0.weight[:,:64]',36864)]
COMPONENTS = ['MSE_all_profiles','SSIM_loss_all_profiles','ArcFace_loss_all_profiles','landmark_structure_degraded']
BUDGETS = {'worker_seconds':600,'external_seconds':630,'kill_grace_seconds':30,
    'export_seconds':180,'export_external_seconds':210,'minimum_free_disk_bytes':3*1024**3,
    'disk_reserve_bytes':512*1024**2,'return_uncompressed_bytes':768*1024**2,
    'peak_vram_bytes':20*1024**3,'original_forward_calls':20,'repaired_forward_calls':20,
    'recognizer_forward_calls':60,'gradient_queries':160}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate(p):
    assert p['format']=='own-DGP-connected-head4-gradient-ablation-v1'
    assert p['original_state']==STATE and p['original_checkpoint_sha256']==CHECKPOINT
    assert p['budgets']==BUDGETS and p['components']==COMPONENTS
    assert p['parameter_parts']==[{'name':name,'elements':n} for name,n in PARTS]
    assert p['gradient_shape']==[2,20,4,147456] and p['optimizer_updates']==p['epochs']==0
    assert len(p['cases'])==100 and len(p['references'])==20 and len(p['cohorts'])==2
    assert [c['id'] for c in p['cases']]==[cid for co in p['cohorts'] for cid in co['case_ids']]
    assert len({c['id'] for c in p['cases']})==100
    assert all(c['role']=='train' for c in p['cases']) and all(r['role']=='train' for r in p['references'])
    for begin in range(0,100,5):
        selected=p['cases'][begin:begin+5]
        assert [c['profile'] for c in selected]==['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
        assert len({c['source_person_or_reference'] for c in selected})==1
    for key in ['native_DEV_or_reserved_final_used','app_promotion','automatic_follow_on','resume_permitted','goal_complete']:
        assert p[key] is False
    assert p['manual_tmux_required'] and p['all_individual_gradient_vectors_saved']
    assert p['raw_CPU_replay_max_abs']==3e-6 and p['embedding_CPU_replay_max_abs']==5e-5
    assert p['initial_GPU_original_repaired_max_abs']==0
    assert p['unchanged_quality_requirements']=={'early_structure_gain':.01,'final_structure_gain':.1,
        'MSE_regression_tolerance':1e-12,'SSIM_ArcFace_regression_tolerance':1e-6,'brightness_fraction_maximum':.2}
    assert p['quality_requirements_not_tested_by_gradient_success']


def verified_assets(root,pin):
    root=Path(root).resolve(); assert sha(root/'protocol.json')==pin
    p=read(root/'protocol.json'); validate(p)
    for name,digest in p['assets_sha256'].items():
        path=root/name
        assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
        assert sha(path)==digest,name
    allowed={'protocol.json','diagnostic.log','diagnostic_exit_code.txt','supervisor_receipt.json','export_manifest.json','export.log','export_exit_code.txt'}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            name=path.relative_to(root).as_posix()
            assert name in p['assets_sha256'] or name in allowed or name.startswith('outputs/'),name
    return p
