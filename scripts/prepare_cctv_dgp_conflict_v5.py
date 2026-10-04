"""Prepare verified additive V5 assets; no model forwards, backward or updates."""
import ast
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from cctv_dgp_pilot import read,sha,write
from cctv_dgp_conflict_training_v5 import (configure_paths,verify_protocol_v5,ARMS,
    REQUIRED_ASSETS,START_SHA,START_STATE,V4_RESULTS_SHA,V4_AUDIT_SHA)
from cctv_dgp_objective_diagnostic_v4 import verify_recipe
from derive_cctv_dgp_conflict_v5 import derive_sources

PARENT=ROOT/'outputs/cctv_dgp_vm_bundle_v1'
PERCEPTUAL=ROOT/'outputs/cctv_dgp_perceptual_vm_v3'
DIAGNOSTIC=ROOT/'outputs/cctv_dgp_objective_vm_v4'
RETURN=ROOT/'outputs/cctv_dgp_normfix_return_v2/outputs/cctv_dgp_pilot'
V3_RETURN=ROOT/'outputs/cctv_dgp_perceptual_return_v3/outputs/cctv_dgp_perceptual_v3'
V4_RETURN=ROOT/'outputs/cctv_dgp_objective_return_v4/outputs/cctv_dgp_objective_diagnostic_v4'
CAPSULE=ROOT/'outputs/cctv_dgp_objective_return_v4/local_independent_audit.json'
DEST=ROOT/'outputs/cctv_dgp_conflict_vm_v5'
ARCHIVE=ROOT/'outputs/cctv-dgp-conflict-v5.tar.gz'


def main():
    if DEST.exists() or ARCHIVE.exists() or Path(str(ARCHIVE)+'.sha256').exists():
        raise ValueError('Preserve existing V5 preparation/transfer files')
    parent, diagnostic=verify_recipe(PARENT,DIAGNOSTIC,PERCEPTUAL,RETURN,V3_RETURN)
    if sha(V4_RETURN/'results.json')!=V4_RESULTS_SHA or sha(CAPSULE)!=V4_AUDIT_SHA:
        raise ValueError('Audited V4 return/receipt differs')
    from audit_cctv_dgp_objective_diagnostic_v4 import audit
    if read(CAPSULE)!=audit(PARENT,DIAGNOSTIC,V4_RETURN,PERCEPTUAL,RETURN,V3_RETURN):
        raise ValueError('V4 independent audit does not reproduce')
    for name,expected in derive_sources().items():
        if (ROOT/name).read_text(encoding='utf-8')!=expected:
            raise ValueError('Derived V5 executable differs: '+name)
    protocol=read(PERCEPTUAL/'perceptual_protocol_v3.json')
    protected=(set(parent['assets_sha256']) | set(protocol['new_assets_sha256'])
               | set(protocol['existing_runtime_assets_sha256']) | set(diagnostic['assets_sha256'])
               | {'protocol.json','protocol.sha256','normfix_v2.json',
                  'perceptual_protocol_v3.json','perceptual_protocol_v3.sha256',
                  'objective_diagnostic_protocol_v4.json','objective_diagnostic_protocol_v4.sha256'})
    names=REQUIRED_ASSETS | {'conflict_protocol_v5.json','conflict_protocol_v5.sha256'}
    if names & protected:
        raise ValueError('V5 would replace historical assets')
    for name in REQUIRED_ASSETS-{'v4_local_audit_for_training.json'}:
        if name.endswith('.py'):
            ast.parse((ROOT/name).read_text(encoding='utf-8'),feature_version=(3,10))
        if name.endswith('.sh') and b'\r' in (ROOT/name).read_bytes():
            raise ValueError('VM launcher requires LF')
    DEST.mkdir()
    for name in sorted(REQUIRED_ASSETS-{'v4_local_audit_for_training.json'}):
        target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    shutil.copyfile(CAPSULE,DEST/'v4_local_audit_for_training.json')
    pins={p.relative_to(DEST).as_posix():sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    import copy
    fixed=copy.deepcopy(parent);fixed['arms']=ARMS
    write(DEST/'conflict_protocol_v5.json',{
        'format':'cctv-two-objective-conflict-pilot-v5','date':'2026-10-04',
        'diagnostic_protocol_sha256':sha(DIAGNOSTIC/'objective_diagnostic_protocol_v4.json'),
        'diagnostic_results_sha256':V4_RESULTS_SHA,'diagnostic_local_audit_sha256':V4_AUDIT_SHA,
        'starting_checkpoint_sha256':START_SHA,'starting_state_hash':START_STATE,
        'assets_sha256':pins,'protocol':fixed,'runtime_cap_seconds':1800,
        'expected_training_autograd_grad_calls':904,'expected_preflight_autograd_grad_calls':8,
        'native_cases_used':0,'native_reserved_used':False,'production_promotion_permitted':False,
        'purpose':'Compare identity weight 0.4 against symmetric two-objective PCGrad at identity 0.1; same data/optimizer/guards.',
        'vm_training_pending':True,
    })
    with (DEST/'conflict_protocol_v5.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(sha(DEST/'conflict_protocol_v5.json')+'\n')
    configure_paths(DEST,RETURN,PERCEPTUAL,DIAGNOSTIC,V3_RETURN,V4_RETURN)
    verify_protocol_v5(PARENT)
    with tarfile.open(ARCHIVE,'x:gz',compresslevel=5) as archive:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                info=tarfile.TarInfo('cctv_dgp_vm_bundle/'+path.relative_to(DEST).as_posix())
                info.size,info.mtime,info.mode=path.stat().st_size,0,0o644
                with path.open('rb') as stream:
                    archive.addfile(info,stream)
    with Path(str(ARCHIVE)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(sha(ARCHIVE)+'  '+ARCHIVE.name+'\n')
    print({'complete':True,'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha(ARCHIVE),
           'protocol_sha256':sha(DEST/'conflict_protocol_v5.json'),'new_files':len(names),
           'local_model_forwards':0,'local_backward_calls':0,'local_optimizer_updates':0,
           'vm_training_pending':True})


if __name__=='__main__':
    main()
