"""Independent saved-gradient arithmetic and original-state readback; no autograd."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from import_cctv_dgp_v26_gradient_diagnostic_v1 import read,write,sha,require,PIN
from verify_cctv_dgp_v26_gradient_diagnostic_v1 import source_contract,expected_layout

RETURN=ROOT/'outputs/cctv_dgp_v26_gradient_diagnostic_v1_return'
BUNDLE=ROOT/'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm'
OUT=ROOT/'outputs/cctv_dgp_v26_gradient_diagnostic_v1_independent_audit.json'


def numeric(actual,expected,tolerance,path=''):
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'Keys differ: '+path)
        for k,v in expected.items():numeric(actual[k],v,tolerance,path+'/'+str(k))
    elif isinstance(expected,list):
        require(len(actual)==len(expected),'Length differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):numeric(a,b,tolerance,path+'/'+str(i))
    else:require(abs(actual-expected)<=tolerance,'Numeric receipt differs: '+path)


def matrix_statistics(array,layout):
    import numpy as np
    require(array.dtype==np.float64 and array.shape==(7,53781) and np.isfinite(array).all(),'Finite float64 gradient layout')
    require(layout==expected_layout(),'Original26 parameter names/shapes/offsets')
    norms=np.sqrt(np.einsum('ij,ij->i',array,array));gram=np.einsum('ik,jk->ij',array,array)
    cos=[]
    for i in range(7):cos.append([float(gram[i,j]/(norms[i]*norms[j])) if norms[i]*norms[j] else 0. for j in range(7)])
    partitions={r['name']:{'component_norms':np.linalg.norm(array[:,r['start']:r['end']],axis=1).tolist(),
        'total_norm':float(np.linalg.norm(array[:,r['start']:r['end']].sum(0)))} for r in layout}
    return {'component_norms':norms.tolist(),'component_gram':gram.tolist(),'component_cosines':cos,
        'total_gradient_norm':float(np.linalg.norm(array.sum(0))),'per_parameter_gradients':partitions}


def main():
    import numpy as np
    started=time.monotonic();require(not OUT.exists(),'Preserve original gradient audit')
    imported=read(RETURN.with_name(RETURN.name+'_import.json'))
    for name,digest in imported['files_sha256'].items():require(sha(RETURN/name)==digest,'Returned evidence changed: '+name)
    require(sha(RETURN/'protocol.json')==PIN==sha(BUNDLE/'protocol.json'),'Frozen protocol changed')
    p=read(BUNDLE/'protocol.json')
    for name,digest in p['assets_sha256'].items():require(sha(RETURN/name)==digest==sha(BUNDLE/name),'Frozen worker source: '+name)
    source_contract((RETURN/'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py').read_text())
    parent=ROOT/'outputs/cctv_dgp_batchmatched_identity_vm_v26';old_return=ROOT/'outputs/cctv_dgp_batchmatched_identity_v26_return'
    require(sha(parent/'protocol.json')==p['closed_V26_protocol_sha256'],'Original V26 protocol changed')
    original=read(parent/'protocol.json')
    for name,digest in original['assets_sha256'].items():require(sha(parent/name)==digest,'Original V26 asset: '+name)
    for name,digest in p['closed_V26_inputs_sha256'].items():require(sha(old_return/name)==digest,'Original V26 saved input: '+name)
    cache=read(old_return/'outputs/frozen_DGP_features.json')
    for name,digest in cache['files_sha256'].items():require(sha(old_return/'outputs/frozen_DGP_features'/name)==digest,'Frozen feature changed: '+name)
    require(len(cache['files_sha256'])==250 and read(old_return/'outputs/early_structure_stop.json')['pass'] is False,'Retain failed V26')
    r=read(RETURN/'outputs/results.json');require(r['complete'] and r['protocol_sha256']==PIN,'Completed diagnostic result')
    require(r['terms']==p['terms'] and [s['update'] for s in r['snapshots']]==[0,50],'Fixed terms/states')
    for k,value in {'head_batches':20,'component_gradient_calls':140,'recognizer_forwards':70,'optimizer_updates':0,'DGP_forwards':0}.items():
        require(type(r[k]) is int and r[k]==value,'Diagnostic counter: '+k)
    require(r['original_evidence_unchanged'] and not r['new_checkpoint_created'] and not r['automatic_follow_on'] and not r['app_promotion'] and not r['goal_complete'],'Frozen-state/no-promotion scope')
    require(r['frozen_recognizer_state']==p['frozen_recognizer_state'],'Frozen recognizer state')
    require(0<=r['seconds']<420 and 0<r['peak_allocated_VRAM_bytes']<=20*1024**3,'Worker timing/VRAM cap')
    supervisor=read(RETURN/'supervisor_receipt.json')
    require(supervisor['complete'] and supervisor['protocol_sha256']==PIN and supervisor['cap_seconds']==480 and supervisor['kill_grace_seconds']==30,'Supervisor bounds')
    require(supervisor['within_external_bound'] and r['seconds']<=supervisor['seconds']<=510 and supervisor['trainer_exit_code']==0 and supervisor['optimizer_updates']==0,'Supervisor timing/exit/update receipt')
    require((RETURN/'trainer_exit_code.txt').read_text().strip()=='0' and not (RETURN/'outputs/failure.json').exists(),'Successful measurement only')
    logs=[json.loads(line) for line in (RETURN/'trainer.log').read_text().splitlines() if line.strip()]
    require([v['snapshot'] for v in logs]==[0,50] and 0<logs[0]['seconds']<logs[1]['seconds']<=r['seconds'],'Finite diagnostic log')
    loss=read(ROOT/'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1/results.json')
    arrays=[];rows=[];maximum_scalar_difference=0.
    for snapshot,log in zip(r['snapshots'],logs):
        update=snapshot['update'];require(snapshot['head_state_before_after']==p['head_states'][str(update)],'Saved head state unchanged')
        file=RETURN/'outputs'/('gradient_components_update'+str(update)+'.npy')
        require(sha(file)==snapshot['gradient_array_sha256'],'Saved gradient file hash')
        matrix=np.load(file,allow_pickle=False);arrays.append(matrix)
        stats=matrix_statistics(matrix,r['parameter_layout'])
        numeric({k:snapshot[k] for k in stats},stats,1e-12,'matrix_statistics')
        numeric(log['component_norms'],snapshot['component_norms'],1e-15,'log gradient norms')
        numeric(log['objective'],snapshot['objective'],1e-15,'log objective')
        require(len(snapshot['batches'])==10 and snapshot['maximum_saved_raw_difference']<=2e-6,'Original head raw replay bound')
        for index,b in enumerate(snapshot['batches']):
            require(b['ids']==[c['id'] for c in original['cases'][index*5:index*5+5]],'Original batch cases/order')
            require(len(b['cohort_weighted_component_values'])==len(b['cohort_weighted_component_gradient_norms'])==7,'Seven finite batch terms')
            require(all(np.isfinite(v) and v>=0 for v in b['cohort_weighted_component_values']+b['cohort_weighted_component_gradient_norms']),'Finite/nonnegative component values and norms')
        sums=np.sum([b['cohort_weighted_component_values'] for b in snapshot['batches']],0)
        numeric(snapshot['component_values'],sums.tolist(),1e-14,'cohort means')
        numeric(snapshot['objective'],float(sums.sum()),1e-14,'objective term sum')
        sum_norms=np.sum([b['cohort_weighted_component_gradient_norms'] for b in snapshot['batches']],0)
        require(np.all(np.asarray(stats['component_norms'])<=sum_norms+1e-12),'Cohort gradient triangle inequality')
        differences={}
        for term,value in zip(p['terms'],snapshot['component_values']):
            difference=abs(value-loss['groups']['all']['states'][str(update)]['terms'][term]);differences[term]=difference
            limit=p['prospective_return_checks']['CPU_saved_loss_scalar_tolerances']['ArcFace_regression' if term=='ArcFace_regression' else 'nonidentity_terms']
            require(difference<=limit,'Prospective fixed CPU/CUDA scalar bound: '+term);maximum_scalar_difference=max(maximum_scalar_difference,difference)
        reward=matrix[:3].sum(0);penalty=matrix[3:].sum(0)
        cosine=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b) else None
        rows.append({'update':update,'objective':snapshot['objective'],'component_values':snapshot['component_values'],
            'component_norms':stats['component_norms'],'CPU_saved_loss_component_differences':differences,
            'degraded_reward_gradient_norm':float(np.linalg.norm(reward)),'preservation_penalty_gradient_norm':float(np.linalg.norm(penalty)),
            'reward_penalty_cosine':cosine(reward,penalty),'total_gradient_norm':stats['total_gradient_norm'],
            'landmark_ArcFace_cosine':cosine(matrix[0],matrix[6]),'landmark_SSIM_cosine':cosine(matrix[0],matrix[5]),
            'ArcFace_to_landmark_gradient_norm_ratio':float(np.linalg.norm(matrix[6])/np.linalg.norm(matrix[0])),
            'batch_sum_gradient_norms':sum_norms.tolist(),'per_parameter_gradient_statistics':stats['per_parameter_gradients']})
    exact=0
    for c in original['cases']:
        baseline=np.load(parent/c['raw_dgp'],allow_pickle=False)
        saved=np.load(old_return/'outputs/update0'/(c['id']+'.npy'),allow_pickle=False)
        require(np.array_equal(baseline,saved),'Initial raw output differs: '+c['id']);exact+=1
    # Corrected reference must stay exactly zero in every initial batch and array.
    require(rows[0]['component_values'][6]==0 and rows[0]['component_norms'][6]==0 and exact==50,'Initial corrected identity proof differs')
    require(np.count_nonzero(arrays[0][6])==0 and all(b['cohort_weighted_component_values'][6]==0 and b['cohort_weighted_component_gradient_norms'][6]==0 for b in r['snapshots'][0]['batches']),'Initial corrected identity batch/gradient must be exactly zero')
    bindings={file.relative_to(ROOT).as_posix():sha(file) for file in [
        RETURN.with_name(RETURN.name+'_import.json'),RETURN/'outputs/results.json',RETURN/'supervisor_receipt.json',
        ROOT/'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1/results.json',ROOT/'scripts/cctv_dgp_v26_gradient_diagnostic_v1_vm.py',
        ROOT/'scripts/verify_cctv_dgp_v26_gradient_diagnostic_v1.py',ROOT/'scripts/import_cctv_dgp_v26_gradient_diagnostic_v1.py',Path(__file__)]}
    result={'complete':True,'date':'2026-10-06','scope':'Independent matrix/statistic/source/state/count/timing readback; no local derivative or optimizer trajectory reconstruction',
        'protocol_sha256':PIN,'archive_sha256':imported['archive_sha256'],'archive_bytes':imported['archive_bytes'],
        'returned_files_verified':len(imported['files_sha256']),'original_assets_verified':len(original['assets_sha256']),
        'original_saved_inputs_verified':len(p['closed_V26_inputs_sha256']),'frozen_feature_arrays_verified':250,
        'gradient_arrays_verified':2,'component_rows_verified':14,'parameter_groups_per_state':26,'fixed_state_gradient_values_verified':2*7*53781,
        'snapshots':rows,'initial_raw_equals_cached_baseline_cases':exact,'maximum_CPU_saved_loss_scalar_difference':maximum_scalar_difference,
        'diagnostic_worker_seconds':r['seconds'],'diagnostic_supervisor_seconds':supervisor['seconds'],
        'VM_gradient_calls_verified':140,'VM_optimizer_updates_verified':0,'corrected_initial_identity_exact_zero_verified':True,
        'gradient_trajectory_causality_proven':False,'original_V26_quality_failure_retained':True,
        'source_bindings_sha256':bindings,'local_neural_calls':0,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'VM_actions':False,'new_training_recipe':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(OUT,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['snapshots','source_bindings_sha256']},indent=2))
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='per_parameter_gradient_statistics'},indent=2))


if __name__=='__main__':main()
