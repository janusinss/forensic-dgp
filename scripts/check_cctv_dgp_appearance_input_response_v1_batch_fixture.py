"""Saved five-case fixture with explicitly synthetic RGB gradients; no autograd."""
import hashlib
from pathlib import Path
import numpy as np

from import_cctv_dgp_appearance_input_response_v1_return import ROOT,OUT,SOURCE,BANK,read,sha,write
from audit_cctv_dgp_appearance_input_response_v1_return import audit_batch,fixed_grid


def main():
    folder=ROOT/'scratch/appearance_input_response_fixture_v1';assert not folder.exists();folder.mkdir(parents=True)
    p=read(SOURCE/'protocol.json');old=read(BANK/'protocol.json');block=p['reference_blocks'][0]
    plan=dict(signed_labels=[r['label'] for r in p['step_plan']]);parent=SOURCE/'outputs/conditioning_response_v1'
    stem=f"b{block['begin']//5:03d}";before=np.load(parent/'initial'/(stem+'.npy'),allow_pickle=False)
    with np.load(parent/'initial'/(stem+'_embeddings.npz'),allow_pickle=False) as f:v=f['raw'].copy()
    np.savez_compressed(folder/(stem+'_embeddings.npz'),raw=v)
    # Analytical fixture coefficients are not computed from a model, loss, or neural derivative.
    g=np.zeros_like(before);g[:,80:88,96:104]=np.float32(.125)
    gradient_file=folder/(stem+'_RGB_gradients.npy');np.save(gradient_file,g,allow_pickle=False)
    refs={r['id']:r for r in old['references']};cases=old['cases'][block['begin']:block['begin']+5]
    np.save(folder/(stem+'_grid.npy'),np.stack([fixed_grid(refs[c['source_person_or_reference']]['matrix112']) for c in cases]),allow_pickle=False)
    rows={r['id']:r for r in read(parent/'initial/snapshot.json')['rows']};records=[]
    for i,c in enumerate(cases):
        value=1-float(v[i].astype(np.float64)@v[10+i].astype(np.float64));projections=[]
        for label in plan['signed_labels']:
            after=np.load(parent/label/(stem+'.npy'),allow_pickle=False)
            score=next(r for r in read(parent/label/'snapshot.json')['rows'] if r['id']==c['id'])
            projections.append(dict(label=label,saved_RAW_sha256=sha(parent/label/(stem+'.npy')),
                input_projection=float(np.sum(g[i].astype(np.float64)*(after[i].astype(np.float64)-before[i].astype(np.float64)))),
                actual_scientific_loss=1-score['raw']['ArcFace_observed_fixed']))
        records.append(dict(id=c['id'],source=c['source'],profile=c['profile'],reference=c['source_person_or_reference'],
            begin=block['begin'],case_index=i,derivative_scalar=value,fresh_double_loss=value,
            scientific_cached_loss=1-rows[c['id']]['raw']['ArcFace_observed_fixed'],scalar_alignment_bound=1e-6,
            gradient_norm=float(np.linalg.norm(g[i].astype(np.float64))),gradient_HWC_bytes_sha256=hashlib.sha256(g[i].tobytes()).hexdigest(),
            cross_case_gradient_maximum=0,projections=projections))
    data=dict(begin=block['begin'],case_ids=block['case_ids'],rows=records,initial_RAW_sha256=sha(parent/'initial'/(stem+'.npy')),
        gradients_sha256=sha(gradient_file),embeddings_sha256=sha(folder/(stem+'_embeddings.npz')),grid_sha256=sha(folder/(stem+'_grid.npy')),
        maximum_initial_embedding_difference=0.,original_forward_embedding_rtol=1e-6,original_forward_embedding_atol=1e-6)
    record_file=folder/(stem+'.json');write(record_file,data)
    checked,gradients=audit_batch(folder,plan,old,block);assert len(checked['rows'])==5 and np.array_equal(gradients,g)
    retained=record_file.read_bytes();data['rows'][0]['projections'][0]['input_projection']+=.001
    record_file.write_text(__import__('json').dumps(data,indent=2)+'\n',encoding='utf-8')
    try:audit_batch(folder,plan,old,block)
    except AssertionError:projection_rejected=True
    else:raise AssertionError('Incorrect input projection accepted')
    record_file.write_bytes(retained)
    changed=g.copy();changed[0,0,0,0]=1.;np.save(gradient_file,changed,allow_pickle=False)
    try:audit_batch(folder,plan,old,block)
    except AssertionError:gradient_rejected=True
    else:raise AssertionError('Corrupted gradient bytes accepted')
    np.save(gradient_file,g,allow_pickle=False);assert sha(gradient_file)==data['gradients_sha256']
    audit_batch(folder,plan,old,block)
    write(OUT/'cctv_dgp_appearance_input_response_v1_batch_fixture.json',dict(complete=True,checker_sha256=sha(__file__),
        prospective_batch_auditor_sha256=sha(ROOT/'scripts/audit_cctv_dgp_appearance_input_response_v1_return.py'),
        five_saved_cases_and30_pixel_projections_checked=True,incorrect_projection_rejected=projection_rejected,
        changed_gradient_file_rejected=gradient_rejected,fixture_restored_exactly=True,
        gradients_are_synthetic_analytical_coefficients_not_neural=True,
        actual_neural_gradient_or_training_calls=0,fixture_is_not_VM_evidence=True,model_qualified=False))
    print(dict(complete=True,saved_cases=5,projections=30,tampered_projection_and_gradient_rejected=True,actual_neural_or_training_calls=0))


if __name__=='__main__':main()
