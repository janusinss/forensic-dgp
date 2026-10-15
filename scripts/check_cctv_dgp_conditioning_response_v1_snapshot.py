"""Replay real retained image bytes, then reject one deliberately corrupt PNG."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from audit_cctv_dgp_conditioning_response_v1_return import audit_snapshot, schema
from import_cctv_dgp_conditioning_response_v1_return import ROOT, OUT, PARENT, read, sha, write


def main():
    previous=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_return'
    folder=ROOT/'scratch/conditioning_response_snapshot_fixture_v1_r1'
    assert not folder.exists();folder.mkdir(parents=True)
    source=previous/'outputs/raw_conditioning_learning_v1_r1'
    shutil.copytree(source/'initial',folder/'initial')
    shutil.copyfile(source/'initial_conditioning.npy',folder/'initial_conditioning.npy')
    snapshot=read(folder/'initial/snapshot.json');snapshot['exact_controls']=True
    p=read(previous/'protocol.json');old=read(PARENT/'protocol.json')
    for b in p['reference_blocks']:
        stem=f"b{b['begin']//5:03d}"
        with np.load(folder/'initial'/(stem+'_embeddings.npz'),allow_pickle=False) as archive:
            vectors={k:archive[k] for k in archive.files}
        for i,cid in enumerate(b['case_ids']):
            row=next(r for r in snapshot['rows'] if r['id']==cid)
            row['fresh_recognizer_scores']={s:float(vectors[s][i]@vectors[s][10+i]) for s in ['raw','png']}
    (folder/'initial/snapshot.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    virtual=schema(previous);names=p['conditioning_names'];edges=np.cumsum([0]+[int(np.prod(p['conditioning_shapes'][n])) for n in names])
    good,groups,cells=audit_snapshot(folder/'initial',p,old,virtual,names,edges)
    assert len(cells)==100 and set(groups)==set(snapshot['groups'])
    differences=[]
    for stage,values in groups.items():
        for key,row in values.items():
            for metric,value in row.items():
                difference=abs(value-snapshot['groups'][stage][key][metric])
                assert difference<=1e-13
                if difference:differences.append(dict(stage=stage,group=key,metric=metric,absolute_difference=difference))
    cid=p['case_ids'][0];path=folder/'initial/png'/(cid+'.png');retained=source/'initial/png'/(cid+'.png')
    original_hash=sha(path);a=np.asarray(Image.open(path).convert('RGB')).copy();a[0,0,0]^=1;Image.fromarray(a).save(path)
    rejected=False
    try:audit_snapshot(folder/'initial',p,old,virtual,names,edges)
    except AssertionError:rejected=True
    finally:shutil.copyfile(retained,path)
    assert rejected and sha(path)==original_hash
    protected=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(OUT/'cctv_dgp_conditioning_response_v1_snapshot_fixture.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        scope='Real retained R1 initial bytes copied into a declared extended-receipt fixture; not a new VM run',
        source_protocol_sha256=sha(previous/'protocol.json'),RAW_cases_checked=100,PNG_cases_checked=100,
        altered_PNG_rejected=True,fixture_PNG_restored_exactly=True,fresh_recognizer_fields_from_saved_vectors_only=True,
        original_metric_replay_bound_unchanged=1e-13,actual_replayed_group_differences=differences,
        protected_app_and_checkpoint_unchanged=True,local_neural_gradient_optimizer_or_weight_proposal_calls=0,
        model_qualified=False,goal_complete=False))
    print(dict(complete=True,RAW_cases=100,PNG_cases=100,corrupt_PNG_rejected=True,neural_or_training_calls=0))


if __name__=='__main__':main()
