"""Restore the retained float32 identity scoring before manual publication."""
from pathlib import Path
from datetime import datetime,timezone
import sys
import numpy as np
from import_cctv_dgp_live_conditioning_v1_r1_return import ROOT,OUT,PARENT,read,sha,write


def save(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


def main():
    packet=OUT/'cctv_dgp_conditioning_learning_v1_r1_vm';p=read(packet/'protocol.json')
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    differences=[]
    for block in p['reference_blocks']:
        _,emb=unpack(PARENT/f"outputs/bank_comparison_v1/baseline/packs/b{block['begin']//5:03d}.npz",block['case_ids'])
        for i in range(5):
            for j in [0,1]:
                original=float(emb[i,j]@emb[i,2])
                draft=float(emb[i,j].astype(np.float64)@emb[i,2].astype(np.float64))
                differences.append(abs(original-draft))
    assert len(differences)==200 and max(differences)>0
    write(OUT/'cctv_dgp_conditioning_learning_v1_r1_scoring_review.json',dict(complete=True,
        UTC=datetime.now(timezone.utc).isoformat(),reviewer_sha256=sha(__file__),rows=200,
        first_draft_protocol_sha256=sha(packet/'protocol.json'),
        maximum_identity_dot_precision_difference=max(differences),
        location='evaluate: float64 identity dot; retained scientific evaluator uses float32',
        fix='Keep scientific identity scoring float32; baseline uses the retained cached embeddings, with live replay still checked',
        original_archive_retained=True,no_VM_training_launched_by_agent=True,quality_thresholds_not_changed=True))
    worker=(packet/'scripts/cctv_dgp_conditioning_learning_v1_r1_vm.py').read_text(encoding='utf-8')
    worker=worker.replace('conditioning_learning_v1_r1','conditioning_learning_v1_r2').replace('conditioning-learning-v1-r1','conditioning-learning-v1-r2')
    before="                            m['ArcFace_observed_fixed']=float(np.dot(emb[i].astype(np.float64),emb[10+i].astype(np.float64)))"
    assert worker.count(before)==1
    worker=worker.replace(before,
        "                            if exact_controls:\n"
        "                                j=0 if stage=='raw' else 1\n"
        "                                m['ArcFace_observed_fixed']=float(expected[i,j]@expected[i,2])\n"
        "                            else:\n"
        "                                m['ArcFace_observed_fixed']=float(emb[i]@emb[10+i])")
    save(ROOT/'scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py',worker)
    for source,target in [
        ('cctv_dgp_conditioning_learning_v1_r1_contract.py','cctv_dgp_conditioning_learning_v1_r2_contract.py'),
        ('prepare_cctv_dgp_conditioning_learning_v1_r1.py','prepare_cctv_dgp_conditioning_learning_v1_r2.py'),
        ('prepare_cctv_dgp_conditioning_learning_v1_r1_return_checker.py','prepare_cctv_dgp_conditioning_learning_v1_r2_return_checker.py'),
        ('audit_cctv_dgp_conditioning_learning_v1_r1_return.py','audit_cctv_dgp_conditioning_learning_v1_r2_return.py'),
        ('audit_cctv_dgp_conditioning_learning_v1_r1_preparation.py','audit_cctv_dgp_conditioning_learning_v1_r2_preparation.py')]:
        text=(ROOT/'scripts'/source).read_text(encoding='utf-8')
        text=text.replace('conditioning_learning_v1_r1','conditioning_learning_v1_r2').replace('conditioning-learning-v1-r1','conditioning-learning-v1-r2')
        if source.startswith('audit_cctv_dgp_conditioning_learning_v1_r1_return'):
            old="                row[stage]=metrics(actual,target,mask,patch,v[i].astype(np.float64),v[10+i].astype(np.float64),mean_actual,prediction8=png if stage=='png' else None)"
            assert text.count(old)==1
            text=text.replace(old,
                "                j=0 if stage=='raw' else 1\n"
                "                score_vector=expected[i,j] if folder.name=='initial' else v[i]\n"
                "                target_vector=expected[i,2] if folder.name=='initial' else v[10+i]\n"
                "                row[stage]=metrics(actual,target,mask,patch,score_vector,target_vector,mean_actual,prediction8=png if stage=='png' else None)")
        if source.endswith('_preparation.py'):
            marker="    assert \"param.requires_grad_(name in names)\" in code and 'candidate.load_state_dict' not in code\n"
            assert text.count(marker)==1
            text=text.replace(marker,marker+
                "    assert \"m['ArcFace_observed_fixed']=float(emb[i]@emb[10+i])\" in code\n"
                "    assert \"m['ArcFace_observed_fixed']=float(expected[i,j]@expected[i,2])\" in code\n"
                "    assert 'float(np.dot(emb[i].astype(np.float64)' not in code\n")
        save(ROOT/'scripts'/target,text)
    print(dict(complete=True,retained_metric_precision_restored=True,rows_checked=200,
        maximum_difference=max(differences),fresh_R2_staged=True))


if __name__=='__main__':main()
