"""Predeclare paired TRAIN diagnostic summaries and exact visual rows."""
from pathlib import Path
import time
from cctv_dgp_actual_step_review_v1_contract import NAME,PROFILES,read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_analysis'


def main():
    start=time.monotonic();assert not OUT.exists();p=read(ROOT/'outputs'/NAME/'protocol.json')
    assert not (ROOT/'outputs/cctv_dgp_actual_step_review_v1_independent_audit.json').exists()
    cases={c['id']:c for c in p['cases']};selected={}
    for role,ids in p['cohorts'].items():
        sources=sorted({cases[cid]['source'] for cid in ids});assert len(sources)==2
        chosen=[]
        for source in sources:
            reference=min(cases[cid]['reference_id'] for cid in ids if cases[cid]['source']==source)
            group=[cid for cid in ids if cases[cid]['reference_id']==reference]
            group.sort(key=lambda cid:PROFILES.index(cases[cid]['profile']))
            assert len(group)==5 and [cases[cid]['profile'] for cid in group]==PROFILES;chosen.extend(group)
        selected[role]=chosen
    rows=[]
    for probe in p['probes']:
        for role,ids in list(selected.items())+[('current_batch',probe['case_ids'])]:
            for cid in ids:rows.append({'update':probe['update'],'role':role,'id':cid,'source':cases[cid]['source'],'profile':cases[cid]['profile']})
    assert len(rows)==250 and len({(r['update'],r['role'],r['id']) for r in rows})==250
    OUT.mkdir()
    write(OUT/'prospective_review.json',{'complete':True,'protocol_sha256':sha(ROOT/'outputs'/NAME/'protocol.json'),
       'checker_sha256':sha(Path(__file__)),'selected_fixed_cohort_ids':selected,'visual_rows':rows,'rows_per_sheet':5,'sheets':50,
       'columns':['input','paired_target','original_DGP','same_before_state','recorded_actual_step','cone_proposal'],
       'exact256_pixel_cells':1500,'selection':'all ten current batches plus the smallest reference ID per source within each unchanged fixed TRAIN cohort',
       'outcome_based_case_or_probe_selection_added_here':False,'post_outcome_TRAIN_probe_selection_from_original_protocol_retained':True,
       'scientific_summary_scope':'all30 conditions and all3150 independently audited slots; raw and PNG reported separately',
       'no_new_capacity_or_quality_gate':True,'full_TRAIN_capacity_tested':False,'native_or_DEV_or_reserved_final_used':False,
       'model_forwards':0,'optimizer_updates':0,'gradient_queries':0,'VM_calls':0,'app_promotion':False,'goal_complete':False,
       'seconds':time.monotonic()-start,'cap_seconds':30})
    print({'complete':True,'visual_rows':250,'sheets':50,'exact_cells':1500,'model_forwards':0,'VM_calls':0},flush=True)


if __name__=='__main__':main()
