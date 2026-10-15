"""Reject false cohort counts and native-CCTV relabeling using non-neural fixtures."""
from copy import deepcopy
from pathlib import Path
import json

from publish_dgp_detector_confidence_v1_r1_results import summarize


def main():
    cases=[];rows=[]
    for stem,family,review,control,pixels in [
        ('covered','face_mask','usable',False,10),
        ('clear','clear_control','usable',True,0),
        ('excluded','hand','needs_clearer',False,200),
    ]:
        for degraded in [False,True]:
            condition='synthetic_degraded_photo' if degraded else 'original_photo'
            c=dict(id=stem+('_degraded' if degraded else '_native'),family=family,
                input_review=review,clear_control=control,condition=condition)
            cases.append(c)
            raw=4 if stem=='covered' and not degraded else 2 if stem=='clear' and not degraded else 0
            rows.append(dict(**c,raw_pixels=raw,proposal_pixels=raw*2,operator_pixels=pixels,
                scores={'operator':dict(selected_fraction=.4 if raw and pixels else 0. if pixels else None,
                                        maximum=.9 if raw and pixels else 1e-6 if pixels else None)}))
    actual=summarize(rows,cases)
    assert actual['original_photo']['eligible_covering_cases']==1
    assert actual['original_photo']['empty_proposal_cases']==0
    assert actual['synthetic_degraded_photo']['eligible_covering_cases']==1
    assert actual['synthetic_degraded_photo']['empty_proposal_cases']==1
    assert actual['matched_photograph_pairs']==3 and len(actual['clear_controls'])==2
    assert len(actual['exclusions'])==2 and not actual['native_CCTV_evidence']
    assert actual['automatic_generation_outputs']==actual['assisted_generation_outputs']==0
    assert not actual['automatic_or_assisted_family_qualified'] and not actual['model_qualified']
    mutations=[]
    broken=deepcopy(rows);broken.pop();mutations.append(('missing paired row',broken))
    broken=deepcopy(rows);broken[0]['condition']='native_CCTV';mutations.append(('native-CCTV relabel',broken))
    broken=deepcopy(rows);broken[4]['input_review']='usable';mutations.append(('exclusion promoted',broken))
    broken=deepcopy(rows);broken[2]['clear_control']=False;mutations.append(('clear control mixed into covering cohort',broken))
    broken=deepcopy(rows);broken.append(deepcopy(broken[0]));mutations.append(('duplicate image',broken))
    broken=deepcopy(rows);broken[1]['proposal_pixels']=1;mutations.append(('empty seed made nonempty by margin',broken))
    broken=deepcopy(rows);broken[1]['operator_pixels']=11;mutations.append(('pair changed assisted footprint',broken))
    rejected=[]
    for name,broken in mutations:
        try:summarize(broken,cases)
        except (AssertionError,ValueError,KeyError):rejected.append(name)
        else:raise AssertionError('Unsafe summary accepted: '+name)
    path=Path(__file__).resolve().parents[1]/'outputs/dgp_detector_confidence_summary_v1_check.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(dict(complete=True,fixtures_not_neural_inference=True,
            clear_controls_and_exclusions_not_in_eligible_denominator=True,
            all_seven_false_cohort_claims_rejected=rejected,neural_or_training_calls=0),f,indent=2)
        f.write('\n')
    print(dict(complete=True,rejected=len(rejected),neural_calls=0))


if __name__=='__main__':main()
