"""Retain and join the ancillary anonymous review; no result reads or inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/codeformer_u96_context_study_v1'
SCRATCH = ROOT/'scratch/u96_isolated_degraded_input_review_v1'


def pin(path):
    b = path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(), 'bytes':len(b),
            'sha256':hashlib.sha256(b).hexdigest()}


def write(path,data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')


def main():
    mapping = json.loads((BASE/'isolated_input_review_map_v1.json').read_bytes())
    source = SCRATCH/'independent_input_review_v1.json'
    review = json.loads(source.read_bytes())
    assert review['complete'] and review['sheets_viewed']==4
    assert review['clean_references_or_model_outputs_seen'] is False
    decisions = {r['anonymous_id']:r for r in review['decisions']}
    assert len(review['decisions'])==len(decisions)==16
    assert set(decisions)=={r['anonymous_id'] for r in mapping['cases']}
    accepted = {'readable','borderline','insufficient','out_of_scope'}
    rows=[]
    for item in mapping['cases']:
        decision = decisions[item['anonymous_id']]
        assert decision['decision'] in accepted and decision['reason']
        assert pin(ROOT/item['original_input']['path'])==item['original_input']
        assert pin(ROOT/item['anonymous_input']['path'])==item['anonymous_input']
        rows.append({**item,'input_only_decision':decision['decision'],
                     'reason':decision['reason'],
                     'earlier_root_label':'usable',
                     'earlier_root_label_was_not_strictly_input_only':True})
    retained=BASE/'isolated_reviewer_decisions_v1.json'
    write(retained,review)
    result={'complete':True, 'review_stage':'post-freeze ancillary development review',
            'criteria':mapping['criteria'], 'mapping':pin(BASE/'isolated_input_review_map_v1.json'),
            'retained_reviewer_record':pin(retained),
            'why_needed':mapping['reason'],
            'frozen_plan_changed':False, 'planned_outputs_excluded':0,
            'reviewer_exposed_to_parent_history_targets_or_model_outputs':False,
            'root_previously_saw_clear_counterparts':True,
            'counts':dict(Counter(r['input_only_decision'] for r in rows)),
            'cases':rows,
            'interpretation':'All planned diagnostic outputs remain. Report any non-readable input as an ancillary uncertainty/insufficiency stratum, not production admission. The original labels are retained as provisional assisted review.',
            'formal_human_or_reserved_FINAL_evaluation':False,
            'new_training_threshold_or_quality_gate_established':False,
            'model_or_HTTP_or_training_calls':0}
    path=BASE/'ancillary_input_only_review_v1.json'
    write(path,result)
    print(json.dumps({'complete':True,'counts':result['counts'],'record':pin(path)}))


if __name__=='__main__':main()
