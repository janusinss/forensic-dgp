"""Magnitude analysis of recorded optimizer steps, with no new fitting."""
import hashlib
import json
import math
import statistics
from pathlib import Path


def summarize(rows):
    values=[r['replay_first_order_loss_change'] for r in rows]
    active=[r for r in rows if r['projection_applied']]
    fractions=[r['projection']['removed_real_norm']/r['projection']['real_norm'] for r in active if r['projection']['real_norm']>0]
    positive=sum(max(0,v) for v in values)
    negative=-sum(min(0,v) for v in values)
    return {'steps':len(rows),'ascent_steps':sum(v>0 for v in values),
        'positive_linearized_sum':positive,'negative_linearized_magnitude_sum':negative,
        'positive_fraction_of_absolute_linearized_sum':positive/(positive+negative) if positive+negative else None,
        'median_absolute_linearized_change':statistics.median(map(abs,values)),
        'max_positive_linearized_change':max(0,max(values)),
        'median_parameter_step_norm':statistics.median(r['parameter_step_norm'] for r in rows),
        'projection_steps':len(active),
        'median_removed_real_fraction_when_projected':statistics.median(fractions) if fractions else None,
        'max_removed_real_fraction_when_projected':max(fractions) if fractions else None,
        'positive_linearized_sum_on_projected_steps':sum(max(0,r['replay_first_order_loss_change']) for r in active)}


def main():
    root=Path('outputs/downloaded_projection/outputs/projection_training_vm')
    result={'scope':'Descriptive aggregates of changing minibatch/state directional derivatives, not cumulative replay loss',
            'optimizer_updates':0,'arms':{}}
    for arm in ('ordinary','projected'):
        path=root/arm/'steps.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==210
        for r in rows:
            assert all(math.isfinite(r[k]) for k in ('replay_first_order_loss_change','parameter_step_norm','pre_clip_norm'))
            p=r['projection']
            assert 0<=p['removed_real_norm']<=p['real_norm']*(1+1e-10)
            assert r['projection_applied']==(arm=='projected' and p['applied'])
        result['arms'][arm]={'steps_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'all':summarize(rows),'epochs':{str(e):summarize([r for r in rows if r['epoch']==e]) for e in range(1,11)}}
    out=Path('outputs/projection_validation/step_magnitudes.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:r['all'] for a,r in result['arms'].items()},indent=2))


if __name__=='__main__':main()
