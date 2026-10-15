"""Compare only saved embedding arithmetic; no encoder, gradient or new direction."""
from datetime import datetime, timezone
import numpy as np

from import_cctv_dgp_conditioning_response_v1_return import ANALYSIS, PIN, RETURN, read, sha, write
from cctv_dgp_conditioning_response_v1_contract import response_receipt


def main():
    audit = read(ANALYSIS/'independent_audit.json')
    assert audit['complete'] and audit['full_diagnostic_returned'] and audit['protocol_sha256'] == PIN
    p = read(RETURN/'protocol.json');base=RETURN/'outputs/conditioning_response_v1'
    records = [read(base/f'repeat{k:02d}'/'gradients.json')['groups'] for k in range(2)]
    initial = read(base/'initial/snapshot.json')
    stage_labels = ['initial']+[s['label'] for s in p['step_plan']]+['rollback']
    arrays = {}; bindings = {}; frozen_embeddings = []
    for label in stage_labels:
        arrays[label] = {}
        for block in p['reference_blocks']:
            stem=f"b{block['begin']//5:03d}";path=base/label/(stem+'_embeddings.npz')
            bindings[path.relative_to(RETURN).as_posix()]=sha(path)
            with np.load(path,allow_pickle=False) as saved:
                for s in ['raw','png']:
                    v=saved[s];assert v.dtype==np.float32 and v.shape==(15,512) and np.isfinite(v).all()
                    arrays[label][(block['begin'],s)] = v.copy()
    for label in stage_labels:
        for block in p['reference_blocks']:
            for s in ['raw','png']:
                before=arrays['initial'][(block['begin'],s)]
                after=arrays[label][(block['begin'],s)]
                frozen_embeddings.append(dict(label=label,begin=block['begin'],stage=s,
                    baseline_exact=bool(np.array_equal(before[5:10],after[5:10])),
                    target_exact=bool(np.array_equal(before[10:15],after[10:15])),
                    maximum_baseline_difference=float(np.max(np.abs(before[5:10]-after[5:10]))),
                    maximum_target_difference=float(np.max(np.abs(before[10:15]-after[10:15])))))
    rows=[]; central=[]
    for gindex,g in enumerate(records[0]):
        if g['metric']!='ArcFace_loss':continue
        blocks=[b for b in p['reference_blocks'] if b['source']==g['source']]
        profile=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24'].index(g['profile'])
        group_double={};group_float={}
        for label in stage_labels:
            d=[];f=[]
            for b in blocks:
                v=arrays[label][(b['begin'],'raw')]
                d.append(1-float(v[profile].astype(np.float64)@v[10+profile].astype(np.float64)))
                f.append(1-float(v[profile]@v[10+profile]))
            group_double[label]=float(np.mean(d));group_float[label]=float(np.mean(f))
        cached=1-initial['groups']['raw'][g['source']+'/'+g['profile']]['ArcFace_observed_fixed']
        gradient_values=[r[gindex]['value'] for r in records]
        rows.append(dict(key=g['key'],gradient_repeat_values=gradient_values,
                         scientific_cached_loss=cached, fresh_initial_double_loss=group_double['initial'],
                         fresh_initial_float_loss=group_float['initial'],
                         gradient_minus_fresh_double=[v-group_double['initial'] for v in gradient_values],
                         group_double_losses=group_double,group_float_losses=group_float))
        for pair in range(3):
            plus,minus=f'plus{pair:02d}',f'minus{pair:02d}'
            original=next(r for r in audit['central_response_comparisons'] if r['pair']==pair and r['key']==g['key'])
            pp=original['even_linear_prediction']+original['predicted_two_sided_difference']/2
            pm=original['even_linear_prediction']-original['predicted_two_sided_difference']/2
            spread=max(abs(gradient_values[0]-gradient_values[1]),
                       *[abs(v-group_double['initial']) for v in gradient_values])
            fresh=response_receipt(group_double['initial'],group_double[plus],group_double[minus],
                                   original['magnitude'],pp,pm,spread)
            difference=group_double[plus]-group_double[minus]
            float_difference=group_float[plus]-group_float[minus]
            central.append(dict(pair=pair,key=g['key'],canonical_float_scientific_receipt=original,
                                saved_double_arithmetic_diagnostic=fresh,
                                maximum_float_vs_double_central_difference=abs(difference-float_difference),
                                precision_change_is_reporting_only_and_does_not_change_any_scientific_gate=True))
    assert len(rows)==10 and len(central)==30
    write(ANALYSIS/'saved_embedding_precision_review.json',dict(
        complete=True,UTC=datetime.now(timezone.utc).isoformat(),analyzer_sha256=sha(__file__),
        independent_audit_sha256=sha(ANALYSIS/'independent_audit.json'),protocol_sha256=PIN,
        saved_embedding_file_bindings=bindings,frozen_embedding_checks=frozen_embeddings,
        gradient_fresh_initial_and_cached_loss_comparisons=rows,central_comparisons=central,
        all_frozen_embeddings_exact=all(r['baseline_exact'] and r['target_exact'] for r in frozen_embeddings),
        reporting_only_saved_array_arithmetic=True,quality_gates_or_forward_precision_unchanged=True,
        local_neural_calls=0,gradient_queries=0,optimizer_updates=0,
        unique_root_cause_or_correct_all_dimensional_derivative_not_proved=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False,
    ))
    print(dict(complete=True,ArcFace_groups=10,comparisons=30,
               frozen_embeddings_exact=all(r['baseline_exact'] and r['target_exact'] for r in frozen_embeddings),
               pair_diagnostic_double_disagreements=[sum(r['saved_double_arithmetic_diagnostic']['central_comparison_within_diagnostic_bound'] is False
                                                       for r in central if r['pair']==k) for k in range(3)],
               largest_float_double_central_difference=max(r['maximum_float_vs_double_central_difference'] for r in central),
               largest_gradient_fresh_initial_difference=max(abs(d) for r in rows for d in r['gradient_minus_fresh_double'])))


if __name__ == '__main__':main()
