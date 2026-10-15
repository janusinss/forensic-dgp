"""Extra saved-evidence attribution and gallery binding; never creates a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch  # read-only serialized states, no model construction or assignment

from cctv_dgp_spatial_readout_v1_contract import read, write, sha

ROOT = Path(__file__).resolve().parent.parent
RETURN = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_readout_v1_analysis'
OUT = RETURN/'outputs/spatial_readout_v1'


def image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def pixels_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def norm(value):
    return float(np.linalg.norm(value.astype(np.float64)))


def main():
    start = time.monotonic()
    imported = read(ANALYSIS/'import.json')
    audited = read(ANALYSIS/'independent_audit.json')
    assert imported['complete'] and audited['complete']
    assert audited['paired_RAW_PNG_compositions_checked'] == 500
    assert len(audited['sheets']) == 20 and not audited['model_qualified']
    p, old = read(RETURN/'protocol.json'), read(PARENT/'protocol.json')
    assert sha(RETURN/'protocol.json') == imported['protocol_sha256']
    refs, cases = {r['id']: r for r in old['references']}, {r['id']: r for r in old['cases']}
    bounds = read(OUT/'distance_bound.json')
    tables = {label: read(OUT/label/'metrics.json') for label in ['baseline', 'A50', 'B50']}
    analysis = {}
    for route in ['A', 'B']:
        initial = np.load(OUT/(route+'_initial_vector.npy'), allow_pickle=False)
        trace = read(OUT/('fit_'+route+'.json'))['rows']
        layout = read(OUT/(route+'_parameter_layout.json'))
        first = torch.load(OUT/('state_'+route+'0/full_state.pth'), map_location='cpu', weights_only=True)
        last = torch.load(OUT/('state_'+route+'50/full_state.pth'), map_location='cpu', weights_only=True)
        changed = [name for name in first['model'] if not torch.equal(first['model'][name], last['model'][name])]
        assert set(changed) <= set(p['trainable_parameter_names'][route])
        partition = []
        for row in layout:
            name = row['name']
            a, b = first['model'][name].numpy(), last['model'][name].numpy()
            delta = b.astype(np.float64)-a.astype(np.float64)
            v = dict(name=name, elements=int(a.size), changed_elements=int(np.count_nonzero(delta)),
                     delta_L2=norm(delta), squared_distance=float(np.sum(delta*delta)))
            if name.startswith('projectors.') and name.endswith('.weight'):
                neighbor = delta.copy()
                neighbor[:, :, 1, 1] = 0
                v.update(center_delta_L2=norm(delta[:, :, 1, 1]), neighbor_delta_L2=norm(neighbor),
                         initially_zero_neighbors_changed=bool(np.count_nonzero(neighbor)))
            partition.append(v)
        previous, steps = initial.astype(np.float64), []
        for row in trace:
            with np.load(OUT/('trajectory_'+route)/f"update_{row['update']:03d}.npz", allow_pickle=False) as pack:
                actual = pack['actual'].astype(np.float64)
            d = actual-initial.astype(np.float64)
            prior_d = previous-initial.astype(np.float64)
            cosine = float(d@prior_d/(norm(d)*norm(prior_d))) if norm(prior_d) else None
            steps.append(dict(update=row['update'], step_L2=norm(actual-previous),
                              original_distance_L2=norm(d), previous_direction_cosine=cosine,
                              projection_scale=row['projection']['scale']))
            previous = actual
        repeated = []
        for i in range(20):
            first_pass, second_pass = trace[i], trace[i+20]
            assert first_pass['reference'] == second_pass['reference']
            r = dict(reference=first_pass['reference'], first_pre_step_objective=first_pass['objective'],
                     second_pre_step_objective=second_pass['objective'],
                     second_minus_first=second_pass['objective']-first_pass['objective'])
            if i < 10:
                third = trace[i+40]
                assert third['reference'] == first_pass['reference']
                r.update(third_pre_step_objective=third['objective'],
                         third_minus_second=third['objective']-second_pass['objective'])
            repeated.append(r)
        base_rows = {r['case']: r for r in tables['baseline']['rows']}
        after_rows = {r['case']: r for r in tables[route+'50']['rows']}
        per_case = []
        for cid in p['case_ids']:
            before, after = base_rows[cid], after_rows[cid]
            r = dict(case=cid, source=before['source'], profile=before['profile'])
            for stage in ['raw', 'png']:
                b, a = before[stage], after[stage]
                r[stage] = dict(structure_gain_percent=100*(1-a['landmark_high_frequency_MSE']/b['landmark_high_frequency_MSE']),
                    MSE_delta=a['MSE']-b['MSE'], SSIM_delta=a['SSIM']-b['SSIM'],
                    ArcFace_delta=a['ArcFace_observed_fixed']-b['ArcFace_observed_fixed'])
            per_case.append(r)
        groups = {}
        for stage in ['raw', 'png']:
            groups[stage] = {}
            for name, b in tables['baseline']['groups'][stage].items():
                a = tables[route+'50']['groups'][stage][name]
                groups[stage][name] = dict(cases=b['cases'], MSE_delta=a['MSE']-b['MSE'],
                    MSE_relative_percent=100*(a['MSE']/b['MSE']-1), SSIM_delta=a['SSIM']-b['SSIM'],
                    ArcFace_delta=a['ArcFace_observed_fixed']-b['ArcFace_observed_fixed'],
                    structure_gain_percent=100*(1-a['landmark_high_frequency_MSE']/b['landmark_high_frequency_MSE']))
        final_squared = sum(v['squared_distance'] for v in partition)
        assert abs(final_squared-norm(previous-initial.astype(np.float64))**2) < 1e-12
        classes = defaultdict(float)
        for v in partition:
            classes['spatial_connections' if v['name'].startswith('projectors.') else 'original_decoder'] += v['squared_distance']
        for key in list(classes):
            classes[key] = dict(squared_distance=classes[key], fraction_of_final_squared_distance=classes[key]/final_squared)
        preflight = read(OUT/('preflight_'+route+'.json'))
        analysis[route] = dict(selected_tensors=len(layout), changed_tensors=len(changed), tensor_changes=partition,
            displacement_partition=dict(classes), projection_applied_updates=sum(r['projection']['scale'] < 1 for r in trace),
            projection_scale_range=[min(r['projection']['scale'] for r in trace), max(r['projection']['scale'] for r in trace)],
            final_original_distance_L2=norm(previous-initial.astype(np.float64)),
            final_distance_fraction_of_common_radius=norm(previous-initial.astype(np.float64))/bounds['common_radius'],
            steps=steps, repeated_reference_pre_step_losses=repeated,
            repeat_comparison=dict(second_pass_lower_objective_references=sum(r['second_minus_first'] < 0 for r in repeated),
                second_pass_higher_objective_references=sum(r['second_minus_first'] > 0 for r in repeated),
                third_pass_lower_objective_references=sum(r.get('third_minus_second', 0) < 0 for r in repeated),
                third_pass_higher_objective_references=sum(r.get('third_minus_second', 0) > 0 for r in repeated),
                causal_effect_of_batch_size_not_established=True),
            all_selected_tensors_nonzero_in_VM_preflight=preflight['all_nonzero'],
            independently_recomputed_CUDA_gradients=False, group_deltas=groups, per_case_deltas=per_case)
        del first, last
    cells = []
    for block, sheet in zip(p['reference_blocks'], audited['sheets']):
        path = Path(sheet['path'])
        assert path.name == block['reference']+'.png' and sha(path) == sheet['sha256']
        full = image(path)
        assert full.shape == (1320, 1280, 3)
        for y, cid in enumerate(block['case_ids']):
            ref = refs[block['reference']]
            paths = [PARENT/cases[cid]['input'], PARENT/ref['target']]+[
                OUT/label/'png'/(cid+'.png') for label in ['baseline', 'A50', 'B50']]
            for x, src in enumerate(paths):
                tile = full[40+256*y:40+256*(y+1), x*256:(x+1)*256]
                exact = image(src)
                assert tile.shape == exact.shape == (256, 256, 3) and np.array_equal(tile, exact)
                cells.append(dict(sheet=path.name, case=cid, column=x, source=str(src), source_sha256=sha(src),
                                  cell_pixels_sha256=pixels_sha(tile), no_display_resizing=True))
    assert len(cells) == 500
    write(ANALYSIS/'gallery_binding_audit.json', dict(complete=True, sheets=20, cells=500, cells_bound=cells,
        exact_saved_pixels=True, neural_calls=0, visual_observations_proved_by_this_check=False))
    protected = read(ANALYSIS/'protected_before.json')['files']
    assert all(sha(ROOT/name) == digest for name, digest in protected.items())
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), routes=analysis,
        source_roles={'dataset/thumbnails128x128': 'audited high-resolution FFHQ counterparts; legacy source key',
                      'dataset/asian_faces': 'Asian-source photographic replay; not ethnicity or native CCTV'},
        references_per_source=10, paired_exposed_TRAIN_references=20, paired_cases=100,
        common_radius_L2=bounds['common_radius'], current_protocol_sha256=sha(RETURN/'protocol.json'),
        original_checkpoint_and_documents_unchanged=True, all20_gallery500_cells_exact=True,
        unexposed_DEV_or_final_evaluation=False, native_CCTV=False,
        no_unique_cause_or_architecture_capacity_limit_proved=True,
        neural_calls=0, gradient_queries=0, optimizer_updates=0,
        seconds=time.monotonic()-start)
    write(ANALYSIS/'learning_attribution.json', record)
    print(json.dumps({'complete': True, 'routes': {k: {n: v[n] for n in [
        'changed_tensors', 'projection_applied_updates', 'projection_scale_range',
        'displacement_partition', 'repeat_comparison']} for k, v in analysis.items()}, 'gallery_cells': len(cells)}))


if __name__ == '__main__':
    main()
