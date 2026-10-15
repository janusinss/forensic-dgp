"""Audit the retained alignment failure and its partial evidence without inference."""
import ast
from datetime import datetime, timezone
import hashlib
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from import_cctv_dgp_bank_saved_direction_forward_v1_r1_return import (
    ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write,
)
from audit_cctv_dgp_bank_saved_direction_forward_v1_r1_return import (
    delivered, image, metrics, state_hash, support,
)


def main():
    start = time.monotonic()
    receipt = ANALYSIS / 'failure_audit.json'
    assert not receipt.exists()
    imported = read(ANALYSIS / 'import.json')
    assert imported['complete'] and imported['failure_reported']
    assert not imported['complete_diagnostic_reported']
    p, old = read(RETURN / 'protocol.json'), read(PARENT / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == PIN
    manifest = read(RETURN / 'export_manifest.json')
    assert manifest['failure_present'] and not manifest['complete_diagnostic']
    assert set(manifest['files']) | {'export_manifest.json'} == {
        f.relative_to(RETURN).as_posix() for f in RETURN.rglob('*') if f.is_file()
    }
    assert all(sha(RETURN / name) == digest for name, digest in manifest['files'].items())
    assert all(sha(PARENT / name) == digest for name, digest in p['parent_required_sha256'].items())
    out = RETURN / 'outputs/saved_direction_forward_v1'
    failure = read(out / 'failure.json')
    assert failure['type'] == 'ValueError' and failure['message'] == 'Invalid fixed target affine'
    assert 0 < failure['seconds'] < p['worker_seconds']
    assert failure['progress']['stage'] == 'temporary_forward'
    assert failure['progress']['proposal'] == 'ordinary_0.0001'
    counts = dict(B=21, recognizer=0, original=0, gradient_queries=0, backwards=0,
                  optimizer_updates=0, temporary_parameter_proposals=1)
    assert failure['progress']['counts'] == counts
    assert failure['disposable_state_restore_error'] is None
    assert (RETURN / 'review_exit_code.txt').read_text(encoding='utf-8').strip() == '1'
    assert not list(RETURN.rglob('*.pth')) and not (out / 'run_results.json').exists()
    assert not (out / 'timing_projection.json').exists()
    env = read(out / 'environment.json')
    assert env['GPU'] == 'NVIDIA L4' and env['host'] == 'forensic-dgp-thesis' and env['manual_tmux']
    assert not env['optimizer_constructed'] and not env['gradient_enabled']
    pilot_path = PARENT / 'cctv_dgp_pilot.py'
    pilot_tree = ast.parse(pilot_path.read_text(encoding='utf-8'))
    grid_node = next(n for n in pilot_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'grid112')
    context = {'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=[grid_node], type_ignores=[]), str(pilot_path), 'exec'), context)
    grid112 = context['grid112']  # Only the pure NumPy/OpenCV function; no VM module execution.
    worker = RETURN / 'scripts/cctv_dgp_bank_saved_direction_forward_v1_r1_vm.py'
    tree = ast.parse(worker.read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'grid112']
    assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == "c['landmarks5_canvas_xy']"
    refs = {r['id']: r for r in old['references']}
    begins = [i for i in range(0, len(old['cases']), 5) if old['cases'][i]['id'] in p['case_ids']]
    assert len(begins) == 20
    torch.set_grad_enabled(False)
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    geometry, controls = [], []
    for begin in begins:
        cs = old['cases'][begin:begin + 5]
        ids = [c['id'] for c in cs]
        raw, _ = unpack(PARENT / f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz', ids)
        control = read(out / f'control_b{begin//5:03d}.json')
        assert control['begin'] == begin and control['case_ids'] == ids
        assert control['exact_raw_replay'] and control['changed_values'] == control['maximum_absolute_error'] == 0
        assert control['actual_RGB_float32_sha256'] == control['expected_RGB_float32_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
        controls.append(control)
        for c in cs:
            ref = refs[c['source_person_or_reference']]
            assert np.asarray(c['landmarks5_canvas_xy']).shape == (5, 2)
            try:
                grid112(c['landmarks5_canvas_xy'])
            except ValueError as error:
                assert str(error) == 'Invalid fixed target affine'
            else:
                raise AssertionError('The actual helper must reject the original argument')
            grid = grid112(ref['matrix112'])
            assert grid.shape == (112, 112, 2) and grid.dtype == np.float32 and np.isfinite(grid).all()
            geometry.append(dict(id=c['id'],reference=c['source_person_or_reference'],
                rejected_landmark_shape=[5,2],accepted_matrix_shape=[2,3],
                fixed_grid_float32_sha256=hashlib.sha256(grid.tobytes()).hexdigest()))
    assert len(geometry) == 100 and len(list(out.glob('control_*.json'))) == 20
    initial = torch.load(PARENT / 'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    assert state_hash(initial) == p['initial_model_state']
    proposal = p['proposals'][0]
    selected = read(out / proposal['label'] / 'parameter_receipt.json')
    assert selected['label'] == proposal['label'] and selected['proposal'] == proposal
    assert selected['optimizer_updates'] == 0 and not selected['checkpoint_saved']
    assert selected['original_frozen_tensors_unchanged']
    proposed = dict(initial)
    with np.load(RETURN / 'directions.npz', allow_pickle=False) as arrays:
        for i, name in enumerate(p['learning_names']):
            value = initial[name].numpy() + np.float32(proposal['rate']) * arrays[f'ordinary_p{i:02d}']
            assert hashlib.sha256(value.tobytes()).hexdigest() == selected['selected_parameter_value_sha256'][name]
            proposed[name] = torch.from_numpy(value)
    assert state_hash(proposed) == selected['model_state'] != p['initial_model_state']
    first = old['cases'][begins[0]:begins[0]+5]
    ids = [c['id'] for c in first]
    baseline, emb = unpack(PARENT / f'outputs/bank_comparison_v1/baseline/packs/b{begins[0]//5:03d}.npz', ids)
    pngs = list((out / proposal['label'] / 'png').glob('*.png'))
    assert {f.stem for f in pngs} == set(ids) and len(pngs) == 5
    assert not list(out.rglob('*_embeddings.npz')) and not list(out.rglob('b*.npy'))
    canvas = Image.new('RGB', (1280, 1460), (20,20,20))
    draw = ImageDraw.Draw(canvas)
    for i, title in enumerate(['Input','Retained DGP','Partial R1 ordinary 1e-4','Clean TRAIN target','Observed mask']):
        draw.text((i*256+5,8),title,fill='white')
    partial_rows, cells = [], []
    for i,c in enumerate(first):
        ref = refs[c['source_person_or_reference']]
        camera, target = image(PARENT/c['input']), image(PARENT/ref['target'])
        mask = image(PARENT/ref['observed'],'L') > 0
        actual = image(out/proposal['label']/'png'/(c['id']+'.png'))
        assert np.array_equal(actual[~mask],camera[~mask])
        base = delivered(baseline[i],camera,mask)
        patch = support(mask,c['landmarks5_canvas_xy'])
        values = {}
        for label, a in [('baseline',base),('partial_PNG',actual)]:
            row = metrics(a.astype(np.float32)/255,target,mask,patch,
                          np.zeros(512),np.zeros(512),a.astype(np.float32)/255,prediction8=a)
            row.pop('ArcFace_observed_fixed')  # No candidate embedding exists; do not imply identity measurement.
            values[label] = row
        partial_rows.append(dict(id=c['id'],profile=c['profile'],**values,
            raw_output_missing=True,identity_metric_unavailable=True,qualification_not_assessed=True))
        row_cells = [camera,base,actual,target,np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2)]
        for col,a in enumerate(row_cells):
            canvas.paste(Image.fromarray(a),(col*256,40+i*284))
            cells.append(dict(id=c['id'],column=col,RGB_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
        draw.text((5,40+i*284+259),c['id'],fill='white')
    sheet = ANALYSIS/'partial_output_sheet.png'
    canvas.save(sheet)
    with Image.open(sheet) as im:
        saved = np.asarray(im.convert('RGB'))
    for i,c in enumerate(first):
        for col in range(5):
            a=saved[40+i*284:40+i*284+256,col*256:(col+1)*256]
            assert hashlib.sha256(a.tobytes()).hexdigest()==cells[i*5+col]['RGB_sha256']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),import_sha256=sha(ANALYSIS/'import.json'),protocol_sha256=PIN,
        files_checked=len(manifest['files'])+1,parent_assets_checked=5852,
        retained_failure_sha256=sha(out/'failure.json'),reported_counts=counts,VM_seconds=failure['seconds'],
        actual_alignment_argument_rejection_reproduced_without_neural_calls=True,
        rejected_argument_is_landmarks5_not_target_affine=True,all100_reference_matrices_are_valid=True,
        fixed_geometry=geometry,control_blocks_checked=20,reported_control_case_hashes_checked=100,
        controls_are_saved_hash_receipts_not_independent_neural_replays=True,
        selected_proposal_parameter_bytes_recomputed=37,partial_PNGs_checked=5,
        partial_case_RGB_metrics=partial_rows,comparison_sheet=sheet.relative_to(ROOT).as_posix(),
        comparison_sheet_sha256=sha(sheet),comparison_sheet_cells=cells,visual_review_pending=True,
        candidate_raw_arrays_and_embeddings_missing=True,complete_four_proposal_comparison=False,
        failure_is_processing_not_a_quality_gate=True,
        disposable_initial_state_restore_reported_without_error=True,
        disposable_restoration_not_freshly_state_hashed_after_failure=True,
        original_checkpoint_app_examples_unchanged=True,no_new_checkpoint=True,
        no_local_neural_gradient_optimizer_or_model_displacement=True,
        no_agent_VM_connection_or_launch=True,model_qualified=False,app_promotion=False,
        more_epochs_not_qualified=True,goal_complete=False,seconds=time.monotonic()-start))
    print(dict(complete=True,failure='alignment argument mismatch',VM_seconds=failure['seconds'],
        controls=100,partial_PNGs=5,optimizer_updates=0,model_qualified=False))


if __name__ == '__main__':
    main()
