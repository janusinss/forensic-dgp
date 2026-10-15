"""Freeze bounded replay selections from existing journals; no neural calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from cctv_dgp_spatial_readout_v1_contract import read, write, sha, THRESHOLDS

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_readout_v1_analysis'
DESIGN = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_design'
UPDATES = [0, 1, 10, 20, 30, 40, 50]


def main():
    assert read(ANALYSIS/'publication/independent_publication_audit.json')['complete']
    assert not DESIGN.exists(), 'Retain frozen design; no overwrite'
    p = read(SOURCE/'protocol.json')
    out = SOURCE/'outputs/spatial_readout_v1'
    manifest = read(SOURCE/'result_manifest.json')['files']
    radius = read(out/'distance_bound.json')['common_radius']
    sources = {'protocol.json': sha(SOURCE/'protocol.json')}
    for name, digest in p['assets_sha256'].items():
        assert sha(SOURCE/name) == digest
        sources[name] = digest
    required = ['result_manifest.json', 'outputs/spatial_readout_v1/distance_bound.json',
        'outputs/spatial_readout_v1/cache_receipt.json', 'outputs/spatial_readout_v1/environment.json',
        'outputs/spatial_readout_v1/run_results.json']
    for route in ['A', 'B']:
        required += [f'outputs/spatial_readout_v1/{route}_initial_vector.npy',
            f'outputs/spatial_readout_v1/{route}_parameter_layout.json',
            f'outputs/spatial_readout_v1/state_{route}0/full_state.pth',
            f'outputs/spatial_readout_v1/state_{route}0/receipt.json',
            f'outputs/spatial_readout_v1/state_{route}50/full_state.pth',
            f'outputs/spatial_readout_v1/state_{route}50/receipt.json',
            f'outputs/spatial_readout_v1/fit_{route}.json']
        required += [f'outputs/spatial_readout_v1/trajectory_{route}/update_{u:03d}.npz' for u in UPDATES if u]
    for label in ['baseline', 'A50', 'B50']:
        required += [f'outputs/spatial_readout_v1/{label}/packs/r{i:02d}.npz' for i in range(20)]
        required += [f'outputs/spatial_readout_v1/{label}/png/{cid}.png' for cid in p['case_ids']]
        required += [f'outputs/spatial_readout_v1/{label}/metrics.json']
    for name in required:
        digest = sha(SOURCE/name)
        if name != 'result_manifest.json':
            assert digest == manifest[name]['sha256']
        sources[name] = digest
    states = []
    for route in ['A', 'B']:
        initial = np.load(out/(route+'_initial_vector.npy'), allow_pickle=False)
        for update in UPDATES:
            path = (out/(route+'_initial_vector.npy')) if update == 0 else out/('trajectory_'+route)/f'update_{update:03d}.npz'
            if update == 0:
                value = initial.copy()
                key = None
            else:
                with np.load(path, allow_pickle=False) as saved:
                    value = saved['actual'].copy()
                key = 'actual'
            assert value.shape == (p['routes'][route]['elements'],) and value.dtype == np.float32 and np.isfinite(value).all()
            distance = float(np.linalg.norm(value.astype(np.float64)-initial.astype(np.float64)))
            assert distance <= radius
            states.append(dict(route=route, update=update, source_file=path.relative_to(SOURCE).as_posix(),
                npz_key=key, file_sha256=sha(path), vector_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                elements=value.size, original_distance_L2=distance, scientific_radius_L2=radius,
                frozen_state_from=f'outputs/spatial_readout_v1/state_{route}0/full_state.pth',
                diagnostic_replay_not_optimizer_resume=True))
    assert len(states) == 14 and len(p['case_ids']) == 100
    protected = read(ANALYSIS/'publication/publication.json')
    for name, record in protected['documents'].items():
        assert sha(ROOT/name) == record['after_sha256']
    checkpoint = ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    assert sha(checkpoint) == p['original_checkpoint_sha256']
    DESIGN.mkdir()
    budgets = dict(worker_seconds=2400, external_seconds=2430, kill_grace_seconds=30,
        export_seconds=600, export_external_seconds=630, saved_audit_seconds=600,
        minimum_free_bytes=8*1024**3, available_RAM_before_import_bytes=8*1024**3,
        disk_reserve_bytes=1024**3, output_byte_cap=3*1024**3,
        peak_VRAM_bytes=20*1024**3, peak_RSS_bytes=12*1024**3,
        maximum_optimizer_updates=0, maximum_gradient_queries=0, maximum_backwards=0,
        original_batch_calls=20, prior_feature_batch_calls=20, decoder_batch_calls=280,
        recognizer_batch_calls=600, snapshot_seconds=180, cache_seconds=240)
    plan = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        status='DESIGN_FROZEN_EXECUTION_PACKET_NOT_YET_PREPARED',
        name='cctv_dgp_spatial_path_replay_v1_vm', stem='cctv-dgp-spatial-path-replay-v1',
        source_VM_name='cctv_dgp_spatial_readout_v1_vm', source_local=str(SOURCE),
        source_protocol_sha256=sha(SOURCE/'protocol.json'), source_required_sha256=sources,
        parent_VM_name=p['parent_name'] if 'parent_name' in p else 'cctv_dgp_bank_photometric_guard_v1_vm',
        parent_protocol_sha256=p['parent_protocol_sha256'], parent_required_sha256=p['parent_required_sha256'],
        checkpoint_sha256=p['original_checkpoint_sha256'], states=states,
        reference_blocks=p['reference_blocks'], case_ids=p['case_ids'],
        scientific_thresholds=THRESHOLDS, original_distance_maximum_fraction=.001,
        common_radius_L2=radius, paired_TRAIN_cases_per_snapshot=100,
        snapshot_compositions=1400, additional_fresh_original_controls=100,
        full_corpus_epochs=0, new_optimizer_updates=0, gradients=0,
        passed_train_GATE_not_incumbent_selection=True, retrospective_exposed_TRAIN_only=True,
        source_roles='HQ FFHQ counterparts and Asian-source photographic replay reported separately; no ethnicity inference',
        final_identities_or_native_CCTV_or_completion_inputs=0,
        visualization=dict(planned_sheets=40, sheets_per_route=20, size=[2304,1320],
            columns=['Input','TRAIN target','Original DGP (0)','1','10','20','30','40','50'],
            rows=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24'],
            planned_exact_256_cells=1800, directly_review_every_planned_sheet=True),
        budgets=budgets, manual_L4_tmux_only=True, failed_recipe_training_resume=False,
        historical_worker_train_or_optimizer_must_not_be_called=True,
        original_and_endpoint_all100_RAW_PNG_parity_required=True,
        new_transfer_prepared=False, new_transfer_verified=False, new_VM_forward_done=False,
        model_qualified=False, automatic_launch=False,
        evidence_sha256={name:sha(ANALYSIS/name) for name in ['learning_attribution.json','saved_objective_audit.json',
            'visual_observations.json','independent_audit.json','publication/independent_publication_audit.json']})
    write(DESIGN/'design.json', plan)
    receipt=dict(complete=True, states=14, source_bindings=len(sources), parent_bindings=len(p['parent_required_sha256']),
        vectors_finite_and_original_bound_preserved=True, all_source_files_match_verified_return=True,
        design_sha256=sha(DESIGN/'design.json'), model_qualified=False, neural_calls=0,
        optimizer_updates=0, executable_packet_prepared=False,
        status='DESIGN_ONLY_NOT_YET_EXECUTABLE', original_checkpoint_unchanged=True)
    write(DESIGN/'saved_state_selection_audit.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
