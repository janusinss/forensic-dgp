"""Seal a manual VM packet using read-only tensors, arrays and file operations."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from cctv_dgp_spatial_readout_v1_contract import (
    NAME, STEM, PARENT_NAME, PARENT_PIN, CHECKPOINT, THRESHOLDS, BUDGETS,
    read, write, sha, validate_protocol, verify_paths,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs'
PARENT = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
NEARBY = OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_return'
REVIEW = OUT/'cctv_dgp_reconstruction_design_review_v2'


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous()
        h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        h.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return h.hexdigest()


def main():
    assert sha(PARENT/'protocol.json') == PARENT_PIN
    old, previous = read(PARENT/'protocol.json'), read(NEARBY/'protocol.json')
    decision = read(REVIEW/'user_decision.json')
    assert decision['human_answer'] == 'apply the best  approach to improve'
    assert decision['required_architecture_discussion_resolved'] and decision['no_training_or_gradient_launch_authorized_to_agent'] is True
    assert sha(NEARBY/'readouts.npz') == previous['assets_sha256']['readouts.npz']
    packet = OUT/NAME
    archive = OUT/(STEM+'-execution.tar.gz')
    assert not packet.exists() and not archive.exists(), 'Preserve an existing packet'
    old_bindings = old['assets_sha256']
    blocks = previous['reference_blocks']
    cases = {c['id']: c for c in old['cases']}
    refs = {r['id']: r for r in old['references']}
    needed = {n for n in old_bindings if '/' not in n or n.startswith(('models/', 'bank_implementation/', 'prior_source/', 'weights/'))}
    needed |= {'initializers/initial_B.pth', 'initializers/mean_style.npy'}
    for block in blocks:
        needed |= {cases[c]['input'] for c in block['case_ids']}
        needed |= {refs[block['reference']]['target'], refs[block['reference']]['observed']}
    parent_required = {n: old_bindings[n] for n in sorted(needed)}
    for block in blocks:
        name = f"outputs/bank_comparison_v1/baseline/packs/b{block['begin']//5:03d}.npz"
        assert previous['parent_required_sha256'][name] == sha(PARENT/name)
        parent_required[name] = previous['parent_required_sha256'][name]
    verify_paths(PARENT, parent_required)
    import numpy as np
    import torch  # read-only serialization; no neural constructors or assignment
    from PIL import Image
    state = torch.load(PARENT/'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    assert state_hash(state) == '98675d10be6f3405dd4bd0950d9251a61b26b9b161ae62a9ce48f775bc43e2bd'
    current = ['net.head4.block0.weight', 'net.head4.block1.weight', 'live_fusion4'] + [
        f'net.head{i}.block{j}.weight' for i in [1, 2, 3] for j in [0, 1]] + [
        'live_fusion_fine', 'net.smooth.0.bias', 'net.smooth2.0.weight', 'net.smooth2.0.bias', 'net.final.weight', 'net.final.bias']
    names_A = ['current.'+n for n in current]
    names_B = names_A+[f'projectors.{size}.{term}' for size in [16, 32, 64, 128, 256] for term in ['weight', 'bias']]
    assert sum(state[n].numel() for n in names_A) == 609219 and len(names_A) == 15
    target_rows = []
    for block in blocks:
        ref = refs[block['reference']]
        with Image.open(PARENT/ref['target']) as image:
            rgb = np.asarray(image.convert('RGB')).copy()
        actual = hashlib.sha256(rgb.tobytes()).hexdigest()
        assert rgb.shape == (256, 256, 3) and actual == old['canonical_target_RGB_sha256'][ref['id']]
        assert all(cases[c]['role'] == 'train' for c in block['case_ids'])
        target_rows.append(dict(reference=ref['id'], source=ref['source'], canonical_target_RGB_sha256=actual,
            ancestral_RGB_sha256=ref['target_rgb_sha256'], evaluation_source_kind=ref.get('evaluation_source_kind'),
            evaluation_source_kind_field_present='evaluation_source_kind' in ref,
            missing_kind_not_relabelled_as_HQ=True))
    source_groups = sorted({b['source'] for b in blocks})
    buckets = [sorted([i for i, b in enumerate(blocks) if b['source'] == s],
        key=lambda i: hashlib.sha256(('spatial-readout-v1|'+blocks[i]['reference']).encode()).hexdigest()) for s in source_groups]
    assert len(buckets) == 2 and len(buckets[0]) == len(buckets[1]) == 10
    order = [i for pair in zip(*buckets) for i in pair]
    schedule = (order*3)[:50]
    proof_sources = [ROOT/'CCTV_DGP_RECONSTRUCTION_DESIGN_REVIEW_V2.md', REVIEW/'user_decision.json', REVIEW/'boundary_green.txt',
        ROOT/'CCTV_DGP_FROZEN_RECONSTRUCTION_PATH_REVIEW_V1.md', ROOT/'CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md',
        ROOT/'CCTV_DGP_MIXED_V9_RESULTS.md', ROOT/'CCTV_DGP_BANK_AB_COMPARISON_V1_RESULTS.md',
        ROOT/'CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md']
    assert all(path.is_file() for path in proof_sources)
    packet.mkdir()
    (packet/'scripts').mkdir()
    (packet/'evidence').mkdir()
    origins = {}
    for name in ['cctv_dgp_spatial_readout_v1_contract.py', 'cctv_dgp_spatial_readout_v1_model.py',
                 'audit_cctv_dgp_spatial_readout_v1_return.py']:
        source = ROOT/'scripts'/name
        ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10))
        shutil.copyfile(source, packet/name)
        origins[name] = dict(source_path=source.relative_to(ROOT).as_posix(), sha256=sha(source), exact_copy=True)
    worker_name = 'scripts/cctv_dgp_spatial_readout_v1_vm.py'
    shutil.copyfile(ROOT/worker_name, packet/worker_name)
    origins[worker_name] = dict(source_path=worker_name, sha256=sha(ROOT/worker_name), exact_copy=True)
    shutil.copyfile(NEARBY/'readouts.npz', packet/'readouts.npz')
    origins['readouts.npz'] = dict(source_path=(NEARBY/'readouts.npz').relative_to(ROOT).as_posix(), sha256=sha(NEARBY/'readouts.npz'), exact_copy=True)
    for i, source in enumerate(proof_sources):
        name = f'evidence/{i:02d}_'+source.name
        shutil.copyfile(source, packet/name)
        origins[name] = dict(source_path=source.relative_to(ROOT).as_posix(), sha256=sha(source), exact_copy=True)
    shell = '''#!/usr/bin/env bash
set -euo pipefail
PIN="${1:?Supply the frozen protocol SHA256}"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
test "$PWD" = "$HOME/forensic-dgp/cctv_dgp_spatial_readout_v1_vm"
test -n "${TMUX:-}"
exec "$PY" -B -u scripts/cctv_dgp_spatial_readout_v1_vm.py --root . --protocol-sha "$PIN" --supervise
'''
    with (packet/'scripts/run_spatial_readout.sh').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(shell)
    write(packet/'evidence/target_bindings.json', dict(complete=True, targets=target_rows, cases=100,
        native_pixels=0, DEV_pixels=0, final_pixels=0, neural_calls=0, read_only_tensors=True))
    p = dict(name=NAME, format='own256-frozen-bank-spatial-reconstruction-comparison-v1',
        UTC=datetime.now(timezone.utc).isoformat(), parent_protocol_sha256=PARENT_PIN,
        original_checkpoint_sha256=CHECKPOINT, original_state=old['original_state'], recognizer_state=old['recognizer_state'],
        standalone_prior_sha256=old['standalone_prior_sha256'], feature_provider_initial_state=state_hash(state),
        reference_blocks=blocks, case_ids=old['preview_case_ids'], paired_TRAIN_cases=100,
        routes={'A': {'tensors': 15, 'elements': 609219}, 'B': {'tensors': 25, 'elements': 2176419}},
        selected_routes=['A', 'B'], updates_per_route=50, schedule=schedule,
        trainable_parameter_names={'A': names_A, 'B': names_B}, B_readout_kernel=3,
        conditioning_trainable=False, observation_trainable=False, prior_trainable=False, normalization_trainable=False,
        initial_parity_policy='exact_RAW_and_PNG_all100', reconstruction_weights=old['reconstruction_weights'],
        normalizers=old['normalizers'], active_anchor_normalizers=old['active_anchor_normalizers'],
        regression_barrier_coefficient=5, photometric_coefficient=10., brightness_fraction_maximum=.2,
        scientific_thresholds=THRESHOLDS, distance_fraction_maximum=.001,
        common_radius_basis='minimum_initial_A_B_historical_conditioning_L2',
        unconstrained_proposal_execution='shadow_parameters_never_model_forward',
        optimizer=dict(name='Adam', lr=.0001, betas=[.9, .999], eps=1e-8, weight_decay=1e-5,
                       gradient_clip_L2=1., projection='common_original_ball_after_each_step'),
        optimizer_foreach=False, optimizer_fused=False, budgets=BUDGETS,
        evaluator='original cached controls; float32 embedding dot; unchanged RAW/PNG pixel metrics; 15-image evaluation batches',
        initial_cached_control_scores_preserved=True, original_embedded_targets_reused_for_TRAIN_loss=True,
        fresh_GPU_baseline_matches_all100_retained_RAW_required=True,
        intermediate_fit_states='research only; not accepted preserving checkpoints',
        endpoint_gates_required=True, parameter_trajectory_and_stopped_full_states_retained=True,
        independent_CUDA_gradient_replication_claim=False,
        sample_passes_per_route=2.5, full_corpus_epochs=0, native_pixels_used=0, DEV_pixels_used=0,
        final_identity_pixels=0, completion_forwards=0, app_promotion=False, automatic_follow_on=False,
        failed_recipe_resume_allowed=False, agent_training_allowed=False, parent_required_sha256=parent_required,
        source_origins=origins, source_labels_not_ethnicity=True,
        new_random_full_network_training=False, external_prior_not_own_trained_GAN=True,
        estimate_minutes_with_checks_and_export=[20, 40], estimate_not_fresh_CUDA_timing=True,
        all_five_milestones_and_seven_covering_families_unchanged=True,
        assets_sha256={path.relative_to(packet).as_posix(): sha(path) for path in sorted(packet.rglob('*')) if path.is_file()})
    validate_protocol(p)
    write(packet/'protocol.json', p)
    pin = sha(packet/'protocol.json')
    with archive.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=6) as stream:
            for path in sorted(packet.rglob('*')):
                if path.is_file():
                    stream.add(path, arcname=(Path(NAME)/path.relative_to(packet)).as_posix(), recursive=False)
    digest = sha(archive)
    with (OUT/(archive.name+'.sha256')).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), protocol_sha256=pin, archive_sha256=digest,
        bytes=archive.stat().st_size, packet_members=len(p['assets_sha256'])+1,
        parent_assets=len(parent_required), source_origins=origins, packet_status='PREPARED_UNRUN_PENDING_INDEPENDENT_PREPARATION_AUDIT',
        neural_constructors=0, neural_forwards=0, gradient_queries=0, optimizer_updates=0,
        read_only_tensor_inspection=True, model_qualified=False, VM_calls=0)
    write(REVIEW/'preparation.json', record)
    print(json.dumps({k: record[k] for k in ['complete', 'protocol_sha256', 'archive_sha256', 'bytes', 'packet_members', 'parent_assets', 'optimizer_updates']}))


if __name__ == '__main__':
    main()
