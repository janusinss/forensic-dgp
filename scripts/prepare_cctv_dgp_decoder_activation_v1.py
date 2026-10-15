"""Seal a finite manual direction measurement; never launch or construct models."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from cctv_dgp_decoder_activation_v1_contract import (
    NAME, STEM, STAGE, REPAIR_PIN, SOURCE_PIN, PARENT_PIN, REPLAY_PIN, CHECKPOINT,
    STATES, AMPLITUDES, BUDGETS, read, write, sha, validate, verify_transfer,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'


def main():
    prepared = OUT/'cctv_dgp_decoder_activation_v1_preparation'
    receipt = prepared/'preparation.json'
    assert not receipt.exists()
    root = OUT/NAME
    assert not root.exists(), 'Preserve any existing sealed or partial packet'
    repair = OUT/'cctv_dgp_decoder_hinge_repair_v1_return'
    source = OUT/'cctv_dgp_spatial_readout_v1_return'
    parent = OUT/'cctv_dgp_bank_photometric_guard_v1_return'
    replay = OUT/'cctv_dgp_spatial_path_replay_v1_return'
    previous = read(repair/'protocol.json')
    numerical = read(OUT/'cctv_dgp_decoder_hinge_repair_v1_analysis/independent_audit_r1.json')
    assert numerical['complete'] and numerical['model_qualified'] is False
    native = OUT/'cctv_dgp_hinge_native_exclusion_v1_analysis/r1'
    assert read(native/'audit.json')['complete'] and read(native/'independent_readback.json')['complete']
    cpu = read(prepared/'cpu_tests.json')
    assert cpu['complete'] and cpu['exit_code'] == 0
    copied = {
        'cctv_dgp_decoder_activation_v1_contract.py': ROOT/'scripts/cctv_dgp_decoder_activation_v1_contract.py',
        'cctv_dgp_decoder_activation_v1_math.py': ROOT/'scripts/cctv_dgp_decoder_activation_v1_math.py',
        'cctv_dgp_decoder_activation_v1_context.py': ROOT/'scripts/cctv_dgp_decoder_activation_v1_context.py',
        'scripts/cctv_dgp_decoder_activation_v1_vm.py': ROOT/'scripts/cctv_dgp_decoder_activation_v1_vm.py',
        'audit_cctv_dgp_decoder_activation_v1_return.py': ROOT/'scripts/audit_cctv_dgp_decoder_activation_v1_return.py',
        'evidence/test_math.py.txt': ROOT/'scripts/test_cctv_dgp_decoder_activation_v1_math.py',
        'evidence/test_contract.py.txt': ROOT/'scripts/test_cctv_dgp_decoder_activation_v1_contract.py',
        'evidence/test_audit.py.txt': ROOT/'scripts/test_cctv_dgp_decoder_activation_v1_audit.py',
        'evidence/cpu_tests.json': prepared/'cpu_tests.json',
        'evidence/cpu_tests.txt': prepared/'cpu_tests.txt',
        'evidence/post_hinge_review.md': ROOT/'CCTV_DGP_POST_HINGE_REPAIR_REVIEW_V1.md',
        'evidence/repair_results.md': ROOT/'CCTV_DGP_DECODER_HINGE_REPAIR_V1_RESULTS.md',
        'evidence/repair_independent_audit_r1.json': OUT/'cctv_dgp_decoder_hinge_repair_v1_analysis/independent_audit_r1.json',
        'evidence/native_exclusion_audit.json': native/'audit.json',
        'evidence/native_exclusion_readback.json': native/'independent_readback.json',
    }
    assert all(path.is_file() for path in copied.values())
    bound_sources = {name: sha(path) for name, path in copied.items()}
    manifest = read(repair/'result_manifest.json')
    repair_bindings = {name: item['sha256'] for name, item in manifest['files'].items()}
    repair_bindings['result_manifest.json'] = sha(repair/'result_manifest.json')
    gradient_layout = [*[f'MSE_{i}' for i in range(5)], *[f'SSIM_loss_{i}' for i in range(5)],
        *[f'ArcFace_loss_{i}' for i in range(5)], *[f'landmark_structure_{i}' for i in range(1,5)],
        *[f'channel_mean_{i}_{c}' for i in range(5) for c in range(3)], 'objective']
    design = dict(name=NAME, status='PREPARED_UNRUN_DIAGNOSTIC_ONLY', date_local='2026-10-15',
        hypothesis='The retained corrected A1 direction can activate one-sided penalties that its chosen gradient misses; first validate the underlying batch-five activation model before any new direction search.',
        scope='Measurement of one retained direction and four already retained rejected/control states; not a test of global or all-local-direction feasibility.',
        repair_protocol_sha256=REPAIR_PIN, source_protocol_sha256=SOURCE_PIN,
        parent_protocol_sha256=PARENT_PIN, replay_protocol_sha256=REPLAY_PIN,
        checkpoint_sha256=CHECKPOINT, route='A', state_labels=STATES, retained_amplitudes=AMPLITUDES,
        budgets=BUDGETS, derivative_reference_indices=previous['schedule'][:2],
        derivative_reference_ids=[previous['reference_blocks'][i]['reference'] for i in previous['schedule'][:2]],
        gradient_layout=gradient_layout, trainable_tensors=15, trainable_elements=609219,
        trainable_parameter_names=previous['trainable_parameter_names'], common_radius_L2=previous['common_radius_L2'],
        new_optimizer_updates=0, new_parameter_proposals=0, new_full_corpus_epochs=0,
        training_objective_batch=5, historical_diagnostic_embedding_batch=15,
        clipped_gradient_replay_bound=dict(max_abs=2e-7, cosine_min=.999999), objective_replay_max_abs=1e-10,
        gradient_norm_metadata_bound=dict(rtol=2e-6, atol=1e-12),
        clip_factor_metadata_max_ULPs=2,
        scientific_thresholds=dict(structure_gain=.01, brightness_gain_fraction=.2, MSE_slack=1e-12, SSIM_ArcFace_slack=1e-6),
        source_required_sha256=previous['source_required_sha256'], parent_required_sha256=previous['parent_required_sha256'],
        replay_required_sha256=previous['replay_required_sha256'], repair_required_sha256=repair_bindings,
        reference_blocks=previous['reference_blocks'], case_ids=previous['case_ids'],
        automatic_launch=False, failed_recipe_resume_allowed=False, model_qualified=False,
        new_incumbent_selection=False, local_neural_construction_allowed=False, direction_search=False,
        new_amplitudes=False, native_pixels_used=False, DEV_pixels_used=False, final_identity_pixels=False,
        diagnostic_only=True, training_success_not_implied=True,
        structure_gain_definition='Relative reduction in landmark-supported high-frequency luminance MSE; ArcFace is a separate preservation measure.',
        method_origin='Route A reuses the corrected DeblurGAN-v2-compatible conditional reconstruction path. Frozen external GAN features are cached only for source parity; they are not injected into this A decoder. This diagnostic is not a new own-trained GAN or qualified generative face prior.',
        manual_platform='Existing NVIDIA L4/g2-standard-4, human tmux; all gradients remain VM-only',
        limitations=['Only two TRAIN reference batches have derivative measurements; all20 have finite replay controls.',
            'One-sided formula is an affine residual model using selected neural branches and idealized real-valued clipping, not a certified float32/PNG derivative.',
            'The four tested states were retained failures/controls; numerical completion cannot select a checkpoint, justify extra epochs or qualify native/final/covering performance.'])
    root.mkdir()
    for name, path in copied.items():
        destination = root/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(destination) == bound_sources[name]
    write(root/'design.json', design)
    shell = '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -n "${TMUX:-}"
test "$#" -eq 1
python -B -u scripts/cctv_dgp_decoder_activation_v1_vm.py --root . --protocol-sha "$1" --supervise
'''
    with (root/'scripts/run_activation.sh').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(shell)
    assets = {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob('*')) if path.is_file()}
    protocol = dict(design, design_sha256=sha(root/'design.json'), assets_sha256=assets)
    validate(protocol, design)
    write(root/'protocol.json', protocol)
    pin = sha(root/'protocol.json')
    verify_transfer(root, repair, source, parent, replay, pin)
    archive = OUT/(STEM+'-execution.tar.gz')
    sidecar = Path(str(archive)+'.sha256')
    assert not archive.exists() and not sidecar.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=6) as stream:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                stream.add(path, arcname=NAME+'/'+path.relative_to(root).as_posix(), recursive=False)
    with sidecar.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(sha(archive)+'  '+archive.name+'\n')
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=pin, design_sha256=sha(root/'design.json'), packet_assets=len(assets),
        archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        root=str(root), archive=str(archive), sidecar=str(sidecar), copied_source_sha256=bound_sources,
        required_bindings={field:len(protocol[field]) for field in ['repair_required_sha256','source_required_sha256','parent_required_sha256','replay_required_sha256']},
        status='PREPARED_UNRUN; prospective independent transfer audit pending',
        automatic_launch=False, local_model_calls=0, new_parameter_proposals=0,
        gradient_queries=0, optimizer_updates=0, agent_VM_calls=0, model_qualified=False))
    print(dict(complete=True, protocol_sha256=pin, archive_bytes=archive.stat().st_size,
               new_optimizer_updates=0, status='PREPARED_UNRUN'))


if __name__ == '__main__':
    main()
