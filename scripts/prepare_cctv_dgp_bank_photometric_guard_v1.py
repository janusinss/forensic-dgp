"""Seal one new brightness-guard objective; human-only finite L4 execution."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
PRIOR_NAME = 'cctv_dgp_bank_comparison_v1_r4_B_vm'
PRIOR_STEM = 'cctv-dgp-bank-comparison-v1-r4-B'
PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'
NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
STEM = 'cctv-dgp-bank-photometric-guard-v1'
FORMAT = 'own256-conditioned-frozen-bank-photometric-training-guard-v1'
TRANSFER_NAME = 'cctv_dgp_bank_photometric_guard_v1_transfer'
GUARD = dict(name='degraded-observed-brightness-gain-hinge-v1', coefficient=10.,
    fraction_maximum=.2, profiles=['blur_lr24','lowlight_lr32','motion_lr48','compound_lr24'],
    normalizer='unchanged_MSE_normalizer', baseline='original_DGP_before_training',
    mean_control='float32_difference_float64_observed_channel_mean_clip_then_float32',
    inequality='positive_mean_only_MSE_gain <= 0.2 * positive_total_MSE_gain',
    reduction='mean_of_per_image_positive_excess_on_four_degraded_profiles',
    per_image_constraint_stronger_than_aggregate_gate=True, clear_profile_excluded=True,
    clean_target_used_for_TRAIN_loss_only=True, inference_or_display_processing_changed=False,
    coefficient_rationale='normalized hinge 10; twice the retained per-metric regression barrier; finite hypothesis, not tuned on final identities')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1024**2), b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def archive(folder, prefix, path):
    with tarfile.open(path, 'x:gz', compresslevel=1) as tar:
        for file in sorted(folder.rglob('*')):
            if file.is_file():
                assert not file.is_symlink()
                tar.add(file, arcname=prefix+'/'+file.relative_to(folder).as_posix(), recursive=False)
    digest = sha(path)
    with Path(str(path)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest+'  '+path.name+'\n')
    return digest


def main():
    parent = OUT/PRIOR_NAME
    returned = OUT/'cctv_dgp_bank_comparison_v1_r4_B_return'
    analysis = OUT/'cctv_dgp_bank_comparison_v1_r4_B_analysis'
    assert sha(parent/'protocol.json') == sha(returned/'protocol.json') == PIN
    sources = [analysis/name for name in ['independent_audit.json','execution_audit.json',
        'matched_AB_comparison.json','saved_learning_diagnostic.json','visual_review.json']]
    sources += [OUT/'cctv_dgp_bank_feature_DC_probe_v1_analysis'/name for name in ['independent_audit.json','visual_review.json']]
    math_path = OUT/'cctv_dgp_bank_photometric_guard_v1_math_audit.json'; sources.append(math_path)
    assert all(read(path)['complete'] for path in sources)
    assert read(analysis/'execution_audit.json')['actual_optimizer_updates'] == 50
    assert read(analysis/'visual_review.json')['viewed_sheets'] == 26
    assert read(sources[-2])['viewed_sheets'] == 5
    math_audit = read(math_path)
    helper = ROOT/'scripts/cctv_dgp_bank_photometric_guard_v1_loss.py'
    assert sha(helper) == math_audit['helper_sha256'] and math_audit['exact_baseline_zero_penalty_cases'] == 100
    assert math_audit['gradient_queries'] == math_audit['optimizer_updates'] == math_audit['backwards'] == 0
    dest, patch = OUT/NAME, OUT/TRANSFER_NAME
    full, small = OUT/(STEM+'-execution.tar.gz'), OUT/(STEM+'-code-transfer.tar.gz')
    assert not any(path.exists() for path in [dest, patch, full, small, OUT/'cctv_dgp_bank_photometric_guard_v1_preparation.json'])
    old = read(parent/'protocol.json'); dest.mkdir()
    for name, expected in old['assets_sha256'].items():
        assert sha(parent/name) == sha(returned/name) == expected
        target = dest/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent/name,target); assert sha(target) == expected
    for name in ['cctv_dgp_bank_comparison_v1_contract.py','cctv_dgp_bank_comparison_v1_model.py']:
        target = dest/name
        text = replace(target.read_text(encoding='utf-8'),repr(PRIOR_NAME),repr(NAME))
        if name.endswith('_contract.py'):
            text = replace(text,repr(PRIOR_STEM),repr(STEM))
            text = replace(text,repr(old['format']),repr(FORMAT))
            text = replace(text, "\ndef validate(p):", "\nPHOTOMETRIC_GUARD = "+repr(GUARD)+"\n\ndef validate(p):")
            text = replace(text,"    assert p['selected_routes'] == ['B']",
                "    assert p['selected_routes'] == ['B']\n"
                "    assert p['retained_B4_protocol_sha256'] == "+repr(PIN)+"\n"
                "    assert p['photometric_guard'] == PHOTOMETRIC_GUARD\n"
                "    assert p['start_from_unchanged_initial_B_not_failed_B50'] is True\n"
                "    assert p['bank_feature_DC_removal'] is False\n"
                "    assert p['single_training_objective_change_only'] is True")
        target.write_text(text,encoding='utf-8',newline='\n')
    shutil.copyfile(helper,dest/helper.name)
    worker = dest/'scripts/cctv_dgp_bank_comparison_v1_vm.py'
    text = worker.read_text(encoding='utf-8')
    text = replace(text,'    def objective(b, prediction, baseline):',
        '    from cctv_dgp_bank_photometric_guard_v1_loss import brightness_guard\n\n'
        '    def objective(b, prediction, baseline):')
    text = replace(text, "        value = sum(w * v for w, v in zip(p['reconstruction_weights'], normal)) + anchors + barrier",
        "        photo, photo_terms = brightness_guard(prediction, baseline, b['target'], b['mask'],\n"
        "            NORMALIZERS[0], p['photometric_guard']['coefficient'], p['photometric_guard']['fraction_maximum'])\n"
        "        value = sum(w * v for w, v in zip(p['reconstruction_weights'], normal)) + anchors + barrier + photo")
    text = replace(text,"            'anchors': float(anchors.detach()), 'barrier': float(barrier.detach())}",
        "            'anchors': float(anchors.detach()), 'barrier': float(barrier.detach()), **photo_terms}")
    text = replace(text,'                    optimizer.zero_grad(set_to_none=True); objective_mean = 0.',
        '                    optimizer.zero_grad(set_to_none=True); objective_mean = 0.; photo_mean = 0.')
    text = replace(text,"                        (loss / 5).backward(); counts['backwards'] += 1; objective_mean += terms['objective']/5",
        "                        (loss / 5).backward(); counts['backwards'] += 1; objective_mean += terms['objective']/5\n"
        "                        photo_mean += terms['photometric_guard']/5")
    text = replace(text,"                    history.append({'update': update, 'objective': objective_mean, 'gradient_norm_before_clip': grad_norm, 'reference_starts': starts})",
        "                    history.append({'update': update, 'objective': objective_mean, 'photometric_guard': photo_mean,\n"
        "                        'gradient_norm_before_clip': grad_norm, 'reference_starts': starts})")
    text = replace(text,"                progress['stage'] = 'snapshot50'",
        "                write(out / ('fit_seconds_' + route + '.json'), {'complete': True, 'route': route,\n"
        "                    'seconds': time.monotonic()-fit_start, 'cap_seconds': BUDGETS['arm_fit_seconds'],\n"
        "                    'updates': progress['route_updates'], 'includes_post_update1_gradient_checks': True})\n"
        "                progress['stage'] = 'snapshot50'")
    worker.write_text(text,encoding='utf-8',newline='\n')
    # Adapt only the data-only installer; retain the historical installer asset.
    source_installer = ROOT/'scripts/install_cctv_dgp_bank_B4_from_B3_v1.py'
    text = source_installer.read_text(encoding='utf-8')
    text = replace(text,"SOURCE_NAME = 'cctv_dgp_bank_comparison_v1_r3_B_vm'","SOURCE_NAME = "+repr(PRIOR_NAME))
    text = replace(text,"TARGET_NAME = 'cctv_dgp_bank_comparison_v1_r4_B_vm'","TARGET_NAME = "+repr(NAME))
    text = replace(text,"TRANSFER_NAME = 'cctv_dgp_bank_comparison_B4_transfer'","TRANSFER_NAME = "+repr(TRANSFER_NAME))
    text = replace(text,"SOURCE_PIN = 'f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35'","SOURCE_PIN = "+repr(PIN))
    text = replace(text,"    assert p['retained_B3_protocol_sha256'] == SOURCE_PIN","    assert p['retained_B4_protocol_sha256'] == SOURCE_PIN")
    before = "    failure_path = source / 'outputs/bank_comparison_v1/failure.json'\n    failure = read(failure_path)\n    assert failure['message'] == \"module 'hashlib' has no attribute 'file_digest'\"\n    assert all(value == 0 for value in failure['progress']['counts'].values())"
    after = "    failure_path = source / 'outputs/bank_comparison_v1/quality_stop_B.json'\n    failure = read(failure_path)\n    assert failure['type'] == 'quality_gate_failure' and failure['outcome']['updates'] == 50\n    assert failure['outcome']['capacity_pass'] is False and failure['failed_recipe_resume_allowed'] is False"
    text = replace(text,before,after)
    text = replace(text,"    assert supervisor['complete'] and supervisor['optimizer_updates'] == 0 and supervisor['trainer_exit_code'] == 1",
        "    assert supervisor['complete'] and supervisor['optimizer_updates'] == 50 and supervisor['trainer_exit_code'] == 2")
    text = replace(text,'retained B3 assets','retained B4 assets')
    installer = dest/'scripts/install_packet_from_B4.py'; installer.write_text(text,encoding='utf-8',newline='\n')
    provenance = dest/'provenance/photometric_guard_v1'; provenance.mkdir()
    for path in sources:
        name = path.parent.name+'__'+path.name
        shutil.copyfile(path,provenance/name)
    for path,name in [(returned/'protocol.json','B4_protocol.json'),
        (returned/'outputs/bank_comparison_v1/outcome_B.json','B4_outcome_B.json'),
        (returned/'outputs/bank_comparison_v1/quality_stop_B.json','B4_quality_stop_B.json')]: shutil.copyfile(path,provenance/name)
    decision = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        hypothesis='Existing training omits the retained brightness-only gain requirement; add a differentiable observed-pixel hinge, preserving useful bank spatial features.',
        one_scientific_change='additional degraded TRAIN photometric objective term', guard=GUARD,
        global_bank_feature_DC_removal_rejected=True, removing_DC_damaged_clear_structure=True,
        start='unchanged hash-pinned initial_B at permanent original DGP; never rejected B50',
        initializer_weights_architecture_frozen_state_optimizer_data_schedule_gates_caps_unchanged=True,
        guard_reduces_brightness_dependence_not_guaranteed_structure_gain=True,
        finite_quality_comparison_only=True, maximum_updates=50, full_epochs=0,
        native_DEV_unpaired=True, final_identities=0, local_gradient_or_optimizer_calls=0,
        independent_return_and_all_visual_sheets_required_before_more_training=True,
        after_third_unsuccessful_trained_comparison_stop_modification_and_ask_diagnostic_question=True,
        evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in sources},
        all_five_milestones_and_seven_completion_families_still_required=True,
        app_promotion=False, goal_complete=False)
    write(provenance/'decision.json',decision)
    p = copy.deepcopy(old)
    p.update(format=FORMAT,UTC=datetime.now(timezone.utc).isoformat(),retained_B4_protocol_sha256=PIN,
        photometric_guard=GUARD,start_from_unchanged_initial_B_not_failed_B50=True,
        bank_feature_DC_removal=False,single_training_objective_change_only=True)
    p['assets_sha256'] = {path.relative_to(dest).as_posix():sha(path) for path in sorted(dest.rglob('*')) if path.is_file()}
    write(dest/'protocol.json',p); pin = sha(dest/'protocol.json')
    spec = importlib.util.spec_from_file_location('guard_contract',dest/'cctv_dgp_bank_comparison_v1_contract.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.verify(dest,pin)
    for name in p['assets_sha256']:
        if name.endswith('.py'): ast.parse((dest/name).read_text(encoding='utf-8-sig'),filename=name,feature_version=(3,10))
    full_hash = archive(dest,NAME,full)
    patch.mkdir(); shutil.copyfile(dest/'protocol.json',patch/'protocol.json')
    shutil.copyfile(parent/'protocol.json',patch/'source_protocol.json'); shutil.copyfile(installer,patch/'install_packet.py')
    overlays = {name:expected for name,expected in p['assets_sha256'].items() if old['assets_sha256'].get(name) != expected}
    for name in overlays:
        target = patch/'overlay'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(dest/name,target)
    write(patch/'transfer.json',dict(complete=True,source_name=PRIOR_NAME,target_name=NAME,
        protocol_sha256=pin,source_protocol_sha256=PIN,overlay_assets=overlays,installer_sha256=sha(installer),
        copy_cap_seconds=300,minimum_free_after_install_bytes=16*1024**3,
        source_or_research_files_deleted=0,model_gradient_or_training_calls=0))
    small_hash = archive(patch,TRANSFER_NAME,small)
    result = dict(complete=True,name=NAME,stem=STEM,route='B',protocol_sha256=pin,
        archive_sha256=full_hash,archive_bytes=full.stat().st_size,packet_assets=len(p['assets_sha256']),
        source_B4_protocol_sha256=PIN,transfer_name=TRANSFER_NAME,transfer_stem=STEM+'-code-transfer',
        transfer_archive_sha256=small_hash,transfer_archive_bytes=small.stat().st_size,
        transfer_manifest_sha256=sha(patch/'transfer.json'),overlay_assets=len(overlays),
        reused_assets=len(p['assets_sha256'])-len(overlays),math_audit_sha256=sha(math_path),
        changed_scientific_recipe='single additional degraded TRAIN brightness-gain hinge',
        architecture_initializer_weights_optimizer_data_schedule_gates_caps_unchanged=True,
        start_not_failed_B50=True,manual_VM_run_pending=True,VM_storage_inventory_and_cleanup_pending=True,
        local_neural_gradient_or_optimizer_calls=0,model_qualified=False,app_promotion=False,goal_complete=False)
    write(OUT/'cctv_dgp_bank_photometric_guard_v1_preparation.json',result)
    print(result,flush=True)


if __name__ == '__main__': main()
