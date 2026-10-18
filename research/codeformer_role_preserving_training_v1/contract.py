"""Standard-library-only prospective finite role-preserving restorer contract.

Importing this module performs no neural work, state loading or training.
The independent source/data/transport reviews and a manual VM launch remain
required. Constants describe a new experiment, not an achieved quality result.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

FORMAT = 'codeformer-role-preserving-training-finite-manual-v1'
MODULE = 'research.codeformer_role_preserving_training_v1.worker'
VM_NAME = 'cctv_codeformer_role_preserving_training_v1_vm'
PARENT_NAME = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
OUTPUTS = 'outputs/role_preserving_training_v1'
PREFIX = 'cctv-codeformer-role-preserving-training-v1'
LOCK_NAME = '.role_preserving_training_v1.lock'
PARENT_PROTOCOL_SHA = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
PARENT_PROTOCOL_CANONICAL_SHA = '3b0644acae19f82d429483c4a64fddc6f5e61a329add05a80d2c3f727624e30e'
DEPENDENCIES_SHA256 = '0cf7922ff74d9a01139d943b9f27b3b15a1d6b0e0d3e5bbc6d668c77fcaff8b9'
RETAINED_OUTPUTS_SHA256 = '915bb9084ce5752f8a4cd4df0eb4d6a6b44c0f12b09b4703d25efa2764e2ec74'
HEX = re.compile(r'[0-9a-f]{64}')
REFERENCE_ID = re.compile(r'tr_ffhq_[0-9]{5}')
DESIGN_DOCUMENT = 'research/codeformer_role_preserving_training_v1/DESIGN.md'
DESIGN_DOCUMENT_SHA256 = '44a54d4d1c90e1a76fea89971703447080e78bb5608e7fd116e9810c1e12a0c9'

ASSETS = {
    'initializer': {'root': 'adain', 'path': 'weights/codeformer.pth',
        'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'bytes': 376637898},
    'full_coarse_checkpoint': {'root': 'adain',
        'path': 'outputs/adain_gradient_isolation_v1/stopped_checkpoint.pt',
        'sha256': '43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b', 'bytes': 855230072},
    'hq_teacher': {'root': 'adain', 'path': 'weights/vqgan_code1024.pth',
        'sha256': '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f', 'bytes': 255078528},
    'perceptual_features': {'root': 'adain', 'path': 'perceptual/vgg16_features.pth',
        'sha256': 'a5b412426cfb06d2196179871dff81a905f97688cd46f2f16df1b63d5b251279', 'bytes': 58867573},
    'perceptual_calibration': {'root': 'adain', 'path': 'perceptual/lpips_vgg_v01.pth',
        'sha256': 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868', 'bytes': 7289}}
MODEL_DIGESTS = {
    'original': '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
    'base': '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'}
LOSS_WEIGHTS = {'token_ce': .5, 'latent_mse': 1., 'coarse_rgb_l1': 4.,
    'pixel_l1_rgb': 1., 'lpips_vgg': 1., 'behavior_rgb_l1': 2., 'behavior_token_kl': .1}
NORMAL_TERMS = ('behavior_rgb_l1', 'behavior_token_kl')
DARK_TERMS = ('token_ce', 'latent_mse', 'coarse_rgb_l1', 'pixel_l1_rgb', 'lpips_vgg',
              'behavior_rgb_l1', 'behavior_token_kl')
OBJECTIVE = {
    'normal_slots': [1, 3], 'dark_slots': [2, 4],
    'normal_terms': list(NORMAL_TERMS), 'dark_eligible_terms': list(DARK_TERMS),
    'dark_ineligible_terms': [x for x in DARK_TERMS if x not in ('pixel_l1_rgb', 'lpips_vgg')],
    'weights': LOSS_WEIGHTS, 'normal_teacher': 'untouched-original-CodeFormer',
    'dark_behavior_teacher': 'frozen-full-coarse-AdaIN50',
    'normal_clean_supervision': False, 'normal_clean512_argument': None,
    'normal_codes_argument': None, 'normal_LPIPS_forwards': 0,
    'normal_spatial_RGB_preservation': 'all delivered256 RGB pixels; mean L1, not only mean luminance',
    'behavior_RGB': 'clamp((native512.float()+1)/2,0,1) then bilinear256 align_corners=False',
    'behavior_token_KL': 'KL(teacher||candidate), softmax/log_softmax T=2, sum vocabulary then mean batch/token, times T^2',
    'dark_clean_RGB': '(native512.float()+1)/2; unclipped RGB512',
    'dark_coarse_RGB': 'mean L1(avg_pool2d(predicted_RGB512,32,32),avg_pool2d(clean_RGB512,32,32))',
    'dark_pixel_RGB': 'mean L1(unclipped predicted_RGB512,clean_RGB512); eligible only',
    'dark_LPIPS': 'frozen LPIPS-v0.1/VGG16(native512,clean_RGB512*2-1); eligible only',
    'dark_HQ': 'frozen HQ indices; feature target from frozen restoration dictionary at those indices',
    'all_behavior_targets_fresh_per_observation': True,
    'all_candidate_terms_fresh_per_observation': True, 'cached_gradient_reuse': False,
    'gradient_accumulation_divisor': 4, 'role_given_to_network_or_inference': False,
    'tone_head': False, 'output_centering': False, 'brightness_gate': False,
    'weight_or_rate_search': False, 'architecture_changed': False}
TRAINING = {
    'max_updates': 50, 'microbatch': 1, 'accumulate': 4, 'snapshots': [0, 50],
    'seed': 20261015, 'retained_data_updates': [1, 50], 'TRAIN_observations': 200,
    'normal_observations': 100, 'dark_observations': 100, 'HQ_selected_references': 100,
    'candidate_initialization': 'full_coarse_checkpoint.model only',
    'candidate_tensors': 515, 'trainable_tensors': 350, 'frozen_tensors': 165,
    'encoder_token_tensors': 278, 'fusion_tensors': 72,
    'trainable_prefixes': ['encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'fuse_convs_dict'],
    'frozen_prefixes': ['quantize', 'generator'],
    'optimizer': 'Adam', 'betas': [.9, .99], 'eps': 1e-8, 'weight_decay': 0.,
    'encoder_token_lr': 1e-5, 'fusion_lr': 2e-5, 'warmup_updates': 50,
    'schedule': 'new-run linear warmup: base_lr*new_update/50 for new_update1..50 only',
    'historical_optimizer_RNG_sampler_schedule_installed': False,
    'fresh_optimizer_empty_state_required': True, 'post50_schedule_authorized': False,
    'autocast': 'disabled-FP32', 'grad_scaler': False, 'gradient_clipping': None,
    'GAN': False, 'EMA': False, 'resume': False, 'loss_weights': LOSS_WEIGHTS,
    'distillation_temperature': 2., 'pair_policy': 'paired-linear-light-keyed-fields-v2',
    'severe_edges': [24, 28, 32, 40],
    'control_variants': ['clear256', 'clear128', 'clear64', 'light256', 'light128', 'light64'],
    'cycle': 'u=update-1; Aedge=edges[u%4]; Bvariant=variants[(u//4)%6]',
    'slot_roles': {'1': 'normal', '2': 'dark', '3': 'normal', '4': 'dark'},
    'max_gradient_queries': 5, 'max_backward_calls': 200}
GRADIENT_ROUTE = {
    'policy': 'candidate-AdaIN-style-detach-only-v1',
    'witness_update': 3, 'witness_slots': [1, 2], 'candidate_forwards': 2,
    'behavior_teacher_forwards': 2, 'perceptual_forwards': 1,
    'component_gradient_queries': 4, 'before_Adam_construction': True,
    'same_role_helper_as_training': True, 'normal_terms': list(NORMAL_TERMS),
    'dark_terms': list(DARK_TERMS),
    'image_partition': ['coarse_rgb_l1', 'pixel_l1_rgb', 'lpips_vgg', 'behavior_rgb_l1'],
    'token_HQ_partition': ['token_ce', 'latent_mse', 'behavior_token_kl'],
    'isolated_image_encoder_and_predictor': 'all None; not merely numerical zero',
    'isolated_image_fusion': 'finite and aggregate nonzero',
    'token_HQ_encoder_and_predictor': 'finite and aggregate nonzero',
    'individual_dark_parent_retention_gradient_nonzero_required': False,
    'unscoped_initial_candidate_capture_reused_for_forward_parity': True,
    'forward_equality': 'bit-exact native/raw/logits/features/tokens and common role terms',
    'candidate_all_module_modes_and_RNG_match_initial_capture': True,
    'CPU_CUDA_Python_NumPy_RNG_restored': True,
    'parameters_grad_None_before_after': True,
    'no_historical_original_route_witness': True,
    'full_gradient_vectors_exported': False,
    'gradient_norms_are_source_bound_not_independent_replay': True}
PERCEPTUAL_PREREQUISITE = {
    'forwards': 1, 'gradient_queries': 1, 'before_Adam_construction': True,
    'prediction_source': 'detached saved dark witness native512 as fresh requires-grad leaf',
    'target_source': 'same retained clean512*2-1',
    'finite_nonzero_prediction_gradient_required': True, 'new_CodeFormer_forwards': 0,
    'parameter_grad_fields_unchanged': True, 'RNG_restored': True}
INITIALIZER_PARITY = {
    'candidate_forwards': 2, 'parent_forwards': 2, 'perceptual_forwards': 1,
    'cases': ['train_u0003_slot1', 'train_u0003_slot2'],
    'candidate_unscoped_capture_same_modes_RNG_as_scoped_witness': True,
    'candidate_unscoped_full_role_helper': True,
    'scoped_vs_unscoped_role_terms': 'bit-exact full actual normal/dark terms and weighted totals',
    'candidate_vs_frozen_parent': 'bit-exact native/raw/logits/features/tokens',
    'original_teacher_identity': 'strict original file SHA and full515-tensor digest; no extra original benchmark forward',
    'snapshot0_vs_retained_parent_all406': 'byte/array equality of RAW and PNG; no extra neural calls'}
EVALUATION = {
    'validation_references': 53, 'profiles': ['clear', 'light', 'blur', 'jpeg', 'lowlight', 'compound'],
    'comparison_cases': 88, 'native_unpaired': 48, 'old_paired': 40, 'cases_per_snapshot': 406,
    'snapshots': [0, 50], 'RAW_and_PNG_retained': True, 'views': ['raw', 'native_png'],
    'clear_cases': 61, 'clear_metrics': ['mae'],
    'normal_blur_cases': 53, 'blur_photometric_metrics': ['luminance_bias', 'coarse_rgb_mae'],
    'mean_regression_limit': .01, 'worst_case_regression_limit': .03,
    'normal_guard_baseline': 'exact retained original-CodeFormer snapshot0000, not new fullcoarse snapshot0',
    'snapshot0_known_failed_guards_recorded_without_abort': True,
    'snapshot50_preservation_required': True,
    'dark_retention_baseline': 'exact full-coarse parent; new snapshot0 parity required',
    'dark_retention_profiles': ['lowlight', 'compound'], 'dark_cases_per_profile': 53,
    'dark_retention_view': 'native_png',
    'dark_retention_rules': ['mean MAE <= parent mean MAE', 'mean SSIM >= parent mean SSIM'],
    'all_dark_RAW_and_PNG_metrics_recorded': True,
    'native_clean_target': None, 'native_identity_or_accuracy_claim': False,
    'fidelity_weight': 1., 'adain': True, 'alignment': 'none', 'display_processing': 'none',
    'delivered_pixels': 256, 'internal_pixels': 512,
    'validation_target_policy': 'retained direct RGB1024-to-Pillow-LANCZOS256',
    'PNG_conversion': 'floor(clip(RAW,0,1)*255) uint8; observed input padding restored exactly',
    'FINAL_used': False, 'quality_thresholds_changed': False,
    'automatic_qualification_or_app_promotion': False}
LIMITS = {
    'worker_seconds': 3000, 'initialization_seconds': 600, 'witness_seconds': 180,
    'training_seconds': 900, 'snapshot_seconds': 600,
    'preflight_seconds': 300, 'export_seconds': 600, 'external_grace_seconds': 30,
    'initial_free_bytes': 16*1024**3, 'reserve_free_bytes': 4*1024**3,
    'RSS_bytes': 8*1024**3, 'CUDA_reserved_bytes': 22*1024**3,
    'result_payload_bytes': 4*1024**3, 'archive_bytes': 17*1024**3//4,
    'checkpoint_temporary_bytes': 1024**3,
    'initial_checkpoint_file_bytes': 384*1024**2, 'trained_checkpoint_file_bytes': 832*1024**2,
    'regular_file_cap': 7000, 'metadata_bytes': 128*1024**2,
    'PNG_file_bytes': 198656, 'array_file_bytes': 256*1024**2,
    'training_history_line_bytes': 65536, 'log_bytes': 2*1024**2, 'one_JSON_bytes': 2*1024**2,
    'source_packet_files': 64, 'source_packet_bytes': 16*1024**2}


def expected_call_caps(training=None):
    t = TRAINING if training is None else training
    require(t['max_updates'] == 50 and t['snapshots'] == [0, 50], 'Finite new fifty-update pilot only')
    # A-dark is image eligible on24 of50 updates; all50 B-dark are eligible.
    dark_eligible = sum(1 + int(((u-1) % 4) >= 2) for u in range(1, 51))
    return {'optimizer_step_attempts': 50, 'backward_calls': 200, 'gradient_queries': 5,
        'route_witness_forwards': 2, 'route_witness_behavior_teacher_forwards': 2,
        'route_witness_perceptual_forwards': 1, 'perceptual_prerequisite_forwards': 1,
        'candidate_training_forwards': 200, 'perceptual_training_forwards': dark_eligible,
        'original_behavior_teacher_forwards': 100, 'parent_behavior_teacher_forwards': 100,
        'teacher_forwards': 100, 'snapshot_forwards': 2*406,
        'initializer_parity_candidate_forwards': 2, 'initializer_parity_parent_forwards': 2,
        'initializer_parity_perceptual_forwards': 1,
        'training_input_writes': 200}


def require(value, message):
    if not value:
        raise ValueError(message)


COUNTS = expected_call_caps()
FORWARD_COUNTERS = tuple(n for n in COUNTS if n.endswith('_forwards'))
NEURAL_FORWARD_CAP = sum(COUNTS[n] for n in FORWARD_COUNTERS)
RESERVATION_CAP = sum(COUNTS.values())
require(NEURAL_FORWARD_CAP == 1397 and RESERVATION_CAP == 1852, 'Prospective operation census differs')
TRAINING['call_caps'] = COUNTS
CHECKPOINT_PATHS = [OUTPUTS+'/snapshot_0000/checkpoint.pt', OUTPUTS+'/snapshot_0050/checkpoint.pt',
                    OUTPUTS+'/stopped_checkpoint.pt']
STATE_POLICY = {
    'completion_counter': 'completed_training_updates', 'initial_Adam_state_empty': True,
    'checkpoint_fields': ['model', 'optimizer', 'RNG', 'sampler', 'rates', 'protocol', 'actual_counters'],
    'teachers_serialized_in_checkpoint': False, 'actual_initial_and50_checkpoints_required_on_success': True,
    'checkpoint50_written_only_after_all_snapshot50_calls_complete': True,
    'checkpoint50_and_actual_stopped_mutually_exclusive': True,
    'at_most_one_updated_checkpoint_final_or_temporary_slot': True,
    'stopped_checkpoint_or_explicit_exact50_alias_required': True,
    'stopped_alias_file': OUTPUTS+'/stopped_checkpoint_alias.json',
    'alias_only_completed_training_updates': 50,
    'alias_requires_optimizer_inflight_false_and_partial_microbatches_zero': True,
    'alias_requires_exact_committed_model_Adam_RNG_sampler_rates_counters_digest': True,
    'alias_requires_retained_checkpoint50_SHA_and_bytes': True,
    'no_new_reservations_RNG_mutations_or_model_operations_after_checkpoint50_before_alias_terminal': True,
    'alias_forbidden_with_stopped_checkpoint_or_checkpoint_temporary': True,
    'uncertain_or_partial_failure_must_preserve_actual_state_and_temporaries': True,
    'training_success_or_qualification_implied_by_export': False}
PARENT_CONTROLS = {
    'training_parity': {'path': 'training_parity_v1.json',
        'sha256': '743c073b92dd1bcfe37210a854277fb31b5e6dbe562d1229fa05903b41c5c6c6',
        'cases': 200, 'updates': 50, 'historical_behavior_anchors': 50,
        'historical_behavior_anchor_flag_not_new_objective_dispatch': True},
    'hq_cache_parity': {'path': 'hq_cache_parity_v1.json',
        'sha256': 'f21dbe09e4b7af1b7d6df5ebd45c2c7230ed10face3e33a33a4c843949fd75ac',
        'historical_cases': 391, 'selected_fresh_cases': 100},
    'validation_saved': {'manifest': 'validation_saved_v1.json',
        'sha256': 'ff76007be1f55e70e12e5d5c3aad00c53e4564326e6d10eeb1e12a38b9178d84', 'case_count': 318}}
RETURN_MANIFEST = {'path': 'outputs/adain_gradient_isolation_v1/export_manifest_20261016T170106Z-8446018b.json',
    'sha256': '0091134a5653825fb3a06d63e3907d1e57521f607e67da5d39ea856cdceff805', 'bytes': 1183232}
FIXED_RETURNED_DEPENDENCIES = {
    RETURN_MANIFEST['path']: {k: RETURN_MANIFEST[k] for k in ('sha256', 'bytes')},
    'outputs/adain_gradient_isolation_v1/runtime.json': {
        'sha256': 'e0d1b088f94aaf7d539b679a70584515d46b128db8792aebee7acafb8c7277fa', 'bytes': 3690},
    'outputs/adain_gradient_isolation_v1/snapshot_0000/metrics.json': {
        'sha256': 'b7bc3eea98a3c9e2883d13d159aafc57d3e097831af5639f36e546e4dbbaa1fa', 'bytes': 346566},
    'outputs/adain_gradient_isolation_v1/snapshot_0050/metrics.json': {
        'sha256': 'b3255acdce891494163a19828cebcd65d4530856a206d1a32e0269bddcad9451', 'bytes': 369913}}
HISTORICAL_RUNTIME_FILES = (
    'research/codeformer_adaptation_v1/__init__.py', 'research/codeformer_adaptation_v1/common.py',
    'research/codeformer_adaptation_v1/perceptual.py', 'research/codeformer_adaptation_v2/__init__.py',
    'research/codeformer_adaptation_v2/model.py', 'research/codeformer_adain_gradient_isolation_v1/__init__.py',
    'research/codeformer_adain_gradient_isolation_v1/gradient_route.py', 'third_party/__init__.py',
    'third_party/codeformer/__init__.py', 'third_party/codeformer/registry.py',
    'third_party/codeformer/vqgan_arch.py', 'third_party/codeformer/codeformer_arch.py')
REQUIRED_SOURCE_FILES = {'research/__init__.py', DESIGN_DOCUMENT,
    'scripts/supervise_codeformer_role_preserving_training_v1_vm.py',
    *('research/codeformer_role_preserving_training_v1/'+n for n in
      ('__init__.py', 'contract.py', 'data.py', 'model.py', 'worker.py', 'transport.py'))}


def sha256(path, budget=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            if budget is not None:
                budget()
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def unique(rows):
        result = {}
        for name, value in rows:
            require(name not in result, 'Duplicate JSON key')
            result[name] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


DESIGN = {'format': 'codeformer-role-preserving-training-design-v1',
    'training': TRAINING, 'objective': OBJECTIVE, 'evaluation': EVALUATION,
    'gradient_route': GRADIENT_ROUTE, 'initializer_parity': INITIALIZER_PARITY,
    'perceptual_prerequisite': PERCEPTUAL_PREREQUISITE, 'state_policy': STATE_POLICY,
    'counts': COUNTS, 'neural_forwards': NEURAL_FORWARD_CAP,
    'operation_reservations': RESERVATION_CAP, 'limits': LIMITS}
DESIGN_SHA256 = canonical_sha256(DESIGN)


def equal(a, b):
    return canonical_json(a) == canonical_json(b)


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse path forbidden')
            if stat.S_ISREG(info.st_mode):
                require(info.st_nlink == 1, 'Hardlinked files forbidden')
    return path


def safe_relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and '\x00' not in name,
            'Portable relative path required')
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and parts.as_posix() == name and all(
        p not in ('', '.', '..') and not p.endswith((' ', '.')) and
        not re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', p) for p in parts.parts),
        'Unsafe relative path')
    return parts


def safe_path(base, name, must_exist=True):
    base = no_links(base).resolve(strict=True)
    path = no_links(base.joinpath(*safe_relative(name).parts))
    require(path.resolve(strict=must_exist).is_relative_to(base), 'Path escapes declared root')
    if must_exist:
        require(stat.S_ISREG(path.lstat().st_mode), 'Exact regular file required')
    return path


def write_new_json(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def manual_root(root):
    require(platform.system() == 'Linux', 'Manual NVIDIA L4 VM only; local neural calls forbidden')
    root = no_links(root).resolve(strict=True)
    require(root == no_links(Path.home()/'forensic-dgp'/VM_NAME).resolve(strict=False), 'Unexpected manual VM root')
    require(bool(os.environ.get('TMUX')), 'Manual tmux required before framework import')
    return root


def resolve_parent(root):
    root = no_links(root).resolve(strict=True)
    parent = no_links(root.parent/PARENT_NAME).resolve(strict=True)
    require(parent.is_dir() and parent.parent == root.parent and parent != root, 'Exact read-only AdaIN sibling required')
    return parent


def learning_rate(update, base, training=None):
    t = TRAINING if training is None else training
    require(type(update) is int and 1 <= update <= t['max_updates'] == 50, 'New schedule limited to1..50')
    require(type(base) is float and base in (t['encoder_token_lr'], t['fusion_lr']), 'Declared base rate required')
    return base*update/t['warmup_updates']


def role_term_names(role, image_eligible):
    require(role in ('normal', 'dark') and type(image_eligible) is bool, 'Explicit role/eligibility required')
    return NORMAL_TERMS if role == 'normal' else (DARK_TERMS if image_eligible else tuple(
        n for n in DARK_TERMS if n not in ('pixel_l1_rgb', 'lpips_vgg')))


def verify_binding_map(mapping, label):
    require(type(mapping) is dict and mapping, 'Nonempty exact binding map required: '+label)
    require(len({x.casefold() for x in mapping}) == len(mapping), 'Case-colliding bindings')
    for name, pin in mapping.items():
        safe_relative(name)
        require(type(pin) is dict and set(pin) == {'sha256', 'bytes'} and
            isinstance(pin['sha256'], str) and HEX.fullmatch(pin['sha256']) and
            type(pin['bytes']) is int and pin['bytes'] >= 0, 'Invalid hash/size binding: '+label)
    return mapping


def expected_extra_returned_paths(reference_ids):
    require(type(reference_ids) is list and len(reference_ids) == len(set(reference_ids)) == 100 and
        reference_ids == sorted(reference_ids) and all(REFERENCE_ID.fullmatch(x) for x in reference_ids),
        'Exactly100 selected TRAIN photograph IDs required')
    return set(FIXED_RETURNED_DEPENDENCIES) | {
        'outputs/adain_gradient_isolation_v1/HQ_cache/'+rid+'/'+name for rid in reference_ids
        for name in ('indices.npy', 'features.npy', 'record.json')}


def verify_protocol(p):
    require(type(p) is dict and p.get('format') == FORMAT and p.get('module') == MODULE and
        p.get('vm_name') == VM_NAME and p.get('outputs') == OUTPUTS and p.get('export_prefix') == PREFIX,
        'Exact new role-preserving namespace required')
    require(p.get('parent_dir') == PARENT_NAME and p.get('parent_protocol_sha256') == PARENT_PROTOCOL_SHA,
        'Exact read-only AdaIN parent required')
    for name, expected in {'training': TRAINING, 'objective': OBJECTIVE, 'evaluation': EVALUATION,
            'gradient_route': GRADIENT_ROUTE, 'initializer_parity': INITIALIZER_PARITY,
            'perceptual_prerequisite': PERCEPTUAL_PREREQUISITE, 'assets': ASSETS,
            'model_digests': MODEL_DIGESTS, 'call_caps': COUNTS, 'limits': LIMITS,
            'design': DESIGN, 'state_policy': STATE_POLICY,
            **PARENT_CONTROLS}.items():
        require(equal(p.get(name), expected), 'Prospective frozen field differs: '+name)
    require(p.get('design_sha256') == DESIGN_SHA256, 'Exact canonical embedded design required')
    require(equal(p['call_caps'], expected_call_caps(p['training'])), 'Source-derived counts differ')
    require(p.get('status') == 'PREPARED_UNRUN' and p.get('manual_tmux_required') is True and
        p.get('new_training_recipe') is True and p.get('actual_candidate_checkpoints_required') is True,
        'Manual new unrun checkpoint-producing pilot required')
    for name in ('automatic_training', 'automatic_VM_launch', 'automatic_resume', 'model_qualified',
            'automatic_app_promotion', 'quality_gates_changed', 'FINAL_used', 'old_optimizer_loaded',
            'historical_RNG_loaded', 'cached_gradient_reuse', 'tone_head_used'):
        require(p.get(name) is False, 'Prohibited protocol flag: '+name)
    require(p.get('data_manifest') == 'references.json' and p.get('comparison_cases') == 'comparison_cases.json'
        and p.get('data_bundle_file') == 'data_bundle.json', 'Exact data document names required')
    old = p.get('historical_parent_protocol')
    require(type(old) is dict and canonical_sha256(old) == PARENT_PROTOCOL_CANONICAL_SHA,
        'Exact historical protocol is provenance, not this new optimizer recipe')
    catalog = verify_binding_map(p.get('historical_dependency_bindings'), 'historical metadata catalog')
    require(len(catalog) == 2011 and canonical_sha256(catalog) == DEPENDENCIES_SHA256,
        'Exact2011 catalog metadata required; unused physical files need not be included')
    deps = verify_binding_map(p.get('dependencies'), 'minimal read-only physical dependencies')
    require(set(deps) < set(catalog) and all(deps[n] == catalog[n] for n in deps),
        'Physical dependencies must be exact minimal historical catalog subset')
    require(deps.get('protocol.json', {}).get('sha256') == PARENT_PROTOCOL_SHA and
        set(HISTORICAL_RUNTIME_FILES) | {'references.json', 'comparison_cases.json', 'perceptual/manifest.json',
        'training_parity_v1.json', 'hq_cache_parity_v1.json', 'validation_saved_v1.json'} <= set(deps),
        'Required exact historical source/data closure missing')
    for pin in ASSETS.values():
        require(deps.get(pin['path']) == {k: pin[k] for k in ('sha256', 'bytes')}, 'Unbound named asset')
    for pin in PARENT_CONTROLS.values():
        require(deps.get(pin.get('path', pin.get('manifest')), {}).get('sha256') == pin['sha256'],
            'Historical data provenance differs')
    retained = verify_binding_map(p.get('retained_output_dependencies'), 'historical RAW/PNG witnesses')
    require(len(retained) == 1624 and canonical_sha256(retained) == RETAINED_OUTPUTS_SHA256,
        'Exact1624 original/base RAW/PNG witnesses required; metrics are separate bindings')
    extra = verify_binding_map(p.get('extra_returned_dependencies'), 'selected HQ/runtime/metrics witnesses')
    require(set(extra) == expected_extra_returned_paths(p.get('selected_train_reference_ids')),
        'Exact100HQ triplets plus fixed returned metadata required')
    require(all(extra.get(n) == pin for n, pin in FIXED_RETURNED_DEPENDENCIES.items()),
        'Returned runtime/metrics/manifest changed')
    require(not set(extra) & set(retained) and not set(extra) & set(deps) and not set(retained) & set(deps),
        'Read-only dependency namespaces collide')
    files, sizes = p.get('files_sha256'), p.get('files_bytes')
    require(type(files) is dict and type(sizes) is dict and set(files) == set(sizes) and
        REQUIRED_SOURCE_FILES | {'data_bundle.json'} <= set(files) and 'protocol.json' not in files,
        'Exact executing source and neutral data index required')
    verify_binding_map({n: {'sha256': h, 'bytes': sizes[n]} for n, h in files.items()}, 'new static sources')
    protocol_bytes = len((json.dumps(p, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    require(len(files)+1 <= LIMITS['source_packet_files'] and
        sum(sizes.values())+protocol_bytes <= LIMITS['source_packet_bytes'],
        'Source packet cap exceeded')
    require(all(Path(n).suffix in ('.py', '.json', '.md', '.txt') for n in files), 'Static binary/checkpoint forbidden')
    require(files['research/__init__.py'] == hashlib.sha256(b'').hexdigest() and sizes['research/__init__.py'] == 0,
        'Intentional empty transport research namespace required')
    document = p.get('design_document', {})
    require(document.get('path') == DESIGN_DOCUMENT and
        document.get('sha256') == files[DESIGN_DOCUMENT] == DESIGN_DOCUMENT_SHA256,
        'Hash-bound new prospective design required')
    return p
