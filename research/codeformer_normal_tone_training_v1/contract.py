"""Stdlib-only bounded50-update training contract; importing performs no training."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

FORMAT='codeformer-normal-tone-training-finite-manual-v1'
VM_NAME='cctv_codeformer_normal_tone_training_v1_vm'
PARENT_NAME='cctv_codeformer_adain_gradient_isolation_v1_vm'
PARENT_PROTOCOL_SHA='4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
DEPENDENCIES_SHA256='0cf7922ff74d9a01139d943b9f27b3b15a1d6b0e0d3e5bbc6d668c77fcaff8b9'
OUTPUTS='outputs/normal_tone_training_v1'
PREFIX='cctv-codeformer-normal-tone-training-v1'
TRAINING={'max_updates': 50, 'schedule_horizon_updates': 600, 'data_horizon_updates': 600, 'microbatch': 1, 'accumulate': 4, 'snapshots': [0, 50], 'benchmark_updates': 20, 'optimizer': 'Adam', 'betas': [0.9, 0.99], 'eps': 1e-08, 'weight_decay': 0.0, 'encoder_token_lr': 1e-05, 'fusion_lr': 2e-05, 'warmup_updates': 50, 'cosine_end_lr': 2e-06, 'autocast': 'disabled-FP32', 'grad_scaler': False, 'gradient_clipping': None, 'GAN': False, 'EMA': False, 'resume': False, 'max_gradient_queries': 9, 'max_backward_calls': 200, 'loss_weights': {'token_ce': 0.5, 'latent_mse': 1.0, 'pixel_l1_rgb': 1.0, 'lpips_vgg': 1.0, 'coarse_rgb_l1': 4.0, 'behavior_rgb_l1': 2.0, 'behavior_token_kl': 0.1}, 'pair_policy': 'paired-linear-light-keyed-fields-v2', 'severe_edges': [24, 28, 32, 40], 'control_variants': ['clear256', 'clear128', 'clear64', 'light256', 'light128', 'light64'], 'cycle': 'u=update-1; Aedge=edges[u%4]; Bvariant=variants[(u//4)%6]', 'behavior_slot': 3, 'distillation_temperature': 2.0, 'call_caps': {'optimizer_step_attempts': 50, 'backward_calls': 200, 'gradient_queries': 9, 'route_witness_forwards': 4, 'route_witness_perceptual_forwards': 4, 'perceptual_prerequisite_forwards': 1, 'candidate_training_forwards': 200, 'perceptual_training_forwards': 148, 'behavior_teacher_forwards': 50, 'teacher_forwards': 391, 'snapshot_forwards': 812, 'baseline_forwards': 2, 'initializer_parity_candidate_forwards': 2, 'training_input_writes': 200}}
EVALUATION={'validation_references': 53, 'profiles': ['clear', 'light', 'blur', 'jpeg', 'lowlight', 'compound'], 'comparison_cases': 88, 'native_unpaired': 48, 'old_paired': 40, 'clear_views': ['raw', 'native_png'], 'mean_clear_mae_delta_stop': 0.01, 'worst_per_case_clear_mae_delta_stop': 0.03, 'normal_blur_cases': 53, 'mean_blur_photometric_delta_stop': 0.01, 'worst_per_case_blur_photometric_delta_stop': 0.03, 'blur_photometric_metrics': ['luminance_bias', 'coarse_rgb_mae'], 'quality_guard_is_historical_one_percent_gate': False, 'baseline': 'same_VM_update0_FP32', 'fidelity_weight': 1.0, 'adain': True, 'alignment': 'none', 'display_processing': 'none', 'delivered_pixels': 256, 'internal_pixels': 512, 'validation_target_policy': 'retained direct RGB1024-to-Pillow-LANCZOS256'}
GRADIENT_ROUTE={'policy': 'candidate-AdaIN-style-detach-only-v1', 'witness_update': 3, 'witness_slots': [1, 2], 'forwards': 4, 'gradient_queries': 8, 'before_Adam_construction': True, 'forward_equality': 'bit-exact', 'CPU_CUDA_RNG_restored': True}
PARENT_CONTROLS={'validation_saved': {'manifest': 'validation_saved_v1.json', 'sha256': 'ff76007be1f55e70e12e5d5c3aad00c53e4564326e6d10eeb1e12a38b9178d84', 'source_protocol_sha256': 'e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e', 'case_count': 318}, 'prior_baseline': {'metrics': 'provenance/v1/snapshot_0000_metrics.json', 'sha256': 'bc70c534883931b3676b68839c858b9e27def0c994156299e85cd2ade62abda9', 'source_protocol_sha256': 'e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e'}, 'training_parity': {'path': 'training_parity_v1.json', 'sha256': '743c073b92dd1bcfe37210a854277fb31b5e6dbe562d1229fa05903b41c5c6c6', 'cases': 200, 'updates': 50, 'behavior_anchors': 50, 'source_protocol_sha256': '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'}, 'hq_cache_parity': {'path': 'hq_cache_parity_v1.json', 'sha256': 'f21dbe09e4b7af1b7d6df5ebd45c2c7230ed10face3e33a33a4c843949fd75ac', 'cases': 391, 'source_protocol_sha256': '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'}}
ASSETS={'initializer': {'path': 'weights/codeformer.pth', 'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'bytes': 376637898}, 'teacher': {'path': 'weights/vqgan_code1024.pth', 'sha256': '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f', 'bytes': 255078528}, 'perceptual_features': {'path': 'perceptual/vgg16_features.pth', 'sha256': 'a5b412426cfb06d2196179871dff81a905f97688cd46f2f16df1b63d5b251279', 'bytes': 58867573}, 'perceptual_calibration': {'path': 'perceptual/lpips_vgg_v01.pth', 'sha256': 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868', 'bytes': 7289}}
LIMITS={'initialization_seconds': 600, 'witness_seconds': 180, 'benchmark_seconds': 600, 'projected_training_seconds': 900, 'training_seconds': 900, 'snapshot_seconds': 600, 'worker_seconds': 3000, 'export_seconds': 600, 'initial_free_bytes': 17179869184, 'reserve_free_bytes': 4294967296, 'result_payload_bytes': 3758096384, 'archive_bytes': 3865470566, 'checkpoint_temporary_bytes': 1073741824, 'PNG_file_bytes': 198656, 'metadata_bytes': 134217728, 'training_history_line_bytes': 65536, 'CUDA_reserved_bytes': 23622320128, 'preflight_seconds': 300, 'RSS_bytes': 8589934592, 'regular_file_cap': 7000, 'log_bytes': 2097152, 'array_file_bytes': 268435456}
TRAINING['max_gradient_queries']=11
TRAINING['call_caps']['gradient_queries']=11
COUNTS=TRAINING['call_caps']
NORMAL_FORMULA_PREREQUISITE={'shape':[1,3,2,2],'gradient_queries':2,
    'before_Adam_construction':True,'same_helper_as_training':True,'synthetic_only':True,
    'neural_forwards':0,'absolute_tolerance':2e-7,'relative_tolerance':2e-5,
    'bound':'max_abs_error <= atol + rtol * max_abs_expected_analytic_gradient',
    'RNG_restored':True,'positive_tone_direction_and_chroma_retention_required':True}
OBJECTIVE={'prediction_RGB':'(native.float()+1)/2; finite unclipped native512',
'grid':'avg_pool2d kernel32 stride32 ->16x16','original_coarse':'mean(abs(P(rgb)-P(clean512)))',
'luminance_weights':[.2126,.7152,.0722],
'normal_centered':'mean(abs(e-mean_spatial(sum_RGB(luminance_weights*e))))',
'mean_scope':'one differentiable scalar per observation, broadcast equally to all3RGB channels',
'normal_slots':[1,3],'dark_slots':[2,4],'dark_coarse_unchanged':True,'coarse_weight':4.0,
'gradient_accumulation_divisor':4,'fresh_full_objective_every_observation':True,
'cached_gradients_used':False,'inference_output_centering':False,'brightness_gate':False,
'new_loss_weight_grid':False,'fresh_initializer_no_optimizer_resume':True}


def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(1024**2), b''):
            if budget is not None:
                budget()
            digest.update(payload)
    return digest.hexdigest()

def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def no_links(path):
    path = Path(path).absolute()
    for part in [path] + list(path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 1024),
                    'Linked/reparse path forbidden: ' + str(part))
    return path

def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value and '\x00' not in value,
            'Portable relative path required')
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and relative.as_posix() == value and
            all(part not in ('', '.', '..') and not part.endswith((' ', '.')) and
                not re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part)
                for part in relative.parts), 'Unsafe relative path')
    return relative

def safe_path(base, relative, must_exist=True):
    base = no_links(base).resolve(strict=True)
    parts = safe_relative(relative).parts
    path = no_links(base.joinpath(*parts))
    require(path.resolve(strict=must_exist).is_relative_to(base), 'Path escapes declared root')
    if must_exist:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode), 'Exact regular file required')
    return path

def write_new_json(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())

def manual_root(root):
    require(platform.system() == 'Linux', 'Manual NVIDIA L4 VM execution only; local neural calls forbidden')
    root = no_links(root).resolve(strict=True)
    expected = no_links(Path.home() / 'forensic-dgp' / VM_NAME).resolve(strict=False)
    require(root == expected, 'Unexpected manual VM root')
    return root

def resolve_parent(root):
    root=no_links(root).resolve(strict=True)
    parent=no_links(root.parent/PARENT_NAME).resolve(strict=True)
    require(parent.is_dir() and parent.parent==root.parent and parent!=root,'Exact read-only AdaIN sibling required')
    return parent


def equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False,separators=(',',':'))==json.dumps(b,sort_keys=True,allow_nan=False,separators=(',',':'))


def expected_call_caps(training=None):
    t=TRAINING if training is None else training
    maximum=t['max_updates']
    eligible=sum(2+(2 if ((update-1)%4)>=2 else 0) for update in range(1,maximum+1))
    return {'optimizer_step_attempts':maximum,'backward_calls':maximum*4,'gradient_queries':11,
        'route_witness_forwards':4,'route_witness_perceptual_forwards':4,'perceptual_prerequisite_forwards':1,
        'candidate_training_forwards':maximum*4,'perceptual_training_forwards':eligible,
        'behavior_teacher_forwards':maximum,'teacher_forwards':391,
        'snapshot_forwards':len(t['snapshots'])*406,'baseline_forwards':2,
        'initializer_parity_candidate_forwards':2,'training_input_writes':maximum*4}


def verify_protocol(p):
    require(p.get('format')==FORMAT and type(p.get('seed')) is int and p['seed']==20261015,
            'Exact training format/seed required')
    require(p.get('parent_dir')==PARENT_NAME and p.get('parent_protocol_sha256')==PARENT_PROTOCOL_SHA and
            p.get('outputs')==OUTPUTS and p.get('export_prefix')==PREFIX,'Exact newroot/readonlyparent namespace required')
    for key,value in [('training',TRAINING),('evaluation',EVALUATION),('gradient_route',GRADIENT_ROUTE),
                      ('limits',LIMITS),('objective',OBJECTIVE),*PARENT_CONTROLS.items(),('assets',ASSETS),
                      ('normal_formula_prerequisite',NORMAL_FORMULA_PREREQUISITE)]:
        require(equal(p.get(key),value),'Frozen training/evaluation/resource control differs: '+key)
    require(equal(p['training']['call_caps'],expected_call_caps(p['training'])),'Source-derived call caps differ')
    require(p.get('data_manifest')=='references.json' and p.get('comparison_cases')=='comparison_cases.json',
            'Canonical data manifests required')
    require(p.get('status')=='PREPARED_UNRUN' and p.get('manual_tmux_required') is True and
            p.get('new_training_recipe') is True and p.get('actual_candidate_checkpoints_required') is True,
            'Manual unrun checkpoint-producing pilot required')
    for key in ('model_qualified','automatic_app_promotion','automatic_training','quality_gates_changed',
                'resume_authorized','FINAL_used','cached_gradient_reuse'):
        require(p.get(key) is False,'No promotion, continuation or FINAL use authorized: '+key)
    files,sizes=p.get('files_sha256'),p.get('files_bytes')
    require(isinstance(files,dict) and isinstance(sizes,dict) and set(files)==set(sizes) and
            'protocol.json' not in files and files,'Exact own-source manifest required')
    required={'research/codeformer_normal_tone_training_v1/'+name for name in
              ('__init__.py','contract.py','transport.py','worker.py','model.py')}
    required.update({'research/__init__.py','scripts/supervise_codeformer_normal_tone_training_v1.py'})
    require(required<=set(files),'Every executing source module and supervisor must be bound')
    for name,value in files.items():
        safe_relative(name)
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and
                type(sizes[name]) is int and sizes[name]>=0,'Invalid source binding')
    deps=p.get('dependencies')
    require(isinstance(deps,dict) and len(deps)==2011 and 'protocol.json' in deps,
            'Exact2011retained AdaIN dependencies required')
    require(hashlib.sha256(json.dumps(deps,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()==DEPENDENCIES_SHA256,
            'Exact previously audited2011dependency manifest required')
    for name,value in deps.items():
        safe_relative(name)
        require(isinstance(value,dict) and set(value)=={'sha256','bytes'} and
                isinstance(value['sha256'],str) and re.fullmatch('[0-9a-f]{64}',value['sha256']) and
                type(value['bytes']) is int and value['bytes']>=0,'Invalid read-only dependency binding')
    require(deps['protocol.json']['sha256']==PARENT_PROTOCOL_SHA,'Exact AdaIN parent protocol required')
    for value in ASSETS.values():
        require(deps.get(value['path'])=={k:value[k] for k in ('sha256','bytes')},'Asset dependency differs')
    for key,row in PARENT_CONTROLS.items():
        name=row.get('path',row.get('manifest',row.get('metrics')))
        require(deps.get(name,{}).get('sha256')==row['sha256'],'Parent provenance/data join differs: '+key)
    require(all(name in deps for name in (p['data_manifest'],p['comparison_cases'])),'Data manifest dependencies required')
    design=p.get('design',{})
    require(design.get('path')=='instructions/CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md' and
            files.get(design.get('path'))==design.get('sha256'),'Hash-bound prospective training design required')
    return p

CHECKPOINT_PATHS=[OUTPUTS+'/snapshot_0000/checkpoint.pt',OUTPUTS+'/snapshot_0050/checkpoint.pt',OUTPUTS+'/stopped_checkpoint.pt']
