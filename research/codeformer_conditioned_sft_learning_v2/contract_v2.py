"""Pure Conditioned SFT V2 contract, case selection, and Two-Phase L4 training specification.

Importing this module loads zero neural frameworks or image decoders.
"""
import copy
import hashlib
import json
import math

from . import architecture_v2 as arch_v2
from . import conditioning_v2 as cond_v2
from . import objective_v2 as obj_v2

STATUS = 'CONDITIONED_SFT_V2_SOURCE_CONTRACT_NOT_A_SEALED_EXECUTION_PACKET'
STAGE_NAME = 'cctv_codeformer_conditioned_sft_learning_v2_vm'
OUTPUT_RELATIVE = 'outputs/conditioned_sft_learning_v2'
PREFIX = 'cctv-codeformer-conditioned-sft-learning-v2'
PACKAGE = 'research/codeformer_conditioned_sft_learning_v2'
FORMAT = 'conditioned-sft-v2-top-down-dilated-pyramid-freq-head-manual-L4-v1'

EVIDENCE_PINS = {
    'v2_manifest': ('evidence/v2_data_manifest.json', 430694,
                    'f060229847a70b80e8a833a49f4a0764b2e813caad58a991ed2d98d225793a15'),
    'v1_protocol': ('evidence/v1_protocol.json', 1248212,
                    'f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0'),
    'retained_continuation': ('evidence/retained_continuation_protocol.json', 279879,
                              '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'),
    'u64_parent_protocol': ('evidence/u64_parent_protocol.json', 197079,
                            'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'),
    'u64_historical_pixel_audit': ('evidence/u64_historical_pixel_audit.json', 2677742,
                                   '8b73cbb901452faff8b3679bcf9d077d4d8c98a90e461c18af260565fd04c18d'),
    'feasibility_v2_receipt': ('evidence/v2_feasibility_receipt_v1.json', 12509,
                               '005cb11a9a893ab0856a42e1254098cdcc71f70a37a1f24228e7bd1f500dc1a4'),
}

NATIVE_CCTV_DEV_CASES = (
    {
        'id': 'ChokePoint_C1_01',
        'dataset': 'ChokePoint',
        'input': {'path': 'data/native_cctv/choke_dev_01_t033.png', 'bytes': 46552,
                  'sha256': '81f89c45975256b9a0fd0a27a313e03229cd72ad893e6ae1636a816ccb223f35'},
    },
    {
        'id': 'ChokePoint_C1_03',
        'dataset': 'ChokePoint',
        'input': {'path': 'data/native_cctv/choke_dev_03_t033.png', 'bytes': 54635,
                  'sha256': '6ac64ca690e383f0c6d9ba00a3be4b70c3b73fb14b8d788073a17253c52053c1'},
    },
    {
        'id': 'QMUL_SurvFace_24to39',
        'dataset': 'QMUL-SurvFace',
        'input': {'path': 'data/native_cctv/dev_24to39_01_input.png', 'bytes': 21636,
                  'sha256': 'abaf47c7d2ed6f614fae08a7352d7cf698359fd1935a1a816e972e738cdf2816'},
    },
    {
        'id': 'QMUL_SurvFace_ge40',
        'dataset': 'QMUL-SurvFace',
        'input': {'path': 'data/native_cctv/dev_ge40_01_input.png', 'bytes': 11908,
                  'sha256': 'dceea504203ec1a041481533c8b0c658a51c6134886fb02afa8dcaccc7c24e8d'},
    },
)

PHASE1_SCHEDULE = {
    'epochs': 40,
    'identities_per_batch': 2,
    'active_roles_per_identity': 5,
    'batch_size': 10,
    'cases_per_epoch': 280,
    'steps_per_epoch': 28,
    'total_optimizer_updates': 1120,
    'optimizer': 'AdamW',
    'lr_max': 5e-4,
    'lr_min': 2e-5,
    'warmup_epochs': 2,
    'betas': [0.9, 0.99],
    'weight_decay': 1e-4,
    'grad_clip_norm': 1.0,
    'trainable_submodules': ['pyramid', 'dark_refiner'],
    'feature_caching': 'zero_cpu_cache_on_the_fly_batched_gpu',
}

PHASE2_SCHEDULE = {
    'identity_passes': 4,
    'identities_per_pass': 56,
    'total_optimizer_updates': 224,
    'roles_per_update': 6,
    'total_role_backwards': 1344,
    'optimizer': 'AdamW',
    'lr_max': 8e-5,
    'lr_min': 1e-5,
    'warmup_updates': 12,
    'betas': [0.9, 0.99],
    'weight_decay': 5e-5,
    'grad_clip_norm': 1.0,
    'trainable_submodules': ['pyramid', 'dark_refiner', 'head_v2'],
    'snapshots': [56, 112, 168, 224],
    'early_quality_abort': False,
    'best_checkpoint_criterion': 'max_monitor_v2_degraded_ssim7_subject_to_clear_and_dark_nonregression',
}

CHECKPOINT_TAGS = (
    'u0000_init',
    'p1_e040',
    'p2_u0056',
    'p2_u0112',
    'p2_u0168',
    'p2_u0224',
    'best_degraded_ssim',
)

LIMITS = {
    'worker_seconds': 7200,
    'external_seconds': 7260,
    'export_seconds': 600,
    'export_external_seconds': 660,
    'termination_grace_seconds': 20,
    'initial_free_disk_bytes': 24 * 1024**3,
    'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 12 * 1024**3,
    'maximum_CUDA_reserved_bytes': 20 * 1024**3,
    'maximum_output_bytes': 12 * 1024**3,
    'maximum_archive_bytes': 12 * 1024**3,
    'automatic_retry_resume': 0,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json_bytes(raw: bytes, *, expected_pin=None):
    need(type(raw) is bytes and 0 < len(raw) <= 32 * 1024**2, 'Bounded JSON bytes required')
    if expected_pin is not None:
        _, exp_bytes, exp_sha = expected_pin
        need(len(raw) == exp_bytes and hashlib.sha256(raw).hexdigest() == exp_sha,
             'Exact pinned JSON bytes required')

    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, 'Duplicate JSON key refused: ' + str(k))
            out[k] = v
        return out

    def reject(token):
        raise ValueError('Nonfinite JSON token refused: ' + str(token))

    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject)

    def walk(node, depth=0):
        need(depth <= 64, 'JSON nesting cap exceeded')
        if type(node) is float:
            need(math.isfinite(node), 'Nonfinite float refused')
        elif type(node) is dict:
            for child in node.values():
                walk(child, depth + 1)
        elif type(node) is list:
            for child in node:
                walk(child, depth + 1)

    walk(value)
    return value


def _remap_v2_asset(binding, expected_prefix: str, staged_prefix: str):
    need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact binding dict required')
    path = binding['path']
    need(path.startswith(expected_prefix), 'Unexpected asset prefix: ' + path)
    suffix = path[len(expected_prefix):]
    return {'path': staged_prefix + suffix, 'bytes': int(binding['bytes']), 'sha256': str(binding['sha256'])}


def select_v2_cohorts(v2_manifest_raw: bytes, v1_protocol_raw: bytes):
    """Derive deterministic staged bindings for 336 V2 opt, 48 V2 monitor, 40 V1 monitor, 56 old_quality, and 4 native CCTV."""
    m2 = strict_json_bytes(v2_manifest_raw, expected_pin=EVIDENCE_PINS['v2_manifest'])
    p1 = strict_json_bytes(v1_protocol_raw, expected_pin=EVIDENCE_PINS['v1_protocol'])

    need(m2.get('complete') is True and m2.get('zero_monitor_overlap_verified') is True,
         'Verified V2 data manifest required')
    opt_refs = m2['optimization_reference_order']
    mon_refs = m2['monitor_reference_order']
    need(len(opt_refs) == 56 and len(mon_refs) == 8 and not (set(opt_refs) & set(mon_refs)),
         'Exact disjoint 56 optimization / 8 monitor identities required')

    prep_prefix = 'outputs/codeformer_conditioned_sft_learning_v2_preparation/'
    v2_opt = []
    for row in m2['optimization_cases']:
        need(row['cohort'] == 'optimize' and row['optimization_permitted'] is True
             and row['role'] in obj_v2.ROLES_V2 and row['reference_id'] in opt_refs,
             'Invalid V2 optimization row: ' + str(row.get('id')))
        v2_opt.append({
            'id': row['id'],
            'reference_id': row['reference_id'],
            'role': row['role'],
            'split': 'TRAIN',
            'cohort': 'optimize_v2',
            'optimization_permitted': True,
            'input': _remap_v2_asset(row['input'], prep_prefix, 'data/v2/'),
            'target': _remap_v2_asset(row['target'], prep_prefix, 'data/v2/'),
            'input_RGB8_sha256': row['input_RGB8_sha256'],
            'target_RGB8_sha256': row['target_RGB8_sha256'],
        })

    v2_mon = []
    for row in m2['monitor_v2_cases']:
        need(row['cohort'] == 'monitor_v2' and row['optimization_permitted'] is False
             and row['role'] in obj_v2.ROLES_V2 and row['reference_id'] in mon_refs,
             'Invalid V2 monitor row: ' + str(row.get('id')))
        v2_mon.append({
            'id': row['id'],
            'reference_id': row['reference_id'],
            'role': row['role'],
            'split': 'TRAIN',
            'cohort': 'monitor_v2',
            'optimization_permitted': False,
            'input': _remap_v2_asset(row['input'], prep_prefix, 'data/v2/'),
            'target': _remap_v2_asset(row['target'], prep_prefix, 'data/v2/'),
            'input_RGB8_sha256': row['input_RGB8_sha256'],
            'target_RGB8_sha256': row['target_RGB8_sha256'],
        })

    need(len(v2_opt) == 336 and len(v2_mon) == 48, 'Exact 336 V2 opt and 48 V2 monitor rows required')

    v1_sel = p1['selection']
    old_quality = copy.deepcopy(v1_sel['old_quality'])
    v1_monitors = [
        {
            'id': r['id'],
            'reference_id': r['reference_id'],
            'role': r['role'],
            'split': r['split'],
            'cohort': r['cohort'],
            'optimization_permitted': r['optimization_permitted'],
            'input': copy.deepcopy(r['input']),
            'target': copy.deepcopy(r['target']),
            'input_RGB8_sha256': r['prepared_input_RGB8_sha256'],
            'target_RGB8_sha256': r['prepared_target_RGB8_sha256'],
        }
        for r in v1_sel['new_monitors']
    ]
    need(len(old_quality) == 56 and len(v1_monitors) == 40, 'Exact 56 old_quality and 40 V1 monitors required')

    old_guard_refs = {r['reference_id'] for r in v1_sel['old_guards']}
    need(not (old_guard_refs & set(opt_refs)), 'Legacy guard photographs must stay disjoint from 56 V2 opt identities')

    return {
        'optimization_reference_order': list(opt_refs),
        'monitor_reference_order': list(mon_refs),
        'v2_optimization': v2_opt,
        'v2_monitors': v2_mon,
        'v1_monitors': v1_monitors,
        'old_quality': old_quality,
        'native_cctv_dev': copy.deepcopy(list(NATIVE_CCTV_DEV_CASES)),
    }


def prospective_contract_v2():
    return {
        'status': STATUS,
        'stage_name': STAGE_NAME,
        'output_relative': OUTPUT_RELATIVE,
        'prefix': PREFIX,
        'format': FORMAT,
        'architecture': arch_v2.architecture_v2_contract(),
        'conditioning': cond_v2.conditioning_v2_contract(),
        'objective': obj_v2.objective_v2_contract(),
        'phase1_schedule': copy.deepcopy(PHASE1_SCHEDULE),
        'phase2_schedule': copy.deepcopy(PHASE2_SCHEDULE),
        'checkpoint_tags': list(CHECKPOINT_TAGS),
        'limits': copy.deepcopy(LIMITS),
    }
