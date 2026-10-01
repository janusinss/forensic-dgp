"""Verify a read-only source forward and package new reflection-pilot files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_face_occlusion_vm import build_bundle, sha
from scripts.package_face_occlusion_focus_vm import validate_new_members
from scripts.train_reflection_coverage_vm import (
    DATA_PATH, DATA_SHA, PIXELS_SHA, AUDIT_SHA, INVENTORY_PATH,
    MODEL_SHA, OPTIMIZER_SHA, PRIOR_INVENTORIES, supplemental_ids,
)

CHECK_PATH = 'outputs/reflection_coverage_bundle_v1/source_check.json'
CODE_PATHS = (
    'scripts/package_reflection_coverage_vm.py',
    'scripts/train_reflection_coverage_vm.py',
    'scripts/audit_reflection_coverage_data.py',
    'reflection_coverage.py', 'completion_data_v2.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_members(files, prior_inventories):
    validate_new_members(files, prior_inventories)


def verify_source_check(check, model_sha, optimizer_sha, data_sha):
    require(check.get('model_sha256') == model_sha and
            check.get('optimizer_sha256') == optimizer_sha and
            check.get('data_sha256') == data_sha, 'Source forward input bindings differ')
    require(check.get('model_state_unchanged') is True and
            check.get('optimizer_constructed') is False and
            type(check.get('optimizer_updates_locally')) is int and
            check['optimizer_updates_locally'] == 0 and
            type(check.get('moment_step')) is int and check['moment_step'] == 420,
            'Require unchanged source, exact moments and no local optimizer')
    terms = check.get('terms')
    require(isinstance(terms, dict) and
            {'core', 'control', 'reflective', 'padded_probe'}.issubset(terms) and
            all(type(v) in (int, float) and math.isfinite(v) for v in terms.values()),
            'Require measured finite core/control/reflection/padded-support terms')


def verify_data():
    folder = ROOT / DATA_PATH
    require(sha(folder / 'manifest.json') == DATA_SHA and
            sha(folder / 'pixels.pth') == PIXELS_SHA and
            sha(folder / 'audit.json') == AUDIT_SHA, 'Frozen supplemental files changed')
    data = read_json(folder / 'manifest.json')
    audit = read_json(folder / 'audit.json')
    visual = read_json(folder / 'visual_review.json')
    require(audit.get('complete') is True and audit['manifest_sha256'] == DATA_SHA and
            audit['pixel_cache_sha256'] == PIXELS_SHA and
            audit['script_sha256'] == sha(ROOT / 'scripts/audit_reflection_coverage_data.py') and
            audit['optimizer_updates_locally'] == 0, 'Independent frozen data audit differs')
    require(visual.get('accepted_for_bounded_pilot') is True and
            visual['manifest_sha256'] == DATA_SHA and visual['audit_sha256'] == AUDIT_SHA and
            visual['optimizer_updates_locally'] == 0 and
            visual['training_output_quality_verified'] is False, 'Bounded visual review differs')
    for name, digest in {**data['input_hashes'], **data['code_hashes']}.items():
        require(sha(ROOT / name) == digest, 'Registered data input changed: ' + name)
    return data


def verify_prior():
    inventories = []
    for name, digest in PRIOR_INVENTORIES.items():
        require(sha(ROOT / name) == digest, 'Previous inventory changed: ' + name)
        inventory = read_json(ROOT / name)
        for member, expected in inventory.items():
            require(sha(ROOT / member) == expected, 'Previous executed input changed: ' + member)
        inventories.append(inventory)
    return inventories


def prepare():
    import torch
    from face_occlusion_adapter import load_adapter, parameter_groups
    from face_occlusion_focus import validate_source, validate_optimizer
    from completion_inference import load_completion
    from detector_training import load_manifest, ReviewedMasks, segmentation_loss
    from detector_replay import replay_consistency
    from reflection_coverage import supported_segmentation_loss
    from scripts.train_face_occlusion_vm import dependency_versions, validate_schedule, REPLAY_SHA
    from scripts.train_coverage_vm import PARENT_SHA

    destination = ROOT / CHECK_PATH
    require(not destination.exists(), 'Preserve the existing source check; use a new version for changes')
    data = verify_data()
    verify_prior()
    source_root = ROOT / 'outputs/downloaded_face_occlusion_continuation/outputs/face_occlusion_continuation_vm'
    source = source_root / 'epoch_30.pth'
    moments = source_root / 'final_optimizer.pth'
    replay_path = ROOT / 'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    parent_path = ROOT / 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(source) == MODEL_SHA and sha(moments) == OPTIMIZER_SHA and
            sha(replay_path) == REPLAY_SHA and sha(parent_path) == PARENT_SHA,
            'Audited source model/moments/replay/parent changed')
    sys.path.insert(0, str(ROOT / 'outputs/face_extraction_dependencies'))
    torch.set_num_threads(4)
    model, payload = load_adapter(source, 'cpu')
    validate_source(payload)
    moment_payload = torch.load(moments, map_location='cpu', weights_only=True)
    moment_check = validate_optimizer(moment_payload, parameter_groups(model), MODEL_SHA)
    model.requires_grad_(False).eval()
    parent, _ = load_completion(parent_path, 'cpu')
    parent.requires_grad_(False).eval()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    parent_initial = {k: v.clone() for k, v in parent.state_dict().items()}
    replay = torch.load(replay_path, map_location='cpu', weights_only=True)
    protocol = read_json(ROOT / 'outputs/coverage_protocol_v1/protocol.json')
    rows = load_manifest(ROOT / 'dataset/detector_training_extension_v2/manifest.json')
    training = [r for r in rows if r['split'] == 'train']
    validate_schedule(protocol, training, replay)
    real = ReviewedMasks(training, 256)
    batch = data['core_schedule']['batches'][0][0]
    pairs = [real[i] if i < 73 else (replay[i-73][0].float()/255, replay[i-73][1].float()) for i in batch]
    x, mask = (torch.stack([p[j] for p in pairs]) for j in (0, 1))
    tag = torch.tensor([i >= 73 for i in batch], dtype=torch.bool)
    cache = torch.load(ROOT / DATA_PATH / 'pixels.pth', map_location='cpu', weights_only=True)
    pixels = cache['pixels']

    def tensors(ids):
        return tuple(torch.stack([pixels[i][key].float() for i in ids]) / (255 if key == 'input' else 1)
                     for key in ('input', 'mask', 'valid'))

    terms = {}
    with torch.inference_mode():
        logits = model.detect(x)
        supervised = segmentation_loss(logits, mask, .25, .1)
        teacher = replay_consistency(logits[tag], parent.detect(x[tag]), torch.ones(int(tag.sum()), dtype=torch.bool))
        terms['core'] = float(supervised + teacher)
        for arm in ('control', 'reflective'):
            sx, sm, sv = tensors(supplemental_ids(arm, data['supplemental_schedule'][0]))
            terms[arm] = float(supported_segmentation_loss(model.detect(sx), sm, sv))
        # Fixed Asian/FFHQ camera probes explicitly exercise the non-full-support branch.
        probe_ids = [9, 141]
        px, pm, pv = tensors(probe_ids)
        require(not bool(pv[0].all()) and bool(pv[1].all()), 'Padding probe membership/support differs')
        terms['padded_probe'] = float(supported_segmentation_loss(model.detect(px), pm, pv))
    require(all(torch.equal(v, initial[k]) for k, v in model.state_dict().items()) and
            all(torch.equal(v, parent_initial[k]) for k, v in parent.state_dict().items()) and
            all(p.grad is None for p in model.parameters()), 'Read-only source forward changed state/gradients')
    check = {
        'format': 'dgp-reflection-coverage-source-check-v1', 'device': 'cpu',
        'model_sha256': MODEL_SHA, 'optimizer_sha256': OPTIMIZER_SHA, 'data_sha256': DATA_SHA,
        'pixel_cache_sha256': PIXELS_SHA, 'audit_sha256': AUDIT_SHA,
        'visual_review_sha256': sha(ROOT / DATA_PATH / 'visual_review.json'),
        'parent_sha256': PARENT_SHA, 'replay_sha256': REPLAY_SHA,
        'model_state_unchanged': True, 'parent_state_unchanged': True,
        'optimizer_constructed': False, 'optimizer_updates_locally': 0, 'moment_step': 420,
        'optimizer': moment_check, 'terms': terms,
        'core_terms': {'supervised': float(supervised), 'original_parent_consistency': float(teacher)},
        'core_batch_indices': batch, 'supplemental_indices': data['supplemental_schedule'][0],
        'padded_probe_indices': probe_ids, 'logit_shape': list(logits.shape),
        'code_hashes': {name: sha(ROOT / name) for name in CODE_PATHS},
        'prior_inventory_hashes': PRIOR_INVENTORIES,
        'torch': str(torch.__version__),
        'dependencies': dependency_versions(ROOT / 'outputs/face_extraction_dependencies'),
        'scope': 'Read-only CPU input/architecture/loss compatibility; no backward, optimizer or fitting. New CUDA pilot unverified.',
    }
    verify_source_check(check, MODEL_SHA, OPTIMIZER_SHA, DATA_SHA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((json.dumps(check, indent=2) + '\n').encode())
    print(json.dumps(check, indent=2), flush=True)


def package():
    verify_data()
    inventories = verify_prior()
    check = read_json(ROOT / CHECK_PATH)
    verify_source_check(check, MODEL_SHA, OPTIMIZER_SHA, DATA_SHA)
    require(check['pixel_cache_sha256'] == PIXELS_SHA and check['audit_sha256'] == AUDIT_SHA and
            check['visual_review_sha256'] == sha(ROOT / DATA_PATH / 'visual_review.json') and
            check['prior_inventory_hashes'] == PRIOR_INVENTORIES and
            check['code_hashes'] == {name: sha(ROOT / name) for name in CODE_PATHS},
            'Source forward code/evidence changed; preserve it and create a new version')
    files = list(CODE_PATHS) + [
        'scripts/prepare_reflection_source_cohort.py', 'scripts/prepare_reflection_coverage_data.py',
        'scripts/prepare_qualified_reflection_review.py', 'scripts/audit_qualified_reflection_review.py',
        'scripts/prepare_clear_replay_scope_review.py', 'scripts/record_clear_replay_scope_screening.py',
        'tests/test_completion_data_v2.py', 'tests/test_qualified_reflection_review.py',
        'tests/test_qualified_reflection_audit.py', 'tests/test_reflection_source_cohort.py',
        'tests/test_reflection_coverage.py', 'tests/test_reflection_coverage_data.py',
        'tests/test_reflection_coverage_audit.py', 'tests/test_reflection_coverage_vm.py',
        'tests/test_reflection_coverage_package.py', 'REFLECTION_COVERAGE.md', 'REFLECTION_COVERAGE_VM.md',
        CHECK_PATH,
    ]
    for folder in (DATA_PATH, 'outputs/qualified_reflection_v1', 'outputs/qualified_source_review_v1',
                   'outputs/reflection_source_cohort_v2', 'outputs/clear_replay_scope_v1'):
        files.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / folder).rglob('*')) if p.is_file())
    validate_members(files, inventories)
    report = build_bundle(ROOT, files, ROOT / INVENTORY_PATH, ROOT / 'outputs/reflection-coverage-code.tar.gz')
    report.update(source_check_sha256=sha(ROOT / CHECK_PATH), data_manifest_sha256=DATA_SHA,
                  optimizer_updates_locally=0, new_cuda_execution_verified=False)
    (ROOT / INVENTORY_PATH).with_name('build.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check-source', action='store_true')
    args = parser.parse_args()
    require(args.prepare == args.check_source, 'Use --prepare --check-source together for the read-only check')
    prepare() if args.prepare else package()
