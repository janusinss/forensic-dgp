"""Inference-only face-prior feasibility; no optimizer or trainable extension."""
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/cctv_dgp_face_prior_feasibility_v10'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
RETURN = ROOT / 'outputs/cctv_dgp_mixed_return_v9'
RESULT = RETURN / 'outputs/cctv_dgp_mixed_v9'
NATIVE = ROOT / 'outputs/cctv_native_development_v2'
CACHED = ROOT / 'outputs/cctv_native_comparison_v1'
CF_WEIGHT = 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth'
IDENTITY_WEIGHT = 'outputs/cctv_dgp_vm_bundle_v1/weights/w600k_r50.onnx'
DGP_WEIGHT = 'outputs/cctv_dgp_mixed_return_v9/outputs/cctv_dgp_mixed_v9/checkpoints/epoch_20.pth'
PINS = {
    'outputs/cctv_dgp_mixed_vm_v9_r2/mixed_protocol_v9.json': '6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70',
    'outputs/cctv_dgp_mixed_return_v9/local_independent_audit.json': '902d1b5fef6764c270402fbd56e66d9e712514653d43fb6572d20b549fe61241',
    'outputs/cctv_dgp_mixed_return_v9/outputs/cctv_dgp_mixed_v9/results.json': '6d45a93b790160bea91a6db2d915ab031be3fa37f1a4da8d62da484c1a0a4e00',
    'outputs/cctv_native_development_v2/frozen_subset.json': 'c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e',
    'outputs/cctv_native_development_v2/input_review.json': '41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2',
    'outputs/cctv_native_comparison_v1/results.json': '3be73af7b0cdae85f54adc8173fd8f48b42511a604fcd60244e3b1d2565a4c34',
    CF_WEIGHT: '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7',
    IDENTITY_WEIGHT: '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
    DGP_WEIGHT: '5edc4d06edfed92eb83fc73f22be9dbf41f2f7192bf81e1d251ba5b7693875ae',
}
PAIRED_ARMS = ['input', 'cached_v2', 'cached_v9_epoch20_diagnostic',
               'codeformer_input_w1', 'codeformer_dgp_w1_diagnostic']
NATIVE_ARMS = ['input', 'cached_phase3', 'cached_codeformer_w1',
               'v9_epoch20_diagnostic', 'codeformer_dgp_w1_diagnostic']
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def safe(root, name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and bool(p.parts) and '..' not in p.parts and
            '\\' not in name and ':' not in name, 'Unsafe relative asset path')
    dest = (root / name).resolve()
    require(dest.is_relative_to(root.resolve()), 'Asset escapes root')
    return dest


def choose_references(references):
    eligible = [r for r in references if r['role'] == 'validation']
    require(len({r['id'] for r in eligible}) == len(eligible), 'Repeated reference')
    sources = ['dataset/asian_faces', 'dataset/thumbnails128x128']
    require({r['source'] for r in eligible} == set(sources), 'Unexpected validation sources')
    selected = []
    for source in sources:
        items = [r for r in eligible if r['source'] == source]
        require(len(items) >= 5, 'Insufficient fixed validation references')
        selected.extend(sorted(items, key=lambda r: hashlib.sha256(
            ('face-prior-v10:' + r['id']).encode()).hexdigest())[:5])
    return selected


def development_cases(subset):
    cases = [c for c in subset['cases'] if c['role'] == 'development']
    require(len(cases) == 24 and len({c['id'] for c in cases}) == 24,
            'Expected exactly24 different development cases')
    require(all(c['source_file'].startswith('development/') for c in cases),
            'Development source path has wrong role')
    return cases


def rgb(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'Expected exact256 RGB asset')
        return np.asarray(image).copy()


def png(raw, input_rgb, observed):
    require(raw.dtype == np.float32 and raw.shape == (256, 256, 3) and
            np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
            'Invalid raw float output')
    require(observed.shape == (256, 256) and observed.dtype == np.bool_ and observed.any(),
            'Invalid observed support')
    value = np.floor(raw * 255).astype(np.uint8)
    value[~observed] = input_rgb[~observed]
    return value


def prepare():
    require(not OUT.exists(), 'Preserve prior frozen/partial comparison')
    for name, pin in PINS.items():
        require(sha(safe(ROOT, name)) == pin, 'Parent asset differs: ' + name)
    p = read(MIXED / 'mixed_protocol_v9.json')
    audit = read(RETURN / 'local_independent_audit.json')
    result = read(RESULT / 'results.json')
    require(audit['complete'] and result['complete'] and audit['selected_epoch'] == 0,
            'Require audited V9 negative result; diagnostic checkpoint only')
    refs = choose_references(p['references']); ids = {r['id'] for r in refs}
    cases = [c for c in p['validation_cases'] if c['reference_id'] in ids]
    require(len(cases) == 50 and all(sorted(c['profile'] for c in cases if
            c['reference_id'] == r['id']) == sorted(PROFILES) for r in refs),
            'Require every frozen profile for each reference')
    assets = dict(PINS)
    def bind(path, expected=None):
        name = path.relative_to(ROOT).as_posix(); actual = sha(path)
        require(expected is None or actual == expected, 'Changed inherited bytes: ' + name)
        assets[name] = actual
        return name
    paired = []
    for ref in refs:
        bind(MIXED / ref['target'], p['assets_sha256'][ref['target']])
        bind(MIXED / ref['observed'], p['assets_sha256'][ref['observed']])
    for case in cases:
        ref = next(r for r in refs if r['id'] == case['reference_id'])
        files = {'input': bind(MIXED / case['input'], p['assets_sha256'][case['input']]),
                 'target': (MIXED / ref['target']).relative_to(ROOT).as_posix(),
                 'observed': (MIXED / ref['observed']).relative_to(ROOT).as_posix()}
        for arm, stage in [('cached_v2', 'baseline'), ('cached_v9_epoch20_diagnostic', 'epoch20')]:
            name = stage + '/images/' + case['id'] + '.png'
            files[arm] = bind(RESULT / name, result['artifacts_sha256'][name])
        paired.append({**case, 'files': files, 'matrix112': ref['matrix112']})
    subset = read(NATIVE / 'frozen_subset.json'); native = development_cases(subset)
    reviews = read(NATIVE / 'input_review.json'); reviewed = {r['id']: r for r in reviews['rows']}
    require(not reviews['reserved_inputs_viewed'] and set(reviewed) == {c['id'] for c in native},
            'Native input review/role differs')
    baseline = read(CACHED / 'results.json')
    native_rows = []
    for case in native:
        source = bind(NATIVE / case['source_file'], case['source_sha256'])
        files = {'source': source}
        for key, name in [('input', 'images/' + case['id'] + '_input.png'),
                          ('cached_phase3', 'stages/' + case['id'] + '_dgp.npy'),
                          ('cached_codeformer_w1', 'stages/' + case['id'] + '_codeformer.npy')]:
            files[key] = bind(CACHED / name, baseline['artifacts_sha256'][name])
        native_rows.append({**case, 'files': files, 'input_review': reviewed[case['id']]})
    names = ['cctv_dgp_face_prior_v10.py', 'scripts/compare_cctv_dgp_face_prior_v10.py',
             'scripts/audit_cctv_dgp_face_prior_v10.py', 'tests/test_cctv_dgp_face_prior_v10.py',
             'pretrained_face_restoration.py', 'dgp_frozen_inference_v2.py',
             'dgp_face_restoration.py', 'cctv_dgp_frozen_norm.py', 'cctv_dgp_pilot.py']
    names.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / 'models').glob('*.py'))
    names.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / 'third_party/codeformer').glob('*') if path.is_file())
    for name in names:
        bind(ROOT / name)
    plan = {'format': 'dgp-face-prior-feasibility-v10', 'date': '2026-10-04',
        'frozen_before_new_outputs': True, 'assets_sha256': assets,
        'references': refs, 'paired_cases': paired, 'native_cases': native_rows,
        'paired_selection': 'Five validation references/source, ascending SHA256(face-prior-v10:reference_id); all five profiles. No metric filtering or role reassignment.',
        'paired_arms': PAIRED_ARMS, 'native_arms': NATIVE_ARMS,
        'input_policy': 'Exact existing RGB256 pixels, observed-region composite, no extra warp/filter/landmark alignment/enhancement; CodeFormer RGB bilinear512 [-1,1], w1/adainTrue, bilinear256.',
        'conditioning_policy': 'CodeFormer receives the delivered floor-quantized V9 epoch20 DGP diagnostic PNG; no trained prior adapter yet.',
        'native_input_policy': 'Original native RGB128 center-pad, PIL bilinear256, nearest observed mask; verify against old cached inputs.',
        'dgp_epoch20_status': 'Failed V9 appearance safeguards; diagnostic preprocessing arm only, not selected best or production.',
        'contribution': 'Existing trained DGP plus frozen official CodeFormer; inference feasibility, not a newly trained DGP extension.',
        'pretrained_license': 'S-Lab License1.0; retained source license and NOTICE, no new license grant.',
        'budget': {'paired_cases': 50, 'native_cases': 24, 'codeformer_forwards': 124,
                   'dgp_forwards': 24, 'recognizer_forwards': 260, 'cpu_threads': 4,
                   'wall_seconds_after_loading': 1200, 'timing_checkpoint_paired_cases': 5,
                   'optimizer_updates': 0, 'backward_calls': 0, 'device': 'cpu'},
        'review_criteria': ['Compare all five ten-reference paired profile grids at original256 cells; preserve observed contour, eyes/nose/mouth and expression before sharpness.',
            'Review all24 native diagnostics; report input-reviewed six frontal/mild coarse cases separately from seven insufficient cases.',
            'Useful prior feasibility requires coherent feature/detail improvement without new obvious anatomy, changed expression or strong color artifacts on reviewed eligible inputs; otherwise redesign before training.',
            'Report source/profile PNG MSE/SSIM and fixed-affine recognizer similarity on paired proxies separately; no native paired metrics or verified identity accuracy.',
            'No model selection or app promotion in this experiment; any subsequent training needs its own finite pre-output protocol and untouched historic safeguards.'],
        'native_reserved_used': False, 'training': False, 'production_promotion': False,
        'independent_final_review_pending': True}
    OUT.mkdir(); write(OUT / 'frozen_plan.json', plan)
    (OUT / 'frozen_plan.sha256').write_text(sha(OUT / 'frozen_plan.json') + '\n', encoding='ascii')
    print(json.dumps({'prepared': True, 'plan_sha256': sha(OUT / 'frozen_plan.json'),
                      'paired_cases': len(paired), 'native_cases': len(native_rows),
                      'training': False, 'codeformer_forward_budget': 124}))


def verify():
    require((OUT / 'frozen_plan.sha256').read_text(encoding='ascii').strip() == sha(OUT / 'frozen_plan.json'),
            'Frozen plan fingerprint differs')
    p = read(OUT / 'frozen_plan.json')
    require(p['format'] == 'dgp-face-prior-feasibility-v10' and not p['training'] and
            not p['native_reserved_used'] and not p['production_promotion'], 'Wrong comparison scope')
    require(p['paired_arms'] == PAIRED_ARMS and p['native_arms'] == NATIVE_ARMS and
            len(p['paired_cases']) == 50 and len(p['native_cases']) == 24, 'Wrong fixed cohort/arms')
    require(choose_references(read(MIXED / 'mixed_protocol_v9.json')['references']) == p['references'],
            'Fixed reference selection differs')
    for name, pin in p['assets_sha256'].items():
        require(sha(safe(ROOT, name)) == pin, 'Changed frozen asset: ' + name)
    return p
