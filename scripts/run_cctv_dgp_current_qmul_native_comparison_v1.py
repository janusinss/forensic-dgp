"""Complete the missing current-DGP QMUL comparison; frozen local inference only."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import state_hash
from dgp_face_restoration import prepare_crop
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
from pretrained_face_restoration import load_face_restorer, WEIGHTS_SHA256, REVISION

BASE = ROOT/'outputs/cctv_native_development_v2'
OUT = ROOT/'outputs/cctv_dgp_current_qmul_native_comparison_v1'
ARMS = ['resize', 'phase3', 'current_dgp', 'codeformer_w1']
CHECKPOINTS = {'phase3': ('checkpoints/dgp_zamboanga_final.pth', 'b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c'),
    'current_dgp': ('outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth', '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'),
    'codeformer_w1': ('outputs/codeformer_restoration_pretrained_v1/codeformer.pth', WEIGHTS_SHA256)}
PINS = {'frozen_subset.json': 'c063985b258b808efaa897c88593cbdbcee2848d686d3d056219f8a8e555a98e',
        'input_review.json': '41912033f5070547e120a6b384024fa929eeaf3a7ecca408c0c67c53d37d36e2'}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f: f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def main():
    assert not OUT.exists(), 'Preserve every prior/partial comparison; no automatic repeat'
    for n, h in PINS.items(): assert sha(BASE/n) == h
    subset = read(BASE/'frozen_subset.json'); review = read(BASE/'input_review.json')
    assert review['subset_sha256'] == PINS['frozen_subset.json'] and review['reviewed_before_model_outputs']
    assert not review['reserved_inputs_viewed']
    cases = [c for c in subset['cases'] if c['role'] == 'development']; reviews = {r['id']: r for r in review['rows']}
    assert len(cases) == len(reviews) == 24 and set(reviews) == {c['id'] for c in cases}
    assert len({c['global_person_id'] for c in cases}) == 24
    files = [Path(__file__), ROOT/'scripts/verify_cctv_dgp_current_qmul_native_comparison_v1.py',
        BASE/'frozen_subset.json', BASE/'input_review.json', BASE/'selection_policy.json',
        ROOT/'cctv_dgp_pilot.py', ROOT/'dgp_face_restoration.py', ROOT/'dgp_frozen_inference_v2.py',
        ROOT/'cctv_dgp_frozen_norm.py', ROOT/'pretrained_face_restoration.py', ROOT/'third_party/codeformer/LICENSE']
    files.extend((ROOT/n for n, _ in CHECKPOINTS.values()))
    files.extend(sorted((ROOT/'models').glob('*.py'))); files.extend(sorted((ROOT/'third_party/codeformer').glob('*.py')))
    for c in cases:
        assert sha(BASE/c['source_file']) == c['source_sha256']; files.append(BASE/c['source_file'])
        old_input = ROOT/'outputs/cctv_native_comparison_v1/images'/(c['id']+'_input.png')
        assert old_input.is_file(); files.append(old_input)
    files.extend([ROOT/'outputs/cctv_native_comparison_v1/results.json', ROOT/'outputs/cctv_dgp_native_pilot_review_v1/execution.json'])
    for _, (n, h) in CHECKPOINTS.items(): assert sha(ROOT/n) == h
    bindings = {q.relative_to(ROOT).as_posix(): sha(q) for q in files}
    OUT.mkdir(); (OUT/'raw').mkdir(); (OUT/'images').mkdir(); (OUT/'masks').mkdir()
    (OUT/'CODEFORMER_LICENSE').write_bytes((ROOT/'third_party/codeformer/LICENSE').read_bytes())
    write(OUT/'plan.json', {'format': 'current-DGP-QMUL-native-development-matched-comparison-v1',
        'date': '2026-10-09', 'frozen_before_outputs': True, 'cases': cases, 'input_reviews': review['rows'], 'arms': ARMS,
        'checkpoints': CHECKPOINTS, 'sources_sha256': bindings,
        'input_policy': 'Native RGB128 center-pad then PIL bilinear256; identical NumPy float32/255 tensor to all three models, no extra alignment/denoise/degradation',
        'output_policy': 'Separate untouched raw float32 model arrays; PNG floor(raw*float32(255)) with exact input padding restored; no display enhancement',
        'normalization': 'Both DGP checkpoints use retained stored evaluation statistics and five disposable InstanceNorm kernel copies',
        'pretrained_comparison': {'model': 'Official CodeFormer restoration', 'revision': REVISION,
            'fidelity': 1., 'adain': True, 'internal_resolution': 512, 'returned_resolution': 256,
            'alignment_limit': 'No detection/FFHQ alignment; this is the common-crop baseline, not the full publisher pipeline'},
        'input_only_core': [r['id'] for r in review['rows'] if r['pose_review'] == 'frontal_or_mild_approximate' and r['input_structure_review'] == 'coarse'],
        'all24_retained': True, 'insufficient_information_review_rule': 'Seven pre-output insufficient cases request clearer crops; sharper generated detail does not reverse those labels',
        'native_evidence_unpaired': True, 'PSNR': None, 'SSIM': None, 'identity_accuracy': None,
        'reserved_pixels_used': False, 'training': False, 'app_changed': False, 'model_qualification': False,
        'budget': {'worker_seconds_including_loading': 480, 'external_seconds': 540,
            'phase3_forwards': 24, 'current_dgp_forwards': 24, 'codeformer_w1_forwards': 24, 'threads': 4},
        'source_terms': 'QMUL-SurvFace research release; copyright retained by original owners. No unrestricted redistribution permission.',
        'publisher_source': 'https://qmul-survface.github.io/', 'country_or_ethnicity_inferred': False, 'zamboanga_performance': False})
    (OUT/'DERIVATIVE_NOTICE.txt').write_text('Research-only QMUL-SurvFace development derivatives. Copyright belongs to original data owners; retain publisher notices. These generated estimates and prepared canvases are modified observations, not recovered hidden identity. Source: https://qmul-survface.github.io/\n', encoding='utf-8')
    start = time.monotonic(); counts = {k: 0 for k in ARMS[1:]}; records = []; hooks = []
    try:
        torch.set_num_threads(4)
        phase3, p3 = load_frozen_dgp_restorer(ROOT/CHECKPOINTS['phase3'][0], expected_sha256=CHECKPOINTS['phase3'][1])
        primary, pd = load_frozen_dgp_restorer(ROOT/CHECKPOINTS['current_dgp'][0], expected_sha256=CHECKPOINTS['current_dgp'][1])
        cf, pc = load_face_restorer(ROOT/CHECKPOINTS['codeformer_w1'][0])
        models = {'phase3': phase3, 'current_dgp': primary, 'codeformer_w1': cf}
        assert all(not m.training and all(not p.requires_grad for p in m.parameters()) for m in models.values())
        before = {k: state_hash(m) for k, m in models.items()}
        for k, m in models.items():
            hooks.append(m.register_forward_hook(lambda *_a, k=k: counts.__setitem__(k, counts[k]+1)))
        write(OUT/'execution.json', {'plan_sha256': sha(OUT/'plan.json'), 'before': before,
            'torch': torch.__version__, 'CPU_threads': 4, 'optimizer_constructed': False,
            'provenance': {'phase3': p3, 'current_dgp': pd, 'codeformer_w1': pc}})
        with torch.inference_mode():
            for index, c in enumerate(cases):
                assert time.monotonic()-start <= 480, 'Finite inference deadline'
                native = pixels(BASE/c['source_file']); common, observed, _, geometry = prepare_crop(native)
                assert np.array_equal(common, pixels(ROOT/'outputs/cctv_native_comparison_v1/images'/(c['id']+'_input.png')))
                x = torch.from_numpy(common.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
                untouched = x.clone()
                outputs = {'phase3': phase3(x), 'current_dgp': primary(x), 'codeformer_w1': cf(x, fidelity=1.)}
                assert torch.equal(untouched, x)
                images = {'resize': common}; raw_paths = {}; changes = {}
                for arm, value in outputs.items():
                    assert value.shape == (1, 3, 256, 256) and value.dtype == torch.float32
                    raw = value[0].permute(1, 2, 0).numpy().copy()
                    assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
                    n = f'raw/{c["id"]}_{arm}.npy'; np.save(OUT/n, raw, allow_pickle=False); raw_paths[arm] = n
                    png = np.floor(raw*np.float32(255)).astype(np.uint8); png[~observed] = common[~observed]
                    images[arm] = png
                    changes[arm] = float(np.abs(png.astype(np.float64)-common.astype(np.float64))[observed].mean())
                paths = {}
                for arm, image in images.items():
                    n = f'images/{c["id"]}_{arm}.png'; Image.fromarray(image).save(OUT/n); paths[arm] = n
                mask_name = f'masks/{c["id"]}.png'; Image.fromarray(observed.astype(np.uint8)*255).save(OUT/mask_name)
                records.append({'id': c['id'], 'native_size': [c['native_width'], c['native_height']],
                    'source_person_id': c['global_person_id'], 'input_review': reviews[c['id']],
                    'geometry': geometry, 'outputs': paths, 'raw': raw_paths, 'observed': mask_name,
                    'input_tensor_float32_SHA256': hashlib.sha256(x.permute(0, 2, 3, 1).numpy().tobytes()).hexdigest(),
                    'observed_MAE_change_255_diagnostic_only': changes,
                    'input_change_is_quality_or_identity_metric': False})
                print({'case': index+1, 'of': 24, 'id': c['id'], 'seconds': time.monotonic()-start}, flush=True)
        after = {k: state_hash(m) for k, m in models.items()}
        assert after == before and counts == {k: 24 for k in ARMS[1:]}
        pages = []
        for begin in range(0, 24, 6):
            sheet = Image.new('RGB', (4*268, 24+6*292), 'white'); draw = ImageDraw.Draw(sheet)
            for col, arm in enumerate(ARMS): draw.text((col*268+5, 5), arm, fill='black')
            for row, record in enumerate(records[begin:begin+6]):
                y = 24+292*row
                draw.text((5, y), record['id']+' / native'+str(record['native_size'])+' / input='+record['input_review']['input_structure_review'], fill='black')
                for col, arm in enumerate(ARMS): sheet.paste(Image.fromarray(pixels(OUT/record['outputs'][arm])), (col*268+5, y+24))
            n = f'comparison-{begin//6+1:02d}.png'; sheet.save(OUT/n); pages.append(n)
        assert time.monotonic()-start <= 480
        write(OUT/'results.json', {'complete': True, 'plan_sha256': sha(OUT/'plan.json'), 'seconds': time.monotonic()-start,
            'records': records, 'model_forwards': counts, 'before': before, 'after': after,
            'model_states_unchanged': True, 'same_exact_inputs': True, 'pages': pages,
            'optimizer_updates': 0, 'gradient_queries': 0, 'app_changed': False, 'native_evidence_unpaired': True,
            'PSNR': None, 'SSIM': None, 'identity_accuracy': None, 'reserved_pixels_used': False,
            'visual_review_pending': True, 'independent_final_review': False, 'model_qualification': False,
            'artifacts_sha256': {q.relative_to(OUT).as_posix(): sha(q) for q in OUT.rglob('*') if q.is_file()}})
        print({'complete': True, 'forwards': counts, 'seconds': time.monotonic()-start}, flush=True)
    except BaseException as exc:
        write(OUT/'failure.json', {'complete': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
            'seconds': time.monotonic()-start, 'completed_cases': len(records), 'forwards': counts,
            'partial_outputs_preserved': True, 'training_calls': 0})
        raise
    finally:
        for h in hooks: h.remove()


if __name__ == '__main__': main()
