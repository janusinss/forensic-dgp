"""Real saved controls plus analytical activation maps; no neural execution."""
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

from import_cctv_dgp_recognizer_precision_v1_return import ROOT, OUT, INPUT, BANK, SOURCE, read, sha, write
from audit_cctv_dgp_recognizer_precision_v1_return import audit_phase
from cctv_dgp_recognizer_precision_v1_contract import project, select_references


def main():
    root = ROOT / 'scratch/recognizer_precision_fixture_v1_r1'; assert not root.exists(); root.mkdir()
    old = read(BANK / 'protocol.json')
    selected = select_references(read(OUT / 'cctv_dgp_appearance_input_response_v1_analysis/boundary_review.json')['case_records'],
        old['cases'], ['dataset/asian_faces', 'dataset/thumbnails128x128'])
    parent = SOURCE / 'outputs/conditioning_response_v1'
    block = next(b for b in read(SOURCE / 'protocol.json')['reference_blocks']
                 if old['cases'][b['begin']]['source_person_or_reference'] == selected[0]['reference'])
    stem = f"b{block['begin']//5:03d}"; cases = old['cases'][block['begin']:block['begin'] + 5]
    before = np.load(parent / 'initial' / (stem + '.npy'), allow_pickle=False)
    with np.load(parent / 'initial' / (stem + '_embeddings.npz'), allow_pickle=False) as f:
        original_vector = f['raw'].copy()
    original_g = np.load(INPUT / 'outputs/appearance_input_response_v1/repeat00' / (stem + '_RGB_gradients.npy'), allow_pickle=False)
    def create(precision, label):
        folder = root / precision; folder.mkdir(exist_ok=True); tag = stem + '_' + label
        dtype = np.float32 if precision == 'FP32' else np.float64
        with np.load(parent / label / (stem + '_embeddings.npz'), allow_pickle=False) as f:
            vector = f['raw'].astype(dtype)
        if precision == 'FP64':
            vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
        gradient = original_g.astype(dtype)
        np.savez_compressed(folder / (tag + '_embeddings.npz'), raw=vector)
        if label == 'initial':
            np.save(folder / (tag + '_RGB_gradients.npy'), gradient, allow_pickle=False)
        values = np.asarray([-1, 0, 1] * 320).reshape(5, 3, 8, 8)
        packed = np.packbits(np.stack([values > 0, values == 0]).reshape(-1), bitorder='little')
        path = folder / (tag + '_activation.npy'); np.save(path, packed, allow_pickle=False)
        rows = []
        for i, c in enumerate(cases):
            row = dict(id=c['id'], source=c['source'], profile=c['profile'], loss=1 - float(vector[i].astype(np.float64) @ vector[10 + i].astype(np.float64)))
            if label == 'initial':
                row.update(gradient_norm=float(np.linalg.norm(gradient[i].astype(np.float64))),
                    gradient_bytes_sha256=hashlib.sha256(gradient[i].tobytes()).hexdigest(), cross_case_gradient_maximum=0,
                    projections={endpoint: project(gradient[i], before[i],
                        np.load(parent / endpoint / (stem + '.npy'), allow_pickle=False)[i]) for endpoint in ['plus02', 'minus02']})
            rows.append(row)
        torch_dtype = 'torch.float32' if precision == 'FP32' else 'torch.float64'
        r = dict(precision=precision, begin=block['begin'], label=label, rows=rows,
            activations={'analytical_fixture': dict(path=path.name, shape=list(values.shape), packed_sha256=sha(path),
                units=values.size, positive=320, zero=320, negative=320, module_type='AnalyticalPReLU', dtype=torch_dtype)},
            encoder_floating_outputs=[dict(module='analytical_fixture', dtype=torch_dtype)])
        write(folder / (tag + '.json'), r)
        audit_phase(folder, {}, old, block, precision, label)
        return folder, tag, r, vector
    f32 = create('FP32', 'initial'); create('FP64', 'initial')
    folder, tag, r, vector = f32
    changed = json.loads(json.dumps(r)); changed['rows'][0]['projections']['plus02'] += .001
    path = folder / (tag + '.json'); original = path.read_bytes()
    path.write_text(json.dumps(changed), encoding='utf-8')
    try:
        audit_phase(folder, {}, old, block, 'FP32', 'initial')
    except AssertionError:
        projection_rejected = True
    else:
        raise AssertionError('Incorrect actual RGB projection accepted')
    finally:
        path.write_bytes(original)
    # Reproduce the control-input mistake: changed RGB used as the frozen baseline.
    folder, tag, r, vector = create('FP32', 'plus02')
    path = folder / (tag + '_embeddings.npz'); original = path.read_bytes()
    wrong = vector.copy(); wrong[5:10] = wrong[:5]
    np.savez_compressed(path, raw=wrong)
    try:
        audit_phase(folder, {}, old, block, 'FP32', 'plus02')
    except AssertionError:
        baseline_rejected = True
    else:
        raise AssertionError('Moving frozen baseline accepted')
    finally:
        path.write_bytes(original)
    path = folder / (tag + '_activation.npy'); original = path.read_bytes()
    bits = np.unpackbits(np.load(path, allow_pickle=False), bitorder='little'); bits[0] = bits[960] = 1
    np.save(path, np.packbits(bits, bitorder='little'), allow_pickle=False)
    path_json = folder / (tag + '.json'); metadata = path_json.read_bytes()
    changed = read(path_json); changed['activations']['analytical_fixture']['packed_sha256'] = sha(path)
    path_json.write_text(json.dumps(changed), encoding='utf-8')
    try:
        audit_phase(folder, {}, old, block, 'FP32', 'plus02')
    except AssertionError:
        activation_rejected = True
    else:
        raise AssertionError('Contradictory activation map accepted')
    finally:
        path.write_bytes(original); path_json.write_bytes(metadata)
    for precision, label in [('FP32', 'initial'), ('FP64', 'initial'), ('FP32', 'plus02')]:
        audit_phase(root / precision, {}, old, block, precision, label)
    write(OUT / 'cctv_dgp_recognizer_precision_v1_fixture.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), return_auditor_sha256=sha(ROOT / 'scripts/audit_cctv_dgp_recognizer_precision_v1_return.py'),
        saved_case_ids=block['case_ids'], saved_FP32_vectors_and_gradients_used=True,
        promoted_FP64_and_activation_maps_are_analytical_fixtures_not_new_neural_outputs=True,
        actual_pixel_projection_corruption_rejected=projection_rejected,
        wrong_frozen_baseline_rejected=baseline_rejected, contradictory_activation_signs_rejected=activation_rejected,
        scratch_bytes_restored=True, local_neural_or_training_calls=0, model_qualified=False))
    print(dict(complete=True, saved_cases=5, both_gradient_dtypes=True,
        projection_baseline_and_activation_corruption_rejected=True, neural_calls=0))


if __name__ == '__main__':
    main()
