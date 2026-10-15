"""Independently check saved real-browser downloads without model execution."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'scratch/app_flow_20261012_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def decode(value, mime):
    prefix = 'data:' + mime + ';base64,'
    assert value.startswith(prefix)
    return base64.b64decode(value[len(prefix):], validate=True)


def pixels(data, mode='RGB'):
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        assert im.size == (256, 256) and im.format == 'PNG'
        return np.asarray(im.convert(mode)).copy()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    destination = ARTIFACTS / 'download_content_audit.json'
    assert not destination.exists(), 'Preserve the previous audit'
    preflight, flow = read(ARTIFACTS / 'preflight.json'), read(ARTIFACTS / 'ui-flow.json')
    assert flow['complete'] and flow['errors'] == []
    assert flow['layout'] == [{'width': w, 'overflow': False} for w in [375, 768, 1280]]
    for name, digest in preflight['source_and_weight_files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in preflight['input_files_sha256'].items():
        assert sha(name) == digest, name
    for name, digest in preflight['fixture_files_sha256'].items():
        assert sha(ARTIFACTS / name) == digest, name
    source = next(Path(name) for name in preflight['input_files_sha256'] if name.endswith('_clear.png'))
    original = pixels(source.read_bytes())
    proposal = pixels((ARTIFACTS / 'automatic-mask.png').read_bytes(), 'L')
    assert not proposal.any(), 'Uncovered TRAIN fixture must have an empty reviewed proposal'
    rows, raw_arrays = [], {}
    for label, requested, applied, mask_source in [
        ('clear-off', 'off', False, 'automatic_reviewed'),
        ('clear-on', 'on', True, 'automatic_reviewed'),
        ('clear-auto', 'auto', True, 'automatic_reviewed'),
        ('assisted-off', 'off', False, 'assisted_reviewed'),
    ]:
        response_path = ARTIFACTS / (label + '-response.json')
        response = read(response_path)
        metadata = response['metadata']
        output_bytes = decode(response['output'], 'image/png')
        bundle_bytes = decode(response['bundle'], 'application/zip')
        assert output_bytes == (ARTIFACTS / (label + '-output.png')).read_bytes()
        assert output_bytes == (ARTIFACTS / 'downloads' / (label + '-face-estimate.png')).read_bytes()
        assert bundle_bytes == (ARTIFACTS / (label + '-bundle.zip')).read_bytes()
        assert bundle_bytes == (ARTIFACTS / 'downloads' / (label + '-face-review.zip')).read_bytes()
        output = pixels(output_bytes)
        response_original = pixels(decode(response['original'], 'image/png'))
        response_mask = pixels(decode(response['mask'], 'image/png'), 'L')
        assert np.array_equal(response_original, original)
        assert set(np.unique(response_mask)).issubset({0, 255})
        mask = response_mask > 0
        assert metadata['restoration_requested'] == requested
        assert metadata['restoration_applied'] is applied
        assert metadata['mask_source'] == mask_source
        assert metadata['input_review']['decision'] == 'usable'
        assert metadata['width'] == metadata['height'] == 256
        assert metadata['optimizer_updates'] == metadata['backward_calls'] == 0
        assert not metadata['native_usefulness_qualified'] and not metadata['covering_families_qualified']
        assert metadata['original_rgb_sha256'] == array_sha(original)
        assert metadata['native_mask_sha256'] == array_sha(mask)
        assert metadata['output_rgb_sha256'] == array_sha(output)
        assert metadata['additional_mask_expansion'] == 0
        assert metadata['geometry']['native_size'] == metadata['geometry']['output_size'] == [256, 256]
        assert metadata['geometry']['observed_bounds_256'] == [0, 0, 256, 256]
        assert metadata['geometry']['canvas_rgb_sha256'] == array_sha(original)
        assert metadata['geometry']['observed_sha256'] == array_sha(np.ones((256, 256), bool))
        assert metadata['geometry']['removal_mask_sha256'] == array_sha(mask)
        assert metadata['display_processing']['colour_policy']['applied'] is False
        names = {'original.png', 'removal-mask-original.png', 'input-256.png',
                 'removal-mask.png', 'estimate.png', 'processing.json', 'README.txt'}
        if applied:
            names.add('dgp-raw-float32.npy')
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            entries = archive.infolist()
            assert len(entries) == len(names) and {entry.filename for entry in entries} == names
            assert all(not entry.is_dir() and '/' not in entry.filename and '\\' not in entry.filename
                       and entry.file_size <= 4 * 1024**2 for entry in entries)
            assert archive.testzip() is None
            assert json.loads(archive.read('processing.json')) == metadata
            assert np.array_equal(pixels(archive.read('original.png')), original)
            assert np.array_equal(pixels(archive.read('input-256.png')), original)
            assert np.array_equal(pixels(archive.read('removal-mask-original.png'), 'L'), response_mask)
            assert np.array_equal(pixels(archive.read('removal-mask.png'), 'L'), response_mask)
            assert archive.read('estimate.png') == output_bytes
            readme = archive.read('README.txt').decode('utf-8')
            assert 'plausible' in readme.lower() and 'identity' in readme.lower()
            if applied:
                raw = np.load(io.BytesIO(archive.read('dgp-raw-float32.npy')), allow_pickle=False)
                assert raw.dtype == np.float32 and raw.shape == (256, 256, 3)
                assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1
                assert metadata['raw_dgp']['sha256'] == array_sha(raw)
                assert metadata['raw_dgp']['dtype'] == 'float32'
                assert metadata['raw_dgp']['shape'] == [256, 256, 3]
                assert metadata['visible_restoration']['weights_sha256'] == preflight['source_and_weight_files_sha256'][
                    'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth']
                assert not mask.any()
                assert np.array_equal(output, np.floor(raw * np.float32(255)).astype(np.uint8))
                raw_arrays[label] = raw
            else:
                assert metadata['raw_dgp'] is None and metadata['visible_restoration'] is None
                assert np.array_equal(output[~mask], original[~mask])
        if label.startswith('clear-'):
            assert not mask.any() and metadata['completion'] == {'bypassed': 'empty mask'}
        else:
            assert int(mask.sum()) == metadata['quality_signals']['removed_pixels'] == 52
            assert np.array_equal(pixels((ARTIFACTS / 'downloads/assisted-removal-mask.png').read_bytes(), 'L'), response_mask)
            assert metadata['completion']['backend'] == 'codeformer-inpainting'
            assert metadata['completion']['weights_sha256'] == preflight['source_and_weight_files_sha256'][
                'checkpoints/codeformer_inpainting.pth']
        rows.append(dict(label=label, response_sha256=sha(response_path), selected_mask_pixels=int(mask.sum()),
                         output_png_sha256=hashlib.sha256(output_bytes).hexdigest(),
                         bundle_sha256=hashlib.sha256(bundle_bytes).hexdigest(), bundle_members=sorted(names),
                         metadata_and_pixel_hashes_match=True, browser_download_bytes_equal_API=True,
                         restoration_requested=requested, restoration_applied=applied, mask_source=mask_source,
                         observed_RGB_exact_when_restoration_off=not applied))
    assert np.array_equal(raw_arrays['clear-on'], raw_arrays['clear-auto']), 'Frozen repeated raw inference changed'
    generate_requests = [request for request in flow['requests'] if request['url'].endswith('/face/generate')]
    assert len(generate_requests) == 4
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
                  preflight_sha256=sha(ARTIFACTS / 'preflight.json'), ui_flow_sha256=sha(ARTIFACTS / 'ui-flow.json'),
                  cases=rows, source_model_and_fixture_hashes_unchanged=True,
                  On_and_Auto_raw_arrays_identical=True, automatic_empty_mask_and_assisted_52_pixel_mask_separate=True,
                  actual_downloads_checked=8, saved_neural_outputs_checked=True,
                  new_neural_gradient_optimizer_calls=0, no_new_model_execution=True,
                  square_photographic_TRAIN_fixture_only=True, native_non_square_flow_pending=True,
                  backend_negative_API_cases_pending=True, visual_review_pending=True,
                  full_app_flow_or_model_quality_qualification=False, all_seven_covering_families_qualified=False)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(complete=True, downloaded_PNGs=4, downloaded_bundles=4,
                         On_Auto_raw_parity=True, visible_RGB_exact_when_Off=True, quality_qualified=False)))


if __name__ == '__main__':
    main()
