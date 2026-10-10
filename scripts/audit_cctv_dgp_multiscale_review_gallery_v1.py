"""Independent exact-cell and full-return integrity audit; no model calls."""
import gzip
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not (OUT / 'gallery_independent_audit.json').exists()
    plan = read(OUT / 'plan.json')
    preparation = read(OUT / 'preparation.json')
    manifest = read(OUT / 'gallery/gallery_manifest.json')
    audit_file = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_file)
    assert preparation['complete'] and preparation['full_return_gzip_CRC_verified']
    assert plan['independent_audit_sha256'] == manifest['independent_audit_sha256'] == sha(audit_file)
    assert plan['protocol_sha256'] == audit['protocol_sha256'] == manifest['protocol_sha256']
    assert preparation['gallery_manifest_sha256'] == sha(OUT / 'gallery/gallery_manifest.json')
    assert len(manifest['pages']) == 64 and len(manifest['cells']) == 2280
    assert len({row['name'] for row in manifest['pages']}) == 64
    count = 0
    for row in manifest['pages']:
        page = ROOT / row['file']
        assert page.resolve().is_relative_to((OUT / 'gallery').resolve()) and sha(page) == row['sha256']
        with Image.open(page) as canvas:
            assert canvas.mode == 'RGB' and canvas.size == (row['width'], row['height'])
            cells = [cell for cell in manifest['cells'] if cell['page'] == row['name']]
            assert len(cells) == (45 if row['kind']=='paired_TRAIN' else 20)
            for cell in cells:
                source = ROOT / cell['source']
                assert source.resolve().is_relative_to((ROOT / 'outputs').resolve()) and sha(source) == cell['source_sha256']
                assert cell['width'] == cell['height'] == 256
                copied = canvas.crop((cell['x'], cell['y'], cell['x']+256, cell['y']+256))
                with Image.open(source) as original:
                    assert original.size == (256,256)
                    original = original.convert('RGB')
                    assert copied.tobytes() == original.tobytes()
                    assert hashlib.sha256(original.tobytes()).hexdigest() == cell['RGB_sha256']
                count += 1
    assert count == 2280 and manifest['enhanced_or_rescaled_cells'] == 0
    archive = ROOT / 'outputs/cctv-dgp-multiscale-calibration-v1-results.tar.gz'
    assert sha(archive) == audit['archive_sha256']
    total = 0
    with gzip.open(archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            total += len(chunk)
            assert total <= 2684354560+1024**2
    assert total == preparation['uncompressed_stream_bytes']
    result = dict(complete=True, pages=64, paired_pages=40, native_pages=24, exact_cells_verified=2280,
                  full_gzip_CRC_verified=True, enhanced_or_rescaled_cells=0,
                  checker_sha256=sha(Path(__file__)), gallery_manifest_sha256=sha(OUT / 'gallery/gallery_manifest.json'),
                  source_bound_audit_sha256=sha(audit_file), visual_review_pending=True,
                  local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
                  model_qualification=False, goal_complete=False)
    with (OUT / 'gallery_independent_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(dict(complete=True, exact_cells_verified=2280, pages=64, visual_review_pending=True))


if __name__ == '__main__':
    main()
