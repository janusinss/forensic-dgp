"""Saved-only page manifest/download audit; no HTTP/browser/model execution."""
import builtins
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'scratch/u96-postfusion-ui-v1'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_page_saved_audit'


def main():
    original_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('No Torch/model execution in saved UI audit')
        return original_import(name, *args, **kwargs)
    builtins.__import__ = blocked
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from PIL import Image
    import research_postfusion_restoration_v1 as page
    checks = []
    def need(ok, name):
        checks.append({'name': name, 'pass': bool(ok)})
        if not ok: raise ValueError(name)
    def digest(raw): return hashlib.sha256(raw).hexdigest()
    def pin(path):
        raw = path.read_bytes()
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': digest(raw)}
    def load(path):
        need(0 < path.stat().st_size <= 8 * 1024 ** 2, 'Bounded saved JSON ' + path.name)
        return page.strict_json(path.read_text(encoding='utf-8'))
    def png(raw):
        need(0 < len(raw) <= 8 * 1024 ** 2, 'Bounded downloaded PNG')
        with Image.open(io.BytesIO(raw)) as image:
            need(image.format == 'PNG' and image.size == (256, 256) and image.mode == 'RGB', 'Exact256 RGB PNG')
            image.load(); return np.asarray(image).copy()
    def array(raw):
        stream = io.BytesIO(raw)
        version = np.lib.format.read_magic(stream)
        need(version in ((1, 0), (2, 0)), 'Supported NPY header')
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = reader(stream, max_header_size=4096)
        need(shape == (256, 256, 3) and dtype == np.dtype('float32') and not dtype.hasobject and not fortran
             and len(raw) == stream.tell() + 256 * 256 * 3 * 4, 'Exact bounded RAW NPY header before allocation')
        value = np.load(io.BytesIO(raw), allow_pickle=False)
        need(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1, 'Finite delivered RAW range')
        return value
    manifest_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_page_preparation/experimental_test_manifest_v1.json'
    need(pin(manifest_path)['sha256'] == '8c63f9f1ce080e1397df8c7708f2ed2c36601e47edce8d07a9b8036546dd4a68', 'Exact root experimental page test manifest')
    manifest = page.configure_manifest(manifest_path, pin(manifest_path)['sha256'])
    need(page.engine.models == {} and 'torch' not in sys.modules, 'All source/proof/model hashes verified without model loading')
    host_path = ROOT / 'scratch/codeformer_postfusion_research_server_20261022_host_v1.json'
    host = load(host_path); flow = load(UI / 'flow-report-r1.json'); layout = load(UI / 'layout-report.json')
    response = load(UI / 'processing-response.json'); first_stop = load(UI / 'first-flow-assertion-stop.json')
    need(host['test_manifest_sha256'] == pin(manifest_path)['sha256'] and host['host'] == '127.0.0.1'
         and host['url'] == flow['url'] and host['CPU_threads'] == 4, 'Saved actual host URL and manifest/thread binding')
    need(flow['complete'] is True and [row['width'] for row in flow['layouts']] == [375, 768, 1280]
         and all(row['overflow'] is False for row in flow['layouts'])
         and flow['pageErrors'] == flow['unexpectedConsoleErrors'] == [], 'Saved root browser flow reports three widths and no unexpected errors')
    need(flow['after']['neural_forward_attempts'] - flow['before']['neural_forward_attempts'] ==
         flow['after']['neural_forward_returns'] - flow['before']['neural_forward_returns'] == 2,
         'R1 functional flow consumed exactly two new forwards')
    need(flow['refused']['neural_forward_attempts'] == flow['after']['neural_forward_attempts'] == 4
         and flow['refused']['neural_forward_returns'] == flow['after']['neural_forward_returns'] == 4
         and flow['before']['neural_forward_returns'] == 2 and flow['refused']['completed_comparisons'] == 2,
         'Lifetime four forwards/two comparisons retained, including earlier assertion-stop comparison')
    need(len(flow['refusals']) == 3 and all(row['status'] == 422 for row in flow['refusals'])
         and flow['refused']['operator_review_refusals'] == 3, 'Three input review refusals added zero neural calls')
    need(response['test_manifest'] == pin(manifest_path) and response['models'] == manifest['models']
         and response['model_qualified'] is False and response['main_app_default_changed'] is False,
         'Downloaded processing response preserves exact model/manifest and research status')
    raw_zip = (UI / 'u96-postfusion-comparison.zip').read_bytes()
    expected = {'original-upload.png', 'resize-control.png', 'observed-support.png', 'processing.json',
                *[arm + suffix for arm in page.ARMS for suffix in ('.png', '-raw.npy')]}
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as z:
        info = z.infolist()
        need(len(info) == len(expected) and set(z.namelist()) == expected
             and len(raw_zip) < 4 * 1024 ** 2 and sum(row.file_size for row in info) < 4 * 1024 ** 2
             and all(not row.flag_bits & 1 and not stat.S_ISLNK(row.external_attr >> 16) for row in info),
             'Exact eight ordinary unencrypted bounded ZIP members')
        files = {row.filename: z.read(row) for row in info}  # Full member CRC checks.
    need(page.strict_json(files['processing.json'].decode('utf-8')) == response, 'ZIP metadata matches saved response exactly')
    original = png(files['original-upload.png']); prepared = png(files['resize-control.png'])
    support_raw = files['observed-support.png']
    with Image.open(io.BytesIO(support_raw)) as image:
        need(image.format == 'PNG' and image.size == (256, 256) and image.mode == 'L', 'Exact256 support mask')
        support = np.asarray(image).copy() > 0
    need(digest(original.tobytes()) == response['geometry']['input_rgb_sha256']
         and digest(prepared.tobytes()) == response['geometry']['canvas_rgb_sha256']
         and digest(support.tobytes()) == response['geometry']['observed_sha256'], 'Native/prepared/support pixel digests match metadata')
    case = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_run_v1/cases/va_ffhq_00383__local_gaussian_3p0'
    need(np.array_equal(prepared, png((case / 'prepared.png').read_bytes())), 'Same fixed DEV input pixels used for user page comparison')
    comparisons = {}
    for arm, name in zip(page.ARMS, ('incumbent-u96.png', 'postfusion-lr-u64.png')):
        downloaded = (UI / name).read_bytes(); pixels = png(downloaded); raw = array(files[arm + '-raw.npy'])
        row = response['outputs'][arm]
        need(downloaded == files[arm + '.png'] and digest(downloaded) == row['PNG_file_sha256'], arm + ' standalone PNG equals ZIP PNG and file digest')
        need(digest(raw.tobytes()) == row['raw_rgb_sha256'] and digest(pixels.tobytes()) == row['PNG_RGB8_sha256'], arm + ' RAW and delivered pixel digests')
        floor = np.floor(raw * np.float32(255)).astype(np.uint8); floor[~support] = prepared[~support]
        need(np.array_equal(floor, pixels) and np.array_equal(pixels[~support], prepared[~support]), arm + ' exact RAW floor quantization and support pixels')
        previous = array((case / arm / 'raw.npy').read_bytes())
        previous_png = png((case / arm / 'output.png').read_bytes())
        delta = np.abs(raw.astype(np.float64) - previous.astype(np.float64))
        comparisons[arm] = {'RAW_bitwise_equal': np.array_equal(raw, previous),
                            'PNG_RGB_bitwise_equal': np.array_equal(pixels, previous_png),
                            'maximum_RAW_difference': float(delta.max()), 'mean_RAW_difference': float(delta.mean())}
        need(comparisons[arm]['RAW_bitwise_equal'] and comparisons[arm]['PNG_RGB_bitwise_equal'], arm + ' same CPU RAW/PNG as retained exact DEV input')
    artifacts = [pin(path) for path in sorted(UI.iterdir()) if path.is_file()]
    OUT.mkdir(exist_ok=True)
    receipt = {'complete': True, 'saved_UI_evidence_audit_pass': True, 'checks': checks, 'all_checks_pass': True,
               'script': pin(Path(__file__)), 'manifest': pin(manifest_path), 'host_receipt': pin(host_path),
               'artifacts': artifacts, 'same_input_saved_DEV_comparison': comparisons,
               'actual_R1_flow_new_forwards': 2, 'actual_saved_server_lifetime_forwards': 4,
               'earlier_assertion_stop_retained': pin(UI / 'first-flow-assertion-stop.json'),
               'root_browser_reports_read_not_rerun': True, 'screenshots_not_visually_judged_by_this_auditor': True,
               'model_constructor_forward_backward_optimizer_training_VM_HTTP_browser_calls': 0,
               'model_qualified': False, 'main_app_default_changed': False,
               'scope': 'Functional transport and exact saved-pixel consistency only; no model usefulness or identity claim.'}
    with (OUT / 'saved_UI_evidence_audit_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'saved_UI_evidence_audit_pass': True, 'checks': len(checks)}))


if __name__ == '__main__': main()
