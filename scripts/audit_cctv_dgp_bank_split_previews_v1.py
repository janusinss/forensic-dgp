"""Independent preview RGB validation while the full metric audit runs separately."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pixels(path, mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256, 256)
        return np.asarray(image.convert(mode)).copy()


def audit(root, imported, receipt):
    start = time.monotonic(); root = root.resolve()
    assert not receipt.exists()
    imp = read(imported); assert imp['complete'] and imp['every_member_path_and_byte_hash_checked']
    assert sha(root/'protocol.json') == imp['protocol_sha256']
    assert sha(root/'export_manifest.json') == imp['export_manifest_sha256']
    p = read(root/'protocol.json'); manifest = read(root/'export_manifest.json')
    sys.path.insert(0, str(root))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_raw_metrics import deliver
    out = root/'outputs/bank_comparison_v1'
    refs = {r['id']: r for r in p['references']}
    paired, native = 0, 0
    bound = {}
    labels = ['baseline'] + [route+'50' for route in p['selected_routes']]
    if (out/'B50_bank_disabled/metrics.json').exists():
        labels.append('B50_bank_disabled')

    def bind(path):
        relative = path.relative_to(root).as_posix()
        assert sha(path) == manifest['files'][relative]
        bound[relative] = manifest['files'][relative]

    for label in labels:
        metrics = read(out/label/'metrics.json'); bind(out/label/'metrics.json')
        rows = {r['id']: r for r in metrics['rows']}
        for begin in range(0, len(p['cases']), 5):
            cases = p['cases'][begin:begin+5]
            if cases[0]['id'] not in p['preview_case_ids']:
                continue
            ids = [c['id'] for c in cases]
            baseline_path = out/'baseline/packs'/f'b{begin//5:03d}.npz'; bind(baseline_path)
            baseline, _ = unpack(baseline_path, ids)
            pack_path = out/label/'packs'/f'b{begin//5:03d}.npz'; bind(pack_path)
            raw, _ = unpack(pack_path, ids, baseline=None if label == 'baseline' else baseline)
            for c, a in zip(cases, raw):
                camera = pixels(root/c['input'])
                mask = pixels(root/refs[c['source_person_or_reference']]['observed'], 'L') > 0
                assert a.dtype == np.float32 and np.isfinite(a).all() and 0 <= a.min() <= a.max() <= 1
                assert np.array_equal(a[~mask], (camera.astype(np.float32)/np.float32(255))[~mask])
                png_path = out/label/'previews'/(c['id']+'.png'); bind(png_path)
                png = pixels(png_path)
                assert np.array_equal(deliver(a, camera, mask), png)
                assert hashlib.sha256(a.tobytes()).hexdigest() == rows[c['id']]['raw_RGB_float32_sha256']
                assert hashlib.sha256(png.tobytes()).hexdigest() == rows[c['id']]['PNG_RGB_sha256']
                paired += 1
        rows = {r['id']: r for r in metrics['native_unpaired_rows']}
        for c in p['native_development']:
            path = out/label/'native'/(c['id']+'.npy'); bind(path)
            a = np.load(path, allow_pickle=False)
            camera = pixels(root/c['input']); mask = pixels(root/c['observed'], 'L') > 0
            assert a.dtype == np.float32 and a.shape == (256,256,3) and np.isfinite(a).all()
            assert 0 <= a.min() <= a.max() <= 1
            assert np.array_equal(a[~mask], (camera.astype(np.float32)/np.float32(255))[~mask])
            png_path = out/label/'native'/(c['id']+'.png'); bind(png_path)
            png = pixels(png_path)
            assert np.array_equal(deliver(a, camera, mask), png)
            assert hashlib.sha256(a.tobytes()).hexdigest() == rows[c['id']]['raw_RGB_float32_sha256']
            assert hashlib.sha256(png.tobytes()).hexdigest() == rows[c['id']]['PNG_RGB_sha256']
            native += 1
    record = dict(complete=True, checker_sha256=sha(__file__), import_receipt_sha256=sha(imported),
        protocol_sha256=imp['protocol_sha256'], manifest_sha256=imp['export_manifest_sha256'],
        scope='planned preview RGB/PNG compositions only; full metric and state audit separate',
        saved_paired_preview_compositions_checked=paired, native_compositions_checked=native,
        exact_preview_file_bindings=bound, full_metric_and_gate_audit_pending=True,
        local_model_gradient_or_training_calls=0, quality_qualified=False, goal_complete=False,
        seconds=time.monotonic()-start)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, paired_previews=paired, native_compositions=native), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--import-receipt', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(); audit(args.root, args.import_receipt, args.receipt)
