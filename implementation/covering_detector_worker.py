"""One frozen CPU covering proposal. Parent owns its 120-second process cap."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .covering_detector import ROOT, load_frozen_detector, need, proposal_from_probability


def canonical(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def read_png(path, mode):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT / 'scratch') and path.is_file() and not path.is_symlink()
         and 0 < path.stat().st_size <= 1024**2, 'Bounded server-owned detector input required.')
    for ancestor in path.parents:
        need(not ancestor.is_symlink(), 'Linked detector input directories are refused.')
        if ancestor == ROOT:
            break
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        need(image.format == 'PNG' and image.mode == mode and image.size == (256, 256),
             'Detector inputs must be exact RGB256 and binary L256 PNGs.')
        array = np.array(image)
    return array, hashlib.sha256(raw).hexdigest()


def status(output, phase, counters, error=None):
    data = {'phase': phase, **counters}
    if error is not None:
        data.update(failed=True, error_type=type(error).__name__, error=str(error)[:1000])
    target = output / 'status.json'
    temporary = target.with_suffix('.json.tmp')
    with temporary.open('wb') as stream:
        stream.write(canonical(data)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)


def run_request(request):
    need(type(request) is dict and set(request) == {'input_path', 'observed_path', 'output_dir'}
         and all(type(v) is str and v for v in request.values()),
         'Expected input_path, observed_path and output_dir strings only.')
    output = Path(request['output_dir']).absolute()
    need(output.is_relative_to(ROOT / 'scratch'), 'Detector output must remain in server-owned scratch.')
    for ancestor in output.parents:
        need(not ancestor.is_symlink(), 'Linked detector output directories are refused.')
        if ancestor == ROOT:
            break
    output.mkdir(parents=True, exist_ok=True)
    need(output.is_dir() and not output.is_symlink() and not any(output.iterdir()),
         'A fresh ordinary detector output directory is required.')
    counters = {'forward_attempts': 0, 'forward_returns': 0}
    status(output, 'validating', counters)
    started = time.monotonic()
    try:
        rgb, image_sha = read_png(request['input_path'], 'RGB')
        observed8, observed_sha = read_png(request['observed_path'], 'L')
        need(np.isin(observed8, (0, 255)).all() and (observed8 == 255).any(),
             'Observed support must contain binary0/255 pixels and a nonempty captured area.')
        observed = observed8 == 255
        status(output, 'loading', counters)
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        model = load_frozen_detector()
        values = rgb.astype(np.float32) / np.float32(255)
        sample = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            counters['forward_attempts'] = 1
            status(output, 'forward_entered', counters)
            logits = model(sample)
            counters['forward_returns'] = 1
            status(output, 'forward_returned', counters)
            need(logits.shape == (1, 1, 256, 256) and logits.dtype == torch.float32
                 and torch.isfinite(logits).all().item(), 'Invalid detector logits.')
            probability = logits.sigmoid()[0, 0].cpu().numpy().copy()
        raw, proposed, metadata = proposal_from_probability(probability, observed)
        paths = {'probability_path': output / 'detector-probability.npy',
                 'raw_mask_path': output / 'detector-raw-mask.png',
                 'proposed_mask_path': output / 'detector-proposed-mask.png'}
        with paths['probability_path'].open('xb') as stream:
            np.save(stream, probability, allow_pickle=False)
        for key, array in (('raw_mask_path', raw), ('proposed_mask_path', proposed)):
            with paths[key].open('xb') as stream:
                Image.fromarray(array).save(stream, format='PNG')
        metadata.update(input_png_sha256=image_sha, observed_support_png_sha256=observed_sha,
                        input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        probability_array_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                        raw_mask_array_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        proposed_mask_array_sha256=hashlib.sha256(proposed.tobytes()).hexdigest(),
                        seconds=time.monotonic() - started, **counters)
        metadata['outputs_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in paths.values()}
        with (output / 'receipt.json').open('xb') as stream:
            stream.write(canonical(metadata))
        status(output, 'complete', counters)
        return {**{key: str(path) for key, path in paths.items()}, 'metadata': metadata}
    except BaseException as exc:
        status(output, 'failed', counters, error=exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args(argv)
    need(args.request.is_file() and not args.request.is_symlink()
         and 0 < args.request.stat().st_size <= 16384, 'Bounded ordinary request JSON required.')
    result = run_request(json.loads(args.request.read_bytes()))
    with args.result.open('xb') as stream:
        stream.write(canonical(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
