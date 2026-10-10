"""Acquire official standalone GAN assets; no neural inference or training.

This is a preparation component, not a restoration model or a training launch.
The release has no published digest. TLS source/release metadata and a locally
computed SHA256 bind the downloaded file; they do not prove unseen provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


BASICSR_COMMIT = '8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a'
GFPGAN_COMMIT = '7552a7791caad982045a7bbe5634bbf1cd5c8679'
MMAGIC_COMMIT = '0a560bba9b79ebe78574e1d4cbbdd0e798e63568'
FFHQ_COMMIT = '4826aa6ea77aa7f1a7802b938ed7c40afb985cda'
NVIDIA_COMMIT = 'bf0fe0baba9fc7039eae0cac575c1778be1ce3e3'
WEIGHT_NAME = 'StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth'
WEIGHT_BYTES = 204535545
WEIGHT_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/' + WEIGHT_NAME
RELEASE_API = 'https://api.github.com/repos/TencentARC/GFPGAN/releases/tags/v0.1.0'
RELEASE_ID, ASSET_ID = 44621114, 38600867
MAX_SECONDS = 420
SOURCES = {
    'stylegan2_arch.py': f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{BASICSR_COMMIT}/basicsr/archs/stylegan2_arch.py',
    'upfirdn2d.py': f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{BASICSR_COMMIT}/basicsr/ops/upfirdn2d/upfirdn2d.py',
    'fused_act.py': f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{BASICSR_COMMIT}/basicsr/ops/fused_act/fused_act.py',
    'LICENSE_BasicSR.txt': f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{BASICSR_COMMIT}/LICENSE.txt',
    'LICENSE_GFPGAN.txt': f'https://raw.githubusercontent.com/TencentARC/GFPGAN/{GFPGAN_COMMIT}/LICENSE',
    'README_GFPGAN.md': f'https://raw.githubusercontent.com/TencentARC/GFPGAN/{GFPGAN_COMMIT}/README.md',
    'LICENSE_StyleGAN2_NVIDIA.txt': f'https://raw.githubusercontent.com/NVlabs/stylegan2/{NVIDIA_COMMIT}/LICENSE.txt',
    'README_FFHQ.md': f'https://raw.githubusercontent.com/NVlabs/ffhq-dataset/{FFHQ_COMMIT}/README.md',
    'glean_styleganv2_reference.py': f'https://raw.githubusercontent.com/open-mmlab/mmagic/{MMAGIC_COMMIT}/mmagic/models/editors/glean/glean_styleganv2.py',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def acquire(root):
    root = root.resolve()
    workspace = Path(__file__).resolve().parents[1]
    assert root.is_relative_to(workspace / 'outputs'), 'Use a new workspace outputs directory'
    assert not root.exists(), 'Immutable acquisition: existing destination is refused'
    assert shutil.disk_usage(workspace).free >= 2 * 1024**3, 'Local2GiB reserve before acquisition'
    root.mkdir(parents=True)
    plan = {'format': 'standalone-generative-face-bank-acquisition-v1',
        'source_commit': BASICSR_COMMIT, 'sources': SOURCES, 'checkpoint_url': WEIGHT_URL,
        'checkpoint_expected_bytes': WEIGHT_BYTES, 'release_id': RELEASE_ID, 'asset_id': ASSET_ID,
        'total_seconds_cap': MAX_SECONDS, 'socket_timeout_seconds': 30,
        'maximum_total_download_bytes': 250 * 1024**2,
        'neural_calls': 0, 'optimizer_updates': 0, 'app_promotion': False,
        'prior_is_external': True, 'restoration_encoder_acquired': False,
        'publisher_digest_available_at_metadata_check': False}
    write(root / 'plan.json', plan)
    start = time.monotonic()
    total = 0
    bindings = {}

    def download(url, name, cap, expected=None):
        nonlocal total
        target = root / name
        part = root / (name + '.part')
        count = 0
        request = urllib.request.Request(url, headers={'User-Agent': 'forensic-dgp-research-preparation/1'})
        with urllib.request.urlopen(request, timeout=30) as response, part.open('xb') as stream:
            for block in iter(lambda: response.read(2 * 1024**2), b''):
                assert time.monotonic() - start < MAX_SECONDS, 'Finite acquisition time stop'
                count += len(block)
                total += len(block)
                assert count <= cap and total <= 250 * 1024**2, 'Finite acquisition byte stop'
                stream.write(block)
            status = response.status
        assert count > 0 and status == 200
        if expected is not None:
            assert count == expected, 'Release asset size mismatch'
        part.rename(target)
        bindings[name] = {'source_url': url, 'bytes': count, 'sha256': sha(target)}
        print(json.dumps({'asset': name, 'bytes': count, 'seconds': time.monotonic() - start}), flush=True)
        return target

    try:
        release_path = download(RELEASE_API, 'official_release_metadata.json', 2 * 1024**2)
        release = json.loads(release_path.read_text(encoding='utf-8'))
        assert release['id'] == RELEASE_ID and release['tag_name'] == 'v0.1.0'
        assets = [a for a in release['assets'] if a['id'] == ASSET_ID]
        assert len(assets) == 1
        asset = assets[0]
        assert asset['name'] == WEIGHT_NAME and asset['size'] == WEIGHT_BYTES
        assert asset['browser_download_url'] == WEIGHT_URL
        for name, url in SOURCES.items():
            download(url, name, 1024**2)
        checkpoint = download(WEIGHT_URL, WEIGHT_NAME, WEIGHT_BYTES, WEIGHT_BYTES)
        if asset.get('digest'):
            assert asset['digest'] == 'sha256:' + sha(checkpoint)
        write(root / 'acquisition.json', {'complete': True, 'bindings': bindings,
            'seconds': time.monotonic() - start, 'download_bytes': total,
            'publisher_digest': asset.get('digest'), 'neural_calls': 0, 'optimizer_updates': 0,
            'model_qualification': False, 'pretrained_dataset_overlap_unexcluded': True,
            'source_terms_must_be_retained': True})
    except Exception as exc:
        write(root / 'failure.json', {'exception_type': type(exc).__name__, 'message': str(exc),
            'seconds': time.monotonic() - start, 'complete_bindings': bindings,
            'neural_calls': 0, 'optimizer_updates': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    acquire(parser.parse_args().root)
