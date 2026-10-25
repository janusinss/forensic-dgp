"""Manual exclusive byte-copy of the exact two U96 initializer files; no model."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
sys.dont_write_bytecode = True

R1_PROTOCOL_SHA = 'ffa457e9b7573537fb006c02910cd109af0e17d1b4b7a576502c9f7aa6d14cdf'
R1_RESULT_SHA = '6fe0fd6d769cb6ddf95142a1569013320f591b3eb5f9addfba233c894fc8c4c0'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def install(root, expected_sha):
    start = time.monotonic()
    root = Path(root).absolute()
    assert platform.system() == 'Linux', 'Install manually in VM SSH'
    raw = (root / 'protocol.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha
    protocol = json.loads(raw)
    helper = root / 'research/codeformer_u96_postfusion_smoothanchors_v1/packet.py'
    binding = protocol['files'][helper.relative_to(root).as_posix()]
    assert helper.stat().st_size == binding['bytes'] and digest(helper) == binding['sha256']
    # Verified source import under -B/dont_write_bytecode; cached bytecode is never needed.
    spec = importlib.util.spec_from_file_location('verified_learning_packet', helper)
    packet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packet)
    root = packet.directory(root)
    assert root == packet.directory(Path.home() / 'forensic-dgp' / packet.STAGE_NAME)
    p = packet.verify(root, expected_sha, missing_weights_for_install=True)
    assert not (root / 'install.json').exists() and not (root / 'outputs').exists()
    assert all(not (root / value['path']).exists() for value in p['model'].values())
    source = packet.directory(Path.home() / 'forensic-dgp/cctv_codeformer_u96_postfusion_preflight_v1_r1_vm')
    assert packet.sha256(source / 'protocol.json') == R1_PROTOCOL_SHA
    result_path = source / 'outputs/u96_postfusion_preflight_v1_r1/result.json'
    assert packet.sha256(result_path) == R1_RESULT_SHA
    result = packet.strict_json(result_path.read_bytes())
    assert result['diagnostic_pass'] is True
    needed = sum(row['bytes'] for row in p['model'].values()) + p['limits']['initial_free_disk_bytes']
    assert shutil.disk_usage(root).free >= needed, 'Leave8GiB free after exclusive model copies'
    copied = []
    for key, row in p['model'].items():
        src = packet.regular(source / ('weights/codeformer_r2.pth' if key == 'r2' else 'weights/codeformer_u96_adapter.pth'))
        assert src.stat().st_size == row['bytes'] and packet.sha256(src) == row['sha256']
        target = root / row['path']
        if not target.parent.exists():
            target.parent.mkdir()
        packet.directory(target.parent)
        assert not target.exists()
        with src.open('rb') as original, target.open('xb') as output:
            while True:
                assert time.monotonic() - start < 180, 'Finite180s installation stop'
                assert shutil.disk_usage(root).free >= p['limits']['initial_free_disk_bytes']
                block = original.read(1024**2)
                if not block:
                    break
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        assert packet.pin(root, target) == row and packet.sha256(src) == row['sha256']
        copied.append({'key': key, 'source': str(src), 'installed': row})
    packet.verify(root, expected_sha)
    receipt = {'complete': True, 'protocol_sha256': expected_sha, 'source_R1_result_sha256': R1_RESULT_SHA,
               'copies': copied, 'seconds': time.monotonic() - start, 'neural_optimizer_training_calls': 0}
    with (root / 'install.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    install(args.root, args.protocol_sha)
