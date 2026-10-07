"""Download the explicitly requested R2 return; no VM writes or training launch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_download_v1'
STEM = 'cctv-dgp-feature-fusion-v32-r2'
EXPECTED_SHA = '576b3897a9a9589ddad8896e71c827481086b26ffab6e6cbeadb9a5dbc4032e9'
EXPECTED_BYTES = 1324635165
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Retain earlier downloads and transport logs'
    assert shutil.disk_usage(ROOT).free >= 6 * 1024**3, 'Local download and import headroom'
    OUT.mkdir()
    transport.OUT = OUT
    names = [STEM + '-export.json', STEM + '-results.tar.gz.sha256', STEM + '-results.tar.gz']
    for index, name in enumerate(names):
        target = OUT / name
        transport.run(['compute', 'scp'] + transport.BASE +
                      ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag=' + KEY,
                       'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name,
                       str(target)], 'download_' + str(index + 1), 1800 if index == 2 else 90)
        assert target.is_file() and not target.is_symlink()
        if index == 0:
            assert target.stat().st_size < 16000
            exported = json.loads(target.read_text(encoding='utf-8'))
            assert exported['complete'] and exported['archive_sha256'] == EXPECTED_SHA
            assert exported['bytes'] == EXPECTED_BYTES and exported['optimizer_updates'] == 50
            assert exported['training_success_not_implied'] and exported['failure_present']
            assert not exported['run_results_present']
        elif index == 1:
            assert target.stat().st_size < 1000
            assert target.read_text().strip().split() == [EXPECTED_SHA, names[2]]
        else:
            assert target.stat().st_size == EXPECTED_BYTES and sha(target) == EXPECTED_SHA
    placements = []
    for name in names:
        source, target = OUT / name, ROOT / 'outputs' / name
        expected = sha(source)
        reused = target.exists()
        if reused:
            assert target.is_file() and not target.is_symlink() and sha(target) == expected
        else:
            with source.open('rb') as incoming, target.open('xb') as local:
                shutil.copyfileobj(incoming, local, 4 * 1024**2)
            assert target.stat().st_size == source.stat().st_size and sha(target) == expected
        placements.append({'name': name, 'sha256': expected, 'bytes': target.stat().st_size,
                           'already_present_and_matching': reused})
    receipt = {'complete': True, 'snapshot_UTC': datetime.now(timezone.utc).isoformat(),
               'reader_sha256': sha(Path(__file__)), 'archive_sha256': EXPECTED_SHA,
               'archive_bytes': EXPECTED_BYTES, 'placements': placements,
               'files_sha256': {p.relative_to(OUT).as_posix(): sha(p)
                                for p in sorted(OUT.rglob('*')) if p.is_file()},
               'VM_writes': 0, 'training_started_by_agent': False,
               'local_neural_or_gradient_calls': 0, 'quality_acceptance_not_implied': True,
               'seconds': time.monotonic() - started, 'goal_complete': False}
    with (OUT / 'download_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({k: receipt[k] for k in ['complete', 'archive_sha256', 'archive_bytes',
                                           'VM_writes', 'training_started_by_agent', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
