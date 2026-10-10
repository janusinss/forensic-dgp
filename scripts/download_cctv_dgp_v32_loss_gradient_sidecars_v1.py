"""Fetch only two tiny receipts for the already completed human archive download."""
from datetime import datetime, timezone
import json
from pathlib import Path

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_sidecar_download'
STEM = 'cctv-dgp-v32-loss-gradient-v1'
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'


def main():
    status = json.loads((ROOT / 'outputs/cctv_dgp_v32_loss_gradient_status_v1/status_receipt.json').read_text())
    remote = status['remote']['export_receipt']
    assert status['complete'] and remote['complete'] and remote['optimizer_updates'] == 0
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    assert archive.is_file() and archive.stat().st_size == remote['bytes']
    assert transport.sha(archive) == remote['archive_sha256'], 'Completed human transfer only'
    OUT.mkdir(exist_ok=False)
    transport.OUT = OUT
    placements = []
    for index, name in enumerate([STEM + '-export.json', STEM + '-results.tar.gz.sha256']):
        local = OUT / name
        transport.run(['compute', 'scp'] + transport.BASE +
                      ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag=' + KEY,
                       'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name, str(local)],
                      'sidecar_' + str(index), 60)
        assert local.is_file() and not local.is_symlink() and local.stat().st_size < 16000
        if index == 0:
            assert json.loads(local.read_text()) == remote
        else:
            assert local.read_text().strip().split() == [remote['archive_sha256'], archive.name]
        destination = ROOT / 'outputs' / name
        reused = destination.exists()
        if not reused:
            try:
                with destination.open('xb') as stream:
                    stream.write(local.read_bytes())
            except FileExistsError:
                reused = True
        assert transport.sha(destination) == transport.sha(local), 'Retain different existing sidecar'
        placements.append({'name': name, 'sha256': transport.sha(destination), 'reused': reused})
    receipt = {'complete': True, 'snapshot_UTC': datetime.now(timezone.utc).isoformat(),
               'reader_sha256': transport.sha(Path(__file__)),
               'archive_sha256': remote['archive_sha256'], 'archive_bytes': remote['bytes'],
               'archive_downloaded_by_agent': False, 'receipt_downloads': 2,
               'placements': placements, 'VM_writes': 0, 'training_started_by_agent': False,
               'local_model_or_gradient_calls': 0, 'goal_complete': False}
    with (OUT / 'download_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
