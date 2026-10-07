"""Bounded official-source acquisition for a separate completion comparison."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_source_review_v1'
COMMIT = 'd273d891ecdad2e1df106516423a75bc45b2d800'
FILES = ('README.md', 'LICENSE', 'generate_image.py', 'legacy.py', 'requirements.txt')


def main():
    assert not OUT.exists(), 'Preserve an earlier source review'
    OUT.mkdir()
    context = ssl.create_default_context(cafile=str(ROOT / 'scratch/gcloud_windows_trust.pem'))
    start = time.monotonic()
    rows = []
    for name in FILES:
        assert time.monotonic() - start < 150
        url = 'https://raw.githubusercontent.com/fenglinglwb/MAT/' + COMMIT + '/' + name
        request = urllib.request.Request(url, headers={'User-Agent': 'Forensic-DGP-Thesis-Source-Audit/1'})
        with urllib.request.urlopen(request, timeout=20, context=context) as response:
            assert response.status == 200 and response.url == url
            data = response.read(200001)
            assert 0 < len(data) <= 200000
            data.decode('utf-8-sig')
        with (OUT / name).open('xb') as stream:
            stream.write(data)
        rows.append({'file': name, 'source_url': url, 'bytes': len(data),
                     'sha256': hashlib.sha256(data).hexdigest()})
    receipt = {'complete': True, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
               'source_commit': COMMIT, 'source_repository': 'https://github.com/fenglinglwb/MAT',
               'files': rows, 'source_only': True, 'checkpoint_downloaded': False,
               'source_executed': False, 'model_or_gradient_calls': 0, 'optimizer_updates': 0,
               'app_changes': False, 'usefulness_established': False, 'goal_complete': False,
               'seconds': time.monotonic() - start}
    with (OUT / 'acquisition.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
