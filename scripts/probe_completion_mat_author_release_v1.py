"""Bounded anonymous probe of the two release links in the pinned author README."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_author_release_probe_v1'
SOURCE = ROOT / 'outputs/completion_mat_source_review_v1_r1'
URLS = [
    ('author_FFHQ512_file', 'https://mycuhk-my.sharepoint.com/:u:/g/personal/1155137927_link_cuhk_edu_hk/ESwt5gvPs4JOvC76WAEDfb4BSJZNy-qsfJSUZz2kTxYyWw?e=71nHCJ'),
    ('author_models_folder', 'https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/EuY30ziF-G5BvwziuHNFzDkBVC6KBPRg69kCeHIu-BXORA?e=7OwJyE'),
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class PublicRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        self.rows = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.rows.append({'status': code, 'from': req.full_url, 'to': newurl})
        host = urllib.parse.urlsplit(newurl).hostname
        if len(self.rows) > 5 or urllib.parse.urlsplit(newurl).scheme != 'https' or host != 'mycuhk-my.sharepoint.com':
            raise urllib.error.HTTPError(req.full_url, code, 'Public probe stops before external authentication', headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main():
    assert not OUT.exists(), 'Retain prior probes'
    acquisition = json.loads((SOURCE / 'acquisition.json').read_text())
    assert acquisition['complete'] and acquisition['source_commit'] == 'd273d891ecdad2e1df106516423a75bc45b2d800'
    readme = (SOURCE / 'README.md').read_text(encoding='utf-8')
    assert sha(SOURCE / 'README.md') == next(row['sha256'] for row in acquisition['files'] if row['file'] == 'README.md')
    assert all(url in readme for _, url in URLS)
    context = ssl.create_default_context(cafile=str(ROOT / 'scratch/gcloud_windows_trust.pem'))
    start = time.monotonic(); OUT.mkdir(); rows = []
    for label, url in URLS:
        assert time.monotonic() - start < 100
        redirects = PublicRedirects()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context), redirects)
        req = urllib.request.Request(url, headers={'User-Agent': 'Forensic-DGP-Thesis-Public-Release-Audit/1'})
        row = {'label': label, 'source_url': url, 'anonymous_request': True, 'credentials_or_cookies_supplied': False}
        try:
            with opener.open(req, timeout=30) as response:
                row.update({'status': response.status, 'final_url': response.url,
                            'content_type': response.headers.get('Content-Type'),
                            'content_length': response.headers.get('Content-Length'),
                            'content_disposition': response.headers.get('Content-Disposition')})
                limit = 100000 if 'text/' in (row['content_type'] or '') else 512
                body = response.read(limit)
                row['response_prefix_bytes'] = len(body)
                row['response_prefix_sha256'] = hashlib.sha256(body).hexdigest()
                with (OUT / (label + '.response_prefix')).open('xb') as stream:
                    stream.write(body)
                row['full_checkpoint_downloaded'] = False
                row['possible_direct_checkpoint_header'] = 'text/' not in (row['content_type'] or '') and body.startswith(b'\x80')
        except Exception as error:
            row.update({'error_class': type(error).__name__, 'error': str(error), 'status': getattr(error, 'code', None),
                        'full_checkpoint_downloaded': False, 'possible_direct_checkpoint_header': False})
        row['redirects'] = redirects.rows
        rows.append(row)
    receipt = {'complete': True, 'probe_utc': datetime.now(timezone.utc).isoformat(),
               'probe_source_sha256': sha(Path(__file__)), 'author_README_sha256': sha(SOURCE / 'README.md'),
               'source_repository': 'https://github.com/fenglinglwb/MAT',
               'requests': rows, 'checkpoint_downloaded': False, 'network_pickle_loaded': False,
               'source_executed': False, 'model_or_gradient_calls': 0, 'optimizer_updates': 0,
               'app_changes': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    with (OUT / 'probe.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'complete': True, 'requests': [{k: row.get(k) for k in
                     ['label', 'status', 'error', 'possible_direct_checkpoint_header', 'content_type']} for row in rows],
                     'checkpoint_downloaded': False, 'seconds': receipt['seconds']}, indent=2))


if __name__ == '__main__':
    main()
