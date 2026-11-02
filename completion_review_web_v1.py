"""Read-only retained completion comparison. No neural models or generation."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
RELATIVE = 'outputs/completion_mat_mirror_comparison_v1'
CANDIDATE_ROOT = ROOT / 'outputs/completion_migan256_comparison_v1'
PINS = {
    'protocol.json': 'd3b35fe043c198007515f3dc0a221b7c7b369b7edd8d7b9bd45122cc550cf76f',
    'results.json': '856e8a328e711b7ec69639e375f73fc267901b4f0e076cbe8a6a8e9892f25a7c',
    'visual_review.json': '36848b0c2ea718f1b38ecb26f7f61d7ee0cbf5941d711e64598f94885bc9d55e',
    'independent_saved_output_audit.json': 'a13373d57ff148e02c1532053c907faef5368249bd404e665f41c0f99dbb2a23',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def safe_file(root, relative):
    need(type(relative) is str and relative and '\\' not in relative and ':' not in relative, 'Invalid retained relative path')
    rel = PurePosixPath(relative)
    need(not rel.is_absolute() and all(x not in ('.', '..') for x in rel.parts), 'Escaping retained path')
    current = root
    for part in rel.parts:
        current = current / part
        info = current.lstat()
        need(not current.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400, 'Links are not admitted')
    need(current.resolve().is_relative_to(root.resolve()) and stat.S_ISREG(current.stat().st_mode), 'Regular retained file required')
    return current


def checked(root, relative, digest, png=False):
    path = safe_file(root, relative)
    need(path.stat().st_size <= 4 * 1024 ** 2, 'Retained file exceeds bound')
    body = path.read_bytes()
    need(hashlib.sha256(body).hexdigest() == digest, 'Retained evidence fingerprint changed: ' + relative)
    size = None
    if png:
        need(body[:8] == b'\x89PNG\r\n\x1a\n' and len(body) >= 33, 'Retained PNG required')
        size = list(struct.unpack('>II', body[16:24]))
    return body, size


def build_catalogue(asset_root, migan_audit_sha=None):
    asset_root = asset_root.resolve(strict=True)
    docs = {}
    for name, digest in PINS.items():
        body, _ = checked(asset_root, RELATIVE + '/' + name, digest)
        docs[name] = json.loads(body)
    protocol, result, review = (docs[n] for n in ('protocol.json', 'results.json', 'visual_review.json'))
    need(result['protocol_sha256'] == PINS['protocol.json'], 'Results/protocol join differs')
    need(review['protocol_sha256'] == PINS['protocol.json'] and review['results_sha256'] == PINS['results.json'], 'Review joins differ')
    need(review['independent_saved_output_audit_sha256'] == PINS['independent_saved_output_audit.json'], 'Audit join differs')
    need(len(protocol['cases']) == len(result['rows']) == len(review['rows']) == 32, 'Exact frozen comparison required')
    result_rows = {r['id']: r for r in result['rows']}
    review_rows = {r['id']: r for r in review['rows']}
    assets = {}
    cases = []

    def add(relative, digest, label):
        key = str(len(assets) + 1)
        body, size = checked(asset_root, relative, digest, png=True)
        assets[key] = {'relative': relative, 'sha256': digest, 'bytes': len(body), 'size': size, 'label': label}
        return {'url': '/asset/' + key + '.png', 'download': '/download/' + key + '.png', 'size': size, 'sha256': digest}

    for entry in protocol['cases']:
        ident = entry['id']
        need(re.fullmatch(r'[A-Za-z0-9_]+', ident) is not None, 'Unexpected fixed case ID')
        output, note = result_rows[ident], review_rows[ident]
        panels = []
        for field, label in [('input', 'Original / prepared 256'), ('reviewed', 'Reviewed removal mask'), ('output', 'CodeFormer completion baseline')]:
            relative = entry[field]
            image = add(relative, protocol['sources_sha256'][relative], label)
            need(image['size'] == [256, 256], 'Exact 256 comparison panel required')
            panels.append(dict(image, label=label))
        image = add(RELATIVE + '/' + output['output'], output['output_sha256'], 'Converted MAT comparison')
        need(image['size'] == [256, 256], 'Exact 256 MAT panel required')
        panels.append(dict(image, label='Converted MAT comparison'))
        cases.append({'id': ident, 'family': entry['family'], 'condition': entry['condition'],
                      'panels': panels, 'observation': note['observation'],
                      'exact_outside_reviewed_mask': output['exact_outside_reviewed_mask'],
                      'empty_mask_bypass': output['empty_mask_bypass'],
                      'MAT_seconds_historical': output['seconds']})
    sheets = []
    for i, sheet in enumerate(review['sheets'], 1):
        image = add(RELATIVE + '/' + sheet['path'], sheet['sha256'], 'Historical comparison sheet ' + str(i))
        sheets.append(dict(image, label='Sheet ' + str(i)))
    need(len(sheets) == 8, 'All eight historical sheets required')
    catalogue = {'mode': 'retained_outputs_only', 'cases': cases, 'sheets': sheets,
                 'case_count': len(cases), 'sheet_count': len(sheets),
                 'decision': review['decision'], 'assisted_quality_qualified': False,
                 'automatic_quality_qualified': False, 'neural_or_training_calls': 0,
                 'scope': 'Previously exposed development photographs and synthetic degradations; no aligned hidden clean reference or native CCTV qualification.',
                 'document_sha256': PINS}
    if migan_audit_sha is not None:
        attach_migan(catalogue, assets, migan_audit_sha)
    return catalogue, assets


def candidate_relative(value):
    need(type(value) is str and value, 'Candidate file path required')
    path = Path(value)
    if path.is_absolute():
        need(path.resolve().is_relative_to(CANDIDATE_ROOT.resolve()), 'Candidate path escapes its fixed root')
        value = path.resolve().relative_to(CANDIDATE_ROOT.resolve()).as_posix()
    # safe_file separately rejects backslashes, traversal, links and drive paths.
    return value


def attach_migan(catalogue, assets, audit_sha):
    need(type(audit_sha) is str and re.fullmatch(r'[0-9a-f]{64}', audit_sha), 'Exact independent audit SHA256 required')
    audit_body, _ = checked(CANDIDATE_ROOT, 'independent_output_audit.json', audit_sha)
    audit = json.loads(audit_body)
    need(audit.get('complete') is True and audit.get('cases') == 32, 'Complete independent 32-case audit required')
    need(audit.get('model_qualified') is False and type(audit.get('training_calls')) is int and audit['training_calls'] == 0, 'No qualification or training is admitted')
    need(audit.get('protocol', {}).get('sha256') == PINS['protocol.json'], 'Fixed historical case/mask protocol binding differs')
    results_pin = audit.get('results', {})
    need(type(results_pin) is dict and 'path' in results_pin and 'sha256' in results_pin, 'Audit-bound results required')
    results_relative = candidate_relative(results_pin['path'])
    need(results_relative == 'results.json', 'Only fixed candidate results are admitted')
    results_body, _ = checked(CANDIDATE_ROOT, results_relative, results_pin['sha256'])
    need(len(results_body) == results_pin.get('bytes'), 'Candidate results byte count differs')
    results = json.loads(results_body)
    need(results.get('complete') is True and results.get('assisted_quality_qualification') is False, 'Candidate result status differs')
    output_rows = audit.get('outputs', [])
    result_rows = results.get('rows', [])
    need(type(output_rows) is list and len(output_rows) == 32 and type(result_rows) is list and len(result_rows) == 32, 'Exact32 candidate rows required')
    outputs = {r['id']: r for r in output_rows}
    result_by_id = {r['id']: r for r in result_rows}
    expected_ids = {r['id'] for r in catalogue['cases']}
    need(set(outputs) == set(result_by_id) == expected_ids and len(outputs) == 32, 'Candidate case IDs must match baseline exactly')
    additions = []
    for case in catalogue['cases']:
        row, result = outputs[case['id']], result_by_id[case['id']]
        need(row.get('exact_outside_reviewed_mask') is True and result.get('exact_outside_reviewed_mask') is True, 'Candidate preservation audit failed')
        need(row.get('empty_mask_bypass') is result.get('empty_mask_bypass') and row.get('empty_mask_bypass') is case['empty_mask_bypass'], 'Candidate empty-mask controls differ')
        file_pin = row['file']
        relative = candidate_relative(file_pin['path'])
        need(relative == 'images/' + case['id'] + '.png' and result['output'] == relative, 'Fixed candidate output filename required')
        need(file_pin['sha256'] == result['output_sha256'], 'Result/audit output fingerprint differs')
        body, size = checked(CANDIDATE_ROOT, relative, file_pin['sha256'], png=True)
        need(size == [256, 256] and len(body) == file_pin['bytes'], 'Exact RGB256 candidate output required')
        key = str(len(assets) + len(additions) + 1)
        additions.append((key, {'relative': relative, 'sha256': file_pin['sha256'], 'bytes': len(body), 'size': size, 'label': 'MI-GAN FFHQ256 experimental', 'namespace': 'migan'}, case,
                          {'url': '/asset/' + key + '.png', 'download': '/download/' + key + '.png', 'size': size, 'sha256': file_pin['sha256'], 'label': 'MI-GAN FFHQ256 · experimental'}))
    for key, asset, case, panel in additions:
        assets[key] = asset
        case['panels'].append(panel)
        case['MI_GAN_seconds_historical'] = result_by_id[case['id']].get('seconds')
    catalogue['mode'] = 'audited_saved_comparison_with_migan'
    catalogue['candidate'] = {'name': 'MI-GAN FFHQ256', 'independent_audit_sha256': audit_sha, 'results_sha256': results_pin['sha256'], 'model_qualified': False, 'new_generation_enabled': False}


HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completion review — retained comparisons</title><style>
:root{color-scheme:dark;font:15px/1.5 system-ui,sans-serif;background:#060b18;color:#e9edf5}*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:clamp(25px,4vw,38px);margin:4px 0 10px}p{max-width:850px}.eyebrow{color:#86b9e8;font-size:12px;letter-spacing:.1em;text-transform:uppercase}.notice{border-left:3px solid #c19b55;background:#131b29;padding:14px 18px;color:#e0d6bd}.filters{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0}label{display:grid;gap:5px;font-size:13px;color:#b9c9dd}select,button,a.button{font:inherit;background:#111d30;color:#eff6ff;border:1px solid #34485f;border-radius:6px;padding:9px 12px}select{max-width:100%;min-width:170px}button,a{cursor:pointer}a{color:#9acbff}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #c5e0ff;outline-offset:3px}.case-select{flex:1;min-width:0}.case-select select{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.card{background:#0c1424;border:1px solid #24354c;border-radius:8px;padding:12px;min-width:0}.card h2{font-size:14px;min-height:42px;margin:0 0 8px}.card img{width:256px;max-width:100%;height:auto;display:block;margin:auto;background:#7f7f7f;image-rendering:auto}.card a{display:inline-block;margin-top:12px;font-size:13px}.details{padding:18px 0;color:#bed0e4}.details strong{color:#e7effa}.sheet-links{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 24px}.footer{font-size:13px;color:#9cafc7;border-top:1px solid #24354c;padding-top:16px}#error{color:#ffb1b1}#count{align-self:end;color:#a6b8d0;white-space:nowrap}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:480px){main{padding:16px}.grid{grid-template-columns:1fr}.filters>*{width:100%}.card h2{min-height:0}select{width:100%;min-width:0}.card img{width:256px}}@media(prefers-reduced-motion:no-preference){button,a{transition:background .12s}}
</style><main><div class="eyebrow">Face mask + completion / local research review</div><h1>Review what completion already produces</h1><p>Compare the same input and reviewed removal area. Saved outputs are shown for evaluation; no candidate is automatically adopted.</p><p id="candidate-status" class="eyebrow">Retained baseline comparison</p><p class="notice"><strong>Saved outputs only — generation is disabled.</strong> All displayed candidates remain unqualified across the full covering scope. Hidden features are plausible estimates, not verified identity.</p><div id="error" role="alert"></div><div class="filters"><label>Covering type<select id="family" aria-label="Covering type"><option value="all">All types</option></select></label><label>Input condition<select id="condition" aria-label="Input condition"><option value="all">Both conditions</option><option value="original_photo">Original photograph</option><option value="synthetic_degraded_photo">Synthetically degraded</option></select></label><label class="case-select">Case<select id="case" aria-label="Case"></select></label><span id="count"></span></div><div class="grid" id="panels"></div><section class="details" aria-live="polite"><strong id="case-heading"></strong><p id="note"></p><p id="preserve"></p></section><h2>Original comparison sheets</h2><p>These include the historical orange mask overlays. All eight frozen sheets are available without regenerating images.</p><div id="sheets" class="sheet-links"></div><p class="footer">Prepared / delivered panels are 256 × 256. Pixels are served unchanged; the browser can scale their display to fit your screen. Source photographs and synthetic degradations are development evidence, not native CCTV evaluation. Restoration is Off in these retained comparisons to isolate completion. No model or cloud connection is used by this viewer.</p></main><script>
'use strict';let data;const $=id=>document.getElementById(id);function choose(){const rows=data.cases.filter(c=>($('family').value==='all'||c.family===$('family').value)&&($('condition').value==='all'||c.condition===$('condition').value));$('case').replaceChildren(...rows.map(c=>{let o=document.createElement('option');o.value=c.id;o.textContent=c.id.replaceAll('_',' ');return o}));$('count').textContent=rows.length+' cases';show()}
function show(){const c=data.cases.find(r=>r.id===$('case').value);$('panels').replaceChildren();if(!c){$('case-heading').textContent='No matching cases';$('note').textContent='';$('preserve').textContent='';return}for(const p of c.panels){const card=document.createElement('article');card.className='card';const h=document.createElement('h2');h.textContent=p.label;const img=document.createElement('img');img.src=p.url;img.alt=p.label+' for '+c.id;img.width=256;img.height=256;const a=document.createElement('a');a.href=p.download;a.textContent='Download exact PNG';a.setAttribute('download','');card.append(h,img,a);$('panels').append(card)}$('case-heading').textContent=c.id.replaceAll('_',' ')+' · '+(c.condition==='original_photo'?'Original photograph':'Synthetic degradation');$('note').textContent='Saved MAT review: '+c.observation;$('preserve').textContent=(c.empty_mask_bypass?'Empty-mask control: generator bypassed. ':'')+'Saved audit: pixels outside the reviewed removal area were copied exactly. That check does not establish useful anatomy or hidden accuracy.'}
fetch('/api/catalogue',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Retained evidence is unavailable');return r.json()}).then(d=>{data=d;if(d.candidate)$('candidate-status').textContent='MI-GAN FFHQ256 saved candidate added · independent preservation audit passed · usefulness unqualified';for(const f of [...new Set(d.cases.map(c=>c.family))]){const o=document.createElement('option');o.value=f;o.textContent=f.replaceAll('_',' ');$('family').append(o)}$('family').addEventListener('change',choose);$('condition').addEventListener('change',choose);$('case').addEventListener('change',show);for(const s of d.sheets){const a=document.createElement('a');a.href=s.url;a.target='_blank';a.rel='noopener';a.textContent=s.label;$('sheets').append(a)}choose()}).catch(e=>{$('error').textContent=e.message});
</script></html>"""


def make_handler(asset_root, catalogue, assets):
    class Handler(BaseHTTPRequestHandler):
        # Keep image responses on a persistent framed connection. Closing an
        # HTTP/1.0 socket immediately after a large Windows send can surface as
        # a Chromium connection reset although a synchronous reader gets bytes.
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            pass

        def send(self, status, body, content_type='application/json', filename=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if filename:
                self.send_header('Content-Disposition', 'attachment; filename="' + filename + '"')
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):
            host = self.headers.get('Host', '')
            if host != '127.0.0.1:' + str(self.server.server_port):
                return self.send(403, b'{"error":"Loopback host required"}')
            route = urlsplit(self.path).path
            if route == '/':
                return self.send(200, HTML.encode(), 'text/html; charset=utf-8')
            if route == '/health':
                return self.send(200, json.dumps({'ready': True, 'mode': catalogue['mode'], 'generation_enabled': False, 'cases': 32, 'model_qualified': False}).encode())
            if route == '/api/catalogue':
                return self.send(200, json.dumps(catalogue).encode())
            if route == '/favicon.ico':
                return self.send(204, b'', 'image/x-icon')
            match = re.fullmatch(r'/(asset|download)/([1-9][0-9]*)\.png', route)
            if match and match[2] in assets:
                entry = assets[match[2]]
                try:
                    source_root = CANDIDATE_ROOT if entry.get('namespace') == 'migan' else asset_root
                    body, _ = checked(source_root, entry['relative'], entry['sha256'], png=True)
                except (ValueError, OSError):
                    return self.send(409, b'{"error":"Retained evidence changed; request refused"}')
                filename = PurePosixPath(entry['relative']).name if match[1] == 'download' else None
                return self.send(200, body, 'image/png', filename)
            return self.send(404, b'{"error":"Unknown retained resource"}')
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-root', type=Path, default=DEFAULT_ASSETS)
    p.add_argument('--port', type=int, default=0)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--migan-audit-sha', help='Opt in only to the exact independently audited MI-GAN saved comparison')
    args = p.parse_args()
    need(0 <= args.port <= 65535, 'Port out of range')
    catalogue, assets = build_catalogue(args.asset_root, args.migan_audit_sha)
    inventory = {'mode': catalogue['mode'], 'document_pins': PINS, 'assets': assets, 'case_count': len(catalogue['cases']), 'sheet_count': 8, 'neural_or_training_calls': 0}
    if args.verify_only:
        print(json.dumps({'complete': True, 'cases': 32, 'sheets': 8, 'pinned_PNG_assets': len(assets), 'mode': catalogue['mode'], 'neural_or_training_calls': 0}))
        return
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(args.asset_root.resolve(), catalogue, assets))
    scratch = ROOT / 'scratch'
    scratch.mkdir(exist_ok=True)
    address = 'http://127.0.0.1:' + str(server.server_port)
    runtime = {'url': address, 'pid': os.getpid(), 'mode': catalogue['mode'], 'generation_enabled': False, 'asset_root': str(args.asset_root.resolve())}
    (scratch / 'completion-review-runtime-v1.json').write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
    (scratch / 'completion-review-assets-v1.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(runtime), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
