"""All saved predictor-only CE V1 comparison sheets, no inference or enhancement."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import argparse, hashlib, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/completion_codeformer_predictor_ce_v1_return'
RETURN = BASE / 'cctv_completion_codeformer_predictor_ce_v1_results'
STAGE = ROOT / 'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
PROTOCOL = '15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'


def need(ok, message):
    if not ok: raise ValueError(message)


def physical(path):
    path = Path(path).absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def sha(path):
    h = hashlib.sha256()
    with physical(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def pin(path): return {'path': str(Path(path).absolute()), 'bytes': physical(path).stat().st_size, 'sha256': sha(path)}
def read(path): return json.loads(physical(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def checkpin(path, row):
    need(physical(path).is_file() and not physical(path).is_symlink() and physical(path).stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Exact source payload ' + str(path))


def pix(path, mode='RGB'):
    with Image.open(physical(path)) as im:
        need(im.format == 'PNG' and im.mode == mode and im.size == (256, 256), 'Exact native256 pixel source')
        return np.array(im, copy=True)


def pixelpin(value):
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'C_order_SHA256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def admitted():
    t = read(BASE / 'transport_audit.json')
    need(t.get('complete') is True, 'Wait for actual complete transport admission')
    need(sha(RETURN / 'protocol.json') == sha(STAGE / 'protocol.json') == PROTOCOL, 'Exact sealed and returned protocol')
    manifest = read(RETURN / 'export_manifest.json')
    need(manifest['protocol_sha256'] == PROTOCOL, 'Exact export protocol')
    return t, read(RETURN / 'protocol.json'), {r['path']: r for r in manifest['files']}


def render():
    start = time.monotonic(); _, p, manifest = admitted()
    audit = read(BASE / 'independent_saved_output_audit.json')
    need(audit['complete'] and audit['independent'] and audit['outside_FP32_PNG_all_exact'] and audit['all_raw512_to_PNG_exact'], 'Complete independent pixel audit before rendering')
    snapshots = audit['actual_snapshots']; terminal = audit['terminal_new_predictor_updates']
    need(snapshots[0] == 0 and terminal == snapshots[-1], 'Exact actual readouts')
    sources = {r['path']: r for r in p['files']}
    folder = BASE / 'visual_sheets'
    need(not folder.exists() and not (BASE / 'visual_sheet_index.json').exists(), 'Preserve older visual sheets/index')
    folder.mkdir()
    try: font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 15); small = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    except OSError: font = small = ImageFont.load_default()
    inputs, outputs, seen, sheets, rows_index = {}, {}, set(), [], []
    cells_total = 0
    def source(relative, mode='RGB'):
        need(relative in sources, 'Pinned source image required'); path = STAGE / relative
        checkpin(path, sources[relative]); inputs[relative] = pin(path)
        return pix(path, mode), {'source_type': 'sealed_source', 'file': inputs[relative]}
    def result(case, kind, cycle):
        relative = 'outputs/u%04d/%s/%s.png' % (cycle, kind, case['id'])
        need(relative in manifest, 'Planned actual output absent ' + relative)
        path = RETURN / relative; checkpin(path, manifest[relative]); outputs[relative] = pin(path); seen.add(relative)
        return pix(path), {'source_type': 'saved_R1_baseline' if cycle == 0 else 'saved_experimental_predictor_only_readout', 'file': outputs[relative], 'cycle': cycle, 'acceptance_endpoint': cycle == 32, 'intermediate_selection_forbidden': cycle in (8, 16), 'model_qualified': False}
    for kind, key, prefix in [('monitor', 'monitor_cases', 'original_monitor'), ('challenge', 'challenge_cases', 'challenge'), ('public_development', 'development_cases', 'public_unpaired')]:
        cases = p[key]; public = kind == 'public_development'
        cycles = [0, terminal] if public else snapshots
        headers = ['Observed input', 'UNPAIRED: no hidden target' if public else 'Known clean TRAIN target', 'Reviewed removal support'] + [('Retained R1 baseline u0' if cycle == 0 else 'Predictor CE u%d EXPERIMENTAL' % cycle) for cycle in cycles]
        columns = len(headers)
        for offset in range(0, len(cases), 4):
            selected = cases[offset:offset + 4]
            canvas = Image.new('RGB', (columns * 256, 116 + 286 * len(selected)), (241, 239, 232)); draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), prefix + ' | fixed protocol order | ordinary observed-input predictions', fill=(10, 20, 25), font=font)
            draw.text((8, 31), 'All outputs experimental. u8/u16 are safety readouts, never selectable checkpoints. Only final u32 is the acceptance endpoint.', fill=(125, 25, 25), font=small)
            draw.text((8, 50), 'Exact saved 256-pixel cells: no enhancement, resampling or omitted failures. Reviewed assisted masks; copied outside pixels.', fill=(40, 40, 40), font=small)
            for column, text in enumerate(headers): draw.text((column * 256 + 5, 80), text, fill=(10, 20, 25), font=small)
            indexed = []
            for row, case in enumerate(selected):
                observed, observed_pin = source(case['input256']); mask, mask_pin = source(case['mask'], 'L')
                need(np.isin(mask, [0, 255]).all() and int((mask == 255).sum()) == case['mask_pixels'], 'Exact reviewed binary support')
                if public:
                    tile = Image.new('RGB', (256, 256), (222, 219, 211)); note = ImageDraw.Draw(tile)
                    for line, text in enumerate(['PUBLIC DEVELOPMENT', 'Unpaired observed photograph', 'NO clean hidden reference', 'No PSNR/SSIM or identity truth', 'Family: ' + str(case.get('family')), 'Empty support = exact bypass' if not mask.any() else 'Manually reviewed removal area']):
                        note.text((10, 66 + line * 24), text, fill=(50, 50, 50), font=small)
                    target = np.array(tile, copy=True); target_pin = {'source_type': 'derived_no_reference_information_tile', 'not_a_reference_image': True}
                else:
                    target, target_pin = source(case['clean256'])
                overlay = observed.copy(); support = mask == 255
                overlay[support] = np.rint(overlay[support].astype(np.float32) * .4 + np.asarray([230, 45, 45], np.float32) * .6).astype(np.uint8)
                cells = [(observed, observed_pin), (target, target_pin), (overlay, {'source_type': 'derived_reviewed_support_overlay', 'input_file': observed_pin['file'], 'mask_file': mask_pin['file'], 'formula': 'Mask255 only round(input*.4+[230,45,45]*.6); outside untouched'})]
                cells += [result(case, kind, cycle) for cycle in cycles]
                need(len(cells) == columns, 'All declared columns present')
                y = 116 + row * 286
                label = case['id'] + ' | ' + str(case.get('family') if public else case.get('mask_kind')) + ' | ' + str(case.get('condition'))
                draw.text((7, y), label, fill=(10, 20, 25), font=small)
                detail = {'id': case['id'], 'kind': kind, 'mask_kind': case.get('mask_kind'), 'family': case.get('family'), 'condition': case.get('condition'), 'style': case.get('style'), 'display_cells': []}
                for column, (array, provenance) in enumerate(cells):
                    need(array.dtype == np.uint8 and array.shape == (256, 256, 3), 'Exact RGB256 display cells')
                    box = [column * 256, y + 22, (column + 1) * 256, y + 278]
                    canvas.paste(Image.fromarray(array), (box[0], box[1])); cells_total += 1
                    detail['display_cells'].append({'column': column, 'label': headers[column], 'box_xyxy': box, 'pixels': pixelpin(array), **provenance})
                indexed.append(detail); rows_index.append(detail)
            path = folder / ('%s_%02d.png' % (prefix, offset // 4 + 1)); canvas.save(path)
            with Image.open(path) as image: saved = np.array(image, copy=True)
            for row in indexed:
                for cell in row['display_cells']:
                    x0, y0, x1, y1 = cell['box_xyxy']
                    need(pixelpin(saved[y0:y1, x0:x1]) == cell['pixels'], 'Exact displayed source/output cell hash')
            sheets.append({'file': pin(path), 'headers': headers, 'size': list(canvas.size), 'cases': indexed, 'all_displayed_cells_exact': True, 'root_visual_review_pending': True})
    actual = {'outputs/u%04d/%s/%s.png' % (row['cycle'], row['kind'], row['id']) for row in audit['all_outputs']}
    need(actual == seen and len(seen) == audit['actual_evaluation_PNGs'], 'All actual evaluated PNGs included exactly')
    need(len(rows_index) == 80 and len(sheets) == 20, 'All48 paired and32public cases across20sheets')
    for row in [*inputs.values(), *outputs.values()]: checkpin(Path(row['path']), row)
    need(not any(name == 'torch' or name.startswith('torch.') or name.startswith('third_party.codeformer') for name in sys.modules), 'No model runtime imported')
    index = {'format': 'completion-codeformer-predictor-ce-v1-all-saved-visual-index-v1', 'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(), 'renderer': pin(__file__), 'protocol': pin(RETURN / 'protocol.json'), 'source_transport': pin(BASE / 'transport_audit.json'), 'independent_pixel_audit': pin(BASE / 'independent_saved_output_audit.json'), 'fixed_case_order': True, 'output_ranking_or_selection': False, 'snapshots': snapshots, 'terminal_cycle': terminal, 'sheets_count': len(sheets), 'unique_cases': len(rows_index), 'paired_cases': 48, 'public_unpaired_cases': 32, 'actual_evaluated_PNGs': len(seen), 'all_actual_evaluated_PNGs_displayed': True, 'exact256_cells': cells_total, 'per_cell_pixel_SHA256_and_coordinates': True, 'sheets': sheets, 'cases': rows_index, 'source_pins': inputs, 'output_pins': outputs, 'root_visual_review_pending': True, 'scope': 'Fixed adaptive TRAIN paired synthetic and public unpaired development only, assisted reviewed masks. No reserved FINAL, automatic proposal evaluation, local-identity recovery or native CCTV clean truth. All intermediate readouts shown without winner selection; only final32 is prospective quality endpoint.', 'model_qualified': False, 'local_execution': {'model_loads': 0, 'neural_calls': 0, 'gradients': 0, 'optimizer_updates': 0, 'VM_calls': 0}, 'seconds': time.monotonic() - start}
    write(BASE / 'visual_sheet_index.json', index)
    print(json.dumps({'complete': True, 'index': pin(BASE / 'visual_sheet_index.json'), 'sheets': len(sheets), 'cases': len(rows_index), 'outputs': len(seen), 'exact256_cells': cells_total}), flush=True)


class ConfinedStatic(HTMLParser):
    def __init__(self): super().__init__(); self.links = []; self.images = 0
    def handle_starttag(self, tag, attrs):
        need(tag not in ('script', 'iframe', 'form', 'input', 'button', 'embed', 'object'), 'Active content refused')
        attrs = dict(attrs); need(not any(key.startswith('on') for key in attrs), 'Event handlers refused')
        if tag == 'img': self.images += 1
        for key in ('href', 'src'):
            if key in attrs:
                value = attrs[key]
                need(value and not any(c in value for c in (':', '\\', '..', '?', '#')) and not value.startswith('/'), 'Confined static local link')
                path = BASE / value; need(path.is_file() and path.resolve().is_relative_to(BASE.resolve()), 'Existing link in return folder')
                self.links.append(value)


def html_review():
    admitted()
    required = ['visual_sheet_index.json', 'transport_audit.json', 'independent_saved_output_audit.json', 'root_visual_review.json']
    reports = {name: read(BASE / name) for name in required}
    need(all(report.get('complete') is True for report in reports.values()), 'Complete actual independent and root reviews required')
    index, visual, audit = reports['visual_sheet_index.json'], reports['root_visual_review.json'], reports['independent_saved_output_audit.json']
    need(visual.get('index_sha256') == sha(BASE / 'visual_sheet_index.json'), 'Root visual review bound to actual index')
    sections = []
    for prefix, title in [('original_monitor_', '32 paired original clear adaptive TRAIN cases'), ('challenge_', '16 paired synthetic adaptive TRAIN challenges'), ('public_unpaired_', '32 public unpaired development photographs')]:
        articles = []
        for sheet in index['sheets']:
            path = Path(sheet['file']['path']); checkpin(path, sheet['file'])
            if not path.stem.startswith(prefix): continue
            relative = path.relative_to(BASE).as_posix()
            articles.append('<article><h3>' + escape(path.stem) + '</h3><p>' + escape(', '.join(row['id'] for row in sheet['cases'])) + '</p><a href="' + relative + '" target="_blank"><img src="' + relative + '" loading="lazy" alt="Fixed saved completion comparisons"></a><p>Click for original256pixel detail; thumbnails are navigation only.</p></article>')
        sections.append('<section><h2>' + title + '</h2>' + ''.join(articles) + '</section>')
    nav = ''.join('<a href="' + name + '">' + escape(name) + '</a> ' for name in required)
    text = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\' file:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'"><title>Predictor-only CE V1 saved comparisons</title><style>body{background:#f3f1eb;color:#172027;font:16px/1.5 system-ui}main{max-width:1280px;margin:auto;padding:24px}article{background:white;border:1px solid #bbb;padding:16px;margin:20px 0}img{max-width:100%;height:auto;display:block}nav{display:flex;gap:15px;flex-wrap:wrap}.status{padding:18px;border:1px solid #b77;background:#fff2df}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #aaa}</style></head><body><main><h1>Predictor-only CE V1: all saved outputs</h1><div class="status"><strong>Development comparison; no model automatically adopted.</strong><p>R1 is the retained baseline. u8/u16 are safety-only image readouts, not selectable checkpoints. Finalu32 alone is the acceptance endpoint. Exact visible-pixel preservation is separate from plausible hidden anatomy and user acceptance.</p><p>Independent final numeric screen: ' + str(audit['final_prospective_numeric_screen_pass']) + '.</p><p>Root visual outcome: ' + escape(str(visual.get('visual_outcome', visual.get('decision', 'See linked root review.')))) + '</p></div><p>All80cases, allactualsavedoutputs, fixedorder and exact256pixels. No inference, sharpening, enhancement, resampling or externalrequests. Public images have no aligned hidden clean target; adaptive paired TRAIN is not independentFINAL or nativeCCTVtruth.</p><nav>' + nav + '</nav>' + ''.join(sections) + '<p>This page cannot establish exact hidden identity, allcoveringfamily qualification or automaticmask quality. CurrentR1app and RESTORATION/IMPLEMENTATION are unchanged.</p></main></body></html>\n'
    parser = ConfinedStatic(); parser.feed(text)
    need(parser.images == 20, 'All20 sheets in static report')
    path = BASE / 'review.html'; need(not path.exists(), 'Preserve prior static report')
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)
    write(BASE / 'review_html_admission.json', {'format': 'completion-codeformer-predictor-ce-v1-readonly-HTML-admission-v1', 'complete': True, 'HTML': pin(path), 'builder': pin(__file__), 'input_reports': {name: pin(BASE / name) for name in required}, 'sheets_linked': parser.images, 'confined_existing_links': len(parser.links), 'JS_forms_external_network': False, 'model_qualified': False, 'neural_calls': 0, 'VM_calls': 0, 'app_changes': False})
    print(json.dumps({'complete': True, 'HTML': pin(path), 'admission': pin(BASE / 'review_html_admission.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--render', action='store_true'); group.add_argument('--html', action='store_true')
    args = parser.parse_args()
    html_review() if args.html else render()


if __name__ == '__main__': main()
