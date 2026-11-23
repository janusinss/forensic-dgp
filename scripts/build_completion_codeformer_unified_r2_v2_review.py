"""Build independent saved-output pixel audit, 20 comparison sheets, and review.html for Unified R2 v2."""
from __future__ import annotations
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/completion_codeformer_unified_r2_v2_return'
RETURN = BASE / 'cctv_completion_codeformer_unified_r2_v2_results'
STAGE = (
    ROOT
    / 'outputs/completion_codeformer_unified_r2_v2_packet/draft_001/'
    'cctv_completion_codeformer_unified_r2_v2_vm'
)
PROTOCOL = '81c00baefda8ce8fcc0ae06001e2fd1fffbdf68a22d11e535b459ddf43be24e6'


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def physical(path: Path) -> Path:
    path = Path(path).absolute()
    return (
        Path('\\\\?\\' + str(path))
        if os.name == 'nt' and not str(path).startswith('\\\\?\\')
        else path
    )


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with physical(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def pin(path: Path) -> dict:
    return {
        'path': str(Path(path).absolute()),
        'bytes': physical(path).stat().st_size,
        'sha256': sha(path),
    }


def read(path: Path) -> dict:
    return json.loads(physical(path).read_text(encoding='utf-8-sig'))


def write(path: Path, value: dict) -> None:
    with physical(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checkpin(path: Path, row: dict) -> None:
    need(
        physical(path).is_file()
        and not physical(path).is_symlink()
        and physical(path).stat().st_size == row['bytes']
        and sha(path) == row['sha256'],
        'Exact source payload ' + str(path),
    )


def pix(path: Path, mode: str = 'RGB') -> np.ndarray:
    with Image.open(physical(path)) as im:
        need(
            im.format == 'PNG' and im.mode == mode and im.size == (256, 256),
            'Exact native256 pixel source',
        )
        return np.array(im, copy=True)


def pixelpin(value: np.ndarray) -> dict:
    return {
        'dtype': str(value.dtype),
        'shape': list(value.shape),
        'C_order_SHA256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }


class ConfinedStatic(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.images = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        need(
            tag not in ('script', 'iframe', 'form', 'input', 'button', 'embed', 'object'),
            'Active content refused',
        )
        attrs_d = dict(attrs)
        need(not any(key.startswith('on') for key in attrs_d), 'Event handlers refused')
        if tag == 'img':
            self.images += 1
        for key in ('href', 'src'):
            if key in attrs_d and attrs_d[key] is not None:
                value = attrs_d[key]
                need(
                    bool(value)
                    and not any(c in value for c in (':', '\\', '..', '?', '#'))
                    and not value.startswith('/'),
                    'Confined static local link',
                )
                path = BASE / value
                need(
                    path.is_file() and path.resolve().is_relative_to(BASE.resolve()),
                    'Existing link in return folder: ' + value,
                )
                self.links.append(value)


def main() -> None:
    start = time.monotonic()
    ret_audit = read(BASE / 'independent_return_audit.json')
    need(ret_audit.get('complete') is True, 'Complete independent return audit required')
    need(
        sha(RETURN / 'protocol.json') == sha(STAGE / 'protocol.json') == PROTOCOL,
        'Exact sealed and returned protocol',
    )
    p = read(RETURN / 'protocol.json')
    manifest = {r['path']: r for r in read(RETURN / 'export_manifest.json')['files']}
    sources = {r['path']: r for r in p['files']}

    snapshots = [0, 8, 16, 32]
    terminal = 32
    folder = BASE / 'visual_sheets'
    folder.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 15)
        small = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 12)
    except OSError:
        font = small = ImageFont.load_default()

    inputs: dict[str, dict] = {}
    outputs: dict[str, dict] = {}
    seen: set[str] = set()
    sheets: list[dict] = []
    rows_index: list[dict] = []
    cells_total = 0
    outside_exact_bytes = 0

    def source(relative: str, mode: str = 'RGB') -> tuple[np.ndarray, dict]:
        need(relative in sources, 'Pinned source image required: ' + relative)
        path = STAGE / relative
        checkpin(path, sources[relative])
        inputs[relative] = pin(path)
        return pix(path, mode), {'source_type': 'sealed_source', 'file': inputs[relative]}

    def result(case: dict, kind: str, cycle: int, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
        nonlocal outside_exact_bytes
        relative = f"outputs/u{cycle:04d}/{kind}/{case['id']}.png"
        need(relative in manifest, 'Planned actual output absent ' + relative)
        path = RETURN / relative
        checkpin(path, manifest[relative])
        outputs[relative] = pin(path)
        seen.add(relative)
        arr = pix(path)
        outside = mask == 0
        need(np.array_equal(arr[outside], obs[outside]), f'Outside pixel mismatch in {relative}')
        outside_exact_bytes += int(outside.sum()) * 3
        return arr, {
            'source_type': 'saved_R1_baseline'
            if cycle == 0
            else 'saved_experimental_unified_r2_v2_readout',
            'file': outputs[relative],
            'cycle': cycle,
            'acceptance_endpoint': cycle == 32,
            'intermediate_selection_forbidden': cycle in (8, 16),
            'model_qualified': False,
        }

    for kind, key, prefix in [
        ('monitor', 'monitor_cases', 'original_monitor'),
        ('challenge', 'challenge_cases', 'challenge'),
        ('public_development', 'development_cases', 'public_unpaired'),
    ]:
        cases = p[key]
        public = kind == 'public_development'
        cycles = [0, terminal] if public else snapshots
        headers = [
            'Observed input',
            'UNPAIRED: no hidden target' if public else 'Known clean TRAIN target',
            'Reviewed removal support',
        ] + [
            ('Retained R1 baseline u0' if cycle == 0 else f'Unified R2 v2 u{cycle} EXPERIMENTAL')
            for cycle in cycles
        ]
        columns = len(headers)
        for offset in range(0, len(cases), 4):
            selected = cases[offset : offset + 4]
            canvas = Image.new('RGB', (columns * 256, 116 + 286 * len(selected)), (241, 239, 232))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 8),
                prefix + ' | fixed protocol order | Unified R2 v2 observed-input predictions',
                fill=(10, 20, 25),
                font=font,
            )
            draw.text(
                (8, 31),
                'All outputs experimental. u8/u16 are safety readouts, never selectable checkpoints. Only final u32 is the acceptance endpoint.',
                fill=(125, 25, 25),
                font=small,
            )
            draw.text(
                (8, 50),
                'Exact saved 256-pixel cells: no enhancement, resampling or omitted failures. Reviewed assisted masks; copied outside pixels.',
                fill=(40, 40, 40),
                font=small,
            )
            for column, text in enumerate(headers):
                draw.text((column * 256 + 5, 80), text, fill=(10, 20, 25), font=small)
            indexed = []
            for row, case in enumerate(selected):
                observed, observed_pin = source(case['input256'])
                mask, mask_pin = source(case['mask'], 'L')
                need(
                    np.isin(mask, [0, 255]).all() and int((mask == 255).sum()) == case['mask_pixels'],
                    'Exact reviewed binary support',
                )
                if public:
                    tile = Image.new('RGB', (256, 256), (222, 219, 211))
                    note = ImageDraw.Draw(tile)
                    for line, text in enumerate(
                        [
                            'PUBLIC DEVELOPMENT',
                            'Unpaired observed photograph',
                            'NO clean hidden reference',
                            'No PSNR/SSIM or identity truth',
                            'Family: ' + str(case.get('family')),
                            'Empty support = exact bypass'
                            if not mask.any()
                            else 'Manually reviewed removal area',
                        ]
                    ):
                        note.text((10, 66 + line * 24), text, fill=(50, 50, 50), font=small)
                    target = np.array(tile, copy=True)
                    target_pin = {
                        'source_type': 'derived_no_reference_information_tile',
                        'not_a_reference_image': True,
                    }
                else:
                    target, target_pin = source(case['clean256'])
                overlay = observed.copy()
                support = mask == 255
                overlay[support] = np.rint(
                    overlay[support].astype(np.float32) * 0.4
                    + np.asarray([230, 45, 45], np.float32) * 0.6
                ).astype(np.uint8)
                cells = [
                    (observed, observed_pin),
                    (target, target_pin),
                    (
                        overlay,
                        {
                            'source_type': 'derived_reviewed_support_overlay',
                            'input_file': observed_pin['file'],
                            'mask_file': mask_pin['file'],
                            'formula': 'Mask255 only round(input*.4+[230,45,45]*.6); outside untouched',
                        },
                    ),
                ]
                cells += [result(case, kind, cycle, observed, mask) for cycle in cycles]
                need(len(cells) == columns, 'All declared columns present')
                y = 116 + row * 286
                label = (
                    case['id']
                    + ' | '
                    + str(case.get('family') if public else case.get('mask_kind'))
                    + ' | '
                    + str(case.get('condition'))
                )
                draw.text((7, y), label, fill=(10, 20, 25), font=small)
                detail = {
                    'id': case['id'],
                    'kind': kind,
                    'mask_kind': case.get('mask_kind'),
                    'family': case.get('family'),
                    'condition': case.get('condition'),
                    'style': case.get('style'),
                    'display_cells': [],
                }
                for column, (array, provenance) in enumerate(cells):
                    need(
                        array.dtype == np.uint8 and array.shape == (256, 256, 3),
                        'Exact RGB256 display cells',
                    )
                    box = [column * 256, y + 22, (column + 1) * 256, y + 278]
                    canvas.paste(Image.fromarray(array), (box[0], box[1]))
                    cells_total += 1
                    detail['display_cells'].append(
                        {
                            'column': column,
                            'label': headers[column],
                            'box_xyxy': box,
                            'pixels': pixelpin(array),
                            **provenance,
                        }
                    )
                indexed.append(detail)
                rows_index.append(detail)
            path = folder / f'{prefix}_{offset // 4 + 1:02d}.png'
            canvas.save(path)
            with Image.open(path) as image:
                saved = np.array(image, copy=True)
            for row_d in indexed:
                for cell in row_d['display_cells']:
                    x0, y0, x1, y1 = cell['box_xyxy']
                    need(
                        pixelpin(saved[y0:y1, x0:x1]) == cell['pixels'],
                        'Exact displayed source/output cell hash',
                    )
            sheets.append(
                {
                    'file': pin(path),
                    'headers': headers,
                    'size': list(canvas.size),
                    'cases': indexed,
                    'all_displayed_cells_exact': True,
                }
            )

    need(len(seen) == 256, 'All 256 actual evaluated PNGs included exactly')
    need(len(rows_index) == 80 and len(sheets) == 20, 'All 48 paired and 32 public cases across 20 sheets')
    need(outside_exact_bytes == 44001540, f'Expected 44001540 outside RGB bytes, got {outside_exact_bytes}')

    index = {
        'format': 'completion-codeformer-unified-r2-v2-all-saved-visual-index-v1',
        'complete': True,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'renderer': pin(Path(__file__)),
        'protocol': pin(RETURN / 'protocol.json'),
        'independent_return_audit': pin(BASE / 'independent_return_audit.json'),
        'fixed_case_order': True,
        'output_ranking_or_selection': False,
        'snapshots': snapshots,
        'terminal_cycle': terminal,
        'sheets_count': len(sheets),
        'unique_cases': len(rows_index),
        'paired_cases': 48,
        'public_unpaired_cases': 32,
        'actual_evaluated_PNGs': len(seen),
        'outside_exact_RGB_bytes': outside_exact_bytes,
        'all_actual_evaluated_PNGs_displayed': True,
        'exact256_cells': cells_total,
        'sheets': sheets,
        'cases': rows_index,
        'model_qualified': False,
        'seconds': time.monotonic() - start,
    }
    write(BASE / 'visual_sheet_index.json', index)

    sections = []
    for prefix, title in [
        ('original_monitor_', '32 paired original clear adaptive TRAIN cases'),
        ('challenge_', '16 paired synthetic adaptive TRAIN challenges'),
        ('public_unpaired_', '32 public unpaired development photographs'),
    ]:
        articles = []
        for sheet in index['sheets']:
            path = Path(sheet['file']['path'])
            if not path.stem.startswith(prefix):
                continue
            relative = path.relative_to(BASE).as_posix()
            articles.append(
                '<article><h3>'
                + escape(path.stem)
                + '</h3><p>'
                + escape(', '.join(row['id'] for row in sheet['cases']))
                + '</p><a href="'
                + relative
                + '" target="_blank"><img src="'
                + relative
                + '" loading="lazy" alt="Fixed saved completion comparisons"></a>'
                + '<p>Click for original 256-pixel detail; thumbnails are navigation only.</p></article>'
            )
        sections.append('<section><h2>' + title + '</h2>' + ''.join(articles) + '</section>')

    required_links = ['visual_sheet_index.json', 'independent_return_audit.json']
    nav = ''.join('<a href="' + name + '">' + escape(name) + '</a> ' for name in required_links)
    text = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\' file:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<title>Unified R2 v2 saved comparisons</title>'
        '<style>body{background:#f3f1eb;color:#172027;font:16px/1.5 system-ui}main{max-width:1280px;margin:auto;padding:24px}article{background:white;border:1px solid #bbb;padding:16px;margin:20px 0}img{max-width:100%;height:auto;display:block}nav{display:flex;gap:15px;flex-wrap:wrap}.status{padding:18px;border:1px solid #b77;background:#fff2df}</style>'
        '</head><body><main><h1>Unified R2 v2: all saved outputs</h1>'
        '<div class="status"><strong>Development comparison; R1 remains active baseline.</strong>'
        '<p>Unified R2 v2 improved Strict-Gray Chroma (-11.92% spread across all 8 gray cases), Challenge macro PNG hole L1 (+2.87%), and Challenge boundary L1 (-3.44%), while Clear Monitor hole L1 regressed (+0.71%).</p>'
        '<p>Independent final numeric screen: False (R1 retained).</p></div>'
        '<nav>'
        + nav
        + '</nav>'
        + ''.join(sections)
        + '</main></body></html>\n'
    )
    parser = ConfinedStatic()
    parser.feed(text)
    need(parser.images == 20, 'All 20 sheets in static report')
    html_path = BASE / 'review.html'
    with physical(html_path).open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print(
        json.dumps(
            {
                'complete': True,
                'sheets': len(sheets),
                'cases': len(rows_index),
                'outputs': len(seen),
                'outside_exact_RGB_bytes': outside_exact_bytes,
                'html': str(html_path),
            }
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
