"""Exact saved-PNG galleries and separate paired/unpaired return summaries."""
import gzip
import hashlib
import json
from pathlib import Path
import time
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
RETURNED = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1_return'
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_return_review_v1'
RATES = [1e-5, 1e-4, 1e-3]
PARTITIONS = ['deep3', 'decoder15']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rgb(path):
    with Image.open(path) as image:
        assert image.size == (256, 256)
        return image.convert('RGB').copy()


def main():
    started = time.monotonic()
    assert not OUT.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['paired_raw_and_PNG_records'] == 1300 and audit['unpaired_native_records'] == 312
    p = read(PACKET / 'protocol.json')
    assert sha(PACKET / 'protocol.json') == audit['protocol_sha256']
    assert p['visual_review_plan']['total_pages'] == 64
    OUT.mkdir()
    gallery = OUT / 'gallery'
    gallery.mkdir()
    write(OUT / 'plan.json', dict(complete=True, protocol_sha256=audit['protocol_sha256'],
          archive_sha256=audit['archive_sha256'], independent_audit_sha256=sha(audit_path),
          script_sha256=sha(Path(__file__)), paired_pages=40, native_pages=24, total_pages=64,
          paired_all5_profiles=True, both_fitting_pools_shown_for_every_paired_reference=True,
          source_pixels='exact saved delivered PNGs, unscaled 256x256 RGB; no display enhancement',
          raw_arrays_and_metrics_retained_separately=True, native_clean_reference_metrics=False,
          reserved_final_pixels=0, local_gradient_queries=0, local_optimizer_updates=0,
          manual_follow_on_only=True, model_qualification=False, goal_complete=False))
    arms = {arm['id']: arm for arm in p['arms']}
    result = read(RETURNED / 'outputs/results.json')
    summaries = []
    for arm_id, arm in arms.items():
        gate = read(RETURNED / 'outputs' / arm_id / 'quality_gate.json')
        row = dict(arm=arm_id, partition=arm['partition'], pool=arm['pool'], lr=arm['lr'],
                   sampled_capacity_pass=gate['sampled_capacity_pass'], comparisons=gate['comparisons'])
        summaries.append(row)
    write(OUT / 'quality_summary.json', dict(complete=True, protocol_sha256=audit['protocol_sha256'],
          archive_sha256=audit['archive_sha256'], arms=summaries,
          arms_passing_sampled_requirements=sum(row['sampled_capacity_pass'] for row in summaries),
          fitting_exposures=result['progress']['fit_exposures'], optimizer_updates=result['optimizer_updates'],
          completed_epochs=0, paired_role='historically exposed TRAIN; not held-out validation',
          native_role='24 unpaired native development crops, no clean target; qualitative review pending',
          model_qualification=False, visual_review_pending=True, goal_complete=False))
    try:
        regular = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 15)
        title = ImageFont.truetype('C:/Windows/Fonts/consolab.ttf', 20)
    except OSError:
        regular = title = ImageFont.load_default()
    pages = []
    cells = []
    pad = 12
    cell = 256
    top = 96
    row_step = 293

    def build(name, kind, partition, pool, rows, columns, heading, notes):
        canvas = Image.new('RGB', (pad + len(columns)*(cell+pad), top + len(rows)*row_step), '#15171b')
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 8), heading, font=title, fill='#ffffff')
        draw.text((pad, 34), notes, font=regular, fill='#dadde3')
        for i, column in enumerate(columns):
            draw.text((pad+i*(cell+pad), 61), column, font=regular, fill='#e0e3e8')
        ids = []
        for j, (case, sources) in enumerate(rows):
            ids.append(case['id'])
            draw.text((pad, top+j*row_step-15), case['id'], font=regular, fill='#e0e3e8')
            assert len(sources) == len(columns)
            for i, (stage, file) in enumerate(sources):
                image = rgb(file)
                x = pad+i*(cell+pad)
                y = top+j*row_step+4
                canvas.paste(image, (x, y))
                cells.append(dict(page=name, row=j, column=i, x=x, y=y, width=256, height=256,
                     case_id=case['id'], stage=stage, source=file.relative_to(ROOT).as_posix(),
                     source_sha256=sha(file), RGB_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        destination = gallery / name
        canvas.save(destination)
        pages.append(dict(name=name, kind=kind, partition=partition, pool=pool, case_ids=ids,
                          file=destination.relative_to(ROOT).as_posix(), sha256=sha(destination),
                          width=canvas.width, height=canvas.height, columns=columns))

    for partition in PARTITIONS:
        for ref_index, reference in enumerate(p['references']):
            cases = [case for case in p['cases'] if case['source_person_or_reference'] == reference['id']]
            assert len(cases) == 5
            rows = []
            for case in cases:
                sources = [('input', PACKET / case['input']), ('clean_TRAIN_target', PACKET / reference['target']),
                           ('current_DGP', RETURNED / 'outputs/baseline/previews' / (case['id']+'.png'))]
                for rate in RATES:
                    for pool in [0, 1]:
                        arm = f'{partition}_lr{rate:g}_pool{pool}'
                        assert arm in arms
                        sources.append((arm, RETURNED / 'outputs' / arm / 'previews' / (case['id']+'.png')))
                rows.append((case, sources))
            columns = ['Input / resize', 'HQ TRAIN target', 'Current DGP'] + [f'LR {rate:g} / pool {pool}' for rate in RATES for pool in [0, 1]]
            build(f'paired_{partition}_{ref_index:02d}.png', 'paired_TRAIN', partition, None, rows, columns,
                  f'PAIRED TRAIN {reference["id"]} | {partition} | all 5 profiles',
                  'Exact saved PNGs; both independent fitting pools. No final evaluation or native-reference claim.')
    for partition in PARTITIONS:
        for pool in [0, 1]:
            for page_index in range(6):
                cases = p['native_development'][page_index*4:page_index*4+4]
                rows = []
                for case in cases:
                    sources = [('input', PACKET / case['input']),
                               ('current_DGP', RETURNED / 'outputs/baseline/native' / (case['id']+'.png'))]
                    for rate in RATES:
                        arm = f'{partition}_lr{rate:g}_pool{pool}'
                        sources.append((arm, RETURNED / 'outputs' / arm / 'native' / (case['id']+'.png')))
                    rows.append((case, sources))
                build(f'native_{partition}_pool{pool}_{page_index:02d}.png', 'unpaired_native_DEV', partition, pool,
                      rows, ['Input / resize', 'Current DGP']+[f'LR {rate:g}' for rate in RATES],
                      f'UNPAIRED NATIVE DEVELOPMENT | {partition} | pool {pool} | page {page_index+1}/6',
                      'ChokePoint C1 public capture; no aligned clean face. Judge structure, appearance and useful clarity.')
    assert len(pages) == 64 and sum(page['kind']=='paired_TRAIN' for page in pages) == 40
    assert len(cells) == 40*5*9 + 24*4*5
    write(gallery / 'gallery_manifest.json', dict(complete=True, protocol_sha256=audit['protocol_sha256'],
          independent_audit_sha256=sha(audit_path), pages=pages, cells=cells,
          enhanced_or_rescaled_cells=0, raw_arrays_separate=True, all_planned_pages_must_be_viewed=True,
          model_qualification=False, goal_complete=False))
    archive = ROOT / 'outputs/cctv-dgp-multiscale-calibration-v1-results.tar.gz'
    total = 0
    with gzip.open(archive, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            total += len(chunk)
            assert total <= 2684354560+1024**2 and time.monotonic()-started < 600
    write(OUT / 'preparation.json', dict(complete=True, pages=64, exact_cells=len(cells),
          full_return_gzip_CRC_verified=True, uncompressed_stream_bytes=total,
          gallery_manifest_sha256=sha(gallery / 'gallery_manifest.json'), seconds=time.monotonic()-started,
          visual_review_pending=True, local_gradient_queries=0, local_optimizer_updates=0,
          model_qualification=False, goal_complete=False))
    print(dict(complete=True, pages=64, exact_cells=len(cells), visual_review_pending=True), flush=True)


if __name__ == '__main__':
    main()
