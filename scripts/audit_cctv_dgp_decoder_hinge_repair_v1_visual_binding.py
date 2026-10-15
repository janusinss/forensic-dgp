"""Bind the completed direct visual review to imported and audited PNG cells."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    root = Path(__file__).resolve().parents[1]
    returned = root/'outputs/cctv_dgp_decoder_hinge_repair_v1_return'
    analysis = root/'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis'
    receipt = analysis/'visual_binding_audit.json'
    assert not receipt.exists()
    numerical = read(analysis/'independent_audit_r1.json')
    observations = read(analysis/'visual_observations.json')
    manifest = read(returned/'result_manifest.json')['files']
    protocol = read(returned/'protocol.json')
    blocks = {block['reference']: block['case_ids'] for block in protocol['reference_blocks']}
    notes = {note['reference']: note['observation'] for note in observations['notes']}
    assert numerical['complete'] and observations['complete']
    assert len(numerical['sheets']) == len(notes) == len(blocks) == 20
    assert observations['cells'] == len(numerical['cell_bindings']) == 800
    expected = {(b['sheet'], b['case'], b['column']): b['pixels_sha256']
                for b in numerical['cell_bindings']}
    assert len(expected) == 800
    records = []
    for sheet in numerical['sheets']:
        reference = sheet['reference']
        relative = f'outputs/decoder_hinge_repair_v1/visual_sheets/{reference}.png'
        vm_file = returned/relative
        local_file = Path(sheet['path'])
        binding = manifest[relative]
        assert vm_file.stat().st_size == binding['bytes']
        assert digest(vm_file) == binding['sha256']
        assert digest(local_file) == sheet['sha256']
        with Image.open(vm_file) as image:
            vm_pixels = np.array(image.convert('RGB'))
        with Image.open(local_file) as image:
            local_pixels = np.array(image.convert('RGB'))
        assert vm_pixels.shape == local_pixels.shape == (1320, 2048, 3)
        cells = []
        for row, case in enumerate(blocks[reference]):
            for column in range(8):
                box = np.s_[40+256*row:40+256*(row+1), 256*column:256*(column+1), :]
                pixels = vm_pixels[box]
                assert np.array_equal(pixels, local_pixels[box])
                pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
                assert pixel_sha == expected[(reference+'.png', case, column)]
                cells.append(dict(case=case, column=column, pixels_sha256=pixel_sha))
        assert len(cells) == sheet['cells'] == 40 and notes[reference].strip()
        records.append(dict(reference=reference, imported_sheet_sha256=digest(vm_file),
            independently_generated_sheet_sha256=digest(local_file), cells=cells,
            direct_review_observation=notes[reference]))
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        sheets=20, cells=800, all_imported_sheet_hashes_match_manifest=True,
        all_imported_cells_equal_independently_generated_cells=True,
        header_font_bytes_not_used_as_pixel_evidence=True,
        observations_sha256=digest(analysis/'visual_observations.json'),
        independent_audit_r1_sha256=digest(analysis/'independent_audit_r1.json'),
        result_manifest_sha256=digest(returned/'result_manifest.json'),
        records=records, independent_final_human_review=False,
        model_qualified=False, neural_calls=0, gradient_queries=0, optimizer_updates=0)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(complete=True, sheets=20, cells=800, model_qualified=False)))


if __name__ == '__main__':
    main()
