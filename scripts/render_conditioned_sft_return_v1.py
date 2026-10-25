"""Render visual comparison sheets from the actual returned L4 GPU Conditioned SFT V1 run."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.codeformer_conditioned_sft_learning_v1 import conditioning, feasibility

SEALED_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm'
AUDIT_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1'
RETAINED_STAGE = AUDIT_DIR / 'retained/cctv_codeformer_conditioned_sft_learning_v1_vm'
VM_OUT = RETAINED_STAGE / 'outputs/conditioned_sft_learning_v1'
VISUALS_DIR = AUDIT_DIR / 'visuals'


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8).copy()


def render_returned_sheets():
    helper = feasibility._load_pixel_helper()
    protocol = json.loads((SEALED_STAGE / 'protocol.json').read_text(encoding='utf-8'))
    selection = protocol['selection']
    controls = {r['id']: r for r in protocol['controls']}

    call_index = {}
    for rec_file in sorted((VM_OUT / 'calls').glob('*/record.json')):
        rec = json.loads(rec_file.read_text(encoding='utf-8'))
        c = rec['call']
        call_index[(c['arm'], c['bank'], c['id'], c['phase'], c['new_update'])] = rec

    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Returned L4 GPU Multi-Role + Native CCTV Sheet
    ref_id = selection['new_monitors'][0]['reference_id']
    role_rows = [r for r in selection['new_monitors'] if r['reference_id'] == ref_id]
    dark_guards = [r for r in selection['old_quality'] if r['role'] == 'dark']
    filter_guards = [r for r in selection['old_quality'] if r['role'] == 'filter']
    selected_rows = [('new_monitors', r) for r in role_rows]
    if dark_guards:
        selected_rows.append(('old_quality', dark_guards[0]))
    if filter_guards:
        selected_rows.append(('old_quality', filter_guards[0]))

    cell_w, cell_h = 256, 256
    header_h = 44
    row_label_w = 230
    cols = [
        'Observed Input',
        'Clean Target / Ref',
        'Initial u64 (u0000)',
        'Returned Arm A (u0008)',
        'Returned Arm B (u0008)',
        'Exposure-Decoupled B8',
    ]
    canvas_w = row_label_w + len(cols) * cell_w
    canvas_h = header_h + len(selected_rows) * cell_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (16, 19, 25))
    draw = ImageDraw.Draw(canvas)

    for c_idx, col_title in enumerate(cols):
        x0 = row_label_w + c_idx * cell_w
        draw.text((x0 + 10, 14), col_title, fill=(230, 236, 245))

    for r_idx, (bank_name, row) in enumerate(selected_rows):
        y0 = header_h + r_idx * cell_h
        cid = row['id']
        role = row['role']
        inp_u8 = load_png(SEALED_STAGE / row['input']['path'])
        inp_f32 = inp_u8.astype(np.float32) / np.float32(255.0)
        if row.get('target') is not None:
            targ_u8 = load_png(SEALED_STAGE / row['target']['path'])
        else:
            targ_u8 = inp_u8.copy()

        rec_b0 = call_index[('B', bank_name, cid, 'initial_quality', 0)]
        rec_a8 = call_index[('A', bank_name, cid, 'quality_snapshot', 8)]
        rec_b8 = call_index[('B', bank_name, cid, 'quality_snapshot', 8)]

        raw0 = helper.decode_array_bytes((RETAINED_STAGE / rec_b0['outputs']['raw']['path']).read_bytes(), np=np)
        raw_a8 = helper.decode_array_bytes((RETAINED_STAGE / rec_a8['outputs']['raw']['path']).read_bytes(), np=np)
        raw_b8 = helper.decode_array_bytes((RETAINED_STAGE / rec_b8['outputs']['raw']['path']).read_bytes(), np=np)

        gates = conditioning.compute_input_conditioning_numpy(inp_f32, np=np)
        if gates['g_dark'] > 0.0:
            coarse_drift = conditioning.block16_mean_numpy(raw_b8 - raw0, np=np)
            dec_b8 = np.clip(raw_b8 - 2.0 * gates['g_dark'] * coarse_drift, 0.0, 1.0).astype(np.float32)
        else:
            dec_b8 = raw_b8

        u0_u8 = (np.clip(raw0, 0.0, 1.0) * 255.0).astype(np.uint8)
        a8_u8 = (np.clip(raw_a8, 0.0, 1.0) * 255.0).astype(np.uint8)
        b8_u8 = (np.clip(raw_b8, 0.0, 1.0) * 255.0).astype(np.uint8)
        dec_u8 = (np.clip(dec_b8, 0.0, 1.0) * 255.0).astype(np.uint8)

        draw.text((10, y0 + 18), f"{cid}", fill=(240, 244, 250))
        draw.text((10, y0 + 42), f"role: {role}", fill=(165, 180, 200))
        draw.text((10, y0 + 66), f"g_dark={gates['g_dark']:.2f} g_deg={gates['g_deg']:.2f}", fill=(130, 210, 170))

        for c_idx, arr in enumerate([inp_u8, targ_u8, u0_u8, a8_u8, b8_u8, dec_u8]):
            x0 = row_label_w + c_idx * cell_w
            canvas.paste(Image.fromarray(arr), (x0, y0))

    sheet_path = VISUALS_DIR / 'returned_l4_multi_role_sheet_v1.png'
    canvas.save(sheet_path, format='PNG')
    return sheet_path


if __name__ == '__main__':
    p = render_returned_sheets()
    print(str(p))
