"""Generate broader multi-role + native CCTV visual comparison sheet for Conditioned SFT V1."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.codeformer_conditioned_sft_learning_v1 import conditioning, feasibility

OUT_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_feasibility'


def load_png(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8).copy()


def load_npy(path):
    return np.load(path, allow_pickle=False).astype(np.float32)


def render_multi_role_sheet():
    stage_root = feasibility.DEFAULT_STAGE_ROOT
    retained_root = feasibility.DEFAULT_RETURN_AUDIT_ROOT / 'retained/cctv_codeformer_matched_feature_learning_v1_vm'
    vm_out = retained_root / 'outputs/matched_feature_learning_v1'
    helper = feasibility._load_pixel_helper()
    protocol = json.loads((stage_root / 'protocol.json').read_text(encoding='utf-8'))
    selection = protocol['selection']

    # Index call records by (arm, bank, case_id, phase, new_update)
    call_index = {}
    for rec_file in sorted((vm_out / 'calls').glob('*/record.json')):
        rec = json.loads(rec_file.read_text(encoding='utf-8'))
        call = rec['call']
        call_index[(call['arm'], call['bank'], call['id'], call['phase'], call['new_update'])] = rec

    controls = {r['id']: r for r in protocol['controls']}

    # Pick representative cases across all 5 roles + native CCTV guard
    # From new_monitors for the 5 paired roles, plus 1 native CCTV guard from old_quality
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
    row_label_w = 220
    cols = ['Observed Input', 'Clean Target / Ref', 'Incumbent U96 / u64', 'Unmodified B8', 'Conditioned B8 (Ours)', '|Diff Conditioned - Target| x4']
    canvas_w = row_label_w + len(cols) * cell_w
    canvas_h = header_h + len(selected_rows) * cell_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)

    for c_idx, col_title in enumerate(cols):
        x0 = row_label_w + c_idx * cell_w
        draw.text((x0 + 10, 14), col_title, fill=(230, 236, 245))

    for r_idx, (bank_name, row) in enumerate(selected_rows):
        y0 = header_h + r_idx * cell_h
        cid = row['id']
        role = row['role']
        inp_u8 = load_png(stage_root / row['input']['path'])
        inp_f32 = inp_u8.astype(np.float32) / np.float32(255.0)
        if row.get('target') is not None:
            targ_u8 = load_png(stage_root / row['target']['path'])
            targ_f32 = targ_u8.astype(np.float32) / np.float32(255.0)
        else:
            targ_u8 = inp_u8.copy()
            targ_f32 = inp_f32.copy()

        rec_0 = call_index[('B', bank_name, cid, 'initial_quality', 0)]
        rec_b8 = call_index[('B', bank_name, cid, 'quality_snapshot', 8)]
        raw0 = helper.decode_array_bytes(
            (retained_root / rec_0['outputs']['raw']['path']).read_bytes(), np=np
        )
        raw_b8 = helper.decode_array_bytes(
            (retained_root / rec_b8['outputs']['raw']['path']).read_bytes(), np=np
        )

        if cid in controls:
            inc_u8 = load_png(stage_root / controls[cid]['png']['path'])
        else:
            inc_u8 = (np.clip(raw0, 0.0, 1.0) * 255.0).astype(np.uint8)

        unmod_u8 = (np.clip(raw_b8, 0.0, 1.0) * 255.0).astype(np.uint8)
        cond_raw, gates = conditioning.condition_output_numpy(
            raw_b8, inp_f32, np=np, baseline_rgb=raw0, beta=0.005, gamma=1.0
        )
        cond_u8 = (np.clip(cond_raw, 0.0, 1.0) * 255.0).astype(np.uint8)
        diff_u8 = np.clip(np.abs(cond_raw - targ_f32) * 4.0 * 255.0, 0.0, 255.0).astype(np.uint8)

        draw.text((10, y0 + 18), f"{cid}", fill=(240, 244, 250))
        draw.text((10, y0 + 42), f"role: {role}", fill=(165, 180, 200))
        draw.text((10, y0 + 66), f"g_dark={gates['g_dark']:.2f} g_deg={gates['g_deg']:.2f}", fill=(130, 210, 170))

        cells = [inp_u8, targ_u8, inc_u8, unmod_u8, cond_u8, diff_u8]
        for c_idx, arr in enumerate(cells):
            x0 = row_label_w + c_idx * cell_w
            canvas.paste(Image.fromarray(arr), (x0, y0))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'conditioned_multi_role_comparison_v1.png'
    canvas.save(out_path, format='PNG')
    return out_path


if __name__ == '__main__':
    path = render_multi_role_sheet()
    print(str(path))
