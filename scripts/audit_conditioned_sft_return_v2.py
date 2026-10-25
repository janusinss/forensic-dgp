"""Stage 4 Return Auditor & Visual Sheet Renderer for Conditioned SFT V2 L4 Results.

Run immediately after downloading cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz:
    python scripts/audit_conditioned_sft_return_v2.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TESTING_OUT = ROOT.parent / 'Testing' / 'outputs'
LOCAL_OUT = ROOT / 'outputs'
V2_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v2_vm'
V1_AUDIT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1'
AUDIT_OUT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def find_return_archive() -> tuple[Path, Path, Path]:
    name = 'cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz'
    for base in (TESTING_OUT, LOCAL_OUT):
        arc = base / name
        sha = base / (name + '.sha256')
        exp = base / 'cctv-codeformer-conditioned-sft-learning-v2-export.json'
        if arc.is_file() and sha.is_file():
            return arc, sha, exp
    raise FileNotFoundError(f'Return archive {name} not yet present in {TESTING_OUT} or {LOCAL_OUT}.')


def render_grid(rows_spec, headers, out_path: Path):
    pad = 10
    label_w = 240
    header_h = 38
    cell = 256
    ncols = len(headers)
    w = label_w + ncols * (cell + pad) + pad
    h = header_h + len(rows_spec) * (cell + pad) + pad
    canvas = Image.new('RGB', (w, h), (16, 19, 24))
    draw = ImageDraw.Draw(canvas)
    for col_i, title in enumerate(headers):
        x0 = label_w + col_i * (cell + pad)
        draw.text((x0 + 4, 12), title, fill=(236, 239, 244))
    for row_i, (lines, imgs) in enumerate(rows_spec):
        y0 = header_h + row_i * (cell + pad)
        for li, text in enumerate(lines):
            draw.text((10, y0 + 10 + li * 18), text, fill=(154, 165, 180) if li else (236, 239, 244))
        for col_i, im in enumerate(imgs):
            x0 = label_w + col_i * (cell + pad)
            canvas.paste(im, (x0, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format='PNG')


def main():
    arc_path, sha_path, exp_path = find_return_archive()
    actual_sha = sha256_file(arc_path)
    expected_sha = sha_path.read_text(encoding='ascii').strip().split()[0]
    assert actual_sha == expected_sha, f'SHA256 mismatch: {actual_sha} != {expected_sha}'

    if AUDIT_OUT.exists():
        shutil.rmtree(AUDIT_OUT)
    extract_dir = AUDIT_OUT / 'extracted'
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(arc_path, mode='r:gz') as tar:
        members = tar.getmembers()
        for m in members:
            assert not m.name.startswith('/') and '..' not in m.name.split('/'), f'Unsafe member: {m.name}'
        tar.extractall(extract_dir, filter='data')

    stage_ret = extract_dir / 'cctv_codeformer_conditioned_sft_learning_v2_vm'
    out_ret = stage_ret / 'outputs/conditioned_sft_learning_v2'
    result = json.loads((out_ret / 'result.json').read_text(encoding='utf-8'))

    # Copy checkpoints to AUDIT_OUT/checkpoints so serve_conditioned_sft_review_v2.py auto-loads best_degraded_ssim!
    shutil.copytree(out_ret / 'checkpoints', AUDIT_OUT / 'checkpoints')
    if (out_ret / 'restored_best').is_dir():
        shutil.copytree(out_ret / 'restored_best', AUDIT_OUT / 'restored_best')

    # Render paired V2 monitor comparison sheet & native CCTV sheet
    from research import codeformer_conditioned_sft_inference_v2 as inf_v2
    weights_dir = TESTING_OUT / 'cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights'
    restorer = inf_v2.load_conditioned_sft_v2_restorer(
        weights_dir / 'codeformer_r2.pth',
        weights_dir / 'codeformer_u96_adapter.pth',
        V2_STAGE / 'weights/terminal_u0064_heads.npz',
        v2_trainable_path=AUDIT_OUT / 'checkpoints/best_degraded_ssim/trainable_v2.npz',
        v1_b8_trainable_path=V1_AUDIT / 'checkpoints/B_u0008/trainable.npz',
        device='cpu',
    )

    protocol = json.loads((V2_STAGE / 'protocol.json').read_text(encoding='utf-8'))
    mon_rows = [r for r in protocol['selection']['v2_monitors'] if r['reference_id'] == 'tr_ffhq_08698']
    paired_visuals = []
    for r in mon_rows:
        inp_im = Image.open(V2_STAGE / r['input']['path']).convert('RGB')
        targ_im = Image.open(V2_STAGE / r['target']['path']).convert('RGB')
        comp = restorer.restore_compare_all(np.asarray(inp_im, dtype=np.uint8))
        paired_visuals.append((
            [r['id'], f"role: {r['role']}", f"Lap u64={comp['sharpness']['u64']:.4f} -> V2={comp['sharpness']['v2']:.4f}"],
            [
                inp_im,
                Image.fromarray(comp['u64_rgb8']),
                Image.fromarray(comp['v1_b8_rgb8']),
                Image.fromarray(comp['v2_rgb8']),
                targ_im,
            ],
        ))
    render_grid(
        paired_visuals,
        ['Observed Input', 'u64 Baseline', 'V1 Arm B (u0008)', f"V2 Best ({result['best_checkpoint_tag']})", 'Clean Target'],
        AUDIT_OUT / 'visuals/v2_returned_paired_sheet_v1.png',
    )

    native_visuals = []
    for r in protocol['selection']['native_cctv_dev']:
        inp_im = Image.open(V2_STAGE / r['input']['path']).convert('RGB')
        comp = restorer.restore_compare_all(np.asarray(inp_im, dtype=np.uint8))
        native_visuals.append((
            [r['id'], f"dataset: {r['dataset']}", f"Lap u64={comp['sharpness']['u64']:.4f} -> V2={comp['sharpness']['v2']:.4f}"],
            [
                inp_im,
                Image.fromarray(comp['u64_rgb8']),
                Image.fromarray(comp['v1_b8_rgb8']),
                Image.fromarray(comp['v2_rgb8']),
            ],
        ))
    render_grid(
        native_visuals,
        ['Native CCTV Input', 'u64 Baseline', 'V1 Arm B (u0008)', f"V2 Best ({result['best_checkpoint_tag']})"],
        AUDIT_OUT / 'visuals/v2_returned_native_cctv_sheet_v1.png',
    )

    summary_out = {
        'available': True,
        'audited_UTC': datetime.now(timezone.utc).isoformat(),
        'archive_bytes': arc_path.stat().st_size,
        'archive_sha256': actual_sha,
        'member_count': len(members),
        'best_checkpoint_tag': result['best_checkpoint_tag'],
        'step0_exact_u64_max_abs_diff': result['step0_exact_u64_max_abs_diff'],
        'frozen_backbone_digests': result['frozen_backbone_digests'],
        'init_summary': result['init_summary']['v2_monitor_role_means'],
        'best_summary': result['best_summary']['v2_monitor_role_means'],
        'all_checkpoint_summaries': result['all_checkpoint_summaries'],
    }
    (AUDIT_OUT / 'return_audit_v2.json').write_bytes(
        (json.dumps(summary_out, indent=2, sort_keys=True) + '\n').encode('utf-8')
    )
    shutil.rmtree(extract_dir)
    print(json.dumps(summary_out, indent=2))


if __name__ == '__main__':
    main()
