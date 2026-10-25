"""Interactive Conditioned SFT V2 browser upload, multi-model restoration, and Stage 1-4 review server."""
import argparse
import base64
import io
import json
from pathlib import Path
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
import numpy as np
from PIL import Image, ImageDraw
import uvicorn

from research import codeformer_conditioned_sft_inference_v2 as inf_v2

ROOT = Path(__file__).resolve().parent
TESTING_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
WEIGHTS_DIR = (
    ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights'
    if (ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_r2.pth').is_file()
    else TESTING_ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights'
)
V1_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm'
V2_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v2_vm'
FEAS_V2_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_feasibility'
PREP_V2_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_source_preparation'
B8_CKPT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/checkpoints/B_u0008/trainable.npz'
V2_RETURN_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1'
V2_BEST_CKPT = V2_RETURN_DIR / 'checkpoints/best_degraded_ssim/trainable_v2.npz'

PRESET_CASES = {
    'global_blur': ('Paired Global Blur (sigma=2.0)', V1_STAGE / 'data/inputs/tr_ffhq_08698__global_gaussian2p0.png'),
    'compound_cctv': ('Compound LR64 + JPEG70', V1_STAGE / 'data/inputs/tr_ffhq_08698__compound_g1p2_lr64_jpeg70.png'),
    'chokepoint_native': ('Native CCTV ChokePoint C1', V2_STAGE / 'data/native_cctv/choke_dev_01_t033.png'),
    'lowlight_noise': ('Lowlight 0.35x + Sensor Noise', V1_STAGE / 'data/inputs/tr_ffhq_08698__lowlight0p35_noise0p006.png'),
}


def _png_data_url(rgb_u8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8, mode='RGB').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _render_composite_strip(comp: dict) -> str:
    panels = [
        ('Observed Input', comp['input_rgb8'], comp['sharpness']['input']),
        ('u64 Baseline', comp['u64_rgb8'], comp['sharpness']['u64']),
    ]
    if comp['v1_b8_rgb8'] is not None:
        panels.append(('V1 Arm B (u0008)', comp['v1_b8_rgb8'], comp['sharpness']['v1_b8']))
    v2_label = 'V2 Trained (L4)' if comp['v2_checkpoint'] else 'V2 Step-0 Init'
    panels.append((v2_label, comp['v2_rgb8'], comp['sharpness']['v2']))

    pad = 12
    header_h = 44
    w = pad + len(panels) * (256 + pad)
    h = header_h + 256 + pad
    canvas = Image.new('RGB', (w, h), (16, 19, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, arr, sharp) in enumerate(panels):
        x0 = pad + idx * (256 + pad)
        draw.text((x0 + 4, 8), title, fill=(236, 239, 244))
        draw.text((x0 + 4, 24), f'Lap sharpness: {sharp:.5f}', fill=(154, 165, 180))
        canvas.paste(Image.fromarray(arr, mode='RGB'), (x0, header_h))
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _decode_to_256_rgb(raw_bytes: bytes) -> np.ndarray:
    if not raw_bytes or len(raw_bytes) > 16 * 1024 * 1024:
        raise HTTPException(status_code=400, detail='Upload a valid PNG or JPEG image up to 16 MB.')
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            rgb = im.convert('RGB')
            if rgb.size != (256, 256):
                # Center-crop square then Lanczos resize to 256x256
                w, h = rgb.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                rgb = rgb.crop((left, top, left + side, top + side)).resize((256, 256), Image.Resampling.LANCZOS)
            return np.asarray(rgb, dtype=np.uint8).copy()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not decode image: {exc}') from exc


def build_app():
    app = FastAPI(title='CCTV CodeFormer Conditioned SFT V2 Workbench')
    restorer_lock = threading.Lock()
    restorer_state = {'instance': None, 'ckpt_mtime': None}

    def get_restorer():
        with restorer_lock:
            ckpt_path = V2_BEST_CKPT if V2_BEST_CKPT.is_file() else None
            mtime = ckpt_path.stat().st_mtime_ns if ckpt_path else None
            if restorer_state['instance'] is None or restorer_state['ckpt_mtime'] != mtime:
                restorer_state['instance'] = inf_v2.load_conditioned_sft_v2_restorer(
                    WEIGHTS_DIR / 'codeformer_r2.pth',
                    WEIGHTS_DIR / 'codeformer_u96_adapter.pth',
                    V1_STAGE / 'weights/terminal_u0064_heads.npz',
                    v2_trainable_path=ckpt_path,
                    v1_b8_trainable_path=B8_CKPT if B8_CKPT.is_file() else None,
                    device='cpu',
                )
                restorer_state['ckpt_mtime'] = mtime
            return restorer_state['instance']

    @app.get('/favicon.ico')
    def favicon():
        return Response(status_code=204)

    @app.get('/api/v2-status')
    def api_v2_status():
        feas = json.loads((FEAS_V2_DIR / 'v2_feasibility_receipt_v1.json').read_text(encoding='utf-8'))
        audit_p = PREP_V2_DIR / 'independent_packet_audit_v2.json'
        pkt = json.loads(audit_p.read_text(encoding='utf-8')) if audit_p.is_file() else {}
        ret_p = V2_RETURN_DIR / 'return_audit_v2.json'
        ret = json.loads(ret_p.read_text(encoding='utf-8')) if ret_p.is_file() else {'available': False}
        return {
            'v2_trained_checkpoint_present': V2_BEST_CKPT.is_file(),
            'feasibility': feas,
            'packet_audit': pkt,
            'return_audit': ret,
        }

    @app.post('/api/restore-upload')
    async def api_restore_upload(file: UploadFile = File(...)):
        raw = await file.read()
        rgb_u8 = _decode_to_256_rgb(raw)
        restorer = get_restorer()
        with restorer_lock:
            comp = restorer.restore_compare_all(rgb_u8)
        return JSONResponse({
            'filename': file.filename or 'uploaded.png',
            'v2_trained_loaded': comp['v2_checkpoint'] is not None,
            'v2_gamma': comp['v2_gamma'],
            'conditioning': comp['conditioning'],
            'sharpness': comp['sharpness'],
            'images': {
                'input': _png_data_url(comp['input_rgb8']),
                'u64': _png_data_url(comp['u64_rgb8']),
                'v1_b8': _png_data_url(comp['v1_b8_rgb8']) if comp['v1_b8_rgb8'] is not None else None,
                'v2': _png_data_url(comp['v2_rgb8']),
                'composite_strip': _render_composite_strip(comp),
            },
        })

    @app.post('/api/restore-preset/{preset_key}')
    def api_restore_preset(preset_key: str):
        if preset_key not in PRESET_CASES:
            raise HTTPException(status_code=404, detail='Unknown preset case.')
        label, path = PRESET_CASES[preset_key]
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Preset image file missing.')
        rgb_u8 = _decode_to_256_rgb(path.read_bytes())
        restorer = get_restorer()
        with restorer_lock:
            comp = restorer.restore_compare_all(rgb_u8)
        return JSONResponse({
            'filename': label,
            'v2_trained_loaded': comp['v2_checkpoint'] is not None,
            'v2_gamma': comp['v2_gamma'],
            'conditioning': comp['conditioning'],
            'sharpness': comp['sharpness'],
            'images': {
                'input': _png_data_url(comp['input_rgb8']),
                'u64': _png_data_url(comp['u64_rgb8']),
                'v1_b8': _png_data_url(comp['v1_b8_rgb8']) if comp['v1_b8_rgb8'] is not None else None,
                'v2': _png_data_url(comp['v2_rgb8']),
                'composite_strip': _render_composite_strip(comp),
            },
        })

    @app.get('/assets/v2-paired-feasibility.png')
    def asset_paired_feas():
        return FileResponse(FEAS_V2_DIR / 'v2_paired_frequency_fusion_sheet_v1.png', media_type='image/png')

    @app.get('/assets/v2-native-feasibility.png')
    def asset_native_feas():
        return FileResponse(FEAS_V2_DIR / 'v2_native_cctv_frequency_fusion_sheet_v1.png', media_type='image/png')

    @app.get('/assets/v2-return-paired.png')
    def asset_return_paired():
        p = V2_RETURN_DIR / 'visuals/v2_returned_paired_sheet_v1.png'
        if p.is_file():
            return FileResponse(p, media_type='image/png')
        return FileResponse(FEAS_V2_DIR / 'v2_paired_frequency_fusion_sheet_v1.png', media_type='image/png')

    @app.get('/assets/v2-return-native.png')
    def asset_return_native():
        p = V2_RETURN_DIR / 'visuals/v2_returned_native_cctv_sheet_v1.png'
        if p.is_file():
            return FileResponse(p, media_type='image/png')
        return FileResponse(FEAS_V2_DIR / 'v2_native_cctv_frequency_fusion_sheet_v1.png', media_type='image/png')

    @app.get('/', response_class=HTMLResponse)
    def index():
        st = api_v2_status()
        pkt = st['packet_audit']
        ret = st['return_audit']
        feas_rows = ''
        for row in st['feasibility'].get('paired_cases', []):
            if not row.get('has_target'):
                continue
            u64_m = row['arms']['u64']
            b8_m = row['arms']['V1_B8_dark_aligned']
            em_m = row['arms']['V2_Exposure_Matched_SFT_Target']
            feas_rows += f"""
            <tr>
              <td class="mono">{row['id']}</td>
              <td class="num">{u64_m['png_mae']:.5f} / {u64_m['png_ssim7']:.4f} / {u64_m['central_laplacian_sharpness']:.5f}</td>
              <td class="num">{b8_m['png_mae']:.5f} / {b8_m['png_ssim7']:.4f} / {b8_m['central_laplacian_sharpness']:.5f}</td>
              <td class="num highlight-val">{em_m['png_mae']:.5f} / {em_m['png_ssim7']:.4f} / {em_m['central_laplacian_sharpness']:.5f}</td>
            </tr>
            """

        ckpt_status_text = (
            f"Loaded L4 Trained V2 ({ret.get('best_checkpoint_tag', 'best_degraded_ssim')})"
            if st['v2_trained_checkpoint_present']
            else 'Step-0 Exact u64 Parity Initializer (Awaiting Manual NVIDIA L4 VM Run)'
        )
        p_sha = pkt.get('protocol_sha256', 'pending')
        a_sha = pkt.get('archive_sha256', 'pending')
        a_bytes = f"{pkt.get('archive_bytes', 0):,}" if pkt.get('archive_bytes') else 'pending'

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CCTV CodeFormer Conditioned SFT V2 Workbench</title>
  <style>
    :root {{
      --canvas: #101318;
      --surface: #171B22;
      --surface-alt: #1E242E;
      --ink: #ECEFF4;
      --muted: #9AA5B4;
      --accent: #38BDF8;
      --accent-ink: #082032;
      --border: #28303D;
      --danger: #F87171;
      --success: #34D399;
      --header-height: 4rem;
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{
      scrollbar-gutter: stable;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }}
    body {{
      margin: 0;
      min-height: 100dvh;
      display: flex;
      flex-direction: column;
      background: var(--canvas);
      color: var(--ink);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
      max-width: 100%;
    }}
    .topbar {{
      height: var(--header-height);
      border-bottom: 1px solid var(--border);
      background: var(--surface);
      display: flex;
      align-items: center;
      padding: 0 16px;
    }}
    .topbar-inner {{
      max-width: 1240px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-width: 0;
    }}
    .brand-title {{
      font-size: 1rem;
      font-weight: 600;
      margin: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .model-state {{
      font-size: 0.8125rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    main {{
      flex: 1;
      max-width: 1240px;
      width: 100%;
      margin: 0 auto;
      padding: 24px 16px 40px;
      min-width: 0;
    }}
    h1 {{
      font-size: clamp(1.35rem, 2.4vw, 1.85rem);
      line-height: 1.25;
      margin: 0 0 8px;
      max-width: 65ch;
    }}
    .lead {{
      margin: 0 0 20px;
      color: var(--muted);
      max-width: 68ch;
      font-size: 0.95rem;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }}
    .tab-btn {{
      min-height: 44px;
      padding: 10px 16px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      font-size: 0.9375rem;
      font-weight: 500;
      cursor: pointer;
      transition: background 120ms ease, color 120ms ease, transform 60ms ease-out, border-color 120ms ease;
    }}
    .tab-btn:hover {{
      background: var(--surface-alt);
      border-color: var(--accent);
    }}
    .tab-btn:active {{
      transform: scale(0.97);
    }}
    .tab-btn:focus-visible, .action-btn:focus-visible, .preset-btn:focus-visible, .file-input:focus-visible, .dl-link:focus-visible {{
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }}
    .tab-btn[aria-selected="true"] {{
      background: var(--accent);
      color: var(--accent-ink);
      border-color: var(--accent);
      font-weight: 600;
    }}
    .panel {{
      display: none;
      min-width: 0;
    }}
    .panel.active {{
      display: block;
    }}
    .surface-box {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
      min-width: 0;
    }}
    .upload-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
      min-width: 0;
    }}
    .file-label {{
      font-size: 0.9375rem;
      font-weight: 500;
    }}
    .file-input {{
      font-size: 16px;
      min-height: 44px;
      padding: 8px;
      color: var(--ink);
      background: var(--canvas);
      border: 1px solid var(--border);
      border-radius: 6px;
      width: 100%;
      max-width: 340px;
      min-width: 0;
      cursor: pointer;
    }}
    .file-input[aria-invalid="true"] {{
      border-color: var(--danger);
    }}
    .action-btn, .preset-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-height: 44px;
      padding: 10px 16px;
      border-radius: 6px;
      border: 1px solid var(--border);
      font-size: 16px;
      font-weight: 500;
      cursor: pointer;
      transition: background 120ms ease, color 120ms ease, transform 60ms ease-out, border-color 120ms ease;
    }}
    .action-btn {{
      background: var(--accent);
      color: var(--accent-ink);
      border-color: var(--accent);
      font-weight: 600;
      min-width: 212px;
    }}
    .action-btn:hover:not(:disabled) {{ filter: brightness(1.08); }}
    .action-btn:active:not(:disabled), .preset-btn:active:not(:disabled), .dl-link:active {{ transform: scale(0.97); }}
    .action-btn:disabled, .preset-btn:disabled {{
      cursor: not-allowed;
      opacity: 0.6;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .spinner {{
      width: 14px;
      height: 14px;
      border: 2px solid currentColor;
      border-right-color: transparent;
      border-radius: 50%;
      animation: spin 650ms linear infinite;
      display: none;
    }}
    .is-loading .spinner {{
      display: inline-block;
    }}
    .preset-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .preset-btn {{
      background: var(--canvas);
      color: var(--ink);
      font-size: 0.875rem;
    }}
    .preset-btn:hover:not(:disabled) {{
      background: var(--surface-alt);
      border-color: var(--accent);
    }}
    .status-msg {{
      margin: 10px 0 0;
      font-size: 0.9rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .status-msg.error {{ color: var(--danger); }}
    .status-msg.success {{ color: var(--success); }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .img-frame {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      min-width: 0;
    }}
    .img-frame h3 {{
      margin: 0 0 4px;
      font-size: 0.9rem;
    }}
    .img-meta {{
      margin: 0 0 8px;
      font-size: 0.8rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .img-frame img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      height: auto;
      display: block;
      object-fit: cover;
      border-radius: 6px;
      background: #0b0d11;
    }}
    .sheet-img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 6px;
      border: 1px solid var(--border);
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.8125rem;
      word-break: break-all;
    }}
    .num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .highlight-val {{ color: var(--accent); font-weight: 600; }}
    .download-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .dl-link {{
      display: inline-flex;
      align-items: center;
      min-height: 44px;
      padding: 10px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--canvas);
      color: var(--ink);
      text-decoration: none;
      font-size: 0.875rem;
      cursor: pointer;
    }}
    .dl-link:hover {{ border-color: var(--accent); }}
    footer {{
      border-top: 1px solid var(--border);
      padding: 16px;
      color: var(--muted);
      font-size: 0.8125rem;
      text-align: center;
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <p class="brand-title" title="CCTV CodeFormer Conditioned SFT V2">CCTV CodeFormer Conditioned SFT V2</p>
      <div class="model-state" title="{ckpt_status_text}">{ckpt_status_text}</div>
    </div>
  </header>

  <main>
    <h1>Multi-Scale Dilated SFT &amp; Frequency-Selective Face Restoration</h1>
    <p class="lead">Upload any 256x256 CCTV crop or run a retained benchmark preset to compare u64, V1 Arm B, and Conditioned SFT V2 side by side.</p>

    <div class="tabs" role="tablist" aria-label="Restoration sections">
      <button type="button" class="tab-btn" role="tab" id="tab-live" aria-selected="true" aria-controls="panel-live">Live Upload &amp; Compare</button>
      <button type="button" class="tab-btn" role="tab" id="tab-feas" aria-selected="false" aria-controls="panel-feas">Stage 1 SFT Headroom Proof</button>
      <button type="button" class="tab-btn" role="tab" id="tab-packet" aria-selected="false" aria-controls="panel-packet">V2 L4 Packet &amp; Provenance</button>
    </div>

    <section id="panel-live" class="panel active" role="tabpanel" aria-labelledby="tab-live">
      <div class="surface-box">
        <form id="upload-form" class="upload-controls">
          <label for="face-file" class="file-label">256x256 Face Image (PNG/JPEG):</label>
          <input id="face-file" name="file" type="file" accept="image/png,image/jpeg" class="file-input" />
          <button id="run-upload-btn" type="submit" class="action-btn"><span class="spinner" aria-hidden="true"></span><span>Restore Uploaded Face</span></button>
        </form>
        <div>
          <span class="file-label">Quick Benchmark Presets:</span>
          <div class="preset-row">
            <button type="button" class="preset-btn" data-preset="global_blur">Global Blur (sigma 2.0)</button>
            <button type="button" class="preset-btn" data-preset="compound_cctv">Compound LR64 + JPEG70</button>
            <button type="button" class="preset-btn" data-preset="chokepoint_native">Native ChokePoint CCTV</button>
            <button type="button" class="preset-btn" data-preset="lowlight_noise">Lowlight 0.35x + Noise</button>
          </div>
        </div>
        <p id="live-status" class="status-msg" role="status">Select an image or click a benchmark preset to run CPU multi-model inference.</p>
      </div>

      <div id="live-results" hidden>
        <div class="compare-grid">
          <div class="img-frame">
            <h3>Observed Input</h3>
            <p id="meta-input" class="img-meta">Lap sharpness: 0.00000</p>
            <img id="img-input" width="256" height="256" alt="Observed 256x256 input face" />
          </div>
          <div class="img-frame">
            <h3>u64 Baseline</h3>
            <p id="meta-u64" class="img-meta">Lap sharpness: 0.00000</p>
            <img id="img-u64" width="256" height="256" alt="u64 baseline restoration" />
          </div>
          <div class="img-frame">
            <h3>V1 Arm B (Dark-Aligned)</h3>
            <p id="meta-b8" class="img-meta">Lap sharpness: 0.00000</p>
            <img id="img-b8" width="256" height="256" alt="Conditioned SFT V1 Arm B restoration" />
          </div>
          <div class="img-frame">
            <h3 id="title-v2">Conditioned SFT V2</h3>
            <p id="meta-v2" class="img-meta">Lap sharpness: 0.00000</p>
            <img id="img-v2" width="256" height="256" alt="Conditioned SFT V2 restoration" />
          </div>
        </div>
        <div class="download-row">
          <a id="dl-strip" class="dl-link" download="v2_comparison_strip.png" href="#">Download Side-by-Side PNG</a>
          <a id="dl-json" class="dl-link" download="v2_comparison_receipt.json" href="#">Download Telemetry JSON</a>
        </div>
      </div>
    </section>

    <section id="panel-feas" class="panel" role="tabpanel" aria-labelledby="tab-feas">
      <div class="surface-box">
        <h2>Exposure-Matched SFT Sharpness Ceiling (MAE / SSIM7 / Central Laplacian)</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Case ID</th>
                <th>u64 Baseline</th>
                <th>V1 Arm B (u0008)</th>
                <th>V2 Exposure-Matched SFT Target</th>
              </tr>
            </thead>
            <tbody>{feas_rows}</tbody>
          </table>
        </div>
      </div>
      <div class="surface-box">
        <h2>Paired 256x256 Multi-Role Comparison Sheet</h2>
        <img class="sheet-img" src="/assets/v2-paired-feasibility.png" width="1800" height="1600" alt="Paired V2 frequency and SFT comparison sheet" />
      </div>
      <div class="surface-box">
        <h2>Unpaired Native CCTV (ChokePoint &amp; QMUL-SurvFace) Sheet</h2>
        <img class="sheet-img" src="/assets/v2-native-feasibility.png" width="1400" height="1000" alt="Native CCTV ChokePoint and QMUL-SurvFace comparison sheet" />
      </div>
    </section>

    <section id="panel-packet" class="panel" role="tabpanel" aria-labelledby="tab-packet">
      <div class="surface-box">
        <h2>Sealed Manual NVIDIA L4 Training Packet</h2>
        <div class="table-wrap">
          <table>
            <tbody>
              <tr><th>Stage Directory</th><td class="mono">~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm</td></tr>
              <tr><th>Protocol SHA256</th><td class="mono" title="{p_sha}">{p_sha}</td></tr>
              <tr><th>Execution Archive SHA256</th><td class="mono" title="{a_sha}">{a_sha}</td></tr>
              <tr><th>Execution Archive Size (Bytes)</th><td class="num">{a_bytes}</td></tr>
              <tr><th>Dataset Split Census</th><td class="num">336 V2 Opt (56 IDs x 6 roles) / 48 V2 Mon / 40 V1 Mon / 56 Retained Quality / 4 Native CCTV</td></tr>
              <tr><th>Two-Phase Schedule</th><td class="num">Phase 1: 40 Epochs (1,680 steps, LR 5e-4) + Phase 2: 224 End-to-End Steps (LR 8e-5)</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <footer>
    Frozen 515-tensor R2 + 24 U96 + 36 u64 backbone. Local server performs inference only (0 neural training calls).
  </footer>

  <script>
    const tabs = Array.from(document.querySelectorAll('.tab-btn'));
    const panels = Array.from(document.querySelectorAll('.panel'));
    tabs.forEach(btn => {{
      btn.addEventListener('click', () => {{
        tabs.forEach(b => b.setAttribute('aria-selected', String(b === btn)));
        panels.forEach(p => p.classList.toggle('active', p.id === btn.getAttribute('aria-controls')));
      }});
    }});

    const statusEl = document.getElementById('live-status');
    const resultsEl = document.getElementById('live-results');
    const uploadBtn = document.getElementById('run-upload-btn');
    const presetBtns = Array.from(document.querySelectorAll('.preset-btn'));

    function setBusy(busy, message) {{
      uploadBtn.disabled = busy;
      uploadBtn.classList.toggle('is-loading', busy);
      presetBtns.forEach(b => {{ b.disabled = busy; }});
      statusEl.className = 'status-msg';
      statusEl.textContent = message;
    }}

    function renderComparison(data) {{
      document.getElementById('img-input').src = data.images.input;
      document.getElementById('img-u64').src = data.images.u64;
      if (data.images.v1_b8) {{
        document.getElementById('img-b8').src = data.images.v1_b8;
        document.getElementById('meta-b8').textContent = 'Lap sharpness: ' + data.sharpness.v1_b8.toFixed(5);
      }}
      document.getElementById('img-v2').src = data.images.v2;
      document.getElementById('title-v2').textContent = data.v2_trained_loaded ? 'Conditioned SFT V2 (L4 Trained)' : 'Conditioned SFT V2 (Step-0 Init)';
      document.getElementById('meta-input').textContent = 'Lap sharpness: ' + data.sharpness.input.toFixed(5);
      document.getElementById('meta-u64').textContent = 'Lap sharpness: ' + data.sharpness.u64.toFixed(5);
      document.getElementById('meta-v2').textContent = 'Lap sharpness: ' + data.sharpness.v2.toFixed(5);

      document.getElementById('dl-strip').href = data.images.composite_strip;
      const receiptBlob = new Blob([JSON.stringify({{
        filename: data.filename,
        v2_trained_loaded: data.v2_trained_loaded,
        v2_gamma: data.v2_gamma,
        conditioning: data.conditioning,
        sharpness: data.sharpness
      }}, null, 2)], {{ type: 'application/json' }});
      document.getElementById('dl-json').href = URL.createObjectURL(receiptBlob);

      resultsEl.hidden = false;
      const c = data.conditioning;
      statusEl.className = 'status-msg success';
      statusEl.textContent = `Completed ${{data.filename}} | g_blur=${{c.g_blur.toFixed(2)}} | g_dark=${{c.g_dark.toFixed(2)}} | g_noise=${{c.g_sensor_noise.toFixed(2)}}`;
    }}

    document.getElementById('upload-form').addEventListener('submit', async (ev) => {{
      ev.preventDefault();
      const fileInput = document.getElementById('face-file');
      if (!fileInput.files || !fileInput.files.length) {{
        statusEl.className = 'status-msg error';
        statusEl.textContent = 'Choose a PNG or JPEG image file first.';
        fileInput.setAttribute('aria-invalid', 'true');
        return;
      }}
      fileInput.removeAttribute('aria-invalid');
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      setBusy(true, 'Running CPU multi-model restoration (~12-18s)...');
      try {{
        const res = await fetch('/api/restore-upload', {{ method: 'POST', body: fd }});
        const payload = await res.json();
        if (!res.ok) throw new Error(payload.detail || 'Upload failed');
        renderComparison(payload);
      }} catch (err) {{
        statusEl.className = 'status-msg error';
        statusEl.textContent = String(err.message || err);
      }} finally {{
        uploadBtn.disabled = false;
        uploadBtn.classList.remove('is-loading');
        presetBtns.forEach(b => {{ b.disabled = false; }});
      }}
    }});

    presetBtns.forEach(btn => {{
      btn.addEventListener('click', async () => {{
        const key = btn.getAttribute('data-preset');
        setBusy(true, `Restoring preset ${{key}} across u64, V1-B8, and V2 (~12-18s)...`);
        try {{
          const res = await fetch(`/api/restore-preset/${{key}}`, {{ method: 'POST' }});
          const payload = await res.json();
          if (!res.ok) throw new Error(payload.detail || 'Preset restore failed');
          renderComparison(payload);
        }} catch (err) {{
          statusEl.className = 'status-msg error';
          statusEl.textContent = String(err.message || err);
        }} finally {{
          uploadBtn.disabled = false;
          uploadBtn.classList.remove('is-loading');
          presetBtns.forEach(b => {{ b.disabled = false; }});
        }}
      }});
    }});
  </script>
</body>
</html>"""

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8092)
    args = parser.parse_args()
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
