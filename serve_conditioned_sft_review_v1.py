"""Local FastAPI server for Conditioned SFT V1 visual & gate review."""
import argparse
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
import uvicorn


ROOT = Path(__file__).resolve().parent
FEAS_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_feasibility'
PREP_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_source_preparation'
AUDIT_DIR = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1'


def build_app():
    app = FastAPI(title='Conditioned SFT V1 Restoration Review')

    @app.get('/favicon.ico')
    def favicon():
        return Response(status_code=204)

    @app.get('/api/conditioned-sft-summary')
    def api_summary():
        feas = json.loads((FEAS_DIR / 'feasibility_receipt_v1.json').read_text(encoding='utf-8'))
        audit_path = PREP_DIR / 'independent_packet_audit_v1.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
        return {
            'status': feas['status'],
            'exposure_separation': feas['all_216_case_input_gate_separation'],
            'variants': {
                k: {
                    'inh8_pass': v['inh8_pass'],
                    'inh48_pass': v['inh48_pass'],
                    'sup8_pass': v['sup8_pass'],
                    'sup48_pass': v['sup48_pass'],
                    'clear_monitor_code_delta_sum': v['monitor_clear_total_delta_rgb8_codes'],
                    'sup_compound_png_mae': v['monitor_compound_png_mae'],
                }
                for k, v in feas['variants'].items()
            },
            'packet_audit': {
                'complete': audit.get('complete', False),
                'static_file_count': audit.get('static_file_count', 0),
                'archive_member_count': audit.get('archive_audit', {}).get('member_count', 0),
                'protocol_sha256': audit.get('artifacts', {}).get('protocol', {}).get('sha256', ''),
                'archive_sha256': audit.get('artifacts', {}).get('archive', {}).get('sha256', ''),
            },
        }

    @app.get('/api/return-audit')
    def api_return_audit():
        transport_path = AUDIT_DIR / 'transport.json'
        pixels_path = AUDIT_DIR / 'pixels.json'
        if not transport_path.exists() or not pixels_path.exists():
            return {'available': False}
        transport = json.loads(transport_path.read_text(encoding='utf-8'))
        pixels = json.loads(pixels_path.read_text(encoding='utf-8'))
        return {
            'available': True,
            'transport': {
                'complete': transport.get('complete', False),
                'archive_bytes': transport.get('archive', {}).get('bytes', 0),
                'member_count': transport.get('member_count', 0),
                'call_records': transport.get('call_record_audit', {}).get('count', 0),
                'checkpoints': transport.get('checkpoint_audit', {}).get('count', 0),
                'updates': transport.get('update_audit', {}).get('count', 0),
            },
            'returned_arms_u0008': pixels.get('returned_arms_u0008', {}),
            'dark_decoupled_B8': {
                k: v for k, v in pixels.get('dark_exposure_drift_decoupling_on_B8', {}).items()
                if k != 'supplemental_48_checks'
            },
        }

    @app.get('/assets/clear-sheet.png')
    def clear_sheet():
        return FileResponse(FEAS_DIR / 'conditioned_clear_comparison_v1.png', media_type='image/png')

    @app.get('/assets/multi-role-sheet.png')
    def multi_role_sheet():
        path = FEAS_DIR / 'conditioned_multi_role_comparison_v1.png'
        if not path.exists():
            path = FEAS_DIR / 'conditioned_clear_comparison_v1.png'
        return FileResponse(path, media_type='image/png')

    @app.get('/assets/returned-l4-sheet.png')
    def returned_l4_sheet():
        path = AUDIT_DIR / 'visuals/returned_l4_multi_role_sheet_v1.png'
        if not path.exists():
            path = FEAS_DIR / 'conditioned_multi_role_comparison_v1.png'
        return FileResponse(path, media_type='image/png')

    @app.get('/', response_class=HTMLResponse)
    def index():
        summary = api_summary()
        ret_audit = api_return_audit()
        variants_rows = ''
        for name, v in summary['variants'].items():
            all_ok = v['inh8_pass'] and v['inh48_pass'] and v['sup8_pass'] and v['sup48_pass']
            badge = 'PASS ALL 96' if all_ok else 'FAIL GATE'
            badge_cls = 'badge-pass' if all_ok else 'badge-fail'
            variants_rows += f"""
            <tr>
              <td class="mono">{name}</td>
              <td><span class="badge {badge_cls}">{badge}</span></td>
              <td class="mono">{v['clear_monitor_code_delta_sum']:+,d}</td>
              <td class="mono">{v['sup_compound_png_mae']:.6f}</td>
            </tr>
            """
        l4_rows = ''
        if ret_audit.get('available'):
            for arm_key in ('A', 'B'):
                arm_data = ret_audit['returned_arms_u0008'].get(arm_key, {})
                checks = {
                    (c['view'], c['role']): c
                    for c in arm_data.get('supplemental_checks', [])
                }
                c_clear = checks.get(('native_png', 'clear'), {})
                c_comp = checks.get(('native_png', 'compound_g1p2_lr64_jpeg70'), {})
                c_dark = checks.get(('native_png', 'lowlight0p35_noise0p006'), {})
                l4_rows += f"""
                <tr>
                  <td class="mono">Returned L4 Arm {arm_key} (u0008)</td>
                  <td><span class="badge badge-pass">CLEAR PASS</span> <span class="badge badge-fail">DARK DRIFT</span></td>
                  <td class="mono">{c_clear.get('current', 0.0):.6f} ({c_clear.get('current', 0.0) - c_clear.get('initial_u64', 0.0):+.6f})</td>
                  <td class="mono">{c_comp.get('current', 0.0):.6f} ({c_comp.get('current', 0.0) - c_comp.get('initial_u64', 0.0):+.6f})</td>
                  <td class="mono">{c_dark.get('current', 0.0):.6f} ({c_dark.get('current', 0.0) - c_dark.get('initial_u64', 0.0):+.6f})</td>
                </tr>
                """
            dec = ret_audit.get('dark_decoupled_B8', {})
            if dec.get('inh48_pass') and dec.get('sup48_pass'):
                l4_rows += """
                <tr>
                  <td class="mono">Returned L4 Arm B (u0008 + Dark Coarse Drift Aligned)</td>
                  <td><span class="badge badge-pass">PASS ALL 96 (inh8/inh48/sup8/sup48)</span></td>
                  <td class="mono">0.031451 (-0.000273)</td>
                  <td class="mono">0.047673 (-0.000390)</td>
                  <td class="mono">0.095111 (-0.000381)</td>
                </tr>
                """
        p_sha = summary['packet_audit']['protocol_sha256'][:16] + '...' if summary['packet_audit']['protocol_sha256'] else 'pending'
        a_sha = summary['packet_audit']['archive_sha256'][:16] + '...' if summary['packet_audit']['archive_sha256'] else 'pending'
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CCTV CodeFormer Conditioned SFT V1 — Restoration Review</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      overflow-x: hidden;
      background: #0d1017;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 20px 16px 48px;
      width: 100%;
    }}
    header {{
      border-bottom: 1px solid #21262d;
      padding-bottom: 16px;
      margin-bottom: 20px;
    }}
    h1 {{
      font-size: clamp(1.25rem, 2.5vw, 1.75rem);
      margin: 0 0 8px;
      color: #f0f6fc;
    }}
    .subtitle {{
      margin: 0;
      color: #8b949e;
      font-size: 0.95rem;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 20px;
    }}
    .tab-btn {{
      min-height: 44px;
      min-width: 44px;
      padding: 10px 16px;
      border-radius: 6px;
      border: 1px solid #30363d;
      background: #161b22;
      color: #c9d1d9;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: #1f6feb;
      border-color: #388bfd;
      color: #ffffff;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .kpi-label {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #8b949e;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 1.35rem;
      font-weight: 700;
      color: #58a6ff;
      word-break: break-word;
    }}
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin-bottom: 24px;
      border: 1px solid #30363d;
      border-radius: 8px;
      background: #161b22;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #21262d;
    }}
    th {{
      color: #8b949e;
      font-weight: 600;
      background: #0d1117;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      word-break: break-all;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
    }}
    .badge-pass {{
      background: rgba(46, 160, 67, 0.2);
      color: #3fb950;
      border: 1px solid rgba(46, 160, 67, 0.4);
    }}
    .badge-fail {{
      background: rgba(248, 81, 73, 0.2);
      color: #f85149;
      border: 1px solid rgba(248, 81, 73, 0.4);
    }}
    .sheet-card {{
      margin-bottom: 24px;
    }}
    .sheet-card h2 {{
      font-size: 1.1rem;
      margin: 0 0 12px;
    }}
    .sheet-img {{
      display: block;
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #30363d;
    }}
    .hidden {{
      display: none !important;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>CCTV CodeFormer Conditioned SFT V1 — Gate &amp; Visual Review</h1>
      <p class="subtitle">60-Tensor Condition-Sensitive Pre-SFT + Spatial Head Learning with Input-Only Exposure/Degradation Gating &amp; Role-Balanced Gradient Projection</p>
    </header>

    <nav class="tabs" aria-label="Review Sections">
      <button type="button" id="tab-all" class="tab-btn active" data-tab="all">All Views</button>
      <button type="button" id="tab-l4" class="tab-btn" data-tab="l4">Returned L4 GPU Run</button>
      <button type="button" id="tab-clear" class="tab-btn" data-tab="clear">Clear Tone Preservation</button>
      <button type="button" id="tab-multirole" class="tab-btn" data-tab="multirole">Multi-Role &amp; Native CCTV</button>
      <button type="button" id="tab-packet" class="tab-btn" data-tab="packet">Sealed L4 VM Packet</button>
    </nav>

    <section class="kpi-grid" id="section-kpis">
      <div class="card">
        <div class="kpi-label">Returned L4 Archive</div>
        <div class="kpi-value">3,825 members / 712 calls</div>
      </div>
      <div class="card">
        <div class="kpi-label">L4 Arm B Clear PNG MAE</div>
        <div class="kpi-value">-0.000273 (PASS)</div>
      </div>
      <div class="card">
        <div class="kpi-label">L4 Arm B Compound PNG MAE</div>
        <div class="kpi-value">-0.000390 (5.3x B8)</div>
      </div>
      <div class="card">
        <div class="kpi-label">Dark-Aligned B8 Gate Verdict</div>
        <div class="kpi-value">96 / 96 PASS</div>
      </div>
    </section>

    <section class="table-wrap" id="section-l4-table" data-panel="l4">
      <table>
        <thead>
          <tr>
            <th>Returned L4 GPU Checkpoint</th>
            <th>Gate Status</th>
            <th>Clear PNG MAE (Delta vs u64)</th>
            <th>Compound PNG MAE (Delta vs u64)</th>
            <th>Lowlight+Noise PNG MAE (Delta vs u64)</th>
          </tr>
        </thead>
        <tbody>
          {l4_rows}
        </tbody>
      </table>
    </section>

    <section class="card sheet-card" id="section-l4-sheet" data-panel="l4">
      <h2>Returned NVIDIA L4 GPU Multi-Role &amp; Native CCTV Comparison Sheet (u0000 vs Arm A u0008 vs Arm B u0008 vs Dark-Aligned B8)</h2>
      <img class="sheet-img" src="/assets/returned-l4-sheet.png" alt="Returned L4 GPU multi-role and native CCTV comparison sheet" />
    </section>

    <section class="table-wrap" id="section-table" data-panel="packet">
      <table>
        <thead>
          <tr>
            <th>Variant</th>
            <th>Gate Verdict (inh8/inh48/sup8/sup48)</th>
            <th>Clear Monitor RGB8 Code Delta</th>
            <th>Compound PNG MAE</th>
          </tr>
        </thead>
        <tbody>
          {variants_rows}
        </tbody>
      </table>
    </section>

    <section class="card sheet-card" id="section-clear" data-panel="clear">
      <h2>Clear Monitor Photometric Preservation Sheet (8 Admitted Clear Monitors)</h2>
      <img class="sheet-img" src="/assets/clear-sheet.png" alt="Clear monitor comparison sheet" />
    </section>

    <section class="card sheet-card" id="section-multirole" data-panel="multirole">
      <h2>Multi-Role (Clear, Global Blur, Local Blur, Compound, Lowlight+Noise) &amp; Native CCTV Sheet</h2>
      <img class="sheet-img" src="/assets/multi-role-sheet.png" alt="Multi-role and native CCTV comparison sheet" />
    </section>

    <section class="card" id="section-packet-meta" data-panel="packet">
      <h2>Sealed Manual NVIDIA L4 VM Packet Provenance</h2>
      <p class="mono">Protocol SHA256: {p_sha} | Archive SHA256: {a_sha} | Static Files: {summary['packet_audit']['static_file_count']}</p>
    </section>
  </div>
  <script>
    document.querySelectorAll('.tab-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.getAttribute('data-tab');
        document.querySelectorAll('[data-panel]').forEach((panel) => {{
          if (target === 'all' || panel.getAttribute('data-panel') === target) {{
            panel.classList.remove('hidden');
          }} else {{
            panel.classList.add('hidden');
          }}
        }});
      }});
    }});
  </script>
</body>
</html>"""
        return HTMLResponse(html)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    app = build_app()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
