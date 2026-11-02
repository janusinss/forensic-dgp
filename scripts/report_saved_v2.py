"""Second-pass index/pixel recheck and static local V2 report, no model work."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/completion_codeformer_coverage_v2_return'
RETURN=PARENT/'cctv_completion_codeformer_coverage_v2_results'
STAGE=ROOT/'outputs/completion_codeformer_coverage_v2_packet/draft_003/cctv_completion_codeformer_coverage_v2_vm'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pix(path,mode='RGB'):
    with Image.open(path) as im:return np.array(im.convert(mode),copy=True)
def same(a,b,message):need(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),message)
def main():
    started=time.monotonic();indexpath=PARENT/'visual_sheet_index.json';index=read(indexpath);p=read(RETURN/'protocol.json');saved=read(PARENT/'independent_saved_output_audit.json');state=read(PARENT/'independent_state_audit.json');need(index['complete'] and saved['complete'] and state['complete'],'Required audits incomplete')
    need(index['sheets_count']==20 and index['unique_cases']==80 and index['available_evaluated_PNGs']==224,'Wrong preview scope')
    case_map={kind:{row['id']:row for row in p[key]} for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases'),('public_development','development_cases')]}
    try:small=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',12)
    except OSError:small=ImageFont.load_default()
    none_note=Image.new('RGB',(256,256),(220,216,208));draw=ImageDraw.Draw(none_note)
    for i,line in enumerate(('UNPAIRED photograph','No aligned clean reference','No hidden-identity metrics')):draw.text((12,65+22*i),line,fill=(60,55,50),font=small)
    unpaired=np.array(none_note);seen_cases=set();seen_pngs=set();cells=0;pages=[]
    for page in index['sheets']:
        path=Path(page['file']['path']);need(pin(path)==page['file'],'Sheet pin changed');sheet=pix(path);need(sheet.shape==(1226,len(page['headers'])*256,3),'Four-row sheet geometry differs')
        rowids=[]
        for row,record in enumerate(page['cases']):
            kind=record['kind'];ident=record['id'];case=case_map[kind][ident];need((kind,ident) not in seen_cases,'Duplicate visual case');seen_cases.add((kind,ident));rowids.append(ident)
            observed=pix(STAGE/case['input256']);mask=pix(STAGE/case['mask'],'L')==255;overlay=observed.copy();overlay[mask]=np.rint(overlay[mask].astype(np.float32)*.4+np.array([230,45,45],np.float32)*.6).astype(np.uint8)
            values=[pix(STAGE/case['clean256']) if kind!='public_development' else unpaired,observed,overlay]
            if kind=='monitor':values.append(pix(Path(record['historical_original']['path'])))
            need(not record['missing_snapshots'],'Unexpected missing snapshot in complete returned evaluation')
            for imagepin in record['all_snapshot_output_pins']:
                image=Path(imagepin['path']);need(pin(image)==imagepin,'Saved output pin changed');values.append(pix(image));seen_pngs.add(image.relative_to(RETURN).as_posix())
            need(len(values)==len(page['headers']),'Column count mismatch')
            for column,value in enumerate(values):same(sheet[104+286*row:360+286*row,column*256:(column+1)*256],value,'Reopened coordinatecell mismatch');cells+=1
        pages.append({'sheet':page['file'],'case_ids':rowids,'reopened_actual_pixel_cells_exact':True})
    allcases={(kind,ident) for kind,rows in case_map.items() for ident in rows};actualpngs={path.relative_to(RETURN).as_posix() for folder in (RETURN/'outputs').glob('u[0-9][0-9][0-9][0-9]') for kind in ('monitor','challenge','public_development') for path in (folder/kind).glob('*.png')}
    need(seen_cases==allcases and len(seen_cases)==80 and seen_pngs==actualpngs and len(seen_pngs)==224,'Nonrankedcase/outputcoverage mismatch')
    clear=saved['clear_incremental_vs_R1'];challenge=saved['challenge_incremental_vs_R1'];cumulative=saved['clear_cumulative_vs_original'];need(not clear['prospective_numeric_screen_pass'] and not challenge['prospective_numeric_screen_pass'],'Report cannot conceal gatefailure')
    need(state['committed_terminal_available'] is False and state['valid_new_recovery_checkpoint'] is False and state['partial_state']['status']=='INCOMPLETE_SERIALIZATION_NOT_A_CHECKPOINT','State status differs')
    report={'format':'completion-v2-visual-index-coordinate-recheck-v1','complete':True,'independent_author':False,'same_author_as_renderer':True,'scope':'Secondimplementationfixedcoordinatecheck against reopenedsource/input/mask/historical/newoutputPNGs. This is not a new independent human visualreview.','index':pin(indexpath),'source_protocol':pin(RETURN/'protocol.json'),'pages':pages,'reopened_sheets':20,'unique_cases':80,'all224evaluatedPNGs_displayed':True,'exact_coordinate_cells':cells,'no_cherrypick_or_outputranking':True,'model_loads':0,'neural_calls':0,'gradients':0,'VM_calls':0,'seconds':time.monotonic()-started}
    receipt=PARENT/'visual_index_coordinate_recheck.json'
    with receipt.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
    sections=[]
    for group,title in [('original_monitor','Original clear TRAIN monitor — paired'),('challenge','New synthetic TRAIN challenges — paired'),('public_DEV','Public development photographs — unpaired')]:
        cards=[]
        for page in index['sheets']:
            path=Path(page['file']['path'])
            if not path.name.startswith(group+'_'):continue
            relative=path.relative_to(PARENT).as_posix();ids=', '.join(r['id'] for r in page['cases'])
            cards.append('<article><h3>'+html.escape(path.stem)+'</h3><p>'+html.escape(ids)+'</p><a href="'+html.escape(relative)+'" target="_blank"><img src="'+html.escape(relative)+'" loading="lazy" alt="'+html.escape(page['title'])+'"></a><p class="hint">Click the sheet to inspect its original pixel size. Embedded thumbnails are only for navigation.</p></article>')
        sections.append('<section><h2>'+html.escape(title)+'</h2>'+''.join(cards)+'</section>')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Completion V2 — retained output review</title><style>body{margin:0;background:#f3f1eb;color:#172027;font:16px/1.5 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}h1,h2,h3{line-height:1.2}h1{font-size:30px}h2{margin-top:38px}.status{background:#fff3d9;border:1px solid #b67925;padding:18px;border-radius:8px}.note{background:#e8eef1;padding:16px;border-radius:8px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #bac2c7;text-align:left;padding:12px}a{color:#124e7a}article{margin:22px 0 34px;padding:18px;background:#fff;border:1px solid #c6cbc8;border-radius:8px}img{display:block;max-width:100%;height:auto;background:#dedbd3}.hint{font-size:13px;color:#4d5960}code{overflow-wrap:anywhere}nav{display:flex;flex-wrap:wrap;gap:14px}summary{cursor:pointer}</style></head><body><main>
<h1>Completion V2: all retained results</h1><p>26 October 2026 · FACE_MASK_COMPLETION research review · Read-only local report</p>
<div class="status"><strong>V2 did not qualify. Trained R1 remains the working completion baseline.</strong><p>All 64 learning-cycle receipts and 224 evaluated PNGs were retained, but the VM worker hit its 1,800-second hard limit while saving the terminal state. The returned <code>terminal_training_state.pth.partial</code> is incomplete serialization, not a usable checkpoint. No V2 model is packaged, adopted or available for inference from this return.</p></div>
<h2>Separate outcomes</h2><table><thead><tr><th>Evidence cohort</th><th>Change versus frozen R1</th><th>Required endpoint result</th></tr></thead><tbody>
<tr><td>32 original clear adaptive TRAIN monitor cases</td><td>__CLEAR__% delivered PNG hole-L1 gain (negative means worse)</td><td>Failed incremental gain and nonworsening requirements; eye-band and single-lens hole errors regressed.</td></tr>
<tr><td>16 new synthetic TRAIN challenges</td><td>+__CHALLENGE__% delivered PNG hole-L1 gain</td><td>Failed complete nonworsening screen: cheek/object hole errors regressed. This average cannot cancel failures on clear cases.</td></tr>
<tr><td>Same original 32 monitor cases, versus original CodeFormer</td><td>+__CUMULATIVE__% cumulative gain</td><td>Below retained R1’s +6.1853%. The historical comparison does not make V2 better than R1.</td></tr>
<tr><td>32 public development photographs</td><td>Unpaired visual evidence only</td><td>No aligned hidden clean reference. No hidden-identity, PSNR/SSIM, ethnicity, native CCTV or Zamboanga-performance claim.</td></tr>
</tbody></table>
<p class="note">Exact copied pixels and useful completion are different requirements. All evaluated outside-mask source pixels and raw-to-PNG compositions pass the independent saved-output audit. Copied covering fragments outside a reviewed mask remain copied source; cloth-like texture or anatomy defects inside that mask are generated content. Passing pixel preservation does not certify hidden anatomy or identity.</p>
<nav><a href="independent_saved_output_audit.json">Saved-output audit</a><a href="independent_state_audit.json">State/timeout audit</a><a href="transport_audit.json">Return integrity audit</a><a href="visual_sheet_index.json">All sheet/output pins</a><a href="visual_index_coordinate_recheck.json">Exact-cell coverage recheck</a></nav>
<p>Every available evaluation image appears below in fixed protocol order, without error ranking or omitted cases: 20 sheets, 80 unique cases, 224 saved evaluation outputs. Original monitor sheets also show original CodeFormer, frozen R1 and every V2 snapshot. “u64” identifies an evaluated snapshot; it does not imply that a usable u64 checkpoint survived.</p>
__SECTIONS__
<h2>Review limits and next action</h2><p>This report only displays retained artifacts. It does not run a model, start a server, change the application, resume training or reconstruct a partial checkpoint. Paired cohorts remain adaptive original TRAIN evidence, not independent identities or FINAL. Automatic covering proposals and full-family usefulness remain unqualified. Keep trained R1 and all V2 evidence while the save failure and failed quality endpoints guide a distinct next decision.</p>
</main></body></html>'''
    page=page.replace('__CLEAR__',format(clear['gain_percent'],'.4f')).replace('__CHALLENGE__',format(challenge['gain_percent'],'.4f')).replace('__CUMULATIVE__',format(cumulative['gain_percent'],'.4f')).replace('__SECTIONS__',''.join(sections))
    target=PARENT/'review.html'
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(page)
    need(page.count('<img ')==20 and '<script' not in page.lower(),'Report imagecount/scriptcheck')
    print(json.dumps({'report':pin(target),'coordinate_recheck':pin(receipt),'sheets':20,'cases':80,'outputs':224,'exact_cells':cells,'seconds':time.monotonic()-started}))
if __name__=='__main__':main()