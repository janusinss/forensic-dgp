"""Independent factual closure of the actual predictor-CE return milestone.
Uses already audited state/metrics and exact displayed image bytes; no model load.
"""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,json,math,os,re
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_return'
CHECKS=0

def need(ok,message):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(message)

def ip(path):
 p=Path(path).resolve();return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p

def sha(path):
 h=hashlib.sha256()
 with ip(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def read(path):return json.loads(ip(path).read_text(encoding='utf-8-sig'))
def pin(path):return {'path':str(Path(path).resolve()),'bytes':ip(path).stat().st_size,'sha256':sha(path)}
def checkpin(row):need(pin(row['path'])==row,'Exact milestone pin '+row['path'])
def near(a,b,message):need(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),message)

class Page(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.csp=[]
 def handle_starttag(self,tag,attrs):
  need(tag not in ('script','form','iframe','frame','object','embed','base'),'Read-only HTML no executable/network submission');d=dict(attrs)
  for k,v in attrs:
   need(not k.lower().startswith('on'),'No inline HTML eventhandlers')
   if k in ('src','href'):self.links.append(v)
  if tag=='meta' and d.get('http-equiv','').lower()=='content-security-policy':self.csp.append(d.get('content',''))

def main():
 mp=BASE/'return_milestone_v1.json';need(sha(mp)=='181b00064b9e3c4862095d651c8298a2e3f00b75d6c8f8c2071c8806a0c0c1d9','Exact root return milestone');m=read(mp);need(m['complete'] is True and m['status']=='EXECUTED_32_UPDATES_VALID_STATE_FAILED_QUALITY_R1_RETAINED' and not m['next_packet_ready'] and not m['next_actual_training_launched'] and not m['goal_complete'] and m['full_goal_status']=='ACTIVE_INCOMPLETE','Current milestone status')
 allpins=[m['author'],m['results_report'],*m['return_evidence'].values(),*m['independent_evidence'].values()]
 for row in allpins:checkpin(row)
 report=ip(m['results_report']['path']).read_text(encoding='utf-8');need(len(ip(m['results_report']['path']).read_bytes())==7533 and sha(m['results_report']['path'])=='85f1aef910241b3d91a3f8ca29740d37dabb4585ce6acf949ab4d231452d0aa0','Exact final report')
 state=read(ROOT/'research/completion_codeformer_predictor_ce_v1_state_review/independent_state_Adam_logits_audit.json');pixel=read(BASE/'independent_saved_output_audit.json');geom=read(BASE/'root_saved_code_geometry_diagnosis.json');closure=read(BASE/'root_endpoint_and_preservation_closure.json');transport=read(BASE/'transport_audit.json');fit=read(ROOT/'research/completion_codeformer_post_predictor_ce_v1_review/FIT_loss_geometric_decomposition.json');visual=read(BASE/'root_visual_review.json');index=read(BASE/'visual_sheet_index.json')
 need(state['complete'] and pixel['complete'] and transport['complete'] and not state['model_qualified'] and not pixel['final_prospective_numeric_screen_pass'],'Independent actual state valid/qualityfailed evidence');need(state['newphase_counts']=={'forward_slots':384,'backward_calls':128,'predictor_optimizer_steps':32,'empty_bypasses':168} and state['actual_hook_counts']=={'calls':384,'generator_calls':256,'code_only_calls':128},'Reported operation counts')
 need(state['model_state']['full_tensors']==497 and state['model_state']['actual_changed_predictor_tensors']==114 and state['model_state']['frozen_bitexact_R1']==383 and state['model_state']['original_prior165_exact'] and state['FIT_CE_replay']['arrays']==128 and state['FIT_CE_replay']['gradient_clip_steps']==29,'Reported actual learning/tensor/CE evidence');need(all(c['cycles_zero_gradient']==0 for c in state['model_state']['actual_predictor_changes']),'All114gradientlogs nonzeroeach32steps')
 need(len(transport['files'])==m['precise_regular_return_count']==835 and m['precise_manifest_payload_count']==834 and any(f['path']=='export_manifest.json' for f in transport['files']),'Precise835regular834payload counts');need('including834 manifest-listed payloads' in report and 'payload_files_rehashed:835' in report and 'precise manifest-listed payload count is834' in report,'Count clarification in final report')
 need(state['terminal_save']['serialized_bytes']==701933925 and state['supervisor']['worker_exit']==2 and not state['supervisor']['timeout'],'Reported complete state andexit2');need(pixel['all_retained_NPZ']==468 and pixel['actual_evaluation_PNGs']==256 and pixel['outside_source_RGB_bytes_exact']==44001540 and len(pixel['historical_R1_48case_compatibility'])==48 and len(pixel['fresh_R1_parity_pairs'])==4,'All raw/output/control scopes')
 claims={}
 for kind,n in [('monitor',32),('challenge',16)]:
  a=closure['separate_cohort_comparisons']['32'][kind];expectedgain=.30756100669542654 if kind=='monitor' else .44602730460013545;near(a['PNG_hole_error_reduction_percent'],expectedgain,'Final PNG gain claim');need(len(a['individual_holes_worse'])==(14 if kind=='monitor' else 5) and len(a['individual_boundaries_worse'])==(15 if kind=='monitor' else 6),'Individual regression counts');claims[kind]={'PNG_gain_percent':a['PNG_hole_error_reduction_percent'],'worse_holes':len(a['individual_holes_worse']),'worse_boundaries':len(a['individual_boundaries_worse'])}
  pg=geom['aggregates'][kind];all0=pg['0']['macro']['all'];all32=pg['32']['macro']['all'];need(all0['tokens']==all32['tokens']==n*256,'Fixed wholegrid count');matches0=round(all0['pooled_token_agreement']*n*256);matches32=round(all32['pooled_token_agreement']*n*256);need((matches0,matches32)==((5183,5182) if kind=='monitor' else (1331,1344)),'Wholegrid cleanlabel matches');need(all32['pooled_changed_tokens']==(386 if kind=='monitor' else 251),'Hardcode actual changed positions');claims[kind].update(matches_before=matches0,matches_after=matches32,changed_positions=all32['pooled_changed_tokens'])
  t0=pg['0']['macro']['touched']['pooled_token_agreement']*100;t32=pg['32']['macro']['touched']['pooled_token_agreement']*100;need(f'{t0:.4f}%' in report and f'{t32:.4f}%' in report,'Root touchedcell rounded percentages supported')
 gray=closure['GRAY_PNG_style']['32'];need(f"{gray['macro']['baseline']:.6f}" in report and f"{gray['macro']['candidate']:.6f}" in report and sum(not r['nonworse'] for k,r in gray.items() if k!='macro')==3 and gray['geometry/eye_band']['nonworse'],'Reported gray macrothreegeometry regression');fa=fit['aggregates']['all']['regions'];fraction=100*fa['touched']['tokens']/fa['all']['tokens'];need(f'{fraction:.3f}%' in report and f"{100*fa['touched']['fraction_all_CE']:.3f}%" in report and f"{100*fa['visible']['fraction_all_CE']:.3f}%" in report,'Saved FITallocation exact rounding');need('not actual neural receptive fields' in report and 'cannot be presented as a before/after learning curve' in report and 'does not prove visible' in report,'Correct diagnosis limits');need('no execution packet or launch guide is ready' in report and '108-call' in report,'Prospective nextdesign status only')
 # The five document transitions preserve every prior byte at its declared boundary.
 docqa=[]
 need(len(m['documents'])==5,'Exactlyfive document changes')
 for d in m['documents']:
  checkpin(d['before']);checkpin(d['after']);old=ip(d['before']['path']).read_bytes();new=ip(d['after']['path']).read_bytes();off=d['old_bytes_offset'];need(new[off:off+len(old)]==old and d['previous_bytes_exact'],'All old document bytes exact')
  if d['mode']=='prepend':need(off+len(old)==len(new) and off>0,'Exact prior suffix')
  else:need(d['mode']=='append' and off==0 and new.startswith(old),'Exact prior prefix')
  docqa.append({'after':d['after'],'backup':d['before'],'old_bytes_offset':off,'all_previous_bytes_exact':True})
 guide0=ip(m['documents'][0]['after']['path']).read_bytes();guide1=ip(m['documents'][1]['after']['path']).read_bytes();need(guide0==guide1 and guide0.startswith(b'# HISTORICAL COMPLETED PILOT') and b'DO NOT RELAUNCH' in guide0[:442] and guide0[442:]==ip(m['documents'][0]['before']['path']).read_bytes(),'Both historicalguide copies identical exact10265suffix')
 # Inspect every raster crop at declared coordinates; do not infer visual usefulness from hashes.
 need(index['sheets_count']==visual['sheets_reviewed']==20 and index['unique_cases']==visual['cases_reviewed']==80 and index['exact256_cells']==496 and index['actual_evaluated_PNGs']==256 and visual['all_actual_saved_outputs_covered']==256 and visual['all496_exact256_cells_examined'],'All20rootreview coverage')
 pages={x['sheet']['path']:x for x in visual['sheet_pages']};need(len(pages)==20 and visual['reviewer']=='Root development assessment, not independent blinded human FINAL' and visual['independent_human_FINAL_pending'] and not visual['candidate_adopted'],'Development visual review notFINAL')
 cells=0;rendered_outputs=set();allcases=set();sheetpins=[]
 for s in index['sheets']:
  checkpin(s['file']);sheetpins.append(s['file']);need(s['file']['path'] in pages and pages[s['file']['path']]['sheet']==s['file'] and pages[s['file']['path']]['all_columns_and_rows_reviewed'] and bool(pages[s['file']['path']]['observations']),'Root inspected each exact sheet with observations')
  with Image.open(ip(s['file']['path'])) as im:need(im.mode=='RGB' and list(im.size)==s['size'],'Actual fixedsheetgeometry');raster=np.array(im)
  for case in s['cases']:
   allcases.add(case['id']);need(len(case['display_cells'])==len(s['headers']),'Every declared column present')
   for c in case['display_cells']:
    box=c['box_xyxy'];x0,y0,x1,y1=box;need(x1-x0==y1-y0==256 and 0<=x0<x1<=raster.shape[1] and 0<=y0<y1<=raster.shape[0],'Exact256displaycell boundingbox');crop=raster[y0:y1,x0:x1];need(hashlib.sha256(crop.tobytes()).hexdigest()==c['pixels']['C_order_SHA256'] and c['pixels']['shape']==[256,256,3] and c['pixels']['dtype']=='uint8','Exact declared cell RGB bytes')
    if 'file' in c:
     checkpin(c['file'])
     with Image.open(ip(c['file']['path'])) as im:original=np.array(im.convert('RGB'))
     need(np.array_equal(crop,original),'Sheetcell noalter/resamplevsactualsourcePNG')
     if c['source_type'] in ('saved_R1_baseline','saved_experimental_predictor_only_readout'):rendered_outputs.add(c['file']['path'])
    elif c['source_type']=='derived_reviewed_support_overlay':
     checkpin(c['input_file']);checkpin(c['mask_file'])
     with Image.open(ip(c['input_file']['path'])) as im:source=np.array(im.convert('RGB'))
     with Image.open(ip(c['mask_file']['path'])) as im:mask=np.array(im)==255
     overlay=source.copy();overlay[mask]=np.rint(source[mask].astype(float)*.4+np.array([230,45,45])*.6).astype(np.uint8);need(np.array_equal(crop,overlay),'Exact reviewed mask-only overlay')
    else:need(c['source_type']=='derived_no_reference_information_tile' and c['not_a_reference_image'] is True,'Public unpaired tile explicitlynotreference')
    cells+=1
 need(cells==496 and len(allcases)==80 and len(rendered_outputs)==256,'All80cases496cells256outputs exact coverage')
 page=Page();html=ip(BASE/'review.html').read_text(encoding='utf-8');page.feed(html);need(len(page.links)==44 and len(page.csp)==1 and "default-src 'none'" in page.csp[0] and "form-action 'none'" in page.csp[0],'HTML44confinedrefs/CSP');urls=[]
 for link in page.links:
  u=urlsplit(link);need(not u.scheme and not u.netloc and not u.query,'No external/network URL');q=(BASE/unquote(u.path)).resolve();need(q.is_relative_to(BASE) and ip(q).is_file(),'Every HTMLref existing confined');urls.append(str(q))
 need({str(Path(s['file']['path']).resolve()) for s in index['sheets']}<set(urls),'All20sheets reachable');need('not selectable checkpoints' in html and 'no aligned hidden clean target' in html and 'Root visual outcome' in html,'Current readonly review status')
 admission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');r1pins=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']];need(len(r1pins)==22,'All22 R1pins')
 for row in r1pins:checkpin(row)
 # Require explicitly incomplete final scope in each new current document portion.
 for text in [report,ip(ROOT/'PROJECT_HANDOFF.md').read_text(encoding='utf-8-sig')[:2021],ip(ROOT/'FACE_MASK_COMPLETION_SPEC_V1.md').read_text(encoding='utf-8-sig')[-1393:]]:
  need('ACTIVE' in text and ('incomplete' in text or 'remain incomplete' in text) and ('FINAL' in text) and ('CPU' in text),'Fullscopeactive/notquietlyshrunk')
 result={'format':'independent-predictor-CE-v1-actual-return-milestone-closure','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'root_milestone':pin(mp),'results_report':m['results_report'],'checks_passed':CHECKS,'all_bound_evidence_pins_exact':True,'scientific_numeric_and_status_claims_supported':True,'metrics':claims,'precise_counts':{'regular_return_files_including_manifest':835,'manifest_listed_payloads':834,'retained_NPZ':468,'evaluation_PNGs':256,'exact_outside_RGB_bytes':44001540},'doc_transition_checks':docqa,'historical_guides_identical_previous10265bytes_exact_at442':True,'visual_material_integrity':{'sheets_rehashed':20,'unique_cases':80,'exact256cells_rechecked':496,'all256distinct_saved_output_cells_match_actualPNGs':True,'mask_overlays_exact':True,'public_information_tiles_explicitlynotreferences':True,'root_all20visual_assessment_records_match_exact_sheetpins':True,'independent_visual_usefulness_review_performed_in_this_QA':False,'independent_human_FINAL_still_pending':True},'HTML':{'attribute_links_checked':44,'unique_existing_confined_targets':len(set(urls)),'noJSforms_externalrequests':True,'all20sheetslinked':True},'all22_current_R1_app_model_pins_exact':True,'next_successor_packet_ready':False,'next_training_launched':False,'working_model':'R1 retained','model_qualified':False,'full_goal_complete':False,'local_actions':{'model_loads':0,'Torch_imported':False,'neural_calls':0,'gradients':0,'optimizer_updates':0,'VM_calls':0,'root_documents_modified':False}}
 target=Path(__file__).parent/'independent_return_milestone_closure.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'checks':CHECKS,'all5prior_document_bytes_exact':True,'sheets':20,'cells':496,'HTMLlinks':44,'R1pins':22,'model_qualified':False,'goal_complete':False},indent=2),flush=True)
if __name__=='__main__':main()
