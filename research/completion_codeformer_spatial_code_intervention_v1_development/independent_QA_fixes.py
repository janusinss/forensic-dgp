from pathlib import Path
import shutil,ast
D=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\research\completion_codeformer_spatial_code_intervention_v1_development');backup=D/'prior_revision_001_before_independent_QA_fixes';backup.mkdir()
for n in ('cctv_completion_codeformer_spatial_code_intervention_v1_vm.py','contracts.py','optimized_budget.py','source_tests.py','source_preparation_tests.json'):shutil.copyfile(D/n,backup/n)
c=D/'contracts.py';s=c.read_text();s+='''

def spatial_effects(summaries):
 """Plain lower-is-better additive effects; no neural selection or thresholds."""
 def endpoint(values):
  a,c,h,v=(values[key] for key in ('A','C','H','V'))
  differences={'H_minus_A':h-a,'V_minus_A':v-a,'C_minus_A':c-a,'C_minus_H_minus_V_plus_A':c-h-v+a}
  return {'values':values,'effects':differences,'exact_signs':{key:(-1 if value<0 else 1 if value>0 else 0) for key,value in differences.items()},'negative_effect_is_lower_endpoint':True}
 def analyze(records,metrics):
  need(records,'Nonempty fixed additive-effect cohort')
  cases=[]
  for item in records:
   cases.append({'id':item['A']['id'],'mask_kind':item['A']['mask_kind'],'condition':item['A']['condition'],'endpoints':{metric:endpoint({arm:item[arm][metric] for arm in ('A','C','H','V')}) for metric in metrics}})
  def macro(items):return {metric:endpoint({arm:sum(r[arm][metric] for r in items)/len(items) for arm in ('A','C','H','V')}) for metric in metrics}
  return {'cases':cases,'macro':macro(records),'geometry':{kind:macro([r for r in records if r['A']['mask_kind']==kind]) for kind in KINDS},'condition':{condition:macro([r for r in records if r['A']['condition']==condition]) for condition in sorted({r['A']['condition'] for r in records})}}
 result={'definition':'Lower-is-better H-A,V-A,C-A,C-H-V+A; all exact signs reported; oracle diagnostics never trained gains or adoption. Negativeinteraction is synergistically lower endpoint, not proof of deployable quality.','paired':{}}
 for kind in ('monitor','challenge'):
  tables={arm:{r['id']:r for r in summaries[(arm,kind)]['records']} for arm in ('A','C','H','V')};ids=list(tables['A'])
  need(all(list(tables[arm])==ids for arm in tables),'All four conditions have same fixed case order')
  joined=[{arm:tables[arm][ident] for arm in tables} for ident in ids]
  result['paired'][kind]=analyze(joined,METRICS)
  if kind=='challenge':
   gray=[r for r in joined if r['A']['style']=='GRAY'];need(len(gray)==8,'Fixed eightGRAY effect cases')
   result['GRAY']=analyze(gray,('GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread'))
 return result
''';ast.parse(s,feature_version=(3,10));c.write_text(s,encoding='utf-8',newline='\n')
p=D/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';s=p.read_text();s=s.replace('output_paths,VM_NAME,','output_paths,spatial_effects,VM_NAME,');s=s.replace("need(len(p['parity_cases'])==4 and len({r['id'] for r in p['parity_cases']})==4,'Four app parity cases')", "need(len(p['parity_cases'])==4 and len({r['id'] for r in p['parity_cases']})==4 and p['parity_cases']==design['data']['parity_cases_exact'],'Four exact prospectively declared app parity cases')")
s=s.replace("            with zipfile.ZipFile(partial) as archive:\n                infos=archive.infolist();need(0<len(infos)<=1024 and len({v.filename for v in infos})==len(infos),'Closed failure ZIP members')", "            with zipfile.ZipFile(partial) as archive:\n                roots=set();storage_records=0\n                infos=archive.infolist();need(0<len(infos)<=1024 and len({v.filename for v in infos})==len(infos),'Closed failure ZIP members')")
s=s.replace("need(not info.is_dir() and not info.flag_bits&1 and info.compress_type in (0,8) and not PurePosixPath(info.filename).is_absolute() and '..' not in PurePosixPath(info.filename).parts and '\\\\' not in info.filename,'Failure ZIP regular canonical members')", "name=PurePosixPath(info.filename)\n                    need(not info.is_dir() and not info.flag_bits&1 and info.compress_type==zipfile.ZIP_STORED and not name.is_absolute() and '..' not in name.parts and ':' not in info.filename and '\\\\' not in info.filename and len(name.parts)>=2 and str(name)==info.filename,'Failure ZIP regular canonical stored members')\n                    need(len(info.filename.encode('utf-8'))+len(info.extra)+len(info.comment)<=512 and info.file_size<=CONTRACT['terminal_bytes'],'Failure ZIP bounded fields/bytes')\n                    roots.add(name.parts[0]);relative='/'.join(name.parts[1:])\n                    need(relative in ('data.pkl','byteorder','version','.data/serialization_id','.format_version','.storage_alignment') or relative.startswith('data/') and relative[5:].isdigit(),'Closed Torch metadata and digit-storage namespace')\n                    if relative.startswith('data/'):storage_records+=1")
s=s.replace("                        for chunk in iter(lambda:source.read(1024**2),b''):budget.guard(terminal=True,emergency=True)\n            digest=sha(partial)", "                        for chunk in iter(lambda:source.read(1024**2),b''):budget.guard(terminal=True,emergency=True)\n                need(len(roots)==1 and storage_records<=1018 and len(archive.comment)<=1024,'Single failure ZIP prefix and bounded tensor storages')\n            digest=sha(partial)")
s=s.replace("'GRAY_style_diagnostics':gray,", "'GRAY_style_diagnostics':gray,'spatial_additive_effects':spatial_effects(all_summaries),")
ast.parse(s,feature_version=(3,10));p.write_text(s,encoding='utf-8',newline='\n')
# Keep original eighteen tests but add exact additive-fixture checks.
t=D/'source_tests.py';s=t.read_text();marker=" def test_18_root_design_and_manual_owned_supervisor(self):";at=s.index(marker);s=s[:at]+''' def test_19_exact_additive_effects_all_endpoints_signs(self):
  rows={}
  for arm,value in [('A',4.),('C',1.),('H',2.),('V',3.)]:
   for cohort in ('monitor','challenge'):
    records=[]
    for index,kind in enumerate(c.KINDS):
     for repeat in range(2):records.append({'id':cohort+str(index)+str(repeat),'mask_kind':kind,'condition':'strict_gray','style':'GRAY',**{key:value for key in (*c.METRICS,'GRAY_raw_mean_channel_spread','GRAY_PNG_mean_channel_spread')}})
    rows[(arm,cohort)]={'records':records}
  actual=c.spatial_effects(rows);endpoint=actual['paired']['monitor']['macro']['PNG_hole_L1'];self.assertEqual(endpoint['effects'],{'H_minus_A':-2.,'V_minus_A':-1.,'C_minus_A':-3.,'C_minus_H_minus_V_plus_A':0.});self.assertEqual(endpoint['exact_signs']['C_minus_H_minus_V_plus_A'],0);self.assertEqual(len(actual['GRAY']['cases']),8);self.assertEqual(actual['GRAY']['geometry']['lower_face']['GRAY_PNG_mean_channel_spread']['effects']['C_minus_A'],-3.)
''' +s[at:];ast.parse(s,feature_version=(3,10));t.write_text(s,encoding='utf-8',newline='\n')
# Prior passing receipt is preserved byteexact above; remove only its original location to publish a new revision receipt.
receipt=D/'source_preparation_tests.json';receipt.unlink()
print({'worker_bytes':p.stat().st_size,'parity_exact_binding':"p['parity_cases']==design['data']['parity_cases_exact']" in p.read_text(),'ZIP_namespace_closed':'Closed Torch metadata and digit-storage namespace' in p.read_text()})
