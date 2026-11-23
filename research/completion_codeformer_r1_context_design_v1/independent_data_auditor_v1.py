"""Independent source-data replay audit, no Torch/models/gradients/VM/FINAL."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from collections import Counter
import hashlib,io,json,math,random,stat,sys,time
import numpy as np
from PIL import Image,ImageFilter,features
BASE=Path(__file__).resolve().parents[2]
ROOT=BASE/'outputs/completion_codeformer_coverage_v2_data'
OLD=BASE/'outputs/completion_codeformer_finetune_v1_packet/cctv_completion_codeformer_finetune_v1_vm'
KINDS=['lower_face','eye_band','image_left_lens','cheek_object'];CONDITIONS=['strict_gray_clear','RGB_mild_degraded','strict_gray_mild_degraded','RGB_clear_irregular']
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def strict_pin(row,rooted=True):
 path=Path(row['path']);need(path.is_file() and not path.is_symlink(),'Missing/linked pinned file')
 if rooted:
  name=PurePosixPath(row['relative_path']);need(not name.is_absolute() and '..' not in name.parts and '\\' not in row['relative_path'],'Traversal');need(path.resolve()==(ROOT/Path(*name.parts)).resolve() and path.resolve().is_relative_to(ROOT.resolve()),'Pinned file escaped data namespace')
 need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Pinned bytes differ '+str(path));return path
CACHE={}
def png(path,size,mode='RGB'):
 path=Path(path)
 if path not in CACHE:
  with Image.open(path) as im:
   need(im.format=='PNG' and im.size==(size,size) and im.mode==mode,'PNG shape/mode differs '+str(path));CACHE[path]=np.array(im,copy=True)
 return CACHE[path]
def eq(a,b,msg):need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),msg)
def gray(a):
 v=a.astype(np.float32);y=v[...,0]*np.float32(.299);y+=v[...,1]*np.float32(.587);y+=v[...,2]*np.float32(.114);q=np.rint(y).clip(0,255).astype(np.uint8);return np.stack((q,q,q),axis=-1)
def seed(label):
 h=hashlib.sha256(('completion-codeformer-coverage-v2|'+label).encode('ascii')).hexdigest();return {'label':label,'sha256':h,'uint64':int(h[:16],16)}
def visibility(mask):
 unavailable=mask!=0;yy,xx=np.mgrid[:256,:256];face=((xx-127.5)/83)**2+((yy-137)/104)**2<=1
 bands={'left_eye':(60,84,119,133),'right_eye':(137,84,196,133),'nose':(108,119,149,166),'mouth':(82,165,174,208)}
 fractions={k:float(unavailable[y1:y2,x1:x2].mean()) for k,(x1,y1,x2,y2) in bands.items()};fraction=float(unavailable[face].mean());all_features=fractions['left_eye']>=.85 and fractions['right_eye']>=.85 and fractions['nose']>=.9 and fractions['mouth']>=.9
 return {'rejected':bool(fraction>=.8 or all_features or unavailable.mean()>=.85),'face_covered_fraction':fraction,'feature_covered_fractions':fractions,'scope':'Approximate frontal-crop geometry, conditional on the reviewed removal area'}
def replay_corruption(row,target):
 c=row['corruption'];need(c['enabled'] is True,'Expected corrupted case')
 need(c['parameter_seed']==seed(row['id']+'|corruption_parameters') and c['noise_seed']==seed(row['id']+'|shared_noise'),'Wrong corruption seed')
 rng=np.random.default_rng(c['parameter_seed']['uint64']);params=(round(float(rng.uniform(.4,.8)),6),int(rng.integers(128,193)),round(float(rng.uniform(0,2)),6),int(rng.integers(75,91)))
 need(params==(c['Gaussian_sigma_pixels256'],c['downsample_side'],c['noise_std_8bit'],c['JPEG_quality']),'Frozen random parameters differ')
 image=Image.fromarray(target).filter(ImageFilter.GaussianBlur(params[0]));image=image.resize((params[1],params[1]),Image.Resampling.BILINEAR).resize((256,256),Image.Resampling.BILINEAR)
 noise=np.random.default_rng(c['noise_seed']['uint64']).normal(0,params[2],(256,256,1)).astype(np.float32);noisy=np.rint(np.array(image).astype(np.float32)+noise).clip(0,255).astype(np.uint8)
 image=Image.fromarray(noisy[:,:,0] if row['style']=='GRAY' else noisy);buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=params[3],subsampling=0,optimize=False,progressive=False);expected=buffer.getvalue()
 jpeg=strict_pin(c['JPEG_bytes']);need(jpeg.read_bytes()==expected,'JPEG encoder bytes differ')
 with Image.open(io.BytesIO(expected)) as image:decoded=np.array(image.convert('RGB'),copy=True)
 eq(decoded,png(row['input256']['path'],256),'JPEG decoded input differs');return {'id':row['id'],'JPEG_bytes_exact':True,'decoded_input_exact':True}
def replay_mask(row,base):
 p=row['mask_transform'];need(p['enabled'] and p['accepted_input_only'] and p['source_mask_changed'] and p['binary'],'Irregular mask flags differ')
 need(-8<=p['translation_x_pixels256']<=8 and -8<=p['translation_y_pixels256']<=8 and .85<=p['scale']<=1.15 and -10<=p['rotation_degrees']<=10 and 0<=p['attempt']<8,'Irregular parameter range')
 need(p['seed']==seed(row['id']+'|irregular_mask|attempt'+str(p['attempt'])),'Irregular seed differs')
 ys,xs=np.nonzero(base);cx=float(xs.mean());cy=float(ys.mean());need(p['center_xy']==[cx,cy],'Mask transform center differs')
 rad=math.radians(p['rotation_degrees']);a=math.cos(rad)/p['scale'];b=math.sin(rad)/p['scale'];d=-math.sin(rad)/p['scale'];e=math.cos(rad)/p['scale'];tx=p['translation_x_pixels256'];ty=p['translation_y_pixels256'];coeff=[a,b,cx-a*(cx+tx)-b*(cy+ty),d,e,cy-d*(cx+tx)-e*(cy+ty)];need(coeff==p['Pillow_inverse_affine_coefficients'],'Mask inverse affine differs')
 affine=np.array(Image.fromarray(base).transform((256,256),Image.Transform.AFFINE,coeff,resample=Image.Resampling.NEAREST,fillcolor=0),copy=True);q=p['boundary_variation'];need(1<=q['amplitude_x']<=2 and 1<=q['amplitude_y']<=2 and 32<=q['period_x_pixels']<=64 and 32<=q['period_y_pixels']<=64,'Boundary displacement range')
 yy,xx=np.mgrid[:256,:256];xm=np.floor(xx+q['amplitude_x']*np.sin(2*math.pi*yy/q['period_y_pixels']+q['phase_x_radians'])+.5).astype(np.int32);ym=np.floor(yy+q['amplitude_y']*np.sin(2*math.pi*xx/q['period_x_pixels']+q['phase_y_radians'])+.5).astype(np.int32);inside=(xm>=0)&(xm<256)&(ym>=0)&(ym<256);rebuilt=np.zeros((256,256),np.uint8);rebuilt[inside]=affine[ym[inside],xm[inside]]
 eq(rebuilt,png(row['mask256']['path'],256,'L'),'Irregular mask pixels do not replay');need(not np.array_equal(base,rebuilt),'Irregular mask did not change original geometry');need(p['geometry_gate']==visibility(rebuilt) and not p['geometry_gate']['rejected'],'Irregular visibility differs');return {'id':row['id'],'binary_affine_boundary_replay_exact':True,'attempts':p['attempt']+1}
def main():
 started=time.monotonic();manifest_path=ROOT/'manifest.json';need(sha(manifest_path)=='393bea14e4ba8a350644bc51d65c68a0eb57ba919b1c9d0954245c1f40935ea7','Final data manifest identity differs');m=read(manifest_path)
 files={r['relative_path']:r for r in m['files']};need(len(files)==len(m['files'])==551,'Data file count/duplicates')
 for row in files.values():strict_pin(row)
 need(sum(r['bytes'] for r in files.values())==m['payload_bytes_before_manifest']==57020130,'Data bytes differ')
 p=read(strict_pin(m['source_protocol'],False));need(m['source_protocol']['sha256']=='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d','Wrong source protocol');oldfiles={r['path']:r for r in p['files']};refs={r['id']:r for r in p['references']};newrefs={r['id']:r for r in m['references']}
 need(len(refs)==len(newrefs)==40 and set(refs)==set(newrefs),'Reference set differs')
 for ref,r in newrefs.items():
  o=refs[ref];need(r['old_reference_record']==o and r['attribution']==o['attribution'] and r['original_role']==r['legacy_completion_split']=='train' and r['role']==o['role'],'Role/provenance/terms changed')
  strict_pin(r['original_source1024'],False)
  with Image.open(r['original_source1024']['path']) as im:need(im.format=='PNG' and im.mode=='RGB' and im.size==(1024,1024),'GenuineHQ source geometry differs')
 fitrefs=sorted(r['id'] for r in p['references'] if r['role']=='fit');monrefs=sorted(r['id'] for r in p['references'] if r['role']=='monitor');need(len(fitrefs)==32 and len(monrefs)==8 and not set(fitrefs)&set(monrefs),'FIT/MONITOR source overlap')
 teachers={r['teacher_key']:r for r in m['teacher_target_sets']};need(len(teachers)==80 and set(teachers)=={ref+'::'+style for ref in refs for style in ('RGB','GRAY')},'Teacher keys differ')
 for key,t in teachers.items():
  ref=t['reference_id'];style=t['style'];need(key==ref+'::'+style,'Teacher appearance identity differs')
  for size in (256,512):
   row=t['target'+str(size)];path=strict_pin(row);a=png(path,size);oldrow=oldfiles[refs[ref]['clean'+str(size)]];source=OLD/oldrow['path'];need(sha(source)==oldrow['sha256'],'Original target changed');b=png(source,size)
   if style=='RGB':need(row['sha256']==oldrow['sha256'] and row['bytes']==oldrow['bytes'],'RGB original bytes changed');eq(a,b,'RGB target pixels differ')
   else:eq(a,gray(b),'GRAY source-matched target differs');need(np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2]),'GRAY target not replicated')
 groups={'fit_anchor_cases':128,'fit_augmented_cases':128,'monitor_original_cases':32,'monitor_challenge_cases':16};allrows=[]
 for key,n in groups.items():need(len(m[key])==n,'Case count '+key);allrows+=m[key]
 need(len(allrows)==len({r['id'] for r in allrows})==304,'Case IDs collide')
 original_by_id={r['id']:r for r in p['fit_cases']+p['monitor_cases']};corruptions=[];masks=[];case_summaries=[]
 for row in allrows:
  need(time.monotonic()-started<300,'Data audit300second stop');ref=row['reference_id'];need(ref in refs and row['original_role']==row['legacy_completion_split']=='train' and row['target_is_known_clean_TRAIN'] is True and row['real_covering_segmentation_truth'] is False,'NonTRAIN label/segmentation overclaim')
  need(row['prepared_bounds']==[0,0,256,256] and row['teacher_key']==ref+'::'+row['style'],'Case teacher/geometry differs');teacher=teachers[row['teacher_key']]
  for field in ('target256','target512'):need(row[field]==teacher[field],'Case target differs from teacher key')
  for field in ('input256','target256','target512','mask256'):strict_pin(row[field])
  target=png(row['target256']['path'],256);observed=png(row['input256']['path'],256);mask=png(row['mask256']['path'],256,'L');need(np.isin(mask,[0,255]).all() and int((mask!=0).sum())==row['mask_pixels'],'Mask not exact binary/pixel count');need(row['visibility']==visibility(mask) and not visibility(mask)['rejected'],'Input-only visibility mismatch')
  original=original_by_id[row['original_case_id']];need(original['reference_id']==ref and original['mask_kind']==row['mask_kind'],'Original geometry/reference mapping changed');base=png(OLD/original['mask'],256,'L')
  if row['mask_transform']['enabled']:masks.append(replay_mask(row,base))
  else:eq(mask,base,'Original delivery mask changed')
  if row['corruption']['enabled']:corruptions.append(replay_corruption(row,target))
  else:eq(observed,target,'Uncorrupted context differs from clean appearance target')
  if row['style']=='GRAY':need(np.array_equal(observed[:,:,0],observed[:,:,1]) and np.array_equal(observed[:,:,1],observed[:,:,2]),'GRAY input lost shared channel equality')
  need(row['case_seed']==seed(row['id']),'Case seed differs')
  case_summaries.append({'id':row['id'],'reference_id':ref,'condition':row['condition'],'style':row['style'],'mask_kind':row['mask_kind'],'corrupted':row['corruption']['enabled'],'irregular':row['mask_transform']['enabled']})
 need({r['id'] for r in m['fit_anchor_cases']}=={r['id'] for r in p['fit_cases']} and {r['id'] for r in m['monitor_original_cases']}=={r['id'] for r in p['monitor_cases']},'Anchor/monitor IDs changed')
 for row in m['fit_anchor_cases']+m['monitor_original_cases']:
  o=original_by_id[row['id']];need(row['input256']['sha256']==oldfiles[refs[row['reference_id']]['clean256']]['sha256'] and row['mask256']['sha256']==oldfiles[o['mask']]['sha256'] and row['style']=='RGB' and not row['corruption']['enabled'] and not row['mask_transform']['enabled'],'Original anchor/MONITOR bytes changed')
 for row in m['fit_augmented_cases']:
  i=fitrefs.index(row['reference_id']);g=KINDS.index(row['mask_kind']);condition=CONDITIONS[(i+g)%4];need(row['condition']==condition and row['style']==('GRAY' if condition.startswith('strict_gray') else 'RGB'),'Aug condition rotation differs')
  need(row['corruption']['enabled']==(condition in ('RGB_mild_degraded','strict_gray_mild_degraded')) and row['mask_transform']['enabled']==(condition=='RGB_clear_irregular'),'Aug operation differs')
 for row in m['monitor_challenge_cases']:
  i=monrefs.index(row['reference_id']);graycase=row['condition']=='strict_gray_mild_degraded';need(graycase or row['condition']=='RGB_irregular_mild_degraded','Challenge condition differs');need(row['mask_kind']==KINDS[(i+(0 if graycase else 2))%4] and row['style']==('GRAY' if graycase else 'RGB') and row['corruption']['enabled'] and row['mask_transform']['enabled']==(not graycase),'Challenge assignment differs')
 augcross=Counter((r['condition'],r['mask_kind']) for r in m['fit_augmented_cases']);challengecross=Counter((r['condition'],r['mask_kind']) for r in m['monitor_challenge_cases']);need(len(augcross)==16 and all(v==8 for v in augcross.values()) and len(challengecross)==8 and all(v==2 for v in challengecross.values()),'Condition/geometry crossing not balanced')
 need(len(corruptions)==80 and len(masks)==40,'Augmented artifact counts differ')
 sampler=read(strict_pin(m['sampler']));need(sampler['historical_original_schedule']==p['schedule'] and sampler['historical_R1_completed_cycles']==sampler['historical_R1_optimizer_step_perroute']==32 and sampler['newphase_cycles']==64,'Parent sampler history changed');sseed=seed('new_phase_sampler|8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d|55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6');need(sampler['separate_local_seed']==sseed,'Sampler seed differs');order=list(range(32));random.Random(sseed['uint64']).shuffle(order);need(sampler['parent_cycle_order_1based']==[i+1 for i in order],'Parent cycle shuffle differs')
 lookup={r['id']:r for r in m['fit_anchor_cases']+m['fit_augmented_cases']};auglookup={r['original_case_id']:r['id'] for r in m['fit_augmented_cases']};expected=[]
 for parent_i in order:
  oldids=p['schedule'][parent_i]
  for first in (True,False):
   a=oldids[:2] if first else oldids[2:];u=oldids[2:] if first else oldids[:2];ids=[a[0],auglookup[u[0]],a[1],auglookup[u[1]]];expected.append(ids)
 need(len(sampler['cycles'])==64,'New sampler cycle count')
 for i,row in enumerate(sampler['cycles']):
  need(row['cycle']==i+1 and row['case_ids']==row['code_case_ids']==row['fusion_case_ids']==expected[i],'Actual sampler order differs');need(len({lookup[n]['reference_id'] for n in row['case_ids']})==4 and sum(lookup[n]['role']=='fit_anchor' for n in row['case_ids'])==2,'Sampler ref/anchor diversity differs')
 for key in ('code_case_ids','fusion_case_ids'):
  flat=[n for row in sampler['cycles'] for n in row[key]];need(len(flat)==len(set(flat))==256 and set(flat)==set(lookup),'Sampler pair coverage differs')
 public=m['public_development_unchanged'];need(len(public)==32 and [r['original_case'] for r in public]==p['development_cases'] and all(r['clean_target'] is None for r in public),'UnpairedDEV altered or used as target')
 for row in public:
  for field,source_field in [('unchanged_input256','input256'),('unchanged_reviewed_mask','mask')]:
   actual=strict_pin(row[field],False);oldpin=oldfiles[row['original_case'][source_field]];need(row[field]['sha256']==oldpin['sha256'] and row[field]['bytes']==oldpin['bytes'],'PublicDEV bytes changed')
 preview=read(strict_pin(m['preview_index']));seen=[]
 for sheet in preview['sheets']:
  path=strict_pin(sheet['file']);canvas=np.array(Image.open(path).convert('RGB'))
  for entry in sheet['cases']:
   row=next(r for r in m['fit_augmented_cases']+m['monitor_challenge_cases'] if r['id']==entry['id']);seen.append(row['id']);target=png(row['target256']['path'],256);source=png(row['input256']['path'],256);mask=png(row['mask256']['path'],256,'L');masked=source.copy();masked[mask==255]=255;cells=[target,source,np.repeat(mask[:,:,None],3,axis=2),masked]
   for column,cell in enumerate(cells):eq(canvas[104+286*entry['row']:360+286*entry['row'],column*256:(column+1)*256],cell,'Saved preview cell mismatch')
 need(len(preview['sheets'])==20 and len(seen)==len(set(seen))==144 and set(seen)=={r['id'] for r in m['fit_augmented_cases']+m['monitor_challenge_cases']},'Preview does not cover all144 new cases')
 final_design=strict_pin(m['controlling_final_design']);need(sha(final_design)=='9566e77f7fb33b4116f744d043ccf5af1ff133222fb69d6ad0b047db324a4682','Controlling final design missing');strict_pin(m['controlling_final_design_document'])
 for row in files.values():strict_pin(row)
 need(sha(manifest_path)=='393bea14e4ba8a350644bc51d65c68a0eb57ba919b1c9d0954245c1f40935ea7','Manifest changed during admission');need('torch' not in sys.modules,'Data audit imported Torch')
 result={'format':'completion-coverage-v2-independent-source-data-admission-v1','complete':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'manifest':pin(manifest_path),'status':'SOURCE_DATA_NUMERICAL_ADMISSION_ONLY;ROOT_VISUAL_AND_TRAINER_PACKET_ADMISSIONS_SEPARATE','payload_files':551,'payload_bytes':57020130,'all_payload_hashes_exact_before_after':True,'controlling_final_design':m['controlling_final_design'],'original_references':{'FIT':32,'adaptive_originalTRAIN_MONITOR':8,'role_and_attribution_exact':True,'all40_genuineRGB1024_sources_rehashed':True,'identity_independence_not_established':True},'teacher_targets':{'sets':80,'styles':['RGB','GRAY'],'PNGs':160,'RGB80_originalbytecopies_exact':True,'GRAY80_pixel_transforms_replayed_exact':True,'latent_code_arrays_not_computed_locally':True,'future_teacher_definition_original_frozen_prequant_latent_nearest512codebook':True},'cases':{'anchor128_originalbytes_exact':True,'monitor32_originalbytes_exact':True,'augmented':128,'challenge':16,'all304_ids_unique':True,'all_masks_binary_visibility_pass':True,'aug_condition_geometry_cells':{a+'__'+b:v for (a,b),v in augcross.items()},'challenge_condition_geometry_cells':{a+'__'+b:v for (a,b),v in challengecross.items()},'summaries':case_summaries},'replay':{'degraded_input_cases':len(corruptions),'all80_seeded_blur_bilinear_sharednoise_JPEGbytes_and_PNG_pixels_exact':True,'irregular_masks':len(masks),'all40_recorded_affine_sine_boundary_pixels_exact':True,'strict_gray_inputs_and_targets_replicated_RGB':True,'degrade_precedes_masking':True,'corruption_limits_are_simulated_not_CCTV_calibrated':True},'sampler':{'actual64cycles_exact_seeded_parent_order_and_crosshalves':True,'two_anchors_two_augmented_four_distinct_references_everycycle':True,'all256_once_perroute_code_fusion_identical':True,'parent_phase_preserved':True,'local_RNG_objects_only':True},'public_DEV':{'cases':32,'inputs_and_masks_exact':True,'no_clean_targets':True,'not_used_for_training_teacher_labels':True,'not_native_CCTV_despite_filenames':True},'preview_artifact_replay':{'sheets':20,'new_cases':144,'all_four_cells_exact':True,'subjective_visual_admission_not_performed_by_this_auditor':True},'terms_and_overlap_limits':m['data_limits'],'local_execution':{'PNG_JPEG_and_metadata_reads_only':True,'preview_images_machine_replayed_not_subjectively_rated':True,'model_or_NPZ_or_FINAL_payload_reads':0,'Torch_imports_or_NN_gradients_optimizers':0,'VM_network_calls':0,'trainer_or_packet_created':False,'original_assets_or_application_sources_edited':False},'mandatory_future_gates':['Root review all20preview sheets/144newcases','Independent source-enforcedtrainer/RNG/Adam/gate/retention/CLI/serialization audit','Verified packet/manualL4 execution and returned output/state audits; no auto training'],'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
 target=ROOT/'independent_data_admission_v1.json'
 with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({'complete':True,'admission':pin(target),'case_counts':[128,128,32,16],'teacher_sets':80,'degradation_replays':80,'mask_replays':40,'preview_sheets':20,'seconds':result['seconds']},indent=2))
if __name__=='__main__':main()