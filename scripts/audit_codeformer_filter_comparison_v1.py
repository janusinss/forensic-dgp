"""Independent saved-pixel metrics for a fixed filter comparison; no neural runtime."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = '7823847ce7a709308fe1f4d35d35e2c14f752b2d5c303f5fa1cb4b7c8de8c3a8'
REVIEW_SHA = '51bff1593cd33ad2b369a7c20d2b5282ed67521df4b62e17a53f082941fc3b50'
Y = np.asarray([.2126,.7152,.0722], np.float64)
ARMS = ('original0','parent50','candidate50')
FLAGGED_REFERENCE = 'va_ffhq_08025'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_json(data):
    def unique(items):
        result = {}
        for key,value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def normalized_rgb(array):
    require(isinstance(array,np.ndarray) and array.ndim==3 and array.shape[2]==3 and
            min(array.shape[:2])>0 and array.dtype in (np.dtype('float32'),np.dtype('float64')) and
            np.isfinite(array).all() and array.min()>=0 and array.max()<=1,
            'Finite normalized RGB floats required')
    return array.astype(np.float64)


def residual_metrics(predicted, clean, support):
    """Reference disagreement; offset/gradient diagnostics are not acceptance gates."""
    predicted,clean=normalized_rgb(predicted),normalized_rgb(clean)
    require(predicted.shape==clean.shape and isinstance(support,np.ndarray) and
            support.dtype==np.bool_ and support.shape==clean.shape[:2] and support.any(),
            'Aligned clean geometry and nonempty boolean observed support required')
    residual=predicted-clean
    observed=residual[support]
    offset=observed.mean(axis=0,dtype=np.float64)
    centered=observed-offset
    mae=float(np.abs(observed).mean(dtype=np.float64))
    mse=float(np.square(observed).mean(dtype=np.float64))
    centered_mse=float(np.square(centered).mean(dtype=np.float64))
    energy=float(np.square(offset).mean(dtype=np.float64))
    closure=mse-(centered_mse+energy)
    valid_x=support[:,1:] & support[:,:-1]
    valid_y=support[1:,:] & support[:-1,:]
    nx,ny=int(valid_x.sum()),int(valid_y.sum())
    dx=np.diff(residual,axis=1)[valid_x]
    dy=np.diff(residual,axis=0)[valid_y]
    sum_x=float(np.abs(dx).sum(dtype=np.float64))
    sum_y=float(np.abs(dy).sum(dtype=np.float64))
    if support.all() and min(clean.shape[:2])>=7:
        ssim=float(structural_similarity(predicted,clean,data_range=1.,channel_axis=2,win_size=7,
                                        gaussian_weights=False,use_sample_covariance=True))
    else:
        ssim=None
    return {'mae':mae,'mse':mse,'psnr':None if mse==0 else -10*math.log10(mse),
        'PSNR_zero_MSE_policy':'null denotes mathematically infinite PSNR; no finite fabricated value',
        'ssim':ssim,'SSIM_policy':'skimage structural_similarity uniform7x7,sample covariance,data_range1,RGB channel mean; only all-observed case',
        'signed_luminance_bias':float((observed@Y).mean(dtype=np.float64)),
        'absolute_luminance_bias':float(abs((observed@Y).mean(dtype=np.float64))),
        'mean_RGB_offset':offset.tolist(),'mean_centered_residual_mae':float(np.abs(centered).mean(dtype=np.float64)),
        'mean_centered_residual_mse':centered_mse,'global_RGB_offset_energy':energy,
        'global_RGB_offset_energy_share':None if mse==0 else energy/mse,
        'MSE_decomposition_closure_residual':closure,
        'adjacent_RGB_residual_gradient_mae':None if nx+ny==0 else (sum_x+sum_y)/(3*(nx+ny)),
        'adjacent_RGB_residual_gradient_x_mae':None if nx==0 else sum_x/(3*nx),
        'adjacent_RGB_residual_gradient_y_mae':None if ny==0 else sum_y/(3*ny),
        'adjacent_x_pairs':nx,'adjacent_y_pairs':ny,'observed_pixels':int(support.sum()),
        'offset_diagnostics_are_not_acceptance_or_identity_metrics':True}


def file_bytes(path, expected_sha=None, expected_size=None, cap=8*1024**2):
    path=Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT),'Evidence outside workspace')
    for part in (path,*path.parents):
        if part==ROOT.parent:break
        info=part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,
                'No linked/reparse evidence')
    before=path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap and
            (expected_size is None or before.st_size==expected_size),'Ordinary bounded exact file required')
    data=path.read_bytes();after=path.stat()
    require((before.st_ino,before.st_mtime_ns,before.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size),
            'Evidence changed across read')
    require(expected_sha is None or digest(data)==expected_sha,'Evidence SHA differs')
    return data


def canonical_relative(name):
    require(isinstance(name,str) and name and ':' not in name and '\\' not in name,'Canonical relative path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix()==name and all(x not in ('','.','..') for x in p.parts),
            'Canonical relative path')
    return name


def rgb_png(data):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format=='PNG' and image.mode=='RGB' and image.size==(256,256) and
                getattr(image,'n_frames',1)==1,'Exact single-frame RGB256 PNG required')
        return np.asarray(image).copy()


def raw_rgb(data):
    require(len(data)<=3*256*256*4+256,'Bounded RAW required')
    stream=io.BytesIO(data); value=np.load(stream,allow_pickle=False)
    require(not stream.read(1),'Trailing RAW bytes')
    require(value.dtype==np.float32 and value.shape==(256,256,3) and value.flags.c_contiguous,
            'Exact contiguous FP32 RGB256 RAW required')
    normalized_rgb(value)
    return value


def scalar_mean_metrics(rows):
    keys=('mae','mse','psnr','ssim','signed_luminance_bias','absolute_luminance_bias',
          'mean_centered_residual_mae','mean_centered_residual_mse','global_RGB_offset_energy',
          'global_RGB_offset_energy_share','adjacent_RGB_residual_gradient_mae')
    return {key:{'mean':None if not any(r[key] is not None for r in rows) else
                float(np.mean([r[key] for r in rows if r[key] is not None],dtype=np.float64)),
                'finite_cases':sum(r[key] is not None for r in rows),'null_cases':sum(r[key] is None for r in rows)}
            for key in keys}


def source_stratum(reference_id):
    return 'appearance_flag_va_ffhq_08025' if reference_id==FLAGGED_REFERENCE else 'other_seven_photographs'


# Saved-output worker schema is reviewed independently before this module gains
# its bounded actual-artifact audit entry point. No inference entry point exists.


MODEL_BINDING_SHA = '5a926fc23c58916374d4ad69bf68fc4066bc469c440d920a2046b09a2761a994'
CRITERIA_SHA = 'fd0ae5aa624f74775a65474957f42cf3cc69e0838febac4f1feb9d063340f903'
METRIC_CONTRACT_SHA = 'ff9082ede323e64e80fdc3ca1ce5882ae495d7f135879f0c9b69208338283098'
HELPER_FREEZE_SHA = 'cf3dcc85817ef1562aee721d1dfa3534d2d26722c9a78bbd8c92692f080d9d49'
REVIEWED_CAPS = {'forward_calls':216,'wall_seconds':3600,'process_RSS_bytes':3*1024**3,
    'minimum_available_RAM_bytes':768*1024**2,'minimum_disk_free_bytes':1024**3,
    'output_files':660,'output_payload_bytes':256*1024**2,'journal_records':700,
    'journal_bytes':2*1024**2,'watchdog_period_seconds':1}
MODELS = {
    'original0': ('original','1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7',
                  '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd'),
    'parent50': ('parent','53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc',
                 '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b'),
    'candidate50': ('r2','d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896',
                    'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f')}


def check_floor(raw,pixels):
    require(np.array_equal(pixels,np.floor(raw*np.float32(255)).astype(np.uint8)),
            'Saved PNG differs from exact FP32 RAW floor conversion')


def stream_model_pin(pin,arm):
    path=(ROOT/pin['path']).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT),'Model outside workspace')
    for part in (path,*path.parents):
        if part==ROOT.parent:break
        info=part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,'Linked model path')
    before=path.stat()
    allowed=3 if arm=='original0' else 1
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==allowed and
            before.st_size==pin['bytes'] and before.st_size<512*1024**2,'Exact model stat/size/link required')
    if arm=='original0':
        require(path==ROOT/'outputs/codeformer_restoration_pretrained_v1/codeformer.pth',
                'Only the exact untouched benchmark permits three hardlinks')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    after=path.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_nlink)==
            (after.st_ino,after.st_size,after.st_mtime_ns,after.st_nlink) and h.hexdigest()==pin['sha256']==MODELS[arm][1],
            'Model bytes changed or lineage differs')
    return {'path':pin['path'],'sha256':h.hexdigest(),'bytes':before.st_size,'hardlink_count':allowed,
            'loaded_or_executed':False}


def png_payload(image):
    stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG');return stream.getvalue()


def comparison_sheets(folder,data,cases,input_bank,target_bank,png_bank):
    """Write exact256 diagnostic panels; model cells stay blank for insufficient inputs."""
    by_id={r['id']:r for r in cases}
    references=[r['reference_id'] for r in data['references']]
    profiles=[r['id'] for r in data['profiles']]
    headers=('Observed / resize control','Clean synthetic target','Untouched CodeFormer','Retained full-coarse parent','Our trained R2 - experimental')
    panels=[];sheets=[]
    destination=folder/'sheets';destination.mkdir()
    for page,profile in enumerate(profiles,1):
        sheet=Image.new('RGB',(1280,2356),'white');draw=ImageDraw.Draw(sheet)
        draw.text((4,2),profile+' | fixed paired synthetic comparison | no extra enhancement | no model qualification',fill='black')
        for col,title in enumerate(headers):draw.text((col*256+3,21),title,fill='black')
        page_cells=[]
        for row,rid in enumerate(references):
            key=rid+'__'+profile;case=by_id[key];y=52+row*288
            note=rid+' | '+case['split']+' | '+case['input_review']['decision']
            if rid==FLAGGED_REFERENCE:note+=' | SOURCE APPEARANCE FLAG: no ordinary uncovered-feature claim'
            draw.text((4,y),note,fill='black')
            tiles=[input_bank[key],target_bank[rid]]
            tiles += [png_bank.get((arm,key)) for arm in ARMS]
            for col,tile in enumerate(tiles):
                x,py=col*256,y+26
                if tile is None:
                    require(case['input_review']['decision']=='insufficient','Blank cell outside prospective skip')
                    draw.rectangle((x,py,x+255,py+255),fill=(235,235,235),outline=(180,180,180))
                    draw.text((x+10,py+95),'NOT RUN: insufficient input',fill='black')
                    draw.text((x+10,py+113),'Request a clearer crop',fill='black')
                    pixel_sha=None
                else:
                    require(tile.shape==(256,256,3) and tile.dtype==np.uint8,'Exact256 cell required')
                    sheet.paste(Image.fromarray(tile),(x,py));pixel_sha=digest(tile.tobytes())
                page_cells.append({'id':key,'reference_id':rid,'split':case['split'],'profile':profile,
                    'decision':case['input_review']['decision'],'column':headers[col],'x':x,'y':py,'width':256,'height':256,
                    'pixel_sha256':pixel_sha,'blank_no_model_result':tile is None})
        array=np.asarray(sheet)
        for cell in page_cells:
            if cell['pixel_sha256'] is not None:
                tile=array[cell['y']:cell['y']+256,cell['x']:cell['x']+256]
                require(digest(tile.tobytes())==cell['pixel_sha256'],'Sheet exact-cell preservation differs')
        path=destination/('%02d_%s.png'%(page,profile))
        payload=png_payload(array)
        with path.open('xb') as stream:stream.write(payload)
        with Image.open(io.BytesIO(payload)) as saved:
            readback=np.asarray(saved)
            for cell in page_cells:
                if cell['pixel_sha256'] is not None:
                    require(digest(readback[cell['y']:cell['y']+256,cell['x']:cell['x']+256].tobytes())==cell['pixel_sha256'],
                            'Encoded sheet readback cell differs')
        pin={'path':str(path.relative_to(folder)),'sha256':digest(payload),'bytes':len(payload),'width':1280,'height':2356,
             'profile':profile,'rows':8,'columns':5,'cases':8}
        sheets.append(pin);panels.extend(dict(c,sheet=pin['path']) for c in page_cells)
    require(len(sheets)==12 and len(panels)==480 and sum(c['blank_no_model_result'] for c in panels)==72,
            '12 pages/480 cells/72 blank model cells required')
    return {'complete':True,'sheets':sheets,'cells':panels,'input_cases':96,'inferred_cases':72,'insufficient_cases':24,
        'exact256_pixel_panels':408,'blank_model_panels':72,'RAW_metric_output_displayed':False,
        'downloaded_PNG_pixels_displayed_exact':True,'no_display_enhancement_or_resampling':True,
        'root_visual_review':'PENDING; auditor creates sheets but does not establish usefulness'}


def strata_summary(metric_rows):
    groups=defaultdict(list)
    for row in metric_rows:
        dims={key:row[key] for key in ('source','split','profile','decision','appearance_stratum')}
        labels=[('all_inferred',{}),('appearance',{'appearance_stratum':dims['appearance_stratum']}),
            ('source',{'source':dims['source']}),('split',{'split':dims['split']}),
            ('profile',{'profile':dims['profile']}),('decision',{'decision':dims['decision']}),
            ('reference',{'reference_id':row['reference_id']}),
            ('source_split_profile', {k:dims[k] for k in ('source','split','profile')}),('full_cross',dims)]
        for label,criteria in labels:
            groups[(label,json.dumps(criteria,sort_keys=True))].append(row)
    summaries=[]
    for (label,criteria),rows in sorted(groups.items()):
        metrics={}
        for view in ('raw','png'):
            metrics[view]={arm:scalar_mean_metrics([r['metrics'][view][arm] for r in rows])
                           for arm in ('resize_control',*ARMS)}
        comparisons={}
        for baseline in ('resize_control','original0','parent50'):
            comparisons[baseline]={}
            for view in ('raw','png'):
                comparisons[baseline][view]={}
                for key,higher in [('mae',False),('ssim',True),('mean_centered_residual_mae',False),
                                   ('adjacent_RGB_residual_gradient_mae',False)]:
                    values=[(r['metrics'][view]['candidate50'][key],r['metrics'][view][baseline][key]) for r in rows]
                    require(all(a is not None and b is not None for a,b in values),'Expected fully observed metric values')
                    comparisons[baseline][view][key]={'candidate_lower':sum(a<b for a,b in values),
                        'candidate_equal':sum(a==b for a,b in values),'candidate_higher':sum(a>b for a,b in values),
                        'preferred_direction':'higher' if higher else 'lower',
                        'mean_candidate_minus_baseline':float(np.mean([a-b for a,b in values],dtype=np.float64))}
        summaries.append({'stratum_type':label,'criteria':json.loads(criteria),'cases':len(rows),
            'means_are_unweighted_per_case_not_population_estimates':True,'metrics':metrics,
            'candidate_comparisons_diagnostic_not_gates':comparisons})
    return {'complete':True,'strata':summaries,'pooled72cases_include_separate63core_and9flagged':True,
        'flagged_reference':FLAGGED_REFERENCE,'flagged_reference_not_ordinary_uncovered_feature_validation':True,
        'offset_energy_share_means':'Mean of per-case RGB offset energy fractions; not aggregate causal/structure explained fraction',
        'PSNR_means':'Finite per-case means; exact-zero/infinite values are separately null and counted',
        'paired_synthetic_only_no_native_CCTV_or_FINAL':True,'quality_gates_changed':False,'model_qualified':False}


def audit(folder,manifest_sha,worker_sha,out):
    import time
    require(not any(name in sys.modules for name in ('torch','tensorflow','jax')),'No neural framework in auditor')
    folder=Path(folder).resolve(strict=True);out=Path(out).absolute()
    require(folder.is_relative_to(ROOT/'outputs') and out.parent.resolve(strict=True).is_relative_to(ROOT/'outputs')
            and not out.exists(),'New exclusive output namespace required')
    started=time.monotonic()
    prep=ROOT/'outputs/codeformer_filter_comparison_v1_preparation'
    contract_data=file_bytes(prep/'metric_contract_v1.json',METRIC_CONTRACT_SHA)
    contract=unique_json(contract_data)
    frozen_helper=file_bytes(prep/'metric_helper_source_at_freeze_v1.py',HELPER_FREEZE_SHA)
    require(Path(__file__).read_bytes().startswith(frozen_helper),'Frozen metric helper bytes changed')
    manifest_data=file_bytes(folder/'manifest.json',manifest_sha,cap=8*1024**2)
    doc=unique_json(manifest_data)
    require(doc['complete'] is True and doc['format']=='codeformer-filter-frozen-comparison-v1' and
            doc['data_manifest_sha256']==DATA_SHA and doc['root_input_review_sha256']==REVIEW_SHA and
            doc['input_review_criteria_sha256']==CRITERIA_SHA and doc['model_binding_sha256']==MODEL_BINDING_SHA and
            doc['source_sha256']==worker_sha,'Completed frozen exact comparison manifest required')
    worker_source=file_bytes(ROOT/'scripts/run_codeformer_filter_comparison_v1.py',worker_sha)
    files=doc['files'];require(len(files)+1<=660,'Finite658 comparison file namespace required')
    observed_total=0
    for name,pin in files.items():
        canonical_relative(name);require(pin['path']==name and set(pin)=={'path','sha256','bytes'},'Exact pin schema')
        file_bytes(folder/name,pin['sha256'],pin['bytes']);observed_total+=pin['bytes']
    require(observed_total+len(manifest_data)<=256*1024**2,'Finite comparison payload cap')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual==set(files)|{'manifest.json'},'Comparison output namespace not closed')

    def get(pin):
        require(pin==files[pin['path']],'Output pin not exactly manifest bound')
        return file_bytes(folder/pin['path'],pin['sha256'],pin['bytes'])

    data=unique_json(get(files['data_manifest.json']))
    review=unique_json(get(files['input_review.json']))
    criteria=unique_json(get(files['input_review_criteria.json']))
    bindings=unique_json(get(files['model_binding.json']))
    for name,fingerprint in [('data_manifest.json',DATA_SHA),('input_review.json',REVIEW_SHA),
                              ('input_review_criteria.json',CRITERIA_SHA),('model_binding.json',MODEL_BINDING_SHA)]:
        require(files[name]['sha256']==fingerprint,'Prospective copied evidence differs')
    require(data['complete'] and review['complete'] and bindings['complete'] and
            criteria['existing_quality_gates_changed'] is False,'Prospective source evidence complete')
    data_root=ROOT/'outputs/codeformer_filter_pairs_v1_data'
    require(digest(file_bytes(data_root/'manifest.json',DATA_SHA))==DATA_SHA,'Original data manifest differs')
    for name,pin in data['files'].items():
        canonical_relative(name);file_bytes(data_root/name,pin['sha256'],pin['bytes'])
    cases=doc['cases'];data_cases={c['id']:c for c in data['cases']}
    labels={r['reference_id']+'__'+r['profile']:r for r in review['cases']}
    require(len(cases)==len(data_cases)==len(labels)==96 and len({c['id'] for c in cases})==96,'Exact96 unique retained cases')
    require(Counter(c['input_review']['decision'] for c in cases)=={'readable':28,'borderline':44,'insufficient':24},
            'Prospective input decisions changed')
    bound_models={}
    for arm,(binding_arm,_,state_sha) in MODELS.items():
        m=next(x for x in bindings['models'] if x['arm']==binding_arm)
        require(m['canonical515_tensor_sha256']==state_sha and m['tensor_count']==515 and m['all515finite_float32_CPU'],
                'Model binding canonical state/schema')
        bound_models[arm]=m
    model_file_proofs={arm:stream_model_pin(m['file'],arm) for arm,m in bound_models.items()}
    arch={pin['path']:pin['sha256'] for pin in bindings['architecture_sources']}
    for pin in bindings['architecture_sources']:file_bytes(ROOT/pin['path'],pin['sha256'],pin['bytes'])
    source_contract=unique_json(get(files['source_contract.json']))
    require(source_contract['source_sha256']==worker_sha and source_contract['caps']==doc['caps'] and
            doc['caps']==REVIEWED_CAPS and 0<=doc['seconds']<=REVIEWED_CAPS['wall_seconds'] and
            source_contract['inputs']==96 and source_contract['skip_insufficient']==24 and source_contract['scoring']=='none' and
            source_contract['architecture']==arch and source_contract['fidelity']==1. and source_contract['adain'] is True and
            source_contract['detach_16'] is True and source_contract['training_updates']==source_contract['backward_calls']==0,
            'Source execution contract differs')
    proofs=unique_json(get(files['model_state_proofs.json']))
    require(proofs['complete'] and proofs['models']==doc['model_state_proofs'] and len(proofs['models'])==3 and
            {r['arm'] for r in proofs['models']}==set(ARMS),'Before/after state proofs')
    for row in proofs['models']:
        arm=row['arm'];require(row['file']==bound_models[arm]['file'] and row['forwards']==72 and row['exact515_state_unchanged'] and
            row['before_state_sha256']==row['after_state_sha256']==MODELS[arm][2],'Frozen canonical model state changed')
    journal=[unique_json(line) for line in get(files['journal.jsonl']).splitlines()]
    require(len(journal)<=700 and len(get(files['journal.jsonl']))<=2*1024**2,'Journal bounds')
    reservations=[r for r in journal if r['event']=='forward_reserved']
    returns=[r for r in journal if r['event']=='forward_returned']
    commits=[r for r in journal if r['event']=='output_committed']
    selected=[c for c in cases if c['input_review']['decision']!='insufficient']
    require(len(selected)==72 and Counter(c['split'] for c in selected)=={'TRAIN':36,'DEV':36} and
            sum(c['reference_id']==FLAGGED_REFERENCE for c in selected)==9,'72 source/split/appearance census')
    expected_order=[(binding_arm,c['id']) for arm,(binding_arm,_,_) in MODELS.items() for c in selected]
    require(len(reservations)==len(returns)==len(commits)==216,'216 reservations/returns/commits required')
    for i,(reserve,returned,commit) in enumerate(zip(reservations,returns,commits),1):
        require(reserve['call']==returned['call']==commit['call']==i and
                (reserve['arm'],reserve['id'])==(returned['arm'],returned['id'])==(commit['arm'],commit['id'])==expected_order[i-1],
                'Forward journal order/identity differs')
        require(journal.index(reserve)<journal.index(returned)<journal.index(commit),'Reservation before return before commit')
    require(journal[-1]['event']=='complete' and journal[-1]['progress']['reserved_forwards']==
            journal[-1]['progress']['returned_forwards']==journal[-1]['progress']['committed_outputs']==216 and
            journal[-1]['progress']['in_flight'] is None,'Complete finite journal closure')
    skipped=unique_json(get(files['skipped_cases.json']))
    require(skipped['complete'] and skipped['refused_from_prospective_input_review'] is True and
            skipped['automatic_threshold_validated'] is False,'Prospective skip policy')
    require({c['id'] for c in skipped['cases']}=={c['id'] for c in cases if not c['inference_planned']},'Exact24 skipped case list')
    expected_files={'data_manifest.json','input_review.json','input_review_criteria.json','model_binding.json',
                    'source_contract.json','skipped_cases.json','runtime.json','model_state_proofs.json','journal.jsonl'}
    png_bank={};input_bank={};target_bank={};metric_rows=[];case_rows=[];worst_closure=0.
    commit_by_call={r['call']:r for r in commits}
    runtime=unique_json(get(files['runtime.json']))
    require(runtime['device']=='cpu' and runtime['torch_threads']==4 and runtime['torch_interop_threads']==1,'Frozen CPU runtime settings')
    resource_samples=[runtime['initial_resources']]+[r['resources'] for r in journal if 'resources' in r]
    require(len(resource_samples)==221,'Expected221 source-bound resource samples')
    for sample in resource_samples:
        require(set(sample)=={'process_RSS_bytes','available_RAM_bytes','disk_free_bytes'} and
                all(type(v) is int and v>0 for v in sample.values()) and
                sample['process_RSS_bytes']<=REVIEWED_CAPS['process_RSS_bytes'] and
                sample['available_RAM_bytes']>=REVIEWED_CAPS['minimum_available_RAM_bytes'] and
                sample['disk_free_bytes']>=REVIEWED_CAPS['minimum_disk_free_bytes'],
                'Recorded resource sample violates prospective cap')
    for case in cases:
        key=case['id'];original_case=data_cases[key]
        require(all(case[k]==value for k,value in original_case.items() if k!='input_review'),
                'Original data case membership/admission changed')
        label=labels[key]
        require(case['input_review']==label and case['appearance_review_required']==(case['reference_id']==FLAGGED_REFERENCE) and
                case['inference_planned']==(label['decision']!='insufficient'),'Pre-output review/appearance flags changed')
        pair=unique_json(file_bytes(data_root/case['pair_record']['path'],case['pair_record']['sha256'],case['pair_record']['bytes']))
        input8=rgb_png(file_bytes(data_root/case['input']['path'],case['input']['sha256'],case['input']['bytes']))
        target_pin=pair['targets']['clean256'];target8=rgb_png(file_bytes(data_root/target_pin['path'],target_pin['sha256'],target_pin['bytes']))
        input_bank[key]=input8;target_bank[case['reference_id']]=target8
        row={'id':key,'reference_id':case['reference_id'],'source':'FFHQ approved aligned photography',
             'split':case['split'],'profile':case['profile'],'decision':label['decision'],
             'appearance_stratum':source_stratum(case['reference_id']),'input_review_reason':label['reason'],
             'input_pin':case['input'],'target256_pin':target_pin,'optimization_permitted_now':False,'metrics':None}
        if not case['inference_planned']:
            require(case['outputs']=={} and not any('/'+key+'/' in name for name in files),'Insufficient input has model output')
            row['skip_reason']='Prospective input-only insufficient decision; no model inference/results. Not an automatic threshold.'
            case_rows.append(row);continue
        require(set(case['outputs'])==set(ARMS),'All three frozen model outputs required')
        support=np.ones((256,256),bool);clean=target8.astype(np.float64)/255.
        metrics={view:{} for view in ('raw','png')}
        metrics['raw']['resize_control']=residual_metrics(input8.astype(np.float32)/np.float32(255),clean,support)
        metrics['png']['resize_control']=residual_metrics(input8.astype(np.float64)/255.,clean,support)
        for arm in ARMS:
            pins=case['outputs'][arm];prefix='outputs/'+arm+'/'+key+'/'
            require(set(pins)=={'raw','png','evidence'} and pins['raw']['path']==prefix+'raw.npy' and
                    pins['png']['path']==prefix+'native.png' and pins['evidence']['path']==prefix+'evidence.json','Exact per-output namespace')
            expected_files.update(p['path'] for p in pins.values())
            raw=raw_rgb(get(pins['raw']));png8=rgb_png(get(pins['png']));check_floor(raw,png8)
            e=unique_json(get(pins['evidence']))
            require(e['format']=='codeformer-filter-frozen-inference-output-v1' and e['id']==key and
                    e['reference_id']==case['reference_id'] and e['split']==case['split'] and e['profile']==case['profile'] and
                    e['input']==case['input'] and e['pair_record']==case['pair_record'] and
                    e['source1024']==pair['source1024'] and e['target_bindings_not_scored_by_worker']==pair['targets'] and
                    e['input_review']==label and e['appearance_review_required']==case['appearance_review_required'] and
                    e['arm']==arm and e['model_binding']==bound_models[arm] and e['architecture_sources']==arch and
                    e['canonical515_state_sha256']==MODELS[arm][2],'Evidence exact source/review/model lineage')
            require(e['raw']==pins['raw'] and e['png']==pins['png'] and e['RAW_array_sha256']==digest(raw.tobytes()) and
                    e['PNG_RGB8_sha256']==digest(png8.tobytes()) and e['RAW_shape']==[256,256,3] and
                    e['RAW_dtype']=='float32' and e['RAW_min']==float(raw.min()) and e['RAW_max']==float(raw.max()),'Exact RAW/PNG fingerprint evidence')
            require(e['observed_support']=='all256x256 pixels' and e['geometry']=={'crop':None,'alignment':None,'warp':None} and
                    e['fidelity']==1. and e['adain'] is e['detach_16'] is True and e['display_processing']=='none' and
                    e['eval_all_modules'] is True and e['parameters_require_grad'] is False and e['no_grad'] is True and
                    e['device']=='cpu' and e['internal_resolution']==512 and e['output_resolution']==256 and
                    e['optimizer_updates']==e['backward_calls']==0 and e['paired_clean_metrics'] is None and
                    e['model_qualified'] is False and e['optimization_permitted_now'] is False,'Frozen inference/geometry/claim policy')
            call=e['forward_reservation'];require(commit_by_call[call]['evidence']==pins['evidence'] and
                    commit_by_call[call]['id']==key and commit_by_call[call]['arm']==MODELS[arm][0],'Journal commit binding differs')
            metrics['raw'][arm]=residual_metrics(raw,clean,support)
            metrics['png'][arm]=residual_metrics(png8.astype(np.float64)/255.,clean,support)
            png_bank[(arm,key)]=png8
        row['metrics']=metrics;metric_rows.append(row);case_rows.append(row)
        worst_closure=max(worst_closure,max(abs(m['MSE_decomposition_closure_residual'])
                         for view in metrics.values() for m in view.values()))
    require(set(files)==expected_files and len(metric_rows)==72 and len(case_rows)==96,'Output exact namespace/census')
    require(doc['actual_forwards']==doc['committed_outputs']==216 and doc['input_insufficient_cases_skipped']==24 and
            doc['actual_training_updates']==doc['actual_optimizer_steps']==doc['actual_backward_calls']==0 and
            doc['paired_metrics_computed'] is False and doc['quality_gates_changed'] is False and doc['model_qualified'] is False,
            'Authoritative finite inference outcome differs')
    require(not any(name in sys.modules for name in ('torch','tensorflow','jax')),'Auditor imported neural framework')
    out.mkdir()
    sheets=comparison_sheets(out,data,cases,input_bank,target_bank,png_bank)
    summary=strata_summary(metric_rows)
    metrics_doc={'complete':True,'cases':case_rows,'inferred':72,'skipped_null_metrics':24,'metric_contract_sha256':METRIC_CONTRACT_SHA,
                 'appearance_flag_not_removed':True,'native_CCTV_or_FINAL_cases':0,'quality_gates_changed':False,'model_qualified':False}
    for name,value in [('per_case_metrics_v1.json',metrics_doc),('stratified_metrics_v1.json',summary),('visual_manifest_v1.json',sheets)]:
        with (out/name).open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    receipt={'format':'codeformer-filter-comparison-independent-saved-output-audit-v1','complete':True,'pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'worker_manifest':{'path':str(folder/'manifest.json'),'sha256':manifest_sha,'bytes':len(manifest_data)},
        'worker_sha256':worker_sha,'auditor_sha256':digest(Path(__file__).read_bytes()),'metric_contract_sha256':METRIC_CONTRACT_SHA,
        'unchanged_frozen_metric_helper_prefix_sha256':HELPER_FREEZE_SHA,'binding_receipt_sha256':MODEL_BINDING_SHA,
        'model_files_freshly_hashed':model_file_proofs,'counts':{'inputs':96,'inferred':72,'skipped':24,'arms':3,'RAW':216,'PNG':216,
            'core_photographs_cases':63,'appearance_flag_cases':9,'inferred_TRAIN':36,'inferred_DEV':36,'control_metric_cases':72,
            'comparison_sheets':12,'exact256_panels':408,'blank_model_panels':72},
        'all216_RAW_geometry_finite_range_and_exact_floor_PNG':True,'all216_lineage_geometry_and_no_enhancement_evidence':True,
        'all216_journal_reservation_return_commit_joins':True,'before_after_frozen515_state_readouts_exact':True,
        'state_readouts_source_bound_not_independent_live_replay':True,'closed_worker_output_files':len(files)+1,
        'resource_envelope':{'samples':len(resource_samples),
            'maximum_recorded_RSS_bytes':max(r['process_RSS_bytes'] for r in resource_samples),
            'minimum_recorded_available_RAM_bytes':min(r['available_RAM_bytes'] for r in resource_samples),
            'minimum_recorded_disk_free_bytes':min(r['disk_free_bytes'] for r in resource_samples),
            'sampled_readouts_not_independent_continuous_watchdog_profile':True},
        'closed_worker_output_bytes':observed_total+len(manifest_data),'all96_paired_source_and_split_membership_retained':True,
        'maximum_absolute_MSE_decomposition_roundoff':worst_closure,'MSE_identity_not_a_quality_gate':True,
        'basic_resize_control_does_not_claim_model_floor_conversion':True,'spatial_and_centered_metrics_not_identity_proof':True,
        'offset_energy_share_is_global_RGB_not_brightness_causal_explanation':True,'new_acceptance_thresholds':False,
        'useful_brightening_remains_desired':True,'root_all12_visual_review_pending':True,'native_CCTV_cases':0,'FINAL_cases':0,
        'new_NN_autograd_optimizer_VM_calls':0,'worker_code_imported':False,'model_checkpoint_loaded':False,
        'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started}
    with (out/'audit_v1.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'complete':True,'pass':True,'RAW':216,'PNG':216,'skipped':24,'sheets':12,'receipt':str(out/'audit_v1.json')}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest-sha',required=True)
    parser.add_argument('--worker-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.manifest_sha,args.worker_sha,args.out)),flush=True)
