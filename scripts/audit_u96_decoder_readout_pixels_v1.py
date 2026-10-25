"""Independent bounded saved-pixel arithmetic; no producer imports or model calls.

Only explicit main() reads an archive. No Torch, pickle, model, VM or subprocess.
The archive is streamed in canonical order; one output directory plus the120
reference arrays are retained. Actual quality comparisons use sealed thresholds
exactly. Floating differences in scalar readback are reported separately.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STAGE_NAME = 'cctv_codeformer_u96_decoder_readout_v1_vm'
RUN = 'outputs/u96_decoder_readout_v1'
PROTOCOL_SHA = 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'
ARCHIVE_SHA = '73c8f4bcca5814cb845058cbc7d553ad3ea1f29e97967449cad66ac848fed2c4'
UPDATES = (0, 1, 4, 16, 32, 64)
ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
VIEWS = ('raw', 'native_png')


def need(ok, text):
    if not ok:
        raise ValueError(text)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sequential_sum(values):
    result = 0.0
    for item in values:
        result += float(item)
    return result


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('Nonfinite JSON constant')
    value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=bad)
    def walk(x):
        if type(x) is float:
            need(math.isfinite(x), 'Finite JSON floats required')
        elif isinstance(x, dict):
            for child in x.values(): walk(child)
        elif isinstance(x, list):
            for child in x: walk(child)
    walk(value)
    return value


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        need(image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 PNG required')
        return np.array(image, copy=True)


def array(data, name):
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    need(version in ((1, 0), (2, 0)), 'NPY1/2 only')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=4096)
    shapes = {'native512.npy': (1, 3, 512, 512), 'logits.npy': (1, 256, 1024), 'LQ.npy': (1, 256, 16, 16)}
    expected = shapes.get(PurePosixPath(name).name, (256, 256, 3))
    need(shape == expected and dtype == np.dtype('float32') and not fortran and not dtype.hasobject,
         'Expected bounded C-order FP32 schema: ' + name)
    need(stream.tell() + math.prod(shape)*4 == len(data), 'NPY extent mismatch')
    result = np.frombuffer(data, dtype=np.float32, offset=stream.tell()).reshape(shape)
    need(np.isfinite(result).all(), 'Nonfinite saved array')
    return result


def masks():
    inside = np.zeros((256, 256), dtype=bool)
    inside[64:192, 64:192] = True
    return {'whole': np.ones_like(inside), 'central128': inside, 'outside': ~inside}


MASKS = masks()


def floor_png(value):
    need(value.shape == (256, 256, 3) and value.dtype == np.float32 and
         np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1, 'Clipped finite RAW required')
    return np.floor(value*np.float32(255)).astype(np.uint8)


def regional(value, target, mask):
    residual = value.astype(np.float64) - target.astype(np.float64)
    horizontal = np.abs(residual[:, 1:] - residual[:, :-1])
    vertical = np.abs(residual[1:] - residual[:-1])
    good_h = mask[:, 1:] & mask[:, :-1]
    good_v = mask[1:] & mask[:-1]
    need(mask.any() and good_h.any() and good_v.any(), 'Nonempty regional pixels and edges required')
    if mask.all():
        edge = .5*(horizontal.mean() + vertical.mean())
    else:
        edge = (horizontal[good_h].sum() + vertical[good_v].sum()) / (3*int(good_h.sum() + good_v.sum()))
    return {'mae': float(np.abs(residual[mask]).mean()), 'adjacent_rgb_error': float(edge)}


def luminance_blocks(value):
    gray = value.astype(np.float64) @ np.array((.2126, .7152, .0722), dtype=np.float64)
    return gray.reshape(16, 16, 16, 16).mean(axis=(1, 3))


def objective(value, target, role, precision=np.float64):
    """Every role uses the same-photograph clean target, including clear/dark."""
    need(role in ROLES, 'Declared objective role required')
    residual = value.astype(precision) - target.astype(precision)
    def terms(mask):
        h = mask[:, 1:] & mask[:, :-1]
        v = mask[1:] & mask[:-1]
        image = np.abs(residual[mask]).mean(dtype=precision)
        edges = np.concatenate((np.abs(residual[:, 1:] - residual[:, :-1])[h].reshape(-1),
                                np.abs(residual[1:] - residual[:-1])[v].reshape(-1)))
        return image, edges.mean(dtype=precision)
    if role == ROLES[0]:
        di, ei = terms(MASKS['central128']); do, eo = terms(MASKS['outside'])
        values = dict(image_inside=float(di), image_outside=float(do), adjacent_inside=float(ei), adjacent_outside=float(eo))
        total = precision(.5)*precision(di+do) + precision(.125)*precision(ei+eo)
    else:
        image, adjacent = terms(MASKS['whole'])
        values = dict(image=float(image), adjacent=float(adjacent))
        total = image + precision(.25)*adjacent
    return float(total), values


def legacy_checks(bank, protocol):
    checks = []
    for role in ('clear', 'dark', 'filter'):
        rows = sorted((r for r in protocol['guards'] if r['role'] == role), key=lambda x:x['id'])
        need(len(rows) == 8, 'Eight retained guards per role')
        for view in VIEWS:
            for metric in (('mae', 'adjacent_rgb_error') if role == 'filter' else ('behavior_distance',)):
                def read(row, result):
                    return result['behavior_distance'][view] if metric == 'behavior_distance' else result['metrics'][view][metric]
                before = [read(r, r) for r in rows]
                now = [read(r, bank[r['id']]) for r in rows]
                values = [x-y for x,y in zip(now,before)] if role == 'filter' else now
                limits = [.03]*8 if role == 'filter' else [max(.03, x) for x in before]
                mean_limit = .01 if role == 'filter' else max(.01, sequential_sum(before)/8)
                mean = sequential_sum(values)/8
                bad = [r['id'] for r, x, limit in zip(rows, values, limits) if x > limit]
                checks.append({'role':role,'view':view,'metric':metric,'mean':mean,'mean_limit':mean_limit,
                    'measurements':dict(zip((r['id'] for r in rows),values)),
                    'failed_ids':bad,'pass':not bad and mean <= mean_limit})
    return checks


def monitor_checks(bank, baseline, protocol, update):
    checks = []
    for role in ROLES:
        ids = sorted(r['id'] for r in protocol['cases'] if r['cohort'] == 'monitor' and r['role'] == role)
        need(len(ids) == 8, 'Eight monitor photographs per role')
        for view in VIEWS:
            for region in (('whole','central128','outside') if role == ROLES[0] else ('whole',)):
                for metric in ('mae','adjacent_rgb_error'):
                    initial = [baseline[k]['regional_metrics'][view][region][metric] for k in ids]
                    current = [bank[k]['regional_metrics'][view][region][metric] for k in ids]
                    delta = [b-a for a,b in zip(initial,current)]
                    mean = sequential_sum(delta)/8
                    checks.append({'role':role,'view':view,'metric':region+'/'+metric,'mean':mean,'worst':max(delta),
                        'pass':mean <= .01 and max(delta) <= .03})
            if role in ('clear','lowlight_linear_0p35'):
                metric = 'U96_behavior_distance' if role == 'clear' else 'U96_illumination_distance'
                values = [bank[k][metric][view] for k in ids]
                checks.append({'role':role,'view':view,'metric':metric,'mean':sequential_sum(values)/8,'worst':max(values),
                    'pass':sequential_sum(values)/8 <= .01 and max(values) <= .03})
            if update == 64 and role in ROLES[:2]:
                region = 'central128' if role == ROLES[0] else 'whole'
                a = sequential_sum(baseline[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                b = sequential_sum(bank[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                need(a > 0, 'Positive initial monitored adjacent error')
                gain = (a-b)/a
                checks.append({'role':role,'view':view,'metric':'new_1_percent_TRAIN_blur_endpoint',
                    'initial_mean':a,'current_mean':b,'gain_fraction':gain,'pass':gain >= .01})
            if update == 64 and role == ROLES[3]:
                for metric in ('mae','adjacent_rgb_error'):
                    a = sequential_sum(baseline[k]['regional_metrics'][view]['whole'][metric] for k in ids)/8
                    b = sequential_sum(bank[k]['regional_metrics'][view]['whole'][metric] for k in ids)/8
                    checks.append({'role':role,'view':view,'metric':'endpoint_dark_'+metric,
                        'initial_mean':a,'current_mean':b,'pass':b <= a})
    return checks


def trend_check(bank, baseline, protocol, update):
    groups = []
    if update == 16:
        for role, region in ((ROLES[1],'whole'),(ROLES[0],'central128')):
            ids = sorted(r['id'] for r in protocol['cases'] if r['cohort'] == 'monitor' and r['role'] == role)
            for view in VIEWS:
                a = sequential_sum(baseline[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                b = sequential_sum(bank[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                groups.append({'role':role,'view':view,'initial_mean':a,'current_mean':b,'pass':b<a})
    passed = update != 16 or any(all(r['pass'] for r in groups if r['role'] == role) for role in ROLES[:2])
    return {'evaluated':update==16,'pass':passed,'comparisons':groups}


def tensor_metadata(value):
    return {'shape':list(value.shape),'dtype':'float32','bytes':int(value.nbytes),
        'RGB_or_tensor_bytes_sha256':sha(value.tobytes()),'minimum':float(value.min()),'maximum':float(value.max()),
        'negative_zero_count':int(np.count_nonzero((value==0)&np.signbit(value)))}


def audit(archive, stage, out):
    started = time.monotonic()
    need(not out.exists() and out.is_relative_to(ROOT/'outputs'), 'Fresh immutable workspace output namespace required')
    need(stat.S_ISREG(archive.lstat().st_mode) and not archive.is_symlink(), 'Ordinary archive required')
    h = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2), b''): h.update(chunk)
    need(h.hexdigest() == ARCHIVE_SHA, 'Exact supplied archive SHA mismatch')
    pdata = (stage/'protocol.json').read_bytes()
    need(sha(pdata) == PROTOCOL_SHA, 'Exact sealed protocol SHA mismatch')
    protocol = strict_json(pdata)
    by_id = {r['id']:r for r in protocol['cases']+protocol['guards']}
    counts, comparisons = Counter(), Counter()
    failures, scalar_differences, objective_rows, summaries = [], [], [], []
    references, evaluated, records, documents, metadata, source_cache = {}, {}, {}, {}, {}, {}
    max_float_difference = 0.0
    def check(ok, label, detail=None):
        comparisons[label] += 1
        if not ok: failures.append({'check':label,'detail':detail})
    def same(actual, saved, label):
        nonlocal max_float_difference
        if isinstance(actual,dict):
            check(isinstance(saved,dict) and set(actual)==set(saved),label+'/keys')
            for k,v in actual.items():
                if isinstance(saved,dict) and k in saved: same(v,saved[k],label+'/'+k)
        elif isinstance(actual,list):
            check(isinstance(saved,list) and len(actual)==len(saved),label+'/length')
            if isinstance(saved,list):
                for i,(a,b) in enumerate(zip(actual,saved)):same(a,b,label+'/'+str(i))
        elif type(actual) is float and type(saved) in (float,int):
            difference=abs(actual-saved); max_float_difference=max(max_float_difference,difference)
            comparisons[label] += 1
            if difference:
                scalar_differences.append({'field':label,'absolute_difference':difference,'actual':actual,'saved':saved})
                if difference > 1e-15: failures.append({'check':label,'difference':difference})
        else: check(actual==saved,label,{'actual':actual,'saved':saved} if actual!=saved else None)
    def source(binding, image=False):
        key=binding['path']
        if key not in source_cache:
            file=stage.joinpath(*PurePosixPath(key).parts)
            need(file.is_file() and not file.is_symlink(), 'Pinned ordinary stage source required')
            data=file.read_bytes()
            check(len(data)==binding['bytes'] and sha(data)==binding['sha256'],'source_pin/'+key)
            source_cache[key]=rgb(data) if image else array(data,key)
        return source_cache[key]
    # Recalculate the older, retained thresholds from their actual saved RGBs;
    # no original/R2 model is replayed and these teachers are never clean GT.
    for guard in protocol['guards']:
        oldraw=source(guard['outputs']['raw'])
        oldpng=source(guard['outputs']['png'],True)
        check(np.array_equal(floor_png(oldraw),oldpng),'legacy_saved_RAW_floor_PNG_exact',guard['id'])
        oldq=oldpng.astype(np.float32)/np.float32(255)
        if guard.get('target') is not None:
            clean=source(guard['target'],True).astype(np.float32)/np.float32(255)
            for view,value in (('raw',oldraw),('native_png',oldq)):
                calculated=regional(value,clean,MASKS['whole'])
                for metric,number in calculated.items():same(number,guard['metrics'][view][metric],'legacy_initial_'+metric)
        if guard.get('behavior_teacher'):
            teacher=guard['behavior_teacher']
            tr=source(teacher['raw']);tq=source(teacher['png'],True).astype(np.float32)/np.float32(255)
            for view,a,b in (('raw',oldraw,tr),('native_png',oldq,tq)):
                number=float(np.abs(a.astype(np.float64)-b.astype(np.float64)).mean())
                same(number,guard['behavior_distance'][view],'legacy_initial_behavior_distance')
    def flush(group):
        if not group:return
        record_path=next((k for k in group if k.endswith('/record.json')),None)
        if record_path is None:return
        record=strict_json(group[record_path]); cid=record['id']; spec=by_id[cid]
        records[record_path]=record; kind=record['kind']; counts[kind]+=1
        stem=record_path.rsplit('/',1)[0]
        need(kind in ('reference','snapshot','training'), 'Declared output kind')
        for binding in (record['outputs']['raw'],record['outputs']['png']):
            same({'path':binding['path'],**metadata[binding['path']]},binding,'output_pin')
        raw=array(group[record['outputs']['raw']['path']],stem+'/raw.npy')
        png=rgb(group[record['outputs']['png']['path']])
        check(np.array_equal(floor_png(raw),png),'RAW_floor_PNG_exact',record_path)
        q=png.astype(np.float32)/np.float32(255)
        for field in ('role','reference_id','input','target'):
            same(record.get(field),spec.get(field),'record_'+field)
        target=None if spec.get('target') is None else source(spec['target'],True).astype(np.float32)/np.float32(255)
        result={'id':cid,'role':spec['role'],'regional_metrics':{},'metrics':{}}
        for view,value in (('raw',raw),('native_png',q)):
            result['regional_metrics'][view]=None if target is None else {name:regional(value,target,mask) for name,mask in MASKS.items()}
            result['metrics'][view]={'mae':None,'adjacent_rgb_error':None} if target is None else result['regional_metrics'][view]['whole']
        same(result['regional_metrics'],record['regional_metrics'],'regional_metrics')
        same(result['metrics'],record['metrics'],'whole_metrics')
        if spec.get('behavior_teacher'):
            teacher=spec['behavior_teacher']; tr=source(teacher['raw']); tq=source(teacher['png'],True).astype(np.float32)/np.float32(255)
            result['behavior_distance']={v:float(np.abs(a.astype(np.float64)-b.astype(np.float64)).mean()) for v,a,b in (('raw',raw,tr),('native_png',q,tq))}
            same(result['behavior_distance'],record['behavior_distance'],'behavior_distance')
        if kind=='reference':
            references[cid]={'raw':raw.copy(),'png':png.copy(),'png_sha':sha(group[record['outputs']['png']['path']]),'stock':record['stock'],'result':result,'record_path':record_path}
        else:
            need(cid in references,'Reference must precede candidate in sorted archive')
            base=references[cid]; br=base['raw']; bq=base['png'].astype(np.float32)/np.float32(255)
            result['U96_behavior_distance']={v:float(np.abs(a.astype(np.float64)-b.astype(np.float64)).mean()) for v,a,b in (('raw',raw,br),('native_png',q,bq))}
            result['U96_illumination_distance']={v:float(np.abs(luminance_blocks(a)-luminance_blocks(b)).mean()) for v,a,b in (('raw',raw,br),('native_png',q,bq))}
            for name in ('U96_behavior_distance','U96_illumination_distance'):same(result[name],record[name],name)
            for label in ('logits','LQ'):same(record['stock'][label],base['stock'][label],'frozen_upstream_'+label)
            initial=(kind=='snapshot' and record['update']==0) or (kind=='training' and record['update']==1)
            if initial:
                check(np.array_equal(raw,br),'initial_RAW_parity',record_path)
                check(sha(group[record['outputs']['png']['path']])==base['png_sha'],'initial_PNG_byte_parity',record_path)
                same(record['stock'],base['stock'],'initial_stock_tuple_parity')
        if 'unclipped' in record['outputs']:
            binding=record['outputs']['unclipped'];un=array(group[binding['path']],binding['path'])
            nchw=np.ascontiguousarray(un.transpose(2,0,1)[None])
            same(tensor_metadata(nchw),record['clipping']['unclipped256'],'unclipped_tensor_metadata')
            same({'path':binding['path'],**metadata[binding['path']]},binding,'unclipped_file_pin')
            counts['unclipped']+=1
        if record['outputs'].get('stock'):
            stock={}
            for label,binding in record['outputs']['stock'].items():
                value=array(group[binding['path']],binding['path']);stock[label]=tensor_metadata(value)
                same(stock[label],record['stock'][label],'stock_actual_'+label)
                same({'path':binding['path'],**metadata[binding['path']]},binding,'stock_file_pin')
                if label=='native512':
                    z=(value.astype(np.float32)+np.float32(1))/np.float32(2)
                    # Exact512→256 bilinear samples average each2×2 cell. FP32 reduction order can differ by one ulp.
                    resized=(z[:,:,0::2,0::2]+z[:,:,0::2,1::2]+z[:,:,1::2,0::2]+z[:,:,1::2,1::2])*np.float32(.25)
                    clipped=np.clip(z,0,1)
                    delivered=(clipped[:,:,0::2,0::2]+clipped[:,:,0::2,1::2]+clipped[:,:,1::2,0::2]+clipped[:,:,1::2,1::2])*np.float32(.25)
                    check(float(np.abs(delivered[0].transpose(1,2,0)-raw).max()) <= np.finfo(np.float32).eps,'actual_native_delivery_FP32_rounding')
                    if 'unclipped' in record['outputs']:
                        check(float(np.abs(resized[0].transpose(1,2,0)-un).max()) <= np.finfo(np.float32).eps,'actual_native_unclipped_FP32_rounding')
                    same(int((value < -1).sum()),record['clipping']['native_below_minus1'],'native_clipping_below')
                    same(int((value > 1).sum()),record['clipping']['native_above1'],'native_clipping_above')
            if kind=='reference':references[cid]['actual_stock']=stock
            elif kind=='training':same(stock,references[cid]['actual_stock'],'first4_actual_stock_array_parity')
            counts['actual_stock_tuples']+=1
        if kind=='training':
            objective_path=stem+'/objective.json'; saved=strict_json(group[objective_path])
            need(target is not None and spec['cohort']=='optimize' and spec['optimization_permitted'] is True,'Clean admitted TRAIN objective only')
            same(saved['photographic_optimization_target'],spec['target'],'objective_clean_target_pin')
            same(saved['comparison_reference_not_optimization_target']['path'],references[cid]['record_path'],'objective_reference_join')
            same(saved['output_record']['sha256'],metadata[record_path]['sha256'],'objective_record_join')
            value64,terms64=objective(raw,target,spec['role'],np.float64)
            value32,terms32=objective(raw,target,spec['role'],np.float32)
            differences={key:abs(saved['terms'][key]-value) for key,value in terms64.items()}
            need(set(terms64)==set(saved['terms']),'Exact objective component keys')
            check(max(differences.values()) <= 5e-7 and abs(value64-saved['objective']) <= 5e-7,
                  'objective_arithmetic_readback_only_tolerance',objective_path)
            same(saved['divided_objective'],float(np.float32(saved['objective'])/np.float32(4)),'objective_divided_four')
            u=record['update'];same(saved['LR'],5e-5*min(1.,u/4),'objective_learning_rate')
            expected_ref=protocol['learning']['optimization_reference_order'][(u-1+(record['slot']-1)*4)%16]
            check(cid==expected_ref+'__'+ROLES[record['slot']-1],'sampler_exact',record_path)
            objective_rows.append({'update':u,'slot':record['slot'],'id':cid,'role':spec['role'],'reported':saved['objective'],
                'recomputed_float64':value64,'recomputed_float32':value32,'terms_float64':terms64,'terms_float32':terms32,
                'reported_terms':saved['terms'],'absolute_float64_difference':abs(value64-saved['objective']),
                'component_absolute_float64_differences':differences,'objective_tolerance_is_not_a_quality_threshold':True})
        evaluated[record_path]=result
        summaries.append({'record_path':record_path,'kind':kind,'update':record['update'],'slot':record['slot'],**result})
        if sum(counts[k] for k in ('reference','snapshot','training'))%100==0:
            print(json.dumps({'independent_saved_records':len(summaries),'of':712}),flush=True)
    group={}; current=None; seen=set(); last=''; total_payload=0
    pattern=re.compile(re.escape(RUN)+r'/(reference/[^/]+|snapshots/u\d{4}/[^/]+|training/u\d{4}/slot[1-4])/[^/]+$')
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=member.name; parts=PurePosixPath(name).parts
            need(member.isfile() and not member.issparse() and not member.issym() and not member.islnk() and
                 not any(k.startswith('GNU.sparse') for k in member.pax_headers),'Ordinary nonsparse member required')
            need(parts and parts[0]==STAGE_NAME and len(parts)>1 and '\\' not in name and ':' not in name and
                 '..' not in parts and PurePosixPath(name).as_posix()==name,'Canonical exact stage member required')
            rel=PurePosixPath(*parts[1:]).as_posix()
            need(rel not in seen and rel>last,'Unique lexicographically sorted members required');seen.add(rel);last=rel
            need(0<=member.size<=16*1024**2,'Bounded individual member required')
            total_payload+=member.size;need(total_payload<=2300*1024**2,'Bounded total payload required')
            payload=tar.extractfile(member).read(member.size+1);need(len(payload)==member.size,'Exact member extent')
            metadata[rel]={'bytes':len(payload),'sha256':sha(payload)}
            if pattern.fullmatch(rel):
                stem=rel.rsplit('/',1)[0]
                if current is not None and current!=stem:flush(group);group={}
                current=stem;group[rel]=payload
            else:
                if current is not None:flush(group);group={};current=None
                if rel.endswith('/bank_and_gates.json') or rel in ('protocol.json',RUN+'/status.json',RUN+'/failure.json'):
                    documents[rel]=strict_json(payload)
            if rel in protocol['files'] and not rel.startswith('weights/'):
                check(metadata[rel]=={k:protocol['files'][rel][k] for k in ('bytes','sha256')},'returned_static_pin/'+rel)
    flush(group)
    same(documents['protocol.json'],protocol,'returned_protocol')
    check(counts['reference']==120 and counts['snapshot']==336 and counts['training']==256,'complete_712_record_census',dict(counts))
    check(len(objective_rows)==256,'complete_256_objective_census')
    baseline={k:v['result'] for k,v in references.items()}
    banks=[]
    for update in UPDATES:
        selected={r['id']:evaluated[f'{RUN}/snapshots/u{update:04d}/{r["id"]}/record.json']
            for r in protocol['cases']+protocol['guards'] if r.get('cohort')=='monitor' or r['id'] in {x['id'] for x in protocol['guards']}}
        check(len(selected)==56,'complete_56_snapshot_bank',update)
        old=legacy_checks(selected,protocol); monitor=monitor_checks(selected,baseline,protocol,update)
        trend=trend_check(selected,baseline,protocol,update)
        saved=documents[f'{RUN}/snapshots/u{update:04d}/bank_and_gates.json']
        for row in old:
            s=saved['legacy']['groups'][row['role']]['views'][row['view']][row['metric']]
            for key in ('pass','mean','mean_limit','failed_ids','measurements'):same(row[key],s[key],'legacy_gate_'+key)
        for row in monitor:
            s=saved['monitor']['groups'][row['role']]['views'][row['view']][row['metric']]
            for key in ('pass','mean','initial_mean','current_mean','gain_fraction'):
                if key in row:
                    reported=(sequential_sum(s['values'].values())/8 if key=='mean' and key not in s else s[key])
                    same(row[key],reported,'monitor_gate_'+key)
        same(trend['pass'],saved['early_trend']['pass'],'trend_gate_exact_pass')
        same(trend['evaluated'],saved['early_trend']['evaluated'],'trend_gate_exact_evaluated')
        for row in trend['comparisons']:
            s=saved['early_trend']['groups'][row['role']]['views'][row['view']]
            for key in ('pass','initial_mean','current_mean'):same(row[key],s[key],'trend_gate_'+key)
        legacy_pass=all(row['pass'] for row in old);monitor_pass=all(row['pass'] for row in monitor)
        same(legacy_pass,saved['legacy']['pass'],'legacy_gate_aggregate')
        same(monitor_pass,saved['monitor']['pass'],'monitor_gate_aggregate')
        banks.append({'update':update,'legacy_pass':legacy_pass,'monitor_pass':monitor_pass,'early_trend':trend,
            'legacy':old,'monitor':monitor,'all_gates_pass':legacy_pass and monitor_pass and trend['pass']})
    terminal=banks[-1]
    failed_checks=[row for row in terminal['legacy']+terminal['monitor'] if not row['pass']]
    endpoints=[row for row in terminal['monitor'] if row['metric']=='new_1_percent_TRAIN_blur_endpoint']
    dark=[row for row in terminal['monitor'] if row['metric'].startswith('endpoint_dark_')]
    report={'complete':True,'audit_type':'INDEPENDENT_ACTUAL_SAVED_PIXEL_OBJECTIVE_AND_GATE_READBACK',
        'UTC':datetime.now(timezone.utc).isoformat(),'archive_sha256':ARCHIVE_SHA,'protocol_sha256':PROTOCOL_SHA,
        'archive_bytes':archive.stat().st_size,'member_count_observed':len(seen),'payload_bytes_observed':total_payload,
        'counts':dict(counts),'checks':sum(comparisons.values()),'checks_by_family':dict(comparisons),
        'saved_numeric_integrity_pass':not failures,'integrity_failure_count':len(failures),'integrity_failures':failures,
        'nonzero_scalar_difference_count':len(scalar_differences),'maximum_scalar_difference':max_float_difference,
        'scalar_integrity_readback_bound':1e-15,'quality_gate_thresholds_exact_no_epsilon':True,
        'objective_scalar_readback_bound':5e-7,'objective_bound_is_not_quality_threshold':True,
        'objective_maximum_absolute_float64_difference':max(r['absolute_float64_difference'] for r in objective_rows),
        'initial_complete_U96_parity_records':60,'actual_native_stock_tuple_records':counts['actual_stock_tuples'],
        'all_snapshots':[{k:b[k] for k in ('update','legacy_pass','monitor_pass','early_trend','all_gates_pass')} for b in banks],
        'terminal_blur_endpoints':endpoints,'terminal_dark_nonregression':dark,'terminal_failed_checks':failed_checks,
        'pilot_quality_gates_pass':terminal['all_gates_pass'],'reported_status':documents[RUN+'/failure.json']['status'],
        'model_qualified':False,'native_CCTV_or_FINAL_qualification':False,'root_visual_review_pending':True,
        'Torch_neural_gradient_optimizer_training_VM_calls':0,'no_RAW_extraction_retained':True,
        'memory_policy':'One streamed output directory plus120 reference RAW/PNG arrays and pinned stage sources; no all-output RAW cache.',
        'elapsed_seconds':time.monotonic()-started}
    out.mkdir(parents=True)
    def write(name,value):
        path=out/name
        with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        data=path.read_bytes();return {'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':sha(data)}
    rows_pin=write('recomputed_saved_pixel_metrics_v1.json',{'complete':True,'records':summaries})
    objective_pin=write('recomputed_clean_objectives_v1.json',{'complete':True,'rows':objective_rows})
    gates_pin=write('recomputed_gates_v1.json',{'complete':True,'snapshots':banks})
    differences_pin=write('scalar_arithmetic_differences_v1.json',{'complete':True,'rows':scalar_differences})
    report['artifacts']={'metrics':rows_pin,'objectives':objective_pin,'gates':gates_pin,'differences':differences_pin}
    pin=write('saved_pixel_audit_v1.json',report)
    print(json.dumps({'complete':True,'saved_numeric_integrity_pass':not failures,'checks':report['checks'],
        'terminal_quality_pass':report['pilot_quality_gates_pass'],'failed_terminal_checks':len(failed_checks),'audit':pin}),flush=True)
    return 0 if not failures else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'outputs/cctv-codeformer-u96-decoder-readout-v1-results.tar.gz')
    parser.add_argument('--stage',type=Path,default=ROOT/'outputs'/STAGE_NAME)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/codeformer_u96_decoder_readout_v1_return_pixel_audit')
    args=parser.parse_args();return audit(args.archive.resolve(),args.stage.resolve(),args.output.resolve())


if __name__=='__main__':
    raise SystemExit(main())
