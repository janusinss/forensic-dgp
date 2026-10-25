"""Read saved B64 images, without Torch, pickle, training, or a VM call.

The sole archive pass validates every member, retains PNG/JSON evidence, and
loads bounded float32 arrays in memory. Gate arithmetic is independent code.
Objective readback reports float64 mathematical reconstruction and float32
reduction differences; it adds no tolerance to any quality decision.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/cctv_codeformer_u96_postfusion_learning_v1_vm'
ARCHIVE = ROOT / 'outputs/cctv-codeformer-u96-postfusion-learning-v1-results.tar.gz'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_return_audit'
RUN = 'outputs/u96_postfusion_learning_v1'
ARCHIVE_SHA = '5f99df00c2eff645d3bc957606766753d3308aaa4e0e22da57ac28edeb8d858a'
PROTOCOL_SHA = '11e3286c64d9de282e46d3fa95ae5c792a402c72a681fba4040d816ae5944903'
PAYLOAD_CAP = 1536 * 1024**2
OVERHEAD_CAP = 16 * 1024**2
SNAPSHOTS = [0, 1, 4, 16, 32, 64]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filepin(path):
    data = Path(path).read_bytes()
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': digest(data)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return filepin(path)


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def png(data):
    with Image.open(io.BytesIO(data)) as im:
        need(im.mode == 'RGB' and im.size == (256, 256), 'Expected exact RGB256 PNG')
        return np.asarray(im).copy()


def array(data):
    value = np.load(io.BytesIO(data), allow_pickle=False)
    need(isinstance(value, np.ndarray) and value.dtype == np.float32 and np.isfinite(value).all(),
         'Expected finite saved float32 array without pickle')
    return value


def regions():
    whole = np.ones((256, 256), dtype=bool)
    inside = np.zeros_like(whole)
    inside[64:192, 64:192] = True
    return {'whole': whole, 'central128': inside, 'outside': ~inside}


REGIONS = regions()
EDGE_COUNTS = {}
for _name, _region in REGIONS.items():
    _h = _region[:, :-1] & _region[:, 1:]
    _v = _region[:-1, :] & _region[1:, :]
    EDGE_COUNTS[_name] = {'pixels': int(_region.sum()), 'horizontal_edges': int(_h.sum()),
                          'vertical_edges': int(_v.sum()), 'adjacent_RGB_components': int((_h.sum()+_v.sum())*3)}


def metrics(value, target):
    residual = value.astype(np.float64) - target.astype(np.float64)
    dh = np.abs(np.diff(residual, axis=1))
    dv = np.abs(np.diff(residual, axis=0))
    answer = {}
    for name, region in REGIONS.items():
        h = region[:, :-1] & region[:, 1:]
        v = region[:-1, :] & region[1:, :]
        if name == 'whole':
            adjacent = float(0.5 * (dh.mean() + dv.mean()))
        else:
            adjacent = float((dh[h].sum() + dv[v].sum()) / (int(h.sum()+v.sum())*3))
        answer[name] = {'mae': float(np.abs(residual[region]).mean()), 'adjacent_rgb_error': adjacent}
    return answer


def illum(value):
    y = value.astype(np.float64) @ np.asarray([.2126, .7152, .0722], dtype=np.float64)
    return y.reshape(16, 16, 16, 16).mean(axis=(1, 3))


def l1_adj(value, target, region, precision):
    residual = value.astype(precision) - target.astype(precision)
    h = region[:, :-1] & region[:, 1:]
    v = region[:-1, :] & region[1:, :]
    image = float(np.abs(residual[region]).mean(dtype=precision))
    horizontal = np.abs(np.diff(residual, axis=1))[h].sum(dtype=precision)
    vertical = np.abs(np.diff(residual, axis=0))[v].sum(dtype=precision)
    adjacent = float(precision(horizontal+vertical) / precision(int(h.sum()+v.sum())*3))
    return image, adjacent


def objective(raw, target, baseline, role, precision):
    whole, inside, outside = REGIONS['whole'], REGIONS['central128'], REGIONS['outside']
    if role == 'local_gaussian_1p2':
        di, ei = l1_adj(raw, target, inside, precision)
        do, eo = l1_adj(raw, target, outside, precision)
        terms = {'inside_image_L1': di, 'outside_image_L1': do,
                 'inside_adjacent_L1': ei, 'outside_adjacent_L1': eo}
        value = precision(precision(.5)*precision(precision(di)+precision(do)) +
                          precision(.125)*precision(precision(ei)+precision(eo)))
    elif role == 'clear':
        di, ei = l1_adj(raw, baseline, whole, precision)
        terms = {'U96_image_anchor_L1': di, 'U96_adjacent_anchor_L1': ei}
        value = precision(2)*precision(precision(di)+precision(.25)*precision(ei))
    else:
        di, ei = l1_adj(raw, target, whole, precision)
        terms = {'target_image_L1': di, 'target_adjacent_L1': ei}
        value = precision(precision(di)+precision(.25)*precision(ei))
        if role == 'lowlight_linear_0p35':
            weights = np.asarray([.2126, .7152, .0722], dtype=precision)
            def pooled(x):
                y = (x.astype(precision)*weights).sum(axis=2, dtype=precision)
                return y.reshape(16, 16, 16, 16).mean(axis=(1, 3), dtype=precision)
            anchor = float(np.abs(pooled(raw)-pooled(baseline)).mean(dtype=precision))
            terms['U96_block16_luminance_anchor_L1'] = anchor
            value = precision(value + precision(2)*precision(anchor))
    return float(value), terms


def legacy_gate(bank, protocol):
    old = {r['id']: r for r in protocol['guards']}
    answer = {'pass': True, 'thresholds_reset_to_U96': False, 'groups': {}}
    for role in ('clear', 'dark', 'filter'):
        ids = sorted(k for k, r in bank.items() if r['role'] == role)
        need(len(ids) == 8, 'Eight same-role legacy guards required')
        group = {'pass': True, 'views': {}}
        for view in ('raw', 'native_png'):
            group['views'][view] = {}
            for field in (('mae', 'adjacent_rgb_error') if role == 'filter' else ('behavior_distance',)):
                def measure(rows, key):
                    return float(rows[key]['behavior_distance'][view] if field == 'behavior_distance'
                                 else rows[key]['metrics'][view][field])
                initial = {k: measure(old, k) for k in ids}
                current = {k: measure(bank, k) for k in ids}
                if role == 'filter':
                    measurements = {k: current[k]-initial[k] for k in ids}
                    limits, meanlimit = dict.fromkeys(ids, .03), .01
                else:
                    measurements = current
                    limits = {k: max(.03, initial[k]) for k in ids}
                    meanlimit = max(.01, sum(initial.values())/8)
                mean = sum(measurements.values())/8
                bad = [k for k in ids if measurements[k] > limits[k]]
                passed = not bad and mean <= meanlimit
                group['views'][view][field] = {'pass': passed, 'initial': initial, 'current': current,
                    'measurements': measurements, 'case_limits': limits, 'mean_limit': meanlimit,
                    'mean': mean, 'failed_ids': bad}
                group['pass'] = group['pass'] and passed
        answer['groups'][role] = group
        answer['pass'] = answer['pass'] and group['pass']
    return answer


def monitor_gate(bank, baseline, protocol, update):
    answer = {'pass': True, 'TRAIN_monitor_only_not_DEV': True, 'endpoint_evaluated': update == 64, 'groups': {}}
    for role in protocol['learning']['slot_roles']:
        ids = sorted(k for k, r in bank.items() if r['role'] == role)
        need(len(ids) == 8, 'Eight same-role monitor inputs required')
        group = {'pass': True, 'views': {}}
        for view in ('raw', 'native_png'):
            checks = {}
            for region in (('whole', 'central128', 'outside') if role == 'local_gaussian_1p2' else ('whole',)):
                for field in ('mae', 'adjacent_rgb_error'):
                    initial = {k: baseline[k]['regional_metrics'][view][region][field] for k in ids}
                    current = {k: bank[k]['regional_metrics'][view][region][field] for k in ids}
                    delta = {k: current[k]-initial[k] for k in ids}
                    mean = sum(delta.values())/8
                    checks[region+'/'+field] = {'pass': max(delta.values()) <= .03 and mean <= .01,
                        'initial': initial, 'current': current, 'deltas': delta, 'mean': mean,
                        'mean_limit': .01, 'worst_limit': .03}
            if role in ('clear', 'lowlight_linear_0p35'):
                field = 'U96_behavior_distance' if role == 'clear' else 'U96_illumination_distance'
                distances = {k: bank[k][field][view] for k in ids}
                checks[field] = {'pass': sum(distances.values())/8 <= .01 and max(distances.values()) <= .03,
                                'values': distances, 'mean_limit': .01, 'worst_limit': .03}
            if update == 64 and role in ('local_gaussian_1p2', 'global_gaussian_1p2'):
                region = 'central128' if role == 'local_gaussian_1p2' else 'whole'
                initial = sum(baseline[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                current = sum(bank[k]['regional_metrics'][view][region]['adjacent_rgb_error'] for k in ids)/8
                gain = (initial-current)/initial
                checks['new_1_percent_TRAIN_blur_endpoint'] = {'pass': gain >= .01, 'gain_fraction': gain,
                    'initial_mean': initial, 'current_mean': current, 'region': region,
                    'not_historical_DGP_feature_gate_or_DEV_qualification': True}
            if update == 64 and role == 'lowlight_linear_0p35':
                for field in ('mae', 'adjacent_rgb_error'):
                    initial = sum(baseline[k]['regional_metrics'][view]['whole'][field] for k in ids)/8
                    current = sum(bank[k]['regional_metrics'][view]['whole'][field] for k in ids)/8
                    checks['endpoint_dark_'+field] = {'pass': current <= initial,
                                                     'initial_mean': initial, 'current_mean': current}
            group['views'][view] = checks
            group['pass'] = group['pass'] and all(x['pass'] for x in checks.values())
        answer['groups'][role] = group
        answer['pass'] = answer['pass'] and group['pass']
    return answer


def main():
    started = time.monotonic()
    need(not OUT.exists(), 'Audit output is immutable and must not preexist')
    OUT.mkdir(parents=True)
    protocol_bytes = (STAGE/'protocol.json').read_bytes()
    need(digest(protocol_bytes) == PROTOCOL_SHA, 'Sealed protocol pin differs')
    protocol = json.loads(protocol_bytes)
    byid = {r['id']: r for r in protocol['cases']+protocol['guards']}
    need(len(byid) == 120, 'Expected96 new rolecases plus24 legacy guards')
    # Read the compressed file through a hash wrapper as tarfile streams it once.
    class HashReader:
        def __init__(self, stream): self.stream, self.hash, self.count = stream, hashlib.sha256(), 0
        def read(self, size=-1):
            value = self.stream.read(size); self.hash.update(value); self.count += len(value); return value
    metadata, docs, arrays, images, raw_bytes, stock_hashes = {}, {}, {}, {}, {}, {}
    payload = 0
    total = 0
    with ARCHIVE.open('rb') as compressed:
        reader = HashReader(compressed)
        with tarfile.open(fileobj=reader, mode='r|gz') as archive:
            for member in archive:
                name = member.name
                parts = PurePosixPath(name).parts
                need(not name.startswith('/') and '..' not in parts and '\\' not in name,
                     'Unsafe archive name: '+name)
                if parts and parts[0] == STAGE.name: parts = parts[1:]
                rel = PurePosixPath(*parts).as_posix()
                if member.isdir(): continue
                need(member.isfile() and rel not in metadata, 'Unsafe, linked, or duplicate archive member')
                need(member.size >= 0 and member.size <= 64*1024**2, 'Single archive member exceeds64MiB')
                payload += member.size
                need(payload <= PAYLOAD_CAP, 'Archive payload exceeds1536MiB')
                data = archive.extractfile(member).read(member.size+1)
                need(len(data) == member.size, 'Archive member length differs')
                metadata[rel] = {'path': rel, 'bytes': len(data), 'sha256': digest(data)}
                if rel.endswith('.json'):
                    docs[rel] = json.loads(data)
                    if rel.startswith(RUN+'/'):
                        dest = OUT/'returned_metadata'/rel[len(RUN)+1:]
                        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
                if rel.endswith('.png'):
                    if rel.startswith(RUN+'/') or rel.startswith('assets/'):
                        im = png(data); images[rel] = im
                        if rel.startswith(RUN+'/'):
                            dest = OUT/'generated_pngs'/rel[len(RUN)+1:]
                            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
                if rel.endswith('.npy') and (rel.startswith(RUN+'/') or rel.startswith('assets/')):
                    value = array(data)
                    stock_hashes[rel] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'bytes': value.nbytes, 'RGB_or_tensor_bytes_sha256': digest(value.tobytes()),
                        'minimum': float(value.min()), 'maximum': float(value.max()),
                        'negative_zero_count': int(np.count_nonzero((value == 0) & np.signbit(value)))}
                    if rel.endswith('/raw.npy') or rel.startswith('assets/'):
                        arrays[rel] = value
                    if rel.endswith('/raw.npy'):
                        need(value.shape == (256,256,3) and value.min() >= 0 and value.max() <= 1,
                             'Invalid deliveredRAW range or shape')
                        raw_bytes[rel] = digest(data)
                        if '/reference/' in rel:
                            dest = OUT/'baseline_raw'/Path(rel).parent.name/'raw.npy'
                            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
                total += 1
                if total % 1000 == 0:
                    print(json.dumps({'archive_members': total, 'payload_MiB': round(payload/1024**2,1)}), flush=True)
        # Consume any gzip/tar trailing bytes through the same compressed hash wrapper.
        while reader.read(1024**2): pass
    need(reader.count == 1166039684 and reader.hash.hexdigest() == ARCHIVE_SHA, 'Returned archive pin differs')
    need(metadata['protocol.json']['sha256'] == PROTOCOL_SHA, 'Returned protocol differs')
    for rel, entry in protocol['files'].items():
        if rel.startswith('weights/'): continue
        need(rel in metadata and all(metadata[rel][k] == entry[k] for k in ('bytes','sha256')),
             'Sealed static return pin differs: '+rel)
    write_json(OUT/'archive_index_v1.json', {'complete': True, 'archive_SHA256': ARCHIVE_SHA,
        'archive_bytes': reader.count, 'payload_bytes': payload, 'members': metadata,
        'no_extractall': True, 'pt_unpickled': False, 'single_archive_stream': True})
    print(json.dumps({'archive_stream_complete': True, 'members': total,
                      'payload_bytes': payload, 'raw_outputs': len(raw_bytes)}), flush=True)

    failures, comparisons = [], Counter()
    def compare(actual, saved, label):
        comparisons[label] += 1
        if actual != saved:
            failures.append({'check': label, 'actual': actual, 'saved': saved})
    # Verify raw/PNG pins before calculating; no archive receipts supply computed metrics.
    records = {key: value for key, value in docs.items() if key.startswith(RUN+'/') and key.endswith('/record.json')}
    need(len(records) == 712 and len(raw_bytes) == 712, 'Exactly712 returned output records andRAWs required')
    baseline = {}
    evaluated = {}
    pixel_rows = []
    objective_rows = []
    first_training_stock = []
    reference_record_paths = {}
    for key, record in records.items():
        if record['kind'] == 'reference': reference_record_paths[record['id']] = key
    need(len(reference_record_paths) == 120, 'Exactly120 completeU96 references required')

    def evaluate(key, record):
        cid = record['id']; row = byid[cid]
        raw = arrays[record['outputs']['raw']['path']]
        image = images[record['outputs']['png']['path']]
        delivery = np.floor(raw*np.float32(255)).astype(np.uint8)
        need(np.array_equal(image, delivery), 'RAW→floorPNG mismatch: '+key)
        for spec in record['outputs'].values():
            if not isinstance(spec, dict) or 'path' not in spec: continue
            need(all(metadata[spec['path']][k] == spec[k] for k in ('bytes','sha256')), 'Record output pin mismatch')
        compare(record['input'], row['input'], 'record_input_binding_exact')
        compare(record['target'], row.get('target'), 'record_target_binding_exact')
        compare(record['role'], row['role'], 'record_role_exact')
        q = image.astype(np.float32)/np.float32(255)
        target = None if row.get('target') is None else images[row['target']['path']].astype(np.float32)/np.float32(255)
        result = {'id': cid, 'role': row['role'], 'regional_metrics': {}, 'metrics': {}}
        for view, value in [('raw', raw), ('native_png', q)]:
            result['regional_metrics'][view] = None if target is None else metrics(value, target)
            result['metrics'][view] = ({'mae':None,'adjacent_rgb_error':None} if target is None
                                       else result['regional_metrics'][view]['whole'])
        compare(result['regional_metrics'], record['regional_metrics'], 'regional_metrics_exact_float64')
        compare(result['metrics'], record['metrics'], 'whole_metrics_exact_float64')
        if row.get('behavior_teacher'):
            teacher = row['behavior_teacher']
            traw = arrays[teacher['raw']['path']]
            tpng = images[teacher['png']['path']].astype(np.float32)/np.float32(255)
            result['behavior_distance'] = {
                'raw': float(np.abs(raw.astype(np.float64)-traw.astype(np.float64)).mean()),
                'native_png': float(np.abs(q.astype(np.float64)-tpng.astype(np.float64)).mean())}
            compare(result['behavior_distance'], record['behavior_distance'], 'legacy_teacher_distance_exact')
        if record['kind'] != 'reference':
            bref = records[reference_record_paths[cid]]
            braw = arrays[bref['outputs']['raw']['path']]
            bpng = images[bref['outputs']['png']['path']].astype(np.float32)/np.float32(255)
            result['U96_behavior_distance'] = {v: float(np.abs(x.astype(np.float64)-b.astype(np.float64)).mean())
                for v,x,b in [('raw',raw,braw),('native_png',q,bpng)]}
            result['U96_illumination_distance'] = {v: float(np.abs(illum(x)-illum(b)).mean())
                for v,x,b in [('raw',raw,braw),('native_png',q,bpng)]}
            compare(result['U96_behavior_distance'], record['U96_behavior_distance'], 'U96_behavior_distance_exact')
            compare(result['U96_illumination_distance'], record['U96_illumination_distance'], 'U96_illumination_distance_exact')
            compare(record['stock']['logits'], bref['stock']['logits'], 'frozen_logits_record_exact')
            compare(record['stock']['LQ'], bref['stock']['LQ'], 'frozen_LQ_record_exact')
        if 'unclipped' in record['outputs']:
            un = dict(stock_hashes[record['outputs']['unclipped']['path']])
            # Saved HWC bytes differ from the worker's NCHW tensor metadata ordering.
            compare(un['minimum'], record['clipping']['unclipped256']['minimum'], 'unclipped_minimum_exact')
            compare(un['maximum'], record['clipping']['unclipped256']['maximum'], 'unclipped_maximum_exact')
        item = {'record': key, 'id': cid, 'kind': record['kind'], 'update': record['update'],
            'slot': record['slot'], 'role': row['role'], 'cohort': row.get('cohort','legacy_guard'),
            'raw_file': metadata[record['outputs']['raw']['path']],
            'png_file': metadata[record['outputs']['png']['path']],
            'raw_RGB8_floor_mismatches': 0, 'raw_minimum': float(raw.min()), 'raw_maximum': float(raw.max()),
            'raw_zero_components': int(np.count_nonzero(raw==0)), 'raw_one_components': int(np.count_nonzero(raw==1)),
            'saved_PNG_RGB8_SHA256': digest(image.tobytes()), **result}
        if record['kind'] != 'reference':
            delta = raw.astype(np.float64)-braw.astype(np.float64)
            item['U96_raw_component_difference'] = {'minimum_signed':float(delta.min()),
                'maximum_signed':float(delta.max()), 'maximum_absolute':float(np.abs(delta).max()),
                'changed_components':int(np.count_nonzero(delta))}
            item['U96_PNG_changed_components'] = int(np.count_nonzero(image-images[bref['outputs']['png']['path']]))
        pixel_rows.append(item)
        return result

    for cid, key in reference_record_paths.items(): baseline[cid] = evaluate(key, records[key])
    for key, record in records.items():
        if record['kind'] != 'reference': evaluated[key] = evaluate(key, record)
    # Equality claims are independently read back from saved arrays and PNG bytes.
    zero_parity = []
    for cid in [r['id'] for r in protocol['cases'] if r['cohort']=='monitor']+[r['id'] for r in protocol['guards']]:
        key = f'{RUN}/snapshots/u0000/{cid}/record.json'; r=records[key]; b=records[reference_record_paths[cid]]
        rawsame = np.array_equal(arrays[r['outputs']['raw']['path']], arrays[b['outputs']['raw']['path']])
        pngsame = metadata[r['outputs']['png']['path']]['sha256']==metadata[b['outputs']['png']['path']]['sha256']
        compare(rawsame, True, 'initial_zerohead_RAW_parity'); compare(pngsame, True, 'initial_zerohead_PNGbyte_parity')
        zero_parity.append({'id':cid,'RAW_exact':rawsame,'PNG_bytes_exact':pngsame})
    for slot in range(1,5):
        key=f'{RUN}/training/u0001/slot{slot}/record.json';r=records[key];b=records[reference_record_paths[r['id']]]
        value={'id':r['id'],'slot':slot,'RAW_exact':np.array_equal(arrays[r['outputs']['raw']['path']],arrays[b['outputs']['raw']['path']]),
            'PNG_bytes_exact':metadata[r['outputs']['png']['path']]['sha256']==metadata[b['outputs']['png']['path']]['sha256'],'stock':{}}
        for label in ['native512','logits','LQ']:
            a=stock_hashes[r['outputs']['stock'][label]['path']];z=stock_hashes[b['outputs']['stock'][label]['path']]
            compare(a,z,'first4_actual_saved_stock_array_exact');compare(a,r['stock'][label],'training_stock_tensor_metadata_exact')
            compare(z,b['stock'][label],'reference_stock_tensor_metadata_exact');value['stock'][label]={'arrays_exact':a==z,'tensor':a}
        compare(value['RAW_exact'],True,'first4_RAW_parity');compare(value['PNG_bytes_exact'],True,'first4_PNGbyte_parity')
        first_training_stock.append(value)
    gates=[]
    for update in SNAPSHOTS:
        selected={r['id']:evaluated[f'{RUN}/snapshots/u{update:04d}/{r["id"]}/record.json']
                  for r in protocol['cases']+protocol['guards'] if r.get('cohort')=='monitor' or r['id'] in {g['id'] for g in protocol['guards']}}
        lg=legacy_gate({k:v for k,v in selected.items() if k in {g['id'] for g in protocol['guards']}},protocol)
        mg=monitor_gate({k:v for k,v in selected.items() if byid[k].get('cohort')=='monitor'},baseline,protocol,update)
        saved=docs[f'{RUN}/snapshots/u{update:04d}/bank_and_gates.json']
        compare(lg,saved['legacy'],'legacy_gate_all_fields_exact_no_epsilon')
        compare(mg,saved['monitor'],'monitor_gate_all_fields_exact_no_epsilon')
        gates.append({'update':update,'legacy':lg,'monitor':mg})
    endpoint=gates[-1]
    endpoint_gains={role:{view:endpoint['monitor']['groups'][role]['views'][view]['new_1_percent_TRAIN_blur_endpoint']
        for view in ['raw','native_png']} for role in ['global_gaussian_1p2','local_gaussian_1p2']}
    numeric_early={'complete':True,'pixel_receipts_exact':not failures,'archive':{'bytes':reader.count,'sha256':ARCHIVE_SHA},
        'protocol_sha256':PROTOCOL_SHA,'endpoint_legacy_pass':endpoint['legacy']['pass'],
        'endpoint_monitor_pass':endpoint['monitor']['pass'],'blur_endpoint':endpoint_gains,
        'snapshot_gate_results':[{'update':g['update'],'legacy_pass':g['legacy']['pass'],'monitor_pass':g['monitor']['pass']} for g in gates],
        'root_visual_review_pending':True,'model_qualified':False,'failures':failures}
    write_json(OUT/'early_endpoint_readback_v1.json',numeric_early)
    print(json.dumps({'endpoint':numeric_early['snapshot_gate_results'],'blur_endpoint':endpoint_gains,
                      'metric_readback_failures':len(failures)}),flush=True)
    # Every recorded objective is reconstructed from saved float32 delivery and teachers.
    for key,saved in docs.items():
        if not key.startswith(RUN+'/training/') or not key.endswith('/objective.json'): continue
        cid=saved['id'];row=byid[cid];r=records[key.replace('/objective.json','/record.json')]
        raw=arrays[r['outputs']['raw']['path']];b=records[reference_record_paths[cid]]
        braw=arrays[b['outputs']['raw']['path']]
        target=images[row['target']['path']].astype(np.float32)/np.float32(255)
        value64,terms64=objective(raw,target,braw,row['role'],np.float64)
        value32,terms32=objective(raw,target,braw,row['role'],np.float32)
        compare(set(saved['terms']),set(terms64),'objective_term_names_exact')
        compare(saved['divided_objective'],saved['objective']/4,'saved_objective_division_exact')
        expectedcid=next(r['id'] for r in protocol['cases'] if r['role']==protocol['learning']['slot_roles'][saved['slot']-1]
            and r['reference_id']==protocol['learning']['optimization_reference_order'][(saved['update']-1+protocol['learning']['sampler_offsets'][saved['slot']-1])%16])
        compare(cid,expectedcid,'objective_sampler_exact')
        objective_rows.append({'path':key,'id':cid,'update':saved['update'],'slot':saved['slot'],
            'saved_float32_CUDA_objective':saved['objective'],'independent_float64_objective':value64,
            'independent_numpy_float32_objective':value32,'saved_minus_float64':saved['objective']-value64,
            'saved_minus_numpy_float32':saved['objective']-value32,'saved_terms':saved['terms'],
            'independent_float64_terms':terms64,'independent_numpy_float32_terms':terms32,
            'maximum_term_absolute_difference_from_float64':max(abs(saved['terms'][k]-terms64[k]) for k in terms64),
            'no_quality_gate_tolerance_added':True})
    need(len(objective_rows)==256,'Exactly256 objectives required')
    objective_summary={'count':256,'maximum_absolute_saved_minus_float64':max(abs(r['saved_minus_float64']) for r in objective_rows),
        'maximum_absolute_saved_minus_numpy_float32':max(abs(r['saved_minus_numpy_float32']) for r in objective_rows),
        'maximum_absolute_term_saved_minus_float64':max(r['maximum_term_absolute_difference_from_float64'] for r in objective_rows),
        'note':'Saved CUDA float32 reductions and independent NumPy CPU float32/float64 arithmetic differ in reduction order. All differences are reported, with no objective or quality pass tolerance invented.'}
    # Exact saved256 visual inventory: each generatedPNG appears in one and only one cell.
    try: font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
    except OSError: font=ImageFont.load_default()
    visual_sheets=[];generated_cells=[]
    def visual(rows, columns, filename, group):
        width=16+len(columns)*264; height=68+len(rows)*300
        sheet=Image.new('RGB',(width,height),(21,23,27));draw=ImageDraw.Draw(sheet)
        draw.text((8,4),f'B64 saved256 pixels | {group} | TRAIN only | no quality/identity qualification',font=font,fill='white')
        for col,title in enumerate(columns):draw.text((8+col*264,30),title,font=font,fill=(220,224,230))
        cells=[]
        for rr,(cid,specs) in enumerate(rows):
            y=68+rr*300;draw.text((8,y-20),cid,font=font,fill=(225,227,232))
            for cc,(source,label,isnew) in enumerate(specs):
                x=8+cc*264;im=images[source];sheet.paste(Image.fromarray(im),(x,y));bounds=[x,y,x+256,y+256]
                draw.text((x,y+258),label[:34],font=font,fill=(205,209,215))
                cell={'id':cid,'column':columns[cc],'source':metadata[source],'bounds_xyxy':bounds,
                      'RGB8_sha256':digest(im.tobytes()),'generated_return_PNG':isnew}
                cells.append(cell)
                if isnew:generated_cells.append({'sheet':filename,**cell})
        dest=OUT/'visuals'/filename;dest.parent.mkdir(parents=True,exist_ok=True);sheet.save(dest)
        # Ensure no overlay, resizing, resampling, or crop drift entered a review cell.
        with Image.open(dest) as readback:
            for cell in cells:need(digest(np.asarray(readback.crop(cell['bounds_xyxy'])).tobytes())==cell['RGB8_sha256'],'Saved visual cell differs')
        visual_sheets.append({'file':filepin(dest),'group':group,'columns':columns,'size':list(sheet.size),'cells':cells})
    readoutids=[r['id'] for r in protocol['cases'] if r['cohort']=='monitor']+[r['id'] for r in protocol['guards']]
    need(len(readoutids)==56,'Expected56 readout cases')
    for offset in range(0,56,4):
        rows=[]
        for cid in readoutids[offset:offset+4]:
            r=byid[cid];b=records[reference_record_paths[cid]]
            teacher=r.get('behavior_teacher')
            if teacher:targetpath,targetlabel=teacher['png']['path'],'Legacy '+teacher['kind']+' behavior teacher'
            else:targetpath,targetlabel=r['target']['path'],'Paired clean target'
            specs=[(r['input']['path'],'Input',False),(targetpath,targetlabel,False),(b['outputs']['png']['path'],'Frozen U96',True)]
            for u in SNAPSHOTS:specs.append((records[f'{RUN}/snapshots/u{u:04d}/{cid}/record.json']['outputs']['png']['path'],f'after{u} commits',True))
            rows.append((cid,specs))
        visual(rows,['Input','Clean / behavior teacher','Frozen U96','u0','u1','u4','u16','u32','u64'],f'readout_{offset//4+1:02d}.png','monitor + legacy readout')
    optids=[r['id'] for r in protocol['cases'] if r['cohort']=='optimize']
    exposures=defaultdict(list)
    for key,r in records.items():
        if r['kind']=='training':exposures[r['id']].append((r['update'],key))
    for offset in range(0,64,4):
        rows=[]
        for cid in optids[offset:offset+4]:
            r=byid[cid];b=records[reference_record_paths[cid]];exp=sorted(exposures[cid]);need(len(exp)==4,'Four saved optimizing exposures per rolecase')
            specs=[(r['input']['path'],'Input',False),(r['target']['path'],'Paired clean target',False),(b['outputs']['png']['path'],'Frozen U96',True)]
            for u,key in exp:specs.append((records[key]['outputs']['png']['path'],f'before commit{u}',True))
            rows.append((cid,specs))
        visual(rows,['Input','Clean target','Frozen U96','Exposure1','Exposure2','Exposure3','Exposure4'],f'optimization_{offset//4+1:02d}.png','optimization images BEFORE shown commit')
    need(len(visual_sheets)==30 and len(generated_cells)==712,'Exactly30 sheets and712 generated image cells required')
    need(len({c['source']['path'] for c in generated_cells})==712,'GeneratedPNG appears in multiple cells')
    need({c['source']['path'] for c in generated_cells}=={r['outputs']['png']['path'] for r in records.values()},'Visual inventory misses generatedPNG')
    visual_pin=write_json(OUT/'visual_manifest_v1.json',{'complete':True,'planned_sheets':30,'exact_saved256_cells':952,
        'generated_PNG_cells':712,'all_generated_PNGs_exactly_once':True,'root_personal_visual_review_pending':True,
        'rendering':'Unmodified256 RGB cells, no resampling, labels outside image pixels. Legacy dark showsR2 behavior teacher, not paired truth.',
        'sheets':visual_sheets})
    failure=docs.get(RUN+'/failure.json')
    report={'format':'complete-U96-B64-independent-saved-pixel-return-audit-v1','complete':True,
        'independent_saved_pixel_integrity_pass':not failures,'pilot_quality_gates_pass':endpoint['legacy']['pass'] and endpoint['monitor']['pass'],
        'model_qualified':False,'root_visual_review_pending':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'archive':{'path':ARCHIVE.relative_to(ROOT).as_posix(),'bytes':reader.count,'sha256':ARCHIVE_SHA,'payload_bytes':payload,
                   'payload_cap_bytes':PAYLOAD_CAP,'stream_member_count':total,'single_bounded_stream':True,'unsafe_extract_used':False,'pt_unpickled':False},
        'protocol':filepin(STAGE/'protocol.json'),'script':filepin(Path(__file__)),
        'independent_method':'No worker imports. NumPy/Pillow saved-output readback. Exact worker-specified historical whole mean-of-means and region internal-edge joint counts; exact <= / >= quality gates without epsilon.',
        'case_counts':dict(Counter(r['kind'] for r in records.values())),'region_internal_edge_counts':EDGE_COUNTS,
        'all712_RAW_to_floorPNG_exact':True,'all712_record_metrics_and_distances_exact':not failures,
        'exact_numeric_readback_comparisons':dict(comparisons),'integrity_failures':failures,'failure_receipt':failure,
        'initial56_zerohead_parity':zero_parity,'first4_grad_enabled_saved_stock_and_delivery_parity':first_training_stock,
        'snapshots_and_independently_recomputed_gates':gates,'blur_endpoint_gains':endpoint_gains,
        'all_saved_pixel_metrics':pixel_rows,'objectives':objective_rows,'objective_reconstruction_summary':objective_summary,
        'visual_manifest':visual_pin,'source_attribution_and_input_admission':'Retained sealed protocol and independent DATA admission audit; no data relabeling.',
        'boundary':{'Torch_imports':0,'neural_calls':0,'gradient_calls':0,'optimizer_calls':0,'training_calls':0,'VM_calls':0,'HTTP_calls':0,'browser_calls':0},
        'material_limits':['64 finite commits ending in retained QUALITY_STOP do not authorize extra epochs, an unchanged rerun, promotion or model qualification.',
            'New blur endpoint is a prospective TRAIN monitor criterion; it is separate from historical DGP feature gates, independent DEV/FINAL, native CCTV and identity claims.',
            'Guard clear original-CodeFormer and darkR2 teachers are behavior references. Legacy dark target is null; its images are not paired clean truth.',
            'Every generated image is placed once in a planned review sheet, but root personal visual review remains pending in this immutable audit.',
            'Saved objective reconstruction uses exact saved float32 inputs and declared formulas. CUDA/CPU reduction-order differences are explicitly reported; no quality tolerance is added.'],
        'seconds':time.monotonic()-started}
    outputpin=write_json(OUT/'saved_pixel_audit_v1.json',report)
    print(json.dumps({'complete':True,'integrity_pass':not failures,'pilot_quality_pass':report['pilot_quality_gates_pass'],
        'failures':len(failures),'generated_PNGs':712,'visual_sheets':30,'objective_summary':objective_summary,'audit':outputpin}),flush=True)


if __name__ == '__main__':
    main()
