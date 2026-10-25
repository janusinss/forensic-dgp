"""Read-only saved PNG threshold diagnosis and exact-cell binding readback.

NumPy/PIL file arithmetic only. No image changes, quantizer changes, new gradient,
parameter proposal/application, model, neural framework, browser or VM operation.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys


class BlockNeural(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in {'torch','torchvision','tensorflow','jax','basicsr'}:
            raise ImportError('Neural imports blocked during saved-PNG diagnosis')
        return None


sys.meta_path.insert(0,BlockNeural())
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit'
PARENT=ROOT/'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE=ROOT/'outputs/cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
OUT=ROOT/'outputs/codeformer_u64_gate_aligned_pilot_v1_png_diagnosis'
OPREFIX='outputs/u64_gate_aligned_pilot_v1/'
ROLES=('local_gaussian_1p2','global_gaussian_1p2','clear','lowlight_linear_0p35')


def need(ok,why):
    if not ok:raise ValueError(why)


def pin(path):
    path=Path(path).absolute();before=path.stat();need(path.is_file()and not path.is_symlink()and before.st_nlink==1,'Ordinary input required')
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    after=path.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Input changed during hashing')
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':after.st_size,'sha256':h.hexdigest()}


def payload_equal(a,b):return (a['bytes'],a['sha256'])==(b['bytes'],b['sha256'])


def ordered_mean(values):
    total=0.
    for v in values:total=total+float(v)
    return total/len(values)


def normalized(codes):return codes.astype(np.float32)/np.float32(255)


def metrics(value,target,central):
    if central:value,target=value[64:192,64:192],target[64:192,64:192]
    residual=value.astype(np.float64)-target.astype(np.float64)
    h=np.abs(np.diff(residual,axis=1));v=np.abs(np.diff(residual,axis=0))
    # Preserve the parent central sum-then-divide and whole mean-of-two-means.
    adjacent=float((h.sum()+v.sum())/(h.size+v.size))if central else float(.5*(h.mean()+v.mean()))
    return {'mae':float(np.abs(residual).mean()),'adjacent_rgb_error':adjacent,
            'horizontal_residual_edge_L1_mean':float(h.mean()),'vertical_residual_edge_L1_mean':float(v.mean()),
            'horizontal_edge_components':int(h.size),'vertical_edge_components':int(v.size)}


def delta_details(raw0,raw1,q0,q1,target_codes,central):
    if central:
        raw0,raw1,q0,q1,target_codes=[a[64:192,64:192]for a in(raw0,raw1,q0,q1,target_codes)]
    code=q1.astype(np.int16)-q0.astype(np.int16);changed=code!=0
    r0=q0.astype(np.int32)-target_codes.astype(np.int32);r1=q1.astype(np.int32)-target_codes.astype(np.int32)
    floats0=normalized(q0).astype(np.float64)-normalized(target_codes).astype(np.float64)
    floats1=normalized(q1).astype(np.float64)-normalized(target_codes).astype(np.float64)
    quantized_delta=normalized(q1).astype(np.float64)-normalized(q0).astype(np.float64)
    edges={}
    for name,axis in [('horizontal',1),('vertical',0)]:
        change=np.abs(np.diff(floats1,axis=axis))-np.abs(np.diff(floats0,axis=axis))
        affected=np.diff(quantized_delta,axis=axis)!=0
        integer_change=np.abs(np.diff(r1,axis=axis))-np.abs(np.diff(r0,axis=axis))
        need(np.all(change[~affected]==0),'Unchanged-code edge changed error')
        edges[name]={'edge_components':int(change.size),'code_change_affected_edge_components':int(affected.sum()),
             'error_increased_edges':int((change>0).sum()),'error_decreased_edges':int((change<0).sum()),
             'error_unchanged_edges':int((change==0).sum()),'exact_normalized_loss_sum_delta':float(change.sum()),
             'exact_normalized_mean_loss_delta':float(change.mean()),
             'integer_RGB8_residual_edge_L1_sum_delta':int(integer_change.sum()),
             'integer_RGB8_increased_edge_loss_units':int(integer_change[integer_change>0].sum()),
             'integer_RGB8_decreased_edge_loss_units':int(integer_change[integer_change<0].sum()),
             'unaffected_edge_loss_delta_exact_zero':True}
    total_edges=sum(v['edge_components']for v in edges.values())
    integer_sum=sum(v['integer_RGB8_residual_edge_L1_sum_delta']for v in edges.values())
    return {'RGB_code_components':int(code.size),'changed_RGB_code_components':int(changed.sum()),
            'changed_spatial_pixels':int(np.any(changed,axis=2).sum()),
            'positive_code_crossings':int((code>0).sum()),'negative_code_crossings':int((code<0).sum()),
            'maximum_positive_code_delta':int(code.max()),'minimum_negative_code_delta':int(code.min()),
            'maximum_absolute_code_delta':int(np.abs(code).max()),'sum_absolute_code_delta':int(np.abs(code).sum()),
            'changed_code_component_fraction':float(changed.mean()),
            'RAW_output_delta_MAD':float(np.abs(raw1.astype(np.float64)-raw0.astype(np.float64)).mean()),
            'PNG_normalized_output_delta_MAD':float(np.abs(quantized_delta).mean()),
            'edges':edges,'total_edge_components':total_edges,
            'integer_RGB8_residual_edge_L1_sum_delta':integer_sum,
            'integer_code_exact_255_adjacent_delta':integer_sum/(255.*total_edges),
            'exact_parent_FP32_normalized_adjacent_delta_from_edges':sum(v['exact_normalized_loss_sum_delta']for v in edges.values())/total_edges,
            'normalization_note':'Integer-code decomposition uses exact code residuals/255; authoritative parent metrics normalize codes to FP32 first. Both are reported without changing any quality threshold.'}


def main():
    OUT.mkdir(exist_ok=False);consumed={}
    def checked(path,binding=None):
        p=pin(path)
        if binding is not None:need(payload_equal(p,binding),'Input differs from authenticated pin: '+str(path))
        consumed[p['path']]=p;return Path(path)
    def doc(path,binding=None):return json.loads(checked(path,binding).read_bytes())
    inventory=doc(AUDIT/'inventory_v1.json');evidence=Path(inventory['evidence_root'])
    transport=doc(ROOT/'outputs/codeformer_u64_gate_aligned_pilot_v1_return_transport/audit_v1.json')
    need(inventory['files']==transport['all_member_pins'],'Inventory differs from independently streamed archive')
    protocol=doc(STAGE/'protocol.json');parent=doc(PARENT/'protocol.json')
    need(payload_equal(pin(STAGE/'protocol.json'),transport['protocol']),'Sealed pilot protocol differs')
    need(parent['files']=={r['parent_relative']:{k:r['actual_local'][k]for k in('bytes','sha256')}for r in transport['parent_bindings']},'Parent source table differs')
    rows={r['id']:r for r in parent['cases']+parent['guards']};banks=[]
    for update in range(3):
        name=OPREFIX+'banks/u'+str(update).zfill(4)+'/bank.json';banks.append(doc(evidence/name,inventory['files'][name]))
    need(all(len(b['records'])==56 and b['complete']is True for b in banks),'All three complete56case banks required')
    need(all(set(b['records'])==set(banks[0]['records'])for b in banks),'Bank membership changed')
    selection=protocol['selection'];monitor_ids={r['id']for r in selection['monitors']};percase=[];means={}
    floor_checks=0;metric_checks=0;unpaired=[]
    for case_id in sorted(banks[0]['records']):
        reference=rows[case_id];central=reference['role']==ROLES[0];target=None
        if reference.get('target')is not None:
            b=reference['target'];need(payload_equal(b,parent['files'][b['path']]),'Parent target pin differs')
            with Image.open(checked(PARENT/b['path'],b))as image:
                need(image.mode=='RGB'and image.size==(256,256),'Exact photographic RGB256 target required');target=np.asarray(image).copy()
        raw_arrays=[];codes=[];values=[];record_pins=[]
        for update,bank in enumerate(banks):
            record=bank['records'][case_id];need(record['id']==case_id and record['update']==update and record['target']==reference.get('target')and record['input']==reference['input'],'Bank/source state join differs')
            checked(evidence/record['receipt']['path'],inventory['files'][record['receipt']['path']]);record_pins.append(record['receipt'])
            raw_binding=record['outputs']['raw'];png_binding=record['outputs']['png']
            need(raw_binding==inventory['files'][raw_binding['path']]and png_binding==inventory['files'][png_binding['path']],'Output pin join differs')
            raw=np.load(checked(evidence/raw_binding['path'],raw_binding),allow_pickle=False)
            need(raw.dtype==np.float32 and raw.shape==(256,256,3)and np.isfinite(raw).all()and raw.min()>=0 and raw.max()<=1,'Actual finite RAW RGB256 required')
            with Image.open(checked(evidence/png_binding['path'],png_binding))as image:
                need(image.mode=='RGB'and image.size==(256,256),'Actual saved PNG RGB256 required');q=np.asarray(image).copy()
            need(np.array_equal(q,(np.clip(raw,0,1)*np.float32(255)).astype(np.uint8)),'Saved PNG differs from exact unchanged floor quantizer');floor_checks+=1
            if target is not None:
                mv={view:metrics(array,normalized(target),central)for view,array in [('raw',raw),('native_png',normalized(q))]}
                region='central128'if central else'whole'
                for view in mv:
                    for field in('mae','adjacent_rgb_error'):
                        need(mv[view][field]==record['regional_metrics'][view][region][field],'Actual independent parent scalar differs');metric_checks+=1
                values.append(mv)
            raw_arrays.append(raw);codes.append(q)
        if target is None:unpaired.append({'id':case_id,'reason':'Legacy dark guard has no aligned target; all3PNGfloor checks still passed'});continue
        pairs={}
        for before,after in [(0,1),(1,2),(0,2)]:
            key='u'+str(after)+'_vs_u'+str(before);detail=delta_details(raw_arrays[before],raw_arrays[after],codes[before],codes[after],target,central)
            detail['RAW_adjacent_delta']=values[after]['raw']['adjacent_rgb_error']-values[before]['raw']['adjacent_rgb_error']
            detail['PNG_adjacent_delta']=values[after]['native_png']['adjacent_rgb_error']-values[before]['native_png']['adjacent_rgb_error']
            detail['quantization_gap_delta']=detail['PNG_adjacent_delta']-detail['RAW_adjacent_delta']
            detail['parent_FP32_normalization_minus_integer_code_delta']=detail['PNG_adjacent_delta']-detail['integer_code_exact_255_adjacent_delta']
            pairs[key]=detail
        percase.append({'id':case_id,'reference_id':reference['reference_id'],'role':reference['role'],'monitor':case_id in monitor_ids,
                        'metric_region':'central128'if central else'whole','three_record_pins':record_pins,'values':values,'comparisons':pairs})
    for role in ROLES:
        cases=[r for r in percase if r['monitor']and r['role']==role];need(len(cases)==8,'Eight original adaptive TRAIN monitor photos required')
        state_means=[{view:{field:ordered_mean([c['values'][u][view][field]for c in cases])for field in('mae','adjacent_rgb_error','horizontal_residual_edge_L1_mean','vertical_residual_edge_L1_mean')}for view in('raw','native_png')}for u in range(3)]
        for u in range(3):
            declared=banks[u].get('current_means',banks[u]['gate']['cohort_means']['current'])
            for view in('raw','native_png'):
                for field in('mae','adjacent_rgb_error'):need(state_means[u][view][field]==declared[view][role][field],'Recomputed eight-photo means differ')
        comparisons={}
        for before,after in[(0,1),(1,2),(0,2)]:
            key='u'+str(after)+'_vs_u'+str(before);parts=[c['comparisons'][key]for c in cases]
            raw_delta=state_means[after]['raw']['adjacent_rgb_error']-state_means[before]['raw']['adjacent_rgb_error']
            png_delta=state_means[after]['native_png']['adjacent_rgb_error']-state_means[before]['native_png']['adjacent_rgb_error']
            code_count=sum(c['RGB_code_components']for c in parts);edge_count=sum(c['total_edge_components']for c in parts)
            comparisons[key]={'RAW_adjacent_mean_delta':raw_delta,'PNG_adjacent_mean_delta':png_delta,
                'quantization_gap_mean_delta':png_delta-raw_delta,
                'horizontal_PNG_mean_delta':state_means[after]['native_png']['horizontal_residual_edge_L1_mean']-state_means[before]['native_png']['horizontal_residual_edge_L1_mean'],
                'vertical_PNG_mean_delta':state_means[after]['native_png']['vertical_residual_edge_L1_mean']-state_means[before]['native_png']['vertical_residual_edge_L1_mean'],
                'changed_RGB_code_components':sum(c['changed_RGB_code_components']for c in parts),'total_RGB_code_components':code_count,
                'changed_code_component_fraction':sum(c['changed_RGB_code_components']for c in parts)/code_count,
                'maximum_absolute_code_delta':max(c['maximum_absolute_code_delta']for c in parts),
                'positive_code_crossings':sum(c['positive_code_crossings']for c in parts),'negative_code_crossings':sum(c['negative_code_crossings']for c in parts),
                'mean_RAW_output_delta_MAD':ordered_mean([c['RAW_output_delta_MAD']for c in parts]),
                'mean_PNG_normalized_output_delta_MAD':ordered_mean([c['PNG_normalized_output_delta_MAD']for c in parts]),
                'affected_residual_edge_components':sum(c['edges'][axis]['code_change_affected_edge_components']for c in parts for axis in('horizontal','vertical')),
                'total_residual_edge_components':edge_count,
                'integer_residual_edge_L1_loss_units_increase':sum(c['edges'][axis]['integer_RGB8_increased_edge_loss_units']for c in parts for axis in('horizontal','vertical')),
                'integer_residual_edge_L1_loss_units_decrease':sum(c['edges'][axis]['integer_RGB8_decreased_edge_loss_units']for c in parts for axis in('horizontal','vertical')),
                'net_integer_RGB8_residual_edge_L1_sum_delta':sum(c['integer_RGB8_residual_edge_L1_sum_delta']for c in parts),
                'integer_code_mean_adjacent_delta':ordered_mean([c['integer_code_exact_255_adjacent_delta']for c in parts]),
                'per_case_mean_contributions':[{'id':c['id'],'PNG_adjacent_delta':c['comparisons'][key]['PNG_adjacent_delta'],
                    'contribution_to_eight_photo_mean':c['comparisons'][key]['PNG_adjacent_delta']/8,
                    'RAW_adjacent_delta':c['comparisons'][key]['RAW_adjacent_delta'],
                    'changed_RGB_code_components':c['comparisons'][key]['changed_RGB_code_components']}for c in cases]}
        means[role]={'TRAIN_monitor_photos':8,'metric_region':cases[0]['metric_region'],'three_state_means':state_means,'comparisons':comparisons}
    local=means[ROLES[0]]['comparisons']
    need(local['u2_vs_u1']['RAW_adjacent_mean_delta']<0 and local['u2_vs_u1']['PNG_adjacent_mean_delta']>0 and
         local['u2_vs_u0']['RAW_adjacent_mean_delta']<0 and local['u2_vs_u0']['PNG_adjacent_mean_delta']>0,'Expected actual local RAW/PNG failure signs differ')
    for binding in consumed.values():need(pin(ROOT/binding['path'])==binding,'Consumed diagnosis source changed')
    receipt={'complete':True,'saved_PNG_threshold_diagnosis_pass':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
        'source':pin(Path(__file__)),'all3_complete56_case_banks_bound':True,'RAW_to_actual_PNG_exact_floor_checks':floor_checks,
        'independent_exact_saved_scalar_checks':metric_checks,'all_four_roles':means,'per_case_diagnosis':percase,
        'unaligned_legacy_dark_guards':unpaired,'consumed_files':list(consumed.values()),'consumed_before_after_exact':True,
        'finding':'The actual update2 localcentral128 monitor PNG adjacent-error mean worsens versus originalu64 and update1 while its RAW mean decreases. Sparse changed RGB8 codes account for every changed PNG residual-edge loss; their signed increases/decreases and per-photo contributions are measured here. This is one measured failure mechanism, not proof that all other quality/generalization issues are PNG-only.',
        'next_source_only_requirement':'Any future training design must account for actual delivered PNG thresholds and independently pass unchanged RAW/PNG preservation gates. This receipt selects no pilot, loss, gradient estimator, parameter step or quantizer change.',
        'quantizer_changed':False,'quality_gates_changed':False,'quantization_epsilon_added':False,'images_altered':False,
        'new_gradient_or_step_proposed':False,'native_wild_DEV_FINAL_evidence':False,'TRAIN_adaptive_monitor_only':True,
        'actual_neural_gradient_optimizer_parameter_training_VM_browser_calls':0,'model_qualified':False,
        'authoritative_quality_stop_retained':True,'no_update1_fallback_or_resumed_training_authorized':True}
    target=OUT/'receipt_v1.json'
    with target.open('x',encoding='utf-8',newline='\n')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'receipt':pin(target),'floor_checks':floor_checks,'metric_checks':metric_checks,
                     'local_central128':means[ROLES[0]],'other_role_u2_vs_u1':{r:means[r]['comparisons']['u2_vs_u1']for r in ROLES[1:]}},indent=2))


if __name__=='__main__':main()
