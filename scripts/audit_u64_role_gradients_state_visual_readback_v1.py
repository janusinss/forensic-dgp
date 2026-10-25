"""Independent saved-state and exact RGB-sheet readback, with neural imports blocked."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import re
import stat
import sys
import zipfile


class NoFramework(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','tensorflow','jax','basicsr','codeformer'}:
            raise ImportError('Neural framework import blocked in independent readback: '+fullname)
        return None


sys.meta_path.insert(0,NoFramework())
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'outputs/codeformer_u64_role_gradients_v1_return_audit'
PARENT = ROOT/'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE = ROOT/'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
OUTPUT = 'outputs/u64_role_gradients_v1/'
EXPECTED_ALL = '47e0707db446d3e39e9c101e8eedd631e49b45ba4f03a5f00aa2ad02e0281a8a'
EXPECTED_NPZ = '5a2d55fe9f136a1fcf035de3c3ca46e363defa5874572bf0dac261838845247b'


def need(ok, why):
    if not ok: raise ValueError(why)


def doc(path):
    def pairs(rows):
        d={}
        for k,v in rows:
            need(k not in d,'Duplicate JSON key'); d[k]=v
        return d
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def pin(path):
    path=Path(path).absolute(); before=path.lstat()
    need(path.resolve().is_relative_to(ROOT.resolve()) and stat.S_ISREG(before.st_mode) and
         before.st_nlink==1 and not path.is_symlink() and not getattr(before,'st_file_attributes',0)&1024,
         'Ordinary confined file required')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    after=path.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),
         'Audit source changed during read')
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':after.st_size,'sha256':h.hexdigest()}


def same_payload(a,b):
    return (a['bytes'],a['sha256'])==(b['bytes'],b['sha256'])


def ast_review(state_path,render_path):
    results={}
    for path in (state_path,render_path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        imports=[]; calls=[]; np_loads=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import): imports += [a.name for a in n.names]
            if isinstance(n,ast.ImportFrom): imports.append(n.module)
            if isinstance(n,ast.Call):
                calls.append(ast.unparse(n.func))
                if ast.unparse(n.func)=='np.load':
                    np_loads.append(any(k.arg=='allow_pickle' and isinstance(k.value,ast.Constant) and k.value.value is False for k in n.keywords))
        need(not set(i.split('.')[0] for i in imports)&{'torch','torchvision','subprocess','tensorflow','jax'},'Source imports framework/process facility')
        if path==state_path:
            need(len(np_loads)==3 and all(np_loads),'Every saved-state np.load must forbid pickle')
            need(not any(c in calls for c in ('np.concatenate','np.stack','list(first.values())','torch.load')),
                 'Unexpected aggregate whole-state/framework decode')
            explanation='Source iterates corresponding initial/closing tensors, with one head tensor lookup, rather than materializing full state dictionaries; retains metadata rows only.'
        else:
            need(not any(c.split('.')[-1] in {'resize','resample','filter','enhance','rotate','thumbnail','transform'} for c in calls),
                 'Renderer has resampling/filter/enhancement call')
            need('canvas.paste' in calls and 'canvas.crop' in calls,'Renderer exact-cell paste/readback required')
            explanation='Renderer opens RGB256 source images, copies and pastes at native dimensions, then verifies saved crop byte hashes.'
        results[path.name]={'source':pin(path),'imports':imports,'review_pass':True,'explanation':explanation}
    return results


def stream_state(path,expected_rows):
    """Read each NPY payload in 1MiB float32 blocks, never instantiate a model/state dict."""
    digests={k:hashlib.sha256() for k in ('all','heads','base','adapters','frozen')}
    rows=[]; counts=Counter(); finite_bytes=0; maximum_block=0
    with zipfile.ZipFile(path) as z:
        infos=z.infolist(); names=[v.filename for v in infos]
        need(len(infos)==len(set(names))==575 and all(re.fullmatch(r'[A-Za-z0-9_.]+\.npy',n) and '..' not in n for n in names),
             'Exact flat 575-array namespace required')
        for info in sorted(infos,key=lambda x:x.filename):
            need(info.compress_type==zipfile.ZIP_STORED and not info.flag_bits&1 and info.file_size<32*1024**2,
                 'Bounded nonencrypted stored NPY required')
            name=info.filename[:-4]; kind=name.split('.')[0]
            need(kind in ('heads','base','adapters'),'Unexpected tensor namespace')
            with z.open(info) as f:
                version=np.lib.format.read_magic(f)
                need(version in ((1,0),(2,0)),'Bounded legacy NPY header required')
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
                need(dtype==np.dtype('<f4') and not dtype.hasobject and not fortran,'Finite C-order little-endian FP32 payload required')
                total=math.prod(shape)*4
                need(total<32*1024**2 and total==expected_rows[name]['bytes'],'Tensor payload size differs')
                target_keys=['all',kind]+(['frozen'] if kind!='heads' else [])
                for key in target_keys:
                    cname=name.removeprefix('base.') if key=='base' else name
                    header=json.dumps([cname,'torch.float32',list(shape)],separators=(',',':')).encode()
                    digests[key].update(len(header).to_bytes(8,'big'));digests[key].update(header)
                h=hashlib.sha256(); left=total
                while left:
                    block=f.read(min(left,1024**2));need(block and len(block)%4==0,'Complete float32 block required')
                    need(np.isfinite(np.frombuffer(block,dtype='<f4')).all(),'Nonfinite saved-state payload')
                    h.update(block)
                    for key in target_keys:digests[key].update(block)
                    finite_bytes+=len(block);maximum_block=max(maximum_block,len(block));left-=len(block)
                need(not f.read(1),'Unexpected trailing NPY payload')
                row={'name':name,'shape':list(shape),'dtype':'float32','bytes':total,'bytes_sha256':h.hexdigest()}
                need(row==expected_rows[name],'Tensor table differs: '+name)
                rows.append(row);counts[kind]+=1
    return {'complete':True,'counts':dict(counts),'canonical_sha256':{k:v.hexdigest() for k,v in digests.items()},
        'tensor_rows':rows,'finite_FP32_payload_bytes':finite_bytes,'maximum_array_payload_read_block_bytes':maximum_block,
        'full_state_or_model_object_instantiation':False,'bounded_numpy_float32_views_used_for_finiteness':True,
        'NPY_headers_and_1MiB_FP32_views_only':True}


def main():
    source_paths=[ROOT/'scripts/audit_u64_role_gradients_states_v1.py',ROOT/'scripts/render_u64_role_gradients_return_v1.py']
    inventory=doc(AUDIT/'inventory_v1.json'); state=doc(AUDIT/'saved_state_audit_v1.json')
    visual=doc(AUDIT/'visual_bindings_v1.json'); review=doc(AUDIT/'root_visual_review_v1.json')
    transport_path=ROOT/'outputs/codeformer_u64_role_gradients_v1_return_transport/transport_receipt_v1.json'
    transport=doc(transport_path); protocol=doc(STAGE/'protocol.json'); parent=doc(PARENT/'protocol.json')
    evidence=Path(inventory['evidence_root']);consumed={}
    def checked(path,binding=None):
        actual=pin(path)
        if binding is not None:need(actual==binding,'Consumed file changed: '+str(path))
        consumed[actual['path']]=actual;return Path(path)
    for p in [*source_paths,AUDIT/'inventory_v1.json',AUDIT/'saved_state_audit_v1.json',AUDIT/'visual_bindings_v1.json',
              AUDIT/'root_visual_review_v1.json',transport_path,STAGE/'protocol.json',PARENT/'protocol.json']:
        checked(p)
    need(transport['complete'] is True and transport['export_all_file_pins_match'] is True and
         inventory['files']==transport['all_member_pins'],'Root inventory/independent transport member table differs')
    need(same_payload(pin(STAGE/'protocol.json'),transport['protocol']),'Current diagnostic protocol differs from independent transport seal')
    need(same_payload(pin(PARENT/'protocol.json'),transport['parent_protocol_actual']),'Current parent protocol differs from independent transport seal')
    ast_result=ast_review(*source_paths)
    need(pin(source_paths[0])==state['source'] and pin(source_paths[1])==visual['source'],'Receipt source pin differs from actual source')
    for b in state['consumed_files']:checked(ROOT/b['path'],b)
    tensor_rows={r['name']:r for r in state['tensor_rows']}
    need(len(tensor_rows)==575 and len(state['tensor_rows'])==575,'Unique 575 tensor table required')
    records=[]
    for phase in ('initial','closing'):
        name=OUTPUT+'full_states/'+phase+'.json'; path=evidence/name
        need(same_payload(pin(path),inventory['files'][name]),'State metadata differs from transport')
        record=doc(path);records.append(record)
        need({r['name']:r for r in record['description']['arrays']}==tensor_rows and
             len(record['description']['arrays'])==575,'State description/table differs')
        name=record['state']['path'];need(same_payload(pin(evidence/name),inventory['files'][name]) and
            same_payload(record['state'],inventory['files'][name]),'Full state payload record differs')
        need(record['complete'] is True and record['canonical_sha256']==record['description']['canonical_sha256']==EXPECTED_ALL,
             'State canonical metadata differs')
    initial=evidence/records[0]['state']['path']; closing=evidence/records[1]['state']['path']
    prior_path=ROOT/'outputs/codeformer_u64_cleananchors_continuation_v1_return_audit/extracted/outputs/u64_cleananchors_continuation_v1/states/initial_u64/state.json'
    prior=doc(prior_path); prior_file=prior_path.parent/'full_state.npz'
    for p in (initial,closing,prior_file):need(pin(p)['sha256']==EXPECTED_NPZ and pin(p)['bytes']==377504068,'Initial/closing/prior NPZ byte mismatch')
    need(prior['canonical_sha256']==EXPECTED_ALL and prior['tensor_count']==575 and prior['update']==0,'Previous original-u64 metadata differs')
    need({r['name']:r for r in prior['arrays']}==tensor_rows,'Previous original-u64 tensor table differs')
    independent=stream_state(initial,tensor_rows)
    need(independent['canonical_sha256']==state['canonical_sha256'] and independent['counts']==state['tensor_counts'],
         'Independent streamed canonical tensor value readback differs')
    need(independent['counts']=={'adapters':24,'base':515,'heads':36} and
         state['canonical_sha256']['heads']==protocol['diagnostic']['initial_head_canonical_sha256'],
         'Frozen/head counts or original-u64 head canonical differs')
    need(state['saved_state_audit_pass'] is True and state['model_qualified'] is False,'State receipt scope differs')
    need(visual['complete'] is True and visual['planned_pages']==visual['actual_pages']==18 and
         visual['output_slots']==list(range(1,73)) and visual['output_cells_once_each']==72 and visual['total_cells']==216,
         'Exact 18-page 72-slot census required')
    need(len(visual['pages'])==18 and len(visual['cells'])==216,'Exact visual metadata row count required')
    need(review['complete'] is True and review['root_personally_inspected_all_planned_sheets'] is True and
         review['sheets']==18 and review['actual_output_cells']==72 and review['total_RGB_cells']==216 and
         review['every_actual_output_covered_once'] is True and review['visual_bindings']==pin(AUDIT/'visual_bindings_v1.json'),
         'Root visual-review join differs')
    need([r['file'] for r in review['pages']]==visual['pages'] and [r['page'] for r in review['pages']]==list(range(1,19)),
         'Personally reviewed pages differ from rendered page pins')
    rows={r['id']:r for r in parent['cases']+parent['guards']};cells={(c['slot'],c['column']):c for c in visual['cells']}
    need(len(cells)==216 and set(cells)=={(s,c) for s in range(1,73) for c in range(3)},'Duplicate/missing visual slot/column')
    expected_consumed=set();cell_readback=[]
    for page_number,page_pin in enumerate(visual['pages'],1):
        page_path=checked(ROOT/page_pin['path'],page_pin)
        with Image.open(page_path) as page:
            need(page.mode=='RGB' and page.size==(820,1272) and getattr(page,'n_frames',1)==1,'Page raster dimensions/mode differ')
            for call in protocol['plan'][(page_number-1)*4:page_number*4]:
                slot=call['slot'];row=rows[call['id']]
                recname=OUTPUT+'calls/'+str(slot).zfill(4)+'/record.json';recpath=evidence/recname
                need(same_payload(pin(recpath),inventory['files'][recname]),'Visual output record differs from transport')
                checked(recpath);expected_consumed.add(recpath.relative_to(ROOT).as_posix());record=doc(recpath)
                need(record['call']==call and record['id']==call['id'] and record['reference_id']==row['reference_id']
                     and record['role']==row['role'] and record['input']==row['input'] and record['target']==row['target'],
                     'Visual record/call/parent semantics differ')
                for col in range(3):
                    cell=cells[(slot,col)];xy=[10+270*col,102+290*((slot-1)%4)]
                    need(cell['page']==page_number and cell['xy']==xy,'Visual layout slot position differs')
                    binding=row['input'] if col==0 else (row['target'] if col==1 else record['outputs']['png'])
                    if col<2:
                        path=PARENT/binding['path']
                        need(same_payload(binding,parent['files'][binding['path']]),'Visual input/target differs from sealed parent')
                    else:
                        path=evidence/binding['path']
                        need(same_payload(binding,inventory['files'][binding['path']]),'Visual output differs from authenticated archive')
                    actual=pin(path);need(same_payload(actual,binding) and actual==cell['source'],'Visual cell source pin differs')
                    checked(path);expected_consumed.add(actual['path'])
                    with Image.open(path) as source:
                        need(source.mode=='RGB' and source.size==(256,256) and getattr(source,'n_frames',1)==1,'Source RGB256 raster required')
                        source_pixels=source.tobytes();x,y=xy
                        delivered=page.crop((x,y,x+256,y+256)).tobytes()
                        need(delivered==source_pixels and hashlib.sha256(source_pixels).hexdigest()==cell['RGB8_sha256'],
                             'Saved page crop differs from exact source RGB pixels')
                        cell_readback.append({'page':page_number,'slot':slot,'column':col,'xy':xy,
                                              'source':actual,'RGB8_sha256':cell['RGB8_sha256'],'exact_pixels':True})
    reported_consumed={r['path']:r for r in visual['consumed_files']}
    need(len(reported_consumed)==len(visual['consumed_files']) and set(reported_consumed)==expected_consumed,
         'Visual consumed source/record namespace differs')
    for binding in reported_consumed.values():checked(ROOT/binding['path'],binding)
    for binding in consumed.values():need(pin(ROOT/binding['path'])==binding,'Consumed file changed after readback')
    result={'complete':True,'independent_state_visual_readback_pass':True,'unresolved_blockers':[],
        'created_UTC':datetime.now(timezone.utc).isoformat(),'source':pin(Path(__file__)),'source_reviews':ast_result,
        'state':{'all575_tensor_rows_match_initial_closing_prior_original_u64':True,
                 'initial_closing_prior_NPZ_bytes':377504068,'initial_closing_prior_NPZ_sha256':EXPECTED_NPZ,
                 'frozen_tensors_unchanged':539,'head_tensors_unchanged':36,'independent_stream_readback':independent,
                 'raw_gradient_fields_claimed_from_saved_values':False},
        'visuals':{'planned_pages':18,'saved_pages':18,'exact_RGB_cells':len(cell_readback),
                   'output_slots_once_each':72,'all_cells_256x256_without_resampling_filters_or_enhancement':True,
                   'all_page_crop_bytes_equal_exact_bound_sources':True,'cells':cell_readback,'pages':visual['pages'],
                   'source_semantics_joined_to_sealed_parent_and_archive':True,
                   'root_personal_all18_page_review_receipt_authenticated':True,
                   'independent_human_FINAL_review':False,'independent_auditor_role':'saved-pixel binding/readback; visual observations remain attributed to root'},
        'consumed_files':list(consumed.values()),'all_consumed_pins_rehashed_after':True,
        'neural_framework_imports_blocked':True,'actual_neural_gradient_optimizer_training_VM_browser_calls':0,
        'model_qualified':False,'integrity_or_baseline_visuals_do_not_imply_quality_improvement':True}
    target=AUDIT/'independent_state_visual_readback_v1.json'
    with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'state_tensor_rows':len(independent['tensor_rows']),'canonical':independent['canonical_sha256'],
        'exact_cells':len(cell_readback),'pages':18,'output_slots_once_each':72,'source_reviews_pass':True,
        'unresolved_blockers':[],'receipt':pin(target)},indent=2))


if __name__=='__main__':main()
