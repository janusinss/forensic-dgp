"""Independent hash-closed saved state readback; CPU only, no neural execution.

Torch is imported only after an exact completed root import receipt, source and
checkpoint byte hashes have been checked. weights_only+mmap loads CPU values;
no returned Python, module/model, optimizer, autograd or inference is executed.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'd0c0b7a5be4577803bc156b611386c3be10068a3e78a47f9348125242f0c79d1'
ARCHIVE = 'c746c7d7edbca0685032f34e84e60299140d0922db2e687d2e13ec59c6cbb86b'
MANIFEST = '5533e7f2da8bc3d5dc7b0e76925351fa85583dba1212266e130c87ac954bf68e'
PREFIX = 'outputs/normal_tone_training_v1/'
TRAINABLE = ('encoder', 'position_emb', 'feat_emb', 'ft_layers', 'idx_pred_layer', 'fuse_convs_dict')
FROZEN = ('generator', 'quantize')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(a, b, bound=1e-12):
    require(finite(a) and finite(b) and abs(a-b) <= bound*max(1., abs(b)), 'Scalar arithmetic differs')


class Evidence:
    def __init__(self, receipt_path):
        self.receipt_path = Path(receipt_path).resolve(strict=True)
        self.receipt_sha = sha(self.receipt_path)
        value = read(self.receipt_path)
        require(value.get('complete') is True and value.get('archive_sha256') == ARCHIVE and
                value.get('protocol_sha256') == PROTOCOL, 'Exact completed root import receipt required')
        self.root = Path(value.get('import_root', value.get('import_directory', ''))).resolve(strict=True)
        require(self.root == (ROOT/'outputs/codeformer_normal_tone_training_v1_return').resolve(strict=True),
                'Exact new imported return directory required')
        self.hashes, self.sizes = value['files_sha256'], value['files_bytes']
        require(set(self.hashes) == set(self.sizes) and len(self.hashes) == 6304,
                'Exact6304-member closed map required')
        self.read_pins = {}
        manifest_name = PREFIX+'export_manifest.json'
        require(self.hashes[manifest_name] == MANIFEST, 'Exact returned manifest required')
        self.manifest = self.json(manifest_name)
        require(self.manifest['protocol_sha256'] == PROTOCOL and
                self.manifest['files_sha256'] == {k:v for k,v in self.hashes.items() if k != manifest_name} and
                self.manifest['files_bytes'] == {k:v for k,v in self.sizes.items() if k != manifest_name},
                'Import map and immutable manifest differ')
        self.p = self.json('protocol.json')
        require(self.hashes['protocol.json'] == PROTOCOL and self.p['training']['max_updates'] == 50,
                'Frozen finite protocol differs')
        for name, digest in self.p['files_sha256'].items():
            require(self.hashes[name] == digest and self.sizes[name] == self.p['files_bytes'][name],
                    'Returned frozen source namespace differs')
        self.dependency_resolution = read(ROOT/'outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json')['local_dependency_resolution']

    def get(self, name):
        relative = PurePosixPath(name)
        require(name in self.hashes and relative.as_posix() == name and not relative.is_absolute() and
                '..' not in relative.parts and ':' not in name and '\\' not in name, 'Unbound/unsafe member')
        path = self.root.joinpath(*relative.parts)
        require(path.resolve(strict=True).is_relative_to(self.root), 'Evidence path escape')
        for ancestor in [path, *path.parents]:
            if ancestor == self.root.parent:
                break
            info = ancestor.lstat()
            require(not ancestor.is_symlink() and not getattr(info, 'st_file_attributes', 0)&1024,
                    'Linked evidence forbidden')
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size == self.sizes[name] and
                sha(path) == self.hashes[name], 'Changed imported bytes: '+name)
        self.read_pins[name] = {'sha256':self.hashes[name], 'bytes':info.st_size,
                                'mtime_ns':info.st_mtime_ns, 'ino':info.st_ino}
        return path

    def json(self, name):
        return read(self.get(name))

    def parent(self, name):
        pin = self.p['dependencies'][name]
        path = Path(self.dependency_resolution[name]).resolve(strict=True)
        require(path.is_relative_to(ROOT) and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'Read-only parent dependency differs: '+name)
        return path

    def close(self):
        require(sha(self.receipt_path) == self.receipt_sha, 'Import receipt changed')
        for name, pin in self.read_pins.items():
            info = (self.root/name).stat()
            require((info.st_size,info.st_mtime_ns,info.st_ino) == (pin['bytes'],pin['mtime_ns'],pin['ino']),
                    'Read evidence changed during audit')
        return {'files_individually_hashed_before_read':len(self.read_pins),
                'read_file_stat_identity_rechecked_at_close':True,'import_receipt_sha256':self.receipt_sha}


def events(p):
    counts = {key:0 for key in ['optimizer_updates','microbatches','gradient_queries','backward_calls',
        'baseline_forwards','initializer_parity_candidate_forwards','snapshot_forwards','teacher_forwards',
        'candidate_training_forwards','perceptual_prerequisite_forwards','perceptual_training_forwards',
        'route_witness_forwards','route_witness_perceptual_forwards','behavior_teacher_forwards',
        'optimizer_step_attempts','training_input_writes']}
    rows, committed, pending = [], 0, 0
    def reserve(kind):
        counts[kind] += 1
        rows.append({'event':len(rows)+1,'before_call':kind,'committed_update':committed,
                     'pending_microbatches':pending,'counts':dict(counts)})
    reserve('perceptual_prerequisite_forwards'); reserve('gradient_queries')
    for _ in range(2):
        reserve('baseline_forwards'); reserve('initializer_parity_candidate_forwards')
    for _ in range(391): reserve('teacher_forwards')
    for _ in range(4):
        reserve('route_witness_perceptual_forwards'); reserve('route_witness_forwards')
        reserve('gradient_queries'); reserve('gradient_queries')
    reserve('gradient_queries'); reserve('gradient_queries')
    for _ in range(406): reserve('snapshot_forwards')
    for update in range(1,51):
        pending = 0
        for slot in range(1,5):
            reserve('training_input_writes')
            if slot == 3: reserve('behavior_teacher_forwards')
            if slot >= 3 or (update-1)%4 >= 2: reserve('perceptual_training_forwards')
            reserve('candidate_training_forwards'); reserve('backward_calls')
            counts['microbatches'] += 1; pending += 1
        reserve('optimizer_step_attempts')
        committed=update;counts['optimizer_updates']=update;pending=0
        if update == 50:
            for _ in range(406): reserve('snapshot_forwards')
    require(len(rows) == 2075 and all(counts[k] == cap for k,cap in p['training']['call_caps'].items()),
            'Independent finite census differs')
    return rows,counts


def sampler_states(ids, history, np):
    ids = sorted(ids)
    require(len(ids)==391 and len(set(ids))==391, 'Exact391TRAIN IDs required')
    rng = np.random.Generator(np.random.PCG64(20261015))
    state = {'format':'codeformer-train-sampler-v2','seed':20261015,
        'reference_ids_sha256':hashlib.sha256(canonical(ids)).hexdigest(),'completed_updates':0,
        'next_update':1,'selection_count':0,'permutation_index':-1,'permutation':[],
        'cursor':0,'rng_state':deepcopy(rng.bit_generator.state),'rotation_history':[]}
    states = {0:deepcopy(state)}
    for row in history:
        update=row['update'];before=hashlib.sha256(canonical(state)).hexdigest();rotations=[]
        def draw(avoid=None):
            if state['cursor']==len(state['permutation']):
                state['permutation']=rng.permutation(ids).tolist();state['cursor']=0;state['permutation_index']+=1
                if avoid is not None and state['permutation'][0]==avoid:
                    amount=1
                    while state['permutation'][amount]==avoid:amount+=1
                    state['permutation']=state['permutation'][amount:]+state['permutation'][:amount]
                    event={'permutation_index':state['permutation_index'],'left_rotation':amount,
                           'avoided_id':avoid,'update':update}
                    rotations.append(event);state['rotation_history'].append(event)
            value=state['permutation'][state['cursor']];state['cursor']+=1;return value
        a=draw();b=draw(a)
        state.update(completed_updates=update,next_update=update+1,selection_count=2*update,
                     rng_state=deepcopy(rng.bit_generator.state))
        selection={'update':update,'id_a':a,'id_b':b,'rotation_events':rotations,
            'state_before_sha256':before,'state_after_sha256':hashlib.sha256(canonical(state)).hexdigest()}
        require(row['TRAIN_selection']==selection,'Independent PCG64 sampler replay differs')
        states[update]=deepcopy(state)
    return states


def audit_metadata(e, np):
    p=e.p;failure=e.json(PREFIX+'failure.json')
    require(failure['complete'] is False and failure['completed_updates']==50 and
            failure['optimizer_inflight'] is False and failure['pending_step_record'] is None and
            failure['error']=='Declared RAW/PNG clear-preservation guard failed; retained snapshot',
            'Expected finite quality stop differs')
    plan,counts=events(p);previous=0.
    for row in plan:
        saved=e.json(PREFIX+'call_trace/%05d.json'%row['event'])
        require({key:saved[key] for key in row}==row,'Call trace order/counter differs')
        require(finite(saved['seconds']) and previous<=saved['seconds']<=3000,'Call clock differs')
        previous=saved['seconds']
    require(failure['counts']==counts and failure['last_call_reservation']==saved,'Terminal count/last event differs')
    history=[json.loads(line) for line in e.get(PREFIX+'training.jsonl').read_text().splitlines() if line.strip()]
    require([row['update'] for row in history]==list(range(1,51)),'Exact50committed history required')
    parity=read(e.parent(p['training_parity']['path']))
    cases={(row['update'],row['slot']):row for row in parity['cases']}
    profiles=Counter();ids=Counter();metadata_diffs=[];scalars=[];seconds=0.
    for row,prior in zip(history,parity['updates']):
        u=row['update'];require(row['optimizer_committed'] is True and len(row['microbatches'])==4 and
            row['TRAIN_selection']==prior['TRAIN_selection'] and row['group_lrs']==prior['group_lrs'],
            'History commit/sampler/LR differs')
        require(seconds<=row['optimization_seconds']<=900,'Optimization clock differs');seconds=row['optimization_seconds']
        for slot,sample in enumerate(row['microbatches'],1):
            expected=cases[(u,slot)];record=sample['pair_record'];normal=slot in (1,3)
            require(sample['slot']==slot and sample['id']==expected['reference_id'] and
                sample['image_eligible']==expected['image_eligible'] and
                all(record[k]==expected[k] for k in ['update','slot','reference_id','source_sha256','profile',
                    'input256_rgb8_sha256','input256_float32_sha256','clean512_rgb8_sha256','clean512_float32_sha256']),
                'History source/target/input/eligibility differs')
            require(record['member']==('normal' if normal else 'dark') and record['role']=='train',
                    'Normal/dark supervision role differs')
            expected_terms={'token_ce','latent_mse','coarse_rgb_l1'}
            if sample['image_eligible']:expected_terms|={'pixel_l1_rgb','lpips_vgg'}
            if slot==3:expected_terms|={'behavior_rgb_l1','behavior_token_kl'}
            require(set(sample['terms'])==set(sample['weighted_terms'])==expected_terms,'Fresh loss eligibility namespace differs')
            for name,value in sample['terms'].items():
                require(finite(value),'Nonfinite loss');close(sample['weighted_terms'][name],value*p['training']['loss_weights'][name])
            weighted=sum(sample['weighted_terms'].values())
            reduction_bound=32*2**-23*max(1.,sum(abs(x) for x in sample['weighted_terms'].values()))
            require(abs(sample['total_before_accumulation_division']-weighted)<=reduction_bound,
                    'Logged fresh objective is inconsistent with weighted terms')
            m=sample['coarse_supervision_metadata']
            require(sample['normal_coarse_mean_is_differentiated'] is normal and m['slot']==slot and
                m['normal_observation'] is normal and m['grid']==[16,16] and m['detached_reporting_only'] is True and
                sample['coarse_supervision']==('luminance-centered-normal' if normal else 'original-full-dark'),
                'Selected normal-only objective metadata differs')
            for name in ('ordinary_coarse_l1','centered_coarse_l1','mean_coarse_luminance_error'):
                require(finite(m[name]),'Nonfinite coarse readout')
            selected=m['centered_coarse_l1' if normal else 'ordinary_coarse_l1']
            close(m['selected_training_coarse_l1'],sample['terms']['coarse_rgb_l1'])
            close(m['selected_minus_reporting'],sample['terms']['coarse_rgb_l1']-selected)
            require(m['selected_term_matches_reporting_exactly']==(sample['terms']['coarse_rgb_l1']==selected),
                    'Coarse metadata comparison flag differs')
            metadata_diffs.append(abs(sample['terms']['coarse_rgb_l1']-selected))
            dtype=sample['candidate_dtypes']
            require(dtype['logits_dtype']==dtype['probabilities_dtype']==dtype['native_output_dtype']=='torch.float32' and
                    dtype['autocast_enabled'] is False and dtype['adain_gradient_route']['policy']=='candidate-AdaIN-style-detach-only-v1',
                    'Training FP32/isolated route declaration differs')
            profiles[record['profile']]+=1;ids[sample['id']]+=1
            scalars.append({'update':u,'slot':slot,'normal':normal,**m})
    refs=read(e.parent(p['data_manifest']))['references']
    states=sampler_states([row['id'] for row in refs if row['historical_role']=='train'],history,np)
    formula=e.json(PREFIX+'normal_formula_prerequisites.json')
    require(formula['complete'] is formula['pass'] is True and formula['gradient_queries']==2 and
        formula['before_Adam_construction'] is True and formula['neural_forwards']==formula['optimizer_updates']==0 and
        formula['CPU_CUDA_RNG_restored'] is True and len(formula['fixtures'])==2,'Formula prerequisite census differs')
    formula_checks=[]
    weights=np.array([.2126,.7152,.0722],dtype=np.float64).reshape(1,3,1,1)
    for row,name in zip(formula['fixtures'],['asymmetric_non_kink','constant_chroma']):
        require(row['fixture']==name and e.json(PREFIX+'normal_formula_prerequisites/'+name+'.json')==row,'Formula identity/record differs')
        value=np.array(row['input_values'],dtype=np.float64).reshape(1,3,2,2)
        mu=(value*weights).sum(axis=1,keepdims=True).mean(axis=(2,3),keepdims=True)
        signs=np.sign(value-mu);expected=signs/12-weights*signs.mean()/4
        actual=np.array(row['actual_gradient_values'],dtype=np.float32).reshape(1,3,2,2)
        recorded=np.array(row['expected_gradient_values'],dtype=np.float64).reshape(1,3,2,2)
        require(np.array_equal(expected,recorded),'Saved analytic derivative differs')
        require(hashlib.sha256(actual.tobytes()).hexdigest()==row['actual_gradient_float32_sha256'] and
                hashlib.sha256(expected.tobytes()).hexdigest()==row['expected_gradient_float64_sha256'],
                'Formula gradient value/hash differs')
        maximum=float(np.abs(actual.astype(np.float64)-expected).max());bound=2e-7+2e-5*float(np.abs(expected).max())
        close(maximum,row['maximum_absolute_error']);close(bound,row['bound'])
        require(row['pass'] is True and maximum<=bound and abs(float(actual.astype(np.float64).sum()))<=12*bound,
                'Formula derivative failed prospective numerical bound')
        formula_checks.append({'fixture':name,'entries':12,'max_error':maximum,'bound':bound})
    snapshots={n:e.json(PREFIX+'snapshot_%04d/metrics.json'%n) for n in (0,50)}
    require(all(v['complete'] is True and len(v['rows'])==406 and len(v['clear'])==61 and len(v['blur'])==53 for v in snapshots.values()),'Full guard/output cohort differs')
    require(snapshots[50]['quality_guard']['pass'] is False and snapshots[50]['normal_blur_photometric_guard']['pass'] is False,
            'Expected preserved quality failure differs')
    return {'failure':failure,'history':history,'snapshots':snapshots,'sampler_states':states,
        'report':{'history_updates':50,'exposures':200,'normal_exposures':100,'dark_exposures':100,
            'unique_photographic_TRAIN_references':len(ids),'profile_counts':dict(profiles),
            'loss_names_weights_finite_logged_total_consistent':True,'full_fresh_graph_and_total_div4_are_source_bound':True,
            'maximum_selected_vs_detached_coarse_reporting_difference':max(metadata_diffs),
            'all_200_coarse_reporting_exact':all(x==0 for x in metadata_diffs),
            'optimization_seconds':seconds,'formula_derivatives':formula_checks,
            'call_trace':{'events':2075,'neural_forwards':1614,'queries':11,'backward':200,'attempts':50,'counts':counts},
            'independent_sampler_replay':True,'coarse_scalars':scalars}}


def tensor_digest(state, torch):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'big'));digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_supervision_arrays(e, np):
    hq=read(e.parent(e.p['hq_cache_parity']['path']))['cases']
    require(len(hq)==391 and len({row['reference_id'] for row in hq})==391,'HQ manifest census differs')
    def array(name,shape,dtype):
        path=e.get(name)
        with path.open('rb') as stream:
            value=np.load(stream,allow_pickle=False)
            require(not stream.read(1),'Trailing NPY bytes')
        require(value.shape==shape and value.dtype==np.dtype(dtype) and np.isfinite(value).all(),
                'Supervision array geometry/dtype/finiteness differs')
        return value
    def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    hq_bytes=0
    for expected in hq:
        folder=PREFIX+'HQ_cache/'+expected['reference_id']+'/'
        row=e.json(folder+'record.json')
        require(row['role']=='train' and row['reference_id']==expected['reference_id'] and
            row['source_path']==expected['source_path'] and row['source_sha256']==expected['source_sha256'] and
            all(row[key]==expected[key] for key in ['clean512_float32_sha256','teacher_state_sha256','restoration_codebook_state_sha256']),
            'HQ target source/teacher/codebook binding differs')
        for label,shape,dtype in [('indices',(1,256),'int64'),('features',(1,256,16,16),'float32')]:
            name=folder+label+'.npy';value=array(name,shape,dtype)
            require(row[label+'_path']==name and row[label+'_sha256']==e.hashes[name] and
                digest(value)==expected[label+'_array_sha256'],'HQ retained exact array parity differs')
            if label=='indices':require(value.min()>=0 and value.max()<1024,'HQ index range differs')
            hq_bytes+=value.nbytes
    anchors=read(e.parent(e.p['training_parity']['path']))['behavior_anchors']
    require([row['update'] for row in anchors]==list(range(1,51)),'Behavior anchor exact census differs')
    anchor_bytes=0;online_nonexact=Counter()
    for expected in anchors:
        folder=PREFIX+'behavior_anchors/update_%04d/'%expected['update']
        online=PREFIX+'online_behavior_anchors/update_%04d/'%expected['update']
        row=e.json(folder+'record.json');parity=e.json(folder+'parity.json')
        require(row['slot']==3 and row['reference_id']==expected['reference_id'] and
            row['teacher_state_sha256']==expected['teacher_state_sha256'] and
            row['input256_float32_sha256']==expected['input256_float32_sha256'],'Used behavior source/teacher differs')
        for label,key,shape in [('delivered256','delivered',(1,3,256,256)),('logits','logits',(1,256,1024))]:
            name=folder+label+'.npy';value=array(name,shape,'float32')
            prior=e.parent(expected[key+'_path'])
            require(sha(prior)==expected[key+'_sha256']==e.hashes[name] and
                row[key+'_path']==name and row[key+'_sha256']==e.hashes[name],
                'Behavior optimization target is not exact retained bytes')
            fresh=array(online+label+'.npy',shape,'float32')
            delta=np.abs(fresh.astype(np.float64)-value.astype(np.float64))
            comparison=parity['comparisons'][key]
            require(comparison['exact_array_equal']==bool(np.array_equal(fresh,value)) and
                comparison['online_tensor_sha256']==digest(fresh) and comparison['used_target_tensor_sha256']==digest(value),
                'Online/used behavior comparison binding differs')
            close(comparison['maximum_absolute'],float(delta.max()));close(comparison['mean_absolute'],float(delta.mean()))
            online_nonexact[key]+=int(not np.array_equal(fresh,value));anchor_bytes+=value.nbytes
    return {'HQ_targets':391,'HQ_physical_arrays':782,'HQ_array_bytes':hq_bytes,
        'HQ_exact_retained_array_hashes':True,'used_behavior_anchors':50,'used_behavior_arrays':100,
        'used_behavior_array_bytes':anchor_bytes,'used_behavior_exact_retained_file_bytes':True,
        'fresh_online_behavior_arrays':100,'online_nonexact_vs_used':dict(online_nonexact),
        'online_arrays_not_used_as_behavior_loss_targets':True,'new_neural_calls':0}


def tree_digest(value, torch):
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):digest.update(b'T'+tensor_digest({'value':item},torch).encode())
        elif isinstance(item,dict):
            digest.update(b'D')
            for key in sorted(item,key=lambda key:(type(key).__name__,str(key))):visit(key);visit(item[key])
        elif isinstance(item,(list,tuple)):
            digest.update(b'L' if isinstance(item,list) else b'U')
            for child in item:visit(child)
        else:digest.update(json.dumps([type(item).__name__,item],allow_nan=False,separators=(',',':')).encode())
    visit(value);return digest.hexdigest()


def inspect_state(state, original, p, expected_sampler, update, torch):
    require(state['format']=='codeformer-normal-tone-training-state-v1' and state['protocol_sha256']==PROTOCOL and
        state['completed_updates']==update and state['schedule']=={'completed_updates':update,'protocol_training':p['training']} and
        state['EMA'] is False and state['no_automatic_resume_supported'] is True and state['resume_safe'] is True and
        state['partial_microbatches_discarded']==0 and state['sampler']==expected_sampler,'Checkpoint committed metadata differs')
    model=state['model'];require(list(model)==list(original) and len(model)==515,'Model key/order/515tensor census differs')
    changes=defaultdict(lambda:{'tensors':0,'changed':0,'max_abs_delta':0.,'delta_squared_l2':0.})
    for name,value in model.items():
        prior=original[name];group=name.split('.')[0]
        require(group in TRAINABLE+FROZEN and isinstance(value,torch.Tensor) and value.device.type=='cpu' and
            value.shape==prior.shape and value.dtype==prior.dtype and torch.isfinite(value).all().item(),
            'CPU state tensor type/geometry/finiteness differs')
        same=torch.equal(value,prior)
        require((group not in FROZEN or same) and (update!=0 or same),'Frozen/initial model changed')
        row=changes[group];row['tensors']+=1;row['changed']+=int(not same)
        if not same:
            delta=value.double()-prior.double();row['max_abs_delta']=max(row['max_abs_delta'],float(delta.abs().max()))
            row['delta_squared_l2']+=float(delta.square().sum())
    require(sum(v['tensors'] for k,v in changes.items() if k in FROZEN)==165 and
        sum(v['tensors'] for k,v in changes.items() if k in TRAINABLE)==350,'Frozen/trainable census differs')
    if update:require(sum(v['changed'] for k,v in changes.items() if k in TRAINABLE)==350,'Some trainable state did not change')
    optimizer=state['optimizer'];require(set(optimizer)=={'state','param_groups'},'Adam state schema differs')
    require([row['name'] for row in optimizer['param_groups']]==['encoder_token','fusion'],'Adam group order differs')
    used=[];steps=Counter();moments=0
    for group in optimizer['param_groups']:
        fusion=group['name']=='fusion';names=[name for name in original if name.split('.')[0] in TRAINABLE and name.startswith('fuse_convs_dict.')==fusion]
        require(len(group['params'])==len(names),'Adam group parameter count differs');used+=group['params']
        rate=p['training']['fusion_lr' if fusion else 'encoder_token_lr']
        close(group['base_lr'],rate);close(group['lr'],rate if update in (0,50) else rate*update/50)
        require(list(group['betas'])==[.9,.99] and group['eps']==1e-8 and group['weight_decay']==0,'Adam hyperparameters differ')
        for identifier,name in zip(group['params'],names):
            entry=optimizer['state'].get(identifier)
            if update==0:require(entry is None,'Initial Adam moments unexpectedly present');continue
            require(set(entry)=={'step','exp_avg','exp_avg_sq'} and float(entry['step'])==50,'Actual saved Adam step differs')
            steps[int(float(entry['step']))]+=1
            for key in ('exp_avg','exp_avg_sq'):
                value=entry[key];require(value.shape==original[name].shape and value.dtype==original[name].dtype and
                    value.device.type=='cpu' and torch.isfinite(value).all().item(),'Adam moment geometry/finiteness differs');moments+=1
            require((entry['exp_avg_sq']>=0).all().item(),'Negative Adam second moment')
    require(len(used)==len(set(used))==350 and set(optimizer['state'])==(set(used) if update else set()),'Adam frozen/duplicate/extra ID')
    rng=state['rng'];require(set(rng)=={'python','numpy','torch_cpu','torch_cuda'} and len(rng['python'])==3 and
        rng['python'][0]==3 and len(rng['python'][1])==625 and rng['numpy']['algorithm']=='MT19937' and
        len(rng['numpy']['keys'])==624 and 0<=rng['numpy']['position']<=624 and len(rng['torch_cuda'])==1,'RNG schema differs')
    for value in [rng['torch_cpu'],*rng['torch_cuda']]:
        require(isinstance(value,torch.Tensor) and value.dtype==torch.uint8 and value.ndim==1 and value.numel()>0 and
            value.device.type=='cpu','CPU/GPU saved RNG array schema differs')
    counts=state['counts'];require(counts['optimizer_updates']==update and counts['microbatches']==4*update and
        counts['backward_calls']==4*update and counts['optimizer_step_attempts']==update,'Checkpoint committed counter differs')
    for key,cap in p['training']['call_caps'].items():require(type(counts[key]) is int and 0<=counts[key]<=cap,'Checkpoint cap differs')
    for row in changes.values():row['delta_l2']=math.sqrt(row.pop('delta_squared_l2'))
    return {'completed_updates':update,'resume_safe_recorded_not_authorized':True,'partial_microbatches':0,
        'model_sha256':tensor_digest(model,torch),'frozen_sha256':tensor_digest({n:v for n,v in model.items() if n.split('.')[0] in FROZEN},torch),
        'optimizer_sha256':tree_digest(optimizer,torch),'RNG_sha256':tree_digest(rng,torch),
        'sampler_sha256':tree_digest(state['sampler'],torch),'schedule_sha256':tree_digest(state['schedule'],torch),
        'groups':dict(changes),'Adam_states':len(optimizer['state']),'Adam_moment_tensors':moments,'Adam_step_histogram':dict(steps),
        'counts':counts,'last_safe_checkpoint':state['last_safe_checkpoint'],'stop_reason':state['stop_reason']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-receipt',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/codeformer_normal_tone_training_v1_return_review/state_v1')
    args=parser.parse_args();started=time.monotonic();e=Evidence(args.import_receipt)
    require(not (args.out/'state_audit_v1.json').exists(),'Preserve previous audit receipt')
    import numpy as np
    metadata=audit_metadata(e,np)
    supervision=audit_supervision_arrays(e,np)
    checkpoint_names=[PREFIX+'snapshot_0000/checkpoint.pt',PREFIX+'snapshot_0050/checkpoint.pt',PREFIX+'stopped_checkpoint.pt']
    paths=[e.get(name) for name in checkpoint_names]
    initializer_path=e.parent(e.p['assets']['initializer']['path'])
    # Every untrusted serialized file has now been regular-file/hash/size bound.
    import torch
    torch.set_num_threads(2)
    original_container=torch.load(initializer_path,map_location='cpu',weights_only=True,mmap=True)
    require(isinstance(original_container,dict) and 'params_ema' in original_container,'Official initializer schema differs')
    original=original_container['params_ema'];checkpoints={};reports={}
    for name,path,update in zip(('initial','update50','stopped'),paths,(0,50,50)):
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        reports[name]=inspect_state(state,original,e.p,metadata['sampler_states'][update],update,torch)
        checkpoints[name]=state
        print(json.dumps({'checkpoint_audited':name,'model_tensors':len(state['model']),'Adam_states':len(state['optimizer']['state'])}),flush=True)
    for field in ('model_sha256','frozen_sha256','optimizer_sha256','RNG_sha256','sampler_sha256','schedule_sha256'):
        require(reports['update50'][field]==reports['stopped'][field],'Snapshot50/stopped committed state differs: '+field)
    require(checkpoints['stopped']['counts']==metadata['failure']['counts'] and
        all(checkpoints['stopped']['counts'][key]==cap for key,cap in e.p['training']['call_caps'].items()),'Stopped terminal counts differ')
    for name,n in [('initial',0),('update50',50)]:
        receipt=metadata['snapshots'][n]['checkpoint']
        require(receipt['path']==checkpoint_names[0 if n==0 else 1] and receipt['sha256']==e.hashes[receipt['path']] and
            receipt['bytes']==e.sizes[receipt['path']] and receipt['resume_safe'] is True,'Snapshot checkpoint pointer differs')
    require(metadata['failure']['checkpoint']['sha256']==e.hashes[checkpoint_names[2]],'Failure stopped checkpoint pointer differs')
    report={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':PROTOCOL,'archive_sha256':ARCHIVE,
        'source_sha256':{Path(__file__).relative_to(ROOT).as_posix():sha(__file__)},'import_root':str(e.root),
        'metadata':metadata['report'],'supervision_arrays':supervision,'checkpoint_readback':reports,'update50_stopped_model_Adam_RNG_sampler_schedule_exact':True,
        'initializer_exact515':True,'frozen_exact165':True,'changed_trainable350':True,
        'checkpoint_load_policy':{'map_location':'cpu','weights_only':True,'mmap':True,'returned_code_executed':False,
            'model_instances':0,'neural_forwards':0,'autograd_queries':0,'optimizer_steps':0,'VM_calls':0},
        'authoritative_outcome':'50committed updates; full clear and blur guards failed; checkpoint is research-only, no automatic resume/promotion',
        'evidence_closure':e.close(),'seconds':time.monotonic()-started,'model_qualified':False,
        'limitations':['Saved states and histories prove consistent committed state, not independent live reconstruction of all CUDA gradients or Adam execution.',
            'Fresh graph/mean differentiation/total divided by4 remain frozen-source-bound; two tiny saved derivative arrays validate only that helper.',
            'Image metrics, target joins and visual usefulness are independently reviewed elsewhere; no inferred hidden identity or native paired quality follows.']}
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'state_audit_v1.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (args.out/'STATE_FINDINGS_V1.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('# Independent normal-tone training saved-state readback\n\n50 genuine committed Adam updates are consistent across200 history exposures,2075 ordered reservations, all350 saved Adam states at step50 and both update50/stopped checkpoints. All515 initializer tensors match the official asset; all165 frozen tensors remain exact and all350 trainable tensors changed. Model, Adam, committed RNG, sampler and schedule agree between update50 and stopped state. No partial microbatch or optimizer-inflight action remains.\n\nThe200 records contain100 normal centered and100 dark full-coarse observations with expected fresh applicable loss namespaces/weights and logged objective arithmetic. The two saved synthetic12-entry gradients pass their prospective analytic bounds. These are CPU saved-value checks; no model, inference, autograd or optimizer ran locally. Full neural-gradient execution remains source-bound.\n\nThe worker stopped on the unchanged clear-preservation gate after50 commits; the retained complete snapshot also fails normal-blur photometric guards. These checkpoints are failed research candidates, not accepted models. No resume or additional epochs are authorized by this audit. Pixel/visual usefulness and historical comparisons require their separate evidence.\n')
    print(json.dumps({'complete':True,'seconds':report['seconds'],'state_audit_sha256':sha(args.out/'state_audit_v1.json')}),flush=True)


if __name__=='__main__':main()
