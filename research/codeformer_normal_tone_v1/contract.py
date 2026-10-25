"""Standard-library-only prospective diagnostic contract. Import executes no model."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

FORMAT = 'codeformer-normal-tone-v1'
CASES_FORMAT = 'codeformer-loss-response-cases-v1'
VM_NAME = 'cctv_codeformer_normal_tone_v1_vm'
PARENT_NAME = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
PARENT_PROTOCOL_SHA = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
SOURCE_UPDATES = [1, 6, 15, 24]
COMPONENTS = ['token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
              'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl']
WEIGHTS = dict(zip(COMPONENTS, [.5, 1., 1., 1., 4., 2., .1]))
BRANCHES = ['baseline', 'full', 'omit_normal_coarse', 'center_normal_luminance', 'momentum_only']
ROOTS = {'adain': {'directory':'cctv_codeformer_adain_gradient_isolation_v1_vm',
                  'protocol_sha256':'4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'},
         'loss_response': {'directory':'cctv_codeformer_loss_response_v1_vm',
                  'protocol_sha256':'23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab'}}
LOSS_RESPONSE_RETURN_MANIFEST_SHA='b1b61858b9c9354972cc89146829d6f1cf387cec19e66ac9e16b115fcdd4c4e1'
COUNTS = {'candidate_source_forwards':32,'candidate_probe_forwards':160,
          'restoration_repeat_forwards':32,'coarse_gradient_queries':32,
          'centered_normal_gradient_queries':16,'optimizer_step_attempts':8,
          'disposable_optimizer_steps':8,'committed_training_updates':0,'backward_calls':0}
LIMITS = {'worker_seconds':1800,'initialization_seconds':300,'source_seconds':600,
          'probe_seconds':900,'export_seconds':600,'preflight_seconds':300,
          'reserve_free_bytes':4*1024**3,'required_free_bytes':10*1024**3,
          'result_payload_bytes':2*1024**3,'archive_bytes':9*1024**3//4,
          'metadata_bytes':128*1024**2,'regular_file_cap':5000,
          'CUDA_reserved_bytes':22*1024**3,'RSS_bytes':8*1024**3,
          'array_file_bytes':256*1024**2,'PNG_file_bytes':3*512*512+4096,'log_bytes':2*1024**2}
OPTIMIZER = {'name':'Adam','betas':[.9,.99],'eps':1e-8,'weight_decay':0.,
             'fixed_group_rates':{'encoder_token':1e-5,'fusion':2e-5},
             'gradient_accumulation_denominator':16,
             'rate_policy':'Recorded parent update50 rates, identical at both measurement states; no scheduler or continuation',
             'branch_restore_required':True,'chained_steps':False,'None_is_not_zero':True}
NUMERICAL = {'fresh_all_coarse_cached_maximum_error':0.0,
             'fixed_state_repeat_maximum_error':0.0,'prior_control_maximum_error':0.0,
             'quality_thresholds_changed':False}
OBJECTIVE = {'prediction_RGB':'(native.float()+1)/2; finite unclipped native512',
             'grid':'avg_pool2d kernel32 stride32 ->16x16',
             'original_coarse':'mean(abs(P(rgb)-P(clean512)))',
             'luminance_weights':[.2126,.7152,.0722],
             'normal_centered':'mean(abs(e-mean_spatial(sum_RGB(luminance_weights*e))))',
             'mean_scope':'one differentiable scalar per observation, broadcast equally to all3RGB channels',
             'normal_slots':[1,3],'dark_slots':[2,4],
             'dark_coarse_unchanged':True,'coarse_weight':4.0,'aggregation_divisor':16,
             'accumulation':'canonical16case order, direct subsequences; no aggregate subtraction or regrouping',
             'inference_output_centering':False,'new_loss_weight_grid':False}


def cached_basis_paths():
    return {state: {term: {'root':'loss_response', 'manifest_path':
        'outputs/loss_response_v1/'+state+'/gradients/'+term+'/manifest.json'}
        for term in COMPONENTS} for state in ('original0','stopped50')}


def prior_controls():
    return {state: {condition: {'root':'loss_response',
        'branch_path':'outputs/loss_response_v1/'+state+'/conditions/'+condition+'/branch.json',
        'outputs_root':'outputs/loss_response_v1/'+state+'/conditions/'+condition}
        for condition in ('full','momentum_only')} for state in ('original0','stopped50')}


def required_loss_response_paths(old_protocol, old_manifest):
    """Exact cached/support namespace, derived without reading array payloads."""
    selected=set(old_protocol['files_sha256']) | {'protocol.json','outputs/loss_response_v1/export_manifest.json'}
    target_prefix='outputs/loss_response_v1/targets/'
    target_pattern=re.compile(re.escape(target_prefix)+r'(tr_ffhq_[0-9]{5})/clean256\.npy\Z')
    target_refs={match.group(1) for name in old_manifest['files_sha256']
                 for match in [target_pattern.fullmatch(name)] if match is not None}
    require(len(target_refs)==8,'Exactly8 prior TRAIN metric target references required')
    for reference in sorted(target_refs):
        for suffix in ('clean256.npy','parity.json'):
            name=target_prefix+reference+'/'+suffix
            require(name in old_manifest['files_sha256'] and name in old_manifest['files_bytes'],
                    'Prior metric target/parity pair missing')
            selected.add(name)
    selected.update('outputs/loss_response_v1/'+name for name in
                    ('completion.json','initialization.json','resource_plan.json'))
    for state in ('original0','stopped50'):
        prefix='outputs/loss_response_v1/'+state+'/'
        selected.update(prefix+name for name in ('source_summary.json','snapshot_complete.json'))
        for term in COMPONENTS:
            folder=prefix+'gradients/'+term+'/'
            selected.add(folder+'manifest.json')
            selected.update(name for name in old_manifest['files_sha256'] if name.startswith(folder) and name.endswith('.npy'))
        for update in SOURCE_UPDATES:
            for slot in range(1,5):
                case='u%04d_s%d' % (update,slot)
                selected.update(prefix+'sources/'+case+'/'+name for name in
                    ('native512.npy','raw256.npy','result.png','tokens.npy','logits.npy','features.npy','record.json'))
                for condition in ('full','momentum_only'):
                    selected.update(prefix+'conditions/'+condition+'/'+case+'/'+name for name in
                        ('native512.npy','raw256.npy','result.png','tokens.npy','record.json'))
        selected.update(prefix+'conditions/'+condition+'/branch.json' for condition in ('full','momentum_only'))
    return selected


def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(1024**2), b''):
            if budget is not None:
                budget()
            digest.update(payload)
    return digest.hexdigest()

def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def no_links(path):
    path = Path(path).absolute()
    for part in [path] + list(path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 1024),
                    'Linked/reparse path forbidden: ' + str(part))
    return path

def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value and '\x00' not in value,
            'Portable relative path required')
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and relative.as_posix() == value and
            all(part not in ('', '.', '..') and not part.endswith((' ', '.')) and
                not re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part)
                for part in relative.parts), 'Unsafe relative path')
    return relative

def safe_path(base, relative, must_exist=True):
    base = no_links(base).resolve(strict=True)
    parts = safe_relative(relative).parts
    path = no_links(base.joinpath(*parts))
    require(path.resolve(strict=must_exist).is_relative_to(base), 'Path escapes declared root')
    if must_exist:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode), 'Exact regular file required')
    return path

def write_new_json(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())

def manual_root(root):
    require(platform.system() == 'Linux', 'Manual NVIDIA L4 VM execution only; local neural calls forbidden')
    root = no_links(root).resolve(strict=True)
    expected = no_links(Path.home() / 'forensic-dgp' / VM_NAME).resolve(strict=False)
    require(root == expected, 'Unexpected manual VM root')
    return root

def resolve_parents(root):
    root = no_links(root).resolve(strict=True)
    parents = {}
    for alias, declaration in ROOTS.items():
        parent = no_links(root.parent / declaration['directory']).resolve(strict=True)
        require(parent.is_dir() and parent.parent == root.parent and parent != root,
                'Exact read-only sibling root required: ' + alias)
        parents[alias] = parent
    require(len(set(parents.values())) == 2, 'Distinct read-only roots required')
    return parents


def verify_protocol(p):
    def same(a,b):
        return json.dumps(a,sort_keys=True,allow_nan=False,separators=(',',':')) == \
               json.dumps(b,sort_keys=True,allow_nan=False,separators=(',',':'))
    require(p.get('format') == FORMAT and p.get('parent_roots') == ROOTS,
            'Protocol or exact two-root identities differ')
    require(type(p.get('seed')) is int and p.get('seed') == 20261015 and
            p.get('outputs') == 'outputs/normal_tone_v1',
            'Fixed seed/output namespace differs')
    require(same(p.get('source_updates'), SOURCE_UPDATES) and p.get('cases_file') == 'cases.json' and
            p.get('snapshots') == ['original0','stopped50'] and p.get('branches') == BRANCHES,
            'Fixed diagnostic census differs')
    require(same(p.get('loss_weights'),WEIGHTS) and same(p.get('optimizer'),OPTIMIZER) and
            same(p.get('numerical'),NUMERICAL) and same(p.get('objective'),OBJECTIVE) and
            same(p.get('limits'),LIMITS) and same(p.get('call_caps'),COUNTS),
            'Frozen mathematical/resource controls differ')
    require(type(p.get('total_neural_forwards')) is int and p.get('total_neural_forwards') == 224 and
            type(p.get('total_gradient_queries')) is int and p.get('total_gradient_queries') == 48,
            'Finite neural/gradient census differs')
    require(p.get('cached_basis')==cached_basis_paths() and p.get('prior_controls')==prior_controls(),
            'Cached basis/prior control joins differ')
    for key in ('automatic_app_promotion','model_qualified','new_training_recipe',
                'export_candidate_checkpoint','quality_gates_changed'):
        require(p.get(key) is False, 'Diagnostic cannot authorize promotion/training: '+key)
    require(p.get('manual_tmux_required') is True and p.get('all_branches_disposable') is True and
            p.get('status') == 'PREPARED_UNRUN', 'Manual disposable unrun status required')
    own,sizes=p.get('files_sha256'),p.get('files_bytes')
    require(isinstance(own,dict) and isinstance(sizes,dict) and set(own)==set(sizes) and
            'protocol.json' not in own and 'cases.json' in own,'Exact own-file manifest required')
    for name,value in own.items():
        safe_relative(name)
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and
                type(sizes[name]) is int and sizes[name]>=0,'Invalid static fingerprint')
    deps=p.get('dependencies')
    require(isinstance(deps,dict) and set(deps)==set(ROOTS),'Exact two-root dependency maps required')
    for alias,mapping in deps.items():
        require(isinstance(mapping,dict) and 'protocol.json' in mapping,'Parent protocol required')
        for name,value in mapping.items():
            safe_relative(name)
            require(set(value)=={'sha256','bytes'} and isinstance(value['sha256'],str) and
                    re.fullmatch('[0-9a-f]{64}',value['sha256']) and type(value['bytes']) is int and
                    value['bytes']>=0,'Invalid parent dependency binding')
        require(mapping['protocol.json']['sha256']==ROOTS[alias]['protocol_sha256'],'Parent protocol hash differs')
    require(len(deps['adain'])==2011,'Exact AdaIN dependency census required')
    require(deps['loss_response'].get('outputs/loss_response_v1/export_manifest.json',{}).get('sha256')==
            LOSS_RESPONSE_RETURN_MANIFEST_SHA,'Exact completed prior return manifest required')
    assets=p.get('assets')
    require(isinstance(assets,dict) and set(assets)=={'initializer','teacher','perceptual_features',
            'perceptual_calibration','original_checkpoint','stopped_checkpoint'},'Exact six assets required')
    for value in assets.values():
        require(isinstance(value,dict) and set(value)=={'root','path','sha256','bytes'} and
                type(value['bytes']) is int and value.get('root')=='adain' and value.get('path') in deps['adain'] and
                {k:value[k] for k in ('sha256','bytes')}==deps['adain'][value['path']],
                'Asset must bind exact AdaIN dependency')
    return p


def verify_cases(document):
    require(document.get('format') == CASES_FORMAT and document.get('source_updates') == SOURCE_UPDATES,
            'Case role/selection differs')
    rows = document.get('cases')
    require(isinstance(rows, list) and len(rows) == 16, 'Exactly16 TRAIN source/probe observations required')
    expected = [(update, slot) for update in SOURCE_UPDATES for slot in (1,2,3,4)]
    require([(r['update'], r['slot']) for r in rows] == expected, 'Canonical case order differs')
    for row in rows:
        require(row['id'] == 'u%04d_s%d' % (row['update'], row['slot']) and
                row['record']['role'] == 'train' and row['reference_id'].startswith('tr_ffhq_') and
                row['reference_id'] == row['record']['reference_id'] and
                row['update'] == row['record']['update'] and row['slot'] == row['record']['slot'] and
                row['profile'] == row['record']['profile'] and
                row['source_path'] == row['record']['source_path'] and
                row['parameters'] == row['record']['parameters'] and
                row['parameters']['behavior_anchor'] == (row['slot'] == 3), 'TRAIN role/case metadata differs')
        prefix = 'outputs/adain_gradient_isolation_v1/training_inputs/update_%04d/slot_%d/' % (row['update'],row['slot'])
        require(row['input_path'] == prefix + 'input.png' and
                row['pair_record_path'] == prefix + 'pair_record.json' and
                row['source_path'] == 'data/sources/' + row['reference_id'] + '.png',
                'Canonical source/input path join differs')
        hq = 'outputs/adain_gradient_isolation_v1/HQ_cache/' + row['reference_id'] + '/'
        require(row['hq_indices_path'] == hq + 'indices.npy' and row['hq_features_path'] == hq + 'features.npy' and
                row['hq_record_path'] == hq + 'record.json', 'Canonical HQ cache join differs')
        behavior_keys = {'behavior_delivered_path','behavior_logits_path','behavior_record_path'}
        require(behavior_keys.issubset(row) if row['slot'] == 3 else not behavior_keys.intersection(row),
                'Behavior target support differs')
        if row['slot'] == 3:
            anchor = 'outputs/adain_gradient_isolation_v1/behavior_anchors/update_%04d/' % row['update']
            require(row['behavior_delivered_path'] == anchor + 'delivered256.npy' and
                    row['behavior_logits_path'] == anchor + 'logits.npy' and
                    row['behavior_record_path'] == anchor + 'record.json', 'Canonical behavior path join differs')
        for key, value in row.items():
            if key.endswith('_path'):
                safe_relative(value)
    require(sum(bool(r['parameters']['image_eligible']) for r in rows) == 12 and
            len({r['reference_id'] for r in rows}) == 8 and
            len({r['id'] for r in rows}) == 16, 'Eligibility/reference census differs')
    require([r['parameters']['variant'] for r in rows if r['slot'] == 1] ==
            ['severe24','severe28','severe32','severe40'], 'Severe-source coverage differs')
    require([r['parameters']['variant'] for r in rows if r['slot'] == 3] ==
            ['clear256','clear128','light256','light64'], 'Protected-control coverage differs')
    return document
