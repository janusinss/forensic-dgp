"""Inert exact-byte packet checks for one manual L4 zero-optimizer preflight."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import sys

STAGE_NAME='cctv_codeformer_u96_postfusion_preflight_v1_vm'
FORMAT='complete-U96-postfusion-zero-optimizer-preflight-v1'
MAX_FILE=512*1024**2
LIMITS={'full_forward_attempts':20,'full_forward_returns':20,'reference_forwards':8,
        'candidate_parity_forwards':8,'candidate_gradient_forwards':4,'gradient_queries':4,
        'optimizer_constructions':0,'optimizer_updates':0,'internal_seconds':1200,
        'external_seconds':1260,'per_phase_seconds':180,'initial_free_disk_bytes':6*1024**3,
        'continuing_disk_reserve':4*1024**3,'maximum_RSS_bytes':10*1024**3,
        'maximum_CUDA_reserved_bytes':18*1024**3,'maximum_output_bytes':128*1024**2}
FIXED={
    'model':{
        'r2':(376609465,'d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'),
        'u96':(212827,'958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0')},
    'prototype':{
        'source':(23240,'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'),
        'design':(15781,'cf4ce39beefa2bef41149de6399f4f8b86cac3bdddf5ab6818651631ef0567be'),
        'review':(4966,'5da647e31ea3c6f6f635ece331943291aed94e043adfe502438541d9b3471ab4')},
    'data':{
        'provenance':(18696,'79251871475cfa66bf25ef47c89f0db066f1cf69eba3cce8c50cea10faffba17'),
        'admission':(7986,'8cba1255fb67cabc760551aa4ece041884de6f1ff7a9ff613b832f785c6cb891'),
        'review':(2056,'43ef8acfc0017ec65544c093ad783972ee4521709d95b7972782c99895dcf458'),
        'audit':(5779,'3609b43b1b37e7b800315b93409fb6c561aeaa1bf7fca7f90ef8fa1343bac18a')}}
REFS=('tr_ffhq_00084','tr_ffhq_00178','tr_ffhq_00323','tr_ffhq_00616')


def need(ok,message):
    if not ok:raise ValueError(message)


def regular(path):
    path=Path(path).absolute()
    for item in (path,*path.parents):
        info=item.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&1024,
             'Linked/reparse packet path refused: '+str(item))
    info=path.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_size<=MAX_FILE,
         'Bounded single-link ordinary file required: '+str(path))
    return path.resolve(strict=True)


def sha256(path):
    digest=hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def pin(root,path):
    root=directory(root);path=regular(path)
    need(path.is_relative_to(root),'File must remain below packet root')
    before=path.stat();digest=sha256(path);after=path.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'File changed during hashing')
    return {'path':path.relative_to(root).as_posix(),'bytes':after.st_size,'sha256':digest}


def strict_json(data):
    def pairs(items):
        result={}
        for key,value in items:
            need(key not in result,'Duplicate JSON key');result[key]=value
        return result
    value=json.loads(data.decode('utf-8'),object_pairs_hook=pairs,
                     parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON refused')))
    def finite(item):
        if type(item) is float:need(math.isfinite(item),'Finite JSON number required')
        elif type(item) is dict:
            for entry in item.values():finite(entry)
        elif type(item) is list:
            for entry in item:finite(entry)
    finite(value);return value


def directory(path):
    path=Path(path).absolute()
    for item in (path,*path.parents):
        info=item.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             not getattr(info,'st_file_attributes',0)&1024,'Ordinary unlinked directory required: '+str(item))
    return path.resolve(strict=True)


def relative(path):
    need(type(path) is str and path and '\\' not in path and ':' not in path,'Canonical relative POSIX path required')
    p=PurePosixPath(path)
    need(bool(p.parts) and path!='.' and not p.is_absolute() and p.as_posix()==path and all(x not in ('','..','.') for x in p.parts),'Packet path escape refused')
    return p


def verify(root,expected_sha):
    root=directory(root);protocol_path=regular(root/'protocol.json')
    need(type(expected_sha) is str and len(expected_sha)==64 and sha256(protocol_path)==expected_sha,'Exact supplied protocol SHA required')
    p=strict_json(protocol_path.read_bytes())
    need(p['format']==FORMAT and p['stage_name']==STAGE_NAME,'Known finite preflight protocol required')
    need(type(p['optimizer_updates']) is int and type(p['optimizer_constructions']) is int and
         p['optimizer_updates']==p['optimizer_constructions']==0,'This packet permits no optimizer')
    need(p['limits']==LIMITS and all(type(v) is int for v in p['limits'].values()),'Exact finite resource/call/time limits required')
    declared=set(p['files']);need('protocol.json' not in declared,'No recursive protocol self-pin')
    allowed_dirs={'outputs'}
    for name,row in p['files'].items():
        relative(name);need(type(row) is dict and set(row)=={'bytes','sha256'},'Exact static file binding required')
        need(type(row['bytes']) is int and row['bytes']>=0 and type(row['sha256']) is str and
             len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256']),'Exact byte count/SHA256 types required')
        actual=pin(root,root/name);need(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'Static packet file differs: '+name)
        allowed_dirs.update(parent.as_posix() for parent in PurePosixPath(name).parents if parent.as_posix()!='.')
    for item in root.rglob('*'):
        if item.is_dir():
            directory(item);name=item.relative_to(root).as_posix()
            need(name in allowed_dirs or name.startswith('outputs/'),'Unexpected directory namespace: '+name);continue
        name=item.relative_to(root).as_posix();regular(item)
        allowed=name in declared or name=='protocol.json' or name.startswith('outputs/') or name in {'run.log','supervisor.json','export.json'}
        need(allowed,'Unexpected packet namespace: '+name)
    witnesses=p['witnesses'];need(len(witnesses)==8 and len({x['id'] for x in witnesses})==8,'Exactly8unique witnesses required')
    need([(x['reference_id'],x['role'],x['id']) for x in witnesses]==
         [(ref,role,ref+'__'+role) for ref in REFS for role in ('clear','local_gaussian_1p2')],
         'Fixed four-reference clear-then-local witness order required')
    need(len({x['reference_id'] for x in witnesses})==4 and sum(x['gradient_witness'] is True for x in witnesses)==4,'Exactly4 TRAIN photographs/gradient witnesses required')
    need(all(type(x['gradient_witness']) is bool and x['role'] in ('clear','local_gaussian_1p2') and x['gradient_witness']==(x['role']=='local_gaussian_1p2') for x in witnesses),'Frozen clear/local roles required')
    for ref in {x['reference_id'] for x in witnesses}:
        pair=[x for x in witnesses if x['reference_id']==ref]
        need(len(pair)==2 and {x['role'] for x in pair}=={'clear','local_gaussian_1p2'},'Exactly one clear/local pair per TRAIN reference required')
        need(pair[0]['target']==pair[1]['target'],'Paired reference must use one exact clean target')
        clear=next(x for x in pair if x['role']=='clear')
        need(clear['input']==clear['target'],'Clear control must equal exact target file')
    for row in witnesses:
        for key in ('input','target'):
            b=row[key];need(b['path'] in declared and b==pin(root,root/b['path']),'Exact witness asset join required')
    for section in ('model','prototype','data'):
        need(type(p[section]) is dict and set(p[section])==set(FIXED[section]),'Exact '+section+' section keys required')
        for name,binding in p[section].items():
            need(type(binding) is dict and set(binding)=={'path','bytes','sha256'},'Exact direct '+section+' pin required: '+name)
            need(binding['path'] in declared and binding==pin(root,root/binding['path']),'Direct metadata/source/model pin must join declared files')
            need((binding['bytes'],binding['sha256'])==FIXED[section][name],'Exact reviewed artifact required: '+section+'/'+name)
    need('torch' not in sys.modules,'Packet verification must remain inert')
    return p


def require_vm(root,p):
    root=directory(root)
    need(platform.system()=='Linux' and sys.version_info>=(3,10),'Manual Linux Python3.10+ runtime required')
    expected=directory(Path.home()/'forensic-dgp'/STAGE_NAME)
    need(root==expected and directory(Path.cwd())==root,'Exact home/stage and working directory required')
    need(bool(os.environ.get('TMUX')),'Launch manually inside tmux')
    need(shutil.disk_usage(root).free>=p['limits']['initial_free_disk_bytes'],'Initial6GiB disk reserve required')
    need(not (root/'outputs/u96_postfusion_preflight_v1/started.json').exists(),'Prior started run is immutable; no automatic rerun')
    return {'system':platform.system(),'python':platform.python_version(),'root':str(root),'tmux':True,
            'initial_free_disk_bytes':shutil.disk_usage(root).free,'actual_training_permitted':False}
