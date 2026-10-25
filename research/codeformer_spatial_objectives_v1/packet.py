"""Closed packet IO, finite resources and operation receipts; no neural imports."""
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import time
import uuid

VM_NAME = 'cctv_codeformer_spatial_objectives_v1_vm'
O = 'outputs/spatial_objectives_v1/'
MiB, GiB = 1024**2, 1024**3
CAPS = {'prefix':10, 'adapter_train':576, 'adapter_stats':240,
        'full_forward':172, 'adapter_full':688, 'backward':144, 'optimizer':48}
LIMITS = {'return_bytes':512*MiB, 'cache_bytes':512*MiB, 'files':2000,
          'json_bytes':4*MiB, 'append_bytes':16*MiB, 'npz_bytes':2*MiB,
          'array_bytes':40*MiB, 'checkpoint_bytes':2*MiB,
          'disk_reserve_bytes':4*GiB, 'initial_free_bytes':6*GiB,
          'RSS_bytes':10*GiB, 'CUDA_reserved_bytes':18*GiB,
          'worker_seconds':2700, 'supervisor_seconds':2760,
          'archive_bytes':544*MiB, 'whole_supervisor_seconds':3120,
          'whole_deadline_cleanup_seconds':40}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(name):
    require(isinstance(name, str) and name and len(name) <= 240 and
            not re.search(r'[\\:\x00-\x1f]', name), 'Bounded POSIX relative name required')
    parts = PurePosixPath(name).parts
    require(not name.startswith('/') and all(p not in ('', '.', '..') for p in parts)
            and str(PurePosixPath(name)) == name, 'Noncanonical or escaping path refused')
    return name


def regular(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        if p.exists() or p.is_symlink():
            s = p.lstat()
            require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse path refused')
    s = path.stat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Unique regular file required')
    return path.resolve(strict=True)


def directory(path):
    path = Path(path).absolute()
    for p in (path, *path.parents):
        s = p.lstat()
        require(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode)
                and not getattr(s, 'st_file_attributes', 0)&1024, 'Unique physical directory chain required')
    return path.resolve(strict=True)


def safe_path(root, name):
    canonical(name); root = directory(root)
    p = root / name
    for ancestor in (p, *p.parents):
        if ancestor == root.parent:
            break
        if ancestor.exists() or ancestor.is_symlink():
            s = ancestor.lstat()
            require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 1024,
                    'Linked/reparse write path refused')
    require(p.absolute().is_relative_to(root), 'Path escaped packet')
    return p


def sha256(path):
    h = hashlib.sha256()
    with regular(path).open('rb') as f:
        for b in iter(lambda:f.read(MiB), b''):
            h.update(b)
    return h.hexdigest()


def pin(root, path):
    p = regular(path)
    return {'path':p.relative_to(directory(root)).as_posix(),
            'bytes':p.stat().st_size, 'sha256':sha256(p)}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':'))+'\n').encode('utf-8')


def read_json(path):
    p = regular(path)
    require(p.stat().st_size <= 8*MiB, 'Metadata exceeds8MiB')
    def unique(pairs):
        d = {}
        for k, v in pairs:
            require(k not in d, 'Duplicate JSON field'); d[k] = v
        return d
    def finite_float(s):
        value=float(s); require(math.isfinite(value), 'Nonfinite JSON exponent'); return value
    return json.loads(p.read_bytes(), object_pairs_hook=unique, parse_float=finite_float,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def inventory(root):
    root = directory(root); result = {}
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(folder)/name; s = p.lstat()
            require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0)&1024,
                    'Linked/reparse directory refused')
        for name in files:
            p = regular(Path(folder)/name); relative = p.relative_to(root).as_posix()
            canonical(relative); result[relative] = p
            require(len(result) <= LIMITS['files'], 'Finite packet file count exceeded')
    return result


def output_names(p):
    names = {O+n for n in ('started.json','preflight.json','initial.json','result.json','failure.json',
                           'events.jsonl','resources.jsonl')}
    for row in p['train_rows']:
        i = row['id']
        for ext in ('.npz','.json'):
            names.add(O+'shared_features0/'+i+ext)
        for site in ('32','64','128','256'):
            names.add('cache/observed/'+i+'/encoder'+site+'.npy')
    for i in p['parity_ids']:
        for ext in ('.npy','.png','.json'):
            names.add(O+'parity/'+i+ext)
    for arm in ('U','S'):
        names.add(O+'branches/'+arm+'.json')
        for u in (0,6,12,24):
            names.add(O+f'checkpoints/{arm}/u{u:04d}.pt')
        for u in (6,12,24):
            for row in p['train_rows']:
                for ext in ('.npz','.json'):
                    names.add(O+f'stats/{arm}/u{u:04d}/'+row['id']+ext)
            for role in ('guards','strong'):
                for row in p[role]:
                    for ext in ('.npy','.png','.json'):
                        names.add(O+f'{role}/{arm}/u{u:04d}/'+row['id']+ext)
                names.add(O+f'{role}/{arm}/u{u:04d}/bank.json')
                if role == 'guards':
                    names.add(O+f'{role}/{arm}/u{u:04d}/gates.json')
    names.update(('launch.json','supervisor.json','trainer.log','export.json'))
    names.update(n+'.partial' for n in tuple(names)
                 if n not in (O+'events.jsonl',O+'resources.jsonl','trainer.log'))
    return sorted(names)


def verify(root, expected_sha, *, returned=False):
    root = directory(root)
    require(re.fullmatch('[0-9a-f]{64}', expected_sha or ''), 'Exact protocol SHA required')
    require(sha256(root/'protocol.json') == expected_sha, 'Protocol SHA differs')
    p = read_json(root/'protocol.json')
    require(p['vm_name'] == VM_NAME and p['max_calls'] == CAPS and p['limits'] == LIMITS,
            'Exact finite execution contract differs')
    require(p['allowed_outputs'] == output_names(p), 'Output namespace differs')
    require(len(p['train_rows']) == 10 and len(p['clean_features']) == 6 and
            len(p['guards']) == 24 and len(p['strong']) == 4 and len(p['sampler']) == 72 and
            len(p['parity_ids']) == 4 and len(set(p['parity_ids'])) == 4, 'Fixed data counts differ')
    ids = {r['id'] for r in p['train_rows']}
    require(len(ids) == 10 and all(r['split'] == 'TRAIN' and r['input_only_decision'] == 'readable'
                                  for r in p['train_rows']), 'Only admitted readable TRAIN inputs allowed')
    require(all(row['id'] in ids and row['update'] == 1+i//3 and row['slot'] == 1+i%3
                for i,row in enumerate(p['sampler'])), 'Exact24x3 sampler differs')
    observed = inventory(root); allowed = set(p['files']) | set(p['allowed_outputs']) | {'protocol.json'}
    require(set(observed) <= allowed, 'Unlisted files in packet')
    required = set(p['files']) - (set(p['vm_only_static']) if returned else set())
    require(required <= set(observed), 'Required static files missing')
    for name in set(p['files']) & set(observed):
        canonical(name); d = p['files'][name]
        require(set(d) == {'bytes','sha256'} and type(d['bytes']) is int and d['bytes'] > 0 and
                re.fullmatch('[0-9a-f]{64}', d['sha256']), 'Invalid static pin')
        require(observed[name].stat().st_size == d['bytes'] and sha256(observed[name]) == d['sha256'],
                'Static source/data hash differs: '+name)
    require(set(p['vm_only_static']) <= set(p['files']) and
            set(p['return_static']) == set(p['files'])-set(p['vm_only_static']), 'Return static partition differs')
    return p


class Writer:
    def __init__(self, root, protocol):
        self.root = directory(root); self.p = protocol
        self.allowed = set(protocol['allowed_outputs']); self.appends = {}
        self.cache_bytes = 0
        self.bytes = sum((self.root/n).stat().st_size for n in protocol['return_static'])
        self.files = len(protocol['files'])+1
        self.bytes += (self.root/'protocol.json').stat().st_size
        for n,f in inventory(self.root).items():
            if n in self.allowed:
                if n.startswith('cache/'):
                    self.cache_bytes += f.stat().st_size
                else:
                    self.bytes += f.stat().st_size
                self.files += 1
        require(self.bytes <= LIMITS['return_bytes'] and self.cache_bytes <= LIMITS['cache_bytes'],
                'Initial output envelope exceeded')

    def path(self, name):
        require(name in self.allowed, 'Write outside sealed output namespace: '+name)
        return safe_path(self.root,name)

    def _check(self, name, amount, *, fresh=True):
        require(type(amount) is int and amount >= 0, 'Finite byte reservation required')
        cap = LIMITS['cache_bytes'] if name.startswith('cache/') else LIMITS['return_bytes']
        current = self.cache_bytes if name.startswith('cache/') else self.bytes
        require(current+amount <= cap and self.files+int(fresh) <= LIMITS['files'], 'Output/cache cap exceeded')
        require(shutil.disk_usage(self.root).free >= LIMITS['disk_reserve_bytes']+amount,
                'Write would cross4GiB reserve')

    def write(self, name, payload):
        require(type(payload) is bytes, 'Exact bytes required'); p = self.path(name)
        require(not p.exists(), 'Existing artifact cannot be overwritten')
        require(not name.endswith('.partial'), 'Partial names are implementation-reserved')
        temporary=self.path(name+'.partial')
        require(not temporary.exists(), 'Retained partial cannot be overwritten')
        self._check(name,len(payload)); require(self.files+2 <= LIMITS['files'], 'Temporary file count cap')
        p.parent.mkdir(parents=True,exist_ok=True); opened=False
        try:
            with temporary.open('xb') as f:
                opened=True
                require(f.write(payload) == len(payload), 'Short write'); f.flush(); os.fsync(f.fileno())
            # Atomic no-overwrite publication. Both names briefly reference one
            # inode; remove only our exact sibling temporary name before pinning.
            os.link(temporary,p)
            temporary.unlink()
        finally:
            if opened:
                retained=temporary if temporary.exists() else p
                amount=retained.stat().st_size
                if name.startswith('cache/'): self.cache_bytes += amount
                else: self.bytes += amount
                self.files += 1+int(temporary.exists() and p.exists())
        return pin(self.root,p)

    def json(self, name, value):
        b = json_bytes(value)
        small = name.startswith((O+'shared_features0/',O+'stats/',O+'parity/',O+'guards/',O+'strong/'))
        limit = 64*1024 if small else 512*1024
        if name.endswith('bank.json'):limit=256*1024
        if name.endswith('gates.json'):limit=128*1024
        if name.startswith(O+'branches/'):limit=MiB
        if name in (O+'result.json','export.json'):limit=2*MiB
        if name=='launch.json':limit=MiB
        require(len(b) <= limit, 'JSON cap exceeded')
        return self.write(name,b)

    def append(self, name, value):
        require(name in (O+'events.jsonl',O+'resources.jsonl'), 'Only two sealed append journals')
        b = json_bytes(value); p = self.path(name); fresh = name not in self.appends
        require(not p.exists() if fresh else p.stat().st_size == self.appends[name], 'Journal changed or reused')
        require(self.appends.get(name,0)+len(b) <= LIMITS['append_bytes'], 'Journal cap exceeded')
        self._check(name,len(b),fresh=fresh); p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb' if fresh else 'ab') as f:
            require(f.write(b) == len(b), 'Short journal write'); f.flush(); os.fsync(f.fileno())
        self.appends[name] = self.appends.get(name,0)+len(b); self.bytes += len(b); self.files += int(fresh)

    def array(self, name, array):
        import numpy as np
        require(name.endswith('.npy') and not array.dtype.hasobject and np.isfinite(array).all(), 'Finite inert NPY required')
        b = io.BytesIO(); np.save(b,array,allow_pickle=False)
        require(b.tell() <= LIMITS['array_bytes'], 'NPY cap exceeded'); return self.write(name,b.getvalue())

    def npz(self, name, arrays):
        import numpy as np
        require(name.endswith('.npz') and all(isinstance(k,str) and re.fullmatch('[0-9]+_(means|moments|pooled)',k)
                and not v.dtype.hasobject and np.isfinite(v).all() for k,v in arrays.items()), 'Finite named inert arrays required')
        b = io.BytesIO(); np.savez(b,**arrays)
        require(b.tell() <= LIMITS['npz_bytes'], 'NPZ cap exceeded'); return self.write(name,b.getvalue())

    def png(self, name, rgb):
        import numpy as np
        from PIL import Image
        require(name.endswith('.png') and rgb.shape == (256,256,3) and rgb.dtype == np.uint8, 'ExactRGB8 PNG256 required')
        b = io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG')
        require(b.tell() <= 200*1024, 'PNG cap exceeded'); return self.write(name,b.getvalue())

    def checkpoint(self, name, value, torch):
        require(name.endswith('.pt') and isinstance(value,dict) and 'base' not in value, 'Adapter-only checkpoint required')
        b = io.BytesIO(); torch.save(value,b)
        require(b.tell() <= LIMITS['checkpoint_bytes'], 'Small checkpoint cap exceeded')
        return self.write(name,b.getvalue())


class Ledger:
    def __init__(self, writer):
        self.writer=writer; self.counts=dict.fromkeys(CAPS,0); self.returns=dict.fromkeys(CAPS,0)
        self.pending={}; self.sequence=0

    def reserve(self, kind, **context):
        require(kind in CAPS and self.counts[kind] < CAPS[kind], 'Operation cap exceeded: '+str(kind))
        self.counts[kind] += 1; self.sequence += 1; token=self.sequence
        self.writer.append(O+'events.jsonl',{'event':'reserved','token':token,'kind':kind,
                                           'count':self.counts[kind],'context':context})
        self.pending[token]=kind; return token

    def returned(self, token, **context):
        require(token in self.pending, 'Unreserved or repeated return')
        kind=self.pending[token]; self.writer.append(O+'events.jsonl',
            {'event':'returned','token':token,'kind':kind,'context':context})
        self.returns[kind] += 1; del self.pending[token]


class Resources:
    def __init__(self, root, writer):
        self.root=Path(root); self.writer=writer; self.started=time.monotonic(); self.torch=None
        self.samples=0

    def check(self, context):
        require(platform.system() == 'Linux', 'Runtime resources require Linux self measurement')
        import resource
        rss=None
        for line in Path('/proc/self/status').read_text(encoding='ascii').splitlines():
            if line.startswith('VmRSS:'): rss=int(line.split()[1])*1024
        peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        seconds=time.monotonic()-self.started; free=shutil.disk_usage(self.root).free
        gpu=0 if self.torch is None else self.torch.cuda.memory_reserved(0)
        require(rss is not None, 'Self RSS missing')
        row={'context':context,'seconds':seconds,'RSS_bytes':rss,'peak_RSS_bytes':peak,
             'CUDA_reserved_bytes':gpu,'free_bytes':free,'sample':self.samples+1}
        self.writer.append(O+'resources.jsonl',row); self.samples+=1
        require(seconds <= LIMITS['worker_seconds'] and max(rss,peak) <= LIMITS['RSS_bytes'] and
                gpu <= LIMITS['CUDA_reserved_bytes'] and free >= LIMITS['disk_reserve_bytes'],
                'Finite runtime memory/time/disk limit exceeded')
        return row

    @contextmanager
    def stage(self, name, seconds):
        require(type(seconds) is int and seconds > 0 and hasattr(signal,'setitimer'), 'Positive Linux stage alarm required')
        old=signal.getsignal(signal.SIGALRM); timer=signal.getitimer(signal.ITIMER_REAL); start=time.monotonic()
        def expired(sig,frame): raise TimeoutError('Stage time cap exceeded: '+str(name))
        signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,min(seconds,timer[0]) if timer[0] else seconds)
        try: yield
        finally:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)
            if timer[0]: signal.setitimer(signal.ITIMER_REAL,max(.000001,timer[0]-(time.monotonic()-start)),timer[1])

    def project(self, name, elapsed, completed, total, cap):
        require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (elapsed,completed,total,cap))
                and completed <= total, 'Finite timing projection required')
        projection=elapsed/completed*total
        self.writer.append(O+'resources.jsonl',{'context':'projection','name':name,'seconds':elapsed,
                           'completed':completed,'total':total,'projected_seconds':projection,'cap_seconds':cap})
        require(projection <= cap, 'Timing projection exceeds finite '+str(name)+' cap')
        return projection
