"""Finite local, inference-only HQ-code intervention. Never trains or promotes.

Clean-reference codes are deliberately unavailable at deployment. This causal
diagnostic compares natural and oracle codes at identical R2 conditioning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs/codeformer_hq_code_oracle_inputs_20261019_v1'
OUT = ROOT / 'outputs/codeformer_hq_code_oracle_local_20261019_v1_r1'
R2 = ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth'
HQ = ROOT / 'outputs/codeformer_teacher_pretrained_v1/vqgan_code1024.pth'
R2_SHA = 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
HQ_SHA = '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f'
MANIFEST_SHA = 'cdb70da025284909c96144fd1884464cf89fea6c7f4167e9c5d99bd834dbc2ed'
ARCH = {'codeformer_arch.py':'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb',
        'vqgan_arch.py':'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf',
        'registry.py':'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(path):
    path = Path(path).absolute()
    need(path.resolve(strict=False).is_relative_to(ROOT), 'Workspace path required')
    for p in (path, *path.parents):
        if p.exists() or p.is_symlink():
            value = p.lstat()
            need(not stat.S_ISLNK(value.st_mode) and not getattr(value, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse paths refused')
    if path.exists() and path.is_file():
        # Historical official teacher has three retained read-only hardlinks.
        # It is never edited; its exact bytes are rehashed before and after.
        need(path.stat().st_nlink == 1 or path == HQ, 'Singly linked evidence required except exact read-only HQ teacher')
    return path


def pin(path):
    path = safe(path); before = path.stat()
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'File changed during hashing')
    return {'bytes': before.st_size, 'sha256': digest}


def new_json(path, value):
    safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def read(path):
    need(safe(path).stat().st_size <= 4 * 1024**2, 'Bounded JSON required')
    return json.loads(path.read_bytes())


def verify():
    import psutil  # Required process/memory guards must load before any output/spawn.
    need(psutil.__version__ == '7.0.0', 'Pinned local psutil7.0.0 runtime required')
    need(pin(INPUT / 'manifest_v1.json')['sha256'] == MANIFEST_SHA, 'Exact frozen input manifest required')
    manifest = read(INPUT / 'manifest_v1.json'); review = read(INPUT / 'root_input_review_v1_r1.json')
    need(review['complete'] is True and review['manifest_sha256'] == MANIFEST_SHA and
         review['inference_only_diagnostic_admitted'] is True and review['training_admitted'] is False,
         'Root input-only diagnostic admission required')
    need(len(manifest['cases']) == 18 and len(manifest['targets']) == 6 and
         {r['id'] for r in manifest['cases']} == {r['id'] for r in review['cases']}, 'Exact18 reviewed cases required')
    evidence = {}
    for name, digest in ((R2, R2_SHA), (HQ, HQ_SHA), *[(ROOT/'third_party/codeformer'/n,s) for n,s in ARCH.items()]):
        value = pin(name); need(value['sha256'] == digest, 'Input model/architecture changed')
        evidence[str(name.relative_to(ROOT)).replace('\\','/')] = value
    for row in manifest['cases']:
        need(row['split'] == 'TRAIN', 'TRAIN-only input required')
        for key in ('input', 'target_HQ512', 'metrics_target256'):
            r = row[key]; value = pin(ROOT/r['path'])
            need(value == {k:r[k] for k in ('bytes','sha256')}, 'Input/target changed')
            evidence[r['path']] = value
    for name in ('manifest_v1.json','plan_v1.json','root_input_review_v1_r1.json'):
        evidence[str((INPUT/name).relative_to(ROOT)).replace('\\','/')] = pin(INPUT/name)
    design = ROOT / 'outputs/codeformer_filter_fusion_next_design_20261019_v1/oracle_design_frozen_v1.json'
    need(read(design)['census']['total_top_level_model_calls']==43,'Exact prospective diagnostic design required')
    evidence[str(design.relative_to(ROOT)).replace('\\','/')] = pin(design)
    evidence[str(Path(__file__).relative_to(ROOT)).replace('\\','/')] = pin(__file__)
    return manifest, review, evidence


def tensor_digest(state, torch):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        header = json.dumps([name,str(t.dtype),list(t.shape)],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'big')); h.update(header); h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def stop_owned_worker_tree(child_pid, child_started):
    """Bound a timeout across Windows venv redirectors; never kill by name."""
    import psutil
    try: parent = psutil.Process(child_pid)
    except psutil.NoSuchProcess: return []
    need(parent.create_time() == child_started, 'Timeout child PID was reused; refuse unrelated process')
    processes = list(reversed(parent.children(recursive=True))) + [parent]
    verified = []
    for process in processes:
        try:
            command = process.cmdline()
            paths = [Path(word).absolute() for word in command if word.endswith('.py')]
            need(Path(__file__).absolute() in paths and '--worker' in command,
                 'Timeout descendant does not match this exact worker; refuse unrelated process')
            verified.append((process, process.create_time()))
        except psutil.NoSuchProcess: pass
    stopped=[]
    for process, created in verified:
        try:
            need(process.create_time()==created,'Timeout descendant PID changed')
            process.kill(); stopped.append(process.pid)
        except psutil.NoSuchProcess: pass
    _,alive=psutil.wait_procs([p for p,_ in verified],timeout=15)
    need(not alive, 'Owned timed-out worker did not terminate')
    return stopped


def worker():
    manifest, review, evidence = verify()
    launch = read(OUT/'launch.json')
    import psutil
    # Windows venv redirectors can introduce an intermediate launcher process.
    need(launch['source_sha256'] == pin(__file__)['sha256'] and
         launch['parent_pid'] in {p.pid for p in psutil.Process().parents()},
         'Exact live supervisor parent required')
    need(not (OUT/'worker_started.json').exists(), 'Preserve original/partial diagnostic; no retry')
    start = time.monotonic(); calls = 0
    new_json(OUT/'worker_started.json', {'pid':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat()})
    sys.path.insert(0,str(ROOT))
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer
    from third_party.codeformer.vqgan_arch import VQAutoEncoder
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(20261019); torch.use_deterministic_algorithms(True)
    process = psutil.Process(); resources = {'maximum_RSS_bytes':0,'minimum_free_disk_bytes':shutil.disk_usage(ROOT).free}
    def check():
        rss = process.memory_info().rss; free = shutil.disk_usage(ROOT).free
        resources['maximum_RSS_bytes'] = max(resources['maximum_RSS_bytes'],rss)
        resources['minimum_free_disk_bytes'] = min(resources['minimum_free_disk_bytes'],free)
        need(time.monotonic()-start <= 900 and free >= 4*1024**3 and rss <= 10*1024**3,
             'Finite900s/4GiB disk reserve/10GiB RSS stop')
    def operation(label, net, x):
        nonlocal calls
        check(); need(calls < 43,'Exact43 top-level neural invocation ceiling')
        calls += 1
        with (OUT/'operations.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps({'event':'attempt','number':calls,'label':label})+'\n'); f.flush(); os.fsync(f.fileno())
        y = net(x)
        check()
        with (OUT/'operations.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps({'event':'return','number':calls,'label':label})+'\n'); f.flush(); os.fsync(f.fileno())
        return y
    def rgb(path, size):
        with Image.open(safe(ROOT/path)) as im:
            need(im.mode=='RGB' and im.size==(size,size),'Exact RGB input geometry required')
            return np.asarray(im).copy()
    def tensor(pixels):
        return torch.from_numpy(pixels.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
    files = {}
    def array(name, a):
        path = OUT/name; safe(path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f: np.save(f,a,allow_pickle=False)
        files[name] = pin(path); return name
    def image(name, a):
        path = OUT/name; safe(path); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f: Image.fromarray(a).save(f,format='PNG')
        files[name] = pin(path); return name
    def output(prefix, native):
        need(native.shape==(1,3,512,512) and torch.isfinite(native).all().item(),'Finite native512 required')
        raw = F.interpolate(((native.float()+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False)[0].permute(1,2,0).contiguous().numpy().copy()
        array(prefix+'-raw.npy',raw); image(prefix+'.png',np.floor(raw*np.float32(255)).astype(np.uint8))
        return raw
    r2_saved = torch.load(R2,weights_only=True,map_location='cpu',mmap=True)['params_ema']
    hq_saved_all = torch.load(HQ,weights_only=True,map_location='cpu',mmap=True)
    need(set(hq_saved_all)=={'params_ema'},'Exact HQteacher loader key required')
    hq_saved = hq_saved_all['params_ema']
    r2 = CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=['32','64','128','256'])
    teacher = VQAutoEncoder(img_size=512,nf=64,ch_mult=[1,2,2,4,4,8],quantizer='nearest',res_blocks=2,attn_resolutions=[16],codebook_size=1024)
    r2.load_state_dict(r2_saved,strict=True); teacher.load_state_dict(hq_saved,strict=True)
    r2.float().eval().requires_grad_(False); teacher.float().eval().requires_grad_(False)
    # The historical V1/V2 contract accepts tiny FP32 EMA drift, not byte
    # equality, and explicitly verifies all 1024 codebook row identities.
    book = r2_saved['quantize.embedding.weight'].double()
    teacher_book = hq_saved['quantize.embedding.weight'].double()
    distances = ((teacher_book**2).sum(1)[:,None] + (book**2).sum(1)[None,:] - 2*teacher_book@book.T).clamp_min(0)
    nearest = distances.topk(2,largest=False,dim=1)
    maximum_book_error = float((book-teacher_book).abs().max().item())
    need(maximum_book_error <= 1e-4 and torch.equal(nearest.indices[:,0],torch.arange(1024)),
         'Historical bounded codebook drift and exact1024 ordered nearest-row identities required')
    compatibility = {'byte_exact_codebook':torch.equal(book,teacher_book),'maximum_codebook_absolute_error':maximum_book_error,
        'nearest_rows_identity':1024,'minimum_second_minus_first_squared_distance':float((nearest.values[:,1]-nearest.values[:,0]).min().item()),
        'HQ_generator_is_R2_generator_byte_exact':all(torch.equal(r2_saved[n],t) for n,t in hq_saved.items() if n.startswith('generator.')),
        'teacher_reconstruction_is_separate_reference_not_exact_R2_prior_ceiling':True}
    before = {'R2':tensor_digest(r2.state_dict(),torch),'HQ':tensor_digest(teacher.state_dict(),torch)}
    teachers = {}; results = []
    original_lookup = r2.quantize.get_codebook_feat
    with torch.inference_mode():
        parity_row = manifest['cases'][0]
        need(parity_row['profile']=='clear','First fixed parity case must be clear')
        parity_input = F.interpolate(tensor(rgb(parity_row['input']['path'],256)),(512,512),mode='bilinear',align_corners=False)*2-1
        pn,pl,pf = operation('unwrapped_R2_parity/'+parity_row['id'],lambda z:r2(z,w=1.,detach_16=True,adain=True),parity_input)
        parity_raw = output('parity/unwrapped_R2',pn)
        parity_upstream = (tensor_digest({'logits':pl},torch),tensor_digest({'lq':pf},torch))
        array('parity/unwrapped_R2-logits.npy',pl.numpy().copy());array('parity/unwrapped_R2-lq.npy',pf.numpy().copy())
        del pn,pl,pf,parity_input
        for row in manifest['targets']:
            rid = row['reference_id']; x = tensor(rgb(row['HQ512']['path'],512))*2-1
            encoded=[]
            hook=teacher.encoder.register_forward_hook(lambda module,args,value:encoded.append(value.detach().clone()))
            try:native,_,stats = operation('HQ/'+rid,teacher,x)
            finally:hook.remove()
            need(len(encoded)==1 and encoded[0].shape==(1,256,16,16),'One exact HQ encoder feature field required')
            native_indices = stats['min_encoding_indices'].detach().long().reshape(1,256)
            # Project genuine HQ features onto the *exact frozen R2 book*.
            # This is explicitly distinct from the native HQ quantizer labels.
            z=encoded[0].permute(0,2,3,1).contiguous().view(-1,256)
            rb=r2.quantize.embedding.weight
            d=(z**2).sum(1,keepdim=True)+(rb**2).sum(1)[None,:]-2*torch.matmul(z,rb.T)
            indices=d.argmin(1).reshape(1,256)
            need(indices.min()>=0 and indices.max()<1024,'HQ indices out of codebook')
            teachers[rid] = indices.clone()
            array('teacher/'+rid+'-codes.npy',indices.numpy().copy())
            array('teacher/'+rid+'-native-HQ-codes.npy',native_indices.numpy().copy())
            array('teacher/'+rid+'-HQ-encoder-features.npy',encoded[0].numpy().copy())
            output('teacher/'+rid,native)
        for row in manifest['cases']:
            target = rgb(row['metrics_target256']['path'],256).astype(np.float64)/255.
            x = F.interpolate(tensor(rgb(row['input']['path'],256)),(512,512),mode='bilinear',align_corners=False)*2-1
            arms = {}; upstream = None
            for arm in ('natural','HQ_code_oracle'):
                capture = []
                def lookup(indices, shape=None):
                    capture.append(indices.detach().clone())
                    return original_lookup(indices if arm=='natural' else teachers[row['reference_id']],shape=shape)
                need('get_codebook_feat' not in r2.quantize.__dict__,'Unexpected preexisting lookup override')
                r2.quantize.get_codebook_feat = lookup
                try:
                    native,logits,lq = operation(row['id']+'/'+arm,lambda z:r2(z,w=1.,detach_16=True,adain=True),x)
                finally:
                    del r2.quantize.__dict__['get_codebook_feat']
                need(len(capture)==1,'Exactly one captured code lookup required')
                codes = capture[0].reshape(1,256)
                current = (tensor_digest({'logits':logits},torch),tensor_digest({'lq':lq},torch),codes.numpy().copy())
                if upstream is None: upstream=current
                else: need(current[:2]==upstream[:2] and np.array_equal(current[2],upstream[2]),'Code intervention changed observed encoder/logits/natural code selection')
                prefix='cases/'+row['id']+'/'+arm
                raw=output(prefix,native)
                if row['id']==parity_row['id'] and arm=='natural':
                    need(np.array_equal(raw,parity_raw) and current[:2]==parity_upstream,
                         'Read-only natural lookup instrumentation changed the actual R2 output')
                array(prefix+'-logits.npy',logits.numpy().copy()); array(prefix+'-lq.npy',lq.numpy().copy())
                array(prefix+'-natural-codes.npy',codes.numpy().copy())
                used=codes if arm=='natural' else teachers[row['reference_id']]
                array(prefix+'-used-codes.npy',used.numpy().copy())
                png=np.floor(raw*np.float32(255)).astype(np.uint8)
                arms[arm]={'RAW_MAE':float(np.abs(raw.astype(np.float64)-target).mean()),
                    'PNG_MAE':float(np.abs(png.astype(np.float64)/255.-target).mean()),
                    'natural_code_mismatch_to_HQ':float((codes!=teachers[row['reference_id']]).float().mean().item()),
                    'used_code_mismatch_to_HQ':float((used!=teachers[row['reference_id']]).float().mean().item()),
                    'logits_tensor_sha256':current[0],'lq_tensor_sha256':current[1]}
                del native,logits,lq,raw
            results.append({'id':row['id'],'reference_id':row['reference_id'],'profile':row['profile'],'arms':arms})
            new_json(OUT/('cases/'+row['id']+'/metrics.json'),results[-1])
            files['cases/'+row['id']+'/metrics.json'] = pin(OUT/('cases/'+row['id']+'/metrics.json'))
            print(json.dumps({'case':len(results),'of':18,'calls':calls,'seconds':time.monotonic()-start}),flush=True)
    after={'R2':tensor_digest(r2.state_dict(),torch),'HQ':tensor_digest(teacher.state_dict(),torch)}
    need(before==after and calls==43 and all(p.grad is None and not p.requires_grad for net in (r2,teacher) for p in net.parameters()),
         'All model states/gradients must remain unchanged; exact43 calls')
    need(all(pin(ROOT/name)==value for name,value in evidence.items()),'Input/source changed during diagnostic')
    result={'complete':True,'evidence_integrity_pass':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'neural_forward_attempts':calls,'neural_forward_returns':calls,'teacher_forwards':6,'CodeFormer_forwards':37,
        'optimizer_updates':0,'gradient_queries':0,'backward_calls':0,'VM_calls':0,'device':'cpu','dtype':'FP32',
        'source_pins':evidence,'state_before':before,'state_after':after,'cases':results,'files':files,'resources':resources,
        'codebook_compatibility':compatibility,'natural_instrumentation_actual_RAW_parity':True,
        'oracle_target_policy':'genuine HQ512 encoder features projected onto exact R2 codebook using stock FP32 squared-distance argmin; native HQ labels retained separately',
        'operations_journal':pin(OUT/'operations.jsonl'),
        'seconds':time.monotonic()-start,'inference_policy':{'fidelity':1.,'adain':True,'internal':512,'delivered':256,'display_processing':'none'},
        'diagnostic_not_deployable':True,'clean_reference_codes_unavailable_at_inference':True,
        'training_admitted':False,'model_qualified':False,'app_default_changed':False,'FINAL_used':False,'native_CCTV_used':False}
    new_json(OUT/'result_v1.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();need(sum((args.verify,args.run,args.worker))==1,'Choose one explicit operation')
    if args.worker:return worker()
    manifest,review,evidence=verify()
    if args.verify:
        print(json.dumps({'complete':True,'cases':18,'planned_neural_forwards':43,'actual_neural_forwards':0,'optimizer_updates':0}));return
    need(not OUT.exists(),'Preserve previous/partial diagnostic; no automatic retry')
    need(shutil.disk_usage(ROOT).free>=5*1024**3,'Require5GiB free before local diagnostic')
    OUT.mkdir(parents=True)
    new_json(OUT/'launch.json',{'parent_pid':os.getpid(),'source_sha256':pin(__file__)['sha256'],'source_pins':evidence,
        'UTC':datetime.now(timezone.utc).isoformat(),'external_timeout_seconds':960,'worker_seconds':900,'planned_forwards':43,'optimizer_updates':0})
    try:
        with (OUT/'worker.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'--worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            import psutil
            child_started=psutil.Process(child.pid).create_time()
            new_json(OUT/'child.json',{'pid':child.pid,'parent_pid':os.getpid()})
            try:code=child.wait(timeout=960)
            except subprocess.TimeoutExpired:
                stopped=stop_owned_worker_tree(child.pid,child_started)
                new_json(OUT/'timeout_tree_stop.json',{'stopped_owned_pids':stopped,'parent_pid':os.getpid()})
                child.wait(timeout=30);raise TimeoutError('External960s diagnostic stop; preserve partial outputs')
        need(code==0,'Diagnostic worker failed; inspect retained worker.log')
        need((OUT/'result_v1.json').is_file(),'Actual result receipt required')
        print(json.dumps({'complete':True,'result':str(OUT/'result_v1.json'),'neural_forwards':43,'optimizer_updates':0}),flush=True)
    except BaseException:
        new_json(OUT/'failure.json',{'complete':False,'traceback':traceback.format_exc(),'model_qualified':False,'retry_allowed':False})
        raise


if __name__=='__main__':main()
