"""CPU stdlib-only audit of fresh100HQ cached arrays; never executes a model."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='67d71dbb38de308b3f0b9299bc9f34844c51c8445149ce35cdac8e98004f1afb'
ARCHIVE='ca152f53a3f733a4cd0ce859ec7d6b4248b47cf5def7c13b7bc52e6dc8697dad'
STAGE=ROOT/'outputs/cctv_codeformer_role_preserving_training_v1_vm'
RETURN=ROOT/'outputs/codeformer_role_preserving_training_v1_return'
OLD=ROOT/'outputs/codeformer_adain_gradient_isolation_v1_return'
REVIEW=ROOT/'outputs/codeformer_role_preserving_training_v1_return_review'
OUT=REVIEW/'hq_v1'
PREFIX='outputs/role_preserving_training_v1/HQ_cache/'


def need(value,message):
    if not value:raise ValueError(message)


def read(path,cap=2*1024**2):
    path=Path(path)
    for parent in (path,*path.parents):
        if parent.exists():
            s=parent.lstat();need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024,'No linked/reparse paths')
    s=path.lstat();need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=cap,'Bounded single-linked regular source required')
    b=path.read_bytes();a=path.stat()
    need((s.st_size,s.st_ino,s.st_mtime_ns)==(a.st_size,a.st_ino,a.st_mtime_ns),'Source changed during read')
    return b


def sha(b):return hashlib.sha256(b).hexdigest()
def load(path):return json.loads(read(path))
def pin(path):
    b=read(path);return {'sha256':sha(b),'bytes':len(b)}


def array_payload(b,kind):
    need(b[:8]==b'\x93NUMPY\x01\x00','NPY v1.0 required')
    width=struct.unpack('<H',b[8:10])[0];need(width<=4096 and len(b)>=10+width,'Bounded NPY header')
    text=b[10:10+width].decode('ascii');need(text.endswith('\n'),'NPY newline required')
    d=ast.literal_eval(text)
    need(set(d)=={'descr','fortran_order','shape'} and d['fortran_order']is False,'Exact C-order plain scalar NPY header')
    expected=('<i8',(1,256),8)if kind=='indices'else('<f4',(1,256,16,16),4)
    need((d['descr'],d['shape'])==expected[:2],'Expected HQ shape/dtype differs')
    payload=b[10+width:];count=1
    for dim in d['shape']:count*=dim
    need(len(payload)==count*expected[2],'Exact NPY payload/EOF differs')
    return payload,d


def main():
    started=time.monotonic();need(not OUT.exists(),'Preserve previous HQ audit');OUT.mkdir()
    pb=read(STAGE/'protocol.json');need(sha(pb)==PROTOCOL,'Frozen source protocol differs');p=json.loads(pb)
    imp=load(REVIEW/'return_import_v1.json');need(imp['complete']is True and imp['archive_sha256']==ARCHIVE and
        imp['protocol_sha256']==PROTOCOL and imp['new_restorer_updates']==0,'Complete exact failure import required')
    hp=OLD/p['hq_cache_parity']['path'];hb=read(hp);need(sha(hb)==p['hq_cache_parity']['sha256'],'Pinned391-row HQ parity source differs')
    h=json.loads(hb);rows={row['reference_id']:row for row in h['cases']}
    need(len(rows)==len(h['cases'])==391 and all(row['role']=='train'for row in rows.values()),'391uniqueTRAIN rows required')
    op=read(OLD/'protocol.json');need(sha(op)==p['parent_protocol_sha256'],'Historical parent protocol differs')
    evidence=[]
    for ref in p['selected_train_reference_ids']:
        oldbase='outputs/adain_gradient_isolation_v1/HQ_cache/'+ref+'/'
        newbase=PREFIX+ref+'/'
        rp=newbase+'record.json';rb=read(RETURN/rp);r=json.loads(rb)
        need(sha(rb)==imp['files_sha256'][rp] and len(rb)==imp['files_bytes'][rp],'Fresh record exact imported bytes')
        need(r['reference_id']==ref and r['fresh_HQ_forward']is True and r['exact_retained_arrays']is True,
            'Fresh HQ record identity differs')
        source=r['sourcebindings'];canonical=rows[ref]
        need(all(source[key]==canonical[key]for key in canonical),'Source/reference/teacher/dictionary391-row bindings differ')
        need(r['teacher_state_sha256']==canonical['teacher_state_sha256'] and
            r['restoration_codebook_state_sha256']==canonical['restoration_codebook_state_sha256'],
            'Top-level teacher/dictionary differs')
        row={'reference_id':ref,'source_sha256':canonical['source_sha256'],
            'teacher_state_sha256':canonical['teacher_state_sha256'],
            'restoration_codebook_state_sha256':canonical['restoration_codebook_state_sha256']}
        for kind in ('indices','features'):
            rel=newbase+kind+'.npy';oldrel=oldbase+kind+'.npy'
            nb=read(RETURN/rel);ob=read(OLD/oldrel)
            n={'sha256':sha(nb),'bytes':len(nb)};expected=p['extra_returned_dependencies'][oldrel]
            need(nb==ob and n==expected,
                'Fresh HQ array exact bytes differ from pinned historical parent')
            need(n['sha256']==imp['files_sha256'][rel]and n['bytes']==imp['files_bytes'][rel] and
                all(r[kind][key]==n[key]for key in n),'Fresh array import/record/file identity differs')
            payload,header=array_payload(nb,kind)
            need(sha(payload)==canonical[kind+'_array_sha256'],'Actual uncompressed array payload differs from canonical391-row parity')
            row[kind]={'fresh_file':rel,'parent_file':oldrel,**n,'payload_sha256':sha(payload),
                'shape':header['shape'],'dtype':header['descr'],'exact_parent_bytes':True,'canonical_payload_pass':True}
        oldrecord=read(OLD/(oldbase+'record.json'))
        need({'sha256':sha(oldrecord),'bytes':len(oldrecord)}==p['extra_returned_dependencies'][oldbase+'record.json'],
            'Pinned historical record differs')
        row['pass']=True;evidence.append(row)
    need(len(evidence)==100 and len({row['reference_id']for row in evidence})==100,'Exactly100TRAIN references required')
    need(not any(name.split('.')[0]in ('torch','numpy','tensorflow','jax')for name in sys.modules),'No framework imports permitted')
    receipt={'format':'codeformer-role-preserving-fresh-HQ-return-audit-v1','complete':True,'pass':True,
        'UTC':datetime.now(timezone.utc).isoformat(),'protocol_sha256':PROTOCOL,'archive_sha256':ARCHIVE,
        'return_import_receipt':pin(REVIEW/'return_import_v1.json'),'historical_parent_protocol':{'sha256':sha(op),'bytes':len(op)},
        'canonical391rowHQ_manifest':{'sha256':sha(hb),'bytes':len(hb)},'selected_TRAIN_references':100,
        'fresh_HQ_arrays':200,'exact_parent_file_bytes':200,'canonical_array_payload_hashes':200,
        'source_teacher_dictionary_joins':100,'rows':evidence,'new_restorer_updates':0,'new_checkpoints':0,
        'new_candidate_native_outputs':0,'model_qualified':False,'actual_local_neural_gradient_optimizer_VM_calls':0,
        'source_sha256':pin(__file__)['sha256'],'seconds':time.monotonic()-started,
        'scope':'Fresh HQ-cache integrity/parity only. The worker stopped before initializer parity and optimizer; no new restoration quality or training success.'}
    with (OUT/'audit.json').open('x',encoding='utf-8',newline='\n')as s:json.dump(receipt,s,indent=2,allow_nan=False);s.write('\n')
    with (OUT/'audit.md').open('x',encoding='utf-8',newline='\n')as s:
        s.write('Fresh HQ-cache parity PASS: 100 selected TRAIN references, 200 exact historical-parent array files, 200 independently parsed NPY payload hashes matched the pinned 391-row canonical metadata. Source, clean-input, frozen teacher and restoration dictionary bindings close for every reference.\n\nNo framework, model, neural, gradient, optimizer, checkpoint or VM execution occurred in this stdlib CPU audit. The returned run has zero new updates, no new checkpoint and no candidate native restoration outputs. HQ-cache integrity is not restoration improvement.\n')
    print(json.dumps({'complete':True,'references':100,'arrays':200,'exact_parent_bytes':200,'canonical_payload_hashes':200,
        'new_restorer_updates':0,'model_qualified':False,'seconds':time.monotonic()-started}))


if __name__=='__main__':main()
