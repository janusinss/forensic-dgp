"""Inert, bounded import of the completed spatial-continuation return."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'cctv-codeformer-spatial-continuation-v1'
SHA = 'c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed'
DEST = ROOT/'outputs/cctv_codeformer_spatial_continuation_v1_return'
REVIEW = ROOT/'outputs/codeformer_spatial_continuation_return_review_v1'
MIB = 1024**2
START = time.monotonic()


def need(value, message):
    if not value: raise ValueError(message)
    if time.monotonic()-START > 300: raise TimeoutError('Import 300-second ceiling')


def parse(payload):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate JSON key'); result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=unique,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def physical(path):
    for current in (path, *path.parents):
        if current.exists() or current.is_symlink():
            s=current.lstat()
            need(not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024,
                 'Symlink/reparse path refused')
    return path


def pin(path):
    physical(path); before=path.stat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink==1, 'Unique regular input required')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(4*MIB),b''): digest.update(b);need(True,'timer')
    after=path.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns)==
         (after.st_ino,after.st_size,after.st_mtime_ns),'Input changed while hashing')
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':after.st_size,'sha256':digest.hexdigest()}


def canonical(name):
    parts=PurePosixPath(name)
    need(type(name) is str and len(name)<=240 and str(parts)==name and not parts.is_absolute()
         and all(p not in ('.','..','') and not p.endswith((' ','.')) and
                 not re.match(r'^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)',p,re.I)
                 for p in parts.parts) and bool(parts.parts)
         and re.fullmatch(r'[A-Za-z0-9_./-]+',name) is not None,'Unsafe member path')
    return name


class Bounded:
    def __init__(self, stream): self.stream=stream;self.count=0
    def read(self, size=-1):
        need(size>=0,'Bounded reads only');b=self.stream.read(size);self.count+=len(b)
        need(self.count<=520*MIB,'Decompressed tar cap');return b


def scan(archive, callback):
    seen={};total=0
    with gzip.open(archive,'rb') as compressed:
        stream=Bounded(compressed)
        with tarfile.open(fileobj=stream,mode='r|') as tar:
            for member in tar:
                name=canonical(member.name)
                need(member.isfile() and not member.issparse() and not member.pax_headers,
                     'Only plain regular members permitted')
                need(name.casefold() not in seen and len(seen)<2000,'Duplicate/case collision/member cap')
                need(0<=member.size<=40*MIB,'Per-member cap')
                total+=member.size;need(total<=512*MIB,'Payload cap')
                seen[name.casefold()]=name
                with tar.extractfile(member) as source:callback(name,member.size,source)
        # Consume and validate gzip trailer, rejecting extra nonzero tar payload.
        for block in iter(lambda:stream.read(MIB),b''):need(not any(block),'Trailing nonzero tar data')
    return {'members':len(seen),'member_payload_bytes':total}


def main():
    outputs=ROOT/'outputs'; archive=outputs/(PREFIX+'-results.tar.gz')
    companion=outputs/(PREFIX+'-export.json'); checksum=Path(str(archive)+'.sha256')
    report=parse(companion.read_bytes()); checksum_bytes=checksum.read_bytes()
    need(report['complete'] is True and report['protocol_sha256']==SHA and
         report['worker_exit']==0 and report['training_success_not_implied'] is True,
         'Wrong or incomplete recovery receipt')
    need(report['bytes']<=544*MIB,'Compressed cap')
    archive_pin=pin(archive)
    need(archive_pin['bytes']==report['bytes'] and archive_pin['sha256']==report['archive_sha256'] and
         checksum_bytes==(report['archive_sha256']+'  '+archive.name+'\n').encode('ascii'),
         'Archive/checksum/receipt disagree')
    stage=ROOT/'outputs/cctv_codeformer_spatial_continuation_v1_vm'
    protocol_bytes=(stage/'protocol.json').read_bytes();need(hashlib.sha256(protocol_bytes).hexdigest()==SHA,'Protocol changed')
    protocol=parse(protocol_bytes);manifest_holder={}; first_pins={}
    allowed=set(protocol['return_static'])|set(protocol['allowed_outputs'])|{'protocol.json'}
    def inspect(name,size,source):
        need(name in allowed and not name.startswith('cache/') and name not in protocol['vm_only_static'],
             'Member outside closed return namespace')
        digest=hashlib.sha256();count=0;content=[]
        for b in iter(lambda:source.read(MIB),b''):
            digest.update(b);count+=len(b)
            if name=='export.json':content.append(b)
        need(count==size,'Truncated member')
        first_pins[name]={'path':name,'bytes':size,'sha256':digest.hexdigest()}
        if name=='export.json':manifest_holder['value']=parse(b''.join(content))
    scanned=scan(archive,inspect)
    need(scanned['members']==report['members'] and scanned['member_payload_bytes']==report['return_payload_bytes'],
         'Recovery counts disagree')
    manifest=manifest_holder['value']
    need(manifest['complete'] is True and manifest['worker_exit']==0 and manifest['protocol_sha256']==SHA,
         'Original manifest boundary')
    rows=manifest['members'];need(len({r['path'] for r in rows})==len(rows),'Duplicate manifest rows')
    need({r['path']:r for r in rows}=={n:r for n,r in first_pins.items() if n!='export.json'},
         'Every member must match unchanged original manifest')
    need(manifest['format']=='codeformer-spatial-continuation-export-manifest-v1','Manifest format differs')
    need(set(protocol['return_static'])|{'protocol.json'}<=set(first_pins),'Missing static return')
    for name in protocol['return_static']:
        need({k:first_pins[name][k] for k in ('bytes','sha256')}==protocol['files'][name],
             'Sealed source/asset changed: '+name)
    need(first_pins['protocol.json']['sha256']==SHA,'Returned protocol differs')
    physical(DEST);need(not DEST.exists(),'Preserve existing import; no overwrite')
    need(shutil.disk_usage(outputs).free>=4*1024**3+scanned['member_payload_bytes'],'Local space reserve')
    DEST.mkdir()
    def copy(name,size,source):
        path=DEST.joinpath(*PurePosixPath(name).parts);physical(path)
        need(path.resolve().is_relative_to(DEST.resolve()),'Destination escape')
        path.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256();count=0
        with path.open('xb') as target:
            for b in iter(lambda:source.read(MIB),b''):
                need(target.write(b)==len(b),'Short import write');digest.update(b);count+=len(b)
        need(first_pins[name]=={'path':name,'bytes':count,'sha256':digest.hexdigest()},'Second-pass bytes differ')
    need(scan(archive,copy)==scanned,'Second-pass namespace differs')
    for name,expected in first_pins.items():
        actual=pin(DEST/name);need(actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256'],
                                  'Physical imported file mismatch')
    need(pin(archive)==archive_pin,'Downloaded archive changed during import')
    REVIEW.mkdir(exist_ok=True)
    receipt={'complete':True,'archive':archive_pin,'export_companion':pin(companion),'checksum':pin(checksum),
             'protocol_sha256':SHA,'destination':DEST.relative_to(ROOT).as_posix(),**scanned,
             'every_member_manifest_and_physical_hash_verified':True,'gzip_trailer_verified':True,
             'static_return_and_closed_namespace_verified':True,'VM_only_assets_not_duplicated':True,
             'files':first_pins,'seconds':time.monotonic()-START,'Torch_neural_optimizer_training_VM_calls':0,
             'quality_audit_pending':True,'model_qualified':False}
    with (REVIEW/'import_v1.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))


if __name__=='__main__':main()
