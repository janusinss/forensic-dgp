"""Independent bounded local transport admission. No model/checkpoint imports."""
from __future__ import annotations
import gzip,hashlib,json,os,re,shutil,stat,tarfile,time
from pathlib import Path,PurePosixPath
BASE=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');SOURCE=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
STEM='cctv-completion-codeformer-predictor-ce-v1';ROOT='cctv_completion_codeformer_predictor_ce_v1_results';EXPECTED='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
DEST=BASE/'outputs/completion_codeformer_predictor_ce_v1_return';PACKET=BASE/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm/protocol.json'
CAP=2816*1024**2;OUTCAP=2560*1024**2;checks=[]
def need(ok,label):
    if not ok:raise ValueError(label)
    checks.append(label)
def xp(path):
    value=os.path.abspath(path);return '\\\\?\\'+value if os.name=='nt' and not value.startswith('\\\\?\\') else value
def parse(body):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('Duplicate JSON '+k)
            out[k]=v
        return out
    def reject(x):raise ValueError('Nonfinite JSON '+x)
    return json.loads(body.decode('utf-8-sig'),object_pairs_hook=unique,parse_constant=reject)
def digest(path):
    h=hashlib.sha256()
    with open(xp(path),'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def plain(path):
    info=os.lstat(xp(path));need(stat.S_ISREG(info.st_mode) and not os.path.islink(xp(path)),'Regular input '+Path(path).name);return info
class Counted:
    def __init__(self,stream):self.stream=stream;self.count=0
    def read(self,size=-1):
        body=self.stream.read(1024**2 if size<0 else size);self.count+=len(body)
        if self.count>CAP:raise ValueError('Raw stream exceeds2816MiB')
        return body
    def tell(self):return self.count

def safe(name):
    need(isinstance(name,str) and name.startswith(ROOT+'/'),'Exact archive root');relative=name[len(ROOT)+1:]
    need(0<len(relative.encode('utf-8'))<=240,'Relative path240UTF8bytes');need('\\' not in relative and ':' not in relative and '\x00' not in relative,'No alternate namespace')
    p=PurePosixPath(relative);need(not p.is_absolute() and p.as_posix()==relative and all(v not in ('','.','..') for v in p.parts),'Canonical relative path')
    for item in p.parts:
        need(item==item.rstrip(' .') and all(ord(c)>=32 for c in item),'Windows canonical component');need(not re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',item,re.I),'No Windows device path')
    need(relative in ('protocol.json','supervisor.json','trainer.log','export_manifest.json') or relative.startswith('outputs/'),'Closed exporter namespace');return relative

def inspect(archive,expected_records=None,extract=False):
    records=[];seen=set();small={};payload=0
    with open(xp(archive),'rb') as raw,gzip.GzipFile(fileobj=raw,mode='rb') as gz:
        counted=Counted(gz)
        with tarfile.open(fileobj=counted,mode='r|',bufsize=1024**2) as tar:
            for member in tar:
                relative=safe(member.name);norm=relative.casefold();need(norm not in seen,'Unique casefold '+relative);seen.add(norm)
                need(member.isfile() and member.type==tarfile.REGTYPE,'Regular tar payload '+relative);need(0<=member.size<=1024**3,'Per-file1GiB '+relative);need(set(member.pax_headers)<={'path','size'},'Restricted PAX '+relative);need(len(records)<1000,'Member cap1000')
                payload+=member.size;need(payload<=OUTCAP+2*1024**2,'Whole output2560MiB plus manifest');src=tar.extractfile(member);need(src is not None,'Readable member '+relative)
                sha=hashlib.sha256();body=[];received=0;destfile=None
                if extract:
                    output=DEST/ROOT/Path(*PurePosixPath(relative).parts);os.makedirs(xp(output.parent),exist_ok=True);destfile=open(xp(output),'xb')
                try:
                    while True:
                        block=src.read(1024**2)
                        if not block:break
                        received+=len(block);need(received<=member.size,'Bounded member '+relative);sha.update(block)
                        if relative in ('protocol.json','supervisor.json','export_manifest.json','outputs/results.json','outputs/failure.json','outputs/progress.json'):
                            need(member.size<=8*1024**2,'Small JSON '+relative);body.append(block)
                        if destfile is not None:destfile.write(block)
                finally:
                    if destfile is not None:destfile.close()
                    src.close()
                need(received==member.size,'Exact size '+relative);record={'path':relative,'bytes':received,'sha256':sha.hexdigest()}
                if expected_records is not None:need(record==expected_records[relative],'Second pass bytes '+relative)
                records.append(record)
                if body:small[relative]=b''.join(body)
                if len(records)%100==0:print({'transport_pass':2 if extract else 1,'files':len(records),'payload_bytes':payload},flush=True)
        while True:
            block=counted.read(1024**2)
            if not block:break
            need(not any(block),'Zero tar trailing padding')
        rawbytes=counted.count
    return records,small,rawbytes,payload

def main():
    started=time.monotonic();archive=SOURCE/(STEM+'-results.tar.gz');receiptpath=SOURCE/(STEM+'-export.json');checksum=SOURCE/(STEM+'-results.tar.gz.sha256')
    need(not DEST.exists(),'Fresh admission root; no overwrite');ai=plain(archive);plain(receiptpath);plain(checksum);receipt=parse(receiptpath.read_bytes());checksumtext=checksum.read_text(encoding='ascii')
    need(receipt.get('format')=='completion-codeformer-predictor-ce-v1-export-receipt-v1' and receipt.get('complete') is True,'Exact completed export receipt');need(receipt.get('protocol_sha256')==EXPECTED and receipt.get('training_success_not_implied') is True and receipt.get('model_qualified') is False,'Receipt scope')
    expected=receipt['archive_sha256'];need(re.fullmatch('[0-9a-f]{64}',expected) is not None,'SHA format');need(checksumtext==expected+'  '+archive.name+'\n','Exact checksum filename/bytes');need(ai.st_size==receipt['bytes']<=CAP,'Archive length/cap');need(digest(archive)==expected,'Full archive SHA256')
    records,small,rawbytes,total=inspect(archive);need(rawbytes==receipt['raw_tar_bytes']<=CAP,'Full gzipEOF/CRC/rawtar length');bypath={r['path']:r for r in records};manifest=parse(small['export_manifest.json']);protocol=parse(small['protocol.json']);supervisor=parse(small['supervisor.json'])
    need(bypath['protocol.json']['sha256']==EXPECTED==digest(PACKET),'Sealed protocol SHA');need(small['protocol.json']==PACKET.read_bytes(),'Sealed protocol exact bytes');need(manifest.get('format')=='completion-codeformer-predictor-ce-v1-return-manifest-v1' and manifest.get('protocol_sha256')==EXPECTED,'Manifest format/protocol')
    entries=manifest['files'];need(isinstance(entries,list) and len(entries)==receipt['payload_files'],'Manifest file count');declared={}
    for r in entries:
        need(set(r)=={'path','bytes','sha256'},'Manifest record schema');safe(ROOT+'/'+r['path']);need(r['path'] not in declared,'Manifest unique path');need(type(r['bytes']) is int and r['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None,'Manifest scalar schema');declared[r['path']]=r
    need(set(declared)==set(bypath)-{'export_manifest.json'},'Exact closed manifest namespace');need(all(bypath[n]==r for n,r in declared.items()),'Every payload independent size/SHA');need(sum(r['bytes'] for r in entries)==receipt['payload_bytes']<=OUTCAP,'Exact payload sum')
    need(manifest.get('worker_exit')==receipt['worker_exit']==supervisor['worker_exit'],'Worker exit join');need(supervisor.get('complete') is True and type(supervisor['worker_exit']) is int,'Terminal supervisor')
    for key in ('terminal_state_committed','partial_files_retained','retained_checkpoint_source_aliases'):need(manifest.get(key)==receipt.get(key),'Receipt/manifest '+key)
    need(manifest.get('raw_and_compressed_cap_bytes')==CAP and manifest.get('whole_output_cap_bytes')==OUTCAP,'Exact exporter caps');need(manifest.get('model_qualified') is False and manifest.get('training_success_not_implied') is True,'Manifest qualification boundary')
    checkpoints=[r for r in entries if any(s.lower() in ('.pth','.pt') for s in PurePosixPath(r['path']).suffixes)];need(len(checkpoints)<=1 and all(r['path'] in ('outputs/terminal_training_state.pth','outputs/terminal_training_state.pth.partial') for r in checkpoints),'One allowed terminal payload');need(sum(r['bytes'] for r in checkpoints)<=1024**3,'Terminal1GiBcap');need(len([r for r in entries if r['path'].endswith('.npz')])<=468,'NPZ468cap');need(len([r for r in entries if r['path'].startswith('outputs/teacher_labels/') and r['path'].endswith('.npz')])==manifest['retained_teacher_label_arrays']==80,'All80 cached label arrays')
    if supervisor['worker_exit'] in (0,2):
        summary=parse(small['outputs/results.json']);need(summary.get('complete') is True and summary.get('protocol_sha256')==EXPECTED and summary.get('model_qualified') is False,'Actual results scope');need(manifest['terminal_state_pin']==declared['outputs/terminal_training_state.pth'],'Exact terminal pin')
    need(shutil.disk_usage(BASE).free>=total+5*1024**3,'Extraction plus5GiB free reserve');os.makedirs(xp(DEST),exist_ok=False)
    try:
        second,_,raw2,total2=inspect(archive,bypath,True);need(second==records and raw2==rawbytes and total2==total,'Second-pass byte identity/CRC');need(digest(archive)==expected,'Full archive SHA after extraction');after=plain(archive);need((ai.st_size,ai.st_mtime_ns,ai.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Source immutable stat')
        audit={'format':'completion-codeformer-predictor-ce-v1-independent-local-transport-v1','complete':True,'quality_adoption_implied':False,'model_qualified':False,'archive':{'path':str(archive),'bytes':ai.st_size,'sha256':expected,'raw_tar_bytes':rawbytes,'gzip_crc_verified':True},'export_receipt_pin':{'bytes':receiptpath.stat().st_size,'sha256':digest(receiptpath)},'checksum_pin':{'bytes':checksum.stat().st_size,'sha256':digest(checksum)},'protocol_sha256':EXPECTED,'admitted_root':str(DEST/ROOT),'payload_files':len(entries),'total_regular_members':len(records),'payload_bytes':receipt['payload_bytes'],'manifest_pin':bypath['export_manifest.json'],'worker_exit':supervisor['worker_exit'],'supervisor':supervisor,'results_summary':parse(small['outputs/results.json']) if 'outputs/results.json' in small else None,'terminal_pin':manifest['terminal_state_pin'],'retained_checkpoint_source_aliases':manifest['retained_checkpoint_source_aliases'],'files':records,'checks_passed':len(checks),'check_labels':checks,'seconds':time.monotonic()-started,'neural_calls':0,'training_calls':0,'checkpoint_loads':0,'VM_connection':False,'extraction_policy':'Closed exact root; canonical bounded Windows components; unique casefold; regular files only; exclusive creation; extended Windows paths; two memberSHA passes and fullgzipEOF/CRC; nooverwrite.'}
        out=DEST/'transport_audit.json'
        with open(xp(out),'x',encoding='utf-8',newline='\n') as f:json.dump(audit,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps({'complete':True,'root':str(DEST/ROOT),'transport_audit_sha256':digest(out),'audit_bytes':out.stat().st_size,'members':len(records),'worker_exit':supervisor['worker_exit'],'seconds':audit['seconds']},indent=2),flush=True)
    except BaseException as exc:
        with open(xp(DEST/'transport_failure.json'),'x',encoding='utf-8',newline='\n') as f:json.dump({'complete':False,'error':repr(exc),'partial_evidence_preserved':True,'archive_sha256':expected,'seconds':time.monotonic()-started},f,indent=2);f.write('\n')
        raise
if __name__=='__main__':main()
