"""Independent actual zero-optimizer probe return transport; no returned code runs."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import gzip,hashlib,json,os,re,shutil,tarfile,time
ROOT=Path(__file__).resolve().parents[2];SOURCE=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs');STEM='cctv-completion-codeformer-same-codebook-probe-v1';ARCHIVE=SOURCE/(STEM+'-results.tar.gz');EXPORT=SOURCE/(STEM+'-export.json');CHECKSUM=SOURCE/(STEM+'-results.tar.gz.sha256');RET=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_return';MEMBER='cctv_completion_codeformer_same_codebook_probe_v1_results';PROTOCOL='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec';ARCHIVE_SHA='e68faa3360b0421c15598a12d472db191395a1790b3b81833f7d611da71577d9';CAP=1536*1024**2

def need(ok,msg):
 if not ok:raise ValueError(msg)
def longpath(p):
 p=Path(p).resolve();return p if str(p).startswith('\\\\?\\') else Path('\\\\?\\'+str(p))
def sha(p):
 h=hashlib.sha256()
 with longpath(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':longpath(p).stat().st_size,'sha256':sha(p)}
def read(p):
 def reject(v):raise ValueError('Nonfinite JSON '+v)
 return json.loads(longpath(p).read_text(encoding='utf-8-sig'),parse_constant=reject)
def main():
 started=time.monotonic();need(not (RET/MEMBER).exists() and (RET/'partial_extraction_001.json').is_file(),'Require fresh canonical root and retained first attempt');need(ARCHIVE.is_file() and not ARCHIVE.is_symlink() and ARCHIVE.stat().st_size==689101051 and ARCHIVE.stat().st_size<=CAP,'Actual archive size/type differs');export=read(EXPORT)
 need(export['complete'] is True and export['format']=='completion-codeformer-same-codebook-probe-v1-export-receipt-v1' and export['archive_sha256']==ARCHIVE_SHA and export['bytes']==ARCHIVE.stat().st_size and export['protocol_sha256']==PROTOCOL,'External receipt differs')
 need(export['worker_exit']==0 and export['terminal_state_committed'] is False and export['partial_files_retained']==[] and export['payload_files']==413 and export['diagnostic_only'] is True and export['new_model_created'] is False and export['model_qualified'] is False and export['export_only_neural_calls']==0,'Diagnostic export scope differs')
 checksum=CHECKSUM.read_text(encoding='ascii').strip();need(checksum==ARCHIVE_SHA+'  '+ARCHIVE.name,'Exact checksum record differs');need(sha(ARCHIVE)==ARCHIVE_SHA,'Full archive SHA mismatch')
 RET.mkdir(exist_ok=True);RUN=RET/MEMBER;RUN.mkdir();need(RUN.resolve().parent==RET.resolve() and RET.resolve().parent==(ROOT/'outputs').resolve(),'Confined new result namespace');seen={};expanded=0
 with ARCHIVE.open('rb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='rb') as gz:
   with tarfile.open(fileobj=gz,mode='r|') as tar:
    for item in tar:
     name=item.name;rel=PurePosixPath(name);need(item.isreg() and not item.issparse() and len(name.encode('utf-8'))<=512 and not rel.is_absolute() and '..' not in rel.parts and ':' not in name and '\\' not in name and str(rel)==name and rel.parts[0]==MEMBER,'Unsafe tar member '+name)
     relative=PurePosixPath(*rel.parts[1:]).as_posix();need(relative and relative!='.' and relative not in seen and len(seen)<1000 and 0<=item.size<=CAP,'Unsafe duplicate/oversize/count tar member');expanded+=item.size;need(expanded<=CAP,'Expanded payload exceeds cap');target=RUN.joinpath(*rel.parts[1:]);need(target.resolve().is_relative_to(RUN.resolve()),'Member escaped run');longpath(target.parent).mkdir(parents=True,exist_ok=True)
     h=hashlib.sha256();count=0
     with tar.extractfile(item) as incoming,longpath(target).open('xb') as output:
      for block in iter(lambda:incoming.read(8*1024*1024),b''):count+=len(block);need(count<=item.size,'Tar member expands beyond declaration');h.update(block);output.write(block)
     need(count==item.size,'Short tar member');seen[relative]={'bytes':count,'sha256':h.hexdigest()}
   # Consume beyond tar end to validate gzip footer/CRC and reject concatenated payload bytes.
   tail=gz.read();need(not tail or not any(tail),'Nonzero data after tar stream')
  need(raw.tell()==ARCHIVE.stat().st_size,'Archive not fully consumed')
 need(len(seen)==414 and 'export_manifest.json' in seen,'Expected413payload+manifest members');manifest=read(RUN/'export_manifest.json');rows=manifest['files'];files={r['path']:r for r in rows};need(len(files)==len(rows)==413 and set(seen)==set(files)|{'export_manifest.json'},'Exact manifest namespace differs')
 for name,row in files.items():need(seen[name]=={'bytes':row['bytes'],'sha256':row['sha256']},'Manifest payload hash mismatch '+name)
 need(sum(r['bytes'] for r in rows)==export['payload_bytes']==690472083,'Payload byte sum differs');need(manifest['protocol_sha256']==PROTOCOL and manifest['worker_exit']==0 and manifest['worker_timeout'] is False and manifest['worker_interruption'] is None and manifest['terminal_state_committed'] is False and manifest['terminal_state_pin'] is None and manifest['partial_files_retained']==[],'Manifest status differs')
 need(manifest['diagnostic_only'] is True and manifest['new_model_created'] is False and manifest['retained_teacher_label_arrays']==0 and manifest['original_base_weights_exported'] is False and manifest['parent_training_state_exported'] is False,'No checkpoint/teacher/source output scope')
 need(not any(n.endswith(('.pth','.pt','.partial')) for n in files),'Unexpected checkpoint/partial member');need(sha(RUN/'protocol.json')==PROTOCOL,'Extracted protocol differs');stage=ROOT/'outputs/completion_codeformer_same_codebook_probe_v1_packet/draft_001/cctv_completion_codeformer_same_codebook_probe_v1_vm';need(longpath(RUN/'protocol.json').read_bytes()==longpath(stage/'protocol.json').read_bytes(),'Published source protocol differs');need(sha(ARCHIVE)==ARCHIVE_SHA,'Archive changed during extraction')
 report={'format':'independent-same-codebook-probe-v1-actual-return-transport-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'archive':pin(ARCHIVE),'external_receipt':pin(EXPORT),'checksum':pin(CHECKSUM),'protocol_sha256':PROTOCOL,'regular_files_including_manifest':414,'manifest_payload_files':413,'manifest_payload_bytes':690472083,'whole_archive_SHA_and_gzip_CRC_consumption_pass':True,'all_payload_manifest_hashes_exact':True,'closed_namespace_and_safe_paths':True,'worker_exit':0,'terminal_state_committed':False,'partial_files_retained':[],'diagnostic_only':True,'new_model_created':False,'state_codes_pixels_visual_pending':True,'model_qualified':False,'result_root':str(RUN.resolve()),'files':seen,'Windows_access':{'extended_path_prefix_used':True,'canonical_result_namespace_unmodified':True,'partial_first_attempt_retained':str(RET/'partial_extraction_001_WINDOWS_MAX_PATH'),'partial_first_attempt_not_in_admitted_manifest':True},'local_actions':{'returned_code_executed':False,'Torch_imported':False,'neural_gradient_optimizer_VM_calls':0},'seconds':time.monotonic()-started}
 out=RET/'transport_audit.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(out),'results':str(RUN.resolve()),'regular_members':414,'payloads':413,'seconds':report['seconds']},indent=2),flush=True)
if __name__=='__main__':main()
