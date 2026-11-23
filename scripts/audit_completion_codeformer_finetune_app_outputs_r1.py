"""Independent audit of retained experimental completion app functionality.
Saved files/JSON/PNG/NPZ/ZIP only; never imports Torch or a model.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time,zipfile,sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'outputs/completion_codeformer_finetune_r1_app'
BROWSER=ROOT/'scratch/completion-codeformer-finetune-r1-playwright.json'
RETURN=ROOT/'outputs/completion_codeformer_finetune_v1_r1_return/cctv_completion_codeformer_finetune_v1_r1_results'
ASSETS=ROOT/'outputs/completion_codeformer_finetune_v1_packet/cctv_completion_codeformer_finetune_v1_vm'
MANIFEST=ROOT/'outputs/completion_codeformer_finetune_v1_r1_candidate/candidate_manifest.json'
MANIFEST_SHA='e8bc39c321d490ee335b683ea6947bc24df47c92e16db30a1412d6fc5380ea77'
WEIGHT_SHA='7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0'
PROTOCOL='8815ad168ab429bf4f30b8d719d3a6de6d1e2667fc38412c89c9c95e74e5ff3d'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pixels(path,mode='RGB'):
    with Image.open(path) as im:
        need(im.format=='PNG' and im.size==(256,256),'ExpectedPNG256');return np.array(im.convert(mode),copy=True)
def equal(a,b,label):need(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b),label)
def array_sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def main():
    start=time.monotonic();dest=APP/'independent_app_functionality_audit.json';need(not dest.exists(),'Preserve completed audit')
    browser=read(BROWSER);need(browser['complete'] and not browser['model_qualified'] and browser['experimental'] and browser['training_calls']==0,'Browser scope/complete differs')
    need(len(browser['tests'])==14 and browser['health']['device']=='cpu' and not browser['health']['restoration_enabled'],'Browser reported tests/health differ')
    need(sha(MANIFEST)==MANIFEST_SHA,'Candidate manifest changed');candidate=read(MANIFEST);weight=Path(candidate['checkpoint']['path']);need(sha(weight)==WEIGHT_SHA==candidate['checkpoint']['sha256'],'Candidate weight changed')
    need(candidate['protocol_sha256']==PROTOCOL and candidate['reconstructed_logical_sha256']=='55feaa296f5164a6bd12995b487c9366c5fe7778f0e9dbe8912fc4ee7c4122d6','Candidate provenance differs')
    protocol=read(RETURN/'protocol.json');need(sha(RETURN/'protocol.json')==PROTOCOL,'Sealed protocol differs')
    expected_names={'upload.bin','original.png','input.png','mask.png','result.png','request.json','receipt.json','app_audit.json','supervisor.json','worker.log','stages.npz'}
    sessions=[];outside_total=0;completed_ids=[];bound_source_ids=[];saved_nonempty=None;selected_case=None
    for number,response in enumerate(browser['requests']):
        ident=response['id'];path=APP/'sessions'/ident;completed_ids.append(ident)
        manifest=read(path/'manifest.json');need(len(manifest['files'])==11,'Expected11manifest files');names=set()
        bindings=[]
        for row in manifest['files']:
            member=Path(row['path']);need(member.parent.resolve()==path.resolve(),'Manifest path escaped session');need(member.name in expected_names,'Unexpected session member');names.add(member.name)
            actual=pin(member);need(actual==row,'Session manifest binding changed');bindings.append(actual)
        need(names==expected_names,'Manifest namespace differs')
        need({f.name for f in path.iterdir() if f.is_file()}==expected_names|{'manifest.json','evidence.zip'},'Session extras/missing files')
        bundle=path/'evidence.zip'
        with zipfile.ZipFile(bundle) as archive:
            infos=archive.infolist();z_names=[f.filename for f in infos]
            need(len(z_names)==13 and len(set(z_names))==13 and set(z_names)==expected_names|{'manifest.json','README.txt'},'ZIP namespace differs')
            need(archive.testzip() is None and sum(f.file_size for f in infos)<20*1024**2,'ZIP CRC/size bound')
            for info in infos:
                need(not info.is_dir() and not info.flag_bits&1 and Path(info.filename).name==info.filename,'ZIP directory/encrypted/traversal')
                expected_body=(b'One plausible estimate, not verified hidden identity.\nWhite mask pixels are generated; black pixels exactly retain input.png.\noriginal.png is the decoded, EXIF-oriented upload; upload.bin preserves uploaded bytes.\nNo automatic detector, restoration model or training was used.\n') if info.filename=='README.txt' else (path/info.filename).read_bytes()
                need(archive.read(info)==expected_body,'ZIP body differs from session/documentedREADME '+info.filename)
        request=read(path/'request.json');receipt=read(path/'receipt.json');supervisor=read(path/'supervisor.json');serveraudit=read(path/'app_audit.json')
        need(request['id']==ident and request['source_id']==response['source_id'],'Browser/request identifier differs');bound_source_ids.append(request['source_id'])
        need(request['mask_reviewed'] and request['visible_structure_confirmed'] and request['assisted'] and not request['automatic_detection_used'],'Reviewed-mask request flags differ')
        need(request['candidate_manifest_sha256']==receipt['candidate_manifest_sha256']==MANIFEST_SHA,'Wrong model manifest selected')
        need(request['training_calls']==request['restoration_calls']==0 and receipt['optimizer_updates']==receipt['gradient_calls']==0 and not receipt['optimizer_constructed'],'Forbidden additionalmodel/training work')
        need(request['worker_sha256']==sha(ROOT/'scripts/completion_codeformer_finetune_r1_worker.py'),'Worker source drift')
        need(request['process_guard_sha256']==sha(ROOT/'completion_process_guard_v1.py'),'Processguard changed')
        for relative,digest in receipt['sources_sha256'].items():need(sha(ROOT/relative)==digest,'Receipt source binding changed '+relative)
        for name,digest in receipt['inputs_sha256'].items():need(sha(path/name)==digest,'Input/request drift '+name)
        for name,digest in receipt['outputs_sha256'].items():need(sha(path/name)==digest,'Output drift '+name)
        source=pixels(path/'input.png');mask8=pixels(path/'mask.png','L');result=pixels(path/'result.png');need(np.isin(mask8,[0,255]).all(),'Binarymask differs');support=mask8==255
        with np.load(path/'stages.npz',allow_pickle=False) as z:
            need(set(z.files)=={'input_rgb','removal_mask','completed_rgb','output_rgb'},'Stagesfields differ');raw=z['completed_rgb'].copy()
            equal(z['input_rgb'],source,'Saved source differs');equal(z['removal_mask'],support.astype(np.uint8),'Saved reviewed support differs');equal(z['output_rgb'],result,'Saved deliveredbytes differ')
        need(raw.shape==(256,256,3) and raw.dtype==np.float32 and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1,'Invalid saved raw256')
        expected=np.rint(raw*np.float32(255)).clip(0,255).astype(np.uint8);expected[~support]=source[~support];equal(expected,result,'Composed raw256-to-PNG replay differs')
        equal(raw[~support],(source.astype(np.float32)/np.float32(255))[~support],'Outside rawFP32 changed');equal(result[~support],source[~support],'Outside sourcebyte changed')
        outside=int((~support).sum())*3;outside_total+=outside
        need(outside==receipt['outside_source_bytes'],'Outside bytecount differs')
        for key,array in [('input_rgb_sha256',source),('removal_mask_sha256',support.astype(np.uint8)),('completed_rgb_sha256',raw),('output_rgb_sha256',result),('outside_source_rgb_sha256',source[~support]),('outside_output_rgb_sha256',result[~support])]:need(array_sha(array)==receipt[key],'Array fingerprint differs '+key)
        calls=int(support.any());need(calls==receipt['neural_calls']==response['neural_calls'] and receipt['empty_mask_bypass']==(not support.any()),'Call/bypass receipt differs')
        need(receipt['complete'] and receipt['device']=='cpu' and receipt['precision']=='FP32' and receipt['threads']==4 and receipt['assisted_only'] and receipt['additional_mask_expansion']==0 and not receipt['visible_restoration_applied'],'Receipt input/output policy differs')
        need(supervisor['status']=='finished' and supervisor['returncode']==0 and supervisor['active_owned_processes']==0 and supervisor['cap_seconds']==120,'Ownedworker terminal status differs')
        need(supervisor['process_ownership']=='windows-job-suspended-before-resume' and supervisor['seconds']==response['seconds'] and 0<supervisor['seconds']<120,'Ownedworker runtime bound differs')
        need(serveraudit['outside_mask_changed_bytes']==0 and serveraudit['raw_png_replay_verified'] and serveraudit['active_owned_processes']==0,'Serveraudit differs')
        case_matches=[]
        for case in protocol['development_cases']:
            if np.array_equal(pixels(ASSETS/case['input256']),source) and np.array_equal(pixels(ASSETS/case['mask'],'L'),mask8):case_matches.append(case)
        need(len(case_matches)==1,'Exactly one fixed publicDEVcase must bind');case=case_matches[0]
        if number==0:
            need(case['id']=='00_cloth_mask_native' and support.sum()==18531 and calls==1,'Expected cloth comparison')
            provenance=receipt['provenance'];need(provenance['weights_sha256']==WEIGHT_SHA and provenance['candidate_manifest_sha256']==MANIFEST_SHA and provenance['protocol_sha256']==PROTOCOL,'Trained model provenance differs')
            need(provenance['logical_state_sha256']==candidate['reconstructed_logical_sha256'] and not provenance['model_qualified'] and not provenance['hidden_identity_verified'],'Model quality/scope differs')
            for relative,loaded in receipt['runtime']['loaded_canonical_sources'].items():need(Path(loaded).resolve().is_relative_to(ROOT.resolve()),'Imported source escapedcanonicalworktree')
            saved_nonempty=(source.copy(),support.copy(),raw.copy(),result.copy());selected_case=case
        else:
            need(case['family']=='clear_glasses_control' and not support.any() and calls==0,'Clear-glasses emptycontrol differs');equal(result,source,'Emptyclear-glasses allpixels differ');equal(raw,source.astype(np.float32)/np.float32(255),'Empty rawbypass differs')
            need(receipt['provenance']['checkpoint_loaded'] is False and receipt['runtime']['torch'] is None and receipt['runtime']['checkpoint'] is None,'Emptyloadedmodel receipt unexpected')
        sessions.append({'id':ident,'case':case,'session_files':bindings,'ZIP':pin(bundle),'ZIP_all13members_exact':True,'raw_stage_scope':'Saved composedFP32 RGB256; internal512 network output is not retained in appstages','raw256_to_PNG_all_exact':True,'outside_source_FP32_and_PNG_exact':True,'outside_source_RGB_bytes':outside,'empty_bypass_allpixels_exact':not bool(support.any()),'neural_calls_reported':calls,'source_policy':'Reviewed removal mask fixed; no additional support margin','supervisor':supervisor,'timing_worker':receipt['timing'],'candidate_provenance':receipt['provenance'],'source_import_locations_receipt':receipt['runtime'].get('loaded_canonical_sources',{})})
    downloads=[]
    for row in browser['downloads']:
        path=Path(row['path']);need(pin(path)==row,'Downloaded file pin changed')
        original=APP/'sessions'/completed_ids[0]/('evidence.zip' if path.suffix=='.zip' else 'result.png');need(path.read_bytes()==original.read_bytes(),'Browserdownload differs from server artifact');downloads.append({'download':pin(path),'server':pin(original),'all_bytes_equal':True})
    need(browser['refusal']['status']==422 and browser['refusal']['body']['error']=='Too little visible face remains. Choose a less-covered or clearer face crop.','Visibility refusal differs')
    directories={p.name for p in (APP/'sessions').iterdir() if p.is_dir()};expected_sessions=set(completed_ids+bound_source_ids)
    need(directories==expected_sessions,'Unexpected generated session afterrefusal')
    for ident in bound_source_ids:need(not any((APP/'sessions'/ident/name).exists() for name in ('supervisor.json','result.png','receipt.json','stages.npz')),'Preparedonly upload has generation artifacts')
    source,support,cpu_raw,cpu_result=saved_nonempty
    vm_folder=RETURN/'outputs/u0032/public_development'
    with np.load(vm_folder/(selected_case['id']+'.npz'),allow_pickle=False) as z:
        vm_input=z['input256'].copy();vm_mask=z['mask256'].copy();vm_raw=z['result256'][0].transpose(1,2,0).copy()
    equal(np.rint(vm_input[0].transpose(1,2,0)*255).astype(np.uint8),source,'CPU/VM observedinput mismatch');equal(vm_mask[0,0].astype(bool),support,'CPU/VM reviewed mask mismatch')
    vm_png=pixels(vm_folder/(selected_case['id']+'.png'));difference=np.abs(cpu_raw.astype(np.float64)-vm_raw.astype(np.float64));byte_difference=np.abs(cpu_result.astype(np.int16)-vm_png.astype(np.int16))
    cpu_vm={'case':selected_case['id'],'same_input_and_mask':True,'same_protocol_and_terminal_logical_state':True,'CPU_vs_VM_PNG_all_pixels_equal':bool(np.array_equal(cpu_result,vm_png)),'PNG_different_RGB_channels':int(np.count_nonzero(byte_difference)),'PNG_different_pixels':int(np.any(byte_difference,axis=2).sum()),'PNG_max_byte_difference':int(byte_difference.max()),'outside_support_different_channels':int(np.count_nonzero(byte_difference[~support])),'composed_FP32_max_absolute_error':float(difference.max()),'composed_FP32_mean_absolute_error_within_mask':float(difference[support].mean()),'PNG':{'CPU':pin(APP/'sessions'/completed_ids[0]/'result.png'),'VM':pin(vm_folder/(selected_case['id']+'.png'))},'interpretation':'Crossruntime savedoutput comparison only. CPU/GPU exact modelparity is not asserted; no additional inference was run.'}
    need(cpu_vm['outside_support_different_channels']==0,'CPU/VM visible preservation differs')
    need(not any(name=='torch' or name=='pretrained_completion' or name.startswith('third_party.codeformer') for name in sys.modules),'Audit imported modelruntime')
    report={'format':'completion-R1-independent-app-functional-output-audit-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'browser_report':pin(BROWSER),'candidate_manifest':pin(MANIFEST),'candidate_checkpoint':pin(weight),'sessions':sessions,'browser_downloads':downloads,'outside_source_RGB_bytes_exact':outside_total,'CPU_VM_selected_case_saved_comparison':cpu_vm,'browser_test_count':len(browser['tests']),'responsive_widths_reported':[375,768,1280],'negative_refusal':{'browser':browser['refusal'],'retained_session_namespace_exact':True,'no_generated_outputs_or_worker_receipts_for_extra_session':True,'limitation':'Browser422, matching session snapshot and sourcegate evidence; not a fresh replay or historical process trace.'},'runtime_limits':{'reference_machine_only':True,'minimum_RAM_certified':False,'maximum_request_seconds':120,'measured_nonempty_seconds':sessions[0]['supervisor']['seconds'],'measured_empty_seconds':sessions[1]['supervisor']['seconds'],'job_private_commit_scope':'Committed private bytes from JobObject including descendants; not RSS or minimumRAM certificate.'},'qualification':{'functional_development_app_pass':True,'quality_or_fullfamily_pass':False,'automatic_detector_qualified':False,'independent_FINAL_or_identity_accuracy':False,'model_qualified':False,'IMPLEMENTATION_promoted':False},'local_auditor_execution':{'model_loads':0,'Torch_imported':False,'neural_forwards':0,'gradients':0,'optimizer_steps':0,'network':0,'VM_calls':0},'audit_seconds':time.monotonic()-start}
    with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'audit':pin(dest),'sessions':len(sessions),'ZIP_members_per_bundle':13,'download_exact':True,'outside_bytes':outside_total,'cpu_vm':cpu_vm,'seconds':report['audit_seconds']}))
if __name__=='__main__':main()