"""Independent static R1 completion app source/admission audit; no neural imports."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,importlib.util,json,re,sys,time
BASE=Path(__file__).resolve().parents[1]
DEST=BASE/'outputs/completion_codeformer_finetune_v1_r1_candidate'
PINS={
'pretrained_completion_finetune_r1.py':'d2ab3890869679ee7501f86f7c7fa1257e238adf5ee6d4e1fc956ec5555f860e',
'scripts/completion_codeformer_finetune_r1_worker.py':'acf3e2a65099dc6117ff4f5e2079e4fcf15aeba0f000772f4f663a12304eccc4',
'completion_codeformer_finetune_r1_web.py':'13ca83675752b115b2f79a8e639cb6af7b7e58bb5e81b01c352bf6f0fdb33045',
 'templates/completion_codeformer_finetune_r1.html':'785bd506026ea9dbe2920c4840e022dcc713c16467ee2a795a34553b8c4c279b'}
MANIFEST='e8bc39c321d490ee335b683ea6947bc24df47c92e16db30a1412d6fc5380ea77'
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):d.update(b)
    return d.hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def funcs(path):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
def main():
    started=time.monotonic();source_pins=[]
    for name,expected in PINS.items():
        path=BASE/name;need(path.is_file() and not path.is_symlink() and sha(path)==expected,'App source drift: '+name);source_pins.append(pin(path))
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
    shared_admission=BASE/'outputs/completion_codeformer_manual_v1/source_admission.json'
    need(sha(shared_admission)=='188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab','Original source admission changed')
    shared=json.loads(shared_admission.read_text(encoding='utf-8'));shared_pins=[]
    for row in shared['source_files']:
        path=Path(row['path']);need(path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Shared original source modified '+str(path));shared_pins.append(pin(path))
    need(sha(BASE/'completion_codeformer_web_v1.py')=='be708101d0ded43a1ea0a1a2072c2b23fc6310d43a6f15e62ea9dded766daa6e','Original app web source changed')
    need(sha(BASE/'completion_process_guard_v1.py')=='1277b0fde51c35337a0391d4089b193aa837f21c7e00ecc9ee4ed7ecd5f687c0','Original owned-worker guard changed')
    worker_new=funcs(BASE/'scripts/completion_codeformer_finetune_r1_worker.py');worker_old=funcs(BASE/'scripts/completion_codeformer_worker_v1.py')
    equal_worker=['sha','array_sha','check_time','load_request','visibility_check256','encode_output','canonical_module','verify_loaded_sources','write_json_exclusive']
    for name in equal_worker:need(worker_new[name]==worker_old[name],'Unintended input/visible/guard worker delta: '+name)
    web_new=funcs(BASE/'completion_codeformer_finetune_r1_web.py');web_old=funcs(BASE/'completion_codeformer_web_v1.py')
    equal_web=['need','sha','pin','save_json','decode_data_url','visibility','audit_worker_result','wait_owned_worker']
    for name in equal_web:need(web_new[name]==web_old[name],'Unintended app decode/replay/ownership delta: '+name)
    new_html=(BASE/'templates/completion_codeformer_finetune_r1.html').read_text(encoding='utf-8');old_html=(BASE/'templates/completion_codeformer_v1.html').read_text(encoding='utf-8')
    for tag in ('script','style'):
        need(re.search('<'+tag+'>(.*?)</'+tag+'>',new_html,re.S).group(1)==re.search('<'+tag+'>(.*?)</'+tag+'>',old_html,re.S).group(1),'Unintended editor/async/style delta')
    need(not any(text in new_html for text in ('Â·','Ã—','â€¦')),'R1 HTML mojibake remains')
    need('Our trained completion R1' in new_html and 'experimental' in new_html and 'not verified recovery of identity' in new_html,'Missing candidate/identity scope label')
    need('mask-reviewed' in new_html and 'structure-confirmed' in new_html and 'mask_reviewed:true' in new_html and 'visible_structure_confirmed:true' in new_html,'Reviewed-mask confirmations missing')
    # The actual manifest validator is stdlib-only and hashes the admitted file;
    # this performs no checkpoint deserialization, Torch import or model call.
    need('torch' not in sys.modules,'Audit unexpectedly already imported Torch')
    spec=importlib.util.spec_from_file_location('independent_r1_loader_source_review',BASE/'pretrained_completion_finetune_r1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metadata=module.validate_candidate_manifest(MANIFEST)
    need('torch' not in sys.modules,'Manifest validation imported Torch')
    need(metadata['checkpoint']['sha256']=='7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0' and metadata['checkpoint']['bytes']==370764381,'Unexpected actual candidate pin')
    need(metadata['independent_state_audit']['sha256']=='e920c3ddd9b8a85b50fd6f324bde8760c13c6df805f6be61d1e4393472d59b06','Wrong state admission')
    new_sources='\n'.join((BASE/name).read_text(encoding='utf-8') for name in PINS if name.endswith('.py'))
    need(not any(marker in new_sources for marker in ('torch.optim.','.backward(','.load_codeformer(', 'pretrained_completion.WEIGHTS_SHA256 =','pretrained_completion.load_codeformer =')),'Unexpected optimizer/baseline mutation path')
    need('WORKER_CAP = 120' in new_sources and 'MAX_SECONDS = 120' in new_sources and "OwnedWorker(command" in new_sources and "('127.0.0.1',args.port)" in new_sources,'Owned/local/timeout boundary absent')
    for item in source_pins:need(sha(Path(item['path']))==item['sha256'],'Source changed during review')
    audit={'format':'completion-codeformer-finetune-r1-independent-app-source-audit-v1','complete':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'sources':source_pins,'candidate_manifest':pin(DEST/'candidate_manifest.json'),'candidate_checkpoint':metadata['checkpoint'],'independent_state_audit':metadata['independent_state_audit'],'shared_original_sources':shared_pins,'original_web_source':pin(BASE/'completion_codeformer_web_v1.py'),'owned_worker_guard':pin(BASE/'completion_process_guard_v1.py'),'unchanged_AST_functions':{'worker':equal_worker,'web':equal_web},'original_editor_JS_and_CSS_exact':True,'HTML_encoding_corrected':True,'review_findings':{'loading':'CPU safe weights_only; exact497 finiteFP32state and logical hash; strictload; separate network instance using unchanged original wrapper; eval/frozen; no optimizer','manifest_boundary':'Explicit operator-supplied CLI SHA pins exacte8bc manifest. Actual validator checked path/size/SHA/protocol/base/state-admission; HTTP cannot override checkpoint or SHA. Loader does not hardcode e8bc literal. Root launch/admission must retain this explicit SHA.','no_global_baseline_contamination':'No original pretrained loader or global weight hash modified; candidate module separate; every request runs a new canonical child process, verifying imported origins before forward.','empty_bypass':'Worker validates prepared input/padding/visibility first, then returns exactsource bytes without Torch or checkpoint access. Server startup requires candidate assets and binds manifest; this is distinct from zero neural calls for empty worker.','reviewed_mask':'Two explicit confirmations; originalpreparedpixels+binary256mask fixed, no additional expansion, padding excluded; empty skipsmodel; approximatevisibility rejects insufficient support','neural_boundary':'Exactly one hooked CodeFormer net forward for nonempty; requires_gradFalse/eval/inference_mode, CPU4threads, no detection/restoration/training','ownership':'Private WindowsJob suspended-before-assignment inherited guard; only requestthread owns lifecycle;120s wait bound, kill/wait/close in ownedtree; shared shutdown only signals event','downloads':'Sessionregistry and fixed asset allowlist/hash, raw/PNG replay before publication, upload/original/input/mask/result +metadata/receipt/NPZ/log/bundle retained; new UUID per generation','UI_scope':'R1 experimental; one plausible hidden estimate; exactoutside prepared256source; visible appearance/fullfamilies/final identity not qualified'},'local_execution':{'manifest_file_validation':True,'checkpoint_deserializations':0,'Torch_imports':0,'neural_forwards':0,'gradients':0,'optimizers':0,'server_launches':0,'VM_calls':0},'admission_scope':'Source approval for one separately authorized opt-in CPU app test after root visual and pixel review. Not app runtime/usefulness/portable deployment qualification.','model_qualified':False,'full_covering_family_qualification':False,'portable_CPU_installation_verified':False,'seconds':time.monotonic()-started}
    target=DEST/'independent_app_source_audit.json'
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'complete':True,'audit':pin(target),'neural_calls':0,'Torch_imports':0,'seconds':audit['seconds']},indent=2))
if __name__=='__main__':main()