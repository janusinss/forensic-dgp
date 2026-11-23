"""Source/binding/request regressions for opt-in R1 completion; no model inference."""
import ast,base64,hashlib,importlib.util,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pretrained_completion_finetune_r1 as loader
import completion_codeformer_finetune_r1_web as web
spec=importlib.util.spec_from_file_location('r1_worker_contract',ROOT/'scripts/completion_codeformer_finetune_r1_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
TEMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,body):path.write_text(json.dumps(body),encoding='utf-8')
def data_url(image):
    out=io.BytesIO();image.save(out,format='PNG');return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()
def candidate_fixture(folder):
    candidate=folder/'candidate.pth';candidate.write_bytes(b'harmless candidate binding fixture')
    audit_root=folder/'return';audit_root.mkdir();audit=audit_root/'audit.json';write_json(audit,{'complete':True})
    body={'format':loader.MANIFEST_FORMAT,'complete':True,'protocol_sha256':loader.PROTOCOL_SHA,'original_checkpoint_sha256':loader.ORIGINAL_SHA,'reconstructed_logical_sha256':'a'*64,'checkpoint':{'path':str(candidate),'bytes':candidate.stat().st_size,'sha256':sha(candidate)},'independent_state_audit':{'path':str(audit),'sha256':sha(audit)},'model_qualified':False}
    manifest=folder/'manifest.json';write_json(manifest,body)
    return candidate,audit_root,audit,body,manifest
def request_fixture(folder,empty=False):
    yy,xx=np.mgrid[:256,:256];rgb=np.stack([xx,yy,(xx+yy)//2],-1).astype(np.uint8)
    mask=np.zeros((256,256),np.uint8)
    if not empty:mask[122:134,122:134]=255
    Image.fromarray(rgb).save(folder/'input.png');Image.fromarray(mask).save(folder/'mask.png')
    write_json(folder/'request.json',{'prepared_bounds':[0,0,256,256]})
    return rgb,mask
def output_fixture(folder):
    rgb,mask=request_fixture(folder);completed=rgb.astype(np.float32)/np.float32(255);completed[mask!=0]=np.float32(.5)
    final=np.rint(completed*255).astype(np.uint8);final[mask==0]=rgb[mask==0]
    Image.fromarray(final).save(folder/'result.png')
    np.savez_compressed(folder/'stages.npz',input_rgb=rgb,removal_mask=mask//255,completed_rgb=completed,output_rgb=final)
    inputs={name:sha(folder/name) for name in ('input.png','mask.png','request.json')}
    receipt={'complete':True,'neural_calls':1,'inputs_sha256':inputs,'outputs_sha256':{name:sha(folder/name) for name in ('result.png','stages.npz')}}
    write_json(folder/'receipt.json',receipt);return rgb,mask,completed,final,inputs,receipt
class R1SourceContracts(unittest.TestCase):
    def test_actual_admitted_candidate_manifest_without_torch(self):
        manifest=loader.CANDIDATE_MANIFEST;expected='e8bc39c321d490ee335b683ea6947bc24df47c92e16db30a1412d6fc5380ea77'
        self.assertEqual(sha(manifest),expected)
        body=loader.validate_candidate_manifest(expected)
        self.assertEqual(body['checkpoint']['sha256'],'7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0')
        self.assertNotIn('torch',sys.modules)
    def test_changed_manifest_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            candidate,ar,audit,body,manifest=candidate_fixture(Path(text))
            with patch.multiple(loader,CANDIDATE_PATH=candidate,CANDIDATE_MANIFEST=manifest,STATE_AUDIT_ROOT=ar):
                loader.validate_candidate_manifest(sha(manifest));expected=sha(manifest);manifest.write_bytes(manifest.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'SHA256'):loader.validate_candidate_manifest(expected)
    def test_changed_checkpoint_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            candidate,ar,audit,body,manifest=candidate_fixture(Path(text));candidate.write_bytes(b'changed')
            with patch.multiple(loader,CANDIDATE_PATH=candidate,CANDIDATE_MANIFEST=manifest,STATE_AUDIT_ROOT=ar):
                with self.assertRaisesRegex(ValueError,'checkpoint bytes'):loader.validate_candidate_manifest(sha(manifest))
    def test_changed_state_audit_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            candidate,ar,audit,body,manifest=candidate_fixture(Path(text));audit.write_text('{"complete":false}')
            with patch.multiple(loader,CANDIDATE_PATH=candidate,CANDIDATE_MANIFEST=manifest,STATE_AUDIT_ROOT=ar):
                with self.assertRaisesRegex(ValueError,'state admission'):loader.validate_candidate_manifest(sha(manifest))
    def test_arbitrary_checkpoint_path_refused(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            candidate,ar,audit,body,manifest=candidate_fixture(Path(text));other=Path(text)/'other.pth';other.write_bytes(candidate.read_bytes())
            with patch.multiple(loader,CANDIDATE_PATH=candidate,CANDIDATE_MANIFEST=manifest,STATE_AUDIT_ROOT=ar):
                with self.assertRaisesRegex(ValueError,'fixed reviewed'):loader.validate_candidate_manifest(sha(manifest),checkpoint=other)
    def test_wrong_protocol_or_qualification_refused(self):
        for key,value in [('protocol_sha256','0'*64),('model_qualified',True),('original_checkpoint_sha256','0'*64)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory(dir=TEMP) as text:
                candidate,ar,audit,body,manifest=candidate_fixture(Path(text));body[key]=value;write_json(manifest,body)
                with patch.multiple(loader,CANDIDATE_PATH=candidate,CANDIDATE_MANIFEST=manifest,STATE_AUDIT_ROOT=ar):
                    with self.assertRaises(ValueError):loader.validate_candidate_manifest(sha(manifest))
    def test_only_cpu_loader_no_optimizer_and_weights_only(self):
        source=(ROOT/'pretrained_completion_finetune_r1.py').read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load']
        self.assertEqual(len(calls),1);kw={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
        self.assertEqual(kw,{'map_location':'cpu','weights_only':True});self.assertNotIn('torch.optim',source)
        with self.assertRaisesRegex(ValueError,'CPU'):loader.load_candidate(manifest_sha='0'*64,device='cuda')
    def test_empty_request_bypasses_checkpoint_and_torch(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            folder=Path(text);rgb,mask=request_fixture(folder,empty=True)
            with patch.object(worker,'complete_nonempty',side_effect=AssertionError('No candidate load allowed')):
                receipt=worker.run_request(folder,checkpoint=folder/'absent.pth',candidate_manifest_sha='f'*64)
            self.assertTrue(receipt['empty_mask_bypass']);self.assertEqual(receipt['neural_calls'],0)
            self.assertEqual(receipt['candidate_manifest_sha256'],'f'*64)
            self.assertTrue(np.array_equal(np.array(Image.open(folder/'result.png')),rgb));self.assertNotIn('torch',sys.modules)
    def test_nonbinary_and_insufficient_masks_refused(self):
        for value in (127,255):
            with self.subTest(value=value),tempfile.TemporaryDirectory(dir=TEMP) as text:
                folder=Path(text);request_fixture(folder);Image.new('L',(256,256),value).save(folder/'mask.png')
                with self.assertRaises(ValueError):worker.load_request(folder)
        self.assertNotIn('torch',sys.modules)
    def test_saved_raw_png_replay(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            folder=Path(text);rgb,mask,completed,final,inputs,receipt=output_fixture(folder)
            self.assertEqual(web.audit_worker_result(folder,inputs,mask),receipt)
    def test_outside_raw_mutation_refused_despite_same_png(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as text:
            folder=Path(text);rgb,mask,completed,final,inputs,receipt=output_fixture(folder);completed[3,3]=np.float32(.5)
            np.savez_compressed(folder/'stages.npz',input_rgb=rgb,removal_mask=mask//255,completed_rgb=completed,output_rgb=final)
            receipt['outputs_sha256']['stages.npz']=sha(folder/'stages.npz');write_json(folder/'receipt.json',receipt)
            with self.assertRaisesRegex(ValueError,'Raw stage changed'):web.audit_worker_result(folder,inputs,mask)
    def test_guard_and_request_limit_reused(self):
        source=(ROOT/'completion_codeformer_finetune_r1_web.py').read_text()
        self.assertIn('from completion_process_guard_v1 import OwnedWorker',source);self.assertIn('WORKER_CAP = 120',source)
        self.assertIn('wait_owned_worker(process, self.shutdown_event, WORKER_CAP)',source)
        self.assertIn('process.kill()',source);self.assertIn('process.wait_tree(timeout=15)',source)
        self.assertIn("'--candidate-manifest-sha', self.candidate_manifest_sha",source)
    def test_existing_wrapper_sources_unchanged(self):
        self.assertEqual(sha(ROOT/'pretrained_completion.py'),'29f2d03d692d762d1d9b39c9747e7becf30a14399651e8c419de2a5b39cae9ce')
        self.assertEqual(sha(ROOT/'completion_codeformer_web_v1.py'),'be708101d0ded43a1ea0a1a2072c2b23fc6310d43a6f15e62ea9dded766daa6e')
    def test_editor_script_style_unchanged_and_experimental_label(self):
        old=(ROOT/'templates/completion_codeformer_v1.html').read_text(encoding='utf-8');new=(ROOT/'templates/completion_codeformer_finetune_r1.html').read_text(encoding='utf-8')
        for tag in ('script','style'):
            self.assertEqual(old.split('<'+tag+'>')[1].split('</'+tag+'>')[0],new.split('<'+tag+'>')[1].split('</'+tag+'>')[0])
        self.assertIn('OUR TRAINED COMPLETION R1 · EXPERIMENTAL',new);self.assertIn('Original CodeFormer is retained as the comparison baseline',new)
if __name__=='__main__':unittest.main()