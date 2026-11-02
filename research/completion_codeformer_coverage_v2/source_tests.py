"""No neural/gradient/training calls: native serialization and pure source contracts."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,io,json,os,runpy,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.completion_codeformer_coverage_v2 import contracts as c
WORKER=ROOT/'scripts/cctv_completion_codeformer_coverage_v2_vm.py'
TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
class SourceOnlyTests(unittest.TestCase):
 def budget(self,folder,clock=None):
  out=Path(folder)/'out';out.mkdir();return c.OutputBudget(out,0,clock=clock or (lambda:0),disk_free=lambda:100*1024**3)
 def test_01_help_has_no_torch_import(self):
  real=__import__
  def guarded(name,*args,**kwargs):
   if name=='torch' or name.startswith('torch.'):raise AssertionError('No Torch may be imported by pureCLI')
   return real(name,*args,**kwargs)
  with patch('builtins.__import__',guarded),patch.object(sys,'argv',[str(WORKER),'--help']),contextlib.redirect_stdout(io.StringIO()):
   with self.assertRaises(SystemExit) as exit:runpy.run_path(str(WORKER),run_name='__main__')
  self.assertEqual(exit.exception.code,0)
 def test_02_python310_AST_and_contract_caps(self):
  ast.parse(WORKER.read_text(encoding='utf-8'),feature_version=(3,10));ast.parse(Path(c.__file__).read_text(encoding='utf-8'),feature_version=(3,10))
  self.assertEqual((c.CONTRACT['cycles'],c.CONTRACT['optimizer_steps_max'],c.CONTRACT['backward_calls_max']),(64,128,512))
  self.assertEqual((c.CONTRACT['forward_slots_planned'],c.CONTRACT['forward_slots_max']),(896,1000))
 def test_03_npz_native_seek_extent_and_replay(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);path=b.out/'teacher_labels/ref_rgb.npz';x=np.arange(4096,dtype=np.float32).reshape(1,16,16,16)
   with b.open(path) as stream:np.savez_compressed(stream,latent=x,codes=np.arange(256,dtype=np.int64))
   self.assertEqual(path.stat().st_size,b.sizes[path.resolve()]);self.assertFalse(hasattr(stream,'fileno'))
   with np.load(path,allow_pickle=False) as z:self.assertTrue(np.array_equal(z['latent'],x))
 def test_04_teacher_oversize_refused_and_partial_kept(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);path=b.out/'teacher_labels/ref_gray.npz'
   with patch.dict(c.CONTRACT,{'teacher_bytes':128}):
    with self.assertRaises(ValueError):
     with b.open(path) as stream:np.savez_compressed(stream,x=np.arange(2048,dtype=np.float32))
   self.assertTrue(path.is_file());self.assertLessEqual(path.stat().st_size,128)
 def test_05_PNG_native_writer_extent(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);path=b.out/'image.png';image=Image.fromarray(np.zeros((256,256,3),np.uint8))
   with b.open(path) as stream:image.save(stream,format='PNG')
   self.assertEqual(path.stat().st_size,b.sizes[path.resolve()]);self.assertLessEqual(path.stat().st_size,c.CONTRACT['PNG_bytes_each'])
 def test_06_pendingterminal_disk_reservation(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);b.disk_free=lambda:c.CONTRACT['stop_free_bytes']+512*1024**2
   with self.assertRaisesRegex(ValueError,'reserved terminal'):b.guard()
 def test_07_after_commit_receipt_in1800_window(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   clock=[0];b=self.budget(text,lambda:clock[0]);partial=b.out/'terminal_training_state.pth.partial';final=b.out/'terminal_training_state.pth'
   with b.open(partial,terminal=True) as stream:stream.write(b'tinyfixture')
   partial.rename(final);b.rename(partial,final);clock[0]=1700
   b.json(b.out/'results.json',{'complete':True},emergency=True);self.assertTrue(b.terminal_committed);self.assertEqual(b.total('terminal'),len(b'tinyfixture'))
   clock[0]=1800
   with self.assertRaisesRegex(ValueError,'deadline'):b.json(b.out/'late.json',{},emergency=True)
 def test_08_overwrite_refused(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'proof.json';b.json(p,{'kept':True});before=p.read_bytes()
   with self.assertRaises(FileExistsError):b.json(p,{'replaced':True})
   self.assertEqual(p.read_bytes(),before)
 def test_09_same_gates_group_regression_and_onepercent(self):
  base={'macro':{k:1. for k in c.METRICS},'geometry':{g:{k:1. for k in c.METRICS} for g in c.KINDS},'condition':{g:{k:1. for k in c.METRICS} for g in ['GRAY','RGB']}}
  good=json.loads(json.dumps(base))
  for group in [good['macro'],*good['geometry'].values(),*good['condition'].values()]:
   for k in group:group[k]=.98
  self.assertTrue(c.compare({'metrics':good},{'metrics':base},conditions=True)['prospective_numeric_screen_pass'])
  good['condition']['RGB']['raw_boundary_L1']=1.001
  self.assertFalse(c.compare({'metrics':good},{'metrics':base},conditions=True)['prospective_numeric_screen_pass'])
 def test_10_no_extra_model_scope_and_original_formulas(self):
  src=WORKER.read_text(encoding='utf-8')
  self.assertIn("mode.add_argument('--verify-transfer'",src);self.assertIn("stdout=subprocess.PIPE",src);self.assertNotIn("stdout=log",src)
  self.assertIn("need(not budget.terminal_committed",src);self.assertIn("parent=torch.load(parent_path,map_location='cpu',weights_only=True)",src)
  self.assertIn("loss=(ce+latent_loss)/4",src);self.assertIn("loss=(rgb_loss+seam)/4",src);self.assertNotIn('adain=True',src)
  self.assertIn("len(restored_cuda)==len(resume_rng['CUDA'])==1",src);self.assertIn("actual_nr[4]==nr['cached_gaussian']",src)
  self.assertIn("'forward_slots':528+counts['forward_slots']",src);self.assertIn("'actual_whole_model_forward_calls':448+model_forwards['calls']",src)
 def test_11_native_torch_serialization_callbacks_and_safe_read(self):
  import torch
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'terminal_training_state.pth.partial';x=torch.arange(32,dtype=torch.float32)
   self.assertFalse(x.requires_grad)
   with b.open(p,terminal=True) as stream:torch.save({'fixture':x},stream)
   self.assertEqual(p.stat().st_size,b.sizes[p.resolve()]);self.assertFalse(hasattr(stream,'fileno'))
   saved=torch.load(p,map_location='cpu',weights_only=True);self.assertTrue(torch.equal(saved['fixture'],x))
 def test_12_native_torch_oversize_before_write(self):
  import torch
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'terminal_training_state.pth.partial';x=torch.arange(4096,dtype=torch.float32)
   with patch.dict(c.CONTRACT,{'terminal_bytes':1024}):
    with self.assertRaises(Exception):
     with b.open(p,terminal=True) as stream:torch.save({'fixture':x},stream)
   self.assertTrue(p.is_file());self.assertLessEqual(p.stat().st_size,1024)
 def test_13_seek_rewrites_do_not_doublecount(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.budget(text);p=b.out/'proof.npz'
   with b.open(p) as stream:stream.write(b'abcd');stream.seek(0);stream.write(b'XY')
   self.assertEqual(b.total('array'),4);self.assertEqual(p.read_bytes(),b'XYcd')
 def test_14_metadata_deadline_doesnot_allow_work_aftercommit(self):
  tree=ast.parse(WORKER.read_text(encoding='utf-8'))
  worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
  forward=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
  self.assertEqual(forward.body[0].value.func.id,'need');self.assertIn('terminal_committed',ast.dump(forward.body[0]))
if __name__=='__main__':unittest.main(verbosity=2)