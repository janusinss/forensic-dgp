"""Writer policy parity and bounded native serialization, no neural computation."""
from pathlib import Path
import ast,hashlib,io,json,os,sys,tempfile,time,unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.completion_codeformer_coverage_v2 import contracts as old
from research.completion_codeformer_writer_v1 import optimized_budget as new
TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
class WriterContracts(unittest.TestCase):
 def pair(self,folder):
  result=[]
  for name,cls in [('old',old.OutputBudget),('new',new.OutputBudget)]:
   out=Path(folder)/name;out.mkdir();b=cls(out,0,clock=lambda:0,disk_free=lambda:100*1024**3);result.append(b)
  return result
 def test_01_python310_no_nativefileno(self):
  source=Path(new.__file__).read_text(encoding='utf-8');ast.parse(source,feature_version=(3,10));self.assertNotIn('def fileno(',source);self.assertNotIn('torch',source)
 def test_02_exactbytes_seek_categorytotals_andrename(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    p=b.out/'terminal_training_state.pth.partial'
    with b.open(p,terminal=True) as stream:stream.write(b'0123456789');stream.seek(3);stream.write(b'ABC');stream.seek(0,2);stream.write(b'END')
    self.assertEqual(p.read_bytes(),b'012ABC6789END');self.assertEqual(b.total('terminal'),13);self.assertEqual(b.total(),13)
    final=b.out/'terminal_training_state.pth';p.rename(final);b.rename(p,final);self.assertTrue(b.terminal_committed);self.assertEqual(b.total('terminal'),13)
    if isinstance(b,new.OutputBudget):self.assertTrue(b.assert_ledger_consistent())
 def test_03_overwrite_cannotchangeledger_orbytes(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.pair(text)[1];p=b.out/'retained.json';b.json(p,{'kept':True});before=p.read_bytes();size=b.total()
   with self.assertRaises(ValueError):b.json(p,{'replaced':True})
   self.assertEqual(p.read_bytes(),before);self.assertEqual(b.total(),size);b.assert_ledger_consistent()
 def test_04_paths_escape_orunsupportedtypes_refused(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.pair(text)[1]
   for p in [b.out.parent/'escaped.json',b.out/'../escaped.json',b.out/'unknown.exe']:
    with self.subTest(path=str(p)),self.assertRaises(ValueError):b.open(p)
   self.assertFalse((b.out.parent/'escaped.json').exists())
 def test_05_total_guard_flags_match_old(self):
  scenarios=[('teacher','teacher_labels/a.npz','teacher_bytes'),('terminal','terminal_training_state.pth.partial','terminal_bytes'),('PNG','a.png','PNG_bytes_each'),('metadata','a.json','metadata_bytes')]
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   pairs=self.pair(text)
   for category,name,cap in scenarios:
    for b in pairs:
     p=b.out/name
     with b.open(p,terminal=category=='terminal') as stream:
      end=old.CONTRACT[cap]+1
      with self.subTest(type=type(b).__module__,category=category),self.assertRaises(ValueError):b.before_write(p,end,terminal=category=='terminal')
     self.assertTrue(p.is_file());self.assertEqual(p.stat().st_size,0)
 def test_06_disksame_pendingterminal_andwrite_floor(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    b.disk_free=lambda:old.CONTRACT['stop_free_bytes']+old.CONTRACT['terminal_bytes']
    with self.assertRaises(ValueError):b.guard()
    b.disk_free=lambda:old.CONTRACT['stop_free_bytes']+1
    p=b.out/'terminal_training_state.pth.partial'
    with b.open(p,terminal=True) as stream:
     with self.assertRaises(ValueError):stream.write(b'AB')
    self.assertEqual(p.stat().st_size,0)
 def test_07_deadlines_samework_terminal_emergency(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    clock=[1679.];b.clock=lambda:clock[0];b.guard();clock[0]=1680.
    with self.assertRaises(ValueError):b.guard()
    b.guard(terminal=True);b.guard(emergency=True);clock[0]=1800.
    with self.assertRaises(ValueError):b.guard(terminal=True)
    with self.assertRaises(ValueError):b.guard(emergency=True)
 def test_08_postcommit_metadata_no_secondterminalreserve(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    p=b.out/'terminal_training_state.pth.partial';final=b.out/'terminal_training_state.pth'
    with b.open(p,terminal=True) as stream:stream.write(b'0123')
    p.rename(final);b.rename(p,final);b.disk_free=lambda:old.CONTRACT['stop_free_bytes']+2*1024**2;b.clock=lambda:1700.
    b.json(b.out/'results.json',{'complete':True},emergency=True);self.assertTrue((b.out/'results.json').is_file())
 def test_09_externalmetadata_late_growth_guarded(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    external=b.out.parent/(b.out.name+'-external.log');external.write_bytes(b'x');b.metadata_external=[external];b.guard()
    with patch.dict(old.CONTRACT,{'metadata_bytes':b.supervisor_reserve}):
     with self.assertRaises(ValueError):b.guard()
 def test_10_metadata_failure_reserve_same(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    cap=old.CONTRACT['metadata_bytes']-b.supervisor_reserve
    p=b.out/'ordinary.json'
    with b.open(p) as stream:
     with self.assertRaises(ValueError):b.before_write(p,cap)
    p2=b.out/'failure.json'
    with b.open(p2,emergency=True) as stream:b.before_write(p2,cap,emergency=True)
 def test_11_teacher_oversize_preserves_prefix_andtotals(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   for b in self.pair(text):
    p=b.out/'teacher_labels/gray.npz'
    with patch.dict(old.CONTRACT,{'teacher_bytes':8}):
     with self.assertRaises(ValueError):
      with b.open(p) as stream:stream.write(b'1234');stream.write(b'56789')
    self.assertEqual(p.read_bytes(),b'1234');self.assertEqual(b.total('teacher'),4)
    if isinstance(b,new.OutputBudget):b.assert_ledger_consistent()
 def test_12_nativeNPZ_PNG_identical(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   pairs=self.pair(text);sha={};x=np.arange(2048,dtype=np.float32)
   for b in pairs:
    p=b.out/'teacher_labels/a.npz'
    with b.open(p) as stream:np.savez_compressed(stream,latent=x,codes=np.arange(256,dtype=np.int64))
    with np.load(p,allow_pickle=False) as a:self.assertTrue(np.array_equal(a['latent'],x))
    image=b.out/'a.png'
    with b.open(image) as stream:Image.fromarray(np.zeros((256,256,3),np.uint8)).save(stream,format='PNG')
    sha[b.out.name]=[hashlib.sha256(z.read_bytes()).hexdigest() for z in (p,image)]
    self.assertFalse(hasattr(stream,'fileno'))
    if isinstance(b,new.OutputBudget):b.assert_ledger_consistent()
   self.assertEqual(sha['old'],sha['new'])
 def test_13_renaming_category_orownedtarget_refused(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.pair(text)[1];p=b.out/'a.npz';q=b.out/'b.npz'
   with b.open(p) as stream:stream.write(b'a')
   with b.open(q) as stream:stream.write(b'b')
   with self.assertRaises(ValueError):b.rename(p,q)
   with self.assertRaises(ValueError):b.rename(p,b.out/'a.png')
   b.assert_ledger_consistent()
 def test_14_counted_namespace_totals_do_not_rescan(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.pair(text)[1]
   for i in range(8):
    with b.open(b.out/(str(i)+'.npz')) as stream:stream.write(b'abc')
   n=b.check_counts['canonical_paths'];self.assertEqual(b.total('array'),24);self.assertEqual(b.total(),24)
   for i in range(20):b.guard()
   self.assertEqual(n,b.check_counts['canonical_paths']);b.assert_ledger_consistent()
 def test_15_native_torch8MiB_readback_andbyteparity(self):
  import torch
  x=torch.arange(2*1024*1024,dtype=torch.float32);self.assertFalse(x.requires_grad)
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   out=Path(text);native=out/'native.pth';torch.save({'fixture':x},native.open('xb'))
   folder=out/'new';folder.mkdir();b=new.OutputBudget(folder,0,clock=lambda:0,disk_free=lambda:100*1024**3);p=folder/'terminal_training_state.pth.partial'
   before=b.check_counts.copy();started=time.perf_counter()
   with b.open(p,terminal=True) as stream:torch.save({'fixture':x},stream)
   seconds=time.perf_counter()-started
   self.assertFalse(hasattr(stream,'fileno'));self.assertEqual(b.total('terminal'),p.stat().st_size)
   self.assertEqual(hashlib.sha256(native.read_bytes()).hexdigest(),hashlib.sha256(p.read_bytes()).hexdigest())
   actual=torch.load(p,map_location='cpu',weights_only=True);self.assertTrue(torch.equal(actual['fixture'],x));b.assert_ledger_consistent()
   RESULTS['tiny_CPU_torch8MiB']={'seconds':seconds,'bytes':p.stat().st_size,'writecallbacks':b.check_counts['write_checks']-before['write_checks'],'canonical_paths':b.check_counts['canonical_paths']-before['canonical_paths'],'native_file_identical':True,'safe_CPU_readback':True,'model_calls':0,'gradients':0}
 def test_16_native_torchfailure_keeps_partial(self):
  import torch
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.pair(text)[1];p=b.out/'terminal_training_state.pth.partial'
   with patch.dict(old.CONTRACT,{'terminal_bytes':1024}):
    with self.assertRaises(Exception):
     with b.open(p,terminal=True) as stream:torch.save({'fixture':torch.arange(4096,dtype=torch.float32)},stream)
   self.assertTrue(p.is_file());self.assertLessEqual(p.stat().st_size,1024);self.assertFalse((b.out/'terminal_training_state.pth').exists());b.assert_ledger_consistent()
 def test_17_linked_output_root_refused_before_resolution(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   out=Path(text)/'root';out.mkdir();real=Path.is_symlink
   with patch.object(Path,'is_symlink',lambda path:path==out or real(path)):
    with self.assertRaisesRegex(ValueError,'Linked output root'):new.OutputBudget(out,0,clock=lambda:0,disk_free=lambda:100*1024**3)
 def test_18_linked_child_component_refused(self):
  with tempfile.TemporaryDirectory(dir=TMP) as text:
   b=self.pair(text)[1];real=Path.is_symlink
   with patch.object(Path,'is_symlink',lambda path:path.name=='linked' or real(path)):
    with self.assertRaisesRegex(ValueError,'Linked output path'):b.open(b.out/'linked'/'evidence.json')
   self.assertFalse((b.out/'linked').exists())
RESULTS={}
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(WriterContracts);started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
 report={'format':'completion-cached-writer-policy-parity-tests-v1','complete':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-started,'native_serialization':RESULTS,'limits_changed':False,'old_source_or_packet_changed':False,'model_calls':0,'gradient_calls':0,'optimizer_operations':0,'VM_calls':0}
 path=ROOT/'research/completion_codeformer_writer_v1/policy_tests_v2.json'
 with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({'receipt':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**report},indent=2))
 raise SystemExit(0 if result.wasSuccessful() else 1)