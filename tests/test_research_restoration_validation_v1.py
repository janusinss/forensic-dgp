"""CPU validation/lineage checks; no model, Torch, browser or VM execution."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import research_restoration_web as web


def actual_validation_module(path,names,namespace):
    """Compile the unchanged current pure functions, avoiding unrelated Torch imports."""
    source=ast.parse(path.read_bytes())
    selected=[node for node in source.body if isinstance(node,ast.FunctionDef)and node.name in names]
    if {node.name for node in selected}!=set(names):raise AssertionError('Actual validation function scope differs')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return types.SimpleNamespace(**{name:namespace[name]for name in names})


crop=actual_validation_module(ROOT/'dgp_face_workflow_v3.py',('decode_crop','review_input'),
    {'np':np,'Image':Image,'ImageOps':ImageOps,'io':io,'UPLOAD_LIMIT':10*1024**2})
geometry=actual_validation_module(ROOT/'dgp_face_restoration.py',('prepare_crop',),
    {'np':np,'Image':Image,'hashlib':hashlib,'POLICY':'dgp256-observed-canvas-v1'})


def image_bytes(mode='RGB',size=(12,17),fmt='PNG'):
    stream=io.BytesIO();Image.new(mode,size,128).save(stream,format=fmt);return stream.getvalue()


class ResearchValidation(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=ROOT/'scratch')
        self.folder=Path(self.temporary.name)
        self.module_patch=patch.dict(sys.modules,{'dgp_face_workflow_v3':crop,'dgp_face_restoration':geometry})
        self.module_patch.start()
        self.simulated_links=set();self.stat_patch=None

    def tearDown(self):
        if self.stat_patch:self.stat_patch.stop()
        self.module_patch.stop();self.temporary.cleanup()

    def test_input_only_review_refuses_before_weight_or_neural_loading(self):
        engine=web.Comparison()
        with patch.object(engine,'load',side_effect=AssertionError('Weights must not load')):
            for status in ('','needs_clearer','out_of_scope','untrusted'):
                with self.subTest(status=status),self.assertRaises(ValueError):engine.generate(image_bytes(),status,'both')
        self.assertEqual(engine.models,{})

    def test_arbitrary_or_unknown_selection_refused_before_decode(self):
        engine=web.Comparison()
        for selection in ('../../weights','C:/secret.pth','AUTO','original,r2',''):
            with self.subTest(selection=selection),self.assertRaises(ValueError):engine.generate(b'bad','usable',selection)
        self.assertEqual(engine.models,{})

    def test_empty_and_oversize_uploads_refused(self):
        for payload in (b'',b'x'*(10*1024**2+1)):
            with self.assertRaises(ValueError):crop.decode_crop(payload)

    def test_non_image_and_unsupported_image_refused(self):
        for payload in (b'not an image',image_bytes(fmt='GIF'),b'\x89PNG\r\n\x1a\ncorrupt'):
            with self.assertRaises(ValueError):crop.decode_crop(payload)

    def test_dimension_limit_refused_before_full_decode(self):
        with self.assertRaises(ValueError):crop.decode_crop(image_bytes(size=(4097,1)))

    def test_geometry_preserves_aspect_support_and_padding(self):
        rgb=np.full((40,20,3),64,dtype=np.uint8)
        canvas,support,removal,meta=geometry.prepare_crop(rgb)
        self.assertEqual(canvas.shape,(256,256,3));self.assertEqual(support.shape,(256,256))
        self.assertEqual(meta['native_center_offset'],[10,0]);self.assertEqual(meta['native_size'],[20,40])
        self.assertEqual(meta['observed_bounds_256'],[64,0,192,256]);self.assertFalse(removal.any())
        self.assertTrue(np.all(canvas[:,0]==128));self.assertTrue(np.all(canvas[:,-1]==128))
        self.assertTrue(np.all(canvas[:,100:156]==64))

    def test_geometry_rejects_wrong_dtype_channels_empty_and_bad_mask(self):
        for rgb in (np.zeros((10,10,3),np.float32),np.zeros((10,10),np.uint8),np.zeros((0,10,3),np.uint8)):
            with self.assertRaises(ValueError):geometry.prepare_crop(rgb)
        with self.assertRaises(ValueError):geometry.prepare_crop(np.zeros((10,10,3),np.uint8),np.zeros((9,10),np.uint8))

    def test_png_serialization_returns_exact_RGB_pixels(self):
        rgb=np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3)
        decoded=np.asarray(Image.open(io.BytesIO(web.png(rgb))).convert('RGB'))
        self.assertTrue(np.array_equal(rgb,decoded))

    def valid_candidate(self):
        p=self.folder/'codeformer_role_preserving_r2.pth';p.write_bytes(b'inert model fixture')
        receipt={'complete':True,'model_qualified':False,'source_checkpoint_sha256':web.FINAL_CHECKPOINT_SHA,
            'model_tensor_sha256':web.FINAL_STATE_SHA,'protocol_sha256':web.PROTOCOL_SHA,
            'export_bytes':p.stat().st_size,'export_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        rp=self.folder/'model_export_v1.json';rp.write_text(json.dumps(receipt),encoding='utf-8')
        return p,rp,receipt

    def test_export_lineage_invalid_or_accepted_claim_refused_without_Torch(self):
        engine=web.Comparison();_,rp,receipt=self.valid_candidate()
        with patch.object(web,'MODEL_DIR',self.folder):
            for key,value in (('complete',False),('model_qualified',True),('source_checkpoint_sha256','0'*64),
                              ('model_tensor_sha256','0'*64),('protocol_sha256','0'*64)):
                with self.subTest(key=key):
                    rp.write_text(json.dumps({**receipt,key:value}),encoding='utf-8')
                    with self.assertRaises(ValueError):engine.candidate_file()

    def test_actual_inference_file_size_or_hash_changes_refused(self):
        engine=web.Comparison();p,rp,receipt=self.valid_candidate()
        with patch.object(web,'MODEL_DIR',self.folder):
            self.assertEqual(engine.candidate_file()[0],p)
            p.write_bytes(b'changed model bytes')
            with self.assertRaises(ValueError):engine.candidate_file()
            p.write_bytes(b'inert model fixture');rp.write_text(json.dumps({**receipt,'export_sha256':'0'*64}),encoding='utf-8')
            with self.assertRaises(ValueError):engine.candidate_file()

    def test_architecture_mutation_is_refused_before_torch_or_model_import(self):
        engine=web.Comparison();self.valid_candidate();fake=types.SimpleNamespace(load=lambda *a,**k:self.fail('Torch load forbidden'))
        with patch.object(web,'MODEL_DIR',self.folder),patch.object(web,'ARCHITECTURE_PINS',{'codeformer_arch.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'architecture source'):engine.load('r2',fake)

    def test_architecture_pins_equal_immutable_VM_parent_sources(self):
        p=json.loads((ROOT/'outputs/cctv_codeformer_role_preserving_training_v1_r2_vm/protocol.json').read_bytes())
        for name,pin in web.ARCHITECTURE_PINS.items():
            relative='third_party/codeformer/'+name
            self.assertEqual(pin,p['dependencies'][relative]['sha256'])
            self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),pin)

    def hardlink(self,source,destination):
        try:os.link(source,destination)
        except OSError:
            # Restricted Windows sessions can forbid creating new hardlinks.
            # Exercise the actual guard with an explicitly simulated nlink2
            # stat; do not claim this fallback created a physical hardlink.
            destination.write_bytes(source.read_bytes())
            self.simulated_links.update((source.absolute(),destination.absolute()))
            if self.stat_patch is None:
                original_stat=Path.stat
                class LinkedStat:
                    st_nlink=2
                    def __init__(self,value):self.original=value
                    def __getattr__(self,name):return getattr(self.original,name)
                def with_link_count(path,*args,**kwargs):
                    value=original_stat(path,*args,**kwargs)
                    if path.absolute()in self.simulated_links:
                        return LinkedStat(value)
                    return value
                self.stat_patch=patch.object(Path,'stat',with_link_count);self.stat_patch.start()
        self.assertGreater(source.stat().st_nlink,1)

    def test_arbitrary_hardlinked_file_rejected_even_with_benchmark_flag(self):
        p=self.folder/'unapproved.pth';p.write_bytes(b'fixture');self.hardlink(p,self.folder/'alias.pth')
        with self.assertRaisesRegex(ValueError,'ordinary'):web.sha(p,allow_benchmark_hardlinks=True)

    def test_fixed_benchmark_hardlink_requires_explicit_opt_in(self):
        p=self.folder/'fixed-original.pth';p.write_bytes(b'fixture');self.hardlink(p,self.folder/'original-alias.pth')
        expected=hashlib.sha256(p.read_bytes()).hexdigest()
        with patch.object(web,'ORIGINAL',p):
            with self.assertRaisesRegex(ValueError,'ordinary'):web.sha(p)
            self.assertEqual(web.sha(p,allow_benchmark_hardlinks=True),expected)

    def test_candidate_hardlinks_and_receipt_hardlinks_remain_rejected(self):
        engine=web.Comparison();p,rp,_=self.valid_candidate()
        with patch.object(web,'MODEL_DIR',self.folder):
            self.hardlink(p,self.folder/'candidate-alias.pth')
            with self.assertRaisesRegex(ValueError,'ordinary'):engine.candidate_file()
        self.folder.joinpath('candidate-alias.pth').unlink()
        with patch.object(web,'MODEL_DIR',self.folder):
            self.hardlink(rp,self.folder/'receipt-alias.json')
            with self.assertRaisesRegex(ValueError,'ordinary'):engine.candidate_file()

    def test_cached_benchmark_still_checks_actual_hash_before_reuse(self):
        engine=web.Comparison();p=self.folder/'fixed-original.pth';p.write_bytes(b'untouched fixture')
        self.hardlink(p,self.folder/'benchmark-alias.pth');expected=hashlib.sha256(p.read_bytes()).hexdigest()
        marker=object();engine.models['original']=marker;engine.pins['original']={'weights_sha256':expected}
        fake=types.SimpleNamespace(load=lambda *a,**k:self.fail('Cached model must not be loaded'))
        with patch.object(web,'ORIGINAL',p),patch.object(web,'ORIGINAL_SHA',expected):
            self.assertIs(engine.load('original',fake)[0],marker)
            p.write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError,'fingerprint'):engine.load('original',fake)

    def test_actual_bundle_block_keeps_upload_and_benchmark_names_unique(self):
        # Execute only the current pure serialization block with deterministic
        # pixel fixtures. No inference helper or torch/model stub is involved.
        module=ast.parse((ROOT/'research_restoration_web.py').read_bytes())
        cls=next(n for n in module.body if isinstance(n,ast.ClassDef)and n.name=='Comparison')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='generate')
        protected=next(n for n in method.body if isinstance(n,ast.Try))
        start=next(i for i,n in enumerate(protected.body)if isinstance(n,ast.Assign)and
            any(isinstance(t,ast.Name)and t.id=='bundle'for t in n.targets))
        serialization=protected.body[start:]
        rgb=np.full((19,13,3),27,np.uint8);canvas=np.full((256,256,3),128,np.uint8)
        observed=np.zeros((256,256),np.bool_);observed[:,64:192]=True
        for arms in (('original',),('r2',),('original','r2')):
            outputs={arm:{'pixels':np.full_like(canvas,80+i),'raw':np.full((256,256,3),.25+i*.1,np.float32)}for i,arm in enumerate(arms)}
            namespace={'io':io,'zipfile':zipfile,'json':json,'np':np,'png':web.png,'data_url':lambda p,mime='image/png':p,
                'rgb':rgb,'canvas':canvas,'observed':observed,'outputs':outputs,'metadata':{'selection':list(arms),'model_qualified':False}}
            function=ast.FunctionDef(name='serialize_actual',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
                body=serialization,decorator_list=[])
            tree=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
            exec(compile(tree,str(ROOT/'research_restoration_web.py'),'exec'),namespace)
            response=namespace['serialize_actual']()
            with zipfile.ZipFile(io.BytesIO(response['bundle']))as archive:
                names=archive.namelist();self.assertEqual(len(names),len(set(names)))
                self.assertEqual(set(names),{'original-upload.png','resize-control.png','observed-support.png','processing.json'}|
                    {arm+suffix for arm in arms for suffix in ('.png','-raw.npy')})
                self.assertTrue(np.array_equal(np.asarray(Image.open(archive.open('original-upload.png'))),rgb))
                self.assertTrue(np.array_equal(np.asarray(Image.open(archive.open('resize-control.png'))),canvas))
                for arm in arms:
                    self.assertTrue(np.array_equal(np.asarray(Image.open(archive.open(arm+'.png'))),outputs[arm]['pixels']))
                    self.assertTrue(np.array_equal(np.load(io.BytesIO(archive.read(arm+'-raw.npy')),allow_pickle=False),outputs[arm]['raw']))


if __name__=='__main__':
    (ROOT/'scratch').mkdir(exist_ok=True)
    unittest.main()
