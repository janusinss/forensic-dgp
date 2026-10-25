"""Fault-injection tests for saved-data parsing/accounting; no Torch runtime."""
import ast
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'scripts/audit_u96_decoder_readout_state_v1.py'
TORCH_PRESENT_BEFORE_IMPORT='torch'in sys.modules
spec=importlib.util.spec_from_file_location('readout_saved_state_audit_under_test',PATH)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
TORCH_PRESENT_AFTER_IMPORT='torch'in sys.modules


def npz_bytes(state):
    b=io.BytesIO();np.savez(b,**state);return b.getvalue()


class SavedStateAuditTests(unittest.TestCase):
    def test_import_is_inert_and_no_training_operations(self):
        tree=ast.parse(PATH.read_text(encoding='utf-8'))
        topimports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        names=[alias.name for node in topimports for alias in node.names]
        self.assertNotIn('torch',names)
        forbidden={'backward','step','Adam','CodeFormer','build_restorer','cuda','to'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,forbidden)
        self.assertEqual(TORCH_PRESENT_BEFORE_IMPORT,TORCH_PRESENT_AFTER_IMPORT)

    def test_exact_two_tensor_census_and_npz_roundtrip(self):
        shapes=audit.shape_schema();self.assertEqual(sum(np.prod(s)for s in shapes.values()),1731)
        state={n:np.arange(np.prod(shape),dtype=np.float32).reshape(shape)for n,shape in shapes.items()}
        result=audit.read_npz(npz_bytes(state),np)
        for n in state:np.testing.assert_array_equal(state[n],result[n])
        self.assertEqual(audit.canonical(state),audit.canonical(dict(reversed(list(state.items())))))

    def test_npz_rejects_wrong_shape_dtype_object_or_nonfinite(self):
        original={n:np.zeros(s,np.float32)for n,s in audit.shape_schema().items()}
        name=next(iter(original))
        for changed in(np.zeros((4,),np.float32),np.zeros((3,),np.float64),np.array([None,None,None],object),
                       np.array([0,np.nan,0],np.float32)):
            altered=dict(original);altered[name]=changed
            with self.assertRaises(ValueError):audit.read_npz(npz_bytes(altered),np)
        with self.assertRaises(ValueError):audit.read_npz(npz_bytes({name:original[name]}),np)

    def test_zip_rejects_path_traversal_duplicates_and_symlink(self):
        for names in(('a/../bad','x'),('same','same')):
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w')as z:
                for n in names:z.writestr(n,b'x')
            with self.assertRaises(ValueError):audit.zip_guard(b.getvalue(),'pt')
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w')as z:
            info=zipfile.ZipInfo('archive/data.pkl');info.external_attr=(0o120777<<16)
            z.writestr(info,b'x')
        with self.assertRaises(ValueError):audit.zip_guard(b.getvalue(),'pt')

    def test_strict_json_duplicate_nan_overflow(self):
        for data in(b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}'):
            with self.assertRaises(ValueError):audit.strict_json(data)
        self.assertEqual(audit.strict_json(b'{"a":[1,2.0,null]}'),{'a':[1,2.,None]})

    def test_declared_counter_order(self):
        c0=audit.expected_counts(0);self.assertEqual((c0['full_forward_attempts'],c0['candidate_snapshot_forwards']),(176,56))
        c4=audit.expected_counts(4);self.assertEqual(c4['candidate_snapshot_forwards'],112)
        c4snap=audit.expected_counts(4,after_snapshot=True);self.assertEqual(c4snap['candidate_snapshot_forwards'],168)
        c64=audit.expected_counts(64,after_snapshot=True)
        self.assertEqual((c64['full_forward_returns'],c64['backward_returns'],c64['optimizer_updates']),(712,256,64))
        self.assertEqual(c64['frozen_U96_adapter_site_returns'],2848)
        for value in(0.0,False,65,-1):
            with self.assertRaises(ValueError):audit.expected_counts(value)

    def test_quality_stop_is_not_screen_success_and_pending_unknown(self):
        status={'counters':{'optimizer_updates':64,'optimizer_update_attempts':64},'optimizer_outcome_unknown':False}
        value=audit.outcome(status,False,{'status':'QUALITY_STOP'},None,{'timeout':False})
        self.assertEqual(value['scientific_status'],'QUALITY_STOP');self.assertFalse(value['model_qualified'])
        value=audit.outcome(status,True,{'status':'QUALITY_STOP'},None,{'timeout':False})
        self.assertTrue(value['optimizer_outcome_unknown'])
        self.assertEqual(value['scientific_status'],'INTERRUPTED_OR_UNRECORDED_OUTCOME')
        with self.assertRaises(ValueError):audit.outcome(status,False,{}, {},None)

    def test_adam_saved_array_recurrence_and_tamper_sensitivity(self):
        p=np.array([.1,-.2,.01],np.float32);g=np.array([.3,-.4,.2],np.float32);zero=np.zeros(3,np.float32)
        m,v,result=audit.adam_predict(p,g,zero,zero,1,1.25e-5,np)
        np.testing.assert_allclose(m,.1*g,rtol=1e-6)
        np.testing.assert_allclose(v,.01*g*g,rtol=1e-6)
        self.assertEqual(audit.comparison(result,result,np,'parameter_update'),(0.,0))
        bad=result.copy();bad[0]+=.001
        self.assertEqual(audit.comparison(bad,result,np,'parameter_update')[1],1)
        m2,v2,p2=audit.adam_predict(result,-g,m,v,2,2.5e-5,np)
        np.testing.assert_allclose(m2,.9*m-.1*g,rtol=1e-6,atol=1e-9)
        self.assertTrue(np.isfinite(p2).all())

    def test_scalar_clean_loss_roles_and_bad_teacher_keys(self):
        terms={'image':.1,'adjacent':.02}
        for role in audit.ROLES[1:]:self.assertAlmostEqual(audit.scalar_objective(role,terms),.105)
        local={'image_inside':.1,'image_outside':.2,'adjacent_inside':.02,'adjacent_outside':.04}
        self.assertAlmostEqual(audit.scalar_objective(audit.ROLES[0],local),.1575)
        with self.assertRaises(ValueError):audit.scalar_objective('clear',{'U96_image_anchor_L1':.1})

    def test_archive_scoped_regular_entries_and_sha(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch')as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((ROOT/'scratch').resolve()))
            path=Path(folder)/'fixture.tar.gz'
            with tarfile.open(path,'w:gz')as t:
                raw=b'{"count":1}';m=tarfile.TarInfo(audit.STAGE+'/fixture.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
            members,jsons,blobs,invalid,transport=audit.read_archive(path,audit.sha(path.read_bytes()),audit.STAGE,audit.OUTPUT)
            self.assertEqual(jsons['fixture.json'],{'count':1});self.assertEqual(transport['payload_files'],1)
            with self.assertRaises(ValueError):audit.read_archive(path,'0'*64,audit.STAGE,audit.OUTPUT)
            for entry in(audit.STAGE+'/../bad.json','other/fixture.json',audit.STAGE+'/bad\\name'):
                other=Path(folder)/'bad.tar.gz'
                with tarfile.open(other,'w:gz')as t:
                    m=tarfile.TarInfo(entry);m.size=1;t.addfile(m,io.BytesIO(b'x'))
                with self.assertRaises(ValueError):audit.read_archive(other,audit.sha(other.read_bytes()),audit.STAGE,audit.OUTPUT)

    def test_archive_rejects_case_collision_and_nonzero_suffix(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'scratch')as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to((ROOT/'scratch').resolve()))
            path=Path(folder)/'bad.tar.gz'
            with tarfile.open(path,'w:gz')as t:
                for n in('x.json','X.json'):
                    m=tarfile.TarInfo(audit.STAGE+'/'+n);m.size=2;t.addfile(m,io.BytesIO(b'{}'))
            with self.assertRaises(ValueError):audit.read_archive(path,audit.sha(path.read_bytes()),audit.STAGE,audit.OUTPUT)
            raw=io.BytesIO()
            with tarfile.open(fileobj=raw,mode='w')as t:
                m=tarfile.TarInfo(audit.STAGE+'/x.json');m.size=2;t.addfile(m,io.BytesIO(b'{}'))
            path.write_bytes(gzip.compress(raw.getvalue()+b'nonzero'))
            with self.assertRaises(ValueError):audit.read_archive(path,audit.sha(path.read_bytes()),audit.STAGE,audit.OUTPUT)

    def test_weights_only_cpu_decode_kwargs_with_fake_module(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w')as z:z.writestr('archive/data.pkl',b'fake')
        class FakeTensor:pass
        class FakeTorch:
            Tensor=FakeTensor
            def load(self,stream,**kw):
                self.kw=kw;return {'key':[1,None,True]}
        fake=FakeTorch();value=audit.torch_cpu_decode(b.getvalue(),fake)
        self.assertEqual(value,{'key':[1,None,True]})
        self.assertEqual(fake.kw,{'map_location':'cpu','weights_only':True})


if __name__=='__main__':unittest.main()
