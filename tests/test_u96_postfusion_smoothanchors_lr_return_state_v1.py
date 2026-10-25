"""Meaningful saved-state parser/recurrence fixtures; no model or optimizer."""
import ast
import copy
import hashlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import uuid
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/audit_u96_postfusion_smoothanchors_lr_state_v1.py'
spec=importlib.util.spec_from_file_location('smooth_state_auditor_under_test',SCRIPT)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def valid_npz():
    return {name:np.zeros(shape,np.float32)for name,shape in audit.shape_schema().items()}


def pack(values):
    stream=io.BytesIO();np.savez(stream,**values);return stream.getvalue()


def status(count=0,attempts=None,unknown=False):
    return {'counters':{'optimizer_updates':count,
        'optimizer_update_attempts':count if attempts is None else attempts},
        'optimizer_outcome_unknown':unknown}


class SavedStateParserTests(unittest.TestCase):
    def test_LR_specific_sealed_protocol_and_schedule(self):
        self.assertEqual(audit.PROTOCOL_SHA,'8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9')
        schedule=[1e-4*min(1.,u/4)for u in range(1,65)]
        self.assertEqual(schedule[:4],[.000025,.00005,.0001*.75,.0001])
        self.assertEqual(set(schedule[3:]),{.0001})
        source=SCRIPT.read_text()
        self.assertIn("maximum_lr=5e-5 if mode=='B1_FIXTURE'else 1e-4",source)
        self.assertIn("complete-U96-postfusion-smoothanchors-lr-resume-v1",source)
        self.assertIn("cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm",source)
        self.assertNotEqual(audit.PROTOCOL_SHA,audit.B1_PROTOCOL_SHA)

    def test_inert_import_and_help_block_numerical_modules(self):
        code=r'''
import ast,importlib.util,sys
from pathlib import Path
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in {'numpy','torch','PIL','psutil','resource'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
p=Path(sys.argv[1]);ast.parse(p.read_text(),feature_version=(3,10))
s=importlib.util.spec_from_file_location('inert_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert 'torch'not in sys.modules and'numpy'not in sys.modules
sys.argv=[str(p),'--help']
try:m.main()
except SystemExit as e:assert e.code==0
else:raise AssertionError('help must exit before numerical dependencies')
'''
        r=subprocess.run([sys.executable,'-B','-c',code,str(SCRIPT)],capture_output=True,text=True,timeout=15)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_strict_json_duplicate_and_nonfinite_rejected(self):
        for raw in(b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e999}',b'{"x":[-1e999]}'):
            with self.subTest(raw=raw),self.assertRaises(ValueError):audit.strict_json(raw)
        self.assertEqual(audit.strict_json(b'{"x":[0,1.0]}'),{'x':[0,1.]})

    def test_valid_dense_heads_roundtrip_canonical(self):
        values=valid_npz();values['heads.32.encoder.bias'][0]=.123
        decoded=audit.read_npz(pack(values),np)
        self.assertEqual(len(decoded),36)
        self.assertEqual(sum(x.size for x in decoded.values()),168960)
        self.assertEqual(audit.canonical(decoded),audit.canonical(values))

    def test_wrong_key_wrong_shape_and_nonfinite_arrays(self):
        for kind in('key','shape','nan','infinity','dtype'):
            values=valid_npz()
            if kind=='key':values['untrusted']=values.pop('heads.32.encoder.bias')
            if kind=='shape':values['heads.32.encoder.bias']=np.zeros((31,),np.float32)
            if kind=='nan':values['heads.32.encoder.bias'][0]=np.nan
            if kind=='infinity':values['heads.32.encoder.bias'][0]=np.inf
            if kind=='dtype':values['heads.32.encoder.bias']=np.zeros((32,),np.float64)
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.read_npz(pack(values),np)

    def test_truncated_duplicate_and_path_zip_rejected(self):
        with self.assertRaises(zipfile.BadZipFile):audit.read_npz(pack(valid_npz())[:-18],np)
        for names in(('same.npy','same.npy'),('../escape.npy',),('a\\b.npy',)):
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w')as z:
                for name in names:z.writestr(name,b'payload')
            with self.subTest(names=names),self.assertRaises(ValueError):audit.zip_guard(stream.getvalue(),'npz')

    def test_npy_declared_allocation_bomb_rejected_before_load(self):
        original=pack(valid_npz());stream=io.BytesIO();header=io.BytesIO()
        np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(1<<34,)})
        with zipfile.ZipFile(io.BytesIO(original))as source,zipfile.ZipFile(stream,'w')as target:
            for member in source.infolist():
                target.writestr(member.filename,header.getvalue()if member.filename=='heads.32.encoder.bias.npy'else source.read(member))
        with self.assertRaisesRegex(ValueError,'declared schema'):audit.read_npz(stream.getvalue(),np)

    def test_partial_status_never_becomes_terminal_success(self):
        self.assertEqual(audit.outcome(status(4),False,{'status':'QUALITY_STOP'},None,None)['scientific_status'],'QUALITY_STOP')
        self.assertTrue(audit.outcome(status(4,5),False,None,None,None)['optimizer_outcome_unknown'])
        self.assertTrue(audit.outcome(status(4),True,None,None,None)['optimizer_outcome_unknown'])
        self.assertIsNone(audit.outcome(None,True,None,None,None)['confirmed_optimizer_updates'])
        self.assertEqual(audit.outcome(status(0),False,{'status':'QUALITY_STOP'},None,None)['confirmed_optimizer_updates'],0)
        for count,pending in((4,False),(64,True)):
            with self.assertRaises(ValueError):
                audit.outcome(status(count),pending,None,{'complete':True,'model_qualified':False},None)
        terminal=audit.outcome(status(64),False,None,{'complete':True,'model_qualified':False},None)
        self.assertEqual(terminal['scientific_status'],'FINITE_SCREEN_PASSED_NOT_QUALIFIED')
        self.assertFalse(terminal['model_qualified'])
        with self.assertRaises(ValueError):audit.outcome(status(True),False,None,None,None)

    def test_sampler_photo_group_coverage_and_wrong_slot_rejection(self):
        order=['train_'+str(i)for i in range(16)]
        rows=[{'id':ref+'_'+role,'reference_id':ref,'role':role,'cohort':'optimize',
            'optimization_permitted':True,'split':'TRAIN'}for ref in order for role in audit.ROLES]
        rows +=[{'id':'monitor_'+str(i),'reference_id':'mon_'+str(i//4),'role':audit.ROLES[i%4],
            'cohort':'monitor','optimization_permitted':False,'split':'TRAIN'}for i in range(32)]
        p={'cases':rows,'learning':{'optimization_reference_order':order}}
        sampler={'exposures':256,'updates':[{'update':u,'slots':[
            order[(u-1+offset)%16]+'_'+role for role,offset in zip(audit.ROLES,[0,4,8,12])]}for u in range(1,65)]}
        _,coverage=audit.check_sampler(p,sampler)
        self.assertEqual(set(coverage.values()),{4})
        broken=copy.deepcopy(sampler);broken['updates'][0]['slots'][0]='monitor_0'
        with self.assertRaises(ValueError):audit.check_sampler(p,broken)

    def test_numPy_Adam_actual_scalar_reference_recurrence(self):
        # Independently calculate scalar Adam moments and parameter trajectory
        # for sign changes/zeros; compare the actual helper, no Torch optimizer.
        h=np.asarray([.2,-.1,0],np.float32);m=np.zeros_like(h);v=np.zeros_like(h)
        scalar_h=[float(x)for x in h];scalar_m=[0.]*3;scalar_v=[0.]*3
        for u,g in enumerate(([.2,-.1,0],[-.3,.05,.2],[0,.01,-.2],[.4,-.2,.1]),1):
            g=np.asarray(g,np.float32);lr=5e-5*min(1.,u/4)
            newm=np.asarray(.9*m+.1*g,np.float32);newv=np.asarray(.99*v+.01*g*g,np.float32)
            pm,pv,ph=audit.adam_formula(h,g,m,v,newm,newv,u,lr,np)
            for i,x in enumerate(g):
                scalar_m[i]=.9*scalar_m[i]+.1*float(x)
                scalar_v[i]=.99*scalar_v[i]+.01*float(x)**2
                scalar_h[i]-=lr*(scalar_m[i]/(1-.9**u))/((scalar_v[i]/(1-.99**u))**.5+1e-8)
            np.testing.assert_allclose(pm,newm,atol=2e-9,rtol=2e-6)
            np.testing.assert_allclose(pv,newv,atol=2e-10,rtol=2e-6)
            np.testing.assert_allclose(ph,scalar_h,atol=2e-8,rtol=2e-6)
            h,m,v=ph,newm,newv
        damaged=h.copy();damaged[0]+=.01
        self.assertGreater(audit.comparison(h,damaged,np,'parameter_update')[1],0)

    def test_archive_footer_corruption_and_duplicate_member_rejection(self):
        def tiny_tar(duplicate=False):
            stream=io.BytesIO()
            with tarfile.open(fileobj=stream,mode='w:gz')as tar:
                for _ in range(2 if duplicate else 1):
                    b=b'{}';info=tarfile.TarInfo('fixture/protocol.json');info.size=len(b);tar.addfile(info,io.BytesIO(b))
            return stream.getvalue()
        fixture=ROOT/'outputs/codeformer_u96_postfusion_smoothanchors_lr_state_audit_preparation'/('fixture_'+uuid.uuid4().hex)
        fixture.mkdir(parents=True,exist_ok=False)
        raw=tiny_tar();p=fixture/'valid.tar.gz';p.write_bytes(raw)
        members,*_=audit.read_archive(p,hashlib.sha256(raw).hexdigest(),'fixture','outputs/o')
        self.assertIn('protocol.json',members)
        p=fixture/'truncated.tar.gz';p.write_bytes(raw[:-4])
        with self.assertRaises((EOFError,OSError,ValueError)):
            audit.read_archive(p,hashlib.sha256(raw[:-4]).hexdigest(),'fixture','outputs/o')
        raw=tiny_tar(True);p=fixture/'duplicate.tar.gz';p.write_bytes(raw)
        with self.assertRaises(ValueError):audit.read_archive(p,hashlib.sha256(raw).hexdigest(),'fixture','outputs/o')

    def test_canonical_names_sparse_and_nonzero_tar_tail_refused(self):
        fixture=ROOT/'outputs/codeformer_u96_postfusion_smoothanchors_lr_state_audit_preparation'/('fixture_'+uuid.uuid4().hex)
        fixture.mkdir(parents=True,exist_ok=False)
        for name in('fixture/a//b','fixture/a/./b'):
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w:gz')as t:
                info=tarfile.TarInfo(name);info.size=2;t.addfile(info,io.BytesIO(b'{}'))
            raw=buffer.getvalue();p=fixture/(sha_name(name)+'.tar.gz');p.write_bytes(raw)
            with self.subTest(name=name),self.assertRaises(ValueError):
                audit.read_archive(p,hashlib.sha256(raw).hexdigest(),'fixture','outputs/o')
        tarbytes=io.BytesIO()
        with tarfile.open(fileobj=tarbytes,mode='w:')as t:
            info=tarfile.TarInfo('fixture/protocol.json');info.size=2;t.addfile(info,io.BytesIO(b'{}'))
        raw=gzip.compress(tarbytes.getvalue()+b'nonzero',mtime=0);p=fixture/'tail.tar.gz';p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'trailing|after final'):
            audit.read_archive(p,hashlib.sha256(raw).hexdigest(),'fixture','outputs/o')
        reader=audit.ExpandedReader(io.BytesIO(b'x'*128));old=audit.MAX_EXPANDED
        try:
            audit.MAX_EXPANDED=64
            with self.assertRaisesRegex(ValueError,'expanded gzip cap'):reader.read(128)
        finally:audit.MAX_EXPANDED=old


def sha_name(value):return hashlib.sha256(value.encode()).hexdigest()[:12]


if __name__=='__main__':unittest.main()
