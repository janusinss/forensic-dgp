"""Pure contracts and mocked lifecycle tests; no Torch/model/gradient calls."""
from __future__ import annotations
import ast
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import signal
import os
import random
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/"scripts/cctv_completion_codeformer_finetune_v1_vm.py"
spec=importlib.util.spec_from_file_location("completion_finetune_contract_target",SOURCE)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)

class Tensor(np.ndarray):
    def clamp_min(self, value): return np.maximum(self,value).view(Tensor)
    def clamp(self, low, high): return np.clip(self,low,high).view(Tensor)

def tensor(value):return np.asarray(value,dtype=np.float32).view(Tensor)

class PureF:
    """NumPy expression harness only; no claim of CUDA-kernel parity."""
    @staticmethod
    def interpolate(value,size,mode,align_corners=None):
        a=np.asarray(value);h,w=a.shape[-2:];oh,ow=size
        if mode=="nearest":
            ys=(np.arange(oh)*h//oh).clip(0,h-1);xs=(np.arange(ow)*w//ow).clip(0,w-1)
            return a[:,:,ys[:,None],xs[None,:]].view(Tensor)
        if mode!="bilinear" or align_corners is not False:raise ValueError("Unexpected interpolation contract")
        y=(np.arange(oh,dtype=np.float64)+.5)*h/oh-.5
        x=(np.arange(ow,dtype=np.float64)+.5)*w/ow-.5
        yl=np.floor(y).astype(int);xl=np.floor(x).astype(int)
        fy=(y-yl).astype(np.float32);fx=(x-xl).astype(np.float32)
        yh=(yl+1).clip(0,h-1);xh=(xl+1).clip(0,w-1);yl=yl.clip(0,h-1);xl=xl.clip(0,w-1)
        top=a[:,:,yl[:,None],xl[None,:]]*(1-fx)[None,None,None,:]+a[:,:,yl[:,None],xh[None,:]]*fx[None,None,None,:]
        bottom=a[:,:,yh[:,None],xl[None,:]]*(1-fx)[None,None,None,:]+a[:,:,yh[:,None],xh[None,:]]*fx[None,None,None,:]
        return (top*(1-fy)[None,None,:,None]+bottom*fy[None,None,:,None]).view(Tensor)
    @staticmethod
    def max_pool2d(value,kernel_size,stride,padding):
        if (kernel_size,stride,padding)!=(7,1,3):raise ValueError("Unexpected ring contract")
        a=np.asarray(value);h,w=a.shape[-2:]
        padded=np.pad(a,((0,0),(0,0),(3,3),(3,3)),constant_values=-np.inf)
        out=np.full_like(a,-np.inf)
        for dy in range(7):
            for dx in range(7):out=np.maximum(out,padded[:,:,dy:dy+h,dx:dx+w])
        return out.view(Tensor)

def lifted(*names):
    tree=ast.parse(SOURCE.read_text(encoding="utf-8"),feature_version=(3,10))
    selected=[]
    for name in names:
        found=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name]
        if len(found)!=1:raise ValueError("Ambiguous lifted function "+name)
        selected.append(copy.deepcopy(found[0]))
    namespace={"F":PureF,"KINDS":pilot.KINDS}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(SOURCE),"exec"),namespace)
    return namespace

def metric_rows(value=1.):
    return [{"id":str(i)+"_"+kind,"mask_kind":kind,
             **{key:value for key in ("raw_hole_L1","raw_boundary_L1","PNG_hole_L1","PNG_boundary_L1")}}
            for i in range(8) for kind in pilot.KINDS]

class ScheduleAndMetrics(unittest.TestCase):
    def test_balanced_schedule_uses_each_fit_pair_once(self):
        cases=[{"reference_id":"fit_%02d"%i,"mask_kind":kind,"id":"fit_%02d__%s"%(i,kind)}
               for i in range(32) for kind in pilot.KINDS]
        plan=pilot.training_schedule(cases);lookup={r["id"]:r for r in cases}
        self.assertEqual(len(plan),32)
        self.assertEqual(sorted(x for cycle in plan for x in cycle),sorted(lookup))
        for cycle in plan:
            self.assertEqual(len({lookup[x]["reference_id"] for x in cycle}),4)
            self.assertEqual([lookup[x]["mask_kind"] for x in cycle],list(pilot.KINDS))
        self.assertEqual(plan,pilot.training_schedule(list(reversed(cases))))
        self.assertEqual(40*2+4*2+32*4*2+4*32+2*28,528)
        self.assertEqual(32*2,64);self.assertEqual(32*4*2,256)

    def test_metrics_accept_improvement_and_refuse_hidden_geometry_regression(self):
        f=lifted("assess","compare");assess=f["assess"];compare=f["compare"]
        baseline={"metrics":assess(metric_rows())}
        improved=metric_rows(.98)
        self.assertTrue(compare({"metrics":assess(improved)},baseline)["prospective_numeric_screen_pass"])
        hidden=metric_rows(.90)
        for row in hidden:
            if row["mask_kind"]=="eye_band":row["PNG_boundary_L1"]=1.0001
        report=compare({"metrics":assess(hidden)},baseline)
        self.assertGreater(report["PNG_macro_hole_gain"],.09)
        self.assertFalse(report["prospective_numeric_screen_pass"])
        self.assertFalse(report["all_raw_PNG_hole_boundary_macro_geometry_nonworse"])
        self.assertFalse(compare({"metrics":assess(metric_rows(.995))},baseline)["prospective_numeric_screen_pass"])

    def test_catastrophic_single_geometry_early_stop_is_not_hidden_by_macro(self):
        f=lifted("assess","compare")
        baseline={"metrics":f["assess"](metric_rows())};rows=metric_rows(.8)
        for row in rows:
            if row["mask_kind"]=="cheek_object":row["raw_hole_L1"]=1.1001
        report=f["compare"]({"metrics":f["assess"](rows)},baseline)
        self.assertTrue(report["catastrophic_regression_stop"])
        self.assertFalse(report["prospective_numeric_screen_pass"])
        self.assertFalse(report["model_qualified"]);self.assertTrue(report["visual_review_pending"])

class GeometryContracts(unittest.TestCase):
    def test_hidden_cover_pixels_cannot_contaminate_app_input(self):
        f=lifted("app_input")
        rgb=tensor(np.full((1,3,256,256),.2))
        mask=tensor(np.zeros((1,1,256,256)));mask[:,:,100:180,90:175]=1
        altered=rgb.copy().view(Tensor);altered[:,:,100:180,90:175]=.99
        left,m=f["app_input"](rgb,mask);right,other_m=f["app_input"](altered,mask)
        self.assertTrue(np.array_equal(left,right))
        self.assertTrue(np.array_equal(m,np.repeat(np.repeat(mask,2,-2),2,-1)))
        self.assertTrue(np.array_equal(m,other_m))
        support=np.repeat(m.astype(bool),3,axis=1)
        self.assertTrue(np.all(left[support]==1))
        self.assertLess(float(np.max(np.abs(left[~support]+.6))),2e-6)

    def test_delivered_composition_keeps_every_visible_float_exact(self):
        f=lifted("delivered")
        rng=np.random.default_rng(7);rgb=tensor(rng.random((1,3,256,256),dtype=np.float32))
        mask=tensor(np.zeros((1,1,256,256)));mask[:,:,70:190,80:180]=1
        raw=tensor(np.full((1,3,512,512),4.))
        output=f["delivered"](rgb,mask,raw)
        outside=np.repeat(mask==0,3,axis=1);inside=~outside
        self.assertTrue(np.array_equal(output[outside],rgb[outside]))
        self.assertTrue(np.all(output[inside]==1))
        bypass=f["delivered"](rgb,tensor(np.zeros_like(mask)),raw)
        self.assertTrue(np.array_equal(bypass,rgb))

    def test_inner_ring_geometry_and_pool_padding(self):
        f=lifted("boundary")
        mask=tensor(np.zeros((1,1,32,32)));mask[:,:,10:20,9:21]=1
        ring=f["boundary"](mask)
        self.assertEqual(int(ring.sum()),10*12-4*6)
        edge=tensor(np.zeros((1,1,32,32)));edge[:,:,:10,:12]=1
        # max-pool inverse padding=-inf does not erode along outer image edges.
        self.assertEqual(int(f["boundary"](edge).sum()),10*12-7*9)

class FakeChild:
    def __init__(self,wait_effects,pid=4242):
        self.pid=pid;self.returncode=None;self.effects=list(wait_effects);self.waits=[]
    def wait(self,timeout=None):
        self.waits.append(timeout)
        value=self.effects.pop(0) if self.effects else 0
        if isinstance(value,BaseException):raise value
        if callable(value):return value(self)
        self.returncode=value;return value
    def poll(self):return self.returncode

class SupervisorContracts(unittest.TestCase):
    def invoke(self,child=None,popen_error=None,pgid=4242):
        with tempfile.TemporaryDirectory(prefix="finetune_contract_") as temporary:
            root=Path(temporary)
            def start(*args,**kwargs):
                self.assertTrue(kwargs["start_new_session"])
                if popen_error:raise popen_error
                return child
            kills=[];handlers={};old_handler=object()
            def change_signal(signum,handler):
                prior=handlers.get(signum,old_handler);handlers[signum]=handler;return prior
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(pilot,"verify",return_value=({},root/"base.pth")))
                stack.enter_context(mock.patch.object(pilot,"require_vm",return_value={"GPU":"fixture"}))
                stack.enter_context(mock.patch.object(pilot.subprocess,"Popen",side_effect=start))
                stack.enter_context(mock.patch.object(pilot.os,"getpgid",return_value=pgid,create=True))
                stack.enter_context(mock.patch.object(pilot.os,"killpg",side_effect=lambda pid,sig:kills.append((pid,sig)),create=True))
                stack.enter_context(mock.patch.object(pilot.signal,"SIGKILL",9,create=True))
                stack.enter_context(mock.patch.object(pilot.signal,"signal",side_effect=change_signal))
                stack.enter_context(mock.patch.object(sys,"argv",["pilot","--root",str(root),"--protocol-sha","fixture","--run"]))
                if child is not None and getattr(child,"signal_on_wait",False):
                    child.effects[0]=lambda owned:handlers[signal.SIGTERM](signal.SIGTERM,None)
                with redirect_stdout(io.StringIO()):code=pilot.main()
            receipt=json.loads((root/"supervisor.json").read_text())
            self.assertIs(handlers[signal.SIGTERM],old_handler)
            return code,receipt,kills
    def test_success_and_timeout_always_write_actual_child_status(self):
        code,r,kills=self.invoke(FakeChild([0]))
        self.assertEqual((code,r["worker_exit"],r["actual_child_exit"]),(0,0,0));self.assertEqual(kills,[])
        child=FakeChild([subprocess.TimeoutExpired("fixture",1800),-9])
        code,r,kills=self.invoke(child)
        self.assertEqual(code,124);self.assertTrue(r["timeout"])
        self.assertEqual(r["actual_child_exit"],-9);self.assertEqual(kills,[(4242,9)])
        self.assertFalse(r["terminal_state_present"])

    def test_interrupt_soft_stop_has_bounded_grace_and_terminal_receipt(self):
        child=FakeChild([KeyboardInterrupt("fixture"),143])
        code,r,kills=self.invoke(child)
        self.assertEqual(code,130);self.assertEqual(r["actual_child_exit"],143)
        self.assertEqual(kills,[(4242,signal.SIGTERM)])
        self.assertTrue(0<=child.waits[1]<=15)
        self.assertIn("KeyboardInterrupt",r["interruption"])
        self.assertIn("no automatic resume",r["partial_state_recoverability"])

    def test_SIGTERM_handler_takes_same_bounded_terminal_path(self):
        child=FakeChild([0,143]);child.signal_on_wait=True
        code,r,kills=self.invoke(child)
        self.assertEqual(code,130);self.assertEqual(r["actual_child_exit"],143)
        self.assertIn("Supervisor termination signal",r["interruption"])
        self.assertEqual(kills,[(4242,signal.SIGTERM)])
        self.assertTrue(0<=child.waits[1]<=15)

    def test_stuck_interrupt_falls_back_to_owned_group_kill(self):
        child=FakeChild([KeyboardInterrupt("fixture"),subprocess.TimeoutExpired("fixture",15),-9])
        code,r,kills=self.invoke(child)
        self.assertEqual(code,130);self.assertEqual(r["actual_child_exit"],-9)
        self.assertEqual(kills,[(4242,signal.SIGTERM),(4242,9)])
        self.assertEqual(r["shutdown_mode"],"interruption_grace_exhausted_SIGKILL")

    def test_spawn_failure_still_records_no_child_without_killing(self):
        code,r,kills=self.invoke(popen_error=OSError("fixture spawn failure"))
        self.assertEqual(code,1);self.assertIsNone(r["actual_child_exit"]);self.assertIsNone(r["child_pid"])
        self.assertIn("fixture spawn failure",r["interruption"]);self.assertEqual(kills,[])


class AtomicTerminalState(unittest.TestCase):
    def test_atomic_state_publication_and_failed_write_retention(self):
        class Weight:
            grad=None
            def detach(self):return self
            def cpu(self):return self
        class Net:
            def state_dict(self):return {"encoder.weight":Weight(),"quantize.embedding.weight":Weight(),"generator.weight":Weight()}
            def named_parameters(self):return [("encoder.weight",Weight())]
        for mode in ("success","serialize_failure","guard_failure"):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(prefix="finetune_atomic_") as temporary:
                out=Path(temporary);namespace=lifted("save_state");observed=[]
                def serialize(snapshot,stream):
                    self.assertEqual(Path(stream.name).name,"terminal_training_state.pth.partial")
                    self.assertFalse((out/"terminal_training_state.pth").exists())
                    self.assertEqual(set(snapshot["trainable_state"]),{"encoder.weight"})
                    self.assertFalse(snapshot["automatic_resume_permitted"])
                    observed.append(snapshot);stream.write(b"fixture checkpoint bytes")
                    if mode=="serialize_failure":raise OSError("fixture serialization failed")
                def guard(**kwargs):
                    self.assertEqual(kwargs,{"terminal_save":True})
                    if mode=="guard_failure":raise ValueError("fixture terminal budget failed")
                namespace.update({"need":pilot.need,"digest_state":lambda prefixes=None:"frozen" if prefixes else "logical",
                    "frozen_state":"frozen","net":Net(),"WEIGHT_SHA":pilot.WEIGHT_SHA,"expected":"fixture protocol",
                    "progress":{"phase":"fixture"},"optimizer_parameter_names":{"code":["encoder.weight"],"fusion":[]},
                    "terminal_cycle":2,"code_optimizer":SimpleNamespace(state_dict=lambda:{}),
                    "fusion_optimizer":SimpleNamespace(state_dict=lambda:{}),"schedule":[["fixture"]],
                    "counts":{"code_optimizer_steps":2},"CONTRACT":pilot.CONTRACT,"out":out,
                    "torch":SimpleNamespace(save=serialize,get_rng_state=lambda:b"CPU",
                        cuda=SimpleNamespace(get_rng_state_all=lambda:[b"CUDA"])),"random":random,"np":np,
                    "guard":guard,"os":os,"sha":pilot.sha})
                if mode=="success":
                    receipt=namespace["save_state"](2,"fixture")
                    self.assertEqual((out/receipt["path"]).read_bytes(),b"fixture checkpoint bytes")
                    self.assertFalse((out/"terminal_training_state.pth.partial").exists())
                else:
                    with self.assertRaises((OSError,ValueError)):namespace["save_state"](2,"fixture")
                    self.assertFalse((out/"terminal_training_state.pth").exists())
                    self.assertEqual((out/"terminal_training_state.pth.partial").read_bytes(),b"fixture checkpoint bytes")
                self.assertEqual(len(observed),1)

if __name__=="__main__":unittest.main(verbosity=2)

