"""Finite saved-step inference contract; no neural import or optimization."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import platform
import sys
import numpy as np

NAME = 'cctv_dgp_actual_step_review_v1_vm'
STEM = 'cctv-dgp-actual-step-review-v1'
RETURN_PREFIX = NAME + '_return'
FORMAT = 'own-DGP-actual-step-finite-inference-review-v1'
UPDATES = [10, 12, 22, 27, 37, 38, 39, 42, 44, 45]
PROPOSALS = ['zero', 'recorded', 'cone']
ROLES = ['not_yet_optimized', 'optimized', 'current_batch']
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
BUDGETS = {'cache_seconds': 300, 'review_seconds': 1500, 'worker_seconds': 1800,
           'export_seconds': 900, 'minimum_free_bytes': 6*1024**3,
           'maximum_allocated_VRAM_bytes': 20*1024**3, 'maximum_output_bytes': 3*1024**3,
           'maximum_return_members': 12000, 'maximum_member_bytes': 16*1024**2,
           'maximum_forward_slots': 3150, 'timing_safety_factor': 1.25}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def safe_path(root, name):
    assert isinstance(name, str) and not any(c in name for c in ['\\', ':', '\x00'])
    parts = PurePosixPath(name)
    assert not parts.is_absolute() and parts.parts and all(p not in ['.', '..'] for p in parts.parts)
    assert name == parts.as_posix()
    path = Path(root)/name
    assert path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink()
    return path


def vm_scope(root):
    assert sys.platform == 'linux', 'Manual finite inference requires the existing Linux VM; no neural imports occurred'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing forensic-dgp-thesis VM only'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve(), 'Distinct actual-step review root required'


def role_ids(p, probe, role):
    return probe['case_ids'] if role == 'current_batch' else p['cohorts'][role]


def output_prefix(update, proposal):
    assert update in UPDATES and proposal in PROPOSALS
    return 'outputs/probes/update'+str(update).zfill(4)+'/'+proposal


def allowed_return_names(p):
    names = {'protocol.json', 'review.log', 'review_exit_code.txt', 'supervisor_receipt.json', 'export_manifest.json',
             'outputs/results.json', 'outputs/failure.json', 'outputs/preflight.json', 'outputs/cache_receipt.json',
             'outputs/timing_projection.json'}
    for probe in p['probes']:
        for proposal in PROPOSALS:
            prefix = output_prefix(probe['update'], proposal)
            names.add(prefix+'/metrics.json')
            for role in ROLES:
                for cid in role_ids(p, probe, role):
                    for suffix in ['.npz', '.png', '_mean_only.png']: names.add(prefix+'/'+role+'/'+cid+suffix)
    return names


def safe_members(members, p):
    allowed = allowed_return_names(p); seen=set(); result=[]; total=0
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk()
        assert m.name.startswith(RETURN_PREFIX+'/')
        name=m.name[len(RETURN_PREFIX)+1:]
        safe_path(Path.cwd(), name)
        assert name in allowed and name.casefold() not in seen
        assert 0 <= m.size <= BUDGETS['maximum_member_bytes']
        total += m.size; seen.add(name.casefold()); result.append((m,name))
        assert total <= BUDGETS['maximum_output_bytes'] and len(seen) <= BUDGETS['maximum_return_members']
    return result,total


def check_proposal(step, proposal):
    assert step['components'].dtype == np.float32 and step['components'].shape == (7,17952)
    assert all(step[k].dtype == np.float32 and step[k].shape == (17952,) and np.isfinite(step[k]).all()
               for k in ['before','after','combined','applied','exp_avg','exp_avg_sq'])
    assert np.isfinite(step['components']).all()
    assert proposal['values'].dtype == np.float32 and proposal['values'].shape == (3,17952) and np.isfinite(proposal['values']).all()
    assert proposal['dual'].dtype == np.float64 and proposal['dual'].shape == (7,) and np.isfinite(proposal['dual']).all()
    assert (proposal['dual'] >= 0).all()
    assert np.array_equal(proposal['values'][0],step['before']) and np.array_equal(proposal['values'][1],step['after'])
    g=step['components'].astype(np.float64); norms=np.linalg.norm(g,axis=1)
    q=np.divide(g,norms[:,None],out=np.zeros_like(g),where=norms[:,None]>0)
    d=step['after'].astype(np.float64)-step['before'].astype(np.float64)
    intended=d-q.T@proposal['dual']; n=float(np.linalg.norm(d)); assert n>0
    assert proposal['intended_delta'].dtype==np.float64 and proposal['intended_delta'].shape==(17952,)
    assert np.allclose(intended,proposal['intended_delta'],rtol=0,atol=1e-14)
    dots=q@intended; assert dots.max() <= 1e-10*n
    assert np.max(np.abs(proposal['dual']*dots)) <= 1e-10*max(n*n,1e-20)
    target=step['before'].astype(np.float64)+proposal['intended_delta']
    rounding=np.abs(proposal['values'][2].astype(np.float64)-target)
    spacing=np.maximum(np.abs(np.spacing(proposal['values'][2])).astype(np.float64),np.finfo(np.float32).tiny)
    assert np.all(rounding <= .50001*spacing+1e-16)
    realized=proposal['values'][2].astype(np.float64)-step['before'].astype(np.float64)
    return {'intended_component_derivatives':(g@intended).tolist(), 'rounded_component_derivatives':(g@realized).tolist(),
            'intended_normalized_constraint_dots':dots.tolist(), 'rounded_normalized_constraint_dots':(q@realized).tolist(),
            'maximum_parameter_rounding_error':float(rounding.max()),'intended_step_norm':float(np.linalg.norm(intended)),
            'rounded_step_norm':float(np.linalg.norm(realized)), 'finite_losses_not_implied':True}


def verify(root,pin):
    root=Path(root).resolve(); assert sha(root/'protocol.json')==pin
    p=read(root/'protocol.json'); assert p['format']==FORMAT and p['budgets']==BUDGETS
    assert p['proposals']==PROPOSALS and [r['update'] for r in p['probes']]==UPDATES
    assert p['optimizer_updates']==0 and p['gradient_queries']==0 and p['new_trained_checkpoint']==False
    assert p['native_or_DEV_or_reserved_final_used']==False and p['app_promotion']==False
    assert len(p['cases'])==145 and len(p['references'])==29
    cases={c['id']:c for c in p['cases']}; assert len(cases)==145 and all(c['role']=='train' for c in cases.values())
    assert list(p['cohorts'])==ROLES[:2] and all(len(v)==len(set(v))==50 for v in p['cohorts'].values())
    assert not set(p['cohorts']['not_yet_optimized']).intersection(p['cohorts']['optimized'])
    for name,digest in p['assets_sha256'].items():
        path=safe_path(root,name); assert path.is_file() and sha(path)==digest,name
    slots=0
    for probe in p['probes']:
        assert len(probe['case_ids'])==5 and [cases[c]['profile'] for c in probe['case_ids']]==PROFILES
        assert len({cases[c]['reference_id'] for c in probe['case_ids']})==1
        with np.load(root/probe['step_arrays'],allow_pickle=False) as a, np.load(root/probe['proposal_arrays'],allow_pickle=False) as b:
            evidence=check_proposal(a,b)
        assert all(abs(evidence[k]-probe['arithmetic'][k])<1e-14 for k in ['maximum_parameter_rounding_error','intended_step_norm','rounded_step_norm'])
        for role in ROLES:
            ids=role_ids(p,probe,role); assert all(cid in cases for cid in ids)
            slots+=len(ids)*len(PROPOSALS)
    assert slots==3150
    return p
