"""Loss and direction arithmetic only; no neural, autograd or optimizer calls."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from cctv_dgp_bank_loss_alignment_v1 import components,hypothetical_adam_direction,summarize

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
PARENT=OUT/'cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS=OUT/'cctv_dgp_bank_path_ablation_v1_analysis'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    receipt=OUT/'cctv_dgp_bank_loss_alignment_v1_math_audit.json'; assert not receipt.exists()
    old=read(PARENT/'protocol.json'); p=read(OUT/'cctv_dgp_bank_path_ablation_v1_return/protocol.json')
    assert read(OUT/'cctv_dgp_bank_partition_v1_analysis/visual_review.json')['complete']
    initial_path=PARENT/'initializers/initial_B.pth'; assert sha(initial_path)==old['assets_sha256']['initializers/initial_B.pth']
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    source=PARENT/'cctv_dgp_bank_comparison_v1_model.py'; assert sha(source)==old['assets_sha256'][source.name]
    tree=ast.parse(source.read_text(encoding='utf-8'))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CURRENT_NAMES' for t in n.targets))
    current=eval(compile(ast.Expression(assignment.value),str(source),'eval'),{'__builtins__':{}},{})
    names=['current.'+n for n in current]+[n for n in initial if n.startswith(('style_conditioner.','spatial_conditioners.','fusions.'))]
    assert len(names)==37
    parameter_bytes=sum(initial[n].numel()*initial[n].element_size() for n in names)
    bound=parameter_bytes*10*4+16*1024**2; assert bound<1024**3
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_bank_comparison_v1_contract import NORMALIZERS,ANCHORS
    base={r['id']:r for r in read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')['rows']}
    rows=read(OUT/'cctv_dgp_bank_path_ablation_v1_return/outputs/path_ablation_v1/enabled/metrics.json')['rows']
    errors=[]
    for begin in range(0,100,5):
        block=rows[begin:begin+5]; originals=[base[r['id']] for r in block]
        def arrays(rs):
            return {key:torch.tensor([r['raw'][metric] for r in rs],dtype=torch.float64) for key,metric in
                    [('MSE','MSE'),('SSIM_loss','SSIM'),('landmark_structure','landmark_high_frequency_MSE')]}
        terms,previous=arrays(block),arrays(originals)
        terms['SSIM_loss']=1-terms['SSIM_loss']; previous['SSIM_loss']=1-previous['SSIM_loss']
        arc=torch.tensor([1-r['raw']['ArcFace_observed_fixed'] for r in block],dtype=torch.float64)
        old_arc=torch.tensor([1-r['raw']['ArcFace_observed_fixed'] for r in originals],dtype=torch.float64)
        mean_mse=torch.tensor([r['raw']['constant_mean_shift_only_MSE'] for r in block],dtype=torch.float64)
        photo=10*torch.relu(torch.relu(previous['MSE']-mean_mse)-.2*torch.relu(previous['MSE']-terms['MSE']))[1:].mean()/NORMALIZERS[0]
        result=components(terms,previous,arc,old_arc,photo,NORMALIZERS,ANCHORS,old['reconstruction_weights'])
        # Independent NumPy reconstruction of the retained scalar objective.
        a,b={k:v.numpy() for k,v in terms.items()},{k:v.numpy() for k,v in previous.items()}
        expected=[old['reconstruction_weights'][i]*value/NORMALIZERS[i] for i,value in enumerate(
            [a['MSE'].mean(),a['SSIM_loss'].mean(),arc.numpy().mean(),a['landmark_structure'][1:].mean()])]
        expected += [a['MSE'][0]/ANCHORS[0]+a['MSE'][1]/ANCHORS[1],
            5*(np.maximum(a['MSE']-b['MSE']-1e-12,0).mean()/NORMALIZERS[0]
               +np.maximum(a['SSIM_loss']-b['SSIM_loss']-1e-6,0).mean()/NORMALIZERS[1]
               +np.maximum(arc.numpy()-old_arc.numpy()-1e-6,0).mean()/NORMALIZERS[2]),float(photo),
            a['MSE'][0]/ANCHORS[0],a['MSE'][1:].mean()/NORMALIZERS[0]]
        error=float(np.max(np.abs(np.array([float(v) for v in result])-expected))); assert error<1e-12
        errors.append(error)
    # Fixed synthetic arrays test Adam arithmetic; no trainable model exists.
    parameters=[torch.tensor([.03*(i+1),-.01*(i+1)],dtype=torch.float32) for i in range(37)]
    grad_rows=[[torch.tensor([.01*(j+1)*(i+1),-.007*(j+2)*(i+1)],dtype=torch.float32) for i in range(37)] for j in range(9)]
    stored=dict(param_groups=[dict(params=list(range(37)),betas=(.9,.999),eps=1e-8,weight_decay=1e-5)],
        state={i:dict(step=torch.tensor(50.),exp_avg=torch.tensor([.01,-.02]),exp_avg_sq=torch.tensor([.03,.04])) for i in range(37)})
    direction_tests=[]
    for state in [None,stored]:
        directions,norm,factor=hypothetical_adam_direction(parameters,grad_rows,state)
        total=[sum(row[i].double().numpy() for row in grad_rows[:7]) for i in range(37)]
        expected_norm=float(np.sqrt(sum(np.square(a).sum() for a in total))); expected_factor=min(1,1/(expected_norm+1e-6))
        assert abs(norm-expected_norm)<1e-5 and abs(factor-expected_factor)<1e-7
        max_error=0.
        for i,(parameter,g,d) in enumerate(zip(parameters,total,directions)):
            effective=g*expected_factor+parameter.double().numpy()*1e-5
            m,v,t=(np.zeros(2),np.zeros(2),1) if state is None else (np.array([.01,-.02]),np.array([.03,.04]),51)
            expected=-(.9*m+.1*effective)/(1-.9**t)/(np.sqrt((.999*v+.001*effective**2)/(1-.999**t))+1e-8)
            assert np.allclose(d.numpy(),expected,rtol=2e-5,atol=2e-5)
            max_error=max(max_error,float(np.abs(d.numpy()-expected).max()))
        summaries=summarize(grad_rows,directions,names)
        for label,summary in summaries.items():
            indices=[names.index(n) for n in summary['parameter_names']]
            matrix=np.array([[sum(float(np.dot(grad_rows[a][i].double().numpy(),grad_rows[b][i].double().numpy())) for i in indices) for b in range(9)] for a in range(9)])
            assert np.allclose(summary['Gram'],matrix,rtol=1e-12,atol=1e-12)
            dots=[sum(float(np.dot(grad_rows[j][i].double().numpy(),directions[i].double().numpy())) for i in indices) for j in range(9)]
            assert np.allclose(summary['Adam_direction_dot_components'],dots,rtol=1e-12,atol=1e-12)
        direction_tests.append(dict(state='initial' if state is None else 'saved_step50',maximum_array_error=max_error,partitions_checked=len(summaries)))
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']; assert all(sha(ROOT/n)==h for n,h in protected.items())
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        helper_sha256=sha(ROOT/'scripts/cctv_dgp_bank_loss_alignment_v1.py'),
        cached_TRAIN_metric_rows_checked=100,cached_scalar_arithmetic_not_fresh_neural_or_gradient_replay=True,
        maximum_scalar_component_error=max(errors),synthetic_array_direction_tests=direction_tests,
        learning_names=names,parameter_elements=sum(initial[n].numel() for n in names),parameter_bytes=parameter_bytes,
        uncompressed_sampled_gradient_upper_bound_bytes=bound,sample_files=4,component_gradient_arrays_per_file=9,
        no_neural_calls=True,local_gradient_queries=0,local_backwards=0,local_optimizer_updates=0,
        actual_L4_gradients_timing_and_direction_results_unrun=True)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,paired_metric_rows=100,parameter_elements=record['parameter_elements'],sample_gradient_bound_bytes=bound,local_gradient_queries=0))


if __name__=='__main__': main()
