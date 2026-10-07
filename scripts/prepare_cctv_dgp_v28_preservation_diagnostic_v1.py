"""Prepare one zero-update final-V28 gradient measurement for manual existing-L4 execution."""
import ast
import copy
import json
from pathlib import Path
import tarfile

from diagnose_cctv_dgp_active_original_decoder_v28_mean_control import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
RETURNED = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_return'
NAME = 'cctv_dgp_v28_preservation_diagnostic_v1_vm'
OUT = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_v28_preservation_diagnostic_v1_preparation'
STEM = 'cctv-dgp-v28-preservation-diagnostic-v1'
WORKER = 'scripts/cctv_dgp_v28_preservation_diagnostic_v1_vm.py'

V28_CHECK = '''def v28_check(parent,p):
    v28=parent.parent/'cctv_dgp_active_original_decoder_vm_v28'
    assert sha(v28/'protocol.json')==p['closed_V28_protocol_sha256']
    old=read(v28/'protocol.json')
    for name,digest in old['assets_sha256'].items():assert sha(v28/name)==digest,name
    for name,digest in p['closed_V28_evidence_sha256'].items():assert sha(v28/name)==digest,'Closed V28 evidence changed: '+name
    result=read(v28/'outputs/results.json')
    assert result['complete'] and result['optimizer_updates']==800 and result['epochs']==80
    assert result['necessary_capacity_pass'] is False and len(result['preservation_failures'])==3
    assert result['candidate_DGP_state']==p['final_V28_DGP_state'] and not result['app_promotion']
    assert old['case_rows']==p['case_rows'] and old['parameter_layout']==p['parameter_layout']
    return v28


'''

GRADIENT_BLOCK = '''        v28=v28_check(parent,p)
        saved_state=torch.load(v28/'outputs/update800/dgp_candidate_v28.pth',map_location='cuda',weights_only=True)
        original_tensors=original.net.state_dict();selected_names={name for name,_ in named}
        assert set(saved_state)==set(original_tensors)
        for name,value in saved_state.items():
            assert value.shape==original_tensors[name].shape and value.dtype==original_tensors[name].dtype and bool(torch.isfinite(value).all())
            if name not in selected_names:assert torch.equal(value,original_tensors[name]),'Frozen V28 state changed: '+name
        candidate.net.load_state_dict(saved_state,strict=True)
        del saved_state
        candidate_state=state_hash(candidate.net);assert candidate_state==p['final_V28_DGP_state']
        displacement=torch.cat([(v.detach().double()-dict(original.net.named_parameters())[name].detach().double()).reshape(-1) for name,v in named]).cpu().numpy().copy()
        np.save(out/'parameter_displacement.npy',displacement,allow_pickle=False)
        total=np.zeros((10,498627),dtype=np.float64);scalar_values=np.zeros(10,dtype=np.float64)
        rows=[];parity=[];keys=['x','base','target','mask','feature','interior','valid7','grid','truth','degraded_weight','clear_weight']
        gradient_folder=out/'gradients';gradient_folder.mkdir()
        replay_folder=out/'final_replay';replay_folder.mkdir()
        for begin in range(0,50,5):
            clock();group=items[begin:begin+5]
            b={key:torch.cat([item[key] for item in group]) for key in keys}
            prediction=candidate(b['x'],b['mask'])
            assert prediction.requires_grad and not torch.is_inference(prediction)
            raw=prediction.detach().permute(0,2,3,1).cpu().numpy().copy()
            delivered=np.stack([np.where(item['mask8'][...,None],np.floor(a*np.float32(255)),item['camera']).astype(np.uint8) for item,a in zip(group,raw)])
            with torch.no_grad():
                vectors=identity.embedding(torch.cat([as_tensor(a,'cuda') for a in delivered]),b['mask'],b['grid']).cpu().numpy().copy()
            for item,a,png,vector in zip(group,raw,delivered,vectors):
                cid=item['case']['id'];saved=np.load(v28/'outputs/update800'/(cid+'.npy'),allow_pickle=False)
                error=float(np.abs(a-saved).max());assert error<=p['fresh_same_VM_raw_tolerance']
                with Image.open(v28/'outputs/update800'/(cid+'.png')) as im:expected_png=np.asarray(im.convert('RGB')).copy()
                assert np.array_equal(png,expected_png),'Retain exact same-batch final delivered PNG: '+cid
                expected_vector=np.load(v28/'outputs/update800'/(cid+'_embedding.npy'),allow_pickle=False)
                vector_error=float(np.abs(vector-expected_vector).max());assert vector_error<=p['fresh_same_VM_vector_tolerance']
                initial_raw=np.load(v28/'outputs/initial_baseline'/(cid+'.npy'),allow_pickle=False)
                baseline_error=float(np.abs(item['base'][0].permute(1,2,0).cpu().numpy()-initial_raw).max());assert baseline_error<=p['fresh_same_VM_raw_tolerance']
                expected_truth=np.load(v28/'outputs/initial_baseline'/(cid+'_target_embedding.npy'),allow_pickle=False)
                truth_error=float(np.abs(item['truth'][0].cpu().numpy()-expected_truth).max());assert truth_error<=p['fresh_same_VM_vector_tolerance']
                np.save(replay_folder/(cid+'.npy'),a,allow_pickle=False);Image.fromarray(png).save(replay_folder/(cid+'.png'))
                np.save(replay_folder/(cid+'_embedding.npy'),vector,allow_pickle=False)
                parity.append({'id':cid,'final_raw_maximum_error':error,'final_PNG_exact':True,'final_vector_maximum_error':vector_error,'initial_raw_maximum_error':baseline_error,'truth_vector_maximum_error':truth_error})
            terms=objective_terms(b,prediction,identity,ns['mean'],ns['feature_errors'],ns['ssim'],normalizers)
            assert list(terms)==p['terms'][:7]
            dc=((prediction-b['base'])*b['mask']).sum((2,3))/b['mask'].sum((2,3)).clamp_min(1)
            base_pixel=ns['mean']((b['base']-b['target']).square(),b['mask'])
            terms['diagnostic_mean_shift_anchor']=dc.square().mean(1)/base_pixel.clamp_min(1e-5)
            terms['diagnostic_clear_SSIM_regression']=b['clear_weight']*terms['SSIM_regression']/5
            terms['diagnostic_clear_ArcFace_regression']=b['clear_weight']*terms['ArcFace_regression']/5
            assert list(terms)==p['terms']
            matrix=np.empty((10,498627),dtype=np.float64);batch_values=[]
            for index,name in enumerate(p['terms']):
                clock();scalar=terms[name].mean()/10
                assert bool(torch.isfinite(scalar))
                pieces=torch.autograd.grad(scalar,parameters,retain_graph=index<9,create_graph=False,allow_unused=False)
                progress['component_gradient_calls']+=1
                assert len(pieces)==12 and all(bool(torch.isfinite(g).all()) for g in pieces)
                matrix[index]=torch.cat([g.detach().reshape(-1).double() for g in pieces]).cpu().numpy()
                value=float(scalar.detach());scalar_values[index]+=value;batch_values.append(value)
            total+=matrix;path=gradient_folder/('batch'+str(begin//5)+'.npy');np.save(path,matrix,allow_pickle=False)
            row={'batch':begin//5,'ids':[item['case']['id'] for item in group],'component_values':batch_values,'component_norms':np.linalg.norm(matrix,axis=1).tolist(),'gradient_array_sha256':sha(path)}
            rows.append(row)
            assert state_hash(candidate.net)==candidate_state and state_hash(original.net)==original_state and state_hash(identity)==identity_state
            assert all(v.grad is None for v in candidate.parameters())
            print(json.dumps({'batch':begin//5+1,'of':10,'gradient_queries':progress['component_gradient_calls'],'component_norms':row['component_norms'],'seconds':time.monotonic()-start}),flush=True)
        norms=np.linalg.norm(total,axis=1);gram=total@total.T;denominator=norms[:,None]*norms[None,:]
        cosine=np.divide(gram,denominator,out=np.zeros_like(gram),where=denominator>0)
        blocks={}
        for parameter in layout:
            block=total[:,parameter['start']:parameter['end']];delta=displacement[parameter['start']:parameter['end']]
            blocks[parameter['name']]={'component_norms':np.linalg.norm(block,axis=1).tolist(),'component_dot_observed_displacement':(block@delta).tolist()}
        np.save(out/'gradient_components.npy',total,allow_pickle=False)
        write(out/'gradient_summary.json',{'complete':True,'terms':p['terms'],'parameter_layout':layout,'batches':rows,'component_values':scalar_values.tolist(),
            'original_seven_term_objective':float(scalar_values[:7].sum()),'component_norms':norms.tolist(),'component_gram':gram.tolist(),'component_cosines':cosine.tolist(),
            'per_parameter_gradients':blocks,'component_dot_observed_displacement':(total@displacement).tolist(),'gradient_array_sha256':sha(out/'gradient_components.npy'),
            'diagnostic_rows_are_not_a_new_training_objective':True})
        assert progress=={'reference_DGP_forwards':10,'candidate_DGP_forwards':10,'recognizer_forwards':30,'component_gradient_calls':100,'optimizer_updates':0,'epochs':0}
        assert state_hash(candidate.net)==candidate_state and state_hash(original.net)==original_state and state_hash(identity)==identity_state
        assert all(v.grad is None for v in candidate.parameters())
        assert all(not v.requires_grad and v.grad is None for v in original.parameters())
        assert all(not v.requires_grad and v.grad is None for v in identity.parameters())
        parent_check(parent,p);v28_check(parent,p);clock()
        write(out/'results.json',{'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-start,**progress,
            'candidate_state_before_after':candidate_state,'original_DGP_state_before_after':original_state,'recognizer_state_before_after':identity_state,
            'fresh_same_batch_replay_rows':parity,'actual_app_context':True,'optimizer_constructed':False,'backwards':0,'new_checkpoint_created':False,
            'V28_failed_gates_retained':True,'frozen_encoder_head4_and_buffers_unchanged':True,'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),
            'diagnostic_only':True,'new_training_recipe_created':False,'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False})
        print(json.dumps({'complete':True,'gradient_queries':100,'optimizer_updates':0,'seconds':time.monotonic()-start}),flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-start,**progress,
            'cause':str(exc),'traceback':traceback.format_exc(),'resume_permitted':False,'optimizer_constructed':False,'backwards':0,'new_checkpoint_created':False,
            'new_training_recipe_created':False,'app_promotion':False,'goal_complete':False})
        raise


'''


def main():
    assert not OUT.exists() and not PREP.exists()
    p28 = read(OLD / 'protocol.json')
    audit = read(ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json')
    control = read(ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/independent_readback.json')
    assert audit['complete'] and audit['training_completed800'] and not audit['necessary_capacity_pass']
    assert control['complete'] and control['fixed_control_is_insufficient']
    assert read(ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_visual_review/visual_review.json')['cases_reviewed'] == 50
    worker_original = OLD / 'scripts/cctv_dgp_active_original_decoder_v28_vm.py'
    assert sha(worker_original) == p28['assets_sha256']['scripts/cctv_dgp_active_original_decoder_v28_vm.py']
    source = worker_original.read_text(encoding='utf-8')
    prefix = source[:source.index('        total = np.zeros((7,498627)')]
    tail = source[source.index('def terminal_record(root):'):]
    source = prefix.replace('def original_functions(parent):', V28_CHECK + 'def original_functions(parent):') + GRADIENT_BLOCK + tail
    replacements = [
        ('"""Finite active-original-decoder L4 pilot. Historical all14 failure retained."""', '"""Zero-update final V28 preservation diagnostic. Every failed gate remains."""'),
        ("'own-DGP-active-original-decoder-capacity-v28'", "'own-DGP-final-v28-preservation-diagnostic-v1'"),
        ("p['optimizer_updates'] == 800 and p['epochs'] == 80", "p['optimizer_updates'] == 0 and p['epochs'] == 0"),
        ("p['component_gradient_calls'] == 70", "p['component_gradient_calls'] == 100"),
        ("assert root.name == 'cctv_dgp_active_original_decoder_vm_v28'", "assert root.name == '" + NAME + "'"),
        ('        sys.path.insert(0, str(root))', "        sys.path.insert(0, str(v28_check(parent,p)))\n        sys.path.insert(0, str(root))"),
        ("'V28 preflight300s/worker1800s cap'", "'Zero-update diagnostic cap300 seconds'"),
        ("'cctv-dgp-active-original-decoder-v28-results.tar.gz'", "'" + STEM + "-results.tar.gz'"),
        ("'cctv_dgp_active_original_decoder_v28_return/'", "'cctv_dgp_v28_preservation_diagnostic_v1_return/'"),
        ("'cctv-dgp-active-original-decoder-v28-export.json'", "'" + STEM + "-export.json'"),
        ("'cap_seconds':2100", "'cap_seconds':330"),
        ("'within_external_bound':elapsed<=2130", "'within_external_bound':elapsed<=360"),
        ("'V28 worker cap1800 seconds'", "'Zero-update diagnostic cap300 seconds'"),
        ('signal.alarm(1800)', 'signal.alarm(300)'),
    ]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    tree = ast.parse(source, feature_version=(3, 10))
    assert not any(isinstance(n, ast.Attribute) and n.attr in ['backward', 'step', 'zero_grad', 'AdamW', 'Adam', 'SGD'] for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'save' and isinstance(n.value, ast.Name) and n.value.id == 'torch' for n in ast.walk(tree))
    assert not any(isinstance(n, ast.Name) and n.id in ['optimizer', 'train'] for n in ast.walk(tree))
    shell = (OLD / 'scripts/run_v28.sh').read_text(encoding='utf-8')
    shell = shell.replace('scripts/cctv_dgp_active_original_decoder_v28_vm.py', WORKER).replace(' 2100s ', ' 330s ')
    assert shell.count(' 330s ') == 1 and shell.count(' 150s ') == 1
    OUT.mkdir();(OUT / 'scripts').mkdir();PREP.mkdir()
    (OUT / WORKER).write_text(source, encoding='utf-8', newline='\n')
    (OUT / 'scripts/run_v28_preservation.sh').write_text(shell, encoding='utf-8', newline='\n')
    assets = {name: sha(OUT / name) for name in [WORKER, 'scripts/run_v28_preservation.sh']}
    keys = ['decoder_parameters','decoder_parameter_tensors','parameter_layout','cases','references','batch_size','batches',
            'historical_CUDA_cache_tolerance','original_DGP_state','frozen_recognizer_state','closed_V27_protocol_sha256',
            'closed_V27_evidence_sha256','closed_R2_protocol_sha256','closed_R2_evidence_sha256','retained_capacity_gates','case_rows']
    p = {key: copy.deepcopy(p28[key]) for key in keys}
    evidence_names = ['outputs/results.json','outputs/execution_receipt.json','outputs/early_structure_stop.json',
                      'outputs/update0/metrics.json','outputs/update800/metrics.json','outputs/gradient_preflight.json',
                      'outputs/update800/dgp_candidate_v28.pth']
    for case in p28['case_rows']:
        cid = case['id']
        evidence_names += ['outputs/initial_baseline/' + cid + suffix for suffix in ['.npy','.png','_target_embedding.npy']]
        evidence_names += ['outputs/update800/' + cid + suffix for suffix in ['.npy','.png','_embedding.npy']]
    p.update({'format':'own-DGP-final-v28-preservation-diagnostic-v1','date':'2026-10-06',
        'purpose':'Measure competing original losses, RGB mean shift and clear-only preservation derivatives at the immutable final800 V28 state; no optimizer or new training recipe.',
        'component_gradient_calls':100,'optimizer_updates':0,'epochs':0,'backwards':0,'new_checkpoint_created':False,
        'terms':p28['terms'] + ['diagnostic_mean_shift_anchor','diagnostic_clear_SSIM_regression','diagnostic_clear_ArcFace_regression'],
        'diagnostic_term_policy':'Original seven objective terms unchanged; mean RGB(pred-base)^2/base_pixel floor1e-5, clear_weight*original SSIM hinge/5, clear_weight*original ArcFace hinge/5. Every component mean/10 gives the whole50-case mean. Additional rows are diagnostic measurements only; their sum is not an optimizer objective.',
        'closed_V28_protocol_sha256':sha(OLD / 'protocol.json'),'closed_V28_evidence_sha256':{name:sha(RETURNED/name) for name in evidence_names},
        'final_V28_checkpoint_sha256':sha(RETURNED/'outputs/update800/dgp_candidate_v28.pth'),
        'final_V28_DGP_state':read(RETURNED/'outputs/results.json')['candidate_DGP_state'],
        'fresh_same_VM_raw_tolerance':2e-6,'fresh_same_VM_vector_tolerance':2e-6,'fresh_same_VM_final_PNG_parity':'Exact all50, same5-case context',
        'CPU_return_raw_tolerance':1e-5,'CPU_return_vector_tolerance':5e-5,
        'gradient_layout':[10,498627],'gradient_arrays':11,'gradient_saved_values_bound':54848970,
        'interpretation':'Local final-state derivatives show first-order objective competition only; they do not identify a unique trajectory cause, establish usefulness, choose loss weights or waive V28 failed gates.',
        'budgets':{'preflight_seconds':300,'worker_seconds':300,'external_seconds':330,'external_kill_grace_seconds':30,
                   'export_seconds':120,'external_export_seconds':150,'external_export_kill_grace_seconds':30,
                   'peak_vram_bytes':20*1024**3,'minimum_free_disk_bytes':2*1024**3,'maximum_export_uncompressed_bytes':600_000_000},
        'assets_sha256':assets,'original_VM_forward_counts':{'reference_DGP_forwards':10,'candidate_DGP_forwards':10,'recognizer_forwards':30},
        'failure_policy':'Stop first source/state/parity/finite/time failure; retain every partial array/log. No optimizer, resume, unchanged retry, native/reserved output or automatic follow-on.',
        'execution':'Human upload/SSH/tmux on existing NVIDIA L4 g2-standard-4 only',
        'native_or_reserved_used':False,'app_promotion':False,'independent_final_review':False,'goal_complete':False,
        'local_basis_sha256':{name:sha(ROOT/name) for name in [
            'scripts/prepare_cctv_dgp_v28_preservation_diagnostic_v1.py',
            'outputs/cctv_dgp_active_original_decoder_v28_independent_audit.json',
            'outputs/cctv_dgp_active_original_decoder_v28_visual_review/visual_review.json',
            'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/results.json',
            'outputs/cctv_dgp_active_original_decoder_v28_mean_control_v1/independent_readback.json']}})
    write(OUT / 'protocol.json', p);pin = sha(OUT / 'protocol.json')
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for name in ['protocol.json', *assets]:tar.add(OUT/name, arcname=NAME+'/'+name, recursive=False)
    archive_digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(archive_digest+'  '+archive.name+'\n')
    write(PREP/'preparation.json',{'complete':True,'preparer_sha256':sha(Path(__file__)),'protocol_sha256':pin,
        'archive_sha256':archive_digest,'archive_bytes':archive.stat().st_size,'packet_files':3,
        'V28_closed_evidence_bindings':len(evidence_names),'gradient_queries_bound':100,'optimizer_updates':0,
        'actual_VM_execution_started':False,'new_training_recipe_created':False,'app_promotion':False,'goal_complete':False})
    print(json.dumps(read(PREP/'preparation.json'), indent=2))


if __name__ == '__main__':main()
