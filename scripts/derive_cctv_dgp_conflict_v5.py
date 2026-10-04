"""Derive separate V5 runner/auditor; preserve all executed V1-V4 sources."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen source anchor differs: '+old[:90])
    return text.replace(old, new)


PREFLIGHT = '''def parameter_gradients(reconstruction_loss, identity_loss, parameters, policy):
    first = torch.autograd.grad(reconstruction_loss, parameters, retain_graph=True, allow_unused=True)
    RUN_CONTEXT['autograd_grad_calls'] += 1
    second = torch.autograd.grad(identity_loss, parameters, allow_unused=True)
    RUN_CONTEXT['autograd_grad_calls'] += 1
    used = tuple(a is not None or b is not None for a, b in zip(first, second))
    r = tuple(a if a is not None else torch.zeros_like(p) for a, p in zip(first, parameters))
    i = tuple(a if a is not None else torch.zeros_like(p) for a, p in zip(second, parameters))
    merged, receipt = combine_gradients(r, i, policy)
    if receipt['combined_l2'] <= 0 or not any(used):
        raise ValueError('Empty combined restoration gradient; stop without an optimizer update')
    return merged, receipt, used


def preflight(root,out,protocol,model,identity,perceptual,deadline):
    import onnxruntime as ort
    before = state_hash(model)
    teacher_before = {'identity':state_hash(identity), 'perceptual':state_hash(perceptual)}
    if before != START_STATE or teacher_before != TEACHERS:
        raise ValueError('V5 preflight starting student/teacher state differs')
    loader = DataLoader(PilotDataset(root,protocol,protocol['training_epochs']['1']),batch_size=8,shuffle=False,num_workers=0)
    batch = device_batch(next(iter(loader)))
    options = ort.SessionOptions(); options.intra_op_num_threads=4; options.inter_op_num_threads=1
    reference = ort.InferenceSession(str(root/protocol['weights']['arcface']),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.no_grad():
        crop = identity_crop(batch['target'][:1],batch['mask'][:1],batch['grid'][:1])*2-1
        expected = reference.run(None,{reference.get_inputs()[0].name:crop.cpu().numpy()})[0]
        actual = identity.encoder(crop).cpu().numpy()
        reference_embed = identity.embedding(batch['target'],batch['mask'],batch['grid'])
    np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-4)
    parameters = tuple(model.parameters())
    checks = {}; folder = out/'preflight_checks'; folder.mkdir()
    perceptual.select_policy('postactivation'); model.eval().requires_grad_(True)
    for arm in protocol['arms']:
        check_clock(deadline)
        generated = composite(model(batch['low']),batch['low'],batch['mask'])
        reconstruction,_ = base_loss(generated,batch['target'],batch['mask'],perceptual)
        identity_term = (1-(identity.embedding(generated,batch['mask'],batch['grid'])*reference_embed).sum(1)).mean()
        identity_gradient = torch.autograd.grad(identity_term,generated,retain_graph=True)[0]
        RUN_CONTEXT['autograd_grad_calls'] += 1
        perceptual_gradient = torch.autograd.grad(perceptual(generated,batch['target']),generated,retain_graph=True)[0]
        RUN_CONTEXT['autograd_grad_calls'] += 1
        for gradient in (identity_gradient,perceptual_gradient):
            if not torch.isfinite(gradient).all() or gradient.abs().sum() <= 0:
                raise ValueError('Nonfinite/empty frozen-supervision input gradient')
        _, gradient_receipt, _ = parameter_gradients(reconstruction,arm['lambda_identity']*identity_term,parameters,arm['gradient_policy'])
        if any(p.grad is not None for m in (model,identity,perceptual) for p in m.parameters()):
            raise ValueError('Preflight unexpectedly accumulated parameter gradients')
        if state_hash(model) != before or {'identity':state_hash(identity),'perceptual':state_hash(perceptual)} != teacher_before:
            raise ValueError('Zero-update V5 preflight changed student/teachers')
        check = {'passed':True, 'arm':arm, 'feature_policy':perceptual.feature_policy,
                 'optimizer_constructed':False, 'optimizer_updates':0, 'autograd_grad_calls':4,
                 'model_state_hash':before, 'model_state_unchanged':True, 'teacher_state_before':teacher_before,
                 'identity_input_gradient_mean':float(identity_gradient.abs().mean()),
                 'perceptual_input_gradient_mean':float(perceptual_gradient.abs().mean()),
                 'policy_gradients':gradient_receipt}
        sub = folder/arm['id']; sub.mkdir(); write(sub/'preflight.json',check); checks[arm['id']]=check
    tail_dataset = PilotDataset(root,protocol,protocol['training_epochs']['1'][-6:])
    with torch.no_grad():
        tail = model(torch.stack([tail_dataset[i]['low'] for i in range(6)]).cuda())
    if not torch.isfinite(tail).all() or state_hash(model) != before:
        raise ValueError('Six-image evaluation changed V5 normalization state')
    if RUN_CONTEXT['autograd_grad_calls'] != 8:
        raise ValueError('V5 zero-update autograd budget differs')
    check_clock(deadline)
    report = {'passed':True, 'batch_size':len(batch['low']), 'optimizer_constructed':False,
              'optimizer_updates':0, 'zero_update_backward_calls':0, 'autograd_grad_calls':8,
              'identity_input_gradient_mean':checks[protocol['arms'][0]['id']]['identity_input_gradient_mean'],
              'model_state_unchanged':True, 'model_state_hash':before, 'teacher_state_before':teacher_before,
              'gradient_policy_checks':checks, 'six_image_eval_state_unchanged':True,
              'six_image_eval_state_hash':before, 'starting_checkpoint_sha256':START_SHA,
              'onnx_conversion_max_error':float(np.abs(expected-actual).max()),
              'calibration_sha256':sha(recipe_module.PERCEPTUAL_DIR/'perceptual_scales_v3.json'),
              'protocol_sha256':protocol_sha_v5(root), 'gpu':torch.cuda.get_device_name(0),
              'vram_bytes':torch.cuda.get_device_properties(0).total_memory,
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(), 'torch':torch.__version__,
              'full_pilot_pending':True}
    write(out/'preflight.json',report)
    print('V5 CUDA preflight passed: both gradient policies, batch8 and batch6, zero updates',flush=True)
    return teacher_before


'''


AUDIT_PREFLIGHT = '''    require(preflight['zero_update_backward_calls']==0 and preflight['autograd_grad_calls']==8
            and preflight['six_image_eval_state_unchanged'] and preflight['six_image_eval_state_hash']==START_STATE
            and preflight['starting_checkpoint_sha256']==execution['starting_checkpoint_sha256']==START_SHA,
            'V5 starting/zero-update/tail preflight differs')
    require(execution['runtime_cap_seconds']==1800 and execution['teacher_states']==TEACHERS
            and execution['gradient_policies']==protocol['arms']
            and report['preflight_autograd_grad_calls']==8 and report['training_autograd_grad_calls']==904
            and report['total_autograd_grad_calls']==912 and not report['cuda_gradients_replayed_by_audit'],
            'V5 frozen teachers/policy/autograd budget differs')
    require(preflight['calibration_sha256']==execution['perceptual_calibration_sha256']==sha(recipe_module.PERCEPTUAL_DIR/'perceptual_scales_v3.json'),
            'Fixed postactivation perceptual binding differs')
    recipe=read(recipe_module.BUNDLE_DIR/'conflict_protocol_v5.json')
    expected_sources={'runtime_sources/'+name for name in recipe['assets_sha256']}
    actual_sources={name for name in report['artifacts_sha256'] if name.startswith('runtime_sources/')}
    require(actual_sources==expected_sources,'V5 returned runtime source inventory differs')
    for name,pin in recipe['assets_sha256'].items():
        require(sha(out/'runtime_sources'/name)==pin,'Executed V5 source/capsule differs: '+name)
    require(set(preflight['gradient_policy_checks'])=={a['id'] for a in protocol['arms']},'Preflight arm coverage differs')
    for arm in protocol['arms']:
        check=preflight['gradient_policy_checks'][arm['id']]
        require(check['passed'] and check['arm']==arm and check['feature_policy']=='postactivation'
                and check['optimizer_updates']==0 and not check['optimizer_constructed'] and check['autograd_grad_calls']==4
                and check['model_state_hash']==START_STATE and check['model_state_unchanged']
                and math.isfinite(check['perceptual_input_gradient_mean']) and check['perceptual_input_gradient_mean']>0
                and math.isfinite(check['identity_input_gradient_mean']) and check['identity_input_gradient_mean']>0
                and check['teacher_state_before']==TEACHERS,'V5 gradient-policy preflight differs')
        same(check['policy_gradients'],rebuild_policy_summary(arm['gradient_policy'],execution['student_parameter_dimension'],check['policy_gradients']['gram_matrix']),'preflight gradient arithmetic',1e-8)
        require(read(out/'preflight_checks'/arm['id']/'preflight.json')==check,'Preflight source report differs')
'''

AUDIT_TRACE = '''def validate_gradient_trace(trace, arm, dimension, position):
    require(trace['gradient_policy']==arm['gradient_policy'] and trace['autograd_grad_calls']==2
            and trace['cumulative_autograd_grad_calls']==8+2*position,'Gradient policy/autograd trace differs')
    rebuilt=rebuild_policy_summary(arm['gradient_policy'],dimension,trace['policy_gradients']['gram_matrix'])
    same(trace['policy_gradients'],rebuilt,'policy gradient arithmetic',1e-8)
    require(rebuilt['combined_l2']>0 and math.isfinite(trace['preclip_norm'])
            and abs(trace['preclip_norm']-rebuilt['combined_l2'])<=5e-5*max(1.,rebuilt['combined_l2']),
            'Actual pre-clipping norm differs from objective Gram')
    return rebuilt


'''


def derive_sources():
    runner = (ROOT/'scripts/run_cctv_dgp_perceptual_vm_v3.py').read_text(encoding='utf-8')
    runner = replace_once(runner, 'from cctv_dgp_perceptual_training_v3 import (configure_paths, verify_protocol_v3,\n    protocol_sha_v3, starting_weights_v3, load_starting_state, CalibratedPerceptualV3, START_SHA, START_STATE)',
        'from cctv_dgp_conflict_training_v5 import (configure_paths, verify_protocol_v5,\n    protocol_sha_v5, starting_weights_v5, load_starting_state, CalibratedPerceptualV3, START_SHA, START_STATE, combine_gradients, TEACHERS)')
    runner = runner.replace('import cctv_dgp_perceptual_training_v3 as recipe_module','import cctv_dgp_conflict_training_v5 as recipe_module')
    runner = runner.replace('verify_bundle = verify_protocol_v3','verify_bundle = verify_protocol_v5')
    runner = runner.replace('protocol_sha_v3(', 'protocol_sha_v5(')
    start, end = runner.index('def single_policy_preflight('), runner.index('def preview(')
    runner = runner[:start] + PREFLIGHT + runner[end:]
    runner = runner.replace('5400','1800').replace('Finite90-minute','Finite30-minute').replace('frozen90-minute','frozen30-minute')
    runner = replace_once(runner, 'RUN_CONTEXT["total_updates"]=0', 'RUN_CONTEXT["total_updates"]=0;RUN_CONTEXT["autograd_grad_calls"]=0')
    runner = replace_once(runner, '    torch.set_num_threads(4);torch.manual_seed(protocol["seed"])', '''    import shutil
    for name in read(recipe_module.BUNDLE_DIR/'conflict_protocol_v5.json')['assets_sha256']:
        target=out/'runtime_sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(recipe_module.BUNDLE_DIR/name,target)
    torch.set_num_threads(4);torch.manual_seed(protocol["seed"])''')
    runner = runner.replace('recipe_module.BUNDLE_DIR/"perceptual_scales_v3.json"','recipe_module.PERCEPTUAL_DIR/"perceptual_scales_v3.json"')
    runner = replace_once(runner, '"teacher_states":teacher_before,"runtime_cap_seconds":1800,',
        '"teacher_states":teacher_before,"runtime_cap_seconds":1800,\n          "gradient_policies":protocol["arms"],"student_parameter_dimension":sum(p.numel() for p in model.parameters()),')
    runner = replace_once(runner, '                total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)\n                optimizer.step();updates+=1;total_updates+=1', '''                parameters=tuple(model.parameters())
                merged,gradient_receipt,used=parameter_gradients(loss,arm["lambda_identity"]*id_loss,parameters,arm["gradient_policy"])
                for p,g,active in zip(parameters,merged,used):
                    p.grad=g if active else None
                preclip=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
                if any(p.grad is not None for m in (identity,perceptual) for p in m.parameters()):
                    raise ValueError('Frozen teacher accumulated a parameter gradient')
                if updates>=protocol['updates_per_arm'] or total_updates>=protocol['expected_total_updates']:
                    raise ValueError('Finite update cap would be exceeded')
                optimizer.step();updates+=1;total_updates+=1''')
    runner = replace_once(runner, '"loss":float(total.item()),"identity_loss":float(id_loss.item()),"components":components,',
        '"loss":float(total.item()),"identity_loss":float(id_loss.item()),"components":components,\n                        "gradient_policy":arm["gradient_policy"],"policy_gradients":gradient_receipt,"preclip_norm":preclip,\n                        "autograd_grad_calls":2,"cumulative_autograd_grad_calls":RUN_CONTEXT["autograd_grad_calls"],')
    runner = replace_once(runner, '    if total_updates!=protocol["expected_total_updates"]:', '''    if RUN_CONTEXT['autograd_grad_calls']!=912:
        raise ValueError('Actual V5 autograd traversal count differs')
    if total_updates!=protocol["expected_total_updates"]:''')
    runner = replace_once(runner, '"total_optimizer_updates":total_updates,"elapsed_seconds":time.monotonic()-start,',
        '"total_optimizer_updates":total_updates,"elapsed_seconds":time.monotonic()-start,\n            "preflight_autograd_grad_calls":8,"training_autograd_grad_calls":904,"total_autograd_grad_calls":RUN_CONTEXT["autograd_grad_calls"],\n            "cuda_gradients_replayed_by_audit":False,')
    runner = runner.replace('outputs/preflight_perceptual_v3','outputs/preflight_conflict_v5').replace('outputs/cctv_dgp_perceptual_v3','outputs/cctv_dgp_conflict_v5')
    runner = replace_once(runner, '    configure_paths(args.bundle_dir,args.parent_return)', '''    configure_paths(args.bundle_dir,args.parent_return,args.perceptual_bundle_dir,args.diagnostic_bundle_dir,args.v3_return,args.v4_return)''')
    runner = replace_once(runner, '    args=parser.parse_args()', '''    parser.add_argument('--perceptual-bundle-dir',type=Path)
    parser.add_argument('--diagnostic-bundle-dir',type=Path)
    parser.add_argument('--v3-return',type=Path)
    parser.add_argument('--v4-return',type=Path)
    args=parser.parse_args()''')
    runner = runner.replace('Matched camera/identity diagnostic','Matched V5 identity-weight/gradient-conflict training pilot')
    runner = runner.replace('Matched CCTV DGP diagnostic complete.', 'V5 matched conflict pilot complete.')
    auditor = (ROOT/'scripts/audit_cctv_dgp_perceptual_results_v3.py').read_text(encoding='utf-8')
    auditor = replace_once(auditor, 'from cctv_dgp_perceptual_training_v3 import (configure_paths, verify_protocol_v3,\n    protocol_sha_v3, starting_weights_v3, load_starting_state, START_SHA, START_STATE)',
        'from cctv_dgp_conflict_training_v5 import (configure_paths, verify_protocol_v5,\n    protocol_sha_v5, starting_weights_v5, load_starting_state, START_SHA, START_STATE, rebuild_policy_summary, TEACHERS)')
    auditor = auditor.replace('import cctv_dgp_perceptual_training_v3 as recipe_module','import cctv_dgp_conflict_training_v5 as recipe_module')
    auditor = auditor.replace('verify_bundle = verify_protocol_v3','verify_bundle = verify_protocol_v5').replace('protocol_sha_v3(', 'protocol_sha_v5(')
    auditor = auditor.replace('perceptual_protocol_v3', 'conflict_protocol_v5').replace('environment_perceptual_v3.txt','environment_conflict_v5.txt').replace('pilot_perceptual_v3.log','pilot_conflict_v5.log').replace('outputs/cctv_dgp_perceptual_v3','outputs/cctv_dgp_conflict_v5')
    start,end=auditor.index("    require(preflight['zero_update_backward_calls']"),auditor.index('    refs={r["id"]:r for r in protocol["references"]};')
    auditor=auditor[:start]+AUDIT_PREFLIGHT+auditor[end:]
    auditor=auditor.replace('report["elapsed_seconds"]<=5400','0<report["elapsed_seconds"]<=1800')
    auditor=replace_once(auditor,'def unit_embedding(path):',AUDIT_TRACE+'def unit_embedding(path):')
    auditor=replace_once(auditor, '                expected_loss=trace[\'components\'][\'pixel\']', '''                validate_gradient_trace(trace,arm,execution['student_parameter_dimension'],position)
                expected_loss=trace['components']['pixel']''')
    auditor=replace_once(auditor, "'perceptual_pilot_version':3,'calibration_sha256':sha(recipe_module.BUNDLE_DIR/'perceptual_scales_v3.json')", ''''conflict_pilot_version':5,'calibration_sha256':sha(recipe_module.PERCEPTUAL_DIR/'perceptual_scales_v3.json'),
            'policy_summaries_rebuilt':position+2,'recorded_vm_autograd_grad_calls':912,'local_backward_calls':0,
            'cuda_gradients_recomputed_by_this_audit':False''')
    auditor=replace_once(auditor, '    configure_paths(args.bundle_dir,args.parent_return)', '    configure_paths(args.bundle_dir,args.parent_return,args.perceptual_bundle_dir,args.diagnostic_bundle_dir,args.v3_return,args.v4_return)')
    auditor=replace_once(auditor, '    args=parser.parse_args()', '''    parser.add_argument('--perceptual-bundle-dir',type=Path)
    parser.add_argument('--diagnostic-bundle-dir',type=Path)
    parser.add_argument('--v3-return',type=Path)
    parser.add_argument('--v4-return',type=Path)
    args=parser.parse_args()''')
    auditor=auditor.replace('Transferred V3 protocol differs','Transferred V5 protocol differs')
    # The current return has a strict LF checksum, including its exact filename.
    auditor=replace_once(auditor, "    raw=Path(str(archive)+\".sha256\").read_bytes().decode('ascii')", '''    require(Path(str(archive)+".sha256").read_bytes()==(sha(archive)+'  '+archive.name+'\\n').encode('ascii'),'Transfer checksum/filename/LF differs')
    raw=Path(str(archive)+".sha256").read_bytes().decode('ascii')''')
    return {'scripts/run_cctv_dgp_conflict_vm_v5.py':runner,
            'scripts/audit_cctv_dgp_conflict_results_v5.py':auditor}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    for name, content in derive_sources().items():
        if args.check:
            if (ROOT/name).read_text(encoding='utf-8') != content:
                raise ValueError('Derived V5 source differs: '+name)
        else:
            with (ROOT/name).open('x',encoding='utf-8',newline='\n') as stream:
                stream.write(content)
    print('Two separate V5 sources verified' if args.check else 'Two separate V5 sources derived; V1-V4 unchanged')
