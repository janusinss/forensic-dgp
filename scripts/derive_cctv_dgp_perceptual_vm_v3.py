"""Derive separate V3 runner/auditor sources; never edit the frozen V1 files."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen source anchor differs: ' + old[:90])
    return text.replace(old, new)


def write_new(name, text):
    with (ROOT / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def derive_sources():
    runner = (ROOT / 'scripts/run_cctv_dgp_pilot_vm.py').read_text(encoding='utf-8')
    runner = replace_once(runner, 'from models import DGPSynthesizer', 'from models import DGPSynthesizer as OriginalSynthesizer')
    marker = 'RUN_CONTEXT={}\n'
    setup = '''from cctv_dgp_perceptual_training_v3 import (configure_paths, verify_protocol_v3,
    protocol_sha_v3, starting_weights_v3, load_starting_state, CalibratedPerceptualV3, START_SHA, START_STATE)
from cctv_dgp_frozen_norm import install_frozen_instance_norm
import cctv_dgp_perceptual_training_v3 as recipe_module
verify_bundle = verify_protocol_v3
PilotPerceptual = CalibratedPerceptualV3


def DGPSynthesizer():
    model = OriginalSynthesizer()
    if install_frozen_instance_norm(model) != 5:
        raise ValueError('Expected five frozen InstanceNorm adapters')
    return model


RUN_CONTEXT={}
'''
    runner = replace_once(runner, marker, setup)
    runner = runner.replace('sha(root/"protocol.json")', 'protocol_sha_v3(root)')
    runner = runner.replace('root/protocol["weights"]["dgp"]', 'starting_weights_v3(root)')
    runner = runner.replace('torch.load(starting_weights_v3(root),map_location="cpu",weights_only=True)',
                            'load_starting_state(root)')
    runner = replace_once(runner, 'if updates>256 or total_updates>512:',
                          'if updates>protocol["updates_per_arm"] or total_updates>protocol["expected_total_updates"]:')
    runner = replace_once(runner, 'def preflight(root,out,protocol,model,identity,perceptual,deadline):',
                          'def single_policy_preflight(root,out,protocol,model,identity,perceptual,deadline):')
    runner = replace_once(runner, '    total=loss+.1*identity_term\n', '''    perceptual_gradient = torch.autograd.grad(perceptual(generated,batch["target"]), generated, retain_graph=True)[0]
    if not torch.isfinite(perceptual_gradient).all() or perceptual_gradient.abs().sum() <= 0:
        raise ValueError('Perceptual supervision has no finite restoration-input gradient')
    total=loss+.1*identity_term
''')
    runner = replace_once(runner, '"identity_input_gradient_mean":float(identity_gradient.abs().mean().item()),',
                          '"identity_input_gradient_mean":float(identity_gradient.abs().mean().item()),\n            "perceptual_input_gradient_mean":float(perceptual_gradient.abs().mean().item()),\n            "feature_policy":perceptual.feature_policy,')
    preflight = '''def preflight(root,out,protocol,model,identity,perceptual,deadline):
    before = state_hash(model)
    checks = {}
    folder = out/'preflight_checks'; folder.mkdir()
    for policy in ('postactivation','preactivation'):
        perceptual.select_policy(policy)
        sub = folder/policy; sub.mkdir()
        single_policy_preflight(root,sub,protocol,model,identity,perceptual,deadline)
        checks[policy] = read(sub/'preflight.json')
    perceptual.select_policy('postactivation')
    dataset = PilotDataset(root,protocol,protocol['validation_cases'][-6:])
    with torch.no_grad():
        tail = model(torch.stack([dataset[i]['low'] for i in range(6)]).cuda())
    if not torch.isfinite(tail).all() or state_hash(model) != before or before != START_STATE:
        raise ValueError('Corrected V3 preflight changed starting state')
    check_clock(deadline)
    combined = dict(checks['postactivation'])
    combined.update(feature_policy_checks=checks, zero_update_backward_calls=2,
                    six_image_eval_state_unchanged=True, six_image_eval_state_hash=before,
                    calibration_sha256=sha(recipe_module.BUNDLE_DIR/'perceptual_scales_v3.json'),
                    starting_checkpoint_sha256=START_SHA)
    write(out/'preflight.json',combined)
    print('V3 CUDA preflight passed: both feature policies, batch6 unchanged, zero updates',flush=True)
    return combined['teacher_state_before']


def preview(out,root,protocol,rows,stage):'''
    runner = replace_once(runner, 'def preview(out,root,protocol,rows,stage):', preflight)
    runner = replace_once(runner, '    for arm in protocol["arms"]:\n',
                          '    for arm in protocol["arms"]:\n        perceptual.select_policy(arm["feature_policy"])\n')
    runner = replace_once(runner, 'record={"arm":arm["id"],"epoch":epoch,"update":updates,"cases":list(batch["case_id"]),',
                          'record={"arm":arm["id"],"feature_policy":perceptual.feature_policy,"epoch":epoch,"update":updates,"cases":list(batch["case_id"]),')
    runner = replace_once(runner, '"initial_model_state":state_hash(model),"initial_buffers_hash":initial_buffers,',
                          '"initial_model_state":state_hash(model),"initial_buffers_hash":initial_buffers,\n          "perceptual_calibration_sha256":sha(recipe_module.BUNDLE_DIR/"perceptual_scales_v3.json"),\n          "starting_checkpoint_sha256":START_SHA,')
    runner = runner.replace('outputs/cctv_dgp_pilot', 'outputs/cctv_dgp_perceptual_v3')
    runner = runner.replace('"outputs/preflight"', '"outputs/preflight_perceptual_v3"')
    runner = replace_once(runner, '    args=parser.parse_args()\n', '''    parser.add_argument('--bundle-dir',type=Path,default=ROOT)
    parser.add_argument('--parent-return',type=Path)
    args=parser.parse_args()
    configure_paths(args.bundle_dir,args.parent_return)
''')

    auditor = (ROOT / 'scripts/audit_cctv_dgp_pilot_results.py').read_text(encoding='utf-8')
    anchor = '\n\n\ndef require(condition,message):'
    imports = '''
from cctv_dgp_perceptual_training_v3 import (configure_paths, verify_protocol_v3,
    protocol_sha_v3, starting_weights_v3, load_starting_state, START_SHA, START_STATE)
import cctv_dgp_perceptual_training_v3 as recipe_module
from audit_cctv_dgp_normfix_results import save_receipt
verify_bundle = verify_protocol_v3


def require(condition,message):'''
    auditor = replace_once(auditor, anchor, '\n'+imports)
    auditor = auditor.replace('sha(root/"protocol.json")', 'protocol_sha_v3(root)').replace("sha(root/'protocol.json')", 'protocol_sha_v3(root)')
    auditor = auditor.replace("root/protocol['weights']['dgp']", 'starting_weights_v3(root)')
    auditor = replace_once(auditor, 'torch.load(root/protocol["weights"]["dgp"],map_location=\'cpu\',weights_only=True)',
                           'load_starting_state(root)')
    auditor = auditor.replace('"protocol.json","protocol.sha256","environment.txt","cuda_runtime_before.txt","pilot.log"',
                              '"perceptual_protocol_v3.json","perceptual_protocol_v3.sha256","environment_perceptual_v3.txt","cuda_runtime_before.txt","pilot_perceptual_v3.log"')
    auditor = auditor.replace('outputs/cctv_dgp_pilot', 'outputs/cctv_dgp_perceptual_v3')
    auditor = replace_once(auditor, '    refs={r["id"]:r for r in protocol["references"]};', '''    require(preflight['zero_update_backward_calls']==2 and preflight['six_image_eval_state_unchanged']
            and preflight['six_image_eval_state_hash']==START_STATE
            and preflight['starting_checkpoint_sha256']==execution['starting_checkpoint_sha256']==START_SHA,
            'V3 starting/zero-update/tail preflight differs')
    require(preflight['calibration_sha256']==execution['perceptual_calibration_sha256']==sha(recipe_module.BUNDLE_DIR/'perceptual_scales_v3.json'),
            'V3 fixed calibration binding differs')
    for policy in ('postactivation','preactivation'):
        check=preflight['feature_policy_checks'][policy]
        require(check['passed'] and check['feature_policy']==policy and check['optimizer_updates']==0
                and check['model_state_hash']==START_STATE and check['model_state_unchanged']
                and math.isfinite(check['perceptual_input_gradient_mean']) and check['perceptual_input_gradient_mean']>0
                and math.isfinite(check['identity_input_gradient_mean']) and check['identity_input_gradient_mean']>0
                and check['teacher_state_before']==execution['teacher_states'] and not check['optimizer_constructed'],
                'V3 feature/identity gradient preflight differs')
        require(read(out/'preflight_checks'/policy/'preflight.json')==check,'V3 preflight source report differs')
    refs={r["id"]:r for r in protocol["references"]};''')
    auditor = replace_once(auditor, "require(trace['arm']==aid and trace['epoch']==epoch", "require(trace['feature_policy']==arm['feature_policy'] and trace['arm']==aid and trace['epoch']==epoch")
    auditor = replace_once(auditor, "'production_promoted':False,'native_visual_review_pending':True,'independent_final_review_pending':True}",
                          "'production_promoted':False,'native_visual_review_pending':True,'independent_final_review_pending':True,\n            'perceptual_pilot_version':3,'calibration_sha256':sha(recipe_module.BUNDLE_DIR/'perceptual_scales_v3.json')}")
    auditor = replace_once(auditor, '    args=parser.parse_args()\n', '''    parser.add_argument('--bundle-dir',type=Path,default=ROOT)
    parser.add_argument('--parent-return',type=Path)
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    configure_paths(args.bundle_dir,args.parent_return)
''')
    auditor = replace_once(auditor, "require(sha(args.extract_to/'protocol.json')==sha(args.root/'protocol.json'),'Transferred protocol differs')",
                          "require(sha(args.extract_to/'perceptual_protocol_v3.json')==protocol_sha_v3(args.root),'Transferred V3 protocol differs')")
    auditor = replace_once(auditor, "    if not (out/'independent_audit.json').exists():\n        write(out/'independent_audit.json',report)",
                          "    save_receipt(out,report,args.receipt)")
    return {'scripts/run_cctv_dgp_perceptual_vm_v3.py': runner,
            'scripts/audit_cctv_dgp_perceptual_results_v3.py': auditor}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, content in derive_sources().items():
        if args.check:
            if (ROOT / name).read_text(encoding='utf-8') != content:
                raise ValueError('Derived V3 source differs: ' + name)
        else:
            write_new(name, content)
    print('Two separate V3 sources verified; frozen originals unchanged' if args.check
          else 'Two separate V3 sources derived; frozen originals unchanged')


if __name__ == '__main__':
    main()
