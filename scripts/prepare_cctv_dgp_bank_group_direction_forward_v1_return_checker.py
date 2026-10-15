"""Check future audit arithmetic and retained gates against actual saved controls."""
import ast
import copy
from datetime import datetime, timezone
import importlib.util
import sys
import numpy as np
from PIL import Image
import torch
from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, PARENT, read, sha, write


def main():
    receipt=OUT/'cctv_dgp_bank_group_direction_forward_v1_return_checker_preparation.json'
    assert not receipt.exists()
    worker_path=ROOT/'scripts/audit_cctv_dgp_bank_group_direction_forward_v1_return.py'
    importer_path=ROOT/'scripts/import_cctv_dgp_bank_group_direction_forward_v1_return.py'
    for file in [worker_path,importer_path]:
        ast.parse(file.read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('prospective_metric_audit',worker_path)
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    torch.set_grad_enabled(False)
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support,capacity
    p=read(PARENT/'protocol.json')
    baseline=read(PARENT/'outputs/bank_comparison_v1/baseline/metrics.json')
    expected={r['id']:r for r in baseline['rows']}
    refs={r['id']:r for r in p['references']}
    checked=[]
    for begin in range(0,3905,5):
        block=p['cases'][begin:begin+5]
        if block[0]['id'] not in p['preview_case_ids']:
            continue
        raw,emb=unpack(PARENT/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz',[c['id'] for c in block])
        for i,c in enumerate(block):
            ref=refs[c['source_person_or_reference']]
            camera=audit.image(PARENT/c['input'])
            target=audit.image(PARENT/ref['target'])
            mask=audit.image(PARENT/ref['observed'],'L')>0
            patch=audit.support(mask,c['landmarks5_canvas_xy'])
            assert np.array_equal(patch,feature_support(mask,c['landmarks5_canvas_xy']))
            png=audit.delivered(raw[i],camera,mask)
            assert np.array_equal(png,audit.image(PARENT/'outputs/bank_comparison_v1/baseline/previews'/(c['id']+'.png')))
            values={
                'raw':audit.metrics(raw[i],target,mask,patch,emb[i,0],emb[i,2],raw[i]),
                'png':audit.metrics(png.astype(np.float32)/255,target,mask,patch,emb[i,1],emb[i,2],png.astype(np.float32)/255,prediction8=png),
            }
            for stage,row in values.items():
                for key,value in row.items():
                    other=expected[c['id']][stage][key]
                    if value is None or isinstance(value,bool):
                        assert value==other
                    else:
                        assert np.isclose(value,other,rtol=0,atol=1e-13),(c['id'],stage,key,value,other)
            checked.append(dict(id=c['id'],source=c['source'],profile=c['profile'],**values))
    assert len(checked)==100
    before=audit.grouped(checked,'raw')
    assert len(before)==17
    fixture=copy.deepcopy(before)
    for value in fixture.values():
        value['MSE'] *= .99
        value['constant_mean_shift_only_MSE']=value['MSE']+.005
        value['landmark_high_frequency_MSE'] *= .98
    tests=[]
    for name,mutator in [
        ('declared_positive_synthetic_fixture',lambda v:None),
        ('clear_MSE_regression',lambda v:v['clear'].update(MSE=before['clear']['MSE']+1e-10)),
        ('SSIM_regression',lambda v:v['all'].update(SSIM=before['all']['SSIM']-2e-6)),
        ('ArcFace_regression',lambda v:v['degraded'].update(ArcFace_observed_fixed=before['degraded']['ArcFace_observed_fixed']-2e-6)),
        ('less_than_one_percent_structure',lambda v:v['degraded'].update(landmark_high_frequency_MSE=before['degraded']['landmark_high_frequency_MSE']*.995)),
        ('excess_brightness_fraction',lambda v:v['degraded'].update(constant_mean_shift_only_MSE=before['degraded']['MSE']*.997)),
    ]:
        after=copy.deepcopy(fixture)
        mutator(after)
        result=audit.compare_groups(before,after)
        retained=capacity(before,after,.01)
        assert result['pass_all_preview_requirements']==retained['pass']
        assert result['pass_all_preview_requirements']==(name=='declared_positive_synthetic_fixture')
        assert result['relative_feature_gain']==retained['relative_feature_gain']
        assert result['brightness_gain_fraction']==retained['brightness_gain_fraction']
        assert result['preservation_failures']==retained['preservation_failures']
        tests.append(dict(name=name,synthetic_fixture_not_model_data=True,retained_capacity_logic_exact=True,
                          expected_pass=name=='declared_positive_synthetic_fixture'))
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        importer_sha256=sha(importer_path),auditor_sha256=sha(worker_path),
        protocol_sha256=sha(OUT/'cctv_dgp_bank_group_direction_forward_v1_vm/protocol.json'),
        prospective_import_and_audit_AST_checked=True,actual_saved_control_cases_checked=100,
        raw_PNG_metric_rows_checked=200,observed_and_feature_masks_recomputed=True,
        exact_retained_PNGs_checked=100,original_capacity_logic_six_boundary_tests=tests,
        synthetic_boundary_fixtures_are_not_restoration_results=True,
        full_future_return_audit_unrun=True,no_future_return_fabricated=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0,
        model_qualified=False,goal_complete=False))
    print(dict(complete=True,actual_control_cases=100,raw_PNG_rows=200,retained_gate_boundary_tests=6))


if __name__=='__main__':
    main()
