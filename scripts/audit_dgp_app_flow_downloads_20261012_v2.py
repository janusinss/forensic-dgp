"""Check real native DEV downloads, refusal receipts and unchanged app sources."""
import ast
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from audit_dgp_app_flow_downloads_20261012_v1 import decode, sha, read, array_sha

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'scratch/app_flow_20261012_v2'


def pixels(data, size=(256,256), mode='RGB'):
    with Image.open(io.BytesIO(data)) as image:
        assert image.size==size and image.format=='PNG'
        return np.asarray(image.convert(mode)).copy()


def main():
    receipt=ART/'download_content_audit.json'
    assert not receipt.exists()
    pre=read(ART/'preflight.json')
    for name,h in pre['source_and_weight_files_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in pre['fixtures_files_sha256'].items():assert sha(ART/name)==h,name
    native=ROOT/pre['native_DEV_input']
    assert sha(native)==pre['native_DEV_SHA256']
    contract=ROOT/'outputs/cctv_dgp_native_contract_audit_20261009_v1/audit.json'
    canonical='outputs/cctv_chokepoint_native_development_v1/native_crops/choke_dev_01_t067.png'
    bindings=read(contract)['encoded_asset_hashes_verified']
    assert bindings[canonical]==sha(ROOT/canonical)==sha(native)
    input_review_path=ROOT/'outputs/cctv_chokepoint_native_development_v1/input_review.json'
    input_review=read(input_review_path)
    frozen=next(r for r in input_review['rows'] if r['id']=='choke_dev_01_t067')
    assert input_review['reviewed_before_model_outputs'] and frozen['input_review']=='usable'
    assert frozen['native_crop_sha256']==sha(native) and not frozen['model_outputs_viewed']
    assert not input_review['reserved_pixels_viewed']
    original=pixels(native.read_bytes(),(138,165))
    proposal=pixels((ART/'native-automatic-mask.png').read_bytes(),(138,165),'L')
    assert not proposal.any()
    assert np.array_equal(pixels((ART/'native-original.png').read_bytes(),(138,165)),original)
    canvas=Image.new('RGB',(165,165),(128,128,128));canvas.paste(Image.fromarray(original),(13,0))
    prepared=np.asarray(canvas.resize((256,256),Image.Resampling.BILINEAR)).copy()
    support=Image.new('L',(165,165),0);support.paste(255,(13,0,151,165))
    observed=np.asarray(support.resize((256,256),Image.Resampling.NEAREST))>0
    assert int(observed.sum())==54784
    bounds=[int(np.nonzero(observed)[1].min()),int(np.nonzero(observed)[0].min()),
            int(np.nonzero(observed)[1].max()+1),int(np.nonzero(observed)[0].max()+1)]
    assert bounds==[20,0,234,256]
    zero_native=np.zeros((165,138),np.uint8);zero=np.zeros((256,256),np.uint8)
    cases=[];outputs={}
    for mode in ['off','on','auto']:
        label='native-'+mode;response_path=ART/(label+'-response.json')
        response=read(response_path);meta=response['metadata'];applied=mode=='on'
        output_bytes=decode(response['output'],'image/png');bundle=decode(response['bundle'],'application/zip')
        assert output_bytes==(ART/(label+'-output.png')).read_bytes()==(ART/'downloads'/(label+'-face-estimate.png')).read_bytes()
        assert bundle==(ART/(label+'-bundle.zip')).read_bytes()==(ART/'downloads'/(label+'-face-review.zip')).read_bytes()
        output=pixels(output_bytes);outputs[mode]=output
        assert np.array_equal(pixels(decode(response['original'],'image/png')),prepared)
        assert np.array_equal(pixels(decode(response['mask'],'image/png'),mode='L'),zero)
        assert meta['restoration_requested']==mode and meta['restoration_applied']==applied
        assert meta['mask_source']=='automatic_reviewed' and meta['input_review']['decision']=='usable'
        assert meta['optimizer_updates']==meta['backward_calls']==0
        assert not meta['native_usefulness_qualified'] and not meta['covering_families_qualified']
        assert meta['original_rgb_sha256']==array_sha(original)
        assert meta['native_mask_sha256']==array_sha(zero_native)
        assert meta['output_rgb_sha256']==array_sha(output)
        geometry=meta['geometry']
        assert geometry['native_size']==[138,165] and geometry['native_center_offset']==[13,0]
        assert geometry['observed_bounds_256']==bounds and geometry['output_size']==[256,256]
        assert geometry['input_rgb_sha256']==array_sha(original)
        assert geometry['canvas_rgb_sha256']==array_sha(prepared)
        assert geometry['observed_sha256']==array_sha(observed)
        assert geometry['removal_mask_sha256']==array_sha(zero.astype(bool))
        assert meta['completion']=={'bypassed':'empty mask'} and meta['additional_mask_expansion']==0
        assert np.array_equal(output[~observed],prepared[~observed])
        assert not meta['display_processing']['colour_policy']['applied']
        names={'original.png','removal-mask-original.png','input-256.png','removal-mask.png','estimate.png','processing.json','README.txt'}
        if applied:names.add('dgp-raw-float32.npy')
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            entries=archive.infolist()
            assert len(entries)==len(names) and {e.filename for e in entries}==names
            assert all(not e.is_dir() and '/' not in e.filename and '\\' not in e.filename and e.file_size<4*1024**2 for e in entries)
            assert archive.testzip() is None and json.loads(archive.read('processing.json'))==meta
            assert np.array_equal(pixels(archive.read('original.png'),(138,165)),original)
            assert np.array_equal(pixels(archive.read('removal-mask-original.png'),(138,165),'L'),zero_native)
            assert np.array_equal(pixels(archive.read('input-256.png')),prepared)
            assert np.array_equal(pixels(archive.read('removal-mask.png'),mode='L'),zero)
            assert archive.read('estimate.png')==output_bytes
            assert 'Hidden regions are plausible estimates' in archive.read('README.txt').decode()
            if applied:
                raw=np.load(io.BytesIO(archive.read('dgp-raw-float32.npy')),allow_pickle=False)
                assert raw.dtype==np.float32 and raw.shape==(256,256,3) and np.isfinite(raw).all()
                assert 0<=raw.min()<=raw.max()<=1 and meta['raw_dgp']['sha256']==array_sha(raw)
                expected=np.floor(raw*np.float32(255)).astype(np.uint8);expected[~observed]=prepared[~observed]
                assert np.array_equal(expected,output)
                assert meta['visible_restoration']['weights_sha256']=='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
            else:
                assert meta['raw_dgp'] is None and meta['visible_restoration'] is None
                assert np.array_equal(output,prepared)
        if mode=='auto':assert meta['quality_signals']['suggest_restoration'] is False
        cases.append(dict(mode=mode,response_sha256=sha(response_path),output_png_sha256=hashlib.sha256(output_bytes).hexdigest(),
            bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_members=sorted(names),
            native_original_dimensions_retained=[138,165],metadata_geometry_and_mask_hashes_checked=True,
            observed_RGB_exact_when_Off_or_Auto=not applied,restoration_applied=applied,
            independent_raw_to_PNG_floor_and_padding_checked=applied,actual_downloads_checked=2))
    assert np.array_equal(outputs['off'],outputs['auto'])
    guards=read(ART/'negative_api_audit.json')
    assert guards['complete'] and guards['requests']==len(guards['guards'])==10
    for row in guards['guards']:
        saved=read(ART/('guard-'+row['label']+'.json'))
        assert saved['status']==row['status']==422 and row['generated_output'] is False
        assert saved['response']['detail']==row['detail']
        assert saved['expected_detail_substring'] in row['detail']
        assert not set(saved['response'])-{'detail'}
    initial=read(ART/'initial_browser_audit.json');ui=read(ART/'native_browser_audit.json');auto=read(ART/'native_auto_browser_audit.json')
    assert all(r['complete'] and not r['errors'] for r in [initial,ui,auto])
    assert all(not r['overflow'] for r in initial['layout']+ui['result_layout']+auto['result_layout'])
    assert len(ui['generations'])==2 and len(ui['result_layout'])+len(auto['result_layout'])==9
    tree=ast.parse((ROOT/'dgp_face_workflow_v3.py').read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DGPFaceWorkflow')
    generate=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    start=next(n for n in generate.body if isinstance(n,ast.With)).lineno
    messages=[n for n in ast.walk(generate) if isinstance(n,ast.Raise) and n.lineno<start]
    assert len(messages)==5
    review=next(n for n in ast.walk(generate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='review_input')
    assert review.lineno<start
    write_data=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),cases=cases,
        preflight_sha256=sha(ART/'preflight.json'),negative_API_audit_sha256=sha(ART/'negative_api_audit.json'),
        native_browser_audit_sha256=sha(ART/'native_browser_audit.json'),native_auto_browser_audit_sha256=sha(ART/'native_auto_browser_audit.json'),
        original_frozen_input_review_sha256=sha(input_review_path),native_contract_sha256=sha(contract),
        canonical_native_source_and_alias_encoded_hashes_match=True,source_model_and_fixture_hashes_unchanged=True,
        native_unpaired=True,clean_reference_absent=True,synthetic_PSNR_or_SSIM_not_reported=True,
        one_native_DEV_case_not_three_independent_subjects=True,actual_downloads_checked=6,
        negative_guard_responses_checked=10,guard_raises_precede_generation_inference_block=True,
        initial_layout_checks=3,result_layout_checks=9,page_and_console_errors=[],
        all_native_padding_and_masks_recomputed=True,Auto_output_identical_to_Off=True,
        new_model_gradient_optimizer_or_training_calls_in_this_saved_output_audit=0,
        visual_review_pending=True,all_seven_families_qualified=False,independent_final_review_complete=False,model_qualified=False,goal_complete=False)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(write_data,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,native_cases=1,generation_modes=3,actual_downloads=6,negative_guards=10,result_layout_checks=9))


if __name__=='__main__':main()
