"""Prospective interface/resource clarifications; no scientific settings change."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile,os

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design'
def need(ok,label):
    if not ok:raise ValueError(label)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def main():
    path=DEST/'pre_release_clarifications_v1.json';need(not path.exists(),'Preserve earlier clarification')
    design=DEST/'root_design.json';need(sha(design)=='1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657','Frozen design exact')
    d=json.loads(design.read_text())
    original=Path('C:/xampp/htdocs/YEAR 4/Testing/checkpoints/codeformer_inpainting.pth')
    need(sha(original)=='b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797','Original full-model file')
    with zipfile.ZipFile(original) as z:
        entries=z.infolist();need(len({r.filename for r in entries})==len(entries) and len(entries)<1615,'Bounded original tensor ZIP')
        storage=[r for r in entries if '/data/' in r.filename and not r.is_dir()]
        total=sum(r.file_size for r in storage)
    # Failure state additionally retains bounded RNG/container metadata, no Adam.
    failure_bound=total+1024**2+1615*1024+10240
    need(failure_bound<384*1024**2,'Actual497 model plus1MiB pickle/headers/RNG reserve fits384MiB')
    resource=d['resource_arithmetic'];need(resource['all104_numeric_data_plus64KiB_each']<560*1024**2,'Exact declared arrays bound')
    record={
      'format':'completion-spatial-code-intervention-v1-prospective-interface-clarifications-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'root_design':pin(design),'author':pin(Path(__file__)),
      'reason':'Resolve implementation interfaces before any source/packet release. Preserve all frozen scientific choices, limits and fixed48/bridgecase membership.',
      'fresh_C_bridge_NPZ':{'same_core_schema_as_HV':True,'input512_saved':False,'arrays':['input256','mask256','result256','raw512','predicted_codes','used_codes','grid_mask','arm_code','oracle_unavailable_clean_information'],'maximum_fields':10,'single_file_cap_bytes':6*1024**2,'explanation':'HistoricalC has no input512 array. Fourordinaryapp parity bundles already retain both512conditioning inputs. The bridge compares matching input256/mask/raw512/result256/predicted/used and in-memoryPNG, without redundantadditional512array.'},
      'grid_selector':'Boolean[1,256,1] fixed from exact256support; clean/predictedlong[1,256,1]. H=where(touched,clean,predicted); V=where(touched,predicted,clean). No learned mask, threshold search, dilation or output-support change.',
      'actual_forward_witness':{'reason':'The owned forward function bypasses net.__call__; quantizer.get_codebook_feat is a direct bound method. A top-levelModule hook alone would not witness these decodes.',
          'method':'Instance-only pass-through wrapper increments quantizer lookup count and calls the exact original bound method with unchangedargs/results; firstgeneratorblock pre-hook and lastgeneratorblock post-hook count starts/completions. All hooks/wrappers restorefinally.',
          'exact_completion_counts':{'typed_dispatch':108,'quantizer_lookup':108,'generator_start':108,'generator_complete':108},'ordinary_app_calls_included':True,'no_global_class_or_model_source_patch':True,'scientific_numerics_changed':False},
      'parity_and_control_truth':{'four_ordinary_pairs':8,'four_fresh_C_bridge_calls':4,'H_cases':48,'V_cases':48,'total_new_full_generation_calls':108,'historical196_A_C_control_files_reused':True,'new_48_A_or_48_full_C_call_claim_forbidden':True},
      'labels_and_return_truth':{'label_input_NPZ_sets':16,'copied_label_NPZs_in_new_return':0,'historical_A_C_NPZs_copied_to_return':0,'new_HV_NPZ':96,'new_parity_NPZ':4,'new_fresh_C_bridge_NPZ':4,'all_new_return_NPZ':104,'new_PNG_only_HV':96,'historical_inputs_retained_exact_at_source':True},
      'failure_state_safety':{'normal_unchanged_checkpoint_saved':False,'unexpected_mutation_nonfinite_capture_actual497':True,'state_capture_attempts_max':1,'actual_RNG_progress_counters_retained':True,'optimizer_objects_or_state':0,'failure384MiB_single_payload_cap':True,'partial_retained':True,
          'zip_member_cap':1615,'pickle_bytes_cap':1024**2,'pre_write_byte_guard':True,'closed_partial_internal_ZIP_CRC_before_commit':True,'safe_hash_finite_receipts_do_not_substitute_original_state':True,'same_inode_alias_export_once_only_if_exact_dev_inode_uid_size_hash_nlink_ge2':True,'scientific_failed_guard_does_not_prevent_resource_bounded_capture':True,'IO_hardkill_can_prevent_recovery':True,'automatic_retry_resume':False},
      'passive_original_ZIP_resource_evidence':{'file':pin(original),'members':len(entries),'tensor_storage_members':len(storage),'tensor_storage_bytes':total,'full_failure_worst_bound_with_metadata_headers':failure_bound,'cap':384*1024**2,'no_checkpoint_load':True},
      'resource_bounds_unchanged':resource,'all_scientific_decisions_unchanged':True,'source_and_transfer_independent_review_still_required':True,'actual_neural_calls':0,'actual_optimizer_updates':0,'VM_calls':0,'runtime_ready':False,'model_qualified':False,
    }
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(pin(path)))
if __name__=='__main__':main()
