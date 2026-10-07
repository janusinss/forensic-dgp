"""Preserve history, record the reviewed V29 limit and one manual coverage pilot."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_v29_development_and_broader_mean_v30_milestone'
PREVIOUS=ROOT/'outputs/dgp_v28_preservation_diagnostic_and_mean_centered_v29_milestone/milestone.json'
DOCS=('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(bindings,mapping=None):
    for name,digest in bindings.items():assert sha(ROOT/(mapping or {}).get(name,name))==digest,name
    return len(bindings)


def main():
    started=time.monotonic();assert not OUT.exists()
    assert sha(PREVIOUS)=='4f81d4a7dca45ab0e8e5c86b6a6a61b63c771531b9acb363f2011078eec0e7be'
    old=read(PREVIOUS);assert verify(old['new_evidence_sha256'])==371
    assert read(PREVIOUS.with_name('independent_readback.json'))['complete']
    audit=read(ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json')
    assert audit['complete'] and audit['training_completed800'] and audit['necessary_capacity_pass']
    paired=read(ROOT/'outputs/cctv_dgp_v29_paired_development_v1/results.json')
    assert len(paired['rows'])==520 and len(paired['diagnostic_preservation_failures'])==21
    assert read(ROOT/'outputs/cctv_dgp_v29_paired_development_v1/saved_output_audit.json')['complete']
    assert read(ROOT/'outputs/cctv_dgp_v29_paired_development_v1/visual_review.json')['preview_faces_reviewed']==50
    native=read(ROOT/'outputs/cctv_dgp_v29_native_development_v1/visual_review.json')
    assert native['cases_reviewed']==24 and not native['restoration_qualified']
    prep=ROOT/'outputs/cctv_dgp_broader_mean_v30_preparation';packet=read(prep/'independent_packet_audit.json')
    assert packet['complete'] and packet['return_boundary_regressions_passed']==7 and packet['five_manual_steps_verified']
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';app=read(app_path)
    assert sha(app_path)==old['app_record_sha256'] and verify({**app['sources_sha256'],**app['evidence_sha256']})==22
    OUT.mkdir();before=OUT/'before_docs';before.mkdir();mapping={}
    prefix='''**Latest research milestone — 7 October 2026: V29 TRAIN capacity audited;520 paired DEV cases and24 native faces reviewed; broader-coverage V30 prepared for manual VM execution.**

The downloaded V29 return matches SHA256
ecf88626b31dee6b0ceb65f6a2a76e534cea8079183e0efd143c6bcb16fb5ae2,
294,561,124 bytes. The original frozen independent checker passes837 files,
38,394,279 gradient values, all four checkpoints and500 forward-only CPU calls.
All800 VM updates complete. Final50-case TRAIN structure gain20.0551%, all17
preservation groups and the original brightness gate pass. All50 TRAIN faces
are viewed. Original checkpoint, encoder/head4, buffers, splits and failures
remain. TRAIN capacity is necessary evidence and does not qualify restoration.

The separately frozen single-crop inference path passes seven contract checks
and50 CPU/VM parity cases: raw error<=2.355e-6, PNG<=one byte, exact padding.
It requires original baseline plus candidate plus the fixed mean projection.
The existing app model/design remains unchanged; this is a benchmark candidate.

All520 paired photographic DEV cases are retained and independently audited.
Aggregate degraded landmark structure gain is1.3516%; asian_faces source
worsens3.4051%, thumbnails128x128 improves2.3437%. There are21 fixed group/metric
regressions,17 in ArcFace. ArcFace is a preservation diagnostic, not identity
accuracy. All50 fixed DEV preview faces are actually viewed: sharper edges
can coexist with changed eyes, mouth or expression. Reject V29 app promotion.
Paired synthetic MSE/PSNR/SSIM do not become native CCTV metrics.

All24 frozen ChokePoint C1 native development faces are independently audited
and actually reviewed against resizing, retained Phase3, original DGP and
declared CodeFormer on identical256 inputs. V29 retains broad face arrangement
but lacks convincing useful clarity over resize. The previously useful02_t033
still needs clearer eyes/nose/lips. Usable inputs are not reclassified from
model softness. Source capture country is unspecified in acquired metadata;
no ethnicity or Zamboanga performance is inferred. No final reserved pixels
are opened. The separate58 cases /45 namespaced final identities remain reserved.

V30 changes optimization coverage from10 to the already approved781 TRAIN
references /3905 existing cases, retaining800 updates /4000 sample exposures.
The first781 batches cover every case once;19 predetermined second-epoch
batches follow. Original initialization, same mean-centered decoder path,
selected12 tensors, seven losses, fixed50 normalizers, AdamW and numeric gates
stay unchanged. This tests the coverage hypothesis, not a proven unique cause.
No V9 trained weights, V29 continuation or unchanged historical pilot is run.
The104 DEV references remain outside optimization; person/pretrained overlap
remains unknown. Every original checkpoint and failed gate remains preserved.

The eight-file746,356-byte packet uploads no images or weights. Protocol:
b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1.
Archive:7a3d58db1e6d14a1c5a589269ec6922cd8fbe8d0f1834f4c242fb1d2cf72ac39.
Independent metadata/packet/source checks pass5467 TRAIN files, exact50-case
source parity, Python3.10 parsing, Windows pre-neural training rejection,
seven return-boundary regressions, readonly Bash syntax and five manual steps.
The original local Bash checker failed because the process sandbox blocked
its signal pipe; the source and failure remain. The distinct R1 checker uses
the separately observed readonly external Bash check. No packet, loss or gate
changed. This is a local setup repair, not a VM/model success.

Actual V30 training has not started. Human gcloud upload/SSH/tmux is required.
Require idle existing L4/g2-standard-4 and6GiB free, without deleting research
assets. Initial proof300s /70 queries, cache900s, fit3600s, worker4500s,
external4800s+30s grace, export900s/external930s+30s grace, allocatedVRAM<=20GiB.
Snapshots0/50/400/800 include all3905 TRAIN outputs and mean controls; only50
raw previews are exported. Early50>=1% structure, final800>=10%, both sources
nonnegative, all17 groups and brightness<=20% remain. Final800 only, no resume,
overwrite, unchanged retry, competing-task termination or automatic promotion.
The prospective independent return checker checks all saved outputs/gates,
the initial proof and frozen partitions, every final3905 case and checkpoint
previews through CPU inference only, with7200s audit cap.

All prior371 bindings and1041/586/50/668/cleanup60/309/66/692/299/697/513 history,
three complete document bodies and app22 bindings are preserved. All seven
covering-family automatic/assisted requirements, useful native restoration,
full app flow and independent final review remain unfinished. Goal active/incomplete.

[V29 results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_MEAN_CENTERED_DECODER_V29_RESULTS.md>) ·
[V30 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BROADER_MEAN_V30_VM.md>)

Earlier bodies below are preserved history. V29 pending/not-started language is
superseded by its audited return and failed development qualification. Only
the distinct V30 coverage pilot is ready for manual execution; no failed
historical pilot is automatically rerun.

'''
    for name in DOCS:
        original=(ROOT/name).read_bytes();(before/name).write_bytes(original)
        mapping[name]=(before/name).relative_to(ROOT).as_posix();split=original.index(b'\n')+1
        (ROOT/name).write_bytes(original[:split]+b'\n'+prefix.encode('utf-8')+original[split:])
    assert verify(old['new_evidence_sha256'],mapping)==371
    evidence={}
    def bind(path):evidence[path.relative_to(ROOT).as_posix()]=sha(path)
    names=[*DOCS,'CCTV_DGP_MEAN_CENTERED_DECODER_V29_RESULTS.md','CCTV_DGP_BROADER_MEAN_V30_VM.md',
           'dgp_mean_centered_inference_v29.py','tests/test_dgp_mean_centered_inference_v29.py',
           'tests/test_cctv_dgp_broader_mean_v30_return.py','scripts/prepare_cctv_dgp_mean_centered_decoder_v29_visual_review.py',
           'scripts/record_cctv_dgp_mean_centered_decoder_v29_visual_review.py','scripts/verify_cctv_dgp_v29_single_input_parity.py',
           'scripts/run_cctv_dgp_v29_native_development_v1.py','scripts/verify_cctv_dgp_v29_native_development_v1.py',
           'scripts/record_cctv_dgp_v29_native_development_review.py','scripts/run_cctv_dgp_v29_paired_development_v1.py',
           'scripts/verify_cctv_dgp_v29_paired_development_v1.py','scripts/record_cctv_dgp_v29_paired_development_review.py',
           'scripts/cctv_dgp_broader_mean_v30_cache.py','scripts/prepare_cctv_dgp_broader_mean_v30.py',
           'scripts/audit_cctv_dgp_broader_mean_v30_return.py','scripts/verify_cctv_dgp_broader_mean_v30_packet.py',
           'scripts/verify_cctv_dgp_broader_mean_v30_packet_r1.py',
           'scripts/record_cctv_dgp_v29_development_and_broader_mean_v30_milestone.py',
           'scripts/verify_cctv_dgp_v29_development_and_broader_mean_v30_milestone.py',
           'outputs/cctv-dgp-mean-centered-decoder-v29-results.tar.gz',
           'outputs/cctv-dgp-mean-centered-decoder-v29-results.tar.gz.sha256','outputs/cctv-dgp-mean-centered-decoder-v29-export.json',
           'outputs/cctv_dgp_mean_centered_decoder_v29_return_import.json','outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json',
           'outputs/cctv-dgp-broader-mean-v30-execution.tar.gz','outputs/cctv-dgp-broader-mean-v30-execution.tar.gz.sha256',
           PREVIOUS.relative_to(ROOT).as_posix(),PREVIOUS.with_name('independent_readback.json').relative_to(ROOT).as_posix()]
    for name in names:bind(ROOT/name)
    for directory in [ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_return',
                      ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_visual_review',
                      ROOT/'outputs/cctv_dgp_v29_single_input_parity_v1',ROOT/'outputs/cctv_dgp_v29_native_development_v1',
                      ROOT/'outputs/cctv_dgp_v29_paired_development_v1',ROOT/'outputs/cctv_dgp_broader_mean_vm_v30',
                      prep,ROOT/'outputs/cctv_dgp_broader_mean_v30_packet_audit_setup_failure',before]:
        for path in directory.rglob('*'):
            if path.is_file():bind(path)
    value={'complete':True,'date':'2026-10-07','scope':'V29 capacity passed but development qualification rejected; one broader-coverage pilot prepared',
           'new_evidence_sha256':evidence,'previous_milestone_sha256':sha(PREVIOUS),'previous371_original_locations':mapping,
           'full_previous_document_bodies_preserved':3,'app_record_sha256':sha(app_path),'app22_bindings_preserved':True,
           'V29_return_archive_sha256':audit['archive_sha256'],'V29_capacity_pass':True,'V29_TRAIN_faces_reviewed':50,
           'V29_paired_DEV_cases_audited':520,'V29_DEV_preview_faces_reviewed':50,'V29_DEV_preservation_failures':21,
           'V29_native_faces_audited_and_reviewed':24,'V29_development_restoration_qualified':False,
           'V30_protocol_sha256':packet['protocol_sha256'],'V30_archive_sha256':packet['archive_sha256'],'V30_archive_bytes':746356,
           'V30_TRAIN_assets_verified':5467,'V30_training_references':781,'V30_training_cases':3905,'V30_updates_bound':800,
           'V30_optimizer_losses_architecture_initialization_and_gates_unchanged':True,'V30_return_regressions_passed':7,
           'original_Bash_setup_failure_retained':True,'V30_distinct_R1_packet_audit_pass':True,
           'V30_actual_VM_training_started':False,'human_manual_VM_execution_required':True,'all_old_failed_gates_preserved':True,
           'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'VM_actions':False,
           'reserved_final_used':False,'independent_final_review_complete':False,'app_promotion':False,
           'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-started}
    with (OUT/'milestone.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:value[k] for k in ['complete','V29_capacity_pass','V29_DEV_preservation_failures','V30_training_cases','V30_actual_VM_training_started','goal_status','seconds']},indent=2))


if __name__=='__main__':main()
