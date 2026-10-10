"""Preserve handoff bytes and record a manual, inference-only diagnostic packet."""
from datetime import datetime,timezone
from pathlib import Path
import time
from cctv_dgp_actual_step_review_v1_contract import NAME,STEM,read,write,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_prepared_milestone'
PRIOR=ROOT/'outputs/cctv_dgp_v41_preservation_trace_milestone'


def main():
    start=time.monotonic();assert not OUT.exists()
    prior=read(PRIOR/'milestone.json');closure=read(PRIOR/'independent_closure_audit.json')
    assert prior['complete'] and closure['complete'] and closure['milestone_sha256']==sha(PRIOR/'milestone.json')
    assert sha(PRIOR/'milestone.json')=='04b0752d2b4a6bf32c2105bc24be0d47e2104993231e9d507c320d8465514be1'
    for name,digest in prior['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    prep=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_packet_preparation.json')
    packet=read(ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation/independent_packet_audit.json')
    assert prep['complete'] and packet['complete'] and prep['protocol_sha256']==packet['protocol_sha256']
    assert packet['VM_calls']==packet['neural_calls']==packet['optimizer_updates']==packet['gradient_queries']==0
    assert not packet['training_started'] and not packet['app_promotion']
    p=read(ROOT/'outputs'/NAME/'protocol.json');assert sha(ROOT/'outputs'/NAME/'protocol.json')==prep['protocol_sha256']
    audit=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json')
    assert audit['complete'] and audit['failure_retained'] and not audit['necessary_capacity_pass']
    stopped=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return/outputs/stopped_spatial_decoder.pth'
    imported=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json')
    assert sha(stopped)==imported['files_sha256']['outputs/stopped_spatial_decoder.pth']
    OUT.mkdir();(OUT/'before_docs').mkdir()
    before=(ROOT/'PROJECT_HANDOFF.md').read_bytes();assert sha(ROOT/'PROJECT_HANDOFF.md')=='6de582b3e3972b0097bc2c1faf110a4de24a90bf84b3d0f05593a392da1b1293'
    (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section=f'''
**Research milestone — 9 October 2026: finite actual-step diagnostic prepared; manual VM execution pending.**

V41 remains failed at50/800: structure gain0.0516710943% against the unchanged1%
requirement and one delivered FFHQ compound preservation failure. Its archive,
checkpoint, full audit, source code, splits and failed gates remain retained.
The earlier "apply the best approach" preference and completed applied-step
architecture review select a separate finite diagnostic before another training
recipe. The historical pending question is superseded for this diagnostic
preparation only; original training code remains unchanged.

The separate packet compares ten recorded V41 before-states and three fixed
parameter proposals: unchanged, recorded actual and one saved-array cone step.
Both preselected50-case TRAIN cohorts remain exact; each current five-profile
batch is included. This yields145 unique cases/29 references,30 conditions and
3,150 finite forward slots. Probe states are selected after TRAIN outcomes and
remain diagnostic. Every slot retains raw floats, delivered PNG and a mean-only
control; raw and delivered objectives/preservation are reported separately.
Float32 proposal rounding is retained, including seven proposals with tiny
positive rounded local constraint derivatives. No finite improvement is assumed.

The independently verified261-member transfer archive is197,588,115bytes.
Protocol SHA256: {prep['protocol_sha256']}
Execution archive SHA256: {prep['archive_sha256']}
All25 corruption/archive/scope/decision regressions pass. Both Windows worker
and supervisor stop before Torch imports. All100 archived-float fixed-filter
arithmetic checks pass without model construction, gradients or parameter
assignment. All Python sources parse for3.10. A read-only Bash syntax check
passes outside the sandbox after retaining its Windows signal-pipe failure.
Two earlier metadata-only preparation failures and their153-file partial copy
remain preserved; their fix distinguishes provenance paths from runtime assets.
Original metadata, labels and scientific inputs are unchanged.

Manual execution remains on the existing L4/g2-standard-4 at ~/forensic-dgp.
Require6GiB free after installation; estimated10–25minutes plus export.
Cache300s, review1,500s, worker1,800s/external1,830s, export900s/external930s,
20GiB allocated VRAM and3GiB encoded returns are enforced. First315-slot timing
must project within1,500s with a1.25 safety factor. Every failure is retained;
no automatic follow-on, historical rerun or cleanup occurs.
The diagnostic has zero optimizer updates, zero gradient queries and no trained
checkpoint. No VM connection or new neural forward occurs during preparation.
The100 local arithmetic checks use fixed filters, not a neural model.

The prospective independent checker is source-bound before the run. It checks
all3,150 raw/PNG/mean-only slots and150 frozen CPU replays under a2,400s bound.
Numerical aggregate allowances preserve exact categorical decisions and do not
change delivered-image gates. Partial failures receive an import-only receipt,
never a full scientific-output audit. No returned code is executed. This packet
is prepared and unrun; it does not establish full TRAIN capacity or qualify a
restoration model.

All14 DGP-primary app bindings remain exact. No frontend or trained-model change
requires another app-flow test here. The earlier functional Playwright evidence
does not qualify restoration/covering quality. Research caches/local backup,
provenance, original checkpoints and all failure records remain retained.
No new native/DEV/reserved-final images enter this packet. Unpaired CCTV stays
separate from paired photographic metrics; source labels do not establish
ethnicity or Zamboanga performance. Clear glasses, ordinary hair and every
visible facial feature remain protected. Useful native restoration, separate
automatic/assisted review for all seven covering families, independent final
review and qualified app flow remain outstanding. Goal active/incomplete.
The complete earlier handoff is archived and preserved below.

[Five exact manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_ACTUAL_STEP_REVIEW_V1_VM.md>)
[Independent packet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_actual_step_review_v1_preparation/independent_packet_audit.json>)

'''
    addition=section.encode('utf-8');(OUT/'new_handoff_section.md').write_bytes(addition);at=before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    names=set(p['source_evidence_sha256'])
    for folder in [ROOT/'outputs'/NAME,ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation',
                   ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation_failure_r1',ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation_failure_r2']:
        names.update(path.relative_to(ROOT).as_posix() for path in folder.rglob('*') if path.is_file())
    names.update(['PROJECT_HANDOFF.md','CCTV_DGP_ACTUAL_STEP_REVIEW_V1_VM.md',
       'outputs/cctv_dgp_actual_step_review_v1_inputs_preparation.json','outputs/cctv_dgp_actual_step_review_v1_packet_preparation.json',
       'outputs/'+STEM+'-execution.tar.gz','outputs/'+STEM+'-execution.tar.gz.sha256',
       'scripts/record_cctv_dgp_actual_step_review_v1_prepared_milestone.py','scripts/verify_cctv_dgp_actual_step_review_v1_prepared_milestone.py',
       'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json',stopped.relative_to(ROOT).as_posix(),
       (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),(OUT/'new_handoff_section.md').relative_to(ROOT).as_posix(),
       (PRIOR/'milestone.json').relative_to(ROOT).as_posix(),(PRIOR/'independent_closure_audit.json').relative_to(ROOT).as_posix()])
    bindings={name:sha(ROOT/name) for name in sorted(names)}
    app=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_audit_recovery_r1/app_preservation_readback.json')['files_sha256']
    for name,digest in app.items():assert sha(ROOT/name)==digest,name
    assert not (ROOT/'outputs'/NAME/'outputs').exists() and not (ROOT/'outputs'/(STEM+'-results.tar.gz')).exists()
    assert time.monotonic()-start<300
    write(OUT/'milestone.json',{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
       'previous_milestone_sha256':sha(PRIOR/'milestone.json'),'previous_closure_sha256':sha(PRIOR/'independent_closure_audit.json'),
       'previous_handoff_path':(OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
       'new_evidence_sha256':bindings,'app_bindings_sha256':app,'protocol_sha256':prep['protocol_sha256'],'archive_sha256':prep['archive_sha256'],
       'prepared_only':True,'diagnostic_cases':145,'diagnostic_references':29,'proposal_conditions':30,'review_forward_slots':3150,
       'earlier_best_approach_authorization_used':True,'original_V41_failed_gate_retained':True,
       'original_metadata_and_checkpoints_and_splits_and_failures_retained':True,'new_training_recipe_prepared':False,'historical_training_code_modified':False,
       'new_neural_model_forwards':0,'fixed_filter_arithmetic_cases':100,'gradient_queries':0,'optimizer_updates':0,'VM_calls':0,
       'new_trained_checkpoint':False,'app_promotion':False,'native_or_DEV_or_reserved_final_used':False,'independent_final_review':False,
       'goal_complete':False,'seconds':time.monotonic()-start,'cap_seconds':300})
    print({'complete':True,'bindings':len(bindings),'prepared_only':True,'VM_calls':0,'goal_complete':False,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
