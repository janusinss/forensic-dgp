"""Record audited human diagnostic and prepared manual V41 without promotion."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import time
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v40_signal_return_v41_prepared_milestone'
PRIOR=ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm_availability_milestone'


def main():
    start=time.monotonic();assert not OUT.exists()
    parent=read(PRIOR/'milestone.json');closed=read(PRIOR/'independent_closure_audit.json')
    assert closed['complete'] and closed['milestone_sha256']==sha(PRIOR/'milestone.json')
    for name,digest in parent['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    audit=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json')
    checked=read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis/independent_analysis_audit.json')
    packet=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_preparation/independent_packet_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and audit['members_verified']==957
    assert checked['complete'] and checked['saved_batches']==40 and checked['exact_unscaled_sheet_cells']==200
    assert packet['complete'] and packet['prepared_only'] and not packet['training_capacity_pass']
    assert not (ROOT/'outputs/cctv-dgp-pcgrad-fit-v41-results.tar.gz').exists()
    OUT.mkdir();(OUT/'before_docs').mkdir()
    before=(ROOT/'PROJECT_HANDOFF.md').read_bytes();(OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section='''
**Research milestone — 8 October 2026: human V40 endpoint diagnostic audited; V41 PCGrad pilot prepared.**

The human returned the learning-signal-v1 archive,303,352,790bytes, SHA256
002cd3765d17644b993646a0320c95d72de31e318b62285c93b1bcc7ba478a10.
The unchanged prospective audit passes in94.566s:957 exact files,280 saved
component-gradient queries,200 raw/300 PNG compositions and40 frozen CPU
replay cases. The VM diagnostic completes in75.072s with zero optimizer or
parameter updates/backwards/epochs and no new trained checkpoint. Historical
raw parity is exact and all four model states remain bound. The earlier
stopped-VM observation predates this human run; no fresh guest-space/status
claim or agent VM execution is made here.

Stopped50 identity-preservation gradient norms are5.328730 and7.522134 times
landmark norms in the two exposed TRAIN cohorts. Raw landmark loss improves
only0.009353% and0.019944%; V40's full-TRAIN delivered gain remains0.0106079234%,
below the unchanged1% early requirement. The preview-cohort summed negative
gradient increases observed-detail loss. These are endpoint observations,
not a reconstructed AdamW trajectory or unique cause. All50 newly selected
optimized faces/200 exact cells are visually reviewed on10 unscaled sheets:
no convincing incremental definition over the original DGP. Source names do
not establish ethnicity; these paired photographic TRAIN cases are not CCTV
or independent evaluation. No input-only labels or split are changed.

After primary-source PCGrad research, one fixed saved-array projection rule
yields nonincreasing first-order component directions in all40 batches. The
independent arithmetic also verifies four aggregate sets and200 exact sheet
cells. This motivates V41, not finite-step preservation or model qualification.
The distinct PCGrad treatment combines the same seven weighted gradients;
architecture, initial seed,3,905 TRAIN cases/781 references,800 paired batches,
AdamW, composition and all numerical gates stay fixed. Per-update gradients,
parameter endpoints, scalar losses and AdamW moments are retained for audit.
No old checkpoint or failed gate is overwritten, resumed or automatically run.

V41 packet verification passes:5,493 files/444,567,499bytes; seven corrupted
step records, nine unsafe archives, three wrong scopes, Windows rejection
before neural imports, fixed800 orders, Python3.10 syntax and read-only Bash.
Protocol:46dbeab9515719fdd5571cdbfdf5e52e6673bc78e44ccff84bd2f3f15e69c56e
Archive:0f1cb2e8b6d11dade6fb0483b4ba1f285ed2c926bc0bf9f0442763b73bae0d53
This is prepared only. Use the five verified single-source gcloud upload,
install,tmux,launch and download steps manually on the existing L4/g2-standard-4
at ~/forensic-dgp. Need6GiB free after installation. Maximum800 updates, unchanged
1%-at50/10%-at800 and all17 preservation/source/mean-only gates remain. Cache900s,
fit3600s, worker4500s/external4800s+30s, export900s/external930s+30s,20GiB allocated
VRAM and3GiB uncompressed return stops are enforced. No training starts here.

The original local audit's wrong-Python/PyTorch absence and full import remain;
the same checker passes in the research venv. The separate analysis's exact
derived-ratio failure (8.881784197001252e-16) and source remain; distinct R1 allows
rtol2e-12/atol0 only for that ratio, without changing arrays or decisions. New
prospective V41 arithmetic allowances do not weaken any delivered-image gate.

All14 DGP-primary app bindings, original/stopped checkpoints, research caches,
local backup, provenance, splits and previous failures remain. V38 development
preservation and completion-quality failures remain binding. Native CCTV remains
unpaired and synthetic PSNR/SSIM separate; reserved45 identities/58 crops stay
unopened. No Zamboanga CCTV evidence or hidden-identity claim exists. Useful
native restoration, all seven covering families with separate automatic/assisted
quality, independent final review and qualified full app flow remain outstanding.
The entire previous handoff is archived and preserved below. Goal active/incomplete.

[Audited diagnostic findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_RESULTS.md>)
[V41 five manual commands](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PCGRAD_FIT_V41_VM.md>)

'''
    addition=section.encode('utf-8');at=before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    names=['PROJECT_HANDOFF.md','CCTV_DGP_V40_LEARNING_SIGNAL_V1_RESULTS.md','CCTV_DGP_PCGRAD_FIT_V41_VM.md',
        'scripts/analyze_cctv_dgp_v40_learning_signal_v1.py','scripts/verify_cctv_dgp_v40_learning_signal_v1_analysis.py',
        'scripts/verify_cctv_dgp_v40_learning_signal_v1_analysis_r1.py','scripts/record_cctv_dgp_v40_learning_signal_v1_visual_review.py',
        'scripts/cctv_dgp_pcgrad_v41.py','scripts/cctv_dgp_pcgrad_v41_evidence.py','scripts/render_cctv_dgp_pcgrad_v41.py',
        'scripts/prepare_cctv_dgp_pcgrad_fit_v41.py','scripts/verify_cctv_dgp_pcgrad_fit_v41_packet.py',
        'scripts/cctv_dgp_pcgrad_fit_v41_vm.py','scripts/audit_cctv_dgp_pcgrad_fit_v41_return.py',Path(__file__).relative_to(ROOT).as_posix(),
        'scripts/verify_cctv_dgp_v40_signal_return_v41_prepared_milestone.py']
    roots=['outputs/cctv_dgp_v40_learning_signal_v1_return','outputs/cctv_dgp_v40_learning_signal_v1_return_environment_failure',
        'outputs/cctv_dgp_v40_learning_signal_v1_analysis','outputs/cctv_dgp_pcgrad_fit_vm_v41','outputs/cctv_dgp_pcgrad_fit_v41_preparation']
    for root in roots:names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/root).rglob('*') if p.is_file())
    names.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/'outputs').iterdir() if p.is_file() and
        (p.name.startswith('cctv-dgp-v40-learning-signal-v1-') or p.name.startswith('cctv_dgp_v40_learning_signal_v1_return_') or
         p.name in ['cctv_dgp_v40_learning_signal_v1_independent_audit.json','cctv-dgp-pcgrad-fit-v41-execution.tar.gz',
                    'cctv-dgp-pcgrad-fit-v41-execution.tar.gz.sha256','cctv_dgp_pcgrad_fit_v41_preparation_execution.log','cctv_dgp_pcgrad_fit_v41_packet_audit_execution.log']))
    names.extend([parent['previous_handoff_path'],(OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
                  (PRIOR/'milestone.json').relative_to(ROOT).as_posix(),(PRIOR/'independent_closure_audit.json').relative_to(ROOT).as_posix()])
    bindings={name:sha(ROOT/name) for name in set(names)};assert time.monotonic()-start<600
    write(OUT/'milestone.json',{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'previous_milestone_sha256':sha(PRIOR/'milestone.json'),
        'previous_handoff_path':(OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
        'document':{'before_sha256':hashlib.sha256(before).hexdigest(),'addition_bytes':len(addition)},'new_evidence_sha256':bindings,
        'human_VM_diagnostic_audited':True,'saved_gradient_queries':280,'diagnostic_optimizer_updates':0,
        'V41_prepared_only':True,'V41_training_started':False,'V40_gate_failure_preserved':True,
        'DGP_primary_app_bindings_unchanged':True,'VM_calls_here':0,'local_gradient_calls':0,'local_optimizer_updates':0,
        'app_promotion':False,'independent_final_review':False,'goal_complete':False,'seconds':time.monotonic()-start,'cap_seconds':600})
    print({'complete':True,'new_bindings':len(bindings),'V41_prepared_only':True,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
