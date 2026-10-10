"""Preserve old handoff bytes, then record audited return, cleanup and next stage."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_capacity_v1_milestone'
sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1'))
from cctv_dgp_head4_capacity_contract_v1 import read,write,sha


def main():
    assert not OUT.exists();packet=ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1';p=read(packet/'protocol.json')
    prep=ROOT/'outputs/cctv_dgp_head4_capacity_v1_preparation';audit=read(prep/'independent_packet_audit.json')
    assert audit['complete'] and audit['VM_launched'] is False and audit['optimizer_updates']==0
    gradient=read(ROOT/'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json')
    assert gradient['complete'] and gradient['connected_route_pass'] and gradient['optimizer_updates']==0
    cleanup=read(ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1/independent_audit.json')
    live=read(ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1/post_apply_inventory.json')
    assert cleanup['complete'] and live['complete'] and live['GPU_compute_idle'] and live['target_archives_remaining']==0
    assert not live['training_launched'] and live['free_bytes']>=14*1024**3
    commands=read(prep/'current_manual_commands_audit.json');assert commands['complete']
    assert sha(ROOT/'CCTV_DGP_HEAD4_CAPACITY_V1_VM_CURRENT.md')==commands['current_guide_sha256']
    original={'current':'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
        'Phase3':'checkpoints/dgp_zamboanga_final.pth','immutable_training_plan':'CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md'}
    protected={k:{'path':v,'sha256':sha(ROOT/v)} for k,v in original.items()}
    assert protected['current']['sha256']==p['original_checkpoint_sha256']
    assert protected['Phase3']['sha256']=='b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c'
    for n,d in p['assets_sha256'].items():assert sha(packet/n)==d,n
    for n,d in p['local_sources'].items():assert sha(ROOT/n)==d,n
    for n,d in p['mixed_TRAIN_assets_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'/n)==d,n
    common=f'''**Latest verified status — 10 October 2026: repaired-head4 gradient return audited; backed-up archive cleanup closed; first manual training stage verified. Full goal incomplete.**

The downloaded L4 gradient diagnostic is complete with zero optimizer updates or
epochs. Independent R2 checks327 members,160 saved vectors,480 component-part
norms,100 initial raw/PNG/embedding records,40 losses and two fresh CPU replays.
Allthree repaired connected pieces have finite nonzero improvement gradients in
both exposed TRAIN cohorts; original equivalents have zero gradients. This is
learning-path evidence, not restored-image usefulness. The original norm-summary
checker failure remains; R2's1e-14 allowance affects derived L2 arithmetic only,
with exact zero/nonzero decisions and unchanged image-quality requirements.

The new independent training initializer preserves all100 CPU outputs exactly,
the full original fusion tensor and620 other original state entries. Three
independently owned pieces/147,456 elements are planned trainable. All stored
normalization, other original parameters and the retained checkpoints remain.
The selected positive reconstruction loss weights are prospectively reviewed
against saved TRAIN gradients, including first-Adam weight decay. Those arithmetic
predictions do not prove finite-step, PNG or native quality improvement.

The new self-contained packet has5505 assets including all5467 unchanged TRAIN
assets,781 canonical targets and3905 cases. Independent packet audit verifies
5506 archive members,90 local source bindings, canonical/clear-target equality,
50 fixed reference batches,100 initial-parity cases, lossless raw storage, safe
incoming boundaries, Windows training refusal, killed-worker failure retention
and allfive gcloud transfer commands. No training has been launched by the agent.

Authorized maintenance removes15 exact regular nlink1 home transfer archive
copies, each matching a complete CRC/hash-verified local backup. It reclaims
{cleanup['observed_recovered_GiB']:.3f}GiB observed. All{cleanup['research_metadata_entries_unchanged']:,} research filesystem metadata
entries and{cleanup['protected_byte_hashes_unchanged']:,} protected byte hashes remain unchanged. Cache bytes are not all
rehashed; their retention is additionally proven by disjoint home-only deletions,
independent-link checks and unchanged full research metadata. No checkpoint,
split, failure, unpacked training folder, cache, runtime or current new packet is
deleted. Fresh post-cleanup inventory reports{live['free_bytes']/1024**3:.3f}GiB free and an idle GPU.
This snapshot is not a guarantee for a future launch. The guide's earlier stopped
API observation is superseded by the fresh running-VM inventory.

Next manual action: **CCTV_DGP_HEAD4_CAPACITY_V1_VM_CURRENT.md**. The original
prepared guide remains. Current commands pin the verified public SSH host key
for the VM's new address. Its445,352,976-byte
execution archive SHA256 is
`c7651ee72f14b3ac1b2e44f2671982b880f21b88172a87d4a03561dc8c32c8b4`;
protocol SHA256 is
`3d6faf634d45867dc2edd7589aa0b5450790e307342f1897f8ea4b859a646532`.
Require14GiB free after installation. Worker2400s/external2430s,
each snapshot900s, fitting300s, export600s/external630s, VRAM20GiB,
return6GiB and free reserve1GiB are enforced. Manual tmux is mandatory.

This first test has a50-update ceiling and250 fitting exposures, checking all3905
TRAIN outputs at0/50. It completes0 full epochs (50/781 of an epoch). Both raw
and delivered PNG must meet the retained1% structure and preservation requirements;
all outputs/embeddings are retained losslessly. Full model/optimizer/scheduler,
RNG/schedule position, code, environment and failed-gate state are exported.
Logical full-state portability is distinguished from unproven bitwise CUDA resume.
Do not repeat a stopped recipe or resume a failed checkpoint.

Additional epochs1,2,5 with maximum5 remain the proposed longer study, requiring
audited preservation and useful native development gains. This50-update stage
does not test or waive the later10% capacity requirement, qualify a model, select
final identities or promote an app checkpoint. All native CCTV evidence remains
unpaired; synthetic paired metrics remain separate. Ancestral training epochs
and full historical subject overlap remain unconfirmed.

Allfive milestones and allseven completion families remain binding: masks,
sunglasses, strong lens glare, hands, obstructing hair, scarves and objects.
Preserve visible appearance/clear glasses/ordinary hair and existing design;
request clearer or less-covered input when insufficient. Show automatic masks
for correction, provide original/mask/one plausible estimate and PNG/bundle
downloads, and report automatic versus assisted results separately. No ethnicity,
hidden-identity or real Zamboanga performance claim is supported. DGP-led Auto
plus override, independent native/final visual review, meaningful regressions
and bundled inline Playwright app-flow verification are still required.

Current evidence: CCTV_DGP_HEAD4_REACTIVATION_V1_RESULTS.md,
CCTV_DGP_HEAD4_TRAINING_V1_DESIGN.md,
outputs/cctv_dgp_head4_capacity_v1_preparation/independent_packet_audit.json and
outputs/cctv_dgp_head4_archive_cleanup_v1/independent_audit.json.
Historical status entries below retain their original bytes and prior decisions.

---

'''
    backup=OUT/'before';backup.mkdir(parents=True);files={}
    for n in ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md']:
        path=ROOT/n;old=path.read_bytes();target=backup/n
        with target.open('xb') as f:f.write(old)
        assert sha(target)==sha(path);path.write_bytes(common.encode('utf-8')+old)
        assert path.read_bytes()==common.encode('utf-8')+target.read_bytes()
        files[n]={'before_sha256':sha(target),'after_sha256':sha(path),'full_previous_bytes_preserved':True}
    for k,v in protected.items():assert sha(ROOT/v['path'])==v['sha256']
    for n,d in p['local_sources'].items():assert sha(ROOT/n)==d
    write(OUT/'closure.json',{'complete':True,'protocol_sha256':sha(packet/'protocol.json'),'packet_sha256':audit['packet_sha256'],
        'canonical_documents':files,'protected':protected,'TRAIN_assets_preserved':5467,'source_bindings_preserved':90,
        'gradient_return_audit_sha256':sha(ROOT/'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json'),
        'packet_audit_sha256':sha(prep/'independent_packet_audit.json'),
        'cleanup_audit_sha256':sha(ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1/independent_audit.json'),
        'post_cleanup_inventory_sha256':sha(ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1/post_apply_inventory.json'),
        'current_manual_commands_audit_sha256':sha(prep/'current_manual_commands_audit.json'),
        'milestone_script_sha256':sha(Path(__file__)),'manual_training_prepared_not_run':True,
        'local_optimizer_updates':0,'app_changed':False,'goal_complete':False})
    print({'complete':True,'handoff_documents':3,'TRAIN_assets_preserved':5467,'source_bindings_preserved':90,
        'manual_training_prepared_not_run':True,'goal_complete':False})


if __name__=='__main__':main()
