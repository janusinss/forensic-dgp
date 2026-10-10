"""Hash-preserved canonical handoff update after verified initialization/packet."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1'))
from cctv_dgp_head4_reactivation_contract_v1 import read,write,sha
OUT=ROOT/'outputs/cctv_dgp_head4_reactivation_v1_milestone'


def main():
    assert not OUT.exists()
    packet=ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1'
    p=read(packet/'protocol.json')
    prep=ROOT/'outputs/cctv_dgp_head4_reactivation_v1_preparation'
    audit=read(prep/'independent_packet_audit.json')
    assert audit['complete'] and not audit['VM_launched']
    for name,digest in p['local_source_bindings'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['assets_sha256'].items():assert sha(packet/name)==digest,name
    corpus=read(ROOT/'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json')['mixed_TRAIN_assets_sha256']
    base=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'
    for name,digest in corpus.items():assert sha(base/name)==digest,name
    assert len(corpus)==5467
    coverage=read(ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json')
    assert coverage['complete'] and coverage['canonical_TRAIN_targets_verified']==781
    parity=read(ROOT/'outputs/cctv_dgp_head4_reactivation_parity_v1/independent_audit.json')
    assert parity['complete'] and all(parity['phase3_dead_branch_value_equality'].values())
    original={'current':'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
        'Phase3':'checkpoints/dgp_zamboanga_final.pth','immutable_training_plan':'CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md'}
    protected={name:{'path':path,'sha256':sha(ROOT/path)} for name,path in original.items()}
    assert protected['current']['sha256']==p['original_checkpoint_sha256']
    assert protected['Phase3']['sha256']==parity['Phase3_checkpoint_sha256']
    before=OUT/'before';before.mkdir(parents=True)
    common='''**Latest verified status — 10 October 2026: original deepest-path numerical limitation isolated; exact-preserving initialization and manual L4 gradient packet verified. Full goal incomplete.**

The current DGP's deepest head is zero on100 exposed TRAIN inputs. Its two kernels
and connected fusion slice are near6.305e-40 and are bitwise unchanged from Phase3.
Other heads remain active. This is one demonstrated branch limitation, not a unique
explanation of all prior failures or proof of image-quality improvement.

A separate fixed initializer copies our current trained adjacent head/fusion and
uses six frozen anchors. All100 initial CPU outputs match the current DGP exactly;
the branch responds on100/100. Independent fresh replays and619 untouched full
state entries pass. No local gradients, optimizer updates, new epochs or app
adoption occurred. Original checkpoints, normalization, thresholds and failures remain.

The full input/target audit now covers3,905 TRAIN cases,781 canonical targets and24
unpaired native development inputs. All5467 TRAIN asset hashes are preserved.
HQ canonical hashes match; legacy thumbnail-hash mismatches are documented, not
corruption. Full-corpus motion cases have limited geometric overlap with native
eye spacing. This does not establish equivalent resolved detail or usable routing.

Next manual action: use **CCTV_DGP_HEAD4_REACTIVATION_V1_VM.md**. The verified
183,353,481-byte packet is a matched original/repaired gradient diagnostic only:
160 queries, zero optimizer updates/epochs,600s worker/630s external stop;3GiB
free required after install. It has not run on the VM. A finite nonzero connected
gradient pass does not waive the1%-at50/10%-at800 or preservation requirements.
Additional epochs1/2/5 require a justified changed training recipe and later
independent capacity/development review. No historical failure is resumed.

All five milestones and seven completion families remain binding. Native CCTV
stays unpaired; no ethnicity or Zamboanga performance claim is made. Reserved-final
identities remain outside tuning. Preserve visible appearance and existing design,
show completion masks for correction, request clearer/less-covered inputs when
insufficient, and retain original/mask/plausible-result downloads and independent
final/inline-Playwright requirements. Direct VM connection remains maintenance
only; all autograd/training is manually launched by the user in tmux.

Current evidence: CCTV_DGP_HEAD4_REACTIVATION_V1_DESIGN.md,
CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md and
outputs/cctv_dgp_head4_reactivation_v1_preparation/independent_packet_audit.json.
The preceding status entries below are retained historical evidence.

---

'''
    files={}
    for name in ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md']:
        path=ROOT/name;data=path.read_bytes();backup=before/name
        with backup.open('xb') as stream:stream.write(data)
        assert sha(backup)==sha(path)
        new=common.encode('utf-8')+data
        path.write_bytes(new)
        assert path.read_bytes().endswith(backup.read_bytes())
        assert path.read_bytes()[:len(common.encode('utf-8'))]==common.encode('utf-8')
        files[name]={'before_sha256':sha(backup),'after_sha256':sha(path),'full_previous_bytes_preserved':True}
    for name,digest in p['local_source_bindings'].items():assert sha(ROOT/name)==digest,name
    for name,item in protected.items():assert sha(ROOT/item['path'])==item['sha256'],name
    receipt={'complete':True,'protocol_sha256':sha(packet/'protocol.json'),'packet_sha256':audit['packet_sha256'],
        'independent_packet_audit_sha256':sha(prep/'independent_packet_audit.json'),
        'full_training_coverage_audit_sha256':sha(ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json'),
        'initial_parity_audit_sha256':sha(ROOT/'outputs/cctv_dgp_head4_reactivation_parity_v1/independent_audit.json'),
        'canonical_documents':files,'protected':protected,'TRAIN_assets_reverified':5467,
        'historical_and_new_source_bindings_reverified':len(p['local_source_bindings']),
        'manual_VM_stage_prepared_not_run':True,'new_trained_checkpoint':False,
        'local_gradient_queries':0,'optimizer_updates':0,'app_changed':False,'goal_complete':False}
    write(OUT/'closure.json',receipt)
    print({'complete':True,'handoff_documents':3,'TRAIN_assets_preserved':5467,
        'source_bindings_preserved':len(p['local_source_bindings']),'VM_launched':False,'goal_complete':False})


if __name__=='__main__':main()
