"""Publish actual diagnostic evidence with complete prior-document backups."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_hq_code_oracle_local_20261019_v1_r1'
REVIEW = OUT / 'review_v1'
PUB = REVIEW / 'publication_v1'
REPORT = ROOT / 'CCTV_CODEFORMER_HQ_CODE_ORACLE_V1_REVIEW.md'
DOCS = ('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')


def pin(path):
    b = path.read_bytes()
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def new(path, payload):
    assert path.resolve(strict=False).is_relative_to(ROOT) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())


def main():
    a=json.loads((REVIEW/'audit_v1.json').read_bytes())
    v=json.loads((REVIEW/'root_visual_review_v1.json').read_bytes())
    r=json.loads((OUT/'result_v1.json').read_bytes())
    assert a['complete'] and a['audit_pass'] and a['actual_top_level_model_calls']==43
    assert r['optimizer_updates']==0 and r['state_before']==r['state_after'] and not r['model_qualified']
    assert v['complete'] and v['sheets_reviewed']==6 and v['audit_sha256']==pin(REVIEW/'audit_v1.json')['sha256']
    assert not PUB.exists() and not REPORT.exists()
    for row in v['sheets']:
        assert pin(ROOT/row['path'])=={k:row[k] for k in ('bytes','sha256')}
    assert all(x['native_HQ_vs_R2_projected_changed_positions']==0 for x in a['all6_HQ_features_native_and_exact_R2_projection_recomputed'])
    evidence={str(p.relative_to(ROOT)):pin(p) for p in (OUT/'result_v1.json',REVIEW/'audit_v1.json',REVIEW/'root_visual_review_v1.json',REVIEW/'visual_manifest_v1.json',ROOT/'outputs/codeformer_filter_fusion_next_design_20261019_v1/oracle_design_frozen_v1.json')}
    table=[]
    for profile,label in (('clear','Clear'),('moderate_existing_gaussian_2p4','Moderate blur'),('strong_fixed_blur3p2_lr64_jpeg50','Strong synthetic blur')):
        x=a['profile_readability_summaries'][profile]['views']['PNG_MAE']
        table.append(f"| {label} | {x['natural_mean']:.6f} | {x['oracle_mean']:.6f} | {x['relative_mean_MAE_reduction_percent']:.2f}% | {x['oracle_better_cases']}/6 |")
    report=f'''# Clean-code oracle diagnostic: modest gains, strong blur unresolved

19 October 2026. User testing found **“They look similar; neither fixes strong blur.”** R2 and fusion50 remain experimental comparisons, with their quality stops retained.

## Actual evidence

A new local CPU FP32 inference run completed in **{r['seconds']:.2f} seconds**. It made **43 top-level model calls**: one unwrapped R2 parity call, six HQ teacher reconstructions, and 18 natural plus 18 oracle R2 calls. **Zero optimizer updates, gradients, backwards or VM calls.** R2 and teacher states stayed exact; the main app default and all checkpoint files stayed unchanged. Peak sampled RSS was {r['resources']['maximum_RSS_bytes']:,} bytes. Worker/external caps were 900/960 seconds; neither fired.

Six fixed, previously admitted TRAIN photographs supplied clear, existing Gaussian2.4, and fixed Gaussian3.2/LR64/JPEG50 inputs. Root reviewed them before outputs: 16 readable observations and two borderline strong cases. These are six photographs, not 18 independent identities. They were historically used in TRAIN. No DEV, native CCTV or FINAL pixels entered this test. The strong profile is a synthetic proxy, not calibrated CCTV or the unknown-origin wild inputs.

The intervention replaces only R2's selected codes. Genuine HQ encoder features from direct1024-to-LANCZOS512 targets are projected onto the exact frozen R2 codebook. AdaIN, observed features, fusion and weights stay fixed. Native HQ labels are retained separately; they agree with projected labels at all captured positions here. The HQ generator differs slightly from R2, so its reconstruction is a separate reference, not an exact R2-prior ceiling. **Clean-reference codes are unavailable at deployment.** This oracle is a diagnostic, not a restoration method users can run on unknown CCTV.

## Saved-image and visual checks

The independent no-neural audit verified 268 output files, 86 attempt/return records, actual unwrapped/natural parity, all 18 pairs of unchanged observed logits/features/natural selections, six projections, RAW/PNG conversion and every paired MAE. Root personally viewed all six exact256-cell sheets.

| Profile, six cases each | R2 PNG MAE | Oracle PNG MAE | Mean MAE reduction | Cases better |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

All six strong cases reduce mean PNG MAE by 2.46%. The four predeclared readable strong cases reduce 2.07% (3/4 better); the two borderline cases remain separate (one improves). This is **paired RGB error reduction**, not the old DGP 1% structure gate, SSIM improvement, identity accuracy or deployment quality. RAW shows the same small, mixed pattern.

Natural and oracle outputs remain visually close. Eye/mouth, tone and texture changes are modest; strong-blur softness and altered accessories remain. The separate HQ teacher also changes some hat/scarf details. High token disagreement even on clear inputs reflects different encoders and clean512 versus upscaled256 evidence; it is not calibrated face or identity confidence.

## Corrected history and next decision

The latest fusion run froze code prediction, but earlier V1/V2/FP32 pilots **already trained encoder/code prediction with HQ code CE and latent losses**. Repeating generic encoder unfreezing or CE would repeat earlier work. See [history comparison](outputs/codeformer_filter_fusion_next_design_20261019_v1/loss_review_peer/HISTORY_COMPARISON.md). The official [CodeFormer joint trainer](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/basicsr/models/codeformer_joint_model.py#L135-L175) is the primary method reference, not proof that another such run will succeed.

This modest conditional oracle result does not justify a code-prediction-only pilot as the solution. Review **selective use of corrupted image features during reconstruction**, rather than the previously rejected global fidelity-off switch. That remains a hypothesis. A distinct finite training recipe, audited readable TRAIN inputs, unchanged preservation checks, resource/source verification and manual VM transfer are still required. No new training packet or trained checkpoint was created by this diagnostic.

The first local startup stopped on missing psutil before neural work. Its source/failure/empty log remain preserved. A host process inventory confirmed no matching live worker. Pinned psutil7.0.0 was installed, actual dependency preflight moved before spawning, and the same scientific plan ran once in a separate R1 directory. This environment repair changed no quality gate.

All original models, stopped states, splits, provenance, failures and migration backups remain. Native CCTV stays source-separated and unpaired; FINAL stays unused. No ethnicity, Zamboanga or exact-identity claim follows. Restoration acceptance, independent final review and main app Auto/override integration remain outstanding; completion development remains deferred. The full goal stays active.

Evidence: [actual result](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/result_v1.json), [independent saved-output audit](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/review_v1/audit_v1.json), [six-sheet root review](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/review_v1/root_visual_review_v1.json), [first comparison sheet](outputs/codeformer_hq_code_oracle_local_20261019_v1_r1/review_v1/sheets/tr_ffhq_03761.png). Existing model testing: [local guide](CCTV_CODEFORMER_FILTER_FUSION_LOCAL_TEST_GUIDE.md).
'''
    header='''# Strong-blur diagnosis updated; restoration acceptance pending — 19 October 2026

User testing found R2 and fusion50 similar; neither fixes strong blur. A new 43-call local inference-only code-oracle diagnostic completed, with independent saved-output checks and all six sheets personally reviewed. Clean-reference codes lowered mean PNG MAE by 2.46% on all six strong synthetic cases (2.07% on the four predeclared readable cases); visible differences remain small. This is not a structure-gate pass, deployable method, new training or accepted model.

[CCTV_CODEFORMER_HQ_CODE_ORACLE_V1_REVIEW.md](CCTV_CODEFORMER_HQ_CODE_ORACLE_V1_REVIEW.md) preserves evidence and corrected history: earlier V1/V2/FP32 already trained encoder/code prediction. Do not repeat those recipes or extend stopped fusion50 unchanged. Review selective corrupted-feature use before a distinct finite manual VM pilot. Original checkpoints, failures, source-separated native/unpaired evidence and FINAL remain preserved; app defaults are unchanged. Completion stays deferred until restoration acceptance. All five milestones and the full goal remain active. Earlier storage cleanup is closed; no deletion occurred here.

Earlier status below remains historical evidence.

---

'''.encode()
    originals={n:(ROOT/n).read_bytes() for n in DOCS};before={n:pin(ROOT/n) for n in DOCS}
    for n,b in originals.items():
        new(PUB/'source_backups'/(n+'.before'),b)
        assert pin(PUB/'source_backups'/(n+'.before'))==before[n]
    for n,b in originals.items():new(PUB/(n+'.staged'),header+b)
    assert all(pin(ROOT/n)==p for n,p in before.items()) and all(pin(ROOT/n)==p for n,p in evidence.items())
    new(REPORT,report.encode());records={}
    for n in DOCS:
        with (PUB/'journal.jsonl').open('ab') as f:
            f.write((json.dumps({'event':'replace_intent','path':n,'before':before[n]})+'\n').encode());f.flush();os.fsync(f.fileno())
        os.replace(PUB/(n+'.staged'),ROOT/n)
        assert (ROOT/n).read_bytes()==header+originals[n]
        records[n]={'before':before[n],'after':pin(ROOT/n),'suffix_offset_bytes':len(header),'backup':str((PUB/'source_backups'/(n+'.before')).relative_to(ROOT))}
        with (PUB/'journal.jsonl').open('ab') as f:
            f.write((json.dumps({'event':'replace_complete','path':n,'after':records[n]['after']})+'\n').encode());f.flush();os.fsync(f.fileno())
    new(PUB/'publication_v1.json',(json.dumps({'complete':True,'evidence':evidence,'documents':records,'report':{'path':str(REPORT.relative_to(ROOT)),**pin(REPORT)},'source':pin(Path(__file__)),'model_qualified':False,'new_training_launched':False,'overall_goal_complete':False},indent=2)+'\n').encode())
    print(json.dumps({'complete':True,'report':str(REPORT),'backups':4,'documents_updated':4,'training_updates':0}))


if __name__=='__main__':main()
