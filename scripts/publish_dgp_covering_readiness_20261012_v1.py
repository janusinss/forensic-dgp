"""Publish a bounded covering recount and preserve the complete handoff history."""
import hashlib
import json
from pathlib import Path

from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import new, prepend_expected

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/dgp_covering_readiness_20261012_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def link(label, path):
    return '['+label+'](<'+(ROOT/path).as_posix()+'>)'


def main():
    a = read(OUT/'audit.json')
    latest = read(ROOT/'outputs/dgp_app_flow_audit_20261012/publication.json')
    assert a['complete'] and a['new_neural_calls'] == a['gradients'] == a['optimizer_updates'] == 0
    assert not a['automatic_qualified'] and not a['assisted_qualified'] and not a['goal_complete']
    assert a['checked_bindings'] == 2529 and a['unique_files'] == 1182
    assert len(a['historical_instruction_bindings']) == 2
    assert len(a['family_counts']) == 7 and a['photo_pairs'] == 18
    for proof in a['historical_instruction_bindings']:
        assert sha(ROOT/proof['retained_snapshot']) == proof['historical_sha256']
    labels = {'masks':'Masks', 'sunglasses':'Sunglasses', 'strong_glare':'Strong glare',
              'hands':'Hands', 'obstructing_hair':'Obstructing hair', 'scarves':'Scarves', 'objects':'Objects'}
    table = []
    for key, name in labels.items():
        counts = a['family_counts'][key]['condition_counts']
        original, degraded = counts['original_photo'], counts['synthetic_degraded_photo']
        table.append('| '+name+' | '+str(original['eligible_cases'])+' | '+
            str(original['empty_automatic_proposals_on_eligible_coverings'])+' | '+
            str(degraded['empty_automatic_proposals_on_eligible_coverings'])+' |')
    report_name = 'CCTV_DGP_COVERING_READINESS_20261012.md'
    report = '''# Covering evidence recount — 12 October 2026

The automatic detector misses many degraded covering examples, while assisted
completion still has generated anatomy and join defects. These are separate
limitations. More epochs of the visible-face restorer alone are not demonstrated
to resolve either one. No new model, mask, application design or training recipe
is adopted by this audit.

## Fresh checks and scope

The arithmetic/image-only recount completed in '''+f"{a['seconds']:.2f}"+''' seconds
under a180-second cap. It verifies2,529 bindings across1,182 unique files from
five retained comparisons: current app Off completion, union conditioning,
fusion disabled, margin blending and the converted MAT comparison. Executable
code, models, inputs, masks and saved artifacts retain exact hashes.

Two instruction documents have newer milestone prefixes. Their historical
protocol versions were independently located as exact SHA256-matched suffixes
of the current documents and retained as separate snapshots. The initial stop
and checker source remain in `outputs/dgp_covering_readiness_20261012_v1_stop/`.
No model/output tolerance or quality gate changed.

There are18 previously exposed photographs, each with a synthetic degradation:
36 requests,32 eligible entries including four empty controls,28 nonempty
assisted estimates and four prior input-only exclusions. These are photographs,
not native CCTV. Matched degradations and reused source-photo masks are not
independent people or annotation trials. Pretraining overlap is unknown; there
is no aligned clean hidden-face reference, hidden PSNR/SSIM, ethnicity, recovered
identity or Zamboanga-performance claim. Reserved final pixels were not used.

## Automatic proposals, separately recounted

Of14 eligible covering photographs, four original and11 degraded automatic
proposals are empty. Across all18 entries, including controls/exclusions, the
counts are five and14. The original uncovered control has11,683 automatic
proposal pixels; its assisted reviewed mask is empty. Both clear-glasses
controls have empty automatic proposals.

| Family | Eligible photos per condition | Empty original proposals | Empty degraded proposals |
| --- | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

Combined hand/mask and sunglasses/glare cases belong to multiple families;
these rows therefore overlap. Counts refer to source photographs, not a
known count of independent identities. Approximate operator removal footprints
are not expert segmentation truth. Their overlap/miss/extra counts and the two
cached detector comparisons were recomputed exactly; better overlap does not
erase those candidates' historical quality failures.

No automatic mask was approved for generation in these retained comparisons.
Consequently they provide zero automatic-generation quality results. All seven
automatic and assisted families remain separately unqualified.

## Assisted preservation and remaining causes

Every eligible Off/candidate output preserves copied visible source pixels
exactly:32 current,32 union,32 fusion-off,32 margin and32 converted-MAT images.
Union, fusion-off and margin also preserve protected support exactly. Each has
four byte-exact uncovered/clear-glasses bypasses; all four excluded inputs stay
excluded. The new recount independently reproduces28 union and28 fusion-off
PNG compositions from saved256 float stages. Earlier independent full pipeline
checks, raw network stages, all output pages and dated visual reviews remain
retained; this recount does not claim a fresh neural replay or visual assessment.

The dated all-case reviews distinguish three causes:

1. Automatic proposal misses require a separately reviewed detection/data path.
2. Covering fragments outside the final mask are exact source copies. Corrections
   require an input-only removal-area review before generation; global dilation
   would risk erasing visible features, clear glasses or ordinary hair.
3. Defects inside generated support require completion-model evidence. Context,
   blending, fusion-off and the converted prior did not qualify the full scope.

Preserved bytes establish processing fidelity, not plausible anatomy or useful
completion. The prior human architecture decision permits diagnosis; this
recount justifies no unchanged failed rerun, new inference sweep or training.
The historical implementing-assistant development reviews remain distinct from
independent final human assessment.

'''+link('Fresh recount and per-case evidence','outputs/dgp_covering_readiness_20261012_v1/audit.json')+'''.
'''+link('Retained fusion diagnostic findings','CCTV_DGP_COMPLETION_FEATURE_FUSION_OFF_V1_RESULTS.md')+'''.
'''+link('Copied-versus-generated support evidence','CCTV_DGP_COMPLETION_PIXEL_SUPPORT_V1_RESULTS.md')+'''.

## Main model and next work

The permanent trained restorer SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
It is the fine-tuned DeblurGAN-v2-compatible baseline. The proposed own
conditioning/reconstruction path uses an attributed external frozen StyleGAN2
prior; the separate completion component is pretrained CodeFormer. Component
origins do not imply useful output or an own-trained generative prior.

The signed small-step main-model diagnostic remains prepared and UNRUN. It
tests the discrepancy between favorable gradients and finite preservation
before more training. Actual gradients, weight proposals and training remain
human-manual L4 tmux work with the existing finite stops and verified transfers.
'''+link('Exact manual transfer, tmux and download commands','CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md')+'''.

Goal ACTIVE. All five milestones, useful native DEV/final review, independent
final assessment, primary DGP Auto/override and full covering-family usefulness
remain required. Keep editable automatic removal preview, visible appearance,
clear glasses/ordinary hair, documented small margin, clearer/less-covered
requests, original/mask/one plausible estimate and PNG/optional bundle downloads.
Keep native unpaired evidence separate from paired synthetic metrics. Original
checkpoints, splits, caches, full states, provenance and gate failures remain
preserved. No VM connection, neural call, model/mask edit or training occurred
in this recount.
'''
    new(ROOT/report_name, report)
    prefix = '''# Latest covering evidence recount — 12 October 2026; main diagnosis UNRUN

A fresh saved-file audit verifies2,529 bindings/1,182 unique files across five
retained completion comparisons. Among14 eligible covering photographs, the
current automatic detector returns empty proposals for4 original/11 degraded
cases; one original uncovered control has11,683 false proposal pixels.
Approximate footprint overlap is not segmentation accuracy. No automatic
proposal was approved for generation; automatic and assisted quality stay
separate and unqualified across all seven families.

All32 eligible copied-source outputs in each comparison are exact; protected
support and four clear/uncovered bypasses are retained. Four prior exclusions
stay excluded. This is processing/provenance evidence from exposed photographs
and matched synthetic degradations, not native CCTV, hidden accuracy or final
quality review. Dated visual failures remain binding. Copied exterior fragments
need input-only area review; generated anatomy needs separate model evidence.
No new model, mask, app change, inference or VM training occurred.

'''+link('Covering recount and distinct failure causes',report_name)+'''.
The signed main-model diagnostic remains verified and UNRUN:
'''+link('Five manual VM steps','CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md')+'''.
Goal ACTIVE; all five milestones and seven automatic/assisted covering families
remain required. The preserved baseline, checkpoints/splits/caches/full states/
provenance/failures and existing app design stay unchanged. Historical bytes below
are retained in full.

'''
    expected = latest['documents']['PROJECT_HANDOFF.md']['after_sha256']
    documents = {'PROJECT_HANDOFF.md': prepend_expected(ROOT/'PROJECT_HANDOFF.md', prefix,
        OUT/'PROJECT_HANDOFF.before.md', expected)}
    receipt = dict(complete=True, audit_sha256=sha(OUT/'audit.json'), report=report_name,
        report_sha256=sha(ROOT/report_name), documents=documents,
        publisher_sha256=sha(Path(__file__)), goal_complete=False, training_started=False)
    with (OUT/'publication.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'complete':True,'report':report_name,'handoff_updated':True,'training_started':False}))


if __name__ == '__main__':
    main()
