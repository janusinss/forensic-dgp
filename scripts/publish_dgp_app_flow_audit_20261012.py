"""Publish bounded real-app verification while preserving the full model goal."""
from datetime import datetime, timezone
from pathlib import Path
from import_cctv_dgp_bank_group_direction_forward_v1_return import ROOT,OUT,ANALYSIS,read,sha,write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected,new,link


def main():
    dest=OUT/'dgp_app_flow_audit_20261012'
    assert not dest.exists()
    evidence=[ROOT/'scratch/app_flow_20261012_v1'/n for n in ['download_content_audit.json','visual_review.json','closure.json']]
    evidence += [ROOT/'scratch/app_flow_20261012_v2'/n for n in ['preflight.json','negative_api_audit.json',
        'native_browser_audit.json','native_auto_browser_audit.json','download_content_audit.json','visual_review.json','closure.json']]
    evidence += [ANALYSIS/'publication.json',ANALYSIS/'publication_audit.json']
    assert all(read(p)['complete'] for p in evidence)
    v1=read(evidence[0]);v2=read(ROOT/'scratch/app_flow_20261012_v2/download_content_audit.json')
    closure=read(ROOT/'scratch/app_flow_20261012_v2/closure.json')
    assert v1['actual_downloads_checked']==8 and v2['actual_downloads_checked']==6
    assert closure['owned_server_process_absence_checked'] and closure['configured_port_available_after_shutdown']
    assert not closure['goal_complete'] and not closure['current_model_usefulness_qualified']
    assert read(ROOT/'scratch/app_flow_20261012_v2/visual_review.json')['planned_screenshots_inspected']==13
    for name,h in read(ROOT/'scratch/app_flow_20261012_v2/preflight.json')['source_and_weight_files_sha256'].items():
        assert sha(ROOT/name)==h
    previous=read(ANALYSIS/'publication.json')
    names=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
           'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    assert all(sha(ROOT/n)==previous['documents'][n]['after_sha256'] for n in names)
    base='cctv_dgp_bank_group_step_check_v1'
    prep=read(OUT/(base+'_preparation.json'))
    assert not (OUT/prep['name']/'outputs').exists() and not (OUT/(base+'_return')).exists()
    report_name='CCTV_DGP_APP_FLOW_AUDIT_20261012.md'
    assert not (ROOT/report_name).exists()
    report=f'''# Local app flow audit — 12 October 2026

**The tested application paths work. Model quality and full goal completion
remain unqualified.** No application source,checkpoint or UI design was changed.
All browser work used the bundled Playwright skill through inline Node execution;
screenshots and diagnostics are under the git-ignored `scratch/` directory.
No scratch JavaScript test file or browser subagent was created.

| Verified scope | Evidence |
|---|---|
| One square photographic TRAIN input | Off,On,Auto and one52-pixel assisted completion request; four successful generations |
| One frozen native DEV input | Same138×165 ChokePoint crop with Off,On,Auto; three successful generations |
| Actual PNG/ZIP downloads | Eight TRAIN and six native files checked byte-for-byte against responses and bundle metadata |
| Backend refusal paths | Ten actual main-app422 responses with explicit actionable reasons and no generated output |
| Responsive browser state | Initial375/768/1280 layouts plus all three native result modes at those widths; no horizontal overflow,page exceptions or unexpected console errors |

The three native mode requests use the same development crop,not three
independent subjects. The assisted52-pixel patch is a functional test,not a
mask/sunglasses/glare/hand/hair/scarf/object quality result.

## Reviewed input and masks

The native crop's encoded bytes match the original frozen native contract and
its input-only review. That review classified readable rough structure before
model outputs. The alias under the multiscale packet is a byte-identical source
copy. Reserved final pixels were not used. Native dimensions remain138×165;
the prepared256×256 canvas does not add observed information.

The automatic proposal is empty for both tested uncovered faces. Native
ordinary hair and visible features therefore remain unselected. The assisted
TRAIN patch is reported separately. Review confirmation,operator input decision,
mask painting/import/edit invalidation,original/removal-area/one-result cards,
PNG downloads and optional review bundles were exercised in the earlier
square flow. Clear glasses and covering-family quality are not newly qualified
by these uncovered input examples.

## Guard and saved-artifact checks

Actual422 responses covered missing input review,needs-clearer,out-of-scope,
unconfirmed removal area,invalid restoration mode,mismatched mask dimensions,
complete covering,near-total covering,constant visible pixels and malformed
image data. Error details request review,a clearer crop or a less-covered crop
as applicable. Source-order checking confirms input-review/visible-support
guard conditions precede the generation inference block. Invalid-request
receipts do not establish that every possible insufficient face is automatically
detected; semantic face usability remains an explicit input-only decision.

All30 source/model/dependency/example bindings remain unchanged. Independent
saved-array checks verify native center-padding,observed bounds[20,0,234,256],
binary removal masks,native original dimensions,metadata RGB hashes,ZIP namespace/
CRC,actual downloaded bytes and declared checkpoint provenance. Off and native
Auto equal the prepared input exactly; prepared padding is unchanged in all
modes. Native On's saved float32 array composes to the delivered PNG through the
declared floor operation with original padding. The square On and Auto raw
arrays are identical. No new local gradients,optimizer updates or training
occurred. Local inference was used for the real application requests.

{link('TRAIN download audit','scratch/app_flow_20261012_v1/download_content_audit.json')}.
{link('Native download and refusal audit','scratch/app_flow_20261012_v2/download_content_audit.json')}.

## Visual outcome and limits

All eight planned square-flow screenshots and13 native-flow screenshots were
inspected. The existing design remains readable across the required widths.
Original,mask and one estimate remain available beside the downloads.
Native Auto correctly chooses preservation for this readable crop; forced On
softens eyes,skin,hair and lip detail. This confirms the override and input
preservation behavior; it does not demonstrate a better restoration model.

{link('Native visual review','scratch/app_flow_20261012_v2/visual_review.json')}.
{link('Native Auto result at1280px','scratch/app_flow_20261012_v2/native-auto-1280.png')}.
{link('Native forced-On output','scratch/app_flow_20261012_v2/native-on-output.png')}.

The native example has no aligned clean reference. It is unpaired evidence;
no native PSNR/SSIM or exact recovered/hidden-identity claim is made. Dataset
labels establish neither ethnicity nor Zamboanga performance. The owned
loopback server was stopped through its own execution session;its process was
absent and the configured port was available afterwards. No other process or
VM workload was stopped.

## Full goal and next manual work

Goal ACTIVE. The latest four model proposals still failed preservation and
the1% structure requirement;best raw gain0.862655% with37 raw/37 PNG
group–metric failures. The new signed small-step diagnostic is verified and
UNRUN;no new incumbent or extra epochs are qualified.
{link('Five manual VM transfer/tmux/download steps','CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md')}.

The permanent baseline SHA256 remains
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
Its component origins and the proposed replacement's own condition/reconstruction
path with an external frozen prior remain declared. Checkpoints,splits,caches,
full states,provenance,gate failures and app design stay retained. Actual training,
gradients and model proposals remain human-manual L4 tmux work;agent VM access
remains hash-bound maintenance only. No agent VM neural run occurred.

All five milestones remain required: frozen native DEV/separate final identities
and provenance;identical-input comparison with raw/display separation;justified
finite improvements;qualified primary DGP with Auto/override;independent useful
DEV/final outputs,regressions and full app review. Tested functional paths are
partial milestone-five evidence;useful native restoration remains unqualified.

All seven covering families remain in scope:masks,sunglasses,strong glare,
hands,obstructing hair,scarves and objects. Each needs reviewed automatic and
assisted results with editable removal preview,visible appearance/clear-glasses/
ordinary-hair preservation,a documented small margin,less-covered requests,
original/mask/one plausible estimate,PNG and optional bundle. Independent final
quality review and full-family scope remain open. Credit/migration notifications
and the November2026 deadline continue to apply.
'''
    new(ROOT/report_name,report)
    prefix=f'''# Latest app verification — 12 October 2026; model improvement still pending

Fresh bundled inline Playwright verified four square TRAIN requests and three
requests on one frozen138×165 native DEV crop. Fourteen actual PNG/ZIP downloads
passed byte/content/metadata checks. Ten actual backend refusal cases passed.
Initial375/768/1280 layouts and all three native result modes at those widths
have no overflow,page exceptions or unexpected console errors. All21 planned
screenshots were inspected. Source/model hashes and existing design are unchanged.

Native Auto equals Off and preserves the prepared input;forced On softens
visible detail. This is functional override/geometry/download evidence from
one unpaired crop,not useful model or full-family qualification. No native
clean-target metrics or reserved final pixels were used. The owned local test
server is stopped. {link('Bounded app verification report',report_name)}.

The four latest finite model proposals remain rejected:best raw gain0.862655%
against1%,with37 raw/37 PNG preservation checks failed. Signed smaller-step
diagnosis remains verified and UNRUN. {link('Five exact manual VM steps','CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md')}.
No extra epochs,new checkpoint or app model promotion is qualified.

Goal ACTIVE. All five milestones and seven covering families remain required,
including independent useful native DEV/final review,primary DGP Auto/override,
editable automatic removal area,separate automatic/assisted results,visible
appearance/clear-glasses/ordinary-hair preservation,clearer/less-covered input
requests,original/mask/one plausible estimate and PNG/optional bundle downloads.
Permanent baseline SHA256646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b
and checkpoints/splits/caches/full states/provenance/failures stay retained.
Unpaired CCTV and paired synthetic evidence stay separate;final identities stay
outside tuning. No ethnicity,Zamboanga or exact hidden-identity claim. Training/
gradients/model proposals stay human-manual L4 tmux work;agent VM access stays
inventory-first,hash-bound maintenance only. Historical bytes below are retained.

'''
    dest.mkdir();backup=dest/'publication_before';backup.mkdir()
    documents={name:prepend_expected(ROOT/name,prefix,backup/name,previous['documents'][name]['after_sha256']) for name in names}
    write(dest/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},documents=documents,
        new_documents_sha256={report_name:sha(ROOT/report_name)},functional_paths_verified_not_model_quality=True,
        model_quality_qualified=False,manual_signed_diagnostic_UNRUN=True,all_goal_milestones_and_covering_families_retained=True,
        no_agent_VM_neural_launch=True,goal_status='active',goal_complete=False))
    print(dict(complete=True,histories_retained=4,app_report_created=True,manual_signed_diagnostic_UNRUN=True,goal_complete=False))


if __name__=='__main__':main()
