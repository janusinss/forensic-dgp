"""Publish the reviewed V1 return; preserve every prior instruction byte.

No neural, optimizer, network or VM operations. This records the root's completed
52-page visual review and visibly supersedes historical launch instructions.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_adaptation_v1_return_review'
SHEETS = REVIEW / 'visual_sheets'
PUBLICATION = REVIEW / 'return_publication_v1'
FILES = (
    'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
    'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
    'CCTV_CODEFORMER_ADAPTATION_V1_PREPARATION.md',
    'CCTV_CODEFORMER_ADAPTATION_V1_VM.md',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def page_note(number):
    if number <= 6:
        return 'HQ blur: fewer conspicuous baseline texture/feature artifacts in many rows; brighter and smoother candidate, with remaining feature differences.'
    if number == 7:
        return 'Last HQ blur rows retain the brighter/smoother tradeoff; first clear controls show smaller changes.'
    if number <= 13:
        return 'HQ clear: changes are generally subtler than severe blur; small tone, detail and expression differences remain. Visual similarity does not waive measured controls.'
    if number == 14:
        return 'Last clear rows then compound: dark-input candidate often less artifacted and closer in exposure, with residual smoothing and feature differences.'
    if number <= 19:
        return 'HQ compound: many dark baseline artifacts reduce and face layout appears more coherent; some eyes, expression, hair and background details still differ from target.'
    if number == 20:
        return 'Last compound rows improve visibility but remain smooth or altered; first JPEG row changes little.'
    if number <= 26:
        return 'HQ JPEG: comparatively subtle changes; preserved broad layout with some smoothing, tone and feature differences.'
    if number == 27:
        return 'Last JPEG and first light rows: small changes on JPEG; several light cases become brighter or softer.'
    if number <= 33:
        return 'HQ light: broad layout mostly retained; some rows oversmooth or brighten despite clearer appearance. Individual detail and tone need preservation.'
    if number == 34:
        return 'Last light row then lowlight: candidate often reduces dark baseline texture artifacts and lifts exposure; residual feature errors and smoothing persist.'
    if number <= 40:
        return 'HQ lowlight: many outputs are more visible and less artifacted than baseline, with incomplete target detail and occasional changes in features/expression.'
    if number <= 43:
        return 'Native ChokePoint: candidate often softer than original CodeFormer; no consistent useful improvement established. Preserve visible glasses, hair and expression; no aligned clean reference.'
    if number <= 46:
        return 'Native QMUL-SurvFace: many severe/turned/obstructed inputs remain implausible or insufficient; candidate may brighten glare-like areas. These are refusal/clearer-crop concerns, not qualified recovered faces.'
    if number == 47:
        return 'Old paired Asian-source: normal blur often overbrightens; clear09306 visibly changes tone/features and is the authoritative preservation failure. Source labels do not establish ethnicity.'
    if number == 48:
        return 'Old paired Asian-source dark cases: artifacts often reduce but some severe outputs still distort eyes/mouth/expression or lack supported detail.'
    if number == 49:
        return 'Old paired Asian-source motion: candidate often brighter/smoother; target feature and tone differences persist.'
    if number == 50:
        return 'Old paired thumbnails blur/clear: blur artifacts often reduce but exposure rises; clear changes smaller. These old low-resolution targets are separate from HQ evaluation.'
    if number == 51:
        return 'Old paired thumbnails dark/compound: many baseline artifacts reduce; changed expression/shape or missing target detail remains in severe cases.'
    return 'Old paired thumbnails motion: candidate smoother/brighter; fewer texture artifacts do not prove target fidelity.'


def main():
    assert not PUBLICATION.exists(), 'Publication already exists; preserve it and review before any new revision'
    manifest_bytes = (SHEETS / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['complete'] and len(manifest['pages']) == 52
    pages, case_ids = [], []
    for number, page in enumerate(manifest['pages'], 1):
        assert page['file'] == f'comparison_{number:02d}.png'
        data = (SHEETS / page['file']).read_bytes()
        assert digest(data) == page['sha256']
        case_ids.extend(page['case_ids'])
        pages.append({**page, 'opened_by_root_reviewer': True,
                      'observations': page_note(number)})
    assert len(case_ids) == len(set(case_ids)) == 406
    originals = {name: (ROOT / name).read_bytes() for name in FILES}
    for name, data in originals.items():
        assert not data.startswith(b'# CodeFormer V1 return audited: quality-stopped')
    now = datetime.now(timezone.utc).isoformat()
    visual = {
        'complete_visual_coverage': True, 'created_utc': now,
        'reviewer': 'root AI agent; descriptive, not blinded human acceptance',
        'pages_reviewed': 52, 'case_pairs_reviewed': 406,
        'snapshot_outputs_represented': 812,
        'review_view': 'delivered native PNG; 256 originals retained; 192px sheet tiles',
        'all_pages_opened_in_tool_at_original_sheet_resolution': True,
        'manifest_sha256': digest(manifest_bytes),
        'case_set_unique_and_complete': True,
        'no_extra_sharpening_or_exposure_correction': True,
        'RAW_view_boundary': 'Array/pixel/PNG relationship independently audited; these sheets display native PNG only',
        'user_acceptance': 'pending', 'independent_final_evaluation': False,
        'model_qualified': False, 'native_CCTV_qualified': False,
        'pages': pages,
    }
    write_new(SHEETS / 'visual_review_v1.json',
              (json.dumps(visual, indent=2, ensure_ascii=False) + '\n').encode())
    prefix = (
        '# CodeFormer V1 return audited: quality-stopped at 200 updates — 15 October 2026\n\n'
        '**CURRENT: 200/1200 optimizer updates; 800 TRAIN exposures; preservation failed.**\n'
        'The first fine-tune has visible improvements and mixed profile metrics, but is\n'
        'not an accepted replacement. One clear control exceeds the unchanged individual\n'
        'RAW/PNG bound. Do not relaunch the V1 commands or resume its stopped state unchanged.\n\n'
        '[Returned result and diagnosis](CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md).\n'
        '[Every comparison page](outputs/codeformer_adaptation_v1_return_review/visual_sheets/INDEX.md).\n'
        'All 52 pages / 406 case pairs were reviewed; user acceptance and reserved-final\n'
        'evaluation remain pending. Native CCTV remains unpaired and unqualified.\n\n'
        'All 350 trainable tensors changed; 165 frozen restoration tensors stayed exact.\n'
        'The archive, checkpoints, failed guard and splits remain preserved. The measured\n'
        'next design addresses exposure/loss imbalance using TRAIN-only data, with the\n'
        'same preservation requirements. No new training has launched, no candidate was\n'
        'adopted, and further completion/detector development remains deferred.\n\n'
        '**The launch-ready / PREPARED / UNRUN text below is retained historical evidence.**\n'
        'This current status supersedes it. The full thesis goal remains active.\n\n---\n\n'
    ).encode('utf-8')
    entries = []
    for name, data in originals.items():
        backup = PUBLICATION / 'prior_documents' / name
        write_new(backup, data)
        target = ROOT / name
        assert target.read_bytes() == data, f'Concurrent edit: {name}'
        target.write_bytes(prefix + data)
        published = target.read_bytes()
        assert published[len(prefix):] == data
        assert backup.read_bytes() == data
        entries.append({'file': name, 'previous_sha256': digest(data),
                        'previous_bytes': len(data), 'prefix_bytes': len(prefix),
                        'published_sha256': digest(published),
                        'preserved_suffix_exact': True,
                        'prior_copy': backup.relative_to(ROOT).as_posix()})
    report = ROOT / 'CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md'
    receipt = {'complete': True, 'created_utc': now,
               'status': 'quality_stopped_200_not_qualified',
               'report_sha256': digest(report.read_bytes()),
               'visual_review_sha256': digest((SHEETS / 'visual_review_v1.json').read_bytes()),
               'documents': entries, 'training_or_VM_calls': 0}
    write_new(PUBLICATION / 'receipt.json',
              (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'documents_updated': len(entries),
                      'prior_bytes_preserved': True, 'pages_reviewed': 52,
                      'model_qualified': False}))


if __name__ == '__main__':
    main()
