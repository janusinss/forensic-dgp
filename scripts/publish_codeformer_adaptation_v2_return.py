"""Publish root-reviewed V2 evidence without altering frozen recipes or prior bytes."""
import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_adaptation_v2_return_review'
SHEETS = REVIEW / 'visual_sheets'
PUBLICATION = REVIEW / 'return_publication_v1'
FILES = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
         'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
         'CCTV_CODEFORMER_ADAPTATION_V2_PREPARATION.md',
         'CCTV_CODEFORMER_ADAPTATION_V2_VM.md')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def page_note(number):
    if number <= 6:
        return 'HQ blur: V2 often reduces original texture/accessory artifacts and is less bright than V1; softening and incorrect eye/expression/detail remain. Pixel and visual results are mixed.'
    if number == 7:
        return 'Last HQ blur rows: reduced false texture but persistent smoothing/feature differences. First clear rows change less; V2 still alters tone.'
    if number <= 13:
        return 'HQ clear: mostly similar broad layout, with brighter tone and some smoothing/detail changes. Page10 includes failed27440; likeness alone does not waive the measured individual bound.'
    if number == 14:
        return 'Last clear rows then compound: V2 reduces many original severe artifacts and is darker than V1; target features/exposure remain imperfect.'
    if number <= 19:
        return 'HQ compound: many original false textures/accessories reduce. V2 less bright than V1, but some eyes/expression/jaw/hair still differ; 21514 is less altered than V1 without establishing identity fidelity.'
    if number == 20:
        return 'Last compound rows show improved visibility versus original but residual feature/smoothing differences; first JPEG row changes subtly.'
    if number <= 26:
        return 'HQ JPEG: broad layout largely retained; V2 changes are subtler than severe cases. Tone and some detail shift; the original is already useful on many rows.'
    if number == 27:
        return 'Last JPEG rows then light: slight changes on mild inputs; brighter/smoother V2 does not consistently improve delivered fidelity.'
    if number <= 33:
        return 'HQ light: faces mostly resemble original, with some brighter tone and reduced texture/detail. Clearer appearance and paired error are distinct assessments.'
    if number == 34:
        return 'Last light then lowlight: V2 improves visibility and removes several original severe texture artifacts; V1 is brighter, V2 less so. Neither state guarantees target appearance.'
    if number <= 40:
        return 'HQ lowlight: many false original textures reduce and visibility improves; V2 generally darker than V1. Some eyes, expression and fine detail remain altered or too smooth.'
    if number <= 43:
        return 'ChokePoint unpaired: V2 often close to original CodeFormer; V1 usually softer. No consistent native improvement established; preserve glasses, hair and expression.'
    if number <= 46:
        return 'QMUL-SurvFace unpaired: V2 less washed out than V1 in several rows, but many tiny/turned/obstructed/dark inputs retain false features, noise or blank facial regions. Insufficient-information handling required.'
    if number == 47:
        return 'Old paired sourceAsian blur/clear: V2 generally less bright than V1 but fails04382/09306clear and alters appearance. Failure rows additionally inspected at256. Source identifiers do not infer ethnicity.'
    if number == 48:
        return 'Old paired sourceAsian dark cases: V2 often reduces original texture but severe01005compound remains implausible; some false glasses and changed eyes/mouth persist. Brightness alone is not supported reconstruction.'
    if number == 49:
        return 'Old paired sourceAsian motion: V2 less bright than V1 but remaining face shape/expression/tone changes prevent general preservation claims.'
    if number == 50:
        return 'Old paired thumbnails blur/clear: V2 reduces several original textures, with less brightening than V1 and remaining facial detail errors. Low-resolution targets are separate from HQ validation.'
    if number == 51:
        return 'Old paired thumbnails compound/lowlight: severe artifacts reduce but some expression/eyes/hair/jaw differences persist. V2 darker than V1 and still not qualified.'
    return 'Old paired thumbnails motion: V2 often between original texture and V1 smoothing/brightness; lips, nose and other fine detail still differ from targets.'


def main(resume=False):
    assert not (PUBLICATION / 'receipt.json').exists(), 'Preserve completed publication'
    assert not PUBLICATION.exists() or (resume and (PUBLICATION / 'first_publication_failure.json').is_file()), 'Explicit retained-failure resume required'
    transport = json.loads((REVIEW / 'archive_import_v1.json').read_text())
    metrics = json.loads((REVIEW / 'metrics_v1/metrics_audit.json').read_text())
    trajectory = json.loads((REVIEW / 'trajectory_v2/trajectory_audit.json').read_text())
    assert transport['complete'] and metrics['complete']
    assert trajectory['audit_finished'] and trajectory['evidence_consistency_pass']
    assert trajectory['completed_updates'] == 50 and not trajectory['model_qualified']
    snapshot = next(r for r in metrics['snapshot_summaries'] if r['snapshot'] == 50)
    assert not snapshot['independent_clear_guard']['pass']
    assert not snapshot['independent_blur_guard']['pass']
    manifest_bytes = (SHEETS / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest['complete'] and len(manifest['pages']) == 52
    pages, cases = [], []
    for number, page in enumerate(manifest['pages'], 1):
        assert page['file'] == f'comparison_{number:02d}.png'
        assert digest((SHEETS / page['file']).read_bytes()) == page['sha256']
        cases.extend(page['case_ids'])
        pages.append({**page, 'opened_by_root_reviewer': True,
                      'observations': page_note(number)})
    assert len(cases) == len(set(cases)) == 406
    failure = json.loads((SHEETS / 'clear_failures_256.json').read_text())
    assert failure['complete'] and failure['resampling'] == 'none'
    assert digest((SHEETS / 'clear_failures_256.png').read_bytes()) == failure['sha256']
    prefix = (
        '# CodeFormer V2 return audited: quality-stopped at 50 updates — 15 October 2026\n\n'
        '**CURRENT: 50/600 updates; real weight changes; declared quality checks failed.**\n'
        'The downloaded archive, training state and every returned RAW/PNG output were\n'
        'independently checked. All 52 sheets / 406 cases and the three failed clear\n'
        'controls at full 256 pixels were reviewed. V2 has useful-looking improvements\n'
        'and higher paired SSIM on 50/53 HQ blur cases, but is not an accepted app model.\n\n'
        '[V2 results, causes and limitations](CCTV_CODEFORMER_ADAPTATION_V2_RETURN_REVIEW.md).\n'
        '[All comparison pages](outputs/codeformer_adaptation_v2_return_review/visual_sheets/INDEX.md).\n'
        '[Latest brightness preference](CCTV_CODEFORMER_ADAPTATION_V2_USER_REVIEW_20261015.md).\n\n'
        'The user accepts moderate brightening when it makes a dark face easier to see.\n'
        'Evaluate that benefit separately from washed-out detail and altered visible\n'
        'features; brightness change alone is not a verdict on usefulness. All three\n'
        'failed clear faces also lose SSIM. Preserve\n'
        'the historical V2 clear and normal-blur photometric failures unchanged. Any\n'
        'future criteria must be declared prospectively, not fitted to those failures.\n\n'
        'All 350 trainable tensors changed; 165 frozen tensors stayed exact. V2 reached\n'
        '200 TRAIN exposures, not 600 updates or epochs. V1 u200 versus V2 u50 brightness\n'
        'differences are descriptive, with unequal budgets and jointly changed recipes.\n'
        'Both stopped states, original checkpoints, splits and gate failures remain\n'
        'protected. Do not relaunch V1/V2 unchanged or append epochs after these stops.\n\n'
        'Next is a bounded zero-optimizer VM diagnostic of training/evaluation precision,\n'
        'photometry and loss paths before choosing further training. Its design review\n'
        'is saved under outputs/codeformer_adaptation_v2_return_review/next_diagnostic_review_v1.md;\n'
        'it is not an execution-ready packet. Actual neural/gradient diagnostic work and\n'
        'all training remain human-launched on L4 in tmux. No new VM run was launched.\n\n'
        'Restoration comes first. Detector/completion development remains deferred until\n'
        'user-accepted restoration. Native CCTV remains unpaired and unqualified; final\n'
        'identities, app adoption/flow and independent final review remain outstanding.\n'
        'The full thesis goal stays active. Earlier PREPARED/UNRUN commands below are\n'
        'retained historical evidence, superseded by this current return status.\n\n---\n\n'
    ).encode('utf-8')
    originals = {}
    for name in FILES:
        target = ROOT / name
        backup = PUBLICATION / 'prior_documents' / name
        current = target.read_bytes()
        prior = backup.read_bytes() if backup.exists() else current
        assert current in (prior, prefix + prior), f'Unexpected concurrent edit: {name}'
        assert not prior.startswith(prefix.split(b'\n')[0]), f'Missing valid prior copy: {name}'
        originals[name] = prior
    now = datetime.now(timezone.utc).isoformat()
    visual = {'complete_visual_coverage': True, 'created_utc': now,
              'reviewer': 'root AI agent; descriptive, not blinded human acceptance',
              'pages_reviewed': 52, 'unique_cases_reviewed': 406,
              'distinct_snapshot_outputs_represented': 1218,
              'all_pages_opened_in_view_image': True,
              'review_view': 'delivered PNG; sheet tiles 192, original256 retained',
              'RAW_boundary': 'Separately audited arrays and exact PNG conversion; not shown by sheets',
              'manifest_sha256': digest(manifest_bytes),
              'full256_failed_clear_cases_additionally_opened': failure['case_ids'],
              'full256_failed_clear_sheet_sha256': failure['sha256'],
              'no_extra_exposure_correction_or_sharpening': True,
              'user_review': 'Moderate helpful brightness acceptable; complete V2 acceptance pending',
              'native_CCTV_qualified': False, 'model_qualified': False,
              'independent_final_evaluation': False, 'pages': pages}
    visual_path = SHEETS / 'visual_review_v1.json'
    if visual_path.exists():
        assert resume, 'Preserve existing visual review'
        previous_visual = json.loads(visual_path.read_text())
        assert previous_visual['pages'] == pages
        assert previous_visual['manifest_sha256'] == visual['manifest_sha256']
        assert previous_visual['full256_failed_clear_sheet_sha256'] == failure['sha256']
    else:
        write_new(visual_path, (json.dumps(visual, indent=2, ensure_ascii=False) + '\n').encode())
    # Retain every old document before any further publication mutation.
    for name, data in originals.items():
        backup = PUBLICATION / 'prior_documents' / name
        if not backup.exists():
            write_new(backup, data)
        assert backup.read_bytes() == data
    entries = []
    for name, data in originals.items():
        backup = PUBLICATION / 'prior_documents' / name
        target = ROOT / name
        current = target.read_bytes()
        assert current in (data, prefix + data), f'Concurrent edit: {name}'
        if current == data:
            staged = PUBLICATION / 'staged_documents' / name
            assert staged.resolve().is_relative_to(ROOT.resolve())
            assert target.resolve() == ROOT.resolve() / name
            assert not target.is_symlink() and target.is_file()
            write_new(staged, prefix + data)
            assert staged.read_bytes() == prefix + data
            assert target.read_bytes() == data, f'Concurrent edit before replace: {name}'
            staged.replace(target)
        assert target.read_bytes()[len(prefix):] == data == backup.read_bytes()
        entries.append({'file': name, 'previous_sha256': digest(data),
                        'previous_bytes': len(data), 'prefix_bytes': len(prefix),
                        'published_sha256': digest(target.read_bytes()),
                        'preserved_suffix_exact': True,
                        'prior_copy': backup.relative_to(ROOT).as_posix()})
    evidence = ['CCTV_CODEFORMER_ADAPTATION_V2_RETURN_REVIEW.md',
                'CCTV_CODEFORMER_ADAPTATION_V2_USER_REVIEW_20261015.md']
    evidence += [str(p.relative_to(ROOT)).replace('\\', '/') for p in [
        REVIEW / 'archive_import_v1.json', REVIEW / 'metrics_v1/metrics_audit.json',
        REVIEW / 'trajectory_v2/trajectory_audit.json',
        REVIEW / 'trajectory_v2/SUMMARY_AND_REPAIR_RECEIPT.json',
        REVIEW / 'independent_status_readback_v1.json',
        REVIEW / 'metrics_supplement_v1/receipt.json',
        REVIEW / 'metrics_supplement_v1/INTERPRETATION.md',
        REVIEW / 'next_diagnostic_review_v1.md', SHEETS / 'visual_review_v1.json']]
    receipt = {'complete': True, 'created_utc': now,
               'status': 'quality_stopped_50_not_qualified', 'documents': entries,
               'evidence': {path: digest((ROOT / path).read_bytes()) for path in evidence},
               'audit_CPU_tests': {'passed': 41, 'runner': 'python -m unittest',
                                   'neural_forwards': 0, 'gradient_queries': 0},
               'supplement_CPU_fixtures_passed': 7,
               'resumed_after_retained_document_write_failure': resume,
               'training_or_VM_calls': 0, 'model_qualified': False}
    write_new(PUBLICATION / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'documents_updated': len(entries),
                      'prior_bytes_preserved': True, 'pages_reviewed': 52,
                      'full256_failed_cases': 3, 'model_qualified': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-publication', action='store_true')
    main(parser.parse_args().resume_publication)
