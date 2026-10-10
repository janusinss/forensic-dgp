"""Combine separately audited diagnostic evidence; no training or new model call."""
from pathlib import Path
import hashlib
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cctv_dgp_actual_step_review_v1_contract import NAME, PROPOSALS, ROLES, read, write, sha, output_prefix

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs' / NAME
ORIGINAL_RETURN = ROOT / 'outputs' / (NAME + '_return')
TAIL_RETURN = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_vm_return'
PARTIAL = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_analysis'
OUT = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_analysis'


def pixels(path):
    with Image.open(path) as image:
        assert image.mode == 'RGB' and image.size == (256, 256)
        return np.asarray(image).copy()


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve every previous analysis'
    p = read(BUNDLE / 'protocol.json')
    prospective_path = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_analysis/prospective_review.json'
    prospective = read(prospective_path)
    partial_audit_path = ROOT / 'outputs/cctv_dgp_actual_step_review_v1_partial_audit/independent_audit.json'
    tail_audit_path = ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_independent_audit.json'
    partial_audit, tail_audit = read(partial_audit_path), read(tail_audit_path)
    partial = read(PARTIAL / 'analysis.json')
    partial_visual = read(PARTIAL / 'visual_review.json')
    partial_pixel_audit = read(PARTIAL / 'independent_arithmetic_and_sheets_audit.json')
    assert partial_audit['complete'] and partial_audit['all3045_raw_PNG_and_mean_only_outputs_checked']
    assert tail_audit['complete'] and tail_audit['all315_outputs_checked']
    assert tail_audit['overlap_files_independently_checked'] == 828
    assert tail_audit['every_original_stored_row_and_categorical_gate_exact']
    assert partial_visual['complete'] and partial_pixel_audit['complete']
    assert partial_pixel_audit['analysis_sha256'] == sha(PARTIAL / 'analysis.json')
    assert partial['protocol_sha256'] == tail_audit['parent_protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    assert partial_audit['CPU_replay']['cases'] == 145 and tail_audit['CPU_replay']['cases'] == 15
    selected = prospective['visual_rows'][-25:]
    assert len(selected) == 25 and all(row['update'] == 45 for row in selected)
    assert prospective['visual_rows'][:225] == read(PARTIAL / 'prospective_review.json')['visual_rows']
    OUT.mkdir()
    plan = {'complete': True, 'parent_protocol_sha256': sha(BUNDLE / 'protocol.json'),
            'tail_protocol_sha256': tail_audit['protocol_sha256'],
            'original_prospective_plan_sha256': sha(prospective_path),
            'partial_scientific_audit_sha256': sha(partial_audit_path),
            'tail_scientific_audit_sha256': sha(tail_audit_path),
            'visual_rows': selected, 'columns': prospective['columns'],
            'sheets': 5, 'exact256_pixel_cells': 150,
            'selection': 'The unchanged last25 rows of the original frozen250-row plan; no outcome-based additions',
            'no_new_case_or_probe_selection': True, 'paired_photographic_TRAIN_diagnostic_only': True,
            'not_native_CCTV_or_final_review': True, 'original_storage_failure_retained': True}
    write(OUT / 'prospective_review.json', plan)
    imports = {ORIGINAL_RETURN: read(ROOT / 'outputs/cctv_dgp_actual_step_review_v1_return_import.json'),
               TAIL_RETURN: read(ROOT / 'outputs/cctv_dgp_actual_step_tail_v1_return_import.json')}
    bindings = {}

    def bind(path):
        name = path.relative_to(ROOT).as_posix()
        bindings[name] = sha(path)
        return bindings[name]

    for path in [prospective_path, partial_audit_path, tail_audit_path, PARTIAL / 'analysis.json',
                 PARTIAL / 'visual_review.json', PARTIAL / 'independent_arithmetic_and_sheets_audit.json',
                 OUT / 'prospective_review.json', BUNDLE / 'protocol.json']:
        bind(path)
    receipts, rows, terms, logical_conditions = {}, [], [], []
    for probe in p['probes']:
        for proposal in PROPOSALS:
            source = TAIL_RETURN if (probe['update'], proposal) == (45, 'cone') else ORIGINAL_RETURN
            path = source / output_prefix(probe['update'], proposal) / 'metrics.json'
            assert bind(path) == imports[source]['files_sha256'][path.relative_to(source).as_posix()]
            receipt = read(path)
            assert receipt['complete'] and receipt['update'] == probe['update'] and receipt['proposal'] == proposal
            receipts[(probe['update'], proposal)] = receipt
            logical_conditions.append({'update': probe['update'], 'proposal': proposal,
                                       'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        baseline = receipts[(probe['update'], 'zero')]
        for proposal in PROPOSALS[1:]:
            candidate = receipts[(probe['update'], proposal)]
            delta = np.array(candidate['objective_term_means']['current_batch']) - np.array(baseline['objective_term_means']['current_batch'])
            terms.append({'update': probe['update'], 'proposal': proposal,
                          'current_batch_raw_objective_term_changes': delta.tolist(),
                          'sum_of_seven_term_changes': float(delta.sum()), 'landmark_term_nonincrease': bool(delta[0] <= 0)})
            for role in ROLES:
                for stage in ('raw', 'PNG'):
                    rows.append({'update': probe['update'], 'proposal': proposal, 'role': role, 'stage': stage,
                                 'cases': candidate['groups'][role][stage]['all']['cases'],
                                 **candidate['comparison_to_same_before_state'][role][stage]})
    assert len(rows) == 120 and len(terms) == 20 and len(logical_conditions) == 30
    assert [row for row in rows if not (row['update'] == 45 and row['proposal'] == 'cone')] == partial['all114_role_stage_state_proposal_rows']
    assert [row for row in terms if not (row['update'] == 45 and row['proposal'] == 'cone')] == partial['all19_current_batch_raw_term_changes']
    summaries = []
    for proposal in PROPOSALS[1:]:
        for role in ROLES:
            for stage in ('raw', 'PNG'):
                subset = [row for row in rows if (row['proposal'], row['role'], row['stage']) == (proposal, role, stage)]
                gains = np.array([row['relative_feature_gain'] for row in subset])
                assert len(subset) == 10
                summaries.append({'proposal': proposal, 'role': role, 'stage': stage, 'states': 10,
                                  'finite_preservation_pass_updates': [row['update'] for row in subset if row['finite_preservation_pass']],
                                  'finite_preservation_failure_updates': [row['update'] for row in subset if not row['finite_preservation_pass']],
                                  'positive_feature_gain_updates': [row['update'] for row in subset if row['relative_feature_gain'] > 0],
                                  'feature_gain_mean': float(gains.mean()), 'feature_gain_minimum': float(gains.min()),
                                  'feature_gain_maximum': float(gains.max()),
                                  'repeated_case_diagnostic_summary_not_population_estimate': True})
    (OUT / 'sheets').mkdir()
    cases = {case['id']: case for case in p['cases']}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    sheets = []
    for begin in range(0, 25, 5):
        chosen = selected[begin:begin + 5]
        canvas = Image.new('RGB', (1816, 1504), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), 'Final-state paired photographic TRAIN diagnostic; exact256-pixel cells; not native CCTV or final evidence', font=font, fill='black')
        for index, label in enumerate(plan['columns']):
            draw.text((280 + 256 * index + 3, 30), label, font=font, fill='black')
        cells = []
        for slot, row in enumerate(chosen):
            assert time.monotonic() - started < 300
            case = cases[row['id']]
            camera = pixels(BUNDLE / case['input'])
            target = pixels(BUNDLE / case['target'])
            original_path = TAIL_RETURN / output_prefix(45, 'zero') / row['role'] / (row['id'] + '.npz')
            assert bind(original_path) == imports[TAIL_RETURN]['files_sha256'][original_path.relative_to(TAIL_RETURN).as_posix()]
            with np.load(original_path, allow_pickle=False) as data:
                original = np.floor(data['original_rgb'] * np.float32(255)).astype(np.uint8)
            with Image.open(BUNDLE / case['observed']) as image:
                mask = np.asarray(image.convert('L')) > 0
            original[~mask] = camera[~mask]
            images = [camera, target, original]
            for relative in (case['input'], case['target'], case['observed']):
                bind(BUNDLE / relative)
            for proposal in PROPOSALS:
                path = TAIL_RETURN / output_prefix(45, proposal) / row['role'] / (row['id'] + '.png')
                assert bind(path) == imports[TAIL_RETURN]['files_sha256'][path.relative_to(TAIL_RETURN).as_posix()]
                images.append(pixels(path))
            yy = 64 + 288 * slot
            draw.text((8, yy + 5), f"update45 / {row['role']}\n{row['id']}\n{row['source']}\n{row['profile']}", font=font, fill='black')
            for column, image in enumerate(images):
                box = (280 + 256 * column, yy, 280 + 256 * (column + 1), yy + 256)
                canvas.paste(Image.fromarray(image), (box[0], box[1]))
                assert np.array_equal(np.asarray(canvas.crop(box)), image)
                cells.append({'row': row, 'column': plan['columns'][column], 'box': list(box),
                              'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
        path = OUT / f'sheets/page{begin // 5 + 46:03d}.png'
        canvas.save(path)
        sheets.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'cells': cells, 'actually_viewed': False})
    assert len(sheets) == 5 and sum(len(sheet['cells']) for sheet in sheets) == 150
    report = {'complete': True, 'parent_protocol_sha256': plan['parent_protocol_sha256'],
              'tail_protocol_sha256': plan['tail_protocol_sha256'], 'logical_conditions': logical_conditions,
              'all120_role_stage_state_proposal_rows': rows, 'all20_current_batch_raw_term_changes': terms,
              'summaries': summaries, 'sheets': sheets, 'source_evidence_sha256': bindings,
              'newly_audited_tail_slots': 315, 'complete_control_slots_replayed_exactly': 210,
              'original_complete_slots': 3045, 'unique_completed_diagnostic_slots': 3150,
              'original_run_remains_storage_stopped': True, 'tail_completed_separately': True,
              'full_prospective_diagnostic_coverage_completed_across_two_retained_runs': True,
              'original_full_checker_not_relabelled_as_pass': True,
              'CPU_replays_executed_across_both_audits': 160,
              'unique_current_batch_proposal_replays': 150,
              'previous_visual_sheets_reviewed': 45, 'new_visual_sheets_pending': 5,
              'paired_photographic_TRAIN_diagnostic_only': True, 'visual_review_complete': False,
              'independent_final_review': False, 'full_TRAIN_capacity_pass': False,
              'native_or_DEV_or_reserved_final_used': False, 'app_promotion': False,
              'optimizer_updates': 0, 'gradient_queries': 0, 'backward_calls': 0,
              'model_forwards_in_this_analysis': 0, 'new_trained_checkpoint': False,
              'checker_sha256': sha(Path(__file__)), 'seconds': time.monotonic() - started, 'cap_seconds': 300}
    write(OUT / 'analysis.json', report)
    print({'complete': True, 'conditions': 30, 'unique_slots': 3150,
           'new_sheets': 5, 'new_exact_cells': 150, 'actually_viewed': False}, flush=True)


if __name__ == '__main__':
    main()
