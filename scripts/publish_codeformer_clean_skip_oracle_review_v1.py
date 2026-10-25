"""Publish actual audited clean-SFT diagnostic and root decision; no inference.

Requires --publish. Exclusively backs up four full prior documents and retains
their exact bytes as suffixes. Never promotes a model or launches training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_clean_skip_oracle_local_20261019_v1'
REVIEW = OUT / 'review_v1'
PUB = REVIEW / 'publication_v1'
REPORT = ROOT / 'CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md'
DOCS = ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
MANIFEST = ROOT / 'outputs/codeformer_hq_code_oracle_inputs_20261019_v1/manifest_v1.json'
MANIFEST_SHA = 'cdb70da025284909c96144fd1884464cf89fea6c7f4167e9c5d99bd834dbc2ed'
DESIGN = ROOT / 'outputs/codeformer_clean_skip_design_20261019_v1/design_frozen_v1.json'
R2_STATE = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'


def need(value, message):
    if not value:
        raise ValueError(message)


def safe(path, exists=True):
    path = Path(path).absolute()
    need(path.resolve(strict=exists).is_relative_to(ROOT), 'Workspace path required')
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            row = part.lstat()
            need(not stat.S_ISLNK(row.st_mode) and not getattr(row, 'st_file_attributes', 0) & 1024,
                 'Linked/reparse publication path refused')
    if exists:
        row = path.stat(); need(stat.S_ISREG(row.st_mode) and row.st_nlink == 1, 'Unique ordinary evidence required')
    return path


def relative(path):
    return str(Path(path).absolute().relative_to(ROOT)).replace('\\', '/')


def pin(path):
    path = safe(path); before = path.stat()
    with path.open('rb') as stream: value = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns),
         'Evidence changed during publication read')
    return {'bytes': before.st_size, 'sha256': value}


def load(path):
    path = safe(path); need(path.stat().st_size <= 4 * 1024**2, 'Bounded review metadata required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'Duplicate review field refused'); result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite metadata refused')))


def new(path, payload):
    path = safe(path, exists=False); need(not path.exists(), 'Preserve previous/partial publication')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def journal(value):
    path = PUB / 'journal.jsonl'
    with path.open('ab') as stream:
        stream.write((json.dumps(value, allow_nan=False) + '\n').encode()); stream.flush(); os.fsync(stream.fileno())


def exact_text(value, field):
    need(isinstance(value, str) and value.strip() and len(value) <= 6000 and '\x00' not in value,
         'Root visual review requires exact nonempty ' + field)
    return value.strip()


def evidence():
    paths = {'result': OUT / 'result_v1.json', 'audit': REVIEW / 'audit_v1.json',
             'visual': REVIEW / 'root_visual_review_v1.json', 'sheets': REVIEW / 'visual_manifest_v1.json',
             'design': DESIGN, 'input_manifest': MANIFEST}
    pins = {name: pin(path) for name, path in paths.items()}
    r, a, v, sheets, design, manifest = (load(paths[name]) for name in ('result', 'audit', 'visual', 'sheets', 'design', 'input_manifest'))
    need(r['complete'] is True and r['neural_forward_attempts'] == r['neural_forward_returns'] == 42 and
         r['clear_capture_forwards'] == 6 and r['paired_forwards'] == 36 and r['SFT_invocations'] == 168 and
         all(r[key] == 0 for key in ('optimizer_updates', 'gradient_queries', 'backward_calls', 'VM_calls')),
         'Exact actual42-call zero-update diagnostic required')
    need(r['model_state_before'] == r['model_state_after'] == R2_STATE and r['hooks_restored'] is True and
         all(r[key] is False for key in ('training_admitted', 'model_qualified', 'app_default_changed', 'FINAL_used', 'native_CCTV_used')),
         'Frozen inference research boundary changed')
    need(a['complete'] is True and a['audit_pass'] is True and a['actual_model_calls'] == 42 and
         a['actual_journal_attempts_returns'] == [42, 42] and a['actual_SFT_invocations'] == 168 and
         a['source_result'] == {'path': relative(paths['result']), **pins['result']} and
         a['saved_output_file_pins_verified'] == 277 and a['all42_RAW_finite_float32_and_PNG_floor_exact'] is True and
         a['all36_saved_RAW_PNG_MAE_recomputed'] is True and len(a['all6_clear_capture_natural_oracle_parity']) == 6 and
         all(row['capture_natural_oracle_RAW_PNG_upstream_exact'] is True for row in a['all6_clear_capture_natural_oracle_parity']) and
         len(a['all18_upstream4arrays_exact']) == 18 and
         all(row['all4_actual_upstream_arrays_exact'] is True for row in a['all18_upstream4arrays_exact']) and
         len(a['all42_source_bound_SFT_records_checked']) == 42 and a['decoder_first32_input_pair_exact'] is True,
         'Actual independent saved-output audit required')
    need(all(a[key] is False for key in ('model_qualified', 'training_admitted', 'app_default_changed', 'FINAL_used', 'native_CCTV_used')) and
         a['visual_manifest'] == {'path': relative(paths['sheets']), **pins['sheets']}, 'Independent audit scope/visual binding differs')
    need(v['complete'] is True and v['root_personal_review_complete'] is True and v['sheets_reviewed'] == 6 and
         v['all18_cases_personally_viewed'] is True and v['audit_sha256'] == pins['audit']['sha256'] and
         v['visual_manifest_sha256'] == pins['sheets']['sha256'], 'Actual root personal review of all six sheets required')
    conclusion = exact_text(v.get('conclusion'), 'conclusion')
    decision = exact_text(v.get('next_action'), 'next_action')
    need(sheets['complete'] is True and sheets['count'] == 6 and len(sheets['sheets']) == len(v['sheets']) == 6,
         'Exact six-sheet manifests required')
    by_path = {row['path']: row for row in sheets['sheets']}
    need(len(by_path) == 6 and {row['path'] for row in v['sheets']} == set(by_path), 'Root review must cover all six exact sheets')
    for row in v['sheets']:
        expected = {key: row[key] for key in ('bytes', 'sha256')}
        need(pin(ROOT / row['path']) == expected == {key: by_path[row['path']][key] for key in ('bytes', 'sha256')},
             'Personally reviewed sheet changed')
        exact_text(row.get('observations'), 'per-sheet observations')
    need(pins['input_manifest']['sha256'] == MANIFEST_SHA and len(manifest['cases']) == 18 and len(manifest['targets']) == 6,
         'Exact six-source18-input set required')
    need(design['census']['total_top_level_model_calls'] == 42 and design['census']['optimizer_updates'] == 0,
         'Frozen42-call diagnostic design differs')
    need(a['degraded_branch_full_arrays_saved'] is False, 'Preserve observed-feature evidence limitation')
    for name, value in r['source_pins'].items():
        need(pin(ROOT / name) == value, 'Diagnostic source/input changed')
    need(r['source_pins'] == a['source_pins'], 'Audit must bind actual producer source/input set')
    all_pins = {relative(path): pins[name] for name, path in paths.items()}
    all_pins.update(r['source_pins'])
    for row in sheets['sheets']:
        all_pins[row['path']] = {key: row[key] for key in ('bytes', 'sha256')}
    return r, a, v, all_pins, conclusion, decision


def text(r, a, v, conclusion, decision):
    summary = a['profile_readability_summaries']; lines = []
    for key, label, count in (('clear', 'Clear', 6), ('moderate_existing_gaussian_2p4', 'Moderate Gaussian blur', 6),
                             ('strong_fixed_blur3p2_lr64_jpeg50', 'Strong synthetic blur', 6),
                             ('readable4_strong', 'Predeclared readable strong', 4)):
        row = summary[key]; need(row['cases'] == count, 'Exact reported profile/readability count required')
        mae = row['views']['PNG']['MAE']; ssim = row['views']['PNG']['SSIM']
        lines.append(f"| {label} ({count}) | {mae['natural_mean']:.6f} | {mae['oracle_mean']:.6f} | {ssim['natural_mean']:.6f} | {ssim['oracle_mean']:.6f} |")
    need(summary['readable16_TRAIN']['cases'] == 16 and summary['borderline2_stress']['cases'] == 2, 'Keep16readable/2stress separation')
    table = '\n'.join(lines)
    report = f'''# Clean-SFT-input oracle diagnostic and root review

19 October 2026. This completed diagnostic changes only the observed encoder features supplied to R2's existing SFT blocks. It creates no trained model, deployable clean-reference method or accepted replacement. The current experimental restoration checkpoints and their earlier quality failures remain preserved.

## Actual execution

The local CPU FP32 worker completed in **{r['seconds']:.2f} seconds**, with **42 top-level R2 calls**: six instrumented clear-feature capture forwards,18 natural and18 clean-SFT-oracle forwards. All168 SFT invocations were recorded. Optimizer updates, gradient queries, backward calls and VM calls were zero. Peak sampled RSS was **{r['resources']['maximum_RSS_bytes']:,} bytes**; minimum sampled free disk was **{r['resources']['minimum_free_disk_bytes']:,} bytes**. The worker/supervisor caps were900/960 seconds, with10GiB RSS and4GiB ongoing disk reserve. The sampled values do not constitute continuous peak measurement.

The frozen method note originally called the capture forwards “unwrapped.” Actual capture forwards carried read-only code lookup, SFT and generator-input instrumentation. They are **instrumented capture references**, not an additional uninstrumented parity experiment. All six capture/natural/clean-self-substitution RAW/PNG and upstream arrays matched exactly. No extra unwrapped call was run or counted.

The six fixed historical TRAIN photographs provide clear, existing Gaussian2.4 and fixed Gaussian3.2/LR64/JPEG50 observations. Root reviewed inputs before outputs:16 readable and two borderline strong stress observations. The four readable strong cases are reported separately from those two stress cases. These are six repeated photographs, not18 independent identities. The strong profile is a synthetic proxy, not calibrated CCTV or unknown-origin wild-image ground truth. No native CCTV, DEV or reserved FINAL pixels entered the diagnostic.

## What changed and what stayed fixed

Each source's existing clear256 input, prepared through the same bilinear256→512 path, supplied four frozen R2 encoder features at32/64/128/256. The oracle substituted those clean-observation features only as the first argument to each SFT block. Natural selected codes, logits, LQ features, post-AdaIN input and every original R2 weight remained fixed. The first32-site decoder input matched between arms; later decoder inputs may change causally after an earlier substitution. Fidelity remained1 and AdaIN stayed on, with no display enhancement or brightness postprocessing.

The independent audit verified277 output file pins,84 call records,24 finite FP32 captured-clean arrays and all42 SFT records. Used oracle feature hashes matched the physically saved clean arrays. Observed degraded feature evidence is limited to **source-bound actual hook hashes**, not saved full degraded feature tensors or an independent neural replay. The paired logits/LQ/codes/post-AdaIN arrays were independently compared exactly. All42 RAW arrays and floored PNGs were checked, and all36 arm-pair MAEs were recomputed. Additional RGB SSIM and adjacent residual-gradient error are descriptive measurements; they are not the old DGP structure gate or identity scores.

## Paired synthetic measurements

| Profile (cases) | R2 PNG MAE | Oracle PNG MAE | R2 PNG SSIM | Oracle PNG SSIM |
|---|---:|---:|---:|---:|
{table}

RAW, all16-readable and two-borderline summaries remain in the independent audit. These are repeated TRAIN diagnostic measurements, not final evaluation or native CCTV performance.

## Root visual conclusion

{conclusion}

## Next action recorded by root

{decision}

A favorable clean-reference intervention would establish conditional feature-route headroom, not whether a small adapter can infer those unavailable features from degraded input. The current18 observations include no dark-camera condition; retaining logits/AdaIN does not itself prove preservation of useful R2 brightening. Any later learned adapter still requires reviewed TRAIN corruption/clear/dark controls, exact initial parity, real gradient-path checks, measured resources, a distinct finite manual L4/tmux protocol and unchanged historical preservation gates. No next training packet or optimizer run is authorized automatically by this publication.

All original weights, failed experiments, source splits, provenance, research caches and migration backups remain. Native CCTV stays source-separated and unpaired; paired synthetic scores remain separate. No ethnicity, Zamboanga performance or exact-identity claim follows. Main app/defaults are unchanged, restoration acceptance and independent final review remain outstanding, and completion development remains deferred until restoration acceptance. The full goal remains active. The earlier local→VM→local storage cleanup is closed; no deletion occurred here.

Evidence: [actual result](outputs/codeformer_clean_skip_oracle_local_20261019_v1/result_v1.json), [independent audit](outputs/codeformer_clean_skip_oracle_local_20261019_v1/review_v1/audit_v1.json), [all six sheets](outputs/codeformer_clean_skip_oracle_local_20261019_v1/review_v1/visual_manifest_v1.json), [root visual decision](outputs/codeformer_clean_skip_oracle_local_20261019_v1/review_v1/root_visual_review_v1.json), [prospective method](outputs/codeformer_clean_skip_design_20261019_v1/METHOD_NOTE.md).
'''
    header = f'''# Clean-SFT feature diagnosis reviewed; restoration acceptance pending

19 October 2026. A local inference-only diagnostic completed42 R2 calls with zero optimizer updates, and its saved arrays plus all six comparison sheets were independently audited and personally reviewed. The oracle uses unavailable clean-reference SFT features; it creates no accepted or deployable model.

Root conclusion: {conclusion}

Next action recorded by root: {decision}

[CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md](CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md) records the actual measurements and limits. Original checkpoints, quality failures, native/unpaired source boundaries and FINAL remain preserved. Future actual training stays manual and finite on the L4 VM. App defaults remain unchanged, completion stays deferred, and the full goal remains active. Earlier storage cleanup is closed.

Earlier status below remains historical evidence.

---

'''
    return report.encode('utf-8'), header.encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(); need(args.publish, 'Use explicit --publish only after peer review and completed actual evidence')
    need(not PUB.exists() and not REPORT.exists(), 'Preserve previous/partial report and publication namespace')
    r, a, v, proof, conclusion, decision = evidence(); report, header = text(r, a, v, conclusion, decision)
    originals = {name: safe(ROOT / name).read_bytes() for name in DOCS}
    before = {name: pin(ROOT / name) for name in DOCS}
    need(all(hashlib.sha256(originals[name]).hexdigest() == before[name]['sha256'] for name in DOCS), 'Document changed during initial read')
    for name, payload in originals.items():
        backup = PUB / 'source_backups' / (name + '.before'); new(backup, payload)
        need(pin(backup) == before[name], 'Full original document backup differs')
    for name, payload in originals.items(): new(PUB / (name + '.staged'), header + payload)
    need(all(pin(ROOT / name) == value for name, value in before.items()) and
         all(pin(ROOT / name) == value for name, value in proof.items()), 'Source/document changed before publication')
    new(REPORT, report); updated = {}
    for name in DOCS:
        journal({'event': 'replace_intent', 'path': name, 'before': before[name]})
        os.replace(PUB / (name + '.staged'), ROOT / name)
        need((ROOT / name).read_bytes() == header + originals[name], 'Exact prior suffix not preserved')
        updated[name] = {'before': before[name], 'after': pin(ROOT / name), 'suffix_offset_bytes': len(header),
            'backup': relative(PUB / 'source_backups' / (name + '.before'))}
        journal({'event': 'replace_complete', 'path': name, 'after': updated[name]['after']})
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(), 'evidence': proof,
        'documents': updated, 'report': {'path': relative(REPORT), **pin(REPORT)}, 'source': pin(__file__),
        'root_conclusion_copied_exact': conclusion, 'root_next_action_copied_exact': decision,
        'model_qualified': False, 'app_default_changed': False, 'new_training_launched': False, 'overall_goal_complete': False}
    new(PUB / 'publication_v1.json', (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode())
    print(json.dumps({'complete': True, 'report': relative(REPORT), 'full_backups': 4, 'documents_updated': 4,
                      'optimizer_updates': 0, 'model_qualified': False, 'overall_goal_complete': False}))


if __name__ == '__main__': main()
