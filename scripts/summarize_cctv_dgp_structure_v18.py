"""Bind completed V18 transfer/audit/development review; no neural execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
RETURN = ROOT / 'outputs/cctv_dgp_structure_return_v18'
OUT = RETURN / 'outputs/structure_v18'
AUDIT = ROOT / 'outputs/cctv_dgp_structure_v18_audit_recovery_r2'
PIN = 'e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb'
ARCHIVE = 'd494351f1f1f36d60f79f696633ff6e0dcc5cdf1f03662f84884f4aae71787e5'

OBSERVATIONS = {
    'clear_stages': 'Geometry mostly retained. Later outputs soften texture and change glasses/skin/background detail, especially tr_ffhq_00084 and tr_ffhq_01210. Clear SSIM and FFHQ-source MSE failures remain.',
    'blur_lr24_stages': 'Update600 has more coherent eyes/mouths than the retained DGP on these fitted faces. Softness, ringing, patchy colour and weak glasses remain; tr_ffhq_00084 frames/lips, tr_asian_00196 hand/cheek/eye area and tr_ffhq_00178 smile are examples.',
    'lowlight_lr32_stages': 'Brightness and fitted facial structure improve. Mottled cheeks, forehead patches, soft glasses and colour blocks remain, including tr_asian_00048, tr_asian_00196, tr_ffhq_00323 and tr_ffhq_01210.',
    'motion_lr48_stages': 'Most trained faces become more coherent with stronger eyes/mouths. Fine glasses/background appearance remains softer; tr_asian_00196 eye/hand texture and tr_ffhq_00323 forehead show patching.',
    'compound_lr24_stages': 'Largest remaining limitation: fitted structure improves but faces stay soft, with purple/red feature patches and weak glasses. tr_asian_00196 eyes/cheek, tr_ffhq_00084 glasses and tr_ffhq_01210 frames/mouth remain unreliable fine detail.',
    'clear_zero_prior': 'Normal and zero-prior outputs retain broadly similar geometry, with differences in softness/texture. Neither control removes the need for the original clear preservation guards.',
    'blur_lr24_zero_prior': 'Removing prior features makes most outputs visibly softer and reduces fitted eye/mouth structure. Normal-path detail is stronger, while glasses and some cheek/teeth details remain weak.',
    'lowlight_lr32_zero_prior': 'Zero-prior outputs lose eye/mouth detail and show large coloured cheek/mouth patches. The normal fitted path is stronger, but its own skin/hand/glasses artifacts remain.',
    'motion_lr48_zero_prior': 'Zero-prior outputs are softer around eyes, lips and smiles; normal outputs better match these trained proxies. Some glasses and facial texture remain incomplete in both paths.',
    'compound_lr24_zero_prior': 'Zero-prior produces stronger coloured patches, soft eye/mouth regions and weak glasses. Normal fitted output is more coherent, yet compound degradation is still visibly limited.'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def summarize():
    archive = ROOT / 'outputs/cctv-dgp-structure-v18-results.tar.gz'
    terminal = read(ROOT / 'outputs/supervisor_completion_v18.json')
    result = read(OUT / 'results.json')
    protocol = read(BUNDLE / 'structure_protocol_v18.json')
    require(sha(archive) == ARCHIVE == terminal['archive_sha256'] and archive.stat().st_size == terminal['bytes'] == 481723919,
            'Exact transferred archive/receipt required')
    require(Path(str(archive) + '.sha256').read_text().split() == [ARCHIVE, archive.name], 'Sidecar differs')
    require(terminal == read(RETURN / 'downloaded_export_receipt.json'), 'Preserved terminal receipt differs')
    require(sha(BUNDLE / 'structure_protocol_v18.json') == sha(RETURN / 'structure_protocol_v18.json') == PIN
            == result['protocol_sha256'] == terminal['protocol_sha256'], 'Protocol differs')
    for name, digest in protocol['assets_sha256'].items():
        require(sha(BUNDLE / name) == sha(RETURN / name) == digest, 'Frozen original/returned asset differs:' + name)
    result_sha = sha(OUT / 'results.json')
    require(result_sha == terminal['results_sha256'], 'Results differ')
    full = read(AUDIT / 'local_full_audit.json')
    completion = read(AUDIT / 'recovery_completion.json')
    require(full['complete'] and full['protocol_sha256'] == PIN and full['results_sha256'] == result_sha
            and completion['originals_unchanged'] and completion['audit_receipt_sha256'] == sha(AUDIT / 'local_full_audit.json')
            and not completion['quality_gates_changed'] and not completion['training_restarted'], 'Full audit/provenance differs')
    require(full['head_forwards'] == full['raw_PNG_metrics'] == full['cosines'] == 250
            and full['original_grid_cells'] == 550 and full['scope']['updates'] == 600
            and full['scope']['exposures'] == 6000 and not full['optimizer_updates'] and not full['backward_calls'], 'Audit scope differs')
    for name in ['training_accumulation_check.json', 'binary64_log10_checks.json']:
        require(read(AUDIT / name)['complete'], 'Numeric check incomplete')
    for key in ['teacher_used', 'validation_used', 'native_used', 'native_reserved_used', 'checkpoint_selected', 'production_promoted']:
        require(result[key] is False, 'Capacity-only scope differs:' + key)
    spec = importlib.util.spec_from_file_location('frozen_v18_capacity_contract', BUNDLE / 'cctv_dgp_structure_v18.py')
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    summaries = {}
    baseline = None
    for snapshot in result['snapshots']:
        metrics = read(OUT / snapshot['metrics'])
        if baseline is None:
            baseline = metrics['summary']
        require(q.strict_preservation(metrics['summary'], baseline) == metrics['preservation'], 'Original quality report differs')
        summaries[str(snapshot['update'])] = {'summary': metrics['summary'], 'preservation': metrics['preservation'],
            'mean_fit_stop_objective': metrics['mean_fit_stop_objective'], 'checkpoint_sha256': sha(OUT / snapshot['checkpoint']),
            'state_hash': snapshot['state_hash'], 'metrics_sha256': sha(OUT / snapshot['metrics'])}
    require(not summaries['600']['preservation']['qualified_for_separate_generalization_protocol'], 'Do not promote failed V18')
    final = read(OUT / 'update600/metrics.json')
    sheets = []
    for name in result['grids']:
        stem = Path(name).stem
        require(stem in OBSERVATIONS, 'Original sheet not reviewed')
        sheets.append({'file': str((OUT / name).resolve()), 'sha256': sha(OUT / name),
                       'original_cells': 60 if stem.endswith('_stages') else 50,
                       'reviewed_at_original_resolution': True, 'observation': OBSERVATIONS[stem]})
    require(len(sheets) == 10 and sum(s['original_cells'] for s in sheets) == 550, 'Complete original-cell review required')
    review = {'complete': True, 'date': '2026-10-05', 'reviewer': 'Codex primary agent; development review, not independent final acceptance',
        'protocol_sha256': PIN, 'results_sha256': result_sha, 'sheets': sheets,
        'scope': 'Ten training photographs/fifty paired synthetic cases only. No native CCTV, held-out validation, reserved review or covering-family completion evidence.',
        'decision': 'Capacity gain observed, unconditional clear preservation failed. Keep V18 experimental and historical failures intact.',
        'insufficient_information': 'Compound inputs expose weak visible fine structure; fitted proxy resemblance does not justify generating hidden identity or replace the clearer-crop rule.',
        'zero_prior_limit': 'Input-sensitivity control on the same fitted head; not a separately trained no-prior baseline.',
        'pretraining_overlap': 'Not excluded; prior/recognizer pretrained corpus overlap cannot be inferred from these results.',
        'source_limit': 'Report dataset/asian_faces and dataset/thumbnails128x128 separately; folder labels are not ethnicity or local CCTV performance.',
        'production_promoted': False, 'independent_final_review_complete': False}
    review_path = ROOT / 'outputs/cctv_dgp_structure_review_v18.json'
    write_new(review_path, review)
    verified = {'complete': True, 'protocol_sha256': PIN, 'archive_sha256': ARCHIVE, 'archive_bytes': archive.stat().st_size,
        'results_sha256': result_sha, 'original_supervisor_success': True, 'source_assets': len(protocol['assets_sha256']),
        'audit_receipt_sha256': sha(AUDIT / 'local_full_audit.json'), 'recovery_completion_sha256': sha(AUDIT / 'recovery_completion.json'),
        'review_receipt_sha256': sha(review_path), 'snapshots': summaries, 'zero_prior_summary': final['zero_prior_summary'],
        'VM_timing_seconds': {k: result[k] for k in ['cache_seconds', 'fit_seconds', 'seconds']},
        'VM_supervisor_seconds': terminal['seconds'], 'local_audit_seconds': full['seconds'],
        'maximum_CPU_head_replay_difference': full['maximum_head_replay_difference'],
        'original_import_failure_preserved': str((RETURN / 'local_audit.log').resolve()),
        'first_recovery_failure_preserved': str((ROOT / 'outputs/cctv_dgp_structure_v18_audit_recovery_r1/audit.log').resolve()),
        'teacher_used': False, 'validation_used': False, 'native_used': False, 'native_reserved_used': False,
        'checkpoint_selected': False, 'production_promoted': False, 'local_optimizer_updates': 0, 'local_backward_calls': 0,
        'next_scope': 'Diagnose input-only preservation/selection processing on training inputs before a separately justified protocol. Do not repeat V18 or waive its failed guards.'}
    write_new(ROOT / 'outputs/cctv_dgp_structure_v18_verified_summary.json', verified)
    for update, data in summaries.items():
        print(json.dumps({'update': update, 'clear': data['summary']['clear'], 'degraded': data['summary']['degraded'], 'preservation': data['preservation']}))
    print(json.dumps({'complete': True, 'reviewed_sheets': 10, 'original_cells': 550, 'source_assets': verified['source_assets'],
                      'production_promoted': False, 'local_optimizer_updates': 0}))


if __name__ == '__main__':
    summarize()
