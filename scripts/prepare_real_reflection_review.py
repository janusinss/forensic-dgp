"""Refresh source-only eyewear qualification against all current references."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'outputs/real_reflection_review_v4'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify_sources(records, train, held, current, benchmark_hashes):
    require(train.isdisjoint(held), 'Fixed training/held-out paths overlap')
    require(len({r['index'] for r in records}) == len(records)
            and len({r['path'] for r in records}) == len(records)
            and len({r['sha256'] for r in records}) == len(records), 'Duplicate source identity or bytes')
    used = {r[k] for r in current for k in ('source_sha256', 'image_sha256') if r.get(k)}
    eligible, excluded = [], []
    for row in records:
        require(row['partition'] == 'train' and row['training_enabled'] is False and row['mask'] is None,
                'Source queue must contain training-only, unlabelled, disabled proposals')
        reason = None
        if row['path'] not in train or row['path'] in held:
            reason = 'outside_fixed_training_split'
        elif row['sha256'] in used:
            reason = 'already_reviewed_source_or_crop'
        elif row['sha256'] in benchmark_hashes:
            reason = 'benchmark_source'
        (excluded if reason else eligible).append({**row, 'reason': reason} if reason else dict(row))
    return eligible, excluded


# Prior source-only page review, not prediction-guided selection or pixel labels.
NATIVE_COVERING = {106, 119, 171, 216, 218, 310, 348, 374}
NATIVE_CONTROL = {52, 157, 207, 208, 217, 219, 230, 232, 240, 241}
DEFERRED = {14, 84, 177, 212, 224, 249}
NOTES = {
    106: 'Mirrored lenses; inspect foreground food and the complete facial covering boundary.',
    119: 'Profile sunglasses; far lens boundary needs native inspection.',
    171: 'Blue/scene-reflective lenses; inspect helmet and chin-strap context.',
    216: 'Dark lenses conceal eyes; trace actual obscuring footprint.',
    218: 'Dark wraparound lenses; inspect visible face and temples.',
    310: 'Dark red lenses; inspect hand near the chin for facial overlap.',
    348: 'Colored scene-reflective lenses; trace covering, not bright pixels alone.',
    374: 'Dark mirrored lenses; inspect helmet and collar overlap.',
    14: 'Strong skin overexposure without clear lens covering; photometric cause unresolved.',
    84: 'Small blurred tinted glasses; visible-eye versus hidden-eye boundary unresolved.',
    177: 'Small blurred dark glasses; insufficient boundary confidence.',
    212: 'Colored spots near lenses; reflection versus physical object unresolved.',
    224: 'Partial lens glint with visible eye detail; covering boundary unresolved.',
    249: 'Small white glints; visible-eye versus reflection boundary unresolved.',
}


def main():
    from PIL import Image, ImageOps, ImageDraw
    from detector_training import load_manifest
    from scripts.audit_real_source_pool import signature

    queue_path = ROOT/'outputs/eyewear_review_v1/queue.json'
    queue = json.loads(queue_path.read_text())
    audit_root = ROOT/'outputs/training_diversity_audit'
    candidate_path = audit_root/'candidate_sources.json'
    roles_path = audit_root/'source_roles_v2.json'
    screen_path = audit_root/'duplicate_screen.json'
    split_path = ROOT/'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json'
    base_path = ROOT/'dataset/detector_glare_review_v3/manifest.json'
    current_path = ROOT/'dataset/detector_training_extension_v2/manifest.json'
    benchmark_path = ROOT/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json'
    crop_path = ROOT/'outputs/face_crop_diagnostic_v1/verification.json'
    inputs = (queue_path, candidate_path, roles_path, screen_path, split_path,
              base_path, current_path, benchmark_path, crop_path, ROOT/'OCCLUSION_POLICY_V3.md')
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    require(not OUT.exists(), 'Preserve completed/partial source review')
    for key, path in (('candidate_manifest_sha256', candidate_path), ('source_role_manifest_sha256', roles_path),
                      ('duplicate_screen_sha256', screen_path), ('split_sha256', split_path)):
        require(queue[key] == sha(path), 'Original source queue binding changed: '+key)
    candidate = json.loads(candidate_path.read_text()); screen = json.loads(screen_path.read_text())
    require(candidate['v3_manifest_sha256'] == sha(base_path)
            and candidate['benchmark_manifest_sha256'] == sha(benchmark_path)
            and screen['candidate_manifest_sha256'] == sha(candidate_path)
            and not screen['decoded_matches'] and not screen['review_flags'], 'Original source-screen lineage differs')
    current = load_manifest(current_path)
    require(len(current) == 105 and Counter(r['split'] for r in current)
            == {'train': 73, 'validation': 25, 'test': 7}, 'Current reviewed membership differs')
    crop = json.loads(crop_path.read_text())
    require(crop['complete'] is True and crop['training_signal']['training_signal_passes'] is False
            and crop['promoted'] is False, 'Completed failed crop evidence differs')
    split = json.loads(split_path.read_text())
    train = {p.replace('\\', '/') for p in split['train']}
    held = {p.replace('\\', '/') for p in split['validation']}
    benchmark = json.loads(benchmark_path.read_text())['cases']
    require(len(benchmark) == 400, 'Original benchmark membership differs')
    roles = {r['index']: r for r in json.loads(roles_path.read_text())['records']}
    records = queue['records']
    require(len(records) == 27 and {r['index'] for r in records}
            == NATIVE_COVERING | NATIVE_CONTROL | DEFERRED | {13, 46, 49}, 'Fixed source review membership differs')
    for row in records:
        source = (ROOT/row['path']).resolve()
        require(source.is_relative_to(ROOT/'dataset') and sha(source) == row['sha256']
                and all(row[k] == roles[row['index']][k] for k in ('path', 'sha256', 'width', 'height', 'source')),
                'Native source bytes or prior roles differ')
    eligible, excluded = qualify_sources(records, train, held, current, {c['source_sha256'] for c in benchmark})
    references = {p: {'roles': ['original_validation'], 'expected': None} for p in held}
    def reference(path, role, digest=None):
        name = path.replace('\\', '/')
        if name in references:
            references[name]['roles'].append(role)
            if digest:
                previous = references[name]['expected']
                require(previous in (None, digest), 'Reference byte bindings contradict')
                references[name]['expected'] = digest
        else:
            references[name] = {'roles': [role], 'expected': digest}
    for row in current:
        reference(row['source'], 'reviewed_source_'+row['split'], row['source_sha256'])
        reference((current_path.parent/row['image']).relative_to(ROOT).as_posix(),
                  'reviewed_crop_'+row['split'], row['image_sha256'])
    for case in benchmark:
        reference(case['source'], 'benchmark_source', case['source_sha256'])
    signatures = {r['index']: signature(ROOT/r['path']) for r in eligible}
    flags = {r['index']: [] for r in eligible}; reference_rows = []
    for i, (name, metadata) in enumerate(sorted(references.items()), 1):
        sig = signature(ROOT/name)
        require(metadata['expected'] is None or sig['sha256'] == metadata['expected'], 'Frozen reference source/crop changed')
        reference_rows.append({'path': name, 'roles': sorted(set(metadata['roles'])), **sig})
        for index, proposed in signatures.items():
            distance = (proposed['phash'] ^ sig['phash']).bit_count()
            exact = proposed['sha256'] == sig['sha256'] or proposed['decoded_sha256'] == sig['decoded_sha256']
            if exact or distance <= 6:
                flags[index].append({'reference': name, 'roles': metadata['roles'], 'exact': exact, 'phash_distance': distance})
        if i % 1000 == 0:
            print('Fresh source reference screen', i, len(references), flush=True)
    pairs = []
    for position, row in enumerate(eligible):
        a = signatures[row['index']]
        for other in eligible[:position]:
            b = signatures[other['index']]; distance = (a['phash'] ^ b['phash']).bit_count()
            exact = a['sha256'] == b['sha256'] or a['decoded_sha256'] == b['decoded_sha256']
            if exact or distance <= 6:
                pairs.append({'a': row['index'], 'b': other['index'], 'exact': exact, 'phash_distance': distance})
    paired = {p[k] for p in pairs for k in ('a', 'b')}
    for row in eligible:
        index = row['index']
        require(signatures[index]['sha256'] == row['sha256'], 'Candidate changed during fresh screen')
        role = 'native_covering_review' if index in NATIVE_COVERING else (
               'native_transparent_control_review' if index in NATIVE_CONTROL else 'deferred_boundary_review')
        row.update(review_role=role, source_signature=signatures[index], overlap_flags=flags[index],
                   within_candidate_overlap=index in paired, mask=None, reviewed=False, training_enabled=False,
                   qualification='pending_overlap_review' if flags[index] or index in paired else 'eligible_for_native_review',
                   rationale=NOTES.get(index, 'Visible eyes through transparent lenses; inspect source before approving an empty mask.'))
    require(all(sha(ROOT/name) == digest for name, digest in bindings.items()), 'Inputs changed during qualification')
    OUT.mkdir(); (OUT/'native').mkdir()
    for row in eligible:
        with Image.open(ROOT/row['path']) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
        path = OUT/'native'/f"{row['index']:03}.png"; rgb.save(path)
        row.update(native_review_image=path.relative_to(ROOT).as_posix(), native_review_sha256=sha(path),
                   native_review_size=list(rgb.size))
    for role in ('native_covering_review', 'native_transparent_control_review', 'deferred_boundary_review'):
        subset = [r for r in eligible if r['review_role'] == role]
        for start in range(0, len(subset), 6):
            page = subset[start:start+6]; sheet = Image.new('RGB', (1152, 420*((len(page)+2)//3)), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, row in enumerate(page):
                with Image.open(ROOT/row['native_review_image']) as image:
                    tile = ImageOps.contain(image, (384, 384), Image.Resampling.NEAREST)
                x, y = j%3*384, j//3*420
                draw.text((x+3, y+2), f"{row['index']} {Path(row['path']).name} {tuple(row['native_review_size'])}", fill='black')
                draw.text((x+3, y+16), row['qualification']+'; no approved mask', fill='black')
                sheet.paste(tile, (x+(384-tile.width)//2, y+32))
            sheet.save(OUT/f'{role}_{start//6}.png')
    (OUT/'reference_signatures.json').write_text(json.dumps(reference_rows, indent=2)+'\n', encoding='utf-8')
    report = {'format': 'dgp-real-reflection-source-review-v4', 'date': '2026-10-02', 'complete': True,
              'script_sha256': sha(__file__), 'test_sha256': sha(ROOT/'tests/test_real_reflection_review.py'),
              'input_sha256': bindings, 'records': eligible, 'excluded': excluded,
              'within_candidate_pairs': pairs, 'reference_count': len(reference_rows),
              'reference_signatures_sha256': sha(OUT/'reference_signatures.json'),
              'artifacts': {p.relative_to(OUT).as_posix(): sha(p) for p in OUT.glob('*.png')},
              'role_counts': dict(Counter(r['review_role'] for r in eligible)),
              'overlap_flagged_sources': sum(bool(r['overlap_flags']) or r['within_candidate_overlap'] for r in eligible),
              'training_enabled': False, 'training_recipe_ready': False, 'new_approved_masks': 0,
              'model_forward_passes': 0, 'optimizer_updates_locally': 0, 'promoted': False,
              'scope': 'Source-only eligibility and native review proposals; no masks, clear labels or source splits changed',
              'limitations': ['Exact/decoded/DCT whole-image screening cannot establish identity separation or exclude all alternate crops.',
                             'Native covering/control proposals require visual inspection and approximate pixel annotation.',
                             'Opaque/scene-reflective examples alone do not resolve missing small/partial real glare diversity.']}
    (OUT/'manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('reference_count', 'role_counts', 'overlap_flagged_sources',
                                            'training_enabled', 'training_recipe_ready', 'new_approved_masks')}
                     | {'excluded_sources': [r['index'] for r in excluded]}), flush=True)


if __name__ == '__main__':
    main()
