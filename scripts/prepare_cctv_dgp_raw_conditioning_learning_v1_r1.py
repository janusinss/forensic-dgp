"""Create a fresh immutable receipt-only correction; never rerun the old pilot."""
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, ANALYSIS, PARENT, read, sha, write

NAME = 'cctv_dgp_raw_conditioning_learning_v1_r1_vm'
STEM = 'cctv-dgp-raw-conditioning-learning-v1-r1'
OLD_NAME = 'cctv_dgp_raw_conditioning_learning_v1_vm'
OLD_STEM = 'cctv-dgp-raw-conditioning-learning-v1'


def text_file(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    original = OUT / OLD_NAME
    p = read(original / 'protocol.json')
    prep = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    red = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_receipt_red.json')
    published = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_publication/publication.json')
    assert prep['complete'] and published['complete'] and not red['complete']
    assert sha(original / 'protocol.json') == prep['protocol_sha256'] == red['protocol_sha256']
    assert red['differences'] == {'maximum_gradient_queries': {'actual': 1280, 'expected': 640}}
    assert p['maximum_gradient_queries'] == 640 and p['recognizer_forward_cap'] == 320
    old_files = {**p['assets_sha256'], 'protocol.json': prep['protocol_sha256']}
    assert all(sha(original / n) == h for n, h in old_files.items())
    assert not (original / 'outputs').exists() and not (original / 'review.log').exists()
    archived = {n: sha(OUT / n) for n in [OLD_STEM + '-execution.tar.gz', OLD_STEM + '-execution.tar.gz.sha256']}
    destination = OUT / NAME
    assert not destination.exists()
    destination.mkdir()
    old_line = 'cases=100,maximum_accepted_updates=2,maximum_gradient_queries=1280,actual_neural_calls=0,'
    new_line = "cases=p['paired_TRAIN_cases'],maximum_accepted_updates=p['maximum_accepted_updates'],maximum_gradient_queries=p['maximum_gradient_queries'],actual_neural_calls=0,"
    for name, digest in p['assets_sha256'].items():
        source = original / name
        target = destination / name.replace('raw_conditioning_learning_v1', 'raw_conditioning_learning_v1_r1')
        target.parent.mkdir(parents=True, exist_ok=True)
        if name in ['conditioning_model.py', 'scripts/run_learning.sh'] or name.endswith('_vm.py'):
            text = source.read_text(encoding='utf-8').replace('raw_conditioning_learning_v1', 'raw_conditioning_learning_v1_r1')
            text = text.replace(OLD_STEM, STEM)
            if name.endswith('_vm.py'):
                assert text.count(old_line) == 1
                text = text.replace(old_line, new_line)
                ast.parse(text, feature_version=(3, 10))
                text_file(ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_r1_vm.py', text)
            text_file(target, text)
        else:
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(target) == digest
    with (OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_receipt_red.json').open('rb') as src, (destination / 'transfer_receipt_review.json').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    q = copy.deepcopy(p)
    q.update(format='finite-RAW32-conditioning-learning-v1-r1', name=NAME, UTC=datetime.now(timezone.utc).isoformat(),
        assets_sha256={f.relative_to(destination).as_posix(): sha(f) for f in destination.rglob('*') if f.is_file()},
        transfer_receipt_repair_of_protocol_sha256=prep['protocol_sha256'],
        transfer_receipt_red_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_receipt_red.json'),
        transfer_receipt_fields_read_directly_from_protocol=True,
        namespace_and_transfer_summary_only_changed_from_RAW32_V1=True)
    excluded = {'format', 'name', 'UTC', 'assets_sha256', 'transfer_receipt_repair_of_protocol_sha256',
                'transfer_receipt_red_sha256', 'transfer_receipt_fields_read_directly_from_protocol',
                'namespace_and_transfer_summary_only_changed_from_RAW32_V1'}
    assert {k:v for k,v in q.items() if k not in excluded} == {k:v for k,v in p.items() if k not in excluded}
    write(destination / 'protocol.json', q)
    pin = sha(destination / 'protocol.json')
    archive = OUT / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for f in sorted(f for f in destination.rglob('*') if f.is_file()):
            tar.add(f, arcname=NAME + '/' + f.relative_to(destination).as_posix(), recursive=False)
    text_file(Path(str(archive) + '.sha256'), sha(archive) + '  ' + archive.name + '\n')
    assert all(sha(original / n) == h for n, h in old_files.items())
    assert all(sha(OUT / n) == h for n, h in archived.items())
    assert all(sha(ROOT / n) == h for n, h in read(ANALYSIS / 'local_protected_before.json')['files_sha256'].items())
    write(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_preparation.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__), name=NAME,
        protocol_sha256=pin, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        packet_assets=len(q['assets_sha256']), parent_file_bindings=len(q['parent_required_sha256']),
        previous_preparation_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json'),
        source_protocol_sha256=prep['protocol_sha256'], old_archive_files_sha256=archived,
        old_packet_and_transfer_and_checkers_unchanged=True, old_packet_local_training_unrun=True,
        previous_human_execution_unknown_and_not_assumed=True,
        recipe_state_losses_optimizer_selection_quality_and_resource_stops_unchanged=True,
        corrected_summary_is_hash_only_and_requires_no_neural_calls=True,
        manual_pilot_unrun=True, prospective_transfer_and_return_audits_required=True,
        local_neural_gradient_optimizer_or_parameter_displacement_calls=0, model_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_bytes=archive.stat().st_size, manual_unrun=True))


if __name__ == '__main__':
    main()
