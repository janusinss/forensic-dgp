"""Seal a narrowly corrected startup fixture; retain the entire failed R2.

No neural model construction, inference, gradients, training or VM execution.
"""
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_conditioning_learning_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

OLD_NAME = 'cctv_dgp_conditioning_learning_v1_r2_vm'
NAME = 'cctv_dgp_conditioning_learning_v1_r3_vm'
STEM = 'cctv-dgp-conditioning-learning-v1-r3'
OLD_PROBE = 'np.arange(256,dtype=np.uint8).reshape(1,1,16,16).repeat(3,axis=1)'
NEW_PROBE = 'np.tile(np.arange(256,dtype=np.uint8)[None,None,None,:],(1,3,256,1))'


def save_text(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)


def main():
    dest = OUT / NAME
    assert not dest.exists(), 'Preserve previously sealed packets'
    p = read(RETURN / 'protocol.json')
    cause = read(ANALYSIS / 'startup_review.json')
    assert cause['complete'] and cause['protocol_sha256'] == PIN == sha(RETURN / 'protocol.json')
    assert cause['finite_structure_gate_reached'] is False and not any(cause['reported_counts'].values())
    old_packet = {**p['assets_sha256'], 'protocol.json': PIN}
    assert all(sha(RETURN / n) == h for n, h in old_packet.items())
    assert all(sha(OUT / OLD_NAME / n) == h for n, h in old_packet.items())
    assert all(sha(PARENT / n) == h for n, h in p['parent_required_sha256'].items())
    old_archives = {n: sha(OUT / n) for n in [
        'cctv-dgp-conditioning-learning-v1-r2-execution.tar.gz',
        'cctv-dgp-conditioning-learning-v1-r2-results.tar.gz',
        'cctv-dgp-conditioning-learning-v1-r2-results.tar.gz.sha256',
        'cctv-dgp-conditioning-learning-v1-r2-export.json']}
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    dest.mkdir()
    edits = []
    for n, h in p['assets_sha256'].items():
        source = RETURN / n
        target_n = n.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
        target = dest / target_n
        target.parent.mkdir(parents=True, exist_ok=True)
        if n == 'scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py':
            original = source.read_text(encoding='utf-8')
            assert original.count(OLD_PROBE) == 1
            text = original.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
            text = text.replace('cctv-dgp-conditioning-learning-v1-r2', STEM)
            text = text.replace(OLD_PROBE, NEW_PROBE)
            ast.parse(text, feature_version=(3, 10))
            save_text(target, text)
            save_text(ROOT / 'scripts/cctv_dgp_conditioning_learning_v1_r3_vm.py', text)
            assert text.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r2').replace(
                STEM, 'cctv-dgp-conditioning-learning-v1-r2').replace(NEW_PROBE, OLD_PROBE) == original
            edits.append(dict(file=n, change='Full256x256 all-byte-level fixture plus immutable namespace'))
        elif n in ['conditioning_model.py', 'scripts/run_learning.sh']:
            original = source.read_text(encoding='utf-8')
            text = original.replace('conditioning_learning_v1_r2', 'conditioning_learning_v1_r3')
            text = text.replace('cctv-dgp-conditioning-learning-v1-r2', STEM)
            assert text != original
            save_text(target, text)
            assert text.replace('conditioning_learning_v1_r3', 'conditioning_learning_v1_r2').replace(
                STEM, 'cctv-dgp-conditioning-learning-v1-r2') == original
            edits.append(dict(file=n, change='Immutable namespace only'))
        else:
            with source.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(target) == h
    with (ANALYSIS / 'startup_review.json').open('rb') as src, (dest / 'startup_review.json').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    q = copy.deepcopy(p)
    q.update(format='finite-conditioning-learning-v1-r3', name=NAME,
             UTC=datetime.now(timezone.utc).isoformat(),
             assets_sha256={f.relative_to(dest).as_posix(): sha(f) for f in dest.rglob('*') if f.is_file()},
             startup_repair_of_protocol_sha256=PIN,
             startup_repair_of_return_archive_sha256=read(ANALYSIS / 'import.json')['archive_sha256'],
             startup_review_sha256=sha(ANALYSIS / 'startup_review.json'),
             repair_scope='256x256 canonical pixel fixture only; no model, loss, threshold or training limit changes')
    excluded = {'format', 'name', 'UTC', 'assets_sha256', 'startup_repair_of_protocol_sha256',
                'startup_repair_of_return_archive_sha256', 'startup_review_sha256', 'repair_scope'}
    assert {n: v for n, v in q.items() if n not in excluded} == {n: v for n, v in p.items() if n not in excluded}
    write(dest / 'protocol.json', q)
    pin = sha(dest / 'protocol.json')
    archive = OUT / (STEM + '-execution.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for f in sorted(f for f in dest.rglob('*') if f.is_file()):
            tar.add(f, arcname=NAME + '/' + f.relative_to(dest).as_posix(), recursive=False)
    save_text(Path(str(archive) + '.sha256'), sha(archive) + '  ' + archive.name + '\n')
    assert all(sha(RETURN / n) == h and sha(OUT / OLD_NAME / n) == h for n, h in old_packet.items())
    assert all(sha(OUT / n) == h for n, h in old_archives.items())
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(OUT / 'cctv_dgp_conditioning_learning_v1_r3_preparation.json', dict(complete=True,
          UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
          protocol_sha256=pin, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
          packet_assets=len(q['assets_sha256']), parent_file_bindings=len(q['parent_required_sha256']),
          source_protocol_sha256=PIN, returned_startup_review_sha256=sha(ANALYSIS / 'startup_review.json'),
          activation_audit_sha256=q['activation_audit_sha256'], exact_changed_assets=edits,
          old_packet_and_return_archive_files_sha256=old_archives, old_packet_and_all_failures_retained=True,
          model_loss_threshold_training_limits_and_initial_state_unchanged=True,
          local_neural_gradient_optimizer_or_parameter_displacement_calls=0, manual_pilot_unrun=True,
          prospective_transfer_and_return_audits_required=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_bytes=archive.stat().st_size, manual_unrun=True))


if __name__ == '__main__':
    main()
