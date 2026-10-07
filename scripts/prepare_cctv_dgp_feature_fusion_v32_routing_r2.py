"""Fix the demonstrated pre-optimizer cache-root mismatch in a distinct R2 packet."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r1'
NAME = 'cctv_dgp_feature_fusion_vm_v32_r2'
STEM = 'cctv-dgp-feature-fusion-v32-r2'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_preparation'
FAILURE = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_failure_v1/host_access_retry'
OLD_PIN = 'e25e5721a5474d767fe48710336105a1eaa843ed2e22d6cb03caebd8e3bd10da'
WORKER = 'scripts/cctv_dgp_feature_fusion_v32_vm.py'
CACHE = 'cctv_dgp_feature_fusion_v32_cache.py'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def routing(text):
    return text.replace(OLD.name, NAME).replace('cctv-dgp-feature-fusion-v32-r1', STEM).replace(
        'cctv_dgp_feature_fusion_v32_r1_return', 'cctv_dgp_feature_fusion_v32_r2_return')


def worker_fix(text):
    text = routing(text)
    text = once(text, "    assert root.name == '" + NAME + "'\n",
                "    assert root.name == '" + NAME + "'\n"
                "    from cctv_dgp_feature_fusion_v32_cache import require_cache_scope\n"
                "    require_cache_scope(root)\n")
    return once(text, '    if a.verify_transfer:\n', '    if a.verify_transfer:\n        vm_scope(root, parent)\n')


def cache_fix(text):
    before = "def load_items(root, p, original, identity, canonical_tensor, clock, write):\n" \
             "    assert sys.platform == 'linux', 'Existing Linux L4 VM only; no local training cache'\n" \
             "    assert root.resolve() == (Path.home() / 'forensic-dgp/cctv_dgp_feature_fusion_vm_v32').resolve()\n"
    after = "def require_cache_scope(root):\n" \
            "    assert sys.platform == 'linux', 'Existing Linux L4 VM only; no local training cache'\n" \
            "    assert root.resolve() == (Path.home() / 'forensic-dgp/" + NAME + "').resolve()\n\n\n" \
            "def load_items(root, p, original, identity, canonical_tensor, clock, write):\n" \
            "    require_cache_scope(root)\n"
    return once(text, before, after)


def main():
    start = time.monotonic()
    assert not BUNDLE.exists() and not PREP.exists() and sha(OLD / 'protocol.json') == OLD_PIN
    failed = read(FAILURE / 'verification.json')
    assert failed['complete'] and failed['original_failure_verified'] and failed['optimizer_updates'] == 0
    assert not failed['optimizer_constructed'] and not failed['new_checkpoint_created']
    repro = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_routing_regression_v1/r1_failure_reproduction.json')
    assert not repro['complete'] and not repro['rows'][0]['actual'] and repro['neural_imports'] == 0
    old = read(OLD / 'protocol.json')
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    for name, digest in old['assets_sha256'].items():
        source = OLD / name; assert sha(source) == digest
        target = BUNDLE / name; target.parent.mkdir(parents=True, exist_ok=True)
        if name in (WORKER, CACHE):
            fixed = worker_fix(source.read_text()) if name == WORKER else cache_fix(source.read_text())
            with target.open('x', encoding='utf-8', newline='\n') as stream: stream.write(fixed)
        else:
            with target.open('xb') as stream: stream.write(source.read_bytes())
    p = copy.deepcopy(old)
    p['routing_revision'] = 2
    p['routing_revision_reason'] = 'Correct R1 cache root to the separate R2 packet; execute the same Linux/exact-root cache guard during transfer verification and before run. Original R1 failed before any optimizer; retain its logs, preflight and export. Training math, selected tensors, cases, schedule, optimizer, losses and all numerical gates are unchanged.'
    p['superseded_R1_protocol_sha256'] = OLD_PIN
    evidence = [f for f in sorted(FAILURE.rglob('*')) if f.is_file()]
    evidence.extend(f for f in sorted((FAILURE.parent).glob('*')) if f.is_file())
    evidence.append(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_routing_regression_v1/r1_failure_reproduction.json')
    p['closed_R1_launch_failure_evidence_sha256'] = {f.relative_to(ROOT).as_posix(): sha(f) for f in evidence}
    prior = [OLD / 'protocol.json', *[OLD / n for n in old['assets_sha256']],
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r1-execution.tar.gz',
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r1-execution.tar.gz.sha256',
             ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_R1_VM.md',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation/independent_packet_audit.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_milestone/milestone.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_milestone/independent_closure_audit.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_milestone/final_readback.json']
    p['prior_R1_packet_evidence_sha256'] = {f.relative_to(ROOT).as_posix(): sha(f) for f in prior}
    basis = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_routing_r2.py',
             ROOT / 'scripts/check_cctv_dgp_feature_fusion_v32_root_routing.py',
             ROOT / 'scripts/read_cctv_dgp_feature_fusion_v32_r1_failure_v1.py']
    p['local_basis_sha256'].update({f.relative_to(ROOT).as_posix(): sha(f) for f in basis})
    p['assets_sha256'] = {f.relative_to(BUNDLE).as_posix(): sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()}
    for f in BUNDLE.rglob('*.py'): ast.parse(f.read_text(), feature_version=(3, 10))
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    template = routing((ROOT / 'scripts/cctv_dgp_feature_fusion_v32_r1_return_audit_template.py').read_text())
    template = template.replace('cctv_dgp_feature_fusion_v32_r1_independent_audit', 'cctv_dgp_feature_fusion_v32_r2_independent_audit').replace('FUSION_V32_R1_PROTOCOL_PIN', 'FUSION_V32_R2_PROTOCOL_PIN')
    assert template.count('FUSION_V32_R2_PROTOCOL_PIN') == 1
    tf = ROOT / 'scripts/cctv_dgp_feature_fusion_v32_r2_return_audit_template.py'
    af = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r2_return.py'
    with tf.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template)
    with af.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template.replace('FUSION_V32_R2_PROTOCOL_PIN', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for f in sorted(BUNDLE.rglob('*')):
            if f.is_file(): tar.add(f, arcname=NAME + '/' + f.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    import prepare_cctv_dgp_feature_fusion_v32 as base
    base.NAME, base.STEM = NAME, STEM
    guide = base.guide(pin, digest).replace('# V32:', '# V32 r2:').replace(
        'tmux new-session -A -s dgp_feature_fusion_v32', 'tmux new-session -A -s dgp_feature_fusion_v32_r2')
    notice = '\nR1 stopped before any optimizer because its cache loader required the base V32\ndirectory. R2 corrects both directory guards and checks the cache guard during\n`--verify-transfer`, before model work. Keep the failed R1 directory and export.\nThis is a packaging repair of the same finite experiment; no numerical gate or\ntraining recipe changes. The R1 full return still needs download and audit.\n'
    split = guide.index('\n') + 1; guide = guide[:split] + notice + guide[split:]
    guide += '\nPreserve the R1 failure first, using these separate **Windows Google Cloud SDK Shell** downloads:\n\n```bat\ncd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n'
    for suffix in ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json']:
        guide += 'gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-feature-fusion-v32-r1' + suffix + '" "."\n'
    guide += '```\n\nExpected R1 archive:169,708,697 bytes; SHA256\n599d84dd2b84fdc05e55b774d115951d251e4013ae777d7f023791bdb60f59e2.\n'
    with (ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_R2_VM.md').open('x', encoding='utf-8', newline='\n') as stream: stream.write(guide)
    bash = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation/bash_syntax.json')
    assert bash['complete'] and sha(BUNDLE / 'scripts/run_v32.sh') == bash['script_sha256']
    bash['inherited_from'] = 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation/bash_syntax.json'
    bash['new_parser_invocations'] = 0; write(PREP / 'bash_syntax.json', bash)
    report = {'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
              'packet_files':11,'routing_revision':2,'R1_zero_update_failure_preserved':True,
              'unchanged_mathematical_training_behavior':True,'selected_tensors':23,'selected_parameters':978243,
              'prospective_return_auditor_sha256':sha(af),'prospective_template_sha256':sha(tf),
              'local_neural_calls':0,'local_gradient_calls':0,'local_optimizer_updates':0,
              'actual_VM_training_started_by_agent':False,'manual_VM_required':True,'goal_complete':False,
              'seconds':time.monotonic()-start}
    write(PREP / 'preparation.json', report); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
