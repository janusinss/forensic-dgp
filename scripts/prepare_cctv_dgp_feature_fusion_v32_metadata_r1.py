"""Preserve V32 and create a metadata-corrected, separately routed R1 packet."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32'
NAME = 'cctv_dgp_feature_fusion_vm_v32_r1'
STEM = 'cctv-dgp-feature-fusion-v32-r1'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation'
OLD_PIN = 'e022fbc655b1b93caefc0572393171126c3db7c37d0e559732b38bfe8a656ee1'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker_routing(text):
    return text.replace('cctv_dgp_feature_fusion_vm_v32', NAME).replace('cctv-dgp-feature-fusion-v32', STEM).replace('cctv_dgp_feature_fusion_v32_return', 'cctv_dgp_feature_fusion_v32_r1_return')


def checker_routing(text):
    text = worker_routing(text)
    for before, after in [('cctv_dgp_feature_fusion_v32_return_import', 'cctv_dgp_feature_fusion_v32_r1_return_import'),
                          ('cctv_dgp_feature_fusion_v32_independent_audit', 'cctv_dgp_feature_fusion_v32_r1_independent_audit')]:
        text = text.replace(before, after)
    return text.replace('FUSION_V32_PROTOCOL_PIN', 'FUSION_V32_R1_PROTOCOL_PIN')


def main():
    started = time.monotonic()
    assert not BUNDLE.exists() and not PREP.exists() and sha(ORIGINAL / 'protocol.json') == OLD_PIN
    old = read(ORIGINAL / 'protocol.json')
    checked = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation/independent_packet_audit.json')
    closed = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone/independent_closure_audit.json')
    assert checked['complete'] and closed['complete'] and checked['regressions_passed'] == 11
    assert not (ORIGINAL / 'outputs').exists()
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    for name, digest in old['assets_sha256'].items():
        source = ORIGINAL / name; assert sha(source) == digest
        target = BUNDLE / name; target.parent.mkdir(parents=True, exist_ok=True)
        if name == 'scripts/cctv_dgp_feature_fusion_v32_vm.py':
            target.write_text(worker_routing(source.read_text()), encoding='utf-8', newline='\n')
        else:
            target.write_bytes(source.read_bytes())
    p = copy.deepcopy(old)
    for key in ['initial_parity', 'normalization', 'gradient_policy']:
        p[key] = p[key].replace('all12', 'all23').replace('selected12', 'selected23').replace('Selected12', 'Selected23')
    p['optimizer']['type'] = 'AdamW selected23 original fusion/decoder tensors'
    p['execution'] = 'Human gcloud transfer/SSH/tmux training on existing NVIDIA L4/g2-standard-4 only; no assistant training launch. Separately authorized VM maintenance remains permitted.'
    p['metadata_revision'] = 1
    p['metadata_revision_reason'] = 'Correct inherited12-tensor descriptions to23 and route a distinct R1 directory/export. All mathematical training behavior, losses, optimizer settings, scientific gates and selected tensors remain identical to the unlaunched V32 packet.'
    p['superseded_unlaunched_V32_protocol_sha256'] = OLD_PIN
    prototypes = [ORIGINAL / 'protocol.json', *[ORIGINAL / n for n in old['assets_sha256']],
                  ROOT / 'outputs/cctv-dgp-feature-fusion-v32-execution.tar.gz',
                  ROOT / 'outputs/cctv-dgp-feature-fusion-v32-execution.tar.gz.sha256',
                  ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation/independent_packet_audit.json',
                  ROOT / 'outputs/cctv_dgp_feature_fusion_v32_forward_review/review.json',
                  ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone/milestone.json',
                  ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone/independent_closure_audit.json']
    p['superseded_packet_evidence_sha256'] = {f.relative_to(ROOT).as_posix(): sha(f) for f in prototypes}
    basis = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_metadata_r1.py']
    p['local_basis_sha256'].update({f.relative_to(ROOT).as_posix(): sha(f) for f in basis})
    p['assets_sha256'] = {f.relative_to(BUNDLE).as_posix(): sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()}
    for f in BUNDLE.rglob('*.py'): ast.parse(f.read_text(), feature_version=(3, 10))
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    template = checker_routing((ROOT / 'scripts/cctv_dgp_feature_fusion_v32_return_audit_template.py').read_text())
    assert template.count('FUSION_V32_R1_PROTOCOL_PIN') == 1
    template_file = ROOT / 'scripts/cctv_dgp_feature_fusion_v32_r1_return_audit_template.py'
    with template_file.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template)
    checker = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r1_return.py'
    with checker.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template.replace('FUSION_V32_R1_PROTOCOL_PIN', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for f in sorted(BUNDLE.rglob('*')):
            if f.is_file(): tar.add(f, arcname=NAME + '/' + f.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + archive.name + '\n')
    import prepare_cctv_dgp_feature_fusion_v32 as base
    base.NAME, base.STEM = NAME, STEM
    guide = base.guide(pin, digest).replace('# V32:', '# V32 r1:').replace('tmux new-session -A -s dgp_feature_fusion_v32', 'tmux new-session -A -s dgp_feature_fusion_v32_r1')
    notice = '\nUse this R1 packet. The preserved unlaunched V32 packet had inherited descriptions\nsaying12 selected tensors although its code selected23. R1 corrects that metadata\nand uses separate output paths; it changes no training recipe, loss or gate.\n'
    position = guide.index('\n') + 1; guide = guide[:position] + notice + guide[position:]
    with (ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_R1_VM.md').open('x', encoding='utf-8', newline='\n') as stream: stream.write(guide)
    # The shell is byte-identical; inherit the already completed read-only parse,
    # rather than performing another host execution for unchanged syntax.
    bash = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation/bash_syntax.json')
    assert sha(BUNDLE / 'scripts/run_v32.sh') == bash['script_sha256']
    bash['inherited_from'] = 'outputs/cctv_dgp_feature_fusion_v32_preparation/bash_syntax.json'
    bash['new_parser_invocations'] = 0
    write(PREP / 'bash_syntax.json', bash)
    write(PREP / 'preparation.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
          'archive_bytes': archive.stat().st_size, 'packet_files': 11, 'metadata_revision': 1,
          'superseded_protocol_sha256': OLD_PIN, 'mathematical_training_behavior_unchanged': True,
          'selected_tensors': 23, 'selected_parameters': 978243,
          'prospective_return_auditor_sha256': sha(checker), 'prospective_template_sha256': sha(template_file),
          'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
          'actual_VM_training_started': False, 'manual_VM_required': True, 'goal_complete': False,
          'seconds': time.monotonic() - started})
    print(json.dumps(read(PREP / 'preparation.json'), indent=2))


if __name__ == '__main__':
    main()
