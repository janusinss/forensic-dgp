"""Publish the verified manual diagnostic guide; preserve every prior doc byte."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/codeformer_path_diagnostic_v1_preparation'
PUBLICATION = EVIDENCE / 'publication_v1'
FILES = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
         'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def main():
    assert not PUBLICATION.exists(), 'Preserve completed or partial publication'
    archive = json.loads((EVIDENCE / 'execution_archive.json').read_text())
    independent = json.loads((EVIDENCE / 'independent_packet_archive_readback_v1.json').read_text())
    source = json.loads((EVIDENCE / 'independent_source_review_r1.json').read_text())
    assert archive['complete'] and independent['complete'] and source['approved_for_protocol_freeze']
    assert archive['protocol_sha256'] == independent['protocol_sha256']
    assert archive['archive_sha256'] == independent['archive_sha256']
    assert sha((ROOT / archive['archive']).read_bytes()) == archive['archive_sha256']
    packet = ROOT / 'outputs/cctv_codeformer_path_diagnostic_v1_vm'
    assert not (packet / 'outputs').exists(), 'Prepared/unrun status requires no run evidence'
    assert sha((packet / 'protocol.json').read_bytes()) == archive['protocol_sha256']
    assert sha((ROOT / source['design_path']).read_bytes()) == source['design_sha256']
    guides = ('CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_VM.md',
              'CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_PREPARATION.md')
    guide = (ROOT / guides[0]).read_text(encoding='utf-8')
    assert archive['protocol_sha256'] in guide and archive['archive_sha256'] in guide
    assert guide.count('gcloud compute scp ') == 5
    assert '--zone=us-central1-a' in guide and 'tmux new-session -A -s codeformer_path_v1' in guide
    assert 'python -B -u scripts/supervise_codeformer_path_diagnostic_v1.py' in guide
    assert 'zero optimizer updates' in guide
    prefix = (
        '# CodeFormer path diagnostic V1 ready for manual transfer — 15 October 2026\n\n'
        '**CURRENT: VERIFIED / PREPARED / UNRUN; zero optimizer updates.**\n'
        'The next step measures training/inference precision and illumination/fusion\n'
        'paths before choosing another restoration training recipe. It is not V3\n'
        'training or a new application model. Moderate helpful brightening remains\n'
        'accepted; protect facial detail, contrast and visible appearance.\n\n'
        '[Five exact upload/tmux/launch/download steps](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_VM.md).\n'
        '[Verified scope and acceptance limits](CCTV_CODEFORMER_PATH_DIAGNOSTIC_V1_PREPARATION.md).\n'
        'Upload: 7,445,777 bytes (7.45 MB). Require 6 GiB free after install; retain\n'
        'a 4 GiB reserve. Worker limit 900 seconds; export limit 300 seconds. Actual\n'
        'VM timing remains unmeasured. Existing V2 assets are read-only dependencies.\n\n'
        'Independent source and archive reviews passed: 36 CPU fixtures, 57 exact\n'
        'archive members, eight exact saved TRAIN inputs and four recorded clean512\n'
        'targets. Actual neural/gradient execution has not run. All new measurements\n'
        'stay human-launched in tmux on the existing L4. No automatic launch or resume.\n\n'
        'V2 remains quality-stopped at 50/600 updates. Paired-image gains, including\n'
        'higher SSIM on 50/53 HQ blur cases, coexist with three clear-face regressions.\n'
        'Its original guard failures, states and the untouched CodeFormer benchmark\n'
        'are preserved. No existing failed recipe receives additional epochs.\n\n'
        'The earlier local offload is complete: 12,394,485,961 logical bytes were\n'
        'removed after verified Drive backup; its free-space readings are historical.\n'
        'No further files were deleted or transferred to Drive during this preparation.\n'
        'Restoration acceptance, native/final evaluation and full app verification\n'
        'remain incomplete; detector/completion development stays deferred. The full\n'
        'thesis goal remains active. Earlier preparation/result statuses below are\n'
        'retained unchanged history, superseded by this current preparation status.\n\n---\n\n'
    ).encode('utf-8')
    originals = {name: (ROOT / name).read_bytes() for name in FILES}
    for name, previous in originals.items():
        assert not previous.startswith(prefix.split(b'\n')[0]), 'Preparation already published'
        write_new(PUBLICATION / 'prior_documents' / name, previous)
        write_new(PUBLICATION / 'staged_documents' / name, prefix + previous)
    entries = []
    for name, previous in originals.items():
        target = ROOT / name
        staged = PUBLICATION / 'staged_documents' / name
        assert target.resolve() == ROOT.resolve() / name
        assert staged.resolve().is_relative_to(ROOT.resolve())
        assert target.is_file() and not target.is_symlink()
        assert target.read_bytes() == previous, 'Concurrent edit: ' + name
        staged.replace(target)
        assert target.read_bytes() == prefix + previous
        entries.append({'file': name, 'previous_bytes': len(previous),
                        'previous_sha256': sha(previous), 'prefix_bytes': len(prefix),
                        'published_sha256': sha(target.read_bytes()),
                        'prior_copy': (PUBLICATION / 'prior_documents' / name).relative_to(ROOT).as_posix(),
                        'unchanged_suffix_verified': True})
    evidence = [*guides, source['design_path']]
    evidence += ['outputs/codeformer_path_diagnostic_v1_preparation/' + name for name in
                 ('stage_inputs.json', 'stage_readback_v1.json', 'independent_source_review_r1.json',
                  'sealed_sources.json', 'execution_archive.json', 'independent_packet_archive_readback_v1.json')]
    receipt = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'prepared_unrun_zero_optimizer_updates',
               'protocol_sha256': archive['protocol_sha256'], 'archive_sha256': archive['archive_sha256'],
               'documents': entries, 'evidence_sha256': {name: sha((ROOT / name).read_bytes()) for name in evidence},
               'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
               'model_qualified': False, 'full_goal_complete': False}
    write_new(PUBLICATION / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'documents_updated': len(entries),
                      'prior_bytes_retained': True, 'manual_diagnostic_prepared_unrun': True,
                      'model_qualified': False}))


if __name__ == '__main__':
    main()
