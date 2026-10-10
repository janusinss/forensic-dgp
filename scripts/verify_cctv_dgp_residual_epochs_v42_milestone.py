"""Independent readback of preparation and preserved history; no neural imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_residual_epochs_v42_milestone'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def main():
    r = json.loads((OUT/'record.json').read_text()); assert r['complete']
    for name, value in r['bindings_sha256'].items(): assert sha(ROOT/name) == value, name
    for name, value in r['historical_source_bindings'].items(): assert sha(ROOT/name) == value, name
    previous = (OUT/'before/PROJECT_HANDOFF.md').read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    assert current.endswith(previous)
    assert sha(OUT/'before/PROJECT_HANDOFF.md') == r['preceding_handoff_sha256']
    assert sha(ROOT/'PROJECT_HANDOFF.md') == r['new_handoff_sha256']
    p = json.loads((ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json').read_text())
    for name, value in p['assets_sha256'].items(): assert sha(ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42'/name) == value
    for name, value in p['local_sources_sha256'].items(): assert sha(ROOT/name) == value
    assert p['training_launched'] is False and p['app_promotion'] is False
    assert p['native_or_reserved_used'] is False and p['goal_complete'] is False
    guide = (ROOT/'CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md').read_text()
    pin = sha(ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json')
    assert pin in guide and 'tmux new-session -A -s dgp_residual_epochs_v42' in guide
    assert guide.count('gcloud compute scp') == 5
    for line in guide.splitlines():
        if line.startswith('gcloud compute scp'):
            assert line.count('janusdominic0@forensic-dgp-thesis:') == 1
    assert '8 GiB free after install' in guide and '3,905 updates' in guide
    normalized_guide = ' '.join(guide.split())
    assert '50 raw inference previews' in normalized_guide and 'not independently recomputed' in normalized_guide
    result = {'complete': True, 'bindings_rechecked': len(r['bindings_sha256']),
        'historical_bindings_rechecked': len(r['historical_source_bindings']),
        'packet_assets_rechecked': len(p['assets_sha256']),
        'exact_previous_handoff_suffix': True, 'manual_commands_consistent': True,
        'VM_connections': 0, 'neural_calls': 0, 'optimizer_updates': 0,
        'training_launched': False, 'app_promoted': False, 'goal_complete': False}
    with (OUT/'independent_readback.json').open('x', encoding='utf-8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(result)


if __name__ == '__main__': main()
