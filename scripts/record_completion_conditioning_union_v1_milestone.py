"""Preserve the entire previous handoff and bind completed conditioning evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_conditioning_union_v1_milestone'
RUN = ROOT / 'outputs/completion_conditioning_union_v1'
PREVIOUS = ROOT / 'outputs/cctv_dgp_v35_return_v36_probe_milestone'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve milestone; no automatic repeat'
    old = read(PREVIOUS / 'milestone.json')
    previous_audit = read(PREVIOUS / 'independent_closure_audit.json')
    assert old['complete'] and previous_audit['complete'] and previous_audit['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    for name, digest in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
        assert time.monotonic() - started < 300
    p, r, audit, visual, review_audit = [read(RUN / name) for name in ['protocol.json', 'results.json',
                           'independent_saved_output_audit.json', 'visual_review.json', 'independent_review_support_audit.json']]
    assert audit['complete'] and review_audit['complete'] and visual['complete'] and r['complete']
    assert r['forwards'] == {'completion': 29, 'internal512': 29, 'DGP': 0, 'detector': 0}
    assert r['state_before'] == r['state_after'] == p['parent_model_state'] and not visual['app_adoption']
    addition = '''
**Completion milestone - 8 October 2026: conditioning isolated; several generated remnants reduced; remaining copied boundaries and glare retained.**

The complete 36-case exposed photographic gallery receives a changed, finite
conditioning test. All final reviewed removal masks stay fixed; only the frozen
model's hidden context changes to the union of two existing reviewed masks.
All nine input pages are actually inspected before protocol freezing, with a
separate audit of 32 masks, 18 pair memberships and 132 preview cells. One fresh
current-mask Off call reproduces its cached PNG and both raw stages exactly.

The existing local CPU runtime completes 29 frozen CodeFormer completion
forwards in 152.31 seconds: one parity check plus 28 changed-context trials.
Four empty controls bypass exactly and four input-only exclusions remain.
The unchanged component state is
255e18cbd071e735e5e9789ba83dc010e4872a415048e33611c02c113590afad.
There are zero DGP/detector forwards, gradients, backwards, optimizer updates,
new checkpoints or VM calls. Worker/external stops are 600/630 seconds; artifacts
are 137,774,023 bytes against a 256 MiB cap. All failures and historical recipes
remain; no failed immutable inference or training is repeated.

The separate 6.28-second saved-output audit verifies all actual neural inputs,
28 raw-to-PNG compositions, 32 outputs, 160 page cells, 397 source bindings and
107 artifacts. The 5,483,088 bytes outside final support and 1,075,314 protected
bytes are exact. This includes clear-frame/ordinary-hair context that was hidden
from the model but must still be copied into the delivered image. Display
selection uses the same final support and no new enhancement.

All 32 estimates and their same-final-support baselines are actually viewed at
256-pixel cell detail. Several estimates contain less regenerated dark eyewear,
obstructing hair, petal or green/brown leaf material. Some central hand/scarf
features are cleaner or remain plausible. Strong glare, upper-hand notches,
lower hand/jaw contacts and several peripheral joins remain unqualified. Softness
or unknown hidden appearance alone is not treated as failure. A saved-pixel
recount confirms that both peripheral flower-tip windows are entirely copied;
the upper-hand notch window is mostly copied. These post-output windows are
processing diagnostics, not new covering labels, mask changes or hidden truth.

The union needs TWO historical assisted masks. It is not a validated automatic
single-mask context rule, and no new automatic output is generated. The source
photo assistance is reused for its degraded pair; these are previously exposed
photos, not native CCTV or independent final images. Pretraining overlap and
hidden appearance remain unknown. No new hidden PSNR/SSIM, recovered identity,
ethnicity or Zamboanga-performance claim follows. Neither automatic nor assisted
whole-scope qualification is established; no variant is adopted into the app.

The own-trained DGP remains the main restorer. All 14 current app/checkpoint
bindings and the existing design, Auto/On/Off routing, review-before-generation
and downloads remain. No frontend change requires another browser run here.
V35's preservation failures remain binding. V36's verified changed-direction
packet remains available for manual L4/tmux execution; no V36 return is present
locally at this milestone, and the agent launches no VM work. Useful native DGP
structure, consistent automatic/assisted covering outputs and independent final
review still remain. Goal active/incomplete.

[Completion conditioning findings and all-case evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_CONDITIONING_UNION_V1_RESULTS.md>)
[V36 exact manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FINITE_CLEARANCE_PROBE_V36_VM.md>)

The complete preceding handoff follows. Its earlier covering observations are
preserved; this diagnostic changes no historical protocol or gate outcome.

'''
    OUT.mkdir()
    (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    saved = OUT / 'before_docs/PROJECT_HANDOFF.md'
    saved.write_bytes(before)
    at = before.index(b'\n') + 1
    handoff.write_bytes(before[:at] + addition.encode('utf-8') + before[at:])
    bindings = {**p['sources_sha256'], **p['app_preservation_sha256']}
    for path in sorted(RUN.rglob('*')):
        if path.is_file():
            bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ['PROJECT_HANDOFF.md', saved.relative_to(ROOT).as_posix(),
                 'CCTV_DGP_COMPLETION_CONDITIONING_UNION_V1_RESULTS.md',
                 'scripts/record_completion_conditioning_union_v1_review.py',
                 'scripts/verify_completion_conditioning_union_v1_review.py',
                 'scripts/record_completion_conditioning_union_v1_milestone.py',
                 'scripts/verify_completion_conditioning_union_v1_milestone.py',
                 'outputs/cctv_dgp_v35_return_v36_probe_milestone/milestone.json',
                 'outputs/cctv_dgp_v35_return_v36_probe_milestone/independent_closure_audit.json',
                 'outputs/cctv_dgp_finite_clearance_probe_v36_preparation/independent_packet_audit.json',
                 'outputs/cctv-dgp-finite-clearance-probe-v36-execution.tar.gz',
                 'outputs/cctv-dgp-finite-clearance-probe-v36-execution.tar.gz.sha256']:
        bindings[name] = sha(ROOT / name)
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    m = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
         'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'previous_handoff_path': saved.relative_to(ROOT).as_posix(),
         'new_evidence_sha256': bindings, 'previous_bindings': len(old['new_evidence_sha256']),
         'document': {'before_sha256': hashlib.sha256(before).hexdigest(),
                      'after_sha256': sha(handoff), 'addition_bytes': len(addition.encode('utf-8'))},
         'completion_forwards_in_comparison': 29, 'completion_forwards_in_recording': 0,
         'DGP_or_detector_forwards': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
         'V36_VM_launches_here': 0, 'app_changes': False, 'app_adoption': False,
         'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
         'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    assert m['seconds'] < 300
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(m, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'bindings': len(bindings), 'seconds': m['seconds'],
                      'milestone_sha256': sha(OUT / 'milestone.json')}), flush=True)


if __name__ == '__main__':
    main()
