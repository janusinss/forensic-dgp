"""Stage bound TRAIN pixels and reviewed source; never call a model or VM."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_bounded_fusion_feasibility_v1 import contract as c

REVIEW = ROOT / 'outputs/codeformer_fp32_comparison_v1_return_review'
PREPARATION = ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_preparation'
PACKET = ROOT / 'outputs' / c.PACKAGE_DIR
PARENT = ROOT / 'outputs/cctv_codeformer_adaptation_v2_vm'
RETURNED = ROOT / 'outputs/codeformer_fp32_comparison_v1_return'
DESIGN = 'CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_DESIGN.md'
CALIBRATION = 'fixed_gate_TRAIN_input_readiness_v1.json'
SELECTION = 'bounded_fusion_feasibility_TRAIN_selection_v1.json'
CAL_SHA = '5d8dbbf3b660a911dd76eac187524ff0097b4fbba24d56b8a71a1a25ff04b064'
SEL_SHA = '44a999a9c88baf77cc393310a1c6452b74d3f329211b2d8fe1f18bb7af42a86e'
SHARED = ('research/__init__.py', 'research/codeformer_adaptation_v1/__init__.py',
    'research/codeformer_adaptation_v1/common.py', 'research/codeformer_adaptation_v2/__init__.py',
    'research/codeformer_adaptation_v2/model.py', 'third_party/__init__.py',
    'third_party/codeformer/__init__.py', 'third_party/codeformer/codeformer_arch.py',
    'third_party/codeformer/vqgan_arch.py', 'third_party/codeformer/registry.py',
    'third_party/codeformer/LICENSE', 'third_party/codeformer/NOTICE.md')
OWN = ('research/codeformer_bounded_fusion_feasibility_v1/__init__.py',
    'research/codeformer_bounded_fusion_feasibility_v1/contract.py',
    'research/codeformer_bounded_fusion_feasibility_v1/worker.py',
    'research/codeformer_bounded_fusion_feasibility_v1/neural.py',
    'research/codeformer_bounded_fusion_feasibility_v1/NEURAL_INTERFACE.md',
    'scripts/supervise_codeformer_bounded_fusion_feasibility_v1.py',
    'scripts/prepare_codeformer_bounded_fusion_feasibility_v1.py')


def write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(payload); output.flush(); os.fsync(output.fileno())


def write_json(path, payload):
    write_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def ident(row):
    return 'train_u%04d_slot%d' % (row['update'], row['slot'])


def stage():
    import numpy as np
    from PIL import Image
    c.require(c.sha256(ROOT / DESIGN) == c.DESIGN_SHA256, 'Prospective design changed')
    c.require(c.sha256(REVIEW / CALIBRATION) == CAL_SHA and c.sha256(REVIEW / SELECTION) == SEL_SHA,
              'Frozen input-only calibration/selection changed')
    chain = c.read_json(REVIEW / 'closed_chain_v1.json')
    published = c.read_json(REVIEW / 'publication_readback_v1.json')
    readiness = c.read_json(REVIEW / 'design_pixel_readiness_v1.json')
    c.require(chain['complete'] and published['complete'] and readiness['complete'], 'Closed prior return/readiness required')
    c.require(not chain['model_qualified'] and not chain['clear_guard_pass'], 'Retain stopped model classification')
    for key, row in c.ASSETS.items():
        source = (PARENT / row['path']) if key == 'original' else (RETURNED / row['path'])
        c.require(source.stat().st_size == row['bytes'] and c.sha256(source) == row['sha256'], 'External VM dependency local copy differs')
    c.require(not PACKET.exists() and not PREPARATION.exists(), 'Preserve prior staging evidence; no automatic overwrite')
    PREPARATION.mkdir(); PACKET.mkdir()
    before_docs = {name: c.sha256(ROOT / name) for name in (
        'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', DESIGN)}
    for name in SHARED + OWN:
        payload = (ROOT / name).read_bytes()
        if name in SHARED:
            c.require(c.sha256(PARENT / name) == sha_bytes(payload), 'Shared neural source drift: ' + name)
        if name.endswith('.py'):
            ast.parse(payload.decode('utf-8-sig'), filename=name, feature_version=(3, 10))
        write_bytes(PACKET / name, payload)
    for name in before_docs:
        write_bytes(PACKET / 'instructions' / name, (ROOT / name).read_bytes())
    provenance = {name: c.sha256(REVIEW / name) for name in (
        CALIBRATION, SELECTION, 'closed_chain_v1.json', 'publication_readback_v1.json',
        'design_pixel_readiness_v1.json', 'independent_report_readback_v2.json',
        'next_design_state_review_v1.json')}
    for name in provenance:
        write_bytes(PACKET / 'provenance' / name, (REVIEW / name).read_bytes())
    calibration = c.read_json(REVIEW / CALIBRATION)
    selected = c.read_json(REVIEW / SELECTION)
    rows = []
    for original in calibration['inputs']:
        uid = ident(original)
        source = ROOT / original['input_path']
        c.require(c.sha256(source) == original['input_sha256'], 'Audited saved input PNG changed')
        destination = 'data/inputs/' + uid + '.png'
        write_bytes(PACKET / destination, source.read_bytes())
        with Image.open(source) as im:
            c.require(im.mode == 'RGB' and im.size == (256, 256), 'Expected exact retained RGB256 PNG')
            rgb8 = np.asarray(im).copy()
        rgb = rgb8.astype(np.float64) / 255.
        z = float((rgb @ np.array([.2126, .7152, .0722]))[32:224, 32:224].mean())
        c.require(abs(z - original['central192_sRGB_mean_luminance']) <= 1e-14, 'Input-only photometry differs')
        rows.append({**original, 'id': uid, 'input_path': destination,
            'input256_rgb8_sha256': sha_bytes(rgb8.tobytes()),
            'input256_float32_sha256': sha_bytes(np.ascontiguousarray(rgb8.astype(np.float32) / np.float32(255)).tobytes())})
    lookup = {row['id']: row for row in rows}
    refs = {row['id']: row for row in c.read_json(PARENT / 'references.json')['references']}
    cohort, referenced = [], {}
    for row in selected['cases']:
        case = dict(lookup[ident(row)])
        ref = refs[case['reference_id']]
        c.require(ref['historical_role'] == 'train' and ref['decision'] == 'accept_clean_restoration', 'TRAIN references only')
        rid = case['reference_id']
        if rid not in referenced:
            source = PARENT / ref['source_path']
            c.require(c.sha256(source) == ref['source_sha256'], 'Audited genuine clean source changed')
            write_bytes(PACKET / ref['source_path'], source.read_bytes())
            with Image.open(source) as im:
                c.require(im.mode == 'RGB' and im.size == (1024, 1024), 'Audited source must be RGB1024')
                target512 = im.resize((512, 512), Image.Resampling.LANCZOS)
                target256 = im.resize((256, 256), Image.Resampling.LANCZOS)
                c.require(sha_bytes(np.asarray(target512).tobytes()) == ref['proposed_512_rgb_sha256'], 'Direct clean512 derivation differs')
                for size, target in [(512, target512), (256, target256)]:
                    import io
                    output = io.BytesIO(); target.save(output, format='PNG')
                    write_bytes(PACKET / ('data/clean%d/%s.png' % (size, rid)), output.getvalue())
            referenced[rid] = ref
        case.update({'source_path': ref['source_path'], 'source_sha256': ref['source_sha256'],
            'target512_path': 'data/clean512/' + rid + '.png', 'target256_path': 'data/clean256/' + rid + '.png'})
        for size in (256, 512):
            case['target%d_sha256' % size] = c.sha256(PACKET / case['target%d_path' % size])
        cohort.append(case)
    write_json(PACKET / 'references.json', {'format': 'bounded-fusion-TRAIN-reference-provenance-v1',
        'references': list(referenced.values()), 'unique_photographic_references': 16,
        'identity_overlap_limits': 'Reference IDs are not independently proven unique people; existing FFHQ/pretraining overlap limits remain',
        'target_derivation': 'Source RGB1024 directly to Pillow LANCZOS512 and LANCZOS256; targets never route into gate or network',
        'native_CCTV_inputs': 0, 'reserved_final_pixels': 0})
    launch = '''import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]))\nfrom research.codeformer_bounded_fusion_feasibility_v1.worker import main\nif __name__ == '__main__':\n    raise SystemExit(main())\n'''
    write_bytes(PACKET / 'scripts/cctv_codeformer_bounded_fusion_feasibility_v1_vm.py', launch.encode())
    shell = '''#!/usr/bin/env bash
set -euo pipefail
test -n "${TMUX:-}" || { echo 'Open the manual tmux session first'; exit 1; }
test "$#" -eq 1
export PYTHONDONTWRITEBYTECODE=1
python -B -u scripts/supervise_codeformer_bounded_fusion_feasibility_v1.py --root . --protocol-sha "$1"
'''
    write_bytes(PACKET / 'scripts/run_bounded_fusion_feasibility_v1.sh', shell.encode())
    runtime = c.read_json(RETURNED / 'outputs/fp32_comparison_v1/runtime.json')
    draft = deepcopy(c.SPEC)
    draft.update({'seed': 20261015, 'cases': cohort, 'calibration_inputs': rows,
        'canary_ids': [ident(row) for row in selected['canaries']],
        'runtime_expected': {key: runtime['device' if key == 'GPU' else key]
            for key in ('torch', 'numpy', 'CUDA', 'GPU', 'capability')},
        'input_policy': 'Exact retained TRAIN PNG bytes decoded RGB8/255 to FP32; not a replay of original continuous training tensors',
        'source_photographic_references': 16, 'final_or_native_inputs': 0,
        'measurement_never_qualifies_or_automatically_advances_training': True,
        'interpretation_rule': 'Compare all32 constant/gated matched cases, separately normal/dark and both stress canaries. Report individual changes, conditional context, preservation and dark visibility together. No target-based gate/beta selection, parameter grid, automatic pass or training advance. Independent return and root visual/user review required; full61/53 DEV guards unchanged for future trained candidates.',
        'prior_return_archive_sha256': chain['archive_sha256'], 'provenance_sha256': provenance})
    write_json(PREPARATION / 'protocol_draft.json', draft)
    staged = {p.relative_to(PACKET).as_posix(): {'sha256': c.sha256(p), 'bytes': p.stat().st_size}
        for p in PACKET.rglob('*') if p.is_file()}
    receipt = {'complete': True, 'status': 'STAGED / UNSEALED / UNRUN', 'files': staged,
        'files_count': len(staged), 'payload_bytes': sum(row['bytes'] for row in staged.values()),
        'document_source_sha256': before_docs, 'shared_sources': list(SHARED),
        'case_count': len(cohort), 'calibration_count': len(rows), 'clean_reference_count': len(referenced),
        'local_neural_gradient_optimizer_VM_calls': 0, 'model_qualified': False,
        'preparer_sha256': c.sha256(__file__)}
    write_json(PREPARATION / 'staging_v1.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('files', 'document_source_sha256', 'shared_sources')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', action='store_true', required=True)
    parser.parse_args()
    stage()
