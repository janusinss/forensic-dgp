"""Publish the verified manual preparation milestone; preserve prior bytes."""
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_preparation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def put(path, value):
    with Path(path).open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    sealed = read(EVIDENCE / 'execution_archive_v1.json')
    reviewed_path = EVIDENCE / 'independent_execution_guide_readback_v1.json'
    assert sha(reviewed_path) == 'c18114080abe7da894d417d492f8823476a75ef4f04e0d802ce6881ad0daf51e'
    reviewed = read(reviewed_path)
    assert reviewed['complete'] and reviewed['approved_for_manual_guide_publication']
    assert not reviewed['model_qualified'] and reviewed['training_authorized'] is False
    assert reviewed['no_run_outputs_in_packet'] and reviewed['neural_gradient_optimizer_VM_network_calls'] == 0
    assert not reviewed['findings']
    assert sealed['complete'] and sealed['status'] == 'VERIFIED / PREPARED / UNRUN'
    assert sha(ROOT / sealed['archive']) == sealed['archive_sha256']
    packet = ROOT / 'outputs/cctv_codeformer_adain_gradient_isolation_v1_vm'
    assert sha(packet / 'protocol.json') == sealed['protocol_sha256']
    assert not (packet / 'outputs').exists()
    guide = ROOT / 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_VM.md'
    report = ROOT / 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_PREPARATION.md'
    assert sha(guide) == 'b3bc125014e3f8bb60f444f4870def45c13f62d9ea1d798c727414586bcaadb1'
    assert sha(report) == '4612b221a4b20f69dd28fa7eaf98b6afeb3913b0da68001f262062827f0ac195'
    assert reviewed['archive_sha256'] == sealed['archive_sha256']
    assert reviewed['protocol_sha256'] == sealed['protocol_sha256']
    assert reviewed['guide_sha256'] == sha(guide)
    assert reviewed['preparation_report_sha256'] == sha(report)
    for name, digest in reviewed['evidence_sha256'].items():
        assert sha(ROOT / name) == digest
    old = read(ROOT / 'outputs/codeformer_bounded_fusion_feasibility_v1_return_review/publication_v1/receipt.json')
    before = {}
    for row in old['documents']:
        path = ROOT / row['path']
        assert sha(path) == row['after_sha256']
        before[row['path']] = path.read_bytes()
    assert len(before) == 4
    prefix = f'''# CodeFormer AdaIN gradient isolation V1 ready for manual VM run — 17 October 2026

**CURRENT: VERIFIED / PREPARED / UNRUN. No new model is accepted.**
[Use the five-step upload/install/tmux/download guide](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_VM.md).
[Preparation evidence](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_PREPARATION.md).
[Prospective single-factor design](CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md).

The new upload is1,159,959 bytes, with31 exact regular archive members and a
portable LF checksum. Protocol SHA-256 `{sealed['protocol_sha256']}`.
Archive SHA-256 `{sealed['archive_sha256']}`.
All1,909 reused dependencies and the staged overlay were freshly checked;
48 CPU fixtures and separate independent source/data/resource/archive/guide
reviews pass. None of these checks is a successful VM training result.

The immediately preceding bounded-fusion measurement is audited, with all18
visual sheets reviewed. Its fixed brightness gate worsened dark PNG MAE on13/16
TRAIN examples against the constant bound, while both bounds sacrificed some
useful stopped-model dark improvement. Reject that unchanged gate. Keep all
original/failure states, outputs, data and provenance. The actual return review
below remains authoritative; no earlier failed checkpoint has been promoted.

The new pilot starts from untouched pretrained CodeFormer and changes only the
candidate training image-gradient route through the encoder's AdaIN style
argument. Forward values at identical weights must remain exact. CE/HQ/token
and fusion learning, losses, rates, first200 canonical TRAIN observations and
the existing RAW/PNG preservation guards remain. Before Adam, the manual VM
must pass fresh391-target parity and a four-CodeFormer/four-LPIPS/eight-query
route witness with RNG/state/mode/callable restoration. A causal benefit is
unproven. No new inference architecture, brightness threshold or epoch extension.

**At most50 actual updates**,200 backwards,9 gradient queries and1,614 neural
forwards. The600-position schedule preserves first50 controls; it does not
authorize600 updates. Worker cap3,000 seconds; export600 seconds; separate
preflight300 seconds, all with finite termination grace. Planning estimate
12–20 minutes excludes transfer and remains unmeasured for this new experiment.
Allow18 GiB free before install; verified sibling copies use1,439,410,618 bytes,
leaving at least16 GiB free and retaining a4 GiB reserve. Preserve any stop and
download its evidence. No automatic retry, resume or unchanged failed rerun.

The sealed packet preserves as-of-stage instruction snapshots; this current
header supersedes their historical pending-preparation status. Executable,
protocol, checkpoint and input hashes stay exact. All older root instruction
bytes are retained below and independently backed up before this prefix.

The goal remains our trained CodeFormer adaptation as the primary256×256
restoration candidate, accurately attributed to its existing architecture.
Restoration comes first, compared with untouched CodeFormer and retained DGP
baselines on identical inputs. Useful moderate brightening is acceptable with
visible structure and appearance preserved. Insufficient information requires
a clearer/more frontal/less-covered crop. Preserve the existing app design.
Keep source-specific native unpaired evidence separate from paired synthetic
scores and reserved labeled FINAL identities; no local Zamboanga, ethnicity or
exact recovered-identity claim. User acceptance, independent final review and
full Auto/override/PNG app verification remain required. Detector/completion
development is deferred until restoration acceptance, with its agreed covering
families, visible/protected pixels, mask correction, plausible estimate and
original/mask/result downloads still in the overall scope.

All new actual training is human-launched on the existing L4 in tmux using the
verified guide. Preparation made no VM connection, cleanup, Drive operation,
local neural/gradient/optimizer call, training launch or app change. Storage
maintenance retains its separate inventory-first/hash-bound authorization.
The full goal remains active and incomplete.

---

'''.encode('utf-8')
    out = EVIDENCE / 'publication_v1'
    out.mkdir(exist_ok=False)
    (out / 'prior').mkdir()
    staged = {}
    for name, value in before.items():
        assert (ROOT / name).read_bytes() == value
        put(out / 'prior' / name, value)
        path = ROOT / (name + '.adain-preparation-stage')
        put(path, prefix + value)
        assert path.read_bytes() == prefix + value
        staged[name] = path
    rows = []
    for name, path in staged.items():
        assert (ROOT / name).read_bytes() == before[name]
        os.replace(path, ROOT / name)
        assert (ROOT / name).read_bytes()[len(prefix):] == before[name]
        assert (out / 'prior' / name).read_bytes() == before[name]
        rows.append({'path': name, 'before_sha256': sha(out / 'prior' / name),
            'after_sha256': sha(ROOT / name), 'prior_bytes': len(before[name]),
            'retained_suffix_offset': len(prefix),
            'backup': (out / 'prior' / name).relative_to(ROOT).as_posix(),
            'prior_bytes_retained_exactly': True})
    receipt = {'complete': True, 'status': 'VERIFIED / PREPARED / UNRUN',
        'documents': rows, 'protocol_sha256': sealed['protocol_sha256'],
        'archive_sha256': sealed['archive_sha256'], 'guide_sha256': sha(guide),
        'preparation_report_sha256': sha(report), 'publisher_sha256': sha(__file__),
        'independent_execution_guide_readback_sha256': sha(reviewed_path),
        'immutable_packet_instruction_snapshots_not_changed': True,
        'local_neural_gradient_optimizer_VM_calls': 0, 'training_started': False,
        'model_qualified': False, 'goal_complete': False}
    put(out / 'receipt.json', (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps({'complete': True, 'documents': len(rows), 'training_started': False}))


if __name__ == '__main__':
    main()
