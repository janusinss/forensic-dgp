"""Prepare a finite manual-training packet; no neural execution or VM mutation."""
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
import time
from cctv_dgp_finite_guard_v1_r1_contract import NAME, STEM, BUDGETS, FRACTIONS, PARITY, definitions, validate, read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm_return'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_preparation'


def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(value)


def shell():
    source = (OLD/'scripts/run_group_conflicts.sh').read_text().replace('group_conflicts_v1', 'finite_guard_v1_r1')
    source = source.replace('1530s', '1830s').replace('330s', '390s')
    source = source.replace('Pass the exact group-conflicts diagnostic protocol SHA256', 'Pass the exact finite-guard protocol SHA256')
    source = source.replace('Diagnostic exit code:', 'Mechanics worker exit code:')
    return source


def guide(pin, digest, size, projection):
    return f'''# Finite preservation checks before another epoch run

Prepared for the existing L4 VM; use only after the independent packet audit
in outputs/cctv_dgp_finite_guard_v1_r1_preparation/ is complete. Not launched.
This separate mechanics study recomputes64 cohort/source/profile gradients.
It tests at most9 placements and accepts at most3 parameter changes. Accepted
changes ARE training, even though no torch optimizer is constructed. Both
100-case TRAIN cohorts now guide fitting; neither is independent DEV/final.

The original1% quality requirement is reported at every placement. A small
accepted mechanics change is not a qualified model or completed epoch. Three
rejected proposals or a failed direction certificate stop and export. No resume,
automatic next study, historical-pilot launch or app promotion is permitted.

Estimated run **10â€“20 minutes**, export **1â€“6 minutes**, extrapolated from
the preceding L4 diagnostic. New64-group runtime/compression is unmeasured.
Enforced: worker1800s/external1830s; cache120s; per state gradients240s,
solver120s, proposals180s; export360s/external390s;30s termination grace;
20GiB allocated VRAM;3GiB uncompressed retained return;512MiB reserve.
Require **7GiB free after installation**, including space for the return archive.
Observed prior-size projection is {projection:,} bytes; live projection and
conservative per-file storage stops still apply. No deletion is bundled.

Packet: {size:,} bytes. Protocol: `{pin}`.
Execution SHA256: `{digest}`.

The VM was stopped when inspected. Start the existing instance manually in
Google Cloud before uploading. To start it from **Windows Google Cloud SDK Shell**:

```bat
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

This command starts billed VM runtime. It has not been executed by Codex.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

If there is less than7GiB free, stop here. Storage maintenance needs a new live
inventory and exact backup/hash-bound candidates. Preserve research assets,
all original checkpoints, splits, gate failures and active work.

3. Open tmux:

```bash
tmux new-session -A -s dgp_finite_guard_v1_r1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_finite_guard_v1_r1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_finite_guard.sh {pin}
```

Detach with Ctrl+B, release, D. Reattach:
`tmux attach-session -t dgp_finite_guard_v1_r1`.
An export receipt with `complete: true` only confirms the archive was created.
It includes the actual number of accepted changes and preserves worker failures.
A zero exit code can also mean all proposals were rejected cleanly.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Return all three files for independent saved-vector/decision/CPU-output audit
and full visual review before any further training decision. Individual autograd
queries are not independently replayed locally. All native CCTV/paired DEV,
independent final, covering-family and full application requirements remain.
The full goal remains active/incomplete. See CCTV_DGP_FINITE_GUARD_V1_R1_DESIGN.md.
'''


def main():
    started = time.monotonic(); assert not OUT.exists() and not PREP.exists()
    parent = read(OLD/'protocol.json')
    for name,digest in parent['assets_sha256'].items(): assert sha(OLD/name) == digest
    for name,digest in parent['local_sources_sha256'].items(): assert sha(ROOT/name) == digest
    audit_path = ROOT/'outputs/cctv_dgp_group_conflicts_v1_independent_audit.json'
    visual_path = ROOT/'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/visual_review.json'
    audit,visual = read(audit_path),read(visual_path)
    assert audit['complete'] and audit['all_stored_row_comparisons_and_failure_decisions_exact']
    assert visual['complete'] and visual['unique_model_outputs_reviewed'] == 400
    assert read(ROOT/'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/gallery_independent_audit.json')['complete']
    source_names = ['contract','candidate','training','vm']
    for name in source_names+['__return_checker__']:
        file = ROOT/'scripts'/('audit_cctv_dgp_finite_guard_v1_r1_return.py' if name == '__return_checker__' else 'cctv_dgp_finite_guard_v1_r1_'+name+'.py')
        ast.parse(file.read_text(),feature_version=(3,10))
    OUT.mkdir(); PREP.mkdir(); mapping = {}
    for name in sorted(parent['assets_sha256']):
        if name.startswith('evidence/') or name in ['cctv_dgp_group_conflicts_v1_contract.py','cctv_dgp_group_conflicts_v1_candidate.py',
                'scripts/cctv_dgp_group_conflicts_v1_vm.py','scripts/run_group_conflicts.sh']: continue
        dest = OUT/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(OLD/name,dest)
        mapping[name] = (OLD/name).relative_to(ROOT).as_posix()
    evidence = {'initial_parameters.npy': OLD/'evidence/initial_parameters.npy',
        'parent_protocol.json': OLD/'protocol.json','parent_results.json':RETURN/'outputs/results.json',
        'parent_independent_audit.json':audit_path,'parent_visual_review.json':visual_path,
        'parent_gallery_audit.json':ROOT/'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/gallery_independent_audit.json',
        'parent_analysis.json':ROOT/'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/finite_direction_analysis.json'}
    for name,origin in evidence.items():
        dest = OUT/'evidence'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(origin,dest)
        mapping['evidence/'+name] = origin.relative_to(ROOT).as_posix()
    for part in source_names:
        name = 'cctv_dgp_finite_guard_v1_r1_'+part+'.py'; dest = OUT/('scripts' if part == 'vm' else '')/name
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/'scripts'/name,dest)
        mapping[dest.relative_to(OUT).as_posix()] = 'scripts/'+name
    text(OUT/'scripts/run_finite_guard.sh',shell())
    p = copy.deepcopy(parent)
    for key in ['terms','scopes','gradient_queries','optimizer_updates','committed_trajectory_updates','trial_variants',
            'parent_result_sha256','parent_audit_sha256','parent_archive_sha256','parent_protocol_sha256']: p.pop(key,None)
    gradient_bytes = (RETURN/'outputs/group_gradients.npz').stat().st_size
    baseline_bytes = sum(q.stat().st_size for q in (RETURN/'outputs/baseline').rglob('*') if q.is_file())
    overhead = int(2*3*gradient_bytes*1.15) + 256*1024**2
    projection = int(10*(baseline_bytes+2*1024**2)*1.25 + overhead)
    assert projection <= BUDGETS['return_uncompressed_bytes']
    p.update({'format':'own-DGP-finite-recertified-mechanics-v1','UTC':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Recompute cohort-specific directions and use actual finite raw/PNG backtracking to test stable preserved microlearning.',
        'architecture_direction':'Unchanged separate current-DGP spatial and decoder copy; change finite learning mechanics and fitting coverage only.',
        'cohort_role_history':{co['name']: 'TRAIN; formerly derivative cohort' if i == 0 else 'TRAIN; formerly cross-check only, now also fitting' for i,co in enumerate(p['cohorts'])},
        'former_cross_cohort_used_for_fitting':True,'group_losses':definitions(p['cases'],p['cohorts']),
        'relative_displacement_fractions':FRACTIONS,'maximum_gradient_queries':960,'maximum_trial_variants':9,
        'maximum_accepted_parameter_changes':3,'accepted_changes_are_actual_training':True,'epochs':0,
        'proposal_rule':'At each state solve bounded64-gradient unit-normalized simplex; require all descent cosines>=1e-7; backtrack2e-5,1e-5,5e-6 using original decoder L2; cast float32 once.',
        'mechanics_acceptance':'Strictly positive overall structure, nonnegative source structure, all unchanged finite preservation/brightness requirements against original AND preceding state in raw AND PNG; negative actual64 derivative dots; no quality qualification.',
        'previous_mean_shift_anchor':'preceding accepted raw output; original anchor retained separately',
        'budgets':BUDGETS,'loss_surrogate_parity':PARITY,'resume_permitted':False,
        'raw_storage':'All64 float64 group vectors and320 query metadata per state; all100 baseline and every proposal raw/PNG/embeddings, both mean-shift anchor metrics (preceding-only PNG is reconstructed), rejected decisions and accepted full cloned states/RNG; individual query vectors explicitly omitted.',
        'parent_protocol_sha256':sha(OLD/'protocol.json'),'parent_archive_sha256':audit['archive_sha256'],
        'parent_audit_sha256':sha(audit_path),'parent_result_sha256':sha(RETURN/'outputs/results.json'),
        'prior_baseline_bytes':baseline_bytes,'prior_32_group_compressed_bytes':gradient_bytes,
        'projected_gradient_and_snapshot_bytes':overhead,'projection_basis':'Six prior32-group compressed sizes plus15percent;256MiB vector/snapshot/metadata allowance;125percent of10 prior baseline traces.',
        'local_audit_maximum_seconds':2400,'copied_source_mapping':mapping})
    p['assets_sha256'] = {q.relative_to(OUT).as_posix():sha(q) for q in sorted(OUT.rglob('*')) if q.is_file()}
    extra = ['scripts/prepare_cctv_dgp_finite_guard_v1_r1.py','scripts/revise_cctv_dgp_finite_guard_v1_storage.py',
        'scripts/audit_cctv_dgp_finite_guard_v1_r1_return.py','scripts/verify_cctv_dgp_finite_guard_v1_r1_packet.py',
        'CCTV_DGP_FINITE_GUARD_V1_R1_DESIGN.md','CCTV_DGP_GROUP_CONFLICTS_V1_RESULTS.md',
        'outputs/cctv_dgp_group_conflicts_v1_independent_audit.json',
        'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/visual_review.json',
        'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/gallery_independent_audit.json',
        'outputs/cctv_dgp_group_conflicts_v1_return_review_v1/finite_direction_analysis.json']
    extra += ['scripts/cctv_dgp_finite_guard_v1_r1_'+name+'.py' for name in source_names]
    p['local_sources_sha256'] = {**parent['local_sources_sha256'],**{n:sha(ROOT/n) for n in extra}}
    validate(p); write(OUT/'protocol.json',p); pin = sha(OUT/'protocol.json')
    for q in OUT.rglob('*.py'): ast.parse(q.read_text(),feature_version=(3,10))
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for q in sorted(OUT.rglob('*')):
            if not q.is_file(): continue
            info = tar.gettarinfo(str(q),arcname=NAME+'/'+q.relative_to(OUT).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with q.open('rb') as stream: tar.addfile(info,stream)
    digest = sha(archive); text(Path(str(archive)+'.sha256'),digest+'  '+archive.name+'\n')
    text(ROOT/'CCTV_DGP_FINITE_GUARD_V1_VM.md',guide(pin,digest,archive.stat().st_size,projection))
    write(PREP/'prepared.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':digest,
        'archive_bytes':archive.stat().st_size,'assets':len(p['assets_sha256']),'cases':100,
        'maximum_training_changes':3,'optimizer_updates_executed':0,'neural_calls':0,
        'VM_training_launched':False,'independent_packet_audit_pending':True,
        'projected_return_bytes':projection,'seconds':time.monotonic()-started})
    print({'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size})


if __name__ == '__main__': main()
