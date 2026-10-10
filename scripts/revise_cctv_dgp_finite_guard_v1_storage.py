"""Preserve first packet; derive R1 with reconstruction of redundant diagnostic PNGs."""
import ast
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = ROOT/'outputs/cctv_dgp_finite_guard_v1_preparation'
    p = json.loads((ROOT/'outputs/cctv_dgp_finite_guard_v1_vm/protocol.json').read_text())
    audit = json.loads((before/'independent_packet_audit.json').read_text())
    assert audit['complete'] and not audit['VM_training_launched']
    for name,digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    copied = []
    names = ['cctv_dgp_finite_guard_v1_contract.py','cctv_dgp_finite_guard_v1_candidate.py',
        'cctv_dgp_finite_guard_v1_training.py','cctv_dgp_finite_guard_v1_vm.py',
        'prepare_cctv_dgp_finite_guard_v1.py','verify_cctv_dgp_finite_guard_v1_packet.py',
        'audit_cctv_dgp_finite_guard_v1_return.py']
    for name in names:
        text = (ROOT/'scripts'/name).read_text().replace('finite_guard_v1','finite_guard_v1_r1').replace('finite-guard-v1','finite-guard-v1-r1')
        if name.endswith('_vm.py'):
            line = "                        Image.fromarray(ppng).save(folder/(cid+'_previous_mean_only.png'))\n"
            assert text.count(line) == 1; text = text.replace(line,'')
        if name.startswith('audit_'):
            line = "        assert np.array_equal(ppng, pixels(folder/label/(cid+'_previous_mean_only.png')))\n"
            assert text.count(line) == 1; text = text.replace(line,'')
        if name.startswith('verify_'):
            line = '    assert "previous_mean_only.png" in worker.read_text()'
            assert text.count(line) == 1
            text = text.replace(line, '    assert "previous_constant_mean_shift_only_MSE" in worker.read_text() and "previous_reference_variant" in worker.read_text()')
        if name.startswith('prepare_'):
            text = text.replace("['contract','candidate','training','vm']", "['contract','candidate','training','vm']")
            # This R1 script derives already prepared sources; no nonexistent R1 builder is claimed.
            text = text.replace('scripts/build_cctv_dgp_finite_guard_v1_r1_sources.py','scripts/revise_cctv_dgp_finite_guard_v1_storage.py')
            text = text.replace('CCTV_DGP_FINITE_GUARD_V1_DESIGN.md','CCTV_DGP_FINITE_GUARD_V1_R1_DESIGN.md')
            text = text.replace('both mean-shift anchors, rejected decisions', 'both mean-shift anchor metrics (preceding-only PNG is reconstructed), rejected decisions')
            # Explicit twenty-KiB/case metadata cushion includes both anchor reports.
            text = text.replace('10*baseline_bytes*1.25 + overhead','10*(baseline_bytes+2*1024**2)*1.25 + overhead')
        if name.startswith('verify_'):
            text = text.replace("10*p['prior_baseline_bytes']*1.25+overhead", "10*(p['prior_baseline_bytes']+2*1024**2)*1.25+overhead")
        destination = ROOT/'scripts'/name.replace('finite_guard_v1','finite_guard_v1_r1')
        ast.parse(text,feature_version=(3,10))
        with destination.open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
        copied.append(destination.relative_to(ROOT).as_posix())
    original_baseline = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm_return/outputs/baseline'
    extra = sum(q.stat().st_size for q in original_baseline.glob('*_mean_only.png'))
    failed_projection = int(12.5*(p['prior_baseline_bytes']+extra)+p['projected_gradient_and_snapshot_bytes'])
    assert failed_projection > p['budgets']['return_uncompressed_bytes']
    receipt = {'complete':True,'initial_protocol_sha256':sha(ROOT/'outputs/cctv_dgp_finite_guard_v1_vm/protocol.json'),
        'initial_packet_audit_sha256':sha(before/'independent_packet_audit.json'),
        'reason':'Redundant preceding-anchor PNG files were absent from baseline storage projection.',
        'prior_duplicate_PNG_bytes_per_variant':extra,'conservative_projection_with_duplicates':failed_projection,
        'cap_bytes':p['budgets']['return_uncompressed_bytes'],
        'initial_packet_superseded_do_not_run':True,'replacement':'cctv_dgp_finite_guard_v1_r1_vm',
        'new_source_files':copied,'old_sources_and_packet_preserved':True,'scientific_thresholds_changed':False,
        'primary_raw_PNG_outputs_omitted':False,'VM_training_launched':False}
    with (before/'storage_preflight_correction.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2); stream.write('\n')
    print({'complete':True,'original_packet_retained':True,'new_sources':len(copied),'neural_calls':0})


if __name__ == '__main__': main()
