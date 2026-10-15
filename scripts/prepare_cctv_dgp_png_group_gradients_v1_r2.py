"""Fresh canonical-PNG-forward repair, only for manual finite L4 measurement."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from import_cctv_dgp_png_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write

BASE = 'cctv_dgp_png_group_gradients_v1_r2'
NAME = BASE + '_vm'
STEM = 'cctv-dgp-png-group-gradients-v1-r2'
OLD_BASE = 'cctv_dgp_png_group_gradients_v1_r1'

PIXEL_PREFLIGHT = '''        # Canonical byte values are checked on the real CUDA device before
        # loading any model or creating any gradient graph.
        progress.update(stage='canonical_PNG_pixel_preflight')
        with torch.inference_mode():
            byte_cpu = np.tile(np.arange(256,dtype=np.uint8),768).reshape(1,3,256,256)
            canonical32 = byte_cpu.astype(np.float32)/np.float32(255)
            canonical64 = byte_cpu.astype(np.float64)/255.
            byte_gpu = torch.from_numpy(byte_cpu).cuda()
            raw_probe = torch.from_numpy(canonical32).cuda()
            rgb_probe,feature_probe = delivered_surrogates(raw_probe,byte_gpu)
            returned32 = rgb_probe.cpu().numpy()
            returned64 = feature_probe.cpu().numpy()
            assert np.array_equal(returned32.view(np.uint32),canonical32.view(np.uint32)), 'Canonical CUDA RGB32 table differs'
            assert np.array_equal(returned64.view(np.uint64),canonical64.view(np.uint64)), 'Canonical CUDA feature64 table differs'
            legacy32 = (byte_gpu.to(torch.float32)/255.).cpu().numpy()
            table = out/'canonical_PNG_pixel_table.npz'
            np.savez_compressed(table,RGB32=returned32.reshape(-1)[:256],feature64=returned64.reshape(-1)[:256],
                legacy_CUDA_division_RGB32=legacy32.reshape(-1)[:256])
            write(out/'canonical_PNG_pixel_preflight.json',dict(complete=True,all256_byte_levels=True,
                RGB32_and_feature64_bit_exact_to_NumPy=True,table_sha256=sha(table),
                canonical_source='NumPy uint8/255 once on CPU, copied without recasting',
                legacy_CUDA_division_mismatch_levels=int(np.count_nonzero(legacy32.reshape(-1)[:256].view(np.uint32)!=canonical32.reshape(-1)[:256].view(np.uint32))),
                local_or_VM_model_forwards_at_preflight=0,gradient_queries_at_preflight=0,optimizer_updates=0,
                backward_is_approximate_and_not_executed_by_pixel_preflight=True))
            del byte_cpu,canonical32,canonical64,byte_gpu,raw_probe,rgb_probe,feature_probe,returned32,returned64,legacy32
        clock()
'''


def one(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)


def build_worker():
    source = RETURN / 'scripts' / (OLD_BASE + '_vm.py')
    old = source.read_text()
    text = old.replace(OLD_BASE + '_vm', NAME).replace('cctv-dgp-png-group-gradients-v1-r1', STEM).replace(
        'outputs/png_group_gradients_v1_r1', 'outputs/png_group_gradients_v1_r2')
    text = one(text, 'from cctv_dgp_png_group_gradients_v1_losses import', 'from cctv_dgp_png_group_gradients_v1_r2_losses import')
    marker = "        original, _ = load_frozen_dgp_restorer"
    text = one(text, marker, PIXEL_PREFLIGHT + marker)
    inverse = text.replace(PIXEL_PREFLIGHT, '').replace(NAME, OLD_BASE + '_vm').replace(STEM, 'cctv-dgp-png-group-gradients-v1-r1').replace(
        'outputs/png_group_gradients_v1_r2', 'outputs/png_group_gradients_v1_r1').replace(
        'from cctv_dgp_png_group_gradients_v1_r2_losses import', 'from cctv_dgp_png_group_gradients_v1_losses import')
    assert inverse == old
    ast.parse(text, feature_version=(3,10))
    return text, sha(source)


def build_contract():
    source = RETURN / 'cctv_dgp_png_group_gradients_v1_contract.py'
    return source.read_text(), sha(source)


def main():
    packet = OUT / NAME
    archive = OUT / (STEM + '-execution.tar.gz')
    receipt = OUT / (BASE + '_preparation.json')
    assert not any(path.exists() for path in [packet, archive, receipt])
    imported, failure = read(ANALYSIS / 'import.json'), read(ANALYSIS / 'failure_audit.json')
    assert imported['complete'] and failure['complete'] and not imported['complete_diagnostic_reported']
    assert read(ANALYSIS / 'canonical_boundary_review.json')['complete']
    failed = read(RETURN / 'outputs/png_group_gradients_v1_r1/failure.json')
    assert failed['progress']['counts'] == dict(B=2,recognizer=0,original=0,gradient_queries=0,backwards=0,optimizer_updates=0,parameter_displacements=0)
    assert 'np.array_equal(pred_png.detach()' in failed['traceback']
    forward_path = OUT / (BASE + '_forward_audit.json')
    forward = read(forward_path)
    helper = ROOT / 'scripts/cctv_dgp_png_group_gradients_v1_r2_losses.py'
    assert forward['complete'] and forward['saved_forward_cases'] == 100 and forward['loss_helper_sha256'] == sha(helper)
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name,digest in protected.items())
    old = read(RETURN / 'protocol.json')
    worker, source_sha = build_worker()
    assert forward['prepared_worker_draft_sha256'] == __import__('hashlib').sha256(worker.encode()).hexdigest()
    packet.mkdir(); (packet / 'scripts').mkdir()
    for name,digest in old['assets_sha256'].items():
        assert sha(RETURN / name) == digest
        if name.startswith('scripts/') or name == 'cctv_dgp_png_group_gradients_v1_losses.py':
            continue
        shutil.copyfile(RETURN / name, packet / name)
    with (packet / 'scripts' / (NAME + '.py')).open('x',encoding='utf-8',newline='\n') as f:f.write(worker)
    shutil.copyfile(helper,packet / helper.name)
    shell = (RETURN / 'scripts/run_png_groups.sh').read_text().replace(OLD_BASE + '_vm.py',NAME + '.py')
    with (packet / 'scripts/run_png_groups.sh').open('x',encoding='utf-8',newline='\n') as f:f.write(shell)
    p = dict(old)
    p.update(name=NAME,stem=STEM,UTC=datetime.now(timezone.utc).isoformat(),
        surrogate_definition='canonical NumPy byte/255 copied to device plus exactly zero identity-gradient terms; exact scientific SSIM plus zero analytic-gradient term',
        canonical_RGB32_and_feature64_values_computed_on_CPU_once=True,
        CPU_neural_or_gradient_calls_not_used_for_canonical_values=True,
        all256_CUDA_byte_levels_required_before_model_loading=True,
        numerical_repair=dict(retained_failed_protocol_sha256=sha(RETURN / 'protocol.json'),
            failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'),
            boundary_review_sha256=sha(ANALYSIS / 'canonical_boundary_review.json'),
            forward_construction_and_pre_model_pixel_test_only=True,
            exact_raw_PNG_and_scientific_metric_checks_not_weakened=True,
            same320_gradient_queries_zero_updates_and_all_execution_limits=True),
        local_saved_value_forward_audit_sha256=sha(forward_path))
    p['derivation_sha256'] = dict(p['derivation_sha256'], repaired_worker_source=source_sha,loss_source=sha(helper))
    p['assets_sha256'] = {path.relative_to(packet).as_posix():sha(path) for path in packet.rglob('*') if path.is_file()}
    assert len(p['assets_sha256']) == 6
    write(packet / 'protocol.json',p); pin=sha(packet / 'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(path for path in packet.rglob('*') if path.is_file()):
            tar.add(file,arcname=NAME+'/'+file.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    importer=(ROOT/'scripts'/('import_'+OLD_BASE+'_return.py')).read_text().replace(OLD_BASE,BASE).replace(
        'cctv-dgp-png-group-gradients-v1-r1',STEM).replace(sha(RETURN/'protocol.json'),pin)
    importer_path=ROOT/'scripts'/('import_'+BASE+'_return.py')
    with importer_path.open('x',encoding='utf-8',newline='\n') as f:f.write(importer)
    auditor=(ROOT/'scripts'/('audit_'+OLD_BASE+'_return.py')).read_text().replace('import_'+OLD_BASE+'_return','import_'+BASE+'_return').replace(
        'outputs/png_group_gradients_v1_r1','outputs/png_group_gradients_v1_r2')
    marker="    run=read(out/'run_results.json');env=read(out/'environment.json')"
    add='''    pixel=read(out/'canonical_PNG_pixel_preflight.json')
    assert pixel['complete'] and pixel['all256_byte_levels'] and pixel['RGB32_and_feature64_bit_exact_to_NumPy']
    assert pixel['local_or_VM_model_forwards_at_preflight']==pixel['gradient_queries_at_preflight']==pixel['optimizer_updates']==0
    table=out/'canonical_PNG_pixel_table.npz';assert sha(table)==pixel['table_sha256']
    with np.load(table,allow_pickle=False) as z:
        assert set(z.files)=={'RGB32','feature64','legacy_CUDA_division_RGB32'}
        assert z['RGB32'].dtype==np.float32 and z['feature64'].dtype==np.float64
        assert all(z[key].shape==(256,) and np.isfinite(z[key]).all() for key in z.files)
        np.testing.assert_array_equal(z['RGB32'].view(np.uint32),(np.arange(256,dtype=np.float32)/np.float32(255)).view(np.uint32))
        np.testing.assert_array_equal(z['feature64'].view(np.uint64),(np.arange(256,dtype=np.float64)/255.).view(np.uint64))
        assert int(np.count_nonzero(z['legacy_CUDA_division_RGB32'].view(np.uint32)!=z['RGB32'].view(np.uint32)))==pixel['legacy_CUDA_division_mismatch_levels']
'''
    auditor=one(auditor,marker,add+marker)
    auditor_path=ROOT/'scripts'/('audit_'+BASE+'_return.py')
    ast.parse(importer,feature_version=(3,10));ast.parse(auditor,feature_version=(3,10))
    with auditor_path.open('x',encoding='utf-8',newline='\n') as f:f.write(auditor)
    assert all(sha(ROOT / name)==value for name,value in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        local_forward_audit_sha256=sha(forward_path),retained_return_import_sha256=sha(ANALYSIS/'import.json'),
        retained_failure_audit_sha256=sha(ANALYSIS/'failure_audit.json'),
        importer_sha256=sha(importer_path),prospective_return_auditor_sha256=sha(auditor_path),
        canonical_pixels_changed_no_scientific_gate_change=True,manual_VM_run_unrun=True,
        no_agent_VM_connection_or_launch=True,local_neural_gradient_optimizer_calls=0,
        more_epochs_qualified=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,name=NAME,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size))


if __name__=='__main__':main()
