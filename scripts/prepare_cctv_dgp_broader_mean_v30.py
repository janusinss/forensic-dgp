"""Prepare one coverage-change pilot; no models, gradients or VM connection."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_vm_v29'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
NAME = 'cctv_dgp_broader_mean_vm_v30'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_broader_mean_v30_preparation'
STEM = 'cctv-dgp-broader-mean-v30'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def worker():
    text = (OLD / 'scripts/cctv_dgp_mean_centered_decoder_v29_vm.py').read_text()
    text = text.replace('own-DGP-mean-centered-original-decoder-capacity-v29', 'own-DGP-broader-mean-centered-original-decoder-v30')
    text = text.replace("p['epochs'] == 80", "p['epochs'] == 800 / 781")
    text = text.replace('cctv_dgp_mean_centered_decoder_vm_v29', NAME)
    text = text.replace('cctv_dgp_mean_centered_v29_training', 'cctv_dgp_broader_mean_v30_training')
    text = text.replace('cctv-dgp-mean-centered-decoder-v29', STEM)
    text = text.replace('cctv_dgp_mean_centered_decoder_v29_return', 'cctv_dgp_broader_mean_v30_return')
    text = text.replace('V29 preflight300s/worker1800s cap', 'V30 proof300s/worker4500s cap')
    text = text.replace('V29 worker cap1800 seconds', 'V30 worker cap4500 seconds').replace('signal.alarm(1800)', 'signal.alarm(4500)')
    text = text.replace("'cap_seconds':2100", "'cap_seconds':4800").replace('elapsed<=2130', 'elapsed<=4830')
    text = text.replace('Need2GiB free; no deletion by this diagnostic', 'Need6GiB free; preserve all historical data')
    text = text.replace('Decoder proof export cap90 seconds', 'V30 export cap900 seconds')
    text = replace_once(text, "        base = parent_check(parent, p)\n", "        broader_check(root, p)\n        base = parent_check(parent, p)\n")
    text = replace_once(text, "        base = parent_check(parent,p)\n        closed_v28_and_diagnostic_check(parent,p)", "        broader_check(root, p)\n        base = parent_check(parent,p)\n        closed_v28_and_diagnostic_check(parent,p)")
    text = replace_once(text, "lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p))", "lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p),broader_check(root,p))")
    text = replace_once(text, "'cases':50,'optimizer_updates':0,'neural_or_gradient_calls':0", "'initial_proof_cases':50,'training_cases':3905,'training_references':781,'optimizer_updates':0,'neural_or_gradient_calls':0")
    text = replace_once(text, "    with tarfile.open(destination,'x:gz',compresslevel=3) as tar:", "    assert shutil.disk_usage(root).free >= sum(path.stat().st_size for path in files) + 16*1024**2, 'Insufficient export space; preserve failure and outputs'\n    with tarfile.open(destination,'x:gz',compresslevel=3) as tar:")
    addition = '''
def broader_check(root, p):
    mixed = root.parent / 'cctv_dgp_mixed_vm_v9_r2'
    assert sha(mixed / 'mixed_protocol_v9.json') == p['mixed_data_protocol_sha256']
    original = read(mixed / 'mixed_protocol_v9.json')
    assert len(original['training_cases']) == 3905
    assert [r for r in original['references'] if r['role'] == 'train'] == p['training_references']
    assert {r['id'] for r in p['training_references']}.isdisjoint({r['id'] for r in original['references'] if r['role'] != 'train'})
    for name, digest in p['mixed_TRAIN_assets_sha256'].items():
        path = (mixed / name).resolve()
        assert path.is_relative_to(mixed) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    closed = root.parent / 'cctv_dgp_mean_centered_decoder_vm_v29'
    for name, digest in p['closed_V29_evidence_sha256'].items():
        path = (closed / name).resolve()
        assert path.is_relative_to(closed) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    result = read(closed / 'outputs/results.json')
    assert result['optimizer_updates'] == 800 and result['necessary_capacity_pass']
    assert p['optimizer_updates'] == 800 and len(p['case_rows']) == 3905
    assert p['initial_proof_case_rows'] == read(root.parent / 'cctv_dgp_feature_skips_vm_v27/protocol.json')['cases']
    assert len(p['mixed_TRAIN_assets_sha256']) == 5467

'''
    text = replace_once(text, '\ndef run(root, parent, p, pin):', addition + '\ndef run(root, parent, p, pin):')
    return text


def training():
    text = (OLD / 'cctv_dgp_mean_centered_v29_training.py').read_text()
    text = replace_once(text, "    out = root/'outputs'", "    from cctv_dgp_broader_mean_v30_cache import load_items\n    items = load_items(root, p, original, identity, canonical_tensor, clock, write)\n    out = root/'outputs'")
    text = replace_once(text, "all(0<=i<50 for i in row)", "all(0<=i<3905 for i in row)")
    text = replace_once(text, "    for begin in range(0,800,10):assert sorted(i for row in schedule[begin:begin+10] for i in row)==list(range(50))", "    assert sorted(i for row in schedule[:781] for i in row)==list(range(3905))\n    assert len({i for row in schedule[781:] for i in row})==95")
    text = text.replace('dgp_candidate_v29.pth', 'dgp_candidate_v30.pth')
    text = text.replace('range(0,50,5)', 'range(0,3905,5)')
    text = replace_once(text, "                    np.save(folder/(cid+'.npy'),a,allow_pickle=False);Image.fromarray(png).save(folder/(cid+'.png'))", "                    if cid in p['preview_case_ids']: np.save(folder/(cid+'.npy'),a,allow_pickle=False)\n                    Image.fromarray(png).save(folder/(cid+'.png'))")
    text = replace_once(text, "                    metric['constant_mean_shift_only_MSE']=", "                    Image.fromarray(mean_png).save(folder/(cid+'_mean_only.png'))\n                    metric['constant_mean_shift_only_MSE']=")
    text = replace_once(text, "                    if update==0:\n                        with Image.open(out/'initial_baseline'", "                    if update==0 and cid in p['preview_case_ids']:\n                        with Image.open(out/'initial_baseline'")
    text = replace_once(text, "'training_outputs':50", "'training_outputs':3905")
    text = text.replace("'fit_cap_seconds':1500,'worker_cap_seconds':1800", "'fit_cap_seconds':3600,'worker_cap_seconds':4500")
    text = text.replace('progress[\'epochs\']=update//10', "progress['epochs']=update/781;progress['completed_epochs']=update//781")
    text = text.replace('<1500', '<3600').replace('<=1500', '<=3600').replace('V29 fitting cap1500 seconds', 'V30 fitting cap3600 seconds')
    text = replace_once(text, "                projected=elapsed+(800-20)*float(np.mean(step_times[1:]))*1.25+120", "                remaining_snapshot_allowance=3*baseline['snapshot_duration_seconds']*1.25\n                projected=elapsed+(800-20)*float(np.mean(step_times[1:]))*1.25+remaining_snapshot_allowance")
    text = replace_once(text, "'overhead_seconds':120,'projected_seconds':projected,'cap_seconds':1500", "'overhead_seconds':remaining_snapshot_allowance,'projected_seconds':projected,'cap_seconds':3600")
    text = replace_once(text, "        clock();folder=out/('update'+str(update));folder.mkdir();rows=[]", "        clock();snapshot_started=time.monotonic();folder=out/('update'+str(update));folder.mkdir();rows=[]")
    text = replace_once(text, "        write(folder/'metrics.json',receipt);snapshots.append(receipt)", "        receipt['snapshot_duration_seconds']=time.monotonic()-snapshot_started\n        write(folder/'metrics.json',receipt);snapshots.append(receipt)")
    text = text.replace("print(f'V29 update", "print(f'V30 update")
    text = replace_once(text, "'trained_tensors':12,", "'trained_tensors':12,'training_references':781,'training_cases':3905,'coverage_changed_only':True,")
    # Independent progression remains explicit while full-cohort snapshots run.
    text = replace_once(text, "                clock();group=items[begin:begin+5];b=batch(list(range(begin,begin+5)))", "                clock();group=items[begin:begin+5];b=batch(list(range(begin,begin+5)))\n                if begin%250==0:print({'V30_snapshot':update,'cases':begin,'of':3905},flush=True)")
    return text


def main():
    started = time.monotonic()
    assert not BUNDLE.exists() and not PREP.exists()
    audit = read(ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json')
    paired = read(ROOT / 'outputs/cctv_dgp_v29_paired_development_v1/results.json')
    assert audit['complete'] and len(paired['diagnostic_preservation_failures']) == 21
    assert read(ROOT / 'outputs/cctv_dgp_v29_paired_development_v1/saved_output_audit.json')['complete']
    assert read(ROOT / 'outputs/cctv_dgp_v29_native_development_v1/visual_review.json')['restoration_qualified'] is False
    old = read(OLD / 'protocol.json'); base = read(PARENT / 'protocol.json')
    mixed = read(MIXED / 'mixed_protocol_v9.json')
    assert sha(MIXED / 'mixed_protocol_v9.json') == '6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70'
    references = [r for r in mixed['references'] if r['role'] == 'train']
    refs = {r['id']: r for r in references}; assert len(refs) == 781
    cases = []
    for row in mixed['training_cases']:
        ref = refs[row['reference_id']]
        cases.append({**row, 'source_person_or_reference': ref['id'], 'role': 'train',
                      'target': ref['target'], 'observed': ref['observed'],
                      'landmarks5_canvas_xy': ref['landmarks5'][0]})
    assert len(cases) == len({c['id'] for c in cases}) == 3905
    data_names = {c['input'] for c in cases} | {r[k] for r in references for k in ['target', 'observed']}
    assert len(data_names) == 5467
    data_hashes = {name: sha(MIXED / name) for name in sorted(data_names)}
    assert all(mixed['assets_sha256'][name] == digest for name, digest in data_hashes.items())
    byid = {c['id']: c for c in cases}
    for c in base['cases']:
        fresh = byid[c['id']]
        for key in ['input', 'target', 'observed']: assert sha(PARENT / c[key]) == data_hashes[fresh[key]]
        assert c['landmarks5_canvas_xy'] == fresh['landmarks5_canvas_xy']
    rng = random.Random(293001); first = list(range(3905)); second = first.copy()
    rng.shuffle(first); rng.shuffle(second)
    flat = first + second[:95]
    schedule = [flat[i:i+5] for i in range(0,4000,5)]
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    (BUNDLE / 'cctv_dgp_mean_centered_decoder_v29.py').write_bytes((OLD / 'cctv_dgp_mean_centered_decoder_v29.py').read_bytes())
    (BUNDLE / 'cctv_dgp_app_input_v28.py').write_bytes((OLD / 'cctv_dgp_app_input_v28.py').read_bytes())
    (BUNDLE / 'cctv_dgp_broader_mean_v30_cache.py').write_bytes((ROOT / 'scripts/cctv_dgp_broader_mean_v30_cache.py').read_bytes())
    (BUNDLE / 'cctv_dgp_broader_mean_v30_training.py').write_text(training(), encoding='utf-8', newline='\n')
    (BUNDLE / 'scripts/cctv_dgp_broader_mean_v30_vm.py').write_text(worker(), encoding='utf-8', newline='\n')
    write(BUNDLE / 'schedule.json', {'seed':293001,'batches':schedule,'updates':800,'first_epoch_complete':781,'second_epoch_batches':19})
    shell = (OLD / 'scripts/run_v29.sh').read_text().replace('cctv_dgp_mean_centered_decoder_v29_vm.py', 'cctv_dgp_broader_mean_v30_vm.py')
    shell = shell.replace('2100s', '4800s').replace('150s', '930s')
    (BUNDLE / 'scripts/run_v30.sh').write_text(shell, encoding='utf-8', newline='\n')
    p = copy.deepcopy(old)
    p.update({'format':'own-DGP-broader-mean-centered-original-decoder-v30','date':'2026-10-07',
              'purpose':'Test broader existing TRAIN coverage after V29 capacity succeeds but520 DEV cases show21 regressions and native24 lacks useful clarity.',
              'design':'Same original initialization, mean-centered spatial path, selected12 parameters, seven losses, fixed initial50 normalizers and AdamW. Only optimization data coverage and its predetermined schedule change.',
              'difference_from_failed_recipes':'V29 repeatedly optimizes10 references over800 updates; V30 uses781 already approved TRAIN references over the same800 updates. No DEV, native or reserved training, no V9 weights/optimizer, no V29 continuation or gate weakening.',
              'epochs':800/781,'training_batches_per_epoch':781,'completed_epochs_bound':1,'second_epoch_batches_bound':19,
              'training_references':references,'case_rows':cases,'initial_proof_case_rows':base['cases'],
              'preview_case_ids':[c['id'] for c in base['cases']],
              'mixed_data_protocol_sha256':sha(MIXED/'mixed_protocol_v9.json'),'mixed_TRAIN_assets_sha256':data_hashes,
              'normalizer_policy':'Retain the exact original50-case gradient preflight and its normalizers before loading the broader cohort; no loss rescaling.',
              'coverage_policy':'First781 updates expose all3905 cases exactly once; next19 expose95 predetermined cases from a second shuffle. Exactly4000 samples and800 updates, as V29. Epoch count consequently changes.',
              'capacity_policy':'All3905 TRAIN outputs at0/50/400/800; same17 group/metric gates,50-update1% structure stop,final80010% structure, both source gains nonnegative and mean-only fraction<=20%. These remain necessary TRAIN gates, not native utility.',
              'return_policy':'All TRAIN PNGs and mean-control PNGs, vectors and metrics at all four states; raw arrays for fixed50 preview cases; checkpoints and initial70-query proof. No full RGB cache export. Separate independent saved-output audit plus all-final3905 forward-only replay and checkpoint-preview replay.',
              'next':'Independently audit returned result and review50 fixed previews; freeze a repeat of all520 photographic DEV cases and native24 comparison. No automatic app promotion or final identity opening.'})
    p['budgets'].update({'cache_seconds':900,'fit_seconds':3600,'worker_seconds':4500,'external_seconds':4800,
                         'export_seconds':900,'external_export_seconds':930,'minimum_free_disk_bytes':6*1024**3,
                         'maximum_export_uncompressed_bytes':3*1024**3,
                         'overhead_seconds':'3 * measured full snapshot0 duration *1.25'})
    returned = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_return'
    closed_names = ['protocol.json',*old['assets_sha256'],'outputs/results.json','outputs/update800/metrics.json',
                    'outputs/update800/dgp_candidate_v29.pth','outputs/gradient_preflight.json','outputs/execution_receipt.json','trainer_exit_code.txt']
    p['closed_V29_evidence_sha256'] = {name: sha(returned / name) for name in closed_names}
    basis = ['CCTV_DGP_MEAN_CENTERED_DECODER_V29_RESULTS.md','scripts/prepare_cctv_dgp_broader_mean_v30.py',
             'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json',
             'outputs/cctv_dgp_v29_paired_development_v1/results.json','outputs/cctv_dgp_v29_paired_development_v1/saved_output_audit.json',
             'outputs/cctv_dgp_v29_paired_development_v1/visual_review.json',
             'outputs/cctv_dgp_v29_native_development_v1/saved_output_audit.json','outputs/cctv_dgp_v29_native_development_v1/visual_review.json',
             'outputs/cctv_dgp_v29_single_input_parity_v1/results.json']
    p['local_basis_sha256'] = {name:sha(ROOT/name) for name in basis}
    p['assets_sha256'] = {f.relative_to(BUNDLE).as_posix():sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()}
    for path in BUNDLE.rglob('*.py'): ast.parse(path.read_text(),feature_version=(3,10))
    write(BUNDLE / 'protocol.json', p)
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME+'/'+path.relative_to(BUNDLE).as_posix(), recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f: f.write(sha(archive)+'  '+archive.name+'\n')
    receipt = {'complete':True,'protocol_sha256':sha(BUNDLE/'protocol.json'),'archive_sha256':sha(archive),
               'archive_bytes':archive.stat().st_size,'packet_files':len(p['assets_sha256'])+1,
               'TRAIN_assets_verified':len(data_hashes),'initial50_exact_source_parity':True,
               'training_cases':3905,'training_references':781,'finite_updates':800,
               'optimizer_losses_architecture_initialization_and_numeric_gates_unchanged':True,
               'original_data_or_weights_uploaded':False,'local_neural_or_gradient_calls':0,
               'actual_VM_training_started':False,'human_manual_VM_execution_required':True,
               'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(PREP/'preparation.json',receipt); print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
