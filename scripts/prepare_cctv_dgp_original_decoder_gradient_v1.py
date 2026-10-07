"""Build a thin manual-VM derivative packet after the selected decoder review."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_preparation'
BUNDLE = ROOT / 'outputs/cctv_dgp_original_decoder_gradient_v1_vm'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
RETURN = ROOT / 'outputs/cctv_dgp_feature_skips_v27_return'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    start = time.monotonic()
    assert not OUT.exists() and not BUNDLE.exists(), 'Preserve previous packet'
    review_path = ROOT / 'outputs/cctv_dgp_original_decoder_review_v1/review.json'
    review = read(review_path)
    readback = read(review_path.with_name('independent_readback.json'))
    assert review['complete'] and readback['complete'] and readback['review_sha256'] == sha(review_path)
    assert review['raw_and_PNG_initial_parity_cases'] == 50 and review['local_gradient_calls'] == review['local_optimizer_updates'] == 0
    for name,digest in review['source_bindings_sha256'].items():
        assert sha(ROOT / name) == digest, name
    original = read(PARENT / 'protocol.json')
    assert sha(PARENT / 'protocol.json') == review['retained_protocol_sha256']
    basis = {name:sha(ROOT/name) for name in [
        'outputs/cctv_dgp_original_decoder_review_v1/review.json',
        'outputs/cctv_dgp_original_decoder_review_v1/independent_readback.json',
        'outputs/cctv_dgp_post_v27_architecture_review_v1/user_architecture_decision.json',
        'outputs/dgp_feature_skips_v27_audit_milestone/milestone.json',
        'outputs/dgp_feature_skips_v27_audit_milestone/independent_readback.json',
        'scripts/prepare_cctv_dgp_original_decoder_gradient_v1.py',
        'scripts/cctv_dgp_original_decoder_candidate_v1.py',
        'scripts/cctv_dgp_original_decoder_gradient_v1_vm.py']}
    BUNDLE.mkdir()
    (BUNDLE / 'scripts').mkdir()
    sources = {'cctv_dgp_original_decoder_candidate_v1.py':'scripts/cctv_dgp_original_decoder_candidate_v1.py',
               'scripts/cctv_dgp_original_decoder_gradient_v1_vm.py':'scripts/cctv_dgp_original_decoder_gradient_v1_vm.py'}
    for destination, source in sources.items():
        content = (ROOT / source).read_bytes()
        ast.parse(content.decode('utf-8'),feature_version=(3,10))
        with (BUNDLE / destination).open('xb') as f: f.write(content)
    shell = (ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm/scripts/run_gradient.sh').read_bytes()
    shell = shell.replace(b'cctv_dgp_v26_gradient_diagnostic_v1_vm.py', b'cctv_dgp_original_decoder_gradient_v1_vm.py')
    shell = shell.replace(b'480s ',b'660s ').replace(b'60s ',b'120s ')
    with (BUNDLE / 'scripts/run_decoder_gradient.sh').open('xb') as f: f.write(shell)
    original_evidence = {name:sha(RETURN/name) for name in ['outputs/failure.json','outputs/early_structure_stop.json',
                                                        'outputs/execution_receipt.json','outputs/cohort_loss_setup.json']}
    protocol = {'format':'own-DGP-original-decoder-zero-update-gradient-proof-v1', 'date':'2026-10-06',
        'purpose':'Measure initial original reconstruction-decoder gradients after V25-V27 stopped small-head recipes and the user-selected architecture review; no fit or model acceptance',
        'design':'Deep-copy original DGPSynthesizer; enable only head1-head4,smooth,smooth2,final weights. Freeze complete encoder/FPN and all evaluation buffers. Use original forward directly, observed support and pinned canonical as_tensor. No new residual head, postprocessing, target-conditioned inference or optimizer.',
        'decoder_parameters':609219,'decoder_parameter_tensors':14,'parameter_layout':review['parameter_layout'],
        'cases':50,'references':10,'batch_size':5,'batches':10,'component_gradient_calls':70,
        'optimizer_updates':0,'epochs':0,'snapshots':['initial only'],
        'terms':['degraded_landmark_detail','degraded_observed_detail','degraded_pixel','clear_baseline_anchor',
                 'pixel_regression','SSIM_regression','ArcFace_regression'],
        'initial_preservation_terms_exact_zero':['clear_baseline_anchor','pixel_regression','SSIM_regression','ArcFace_regression'],
        'initial_parity':'Every same-batch candidate RGB/PNG must exactly equal the fresh frozen DGP baseline before any derivative; both initial states are identical.',
        'normalization':'Pinned as_tensor on CUDA. Same-batch fresh frozen original baseline. Historical CUDA-scalar cache is retained and separately compared; never claim its PNG byte equality or canonical app qualification.',
        'historical_CUDA_cache_tolerance':2e-6,'CPU_CUDA_raw_replay_tolerance':1e-5,
        'original_DGP_state':review['DGP_state_before_after'],
        'frozen_recognizer_state':'9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963',
        'closed_V27_protocol_sha256':sha(PARENT/'protocol.json'),'closed_V27_evidence_sha256':original_evidence,
        'retained_capacity_gates':original['prospective_gates'],
        'loss_policy':'Compile exact pinned V27 mean/feature_errors/ssim and fixed blur/high; import exact corrected V26 batchmatched_scores/objective_terms and V24 assembly/normalizers. Recompute the unchanged normalizer formula from the new fresh baseline; old scalar receipts remain preserved.',
        'gradient_policy':'Each of seven cohort-weighted mean terms uses autograd.grad over all14 decoder tensors; all pieces must be finite and connected. Save every7x609219 float64 batch matrix plus their sum. Require nonzero improvement gradient at every decoder tensor; all four initial preservation values/gradients must be exactly zero in all ten batches.',
        'budgets':{'worker_seconds':600,'external_seconds':660,'external_kill_grace_seconds':30,
                   'export_seconds':90,'external_export_seconds':120,'external_export_kill_grace_seconds':10,
                   'peak_vram_bytes':20*1024**3,'minimum_free_disk_bytes':2*1024**3,
                   'maximum_export_uncompressed_bytes':900_000_000},
        'execution':'Human manual existing NVIDIA L4/g2-standard-4 in ~/forensic-dgp; gcloud uploads and separate PuTTY downloads; tmux. No assistant VM/cloud connection for this diagnostic.',
        'failure_policy':'Retain partial arrays, failures, logs, timing, original checkpoint and old failures. No resume, unchanged retry, follow-on or threshold search.',
        'next':'Independent source/matrix/CPU inference audit before defining any finite training pilot. Initial gradient proof cannot establish useful structure or justify app promotion.',
        'native_or_reserved_used':False,'completion_training':False,'new_checkpoint_created':False,
        'automatic_follow_on':False,'app_promotion':False,'independent_final_review':False,'goal_complete':False,
        'assets_sha256':{name:sha(BUNDLE/name) for name in [*sources,'scripts/run_decoder_gradient.sh']},
        'local_basis_sha256':basis}
    write(BUNDLE/'protocol.json',protocol)
    pin = sha(BUNDLE/'protocol.json')
    archive = ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-execution.tar.gz'
    assert not archive.exists()
    files = [path for path in BUNDLE.rglob('*') if path.is_file()]
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for path in sorted(files): tar.add(path,arcname=BUNDLE.name+'/'+path.relative_to(BUNDLE).as_posix(),recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:
        f.write(digest+'  '+archive.name+'\n')
    OUT.mkdir()
    write(OUT/'preparation.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
        'packet_regular_files':len(files),'uploaded_data_or_weights_bytes':0,'source_bindings_sha256':basis,
        'raw_PNG_initial_local_parity_cases':50,'local_neural_or_gradient_calls':0,
        'VM_gradient_proof_pending':True,'optimizer_updates':0,'new_training_recipe_created':False,
        'VM_actions':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start})
    print(json.dumps(read(OUT/'preparation.json'),indent=2))


if __name__ == '__main__': main()
