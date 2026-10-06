"""Prepare a different finite loss-alignment pilot, preserving all V23 failures."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_cctv_dgp_detail_skip_v23 import read, require, sha, write
OLD = ROOT/'outputs/cctv_dgp_detail_skip_vm_v23'
NEW = ROOT/'outputs/cctv_dgp_degraded_detail_vm_v24'


def prepare():
    started = time.monotonic()
    require(not NEW.exists(), 'Preserve prior V24 packet')
    require(sha(OLD/'protocol.json') == '4cddb98fb5e6a215cb84f98132cfa5a2146ee157c1cb939919ffbf8c5740e863', 'Original V23 protocol differs')
    old = read(OLD/'protocol.json')
    for name, digest in old['assets_sha256'].items():
        require(sha(OLD/name)==digest, 'Original V23 asset differs: '+name)
    audit = read(ROOT/'outputs/cctv_dgp_detail_skip_v23_independent_audit.json')
    review = read(ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic/visual_review.json')
    loss_audit = read(ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/independent_saved_loss_audit.json')
    require(audit['complete'] and not audit['early_structure_stop']['pass'] and review['complete'] and
            loss_audit['complete'] and loss_audit['clear_contribution_fraction_of_net_objective_reduction']>1,
            'Audited failed V23, all50-case review and loss mismatch evidence required')
    NEW.mkdir(); assets={}
    for name, digest in old['assets_sha256'].items():
        new_name = 'lineage/v23_original_'+Path(name).name if name in ['scripts/cctv_dgp_detail_skip_v23.py','scripts/run_v23.sh'] else name
        dest=NEW/new_name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,dest)
        require(sha(dest)==digest,'Copied asset differs');assets[new_name]=digest
    source=(OLD/'scripts/cctv_dgp_detail_skip_v23.py').read_text(encoding='utf-8')
    for before,after in [('dgp-learned-detail-skip-capacity-v23','dgp-degraded-detail-cohort-capacity-v24'),
                         ('cctv_dgp_detail_skip_v23','cctv_dgp_degraded_detail_v24'),
                         ('cctv-dgp-detail-skip-v23','cctv-dgp-degraded-detail-v24'),('run_v23.sh','run_v24.sh'),('V23 update','V24 update')]:
        source=source.replace(before,after)
    tree=ast.parse(source);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loss=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='loss')
    lines=source.splitlines(keepends=True)
    new_loss='''    def loss(ids):
        b=batch(ids); pred=head(b['x'],b['base'],b['mask'])
        terms=objective_terms(b,pred,identity,mean,feature_errors,ssim,normalizers)
        result=sum(terms.values()).mean()
        assert torch.isfinite(result)
        return result
'''
    source=''.join(lines[:loss.lineno-1])+new_loss+''.join(lines[loss.end_lineno:])
    anchor="    out = root / 'outputs'"
    require(source.count(anchor)==1,'Output anchor differs')
    source=source.replace(anchor,"    from cctv_dgp_degraded_objective_v24 import cohort_normalizers, objective_terms\n    normalizers, loss_setup = cohort_normalizers(items, head)\n"+anchor,1)
    source=source.replace("    out.mkdir()", "    out.mkdir()\n    write(out/'cohort_loss_setup.json', loss_setup)",1)
    anchor="['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine']"
    require(source.count(anchor)==1,'Batch anchor differs')
    source=source.replace(anchor,"['x','base','mask','target','feature','interior','valid7','grid','truth','base_cosine','degraded_weight','clear_weight']")
    ast.parse(source,feature_version=(3,10))
    source_path=ROOT/'scripts/cctv_dgp_degraded_detail_v24.py'
    with source_path.open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    (NEW/'scripts').mkdir(exist_ok=True)
    shutil.copy2(source_path,NEW/'scripts/cctv_dgp_degraded_detail_v24.py')
    assets['scripts/cctv_dgp_degraded_detail_v24.py']=sha(NEW/'scripts/cctv_dgp_degraded_detail_v24.py')
    shutil.copy2(ROOT/'scripts/cctv_dgp_degraded_objective_v24.py',NEW/'cctv_dgp_degraded_objective_v24.py')
    assets['cctv_dgp_degraded_objective_v24.py']=sha(NEW/'cctv_dgp_degraded_objective_v24.py')
    shell=(OLD/'scripts/run_v23.sh').read_text(encoding='utf-8').replace('cctv_dgp_detail_skip_v23','cctv_dgp_degraded_detail_v24')
    with (NEW/'scripts/run_v24.sh').open('x',encoding='utf-8',newline='\n') as f:f.write(shell)
    assets['scripts/run_v24.sh']=sha(NEW/'scripts/run_v24.sh')
    lineage=[ROOT/'outputs/cctv_dgp_detail_skip_v23_independent_audit.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_return/outputs/failure.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_return/outputs/early_structure_stop.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic/results.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic/visual_review.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic/independent_saved_diagnostic_audit.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/plan.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/results.json',
             ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/independent_saved_loss_audit.json']
    for index,path in enumerate(lineage):
        name='lineage/closed_v23_'+str(index).zfill(2)+'_'+path.name;shutil.copy2(path,NEW/name);assets[name]=sha(NEW/name)
    p=copy.deepcopy(old);p['format']='dgp-degraded-detail-cohort-capacity-v24';p['assets_sha256']=assets
    p['purpose']='Necessary exposed-training capacity check of corrected degraded-cohort objective; not app/native/final qualification'
    p['hypothesis']='The clear-case target reward and per-case HF normalization lowered the old global objective while the degraded cohort failed. Optimize degraded-cohort average HF error with clear baseline preservation controls; retain the V23 head and all quality gates.'
    p['difference_from_closed_recipes']='Only training-objective cohort/normalizer/control policy changes. V23/V22 head/data/gates remain failed; no longer budget, retry or source/profile override at inference.'
    p['design']['objective']='1.25*degraded_mask*(landmark_HF/cohort_baseline_degraded_HF+.25 observed_HF/cohort_baseline_degraded_HF+.05 per-case_pixel/base_pixel)+.05*clear_mask*RGB_change_from_baseline/base_pixel + unchanged2 pixel-regression/5 SSIM-regression/5 ArcFace-regression penalties on every case. Normalizers frozen from all40 degraded baselines; original floors; no clear target reward.'
    p['design']['clear_policy']='Ten clear cases remain in every epoch as baseline-preservation controls, not rewarded clean-target examples. All original final quality guards still include clear groups. Metadata flags only affect the training loss, never inference.'
    p['closed_v23_protocol_sha256']=sha(OLD/'protocol.json');p['closed_v23_return_sha256']='b71639cabd17476d5989dfd4582d47de9b747b8bf2a611232e5f62cc67bf9fd9'
    p['loss_alignment_evidence']='CPU exact original-loss decomposition on100 saved states, separately arithmetic-audited:101.4223% of net loss reduction arises from clear cases; degraded objective worsens. Not proof of GPU gradient conflict.'
    p['no_further_blind_attempt']='If this distinct objective experiment fails, stop this head recipe and review the architecture/data/loss assumptions before another training recipe; no automatic follow-on.'
    write(NEW/'protocol.json',p);pin=sha(NEW/'protocol.json')
    with (NEW/'protocol.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(pin+'  protocol.json\n')
    archive=ROOT/'outputs/cctv-dgp-degraded-detail-v24-execution.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for file in sorted(NEW.rglob('*')):
            if file.is_file():
                info=tar.gettarinfo(str(file),NEW.name+'/'+file.relative_to(NEW).as_posix());info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                info.mode=0o755 if file.name=='run_v24.sh' else 0o644
                with file.open('rb') as f:tar.addfile(info,f)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    out=ROOT/'outputs/cctv_dgp_degraded_detail_v24_preparation';out.mkdir()
    receipt={'complete':True,'date':'2026-10-06','scope':'New finite objective-alignment transfer; actual VM preflight/training/quality pending',
             'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,'assets':len(assets),
             'parameters':4613,'same50_cases_same800_updates_same80epochs':True,'quality_gates_unchanged':True,
             'loss_changed_prospectively':True,'seconds':time.monotonic()-started,'neural_calls':0,'optimizer_updates':0,'backward_calls':0,'VM_actions':False,'goal_complete':False}
    write(out/'preparation.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    prepare()
