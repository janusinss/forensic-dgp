"""Prepare an immutable full-resolution detail-bypass pilot; no neural/training work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_cctv_dgp_detail_prior_v22_r1 import read, require, sha, write

OLD = ROOT / 'outputs/cctv_dgp_detail_prior_vm_v22_r1'
NEW = ROOT / 'outputs/cctv_dgp_detail_skip_vm_v23'
SOURCE = ROOT / 'scripts/cctv_dgp_detail_skip_v23.py'

HEAD = '''    class DetailHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.direct = nn.Conv2d(13, 3, 3, padding=1)
            self.stem = nn.Conv2d(13, 16, 3, padding=1)
            self.refine = nn.Conv2d(16, 16, 3, padding=1)
            self.tail = nn.Conv2d(16, 3, 1)
            nn.init.zeros_(self.direct.weight)
            nn.init.zeros_(self.direct.bias)
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)
            z = torch.arange(-6, 7, dtype=torch.float32)
            k = torch.exp(-0.5 * (z / 2).square()); k = k / k.sum()
            self.register_buffer('kernel', k)
            self.register_buffer('reflect_indices', torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))))

        def blur(self, value):
            c = value.shape[1]
            vertical = self.kernel.view(1, 1, 13, 1).expand(c, 1, 13, 1)
            horizontal = self.kernel.view(1, 1, 1, 13).expand(c, 1, 1, 13)
            value = F.conv2d(value.index_select(2, self.reflect_indices), vertical, groups=c)
            return F.conv2d(value.index_select(3, self.reflect_indices), horizontal, groups=c)

        def high(self, value):
            return value - self.blur(value)

        def forward(self, x, base, mask):
            features = torch.cat((x, base, self.high(x), self.high(base), mask), dim=1)
            shallow = F.silu(self.refine(F.silu(self.stem(features))))
            q = 0.05 * torch.tanh(self.direct(features) + self.tail(shallow))
            low = self.blur(q * mask) / self.blur(mask).clamp_min(1e-8)
            band = (q - low) * mask
            mean = band.sum((2, 3), keepdim=True) / mask.sum((2, 3), keepdim=True)
            band = band - mean * mask
            return (base + band).clamp(0, 1)
'''

SHELL = r'''#!/usr/bin/env bash
set -u
set -o pipefail
pilot_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
protocol_pin="${1:?Supply the frozen protocol SHA256}"
pilot_python="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
test -x "$pilot_python" || exit 2
test ! -e "$pilot_root/outputs" || exit 3
test ! -e "$pilot_root/trainer.log" || exit 4
cd "$pilot_root" || exit 5
start_mono="$("$pilot_python" -c 'import time;print(time.monotonic())')"
timeout --signal=TERM --kill-after=30s 2100s "$pilot_python" -B -u scripts/cctv_dgp_detail_skip_v23.py --root "$pilot_root" --protocol-sha "$protocol_pin" --run 2>&1 | tee trainer.log
trainer_status=${PIPESTATUS[0]}
end_mono="$("$pilot_python" -c 'import time;print(time.monotonic())')"
printf 'Trainer exit code: %s\n' "$trainer_status"
printf '%s\n' "$trainer_status" > trainer_exit_code.txt
"$pilot_python" -B scripts/cctv_dgp_detail_skip_v23.py --root "$pilot_root" --protocol-sha "$protocol_pin" --record-supervision --supervisor-start "$start_mono" --supervisor-end "$end_mono" --trainer-exit "$trainer_status"
supervision_status=$?
if [ "$supervision_status" -ne 0 ]; then exit "$supervision_status"; fi
timeout --signal=TERM --kill-after=30s 150s "$pilot_python" -B -u scripts/cctv_dgp_detail_skip_v23.py --root "$pilot_root" --protocol-sha "$protocol_pin" --export
export_status=$?
printf 'Export exit code: %s\n' "$export_status"
if [ "$export_status" -ne 0 ]; then exit "$export_status"; fi
exit "$trainer_status"
'''


def replace_once(source, old, new):
    require(source.count(old) == 1, 'Source anchor is not unique: ' + old[:100])
    return source.replace(old, new, 1)


def runtime_source():
    old_path = OLD / 'scripts/cctv_dgp_detail_prior_v22_r1.py'
    require(sha(old_path) == '5e0137e9c5e60220c195f336405887689c64711b2c691cf94cba8e8c56876cac', 'Original runtime differs')
    source = old_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    node = next(n for n in function.body if isinstance(n, ast.ClassDef) and n.name == 'DetailHead')
    lines = source.splitlines(keepends=True)
    source = ''.join(lines[:node.lineno - 1]) + HEAD + ''.join(lines[node.end_lineno:])
    for old, new in [('dgp-learned-detail-capacity-v22', 'dgp-learned-detail-skip-capacity-v23'),
                     ('cctv_dgp_detail_prior_v22_r1', 'cctv_dgp_detail_skip_v23'),
                     ('cctv-dgp-detail-prior-v22-r1', 'cctv-dgp-detail-skip-v23'),
                     ('run_v22.sh', 'run_v23.sh'), ('45443', '4613'), ('V22 update', 'V23 update')]:
        source = source.replace(old, new)
    source = replace_once(source, '    head = DetailHead().cuda().eval()', '''    counts = {'detail_head': 0, 'DGP': 0, 'fixed_recognizer': 0}
    def counter(name):
        def hook(*_): counts[name] += 1
        return hook
    head = DetailHead().cuda().eval()
    head.register_forward_hook(counter('detail_head'))
    head.audit_forward_counts = counts''')
    source = replace_once(source, '    before = state_hash(dgp.net)', "    dgp.net.register_forward_hook(counter('DGP'))\n    before = state_hash(dgp.net)")
    source = replace_once(source, '    identity_state = state_hash(identity)', "    identity.encoder.register_forward_hook(counter('fixed_recognizer'))\n    identity_state = state_hash(identity)")
    source = replace_once(source, "'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}", "'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'neural_forward_counts': dict(counts)}")
    source = replace_once(source, '    snapshots = []', '''    snapshots = []
    fit_start = None
    step_times = []
    optimizer_constructed = False
    def execution(terminal):
        now = time.monotonic()
        write(out/'execution_receipt.json', {'complete': True, 'protocol_sha256': pin, 'terminal': terminal,
            **progress, 'worker_seconds': now-start, 'fit_seconds': None if fit_start is None else now-fit_start,
            'fit_cap_seconds': 1500, 'worker_cap_seconds': 1800,
            'step_times_seconds': step_times, 'neural_forward_counts': dict(head.audit_forward_counts),
            'optimizer_constructed': optimizer_constructed, 'optimizer': 'AdamW new head only',
            'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'VRAM_cap_bytes': 20*1024**3,
            'recognizer_state': state_hash(identity), 'initial_recognizer_state': preflight['recognizer_state'],
            'head_state': state_hash(head), 'app_promotion': False, 'goal_complete': False})''')
    source = replace_once(source, "        head.zero_grad(set_to_none=True)\n        require_vm(root)", '''        direct_gradient = float(head.direct.weight.grad.double().square().sum())
        assert direct_gradient > 0, 'Nonzero full-resolution bypass gradient required'
        write(out/'one_batch_gradient_preflight.json', {'complete': True, 'backwards': 1,
            'optimizer_constructed': False, 'direct_gradient_sum_squares': direct_gradient,
            'per_tensor_gradient_sum_squares': {name: float(value.grad.double().square().sum()) for name,value in head.named_parameters()},
            'recognizer_has_no_gradients': all(value.grad is None for value in identity.parameters()),
            'neural_forward_counts': dict(head.audit_forward_counts)})
        head.zero_grad(set_to_none=True)
        require_vm(root)''')
    source = replace_once(source, '        fit_start=time.monotonic();step_times=[]', '        optimizer_constructed = True\n        fit_start=time.monotonic()')
    source = replace_once(source, "write(out/'timing_update20.json',{'updates':20,'seconds':elapsed,'projected_seconds':projected,'cap_seconds':1500})",
        "write(out/'timing_update20.json',{'updates':20,'seconds':elapsed,'steady_sample_seconds':step_times[1:].copy(),'remaining_updates':780,'safety_factor':1.25,'overhead_seconds':120,'projected_seconds':projected,'cap_seconds':1500})")
    source = replace_once(source, '{float(objective):.6f}', '{float(objective.detach()):.6f}')
    source = replace_once(source, '        head.eval();candidate=snapshots[-1];failures=[]', "        assert time.monotonic()-fit_start <= 1500, 'Final fitting elapsed exceeds1500 seconds'\n        head.eval();candidate=snapshots[-1];failures=[]")
    source = replace_once(source, "        write(out/'results.json',result);print(json.dumps(result),flush=True)",
        "        execution('completed800')\n        write(out/'results.json',result);print(json.dumps(result),flush=True)")
    source = replace_once(source, "        torch.save(head.state_dict(),out/'stopped_head.pth')\n        raise", "        torch.save(head.state_dict(),out/'stopped_head.pth')\n        execution('failed_partial')\n        raise")
    source = replace_once(source, "    files+=[root/'trainer.log'] if (root/'trainer.log').exists() else []", "    files+=[root/name for name in ['trainer.log','trainer_exit_code.txt','supervisor_receipt.json'] if (root/name).exists()]")
    source = replace_once(source, "for mode in ['verify-transfer','preflight','run','export']", "for mode in ['verify-transfer','preflight','run','export','record-supervision']")
    source = replace_once(source, '    args=parser.parse_args();root=args.root.resolve();', "    parser.add_argument('--supervisor-start',type=float);parser.add_argument('--supervisor-end',type=float);parser.add_argument('--trainer-exit',type=int)\n    args=parser.parse_args();root=args.root.resolve();")
    source = replace_once(source, '    if args.export:export(root,p,args.protocol_sha);return', '''    if args.record_supervision:
        assert sys.platform=='linux' and root.is_relative_to((Path.home()/'forensic-dgp').resolve())
        import math
        assert all(value is not None and math.isfinite(value) for value in [args.supervisor_start,args.supervisor_end]) and args.supervisor_end>=args.supervisor_start and args.trainer_exit is not None
        seconds=args.supervisor_end-args.supervisor_start
        write(root/'supervisor_receipt.json', {'complete': True, 'protocol_sha256': args.protocol_sha,
            'seconds': seconds, 'cap_seconds': 2100, 'kill_grace_seconds': 30,
            'within_external_bound': seconds<=2130, 'trainer_exit_code': args.trainer_exit,
            'measurement': 'Parent shell samples Linux system monotonic clock before/after child pipeline; parent creates no CUDA context',
            'training_acceptance_not_implied': True})
        return
    if args.export:export(root,p,args.protocol_sha);return''')
    ast.parse(source, feature_version=(3, 10))
    return source


def prepare():
    started = time.monotonic()
    require(not NEW.exists() and not SOURCE.exists(), 'Preserve existing V23 preparation')
    old = read(OLD / 'protocol.json')
    require(sha(OLD / 'protocol.json') == 'c431af07b8e0bd2310f67fdc1b9ccfac7118adc08dc6cddd22131af6a5a4ca96', 'Original protocol differs')
    for name,digest in old['assets_sha256'].items():
        require(sha(OLD / name) == digest, 'Original asset differs: ' + name)
    diagnostic = ROOT / 'outputs/cctv_dgp_detail_prior_v22_r1_diagnostic'
    require(read(diagnostic/'visual_review.json')['complete'] and read(diagnostic/'independent_saved_diagnostic_audit.json')['complete'], 'Audited/reviewed V22 failure required')
    NEW.mkdir(); assets = {}
    for name,digest in old['assets_sha256'].items():
        new_name = 'lineage/v22_original_' + Path(name).name if name in ['scripts/cctv_dgp_detail_prior_v22_r1.py','scripts/run_v22.sh'] else name
        path = NEW/new_name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/name,path)
        require(sha(path)==digest,'Copied asset differs');assets[new_name]=digest
    text = runtime_source()
    with SOURCE.open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)
    source_copy = NEW/'scripts/cctv_dgp_detail_skip_v23.py';source_copy.parent.mkdir(exist_ok=True);shutil.copy2(SOURCE,source_copy)
    assets['scripts/cctv_dgp_detail_skip_v23.py']=sha(source_copy)
    shell = NEW/'scripts/run_v23.sh'
    with shell.open('x',encoding='utf-8',newline='\n') as stream:stream.write(SHELL)
    assets['scripts/run_v23.sh']=sha(shell)
    lineage = [ROOT/'outputs/cctv_dgp_detail_prior_v22_r1_independent_audit.json',
        diagnostic/'plan.json',diagnostic/'results.json',diagnostic/'visual_review.json',
        diagnostic/'independent_saved_diagnostic_audit.json',diagnostic/'diagnostic_summary.json',
        ROOT/'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/early_structure_stop.json',
        ROOT/'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/failure.json',
        ROOT/'outputs/cctv_dgp_detail_prior_v22_r1_return/outputs/update50/head.pth',
        ROOT/'outputs/cctv_dgp_detail_prior_vm_v22_r1/protocol.json',ROOT/'CCTV_DGP_DETAIL_PRIOR_V22_R1_RESULTS.md']
    for index,path in enumerate(lineage):
        name='lineage/closed_v22_'+str(index).zfill(2)+'_'+path.name
        shutil.copy2(path,NEW/name);assets[name]=sha(NEW/name)
    shutil.copy2(OLD/'schedule.json',NEW/'schedule.json')
    require(sha(NEW/'schedule.json')==sha(OLD/'schedule.json'), 'Schedule changed')
    p=copy.deepcopy(old);p['date']='2026-10-06';p['format']='dgp-learned-detail-skip-capacity-v23'
    p['purpose']='Finite training capacity for a full-resolution input-detail bypass after audited V22 early failure; no app qualification'
    p['hypothesis']='Direct full-resolution trainable input/high-pass conditioning avoids the demonstrated tiny corrective signal through a deep pooled chain. A shallow16-channel branch learns nonlinear changes; original projection/quality guards remain.'
    p['difference_from_closed_recipes']='Replace only the deep pooled/dilated45k head with a4613-parameter direct13->3 3x3 bypass plus13->16->16->3 full256 shallow branch. No pretrained generator/CodeFormer conditioning, brightness recipe, per-image normalization or unchanged V22 repeat. Extra execution evidence/logging does not relax gates.'
    p['assets_sha256']=assets;p['design']['trainable_parameters']=4613
    p['design']['architecture']='All256x256; direct13->3 3x3 plus13->16 3x3 SiLU ->16->16 3x3 SiLU ->16->3 1x1, both RGB tails zero. No pooling, dilation, normalization or label-conditioned route.'
    p['design']['correction']='q=.05*tanh(direct(features)+shallow(features)); unchanged13tap sigma2 observed-normalized high-pass and zero observed RGB mean; clip retainedDGP+band; original padding'
    p['execution']='Manual existing NVIDIA L4/g2-standard-4 under ~/forensic-dgp only; verified transfers and pasteable tmux commands; no assistant cloud/VM action'
    p.pop('execution_revision',None)
    p['prospective_gates']['next_if_pass']='Independent source/execution/step projection/CPU head/recognizer audit and all50-case review before a separately frozen broader/native/canonical app experiment. No automatic follow-on.'
    p['closed_v22_protocol_sha256']=sha(OLD/'protocol.json')
    p['closed_v22_return_sha256']='4a734450e9ea8fca8fb240a5e06b55f99a28316a770869938b2bd2aa05ad9cad'
    p['execution_evidence']='Export complete step samples, fitting elapsed, allocated VRAM peak, component forward counts, one-batch bypass gradient, supervisor monotonic duration and trainer exit code; preserve every partial/final failure'
    write(NEW/'protocol.json',p);pin=sha(NEW/'protocol.json')
    (NEW/'protocol.sha256').write_text(pin+'  protocol.json\n',encoding='ascii')
    packet=ROOT/'outputs/cctv-dgp-detail-skip-v23-execution.tar.gz'
    with tarfile.open(packet,'x:gz',compresslevel=3) as stream:
        for path in sorted(NEW.rglob('*')):
            if path.is_file():
                name=NEW.name+'/'+path.relative_to(NEW).as_posix()
                info=stream.gettarinfo(str(path),arcname=name);info.mode=0o755 if path==shell else 0o644
                with path.open('rb') as payload:stream.addfile(info,payload)
    digest=sha(packet)
    with Path(str(packet)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+packet.name+'\n')
    receipt={'complete':True,'date':'2026-10-06','scope':'Prepared immutable transfer only; CUDA gradients/training/quality pending',
        'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':packet.stat().st_size,'assets':len(assets),
        'trainable_parameters':4613,'cases':50,'updates':800,'epochs':80,'schedule_unchanged':True,'quality_tolerances_unchanged':True,
        'seconds':time.monotonic()-started,'optimizer_updates':0,'backward_calls':0,'neural_calls':0,'VM_actions':False,'app_promotion':False,'goal_complete':False}
    evidence=ROOT/'outputs/cctv_dgp_detail_skip_v23_preparation';evidence.mkdir()
    write(evidence/'preparation.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':prepare()
