"""Prepare/run one read-only final probe AFTER the exact successful R1 apply.

Default prints local preparation only. --run needs the exact completed local R1
apply receipt SHA. No upload, deletion, guest receipt write, checkpoint loading,
protected payload hashing, training, or automatic retry is present here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/vm_archive_offload_20261018_v1'
CLIENT = ROOT / 'scripts/vm_archive_offload_20261018_v1_r1.py'
CLIENT_SHA = '44139813d78161096c26f7c4f37be7a958a5bc285b300a2f6ef805bfa5688480'
TRUST = ROOT / 'scripts/inspect_vm_storage_20261018_v1.py'
TRUST_SHA = '7838ef1c9d0d5d14ed3960348f07c14dfb91c87835c35bd85451bec01a88c3aa'
PREPARED = OUT / 'prepared_r1.json'
PREPARED_SHA = '6013b5497aa8bc7d8e57a42c28be986bcf0d6e6948367b16ceebd3fc69f33c37'
PLAN = OUT / 'exact_plan_r1.json'
PLAN_SHA = 'e5744075ebcef689ed3c552b399874f76b7e7ad9e1cad821fdda293cc733238d'
GUEST_SHA = 'a0282d5a0308c19c67b9ac3de40d4eace425f05e189affb6535279ff14561cb7'
NAMESPACE_SHA = '580204d720f284510cd949460ebd9ed3ad26f7dd0ad562fb1b0cac69522fdde3'
TOTAL_BYTES = 16_692_148_382
APPLY_RECEIPT = OUT / 'apply_r1/receipt.json'

# This exact code is transmitted in the reviewed short, pinned-SSH transport.
# It emits a bounded result only; the namespace map stays in memory.
PROBE = '''import hashlib,json,os,runpy,signal,stat
from pathlib import Path
C=CONFIG
def require(ok,msg):
 if not ok:raise ValueError(msg)
def expire(*args):raise TimeoutError('Final readonly probe240s cap')
signal.signal(signal.SIGALRM,expire);signal.alarm(240)
try:
 require(os.getuid()==1001,'Exact VM uid1001 required')
 source=Path('/home/janusdominic0/vm_archive_offload_20261018_v1/guest_r1.py')
 for p in (source,*source.parents):
  s=p.lstat();require(not stat.S_ISLNK(s.st_mode),'Source ancestor symlink refused')
  if str(p)=='/home/janusdominic0':break
 s=source.lstat();require(stat.S_ISREG(s.st_mode)and s.st_uid==1001 and s.st_nlink==1 and s.st_size<131072,'Exact ordinary owned guest source required')
 require(hashlib.sha256(source.read_bytes()).hexdigest()==C['guest_sha'],'Exact R1 guest hash differs')
 g=runpy.run_path(str(source),run_name='_readonly_final_probe_20261018')
 identity=g['runtime_identity']()
 plan=g['bounded_read_json'](g['AUDIT']/'exact_plan_r1.json',C['plan_sha'])
 require(plan['guest_source']['sha256']==C['guest_sha']and plan['verified']['namespace_sha256']==C['namespace_sha'],'Exact original verified namespace/source binding differs')
 rows=g['validate_request'](plan['request'])
 applied=g['bounded_read_json'](g['AUDIT']/'apply_receipt.json')
 require(applied.get('complete')is True and applied.get('files_removed')==6 and applied.get('logical_bytes_removed')==16692148382 and applied.get('plan_sha256')==C['plan_sha']and applied.get('all_research_namespace_and_protected_content_unchanged')is True,'Exact successful R1 guest apply receipt required')
 require(g['canonical_sha'](applied)==C['apply_canonical_sha'],'Guest apply receipt differs from completed Windows receipt')
 absent=[]
 for row in rows:
  p=g['path_policy'](row['vm_path'])
  require(not os.path.lexists(p),'Original HOME archive or dangling link still exists')
  absent.append(str(p))
 workload_before=g['workload_guard'](rows);readers_before=g['open_reader_guard'](rows)
 space=g['namespace']();actual=g['canonical_sha'](space)
 require(actual==C['namespace_sha'],'Full research namespace metadata differs from original verification')
 workload_after=g['workload_guard'](rows);readers_after=g['open_reader_guard'](rows)
 result={'format':'vm-six-home-final-readonly-probe-v1','complete':True,'identity':identity,'guest_source_sha256':C['guest_sha'],'plan_sha256':C['plan_sha'],'apply_receipt_canonical_sha256':C['apply_canonical_sha'],'exact_six_paths_absent':absent,'research_namespace_sha256':actual,'research_namespace_entries':len(space),'research_namespace_metadata_equals_original_verified':True,'workload_before':workload_before,'open_reader_before':readers_before,'workload_after':workload_after,'open_reader_after':readers_after,'disk':g['disk'](),'protected_payloads_rehashed':False,'apply_protected_before_after_hash_evidence_remains_separate':True,'guest_files_written':0,'files_removed_by_probe':0,'training_calls':0}
 print(json.dumps(result,sort_keys=True,separators=(',',':')),flush=True)
finally:signal.alarm(0)
'''


def require(okay, message):
    if not okay:
        raise ValueError(message)


def canonical_sha(value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return hashlib.sha256(raw).hexdigest()


def check_apply(value):
    require(value.get('complete') is True and value.get('mode') == 'apply'
            and value.get('files_removed') == 6 and value.get('logical_bytes_removed') == TOTAL_BYTES
            and value.get('exact_plan_sha256') == PLAN_SHA and value.get('training_calls') == 0,
            'Exact successful local six-archive R1 apply required before any network call')
    guest = value.get('guest_receipt', {})
    require(guest.get('complete') is True and guest.get('mode') == 'apply'
            and guest.get('files_removed') == 6 and guest.get('logical_bytes_removed') == TOTAL_BYTES
            and guest.get('plan_sha256') == PLAN_SHA
            and guest.get('all_research_namespace_and_protected_content_unchanged') is True
            and guest.get('training_calls') == 0,
            'Complete unchanged-namespace R1 guest receipt required')
    local_payload, remote_payload = value.get('payload', {}), guest.get('payload', {})
    require(local_payload.get('path') == str(OUT / 'apply_r1/apply_payload.json.gz')
            and remote_payload.get('path') == '/home/janusdominic0/vm_archive_offload_20261018_v1/apply_payload.json.gz'
            and local_payload.get('bytes', 0) > 0
            and all(local_payload.get(k) == remote_payload.get(k) for k in ('bytes', 'sha256')),
            'Exact downloaded guest payload hash/size and distinct local/remote paths required')
    return guest


def load_client():
    for path, expected in ((CLIENT, CLIENT_SHA), (TRUST, TRUST_SHA)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Reviewed transport source changed')
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('_readonly_final_probe_client', CLIENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    client = load_client()
    require(client.pin(PREPARED)['sha256'] == PREPARED_SHA, 'Exact R1 preparation required')
    require(client.pin(PLAN)['sha256'] == PLAN_SHA, 'Exact R1 plan required')
    prep, plan = client.read(PREPARED), client.read(PLAN)
    client.runtime_sources(prep)
    require(prep['guest_source']['sha256'] == GUEST_SHA
            and plan['guest_source'] == prep['guest_source']
            and plan['verified']['namespace_sha256'] == NAMESPACE_SHA, 'Frozen final-probe lineage differs')
    return client, prep, plan


def probe_source(guest_apply):
    config = dict(guest_sha=GUEST_SHA, plan_sha=PLAN_SHA, namespace_sha=NAMESPACE_SHA,
                  apply_canonical_sha=canonical_sha(guest_apply))
    return PROBE.replace('CONFIG', repr(config), 1)


def run(apply_sha):
    client, prep, plan = prepare()
    require(len(apply_sha) == 64 and all(c in '0123456789abcdef' for c in apply_sha), 'Exact apply receipt SHA required')
    require(client.pin(APPLY_RECEIPT)['sha256'] == apply_sha, 'Completed local apply receipt changed')
    apply = client.read(APPLY_RECEIPT); guest = check_apply(apply)
    require(client.pin(Path(apply['payload']['path'])) == apply['payload'], 'Completed local apply payload changed')
    source = probe_source(guest)
    argv = client.ssh_args(source)  # Check the Windows short-command cap before API/SSH.
    folder = OUT / 'final_readonly_probe_v1'
    folder.mkdir()
    client.save(folder / 'intent.json', dict(complete=False, probe_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source=client.pin(Path(__file__)), apply_receipt=client.pin(APPLY_RECEIPT),
        exact_plan=client.pin(PLAN), guest_source=prep['guest_source'],
        guest_cap_seconds=240, SSH_cap_seconds=300, protected_payload_rehash_planned=False,
        no_guest_writes_uploads_removal_or_training=True))
    environment = client.trust.environment(prep)
    client.live_api(folder, environment)
    out = client.invoke(folder, 'final_probe', argv, 300, environment, 'read-only final namespace/workload/absence probe')
    result = client.last_receipt(out)
    expected = sorted(r['vm_path'] for r in plan['request']['candidates'])
    require(result.get('complete') is True and sorted(result.get('exact_six_paths_absent', [])) == expected
            and result.get('research_namespace_sha256') == NAMESPACE_SHA
            and result.get('guest_source_sha256') == GUEST_SHA and result.get('plan_sha256') == PLAN_SHA
            and result.get('apply_receipt_canonical_sha256') == canonical_sha(guest)
            and result.get('research_namespace_metadata_equals_original_verified') is True
            and result.get('protected_payloads_rehashed') is False
            and result.get('files_removed_by_probe') == 0 and result.get('guest_files_written') == 0
            and result.get('training_calls') == 0, 'Exact bounded final probe evidence required')
    client.save(folder / 'receipt.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        actual_probe=result, stdout=client.pin(out), source=client.pin(Path(__file__)),
        apply_receipt=client.pin(APPLY_RECEIPT), exact_plan=client.pin(PLAN),
        limitation='Final probe compares full research metadata only. The separate apply audit establishes protected before/after payload hashes.',
        files_removed_by_probe=0, training_calls=0))
    print(json.dumps(dict(complete=True, receipt=client.pin(folder / 'receipt.json'), protected_payloads_rehashed=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--apply-receipt-sha256')
    args = parser.parse_args()
    if args.run:
        require(args.apply_receipt_sha256, 'Successful R1 apply receipt SHA required')
        run(args.apply_receipt_sha256)
    else:
        client, prep, plan = prepare()
        print(json.dumps(dict(complete=True, local_preparation_only=True, VM_API_SSH_calls=0,
                             source=client.pin(Path(__file__)), plan=client.pin(PLAN),
                             exact_guest_sha256=GUEST_SHA, original_namespace_sha256=NAMESPACE_SHA,
                             actual_successful_apply_required_before_run=True)))


if __name__ == '__main__':
    main()
