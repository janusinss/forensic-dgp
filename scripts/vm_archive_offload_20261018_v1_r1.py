"""R1 report-only compatible maintenance transport; original verify evidence retained.

--prepare-or-amend is local only and binds the completed original verification.
--apply requires the exact verified plan and independent preapply closure.
No VM start/stop, research training, broad deletion, automatic retry or phase queue.
"""
import argparse
import ast
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time

import inspect_vm_storage_20261018_v1 as trust

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/vm_archive_offload_20261018_v1'
SOURCE = ROOT / 'scripts/vm_archive_offload_20261018_v1_guest_r1.py'
SCOPE = OUT / 'windows_backup_proof_v1/scope.json'
SCOPE_SHA = '95c0547aed21b715fc49be1fb77c07057036f42bea987af11000670d950336fa'
TRUST_DRIVER_SHA = '7838ef1c9d0d5d14ed3960348f07c14dfb91c87835c35bd85451bec01a88c3aa'
REMOTE = '/home/janusdominic0/vm_archive_offload_20261018_v1'
GUEST_SOURCE = REMOTE + '/guest_r1.py'
GUEST_REQUEST = REMOTE + '/request.json'
BASE = ['--project=forensic-dgp-thesis', '--zone=us-central1-a', '--quiet']
FLAGS = ['-batch', '-hostkey', trust.HOSTKEY]
MAX_LOG = 20 * 1024**2
MAX_PAYLOAD = 256 * 1024**2
TOTAL_BYTES = 16_692_148_382


def require(okay, message):
    if not okay: raise ValueError(message)


def pin(path):
    return trust.file_pin(path)


def read(path):
    return trust.read(path)


def save(path, value):
    return trust.save(path, value)


def local_metadata(path):
    p = Path(path).absolute()
    require(p.is_relative_to(ROOT), 'Windows backup must stay under Testing')
    for q in (p, *p.parents):
        s = q.lstat()
        require(not stat.S_ISLNK(s.st_mode) and not getattr(s, 'st_file_attributes', 0) & 0x400,
                'Linked/reparse local backup forbidden')
        if q == ROOT: break
    s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Owned unique regular local backup required')
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
            'inode': s.st_ino, 'device': s.st_dev, 'nlink': s.st_nlink,
            'mode': s.st_mode, 'attributes': getattr(s, 'st_file_attributes', 0)}


def require_backup(row):
    p = ROOT / row['local_backup_path']
    before = local_metadata(p)
    require(before == row['local_backup_metadata_after'] and
            row['local_backup_metadata_before'] == row['local_backup_metadata_after'] and
            row['local_backup_fresh_full_archive_hash_verified'] is True,
            'Fresh full local backup proof/identity changed')
    return p, before


def hash_backups(scope):
    start = time.monotonic(); result = []
    for row in scope['candidates']:
        p, before = require_backup(row); h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(4 * 1024**2), b''):
                require(time.monotonic() - start <= 1200, '1200s current local backup hash cap')
                h.update(b)
        require(local_metadata(p) == before and h.hexdigest() == row['local_backup_sha256'],
                'Current complete Windows backup bytes changed')
        result.append({'path': row['local_backup_path'], 'sha256': h.hexdigest(),
                       'metadata': before, 'bytes': before['bytes']})
    return {'complete': True, 'all_six_current_full_SHA_verified': True,
            'files': result, 'seconds': time.monotonic() - start}


def build_request(scope):
    require(scope.get('complete') is True and scope.get('candidate_count') == 6
            and scope.get('logical_archive_bytes') == TOTAL_BYTES, 'Exact complete six-backup scope required')
    rows = []
    for r in scope['candidates']:
        require_backup(r)
        m = {k: v for k, v in r['vm_metadata_from_fresh_inventory'].items() if k not in ('path', 'symlink')}
        rows.append({'vm_path': r['vm_path'], 'expected_metadata': m,
            'sha256': r['local_backup_sha256'], 'family': r['family'],
            'backup': {'path': r['local_backup_path'], 'bytes': r['local_backup_metadata_after']['bytes'],
                       'sha256': r['local_backup_sha256'], 'fresh_full_windows_hash_verified': True,
                       'metadata': r['local_backup_metadata_after']},
            'research_root_retained': r['VM_research_root_to_retain']})
    return {'format': 'vm-six-home-archive-maintenance-request-v1', 'complete': True,
            'windows_backup_scope_sha256': SCOPE_SHA, 'candidates': rows,
            'all_actual_training_caches_checkpoints_sources_MD_splits_failures_retained': True,
            'candidate_count': 6, 'logical_archive_bytes': TOTAL_BYTES}


def prepared():
    require(pin(Path(trust.__file__))['sha256'] == TRUST_DRIVER_SHA, 'Reviewed TLS/SSH primitive dependency changed')
    require(pin(SCOPE)['sha256'] == SCOPE_SHA, 'Fresh six-backup scope changed')
    scope = read(SCOPE); request = build_request(scope)
    ast.parse(SOURCE.read_text(encoding='utf-8'), feature_version=(3, 10))
    require(pin(trust.TRUST)['sha256'] == trust.TRUST_SHA, 'Historical exact host-key evidence changed')
    old = read(trust.TRUST)
    require(old.get('complete') is True and old.get('known_historical_hostkey') == trust.HOSTKEY
            and old.get('current_VM_ID') == trust.INSTANCE_ID, 'Pinned historical VM/host-key mismatch')
    require(trust.CA.read_bytes().count(b'-----BEGIN CERTIFICATE-----') > 0, 'Existing CA PEM required')
    return {'complete': True, 'format': 'vm-six-home-archive-preparation-v1',
            'guest_source': pin(SOURCE), 'windows_backup_scope': pin(SCOPE),
            'driver': pin(Path(__file__)), 'reviewed_transport_primitives': pin(Path(trust.__file__)),
            'gcloud_executable': pin(trust.CLOUD), 'TLS_CA_bundle': pin(trust.CA),
            'historical_trust': pin(trust.TRUST), 'pinned_hostkey': trust.HOSTKEY,
            'request': request, 'remote_audit_namespace': REMOTE,
            'caps': {'guest_whole_seconds': 1800, 'hash_seconds': 1200,
                     'SSH_seconds': 1860, 'SCP_seconds': 120, 'API_seconds': 60},
            'network_VM_actions': 0, 'files_removed': 0, 'training_calls': 0}


def invoke(folder, name, argv, cap, environment, action):
    stdout, stderr = folder / (name + '_stdout.log'), folder / (name + '_stderr.log')
    start = time.monotonic(); child = None; stop = None
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            child = subprocess.Popen([str(trust.CLOUD), *argv], stdout=out, stderr=err, env=environment,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            save(folder / (name + '_started.json'), {'pid': child.pid, 'cap_seconds': cap,
                 'action': action, 'TLS_validation_enabled': True, 'hostkey_pinned': trust.HOSTKEY})
            last = start
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > cap or stdout.stat().st_size + stderr.stat().st_size > MAX_LOG:
                    stop = 'time_or_log_cap'; trust.stop_own_transport(child); break
                if time.monotonic() - last >= 30:
                    print(json.dumps({'maintenance_stage': name, 'seconds': round(elapsed, 1),
                                      'own_transport_pid': child.pid, 'still_running': True}), flush=True)
                    last = time.monotonic()
                time.sleep(.25)
            code = child.wait(timeout=15)
        require(stdout.stat().st_size + stderr.stat().st_size <= MAX_LOG, 'Transport output cap')
        receipt = {'complete': code == 0 and stop is None, 'exit_code': code, 'external_stop': stop,
                   'child_pid': child.pid, 'child_terminal': True, 'seconds': time.monotonic() - start,
                   'cap_seconds': cap, 'action': action, 'stdout': pin(stdout), 'stderr': pin(stderr),
                   'TLS_validation_enabled': True, 'hostkey_pinned': trust.HOSTKEY,
                   'no_automatic_retry': True, 'model_training_calls': 0}
        save(folder / (name + '_transport.json'), receipt)
        require(receipt['complete'], 'VM maintenance transport stopped; preserve evidence; no replay')
        return stdout
    except BaseException as e:
        if child is not None and child.poll() is None: trust.stop_own_transport(child)
        save(folder / (name + '_failure.json'), {'complete': False, 'exception_type': type(e).__name__,
             'reason': str(e)[:2048], 'child_terminal': child is None or child.poll() is not None,
             'action': action, 'no_automatic_retry': True})
        raise


def ssh_args(code):
    source = base64.b64encode(code.encode('utf-8')).decode('ascii')
    wrapper = 'import base64;exec(base64.b64decode(' + repr(source) + '))'
    command = 'python3 -B -c ' + shlex.quote(wrapper)
    argv = ['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis', *BASE,
            *['--ssh-flag=' + x for x in FLAGS], '--command=' + command]
    require(len(subprocess.list2cmdline([str(trust.CLOUD), *argv])) < 7900, 'Windows short SSH command cap')
    return argv


def scp_args(local, remote, download=False):
    source, target = (remote, str(local)) if download else (str(local), remote)
    return ['compute', 'scp', source, target, *BASE, *['--scp-flag=' + x for x in FLAGS]]


def runtime_sources(plan):
    for key in ('guest_source', 'windows_backup_scope', 'driver', 'reviewed_transport_primitives',
                'gcloud_executable', 'TLS_CA_bundle', 'historical_trust'):
        require(pin(plan[key]['path']) == plan[key], 'Prepared runtime dependency changed: ' + key)


def live_api(folder, env):
    out = invoke(folder, 'api', ['compute', 'instances', 'describe', trust.INSTANCE,
        *BASE, '--format=json(id,name,status,machineType,zone,guestAccelerators)'], 60, env, 'read-only API identity')
    data = read(out)
    require(str(data.get('id')) == trust.INSTANCE_ID and data.get('name') == trust.INSTANCE
            and data.get('status') == 'RUNNING' and data.get('zone', '').endswith('/us-central1-a')
            and data.get('machineType', '').endswith('/g2-standard-4'), 'Fresh running API identity required')
    gpu = data.get('guestAccelerators', [])
    require(len(gpu) == 1 and gpu[0].get('acceleratorCount') == 1
            and gpu[0].get('acceleratorType', '').endswith('/nvidia-l4'), 'Fresh API L4 identity mismatch')


def launcher(mode, source_sha, document_name, document_sha, review_sha=None):
    doc = REMOTE + '/' + document_name
    args = ['guest_r1.py', '--verify-only', '--request', doc, '--request-sha256', document_sha]
    if mode == 'apply':
        args = ['guest_r1.py', '--apply', '--plan', doc, '--plan-sha256', document_sha,
                '--review', REMOTE + '/independent_review_r1.json', '--review-sha256', review_sha]
    return ('import hashlib,os,runpy,sys;from pathlib import Path;'
            'p=Path(' + repr(GUEST_SOURCE) + ');'
            'assert os.getuid()==1001 and not p.is_symlink();'
            'assert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(source_sha) + ';'
            'sys.argv=' + repr(args) + ';runpy.run_path(str(p),run_name="__main__")')


def last_receipt(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    require(lines, 'Guest final receipt missing')
    return json.loads(lines[-1])


def download_payload(folder, receipt, mode, env):
    recorded = receipt.get('payload', {})
    remote_path = REMOTE + '/' + mode + '_payload.json.gz'
    require(recorded.get('path') == remote_path and recorded.get('bytes', MAX_PAYLOAD + 1) <= MAX_PAYLOAD,
            'Exact bounded guest payload path required')
    local = folder / (mode + '_payload.json.gz')
    invoke(folder, 'download_payload', scp_args(local, 'janusdominic0@forensic-dgp-thesis:' + remote_path,
           download=True), 120, env, 'download maintenance evidence only')
    actual = pin(local)
    require(actual['bytes'] == recorded['bytes'] and actual['sha256'] == recorded['sha256'],
            'Downloaded evidence bytes differ')
    with gzip.open(local, 'rb') as f:
        raw = f.read(MAX_PAYLOAD + 1)
        require(len(raw) <= MAX_PAYLOAD and not f.read(1), 'Bounded decoded maintenance payload required')
    value = json.loads(raw)
    require(value.get('complete') is True, 'Guest payload incomplete')
    return value, actual




def apply(prep, plan_sha, review_name, review_sha):
    runtime_sources(prep)
    plan_path = OUT / 'exact_plan_r1.json'; require(pin(plan_path)['sha256'] == plan_sha, 'Exact reviewed plan SHA required')
    plan = read(plan_path); review_path = Path(review_name).absolute()
    require(review_path.is_relative_to(ROOT) and pin(review_path)['sha256'] == review_sha,
            'Exact workspace independent review SHA required')
    review = read(review_path)
    require(review.get('complete') is True and review.get('approved_for_exact_six_HOME_archive_apply') is True
            and review.get('plan_sha256') == plan_sha and review.get('guest_source_sha256') == prep['guest_source']['sha256']
            and review.get('windows_backup_scope_sha256') == SCOPE_SHA
            and review.get('protected_sha256') == plan['verified']['protected_sha256']
            and review.get('candidate_current_sha256') == plan['verified']['candidate_current_sha256'],
            'Independent exact candidate/backup/protection/source closure required')
    require(plan['guest_source'] == prep['guest_source'] and plan['request'] == prep['request'], 'Prepared exact plan changed')
    folder = OUT / 'apply_r1'; folder.mkdir()
    backup = hash_backups(read(SCOPE))
    save(folder / 'intent.json', {'plan_sha256': plan_sha, 'independent_review': pin(review_path),
         'windows_backups_current': backup, 'no_automatic_retry_or_replay': True})
    env = trust.environment(prep); live_api(folder, env)
    invoke(folder, 'upload_source_r1', scp_args(SOURCE, 'janusdominic0@forensic-dgp-thesis:' + GUEST_SOURCE),
           120, env, 'upload new reviewed report-only guest_r1 source; preserve original guest.py')
    for label, local, name in [('plan', plan_path, 'exact_plan_r1.json'), ('review', review_path, 'independent_review_r1.json')]:
        invoke(folder, 'upload_' + label, scp_args(local, 'janusdominic0@forensic-dgp-thesis:' + REMOTE + '/' + name),
               120, env, 'upload exact reviewed apply evidence only')
    out = invoke(folder, 'guest_apply', ssh_args(launcher('apply', prep['guest_source']['sha256'],
          'exact_plan_r1.json', plan_sha, review_sha)), 1860, env, 'apply exactly six reviewed HOME archive copies')
    receipt = last_receipt(out); require(receipt.get('complete') is True and receipt.get('files_removed') == 6
        and receipt.get('logical_bytes_removed') == TOTAL_BYTES, 'Exact complete six-removal receipt required')
    payload, payload_pin = download_payload(folder, receipt, 'apply', env)
    require(payload['plan_sha256'] == plan_sha and payload['review_sha256'] == review_sha
            and payload['all_research_namespace_and_protected_content_unchanged'] is True,
            'Independent after-state payload binding differs')
    save(folder / 'receipt.json', {'complete': True, 'mode': 'apply', 'guest_receipt': receipt,
         'payload': payload_pin, 'exact_plan_sha256': plan_sha, 'review_sha256': review_sha,
         'files_removed': 6, 'logical_bytes_removed': TOTAL_BYTES,
         'independent_postapply_audit_required': True, 'training_calls': 0})
    print(json.dumps({'complete': True, 'files_removed': 6, 'logical_bytes_removed': TOTAL_BYTES,
                      'receipt': pin(folder/'receipt.json'), 'independent_postapply_audit_required': True}), flush=True)


def prepare_or_amend():
    old_plan_path = OUT / 'exact_plan_v1.json'
    require(pin(old_plan_path)['sha256'] == 'b5cd25a2432cf73e16f8009569f9a405d3bfd61ad1732557487567280db328a0',
            'Exact original verified plan required; never replace it')
    old = read(old_plan_path)
    original_prepared = OUT / 'prepared_v1.json'
    require(pin(original_prepared) == old['prepared_connection'], 'Original completed preparation changed')
    original = read(original_prepared)
    require(original['guest_source']['sha256'] == '6ecf6eff7d9a12109b3b801a0c2829671f6fe0e83b66e31bccef63137acbe004',
            'Exact original verified guest implementation required')
    original_source = Path(original['guest_source']['path'])
    require(pin(original_source) == original['guest_source'], 'Preserved original guest source changed')
    require(pin(Path(original['driver']['path'])) == original['driver'], 'Original client source changed')
    payload = OUT / 'verification_v1/verify_payload.json.gz'
    require(pin(payload)['sha256'] == 'c07efda5f30bd261a04307988b13fd8bcd2be782e51af17a3856975d3720484c'
            and pin(payload) == old['verified']['local_payload'], 'Exact completed VM verification payload required')
    with gzip.open(payload, 'rb') as f:
        raw = f.read(MAX_PAYLOAD + 1)
        require(len(raw) <= MAX_PAYLOAD and not f.read(1), 'Bounded original verification payload')
    verified = json.loads(raw)
    require(verified['complete'] is True and verified['files_removed'] == 0
            and verified['guest_source_sha256'] == original['guest_source']['sha256'], 'Original verification source lineage differs')
    value = prepared()
    require(value['request'] == original['request'] == old['request'], 'R1 must retain the exact original request')
    # Independent report-size arithmetic uses the real saved maps, not an invented small fixture.
    maps = {'namespace_after': verified['namespace_before'], 'protected_after': verified['protected_before']}
    maps_bytes = len((json.dumps(maps, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())
    require(maps_bytes + 8 * 1024**2 <= MAX_PAYLOAD, 'Actual maps plus bounded nonmap reservation exceed256MiB')
    amendment = {'format': 'vm-six-archive-report-only-R1-compatibility-amendment', 'complete': True,
        'original_plan': pin(old_plan_path), 'original_prepared': pin(original_prepared),
        'original_guest_source': original['guest_source'], 'original_client_source': original['driver'],
        'original_verified_payload': pin(payload), 'original_request_sha256': old['request_sha256'],
        'unchanged_VM_scope_training_guards_hash_namespace_and_removal_policy': True,
        'change': 'Store before maps by exact retained verification payload and canonical refs; retain full after maps; separately pin full workload-after JSON',
        'actual_saved_maps_serialized_bytes': maps_bytes, 'reserved_nonmap_bytes': 8 * 1024**2,
        'maximum_decoded_report_bytes': MAX_PAYLOAD, 'prospective_after_report_upper_bytes': maps_bytes + 8 * 1024**2,
        'all_original_source_prepared_plan_verify_bytes_retained': True,
        'original_VM_verification_not_repeated_unchanged': True, 'VM_or_removal_calls': 0}
    prep_pin = save(OUT/'prepared_r1.json', value)
    amended = dict(old)
    amended.update(guest_source=value['guest_source'], driver=value['driver'], prepared_connection=prep_pin,
                   compatibility_lineage=amendment)
    amended['verified'] = dict(old['verified'], guest_source_sha256=original['guest_source']['sha256'])
    plan_pin = save(OUT/'exact_plan_r1.json', amended)
    closure_pin = save(OUT/'report_only_amendment_r1.json', amendment)
    print(json.dumps({'complete': True, 'local_only': True, 'prepared_r1': prep_pin, 'exact_plan_r1': plan_pin,
                      'compatibility_amendment': closure_pin, 'API_SSH_SCP_calls': 0, 'files_removed': 0}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-or-amend', action='store_true'); mode.add_argument('--apply', action='store_true')
    p.add_argument('--plan-sha256'); p.add_argument('--independent-review'); p.add_argument('--review-sha256')
    args = p.parse_args()
    if args.prepare_or_amend:
        prepare_or_amend()
    else:
        require(args.plan_sha256 and args.independent_review and args.review_sha256, 'Exact independently reviewed R1 apply arguments required')
        apply(read(OUT/'prepared_r1.json'), args.plan_sha256, args.independent_review, args.review_sha256)


if __name__ == '__main__':
    main()
