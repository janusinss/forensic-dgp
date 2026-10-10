"""One read-only deletion/disk observation; full protected-file audit stays pending."""
import json
import storage_gcloud_20261009_v1 as transport


def main():
    source = transport.remote_check() + '''import shutil
plan=json.loads(Path('/home/janusdominic0/archive-duplicates_plan_20261009_v1.json').read_text())
ledger=Path('/home/janusdominic0/forensic-dgp/maintenance_storage_20261009_v1/archive-duplicates/deletions.jsonl')
rows=[json.loads(line) for line in ledger.read_text().splitlines()]
assert {row['path'] for row in rows}=={row['path'] for row in plan['files']}
assert all(not os.path.lexists(row['path']) for row in rows)
disk=shutil.disk_usage('/')
print(json.dumps({'complete':True,'scope':'Deletion ledger/absence/disk observation only; protected-file and independent audits pending',
 'archive_copies_removed':len(rows),'logical_bytes_removed':sum(row['bytes'] for row in rows),
 'free_bytes':disk.free,'free_GiB':disk.free/1024**3,'files_removed_by_observer':0,
 'training_or_diagnostic_launched':False,'full_cleanup_closure_complete':False},allow_nan=False))
'''
    transport.ssh(source, 'deletion_stage_observation', 90)
    value = json.loads((transport.LOGS / 'deletion_stage_observation_stdout.log').read_text(encoding='utf-8'))
    assert value['complete'] and value['archive_copies_removed'] == len(transport.transport.read(transport.PLAN)['files'])
    transport.transport.write(transport.OUT / 'deletion_stage_observation.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    main()
