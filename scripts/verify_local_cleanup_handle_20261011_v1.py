"""Verify source-read and exclusive-removal behavior on one tiny owned fixture."""
import gzip
import json
from pathlib import Path
import tarfile

from apply_local_archive_cleanup_20261011_v1 import verify_then_remove
from prepare_local_archive_cleanup_20261011_v1 import ROOT,OUT,metadata,sha,write


def main():
    folder=OUT/'native_handle_fixture'
    assert not folder.exists()
    folder.mkdir()
    source=folder/'source.txt'
    source.write_bytes(b'local-cleanup-native-handle-verification\n')
    archive=ROOT/'outputs/local-cleanup-native-handle-fixture-results.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'x:gz') as f:
        f.add(source,arcname='source.txt',recursive=False)
    manifest=folder/'members.json.gz'
    with gzip.open(manifest,'xb') as f:
        f.write(json.dumps([dict(name='source.txt',bytes=source.stat().st_size,
            sha256=sha(source),retained_path=source.relative_to(ROOT).as_posix(),
            retained_metadata=metadata(source))]).encode('utf-8'))
    row=dict(archive=archive.relative_to(ROOT).as_posix(),archive_sha256=sha(archive),
        metadata=metadata(archive),member_manifest=manifest.relative_to(ROOT).as_posix(),
        member_manifest_sha256=sha(manifest),complete_regular_members=1,
        retained_tree=folder.relative_to(ROOT).as_posix())
    result=verify_then_remove(row)
    assert not archive.exists() and source.read_bytes()==b'local-cleanup-native-handle-verification\n'
    write(OUT/'native_handle_verification.json',dict(complete=True,
        isolated_generated_fixture_only=True,existing_user_files_removed=0,
        source_preserved=True,native_exclusive_archive_removal_verified=True,
        fixture_removal=result,VM_connections_or_training_calls=0))
    print(dict(complete=True,existing_user_files_removed=0,source_preserved=True),flush=True)


if __name__=='__main__':
    main()
