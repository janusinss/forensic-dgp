"""Fresh read-only inventory; reuse reviewed scanner, never historical decisions."""
import hashlib
from pathlib import Path
import inventory_local_research_storage_20261011_v1 as scanner

if __name__ == '__main__':
    scanner.OUT = scanner.ROOT / 'outputs/google_drive_offload_20261015_v1'
    source = Path(scanner.__file__)
    print({'read_only_scanner': str(source),
           'scanner_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}, flush=True)
    scanner.main()
