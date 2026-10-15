"""Import only the sealed manual B r4 return; never import or run its trainer."""
import argparse
import json
from pathlib import Path

import import_cctv_dgp_bank_comparison_return_v1 as importer


NAME = 'cctv_dgp_bank_comparison_v1_r4_B_vm'
STEM = 'cctv-dgp-bank-comparison-v1-r4-B'
PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'
ARCHIVE_SHA = 'aea4c8e14b9a0ff67744498bf776d1c74137bdae5b38030d61885181e35ed2c7'
PREFIX = 'cctv_dgp_bank_comparison_v1_r4_B'


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepared_path = importer.OUTPUT / (PREFIX + '_preparation.json')
    transfer_path = importer.OUTPUT / 'cctv_dgp_bank_comparison_B4_transfer_v1_audit.json'
    prepared, transfer = importer.read(prepared_path), importer.read(transfer_path)
    assert prepared['complete'] and transfer['complete']
    assert prepared['route'] == 'B' and prepared['name'] == NAME and prepared['stem'] == STEM
    assert prepared['protocol_sha256'] == transfer['protocol_sha256'] == PIN
    assert prepared['archive_sha256'] == transfer['archive_sha256'] == ARCHIVE_SHA
    assert transfer['preparation_sha256'] == importer.sha(prepared_path)
    assert prepared['transfer_archive_sha256'] == transfer['transfer_archive_sha256']
    assert transfer['full_and_reuse_transfers_match_exact_packet']
    assert transfer['exact_initializer_weights_data_schedule_scientific_gates_and_resource_caps']
    assert importer.sha(importer.OUTPUT / NAME / 'protocol.json') == PIN
    importer.NAME, importer.STEM, importer.PIN = NAME, STEM, PIN
    importer.PREPARED = importer.OUTPUT / NAME
    importer.RETURN = importer.OUTPUT / (PREFIX + '_return')
    importer.ANALYSIS = importer.OUTPUT / (PREFIX + '_analysis')
    sources = [Path(__file__).resolve(), Path(importer.__file__).resolve(),
               prepared_path, transfer_path]
    bound = {str(path): importer.sha(path) for path in sources}
    importer.main()
    assert all(importer.sha(Path(path)) == expected for path, expected in bound.items())
    receipt = dict(complete=True, route='B', revision='r4', protocol_sha256=PIN,
                   import_receipt_sha256=importer.sha(importer.ANALYSIS / 'import.json'),
                   source_bindings_unchanged=bound, VM_trainer_imported_or_run=False,
                   local_optimizer_gradient_or_training_calls=0, quality_qualified=False)
    with (importer.ANALYSIS / 'route_import.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
