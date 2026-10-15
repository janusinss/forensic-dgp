"""Import only the sealed manual B r3 return; never import or run its trainer."""
import argparse
import json
from pathlib import Path

import import_cctv_dgp_bank_comparison_return_v1 as importer


NAME = 'cctv_dgp_bank_comparison_v1_r3_B_vm'
STEM = 'cctv-dgp-bank-comparison-v1-r3-B'
PIN = 'f34b7b54cd322d26a1a2159cffa28aabb1258fcb484914fd781d70e3e7bc0c35'
ARCHIVE_SHA = '9ed188ce42881d33765544a355e687ef1ad1d34f044276e36fb6d6b929f0fc8b'
PREFIX = 'cctv_dgp_bank_comparison_v1_r3_B'


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepared_path = importer.OUTPUT / (PREFIX + '_preparation.json')
    transfer_path = importer.OUTPUT / 'cctv_dgp_bank_comparison_B3_transfer_v1_audit.json'
    prepared, transfer = importer.read(prepared_path), importer.read(transfer_path)
    assert prepared['complete'] and transfer['complete']
    assert prepared['route'] == 'B' and prepared['name'] == NAME and prepared['stem'] == STEM
    assert prepared['protocol_sha256'] == transfer['protocol_sha256'] == PIN
    assert prepared['archive_sha256'] == transfer['archive_sha256'] == ARCHIVE_SHA
    assert transfer['preparation_sha256'] == importer.sha(prepared_path)
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
    receipt = dict(complete=True, route='B', revision='r3', protocol_sha256=PIN,
                   import_receipt_sha256=importer.sha(importer.ANALYSIS / 'import.json'),
                   source_bindings_unchanged=bound, VM_trainer_imported_or_run=False,
                   local_optimizer_gradient_or_training_calls=0, quality_qualified=False)
    with (importer.ANALYSIS / 'route_import.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
