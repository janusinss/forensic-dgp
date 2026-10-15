"""Future sealed manual VM return import; never imports/executes the trainer."""
import argparse
import json
from pathlib import Path

import import_cctv_dgp_bank_comparison_return_v1 as importer

NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
STEM = 'cctv-dgp-bank-photometric-guard-v1'
PREFIX = 'cctv_dgp_bank_photometric_guard_v1'
PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
FULL_SHA = '3f024ee645c7901ebd14eb0d86560016e4efc2f1375a4b8b0048a1fe4bc10615'


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    preparation = importer.OUTPUT/(PREFIX+'_preparation.json')
    transfer = importer.OUTPUT/(PREFIX+'_transfer_audit.json')
    prep, audit = importer.read(preparation), importer.read(transfer)
    assert prep['complete'] and audit['complete'] and prep['protocol_sha256'] == audit['protocol_sha256'] == PIN
    assert prep['archive_sha256'] == audit['archive_sha256'] == FULL_SHA
    assert audit['preparation_sha256'] == importer.sha(preparation)
    assert audit['full_and_reuse_transfers_match_exact_packet'] and audit['model_computation_AST_unchanged']
    assert audit['initializer_weights_frozen_assets_optimizer_data_schedule_gates_caps_unchanged']
    assert prep['start_not_failed_B50'] and not prep['model_qualified']
    importer.NAME, importer.STEM, importer.PIN = NAME, STEM, PIN
    importer.PREPARED = importer.OUTPUT/NAME
    importer.RETURN, importer.ANALYSIS = importer.OUTPUT/(PREFIX+'_return'), importer.OUTPUT/(PREFIX+'_analysis')
    sources = [Path(__file__).resolve(),Path(importer.__file__).resolve(),preparation,transfer]
    bound = {str(path):importer.sha(path) for path in sources}
    importer.main()
    assert all(importer.sha(Path(path)) == digest for path,digest in bound.items())
    result = dict(complete=True,route='B',revision='photometric_guard_v1',protocol_sha256=PIN,
        import_receipt_sha256=importer.sha(importer.ANALYSIS/'import.json'),source_bindings_unchanged=bound,
        VM_trainer_imported_or_run=False,local_optimizer_gradient_or_training_calls=0,quality_qualified=False)
    with (importer.ANALYSIS/'route_import.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
