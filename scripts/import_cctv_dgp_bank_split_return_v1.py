"""Import a sealed, manually executed bank-comparison route without its trainer."""
import argparse
import json
from pathlib import Path

import import_cctv_dgp_bank_comparison_return_v1 as importer


ROUTES = {
    'A': ('cctv_dgp_bank_comparison_v1_r3_A_vm',
          'cctv-dgp-bank-comparison-v1-r3-A',
          '6ca05c6b87a4588bfea2bc5b80307927c3a2fd47fa419567ee35280b37a78d0b',
          'cctv_dgp_bank_comparison_v1_r3_A'),
    'B': ('cctv_dgp_bank_comparison_v1_r2_B_vm',
          'cctv-dgp-bank-comparison-v1-r2-B',
          '68197fd20bef1eae52c4c232164a027e095d567d4bfb37aea79c4f908853ea42',
          'cctv_dgp_bank_comparison_v1_r2_B'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=ROUTES, required=True)
    route = parser.parse_args().route
    name, stem, pin, prefix = ROUTES[route]
    importer.NAME = name
    importer.STEM = stem
    importer.PIN = pin
    importer.PREPARED = importer.OUTPUT / name
    importer.RETURN = importer.OUTPUT / (prefix + '_return')
    importer.ANALYSIS = importer.OUTPUT / (prefix + '_analysis')
    preparation = importer.OUTPUT / 'cctv_dgp_bank_comparison_v1_r3_preparation.json'
    prepared = importer.read(preparation)
    entry, = [r for r in prepared['routes'] if r['route'] == route]
    assert prepared['complete'] and entry['name'] == name and entry['stem'] == stem
    assert entry['protocol_sha256'] == pin
    wrapper = Path(__file__).resolve()
    original = Path(importer.__file__).resolve()
    bound = {str(path): importer.sha(path) for path in (wrapper, original, preparation)}
    importer.main()
    assert all(importer.sha(Path(path)) == value for path, value in bound.items())
    receipt = dict(complete=True, route=route, protocol_sha256=pin,
                   import_receipt_sha256=importer.sha(importer.ANALYSIS / 'import.json'),
                   source_bindings_unchanged=bound, VM_trainer_imported_or_run=False,
                   local_optimizer_gradient_or_training_calls=0, quality_qualified=False)
    with (importer.ANALYSIS / 'route_import.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
