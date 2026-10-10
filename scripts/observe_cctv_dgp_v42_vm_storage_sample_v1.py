"""Bounded read-only refresh of the already inventoried manual V42 worker."""
import argparse
import re
from pathlib import Path

import observe_cctv_dgp_v42_vm_storage_v1 as observer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--sample', required=True)
    a = parser.parse_args(); assert re.fullmatch('[a-z0-9_]{1,48}', a.sample)
    out = observer.OUT; transport = observer.transport
    assert transport.transport.read(out/'independent_audit.json')['complete']
    transport.OUT = out; transport.LOGS = out/'transport'
    assert not (out/(a.sample+'.json')).exists()
    transport.ssh(observer.SOURCE, a.sample, 90)
    value = transport.transport.read(transport.LOGS/(a.sample+'_stdout.log'))
    assert value['complete'] and value['files_removed'] == value['processes_started_or_stopped'] == 0
    transport.transport.write(out/(a.sample+'.json'), value)
    print(value, flush=True)


if __name__ == '__main__': main()
