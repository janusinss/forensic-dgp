"""Check the actual process WorkingSet counter with a touched 64 MiB allocation."""
from pathlib import Path
import importlib.util
import json
import numpy as np


def main():
    f=Path(__file__).with_name('dgp_detector_process_memory_v1.py')
    assert f.is_file(),'Dependency-free process-memory reader is not implemented'
    spec=importlib.util.spec_from_file_location('memory_reader',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    before=m.resident_bytes();assert isinstance(before,int) and 1024**2<before<4*1024**3
    a=np.full(64*1024**2,1,dtype=np.uint8);assert a.sum()==64*1024**2
    after=m.resident_bytes();assert after-before>=48*1024**2,(before,after)
    result=dict(complete=True,reader='Windows GetProcessMemoryInfo WorkingSetSize',before=before,after=after,
        touched_array_bytes=a.nbytes,minimum_measured_increase_bytes=48*1024**2,neural_or_training_calls=0)
    with (f.resolve().parents[1]/'outputs/dgp_detector_process_memory_v1_check.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(result)


if __name__=='__main__':main()
