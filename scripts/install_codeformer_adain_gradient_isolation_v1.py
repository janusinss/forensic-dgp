"""Manual CPU byte-copy installer only; never starts model execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.prepare_codeformer_adain_gradient_isolation_v1 import stage_dependencies


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args();print(json.dumps(stage_dependencies(args.root,args.protocol_sha),indent=2))
