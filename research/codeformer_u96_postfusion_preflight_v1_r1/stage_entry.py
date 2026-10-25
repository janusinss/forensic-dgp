"""Saved stage CLI; importing this file does not launch a diagnostic."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    from research.codeformer_u96_postfusion_preflight_v1_r1.entry import main
    raise SystemExit(main())
