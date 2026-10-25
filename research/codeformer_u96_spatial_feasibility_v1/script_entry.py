"""Copy this inert bridge to scripts/run_u96_spatial_feasibility_v1.py."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_u96_spatial_feasibility_v1.stage_entry import main


if __name__ == '__main__':
    raise SystemExit(main())
