"""Reuse the reviewed hash-before-extract reader under an exact new stage."""
from datetime import datetime,timezone
import ast
from pathlib import Path
import types

from cctv_dgp_decoder_feasibility_v1_contract import NAME,STEM,STAGE,read,write,sha,verify_transfer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
PREP=OUT/'cctv_dgp_decoder_feasibility_v1_preparation'
RETURN=OUT/'cctv_dgp_decoder_feasibility_v1_return'
ANALYSIS=OUT/'cctv_dgp_decoder_feasibility_v1_analysis'
READER_SHA='24f04a29d7745c0f4e35ae450ee82ea1b05e900003c27e0f6e59d2ea00c0562d'


def reader():
    path=ROOT/'scripts/import_cctv_dgp_decoder_activation_v1_return.py'
    assert sha(path)==READER_SHA,'Preserve reviewed hash-before-extraction reader'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    # Run the local known reader definitions, never returned executable code.
    # Replace only its bound STAGE namespace; the extraction algorithm is exact.
    module=types.ModuleType('feasibility_hash_before_extract_reader')
    module.__file__=str(path)
    exec(compile(tree,str(path),'exec'),module.__dict__)
    module.STAGE=STAGE
    return module


def main():
    prepared=read(PREP/'preparation.json')
    assert read(PREP/'independent_preparation_audit.json')['complete']
    packet=OUT/NAME;pin=sha(packet/'protocol.json')
    assert pin==prepared['protocol_sha256']
    assert not RETURN.exists() and not ANALYSIS.exists(),'Retain previous complete or partial return'
    p=read(packet/'protocol.json')
    imported=reader().import_verified(OUT/(STEM+'-results.tar.gz'),OUT/(STEM+'-results.tar.gz.sha256'),
        OUT/(STEM+'-export.json'),RETURN,p,pin)
    activation,repair,source,parent,replay=[OUT/name for name in [
        'cctv_dgp_decoder_activation_v1_return','cctv_dgp_decoder_hinge_repair_v1_return',
        'cctv_dgp_spatial_readout_v1_return','cctv_dgp_bank_photometric_guard_v1_return',
        'cctv_dgp_spatial_path_replay_v1_return']]
    p,ap,rp,old,data=verify_transfer(RETURN,activation,repair,source,parent,replay,pin)
    ANALYSIS.mkdir()
    write(ANALYSIS/'import.json',dict(imported,UTC=datetime.now(timezone.utc).isoformat(),
        reader_sha256=READER_SHA,stage_rebound_only=STAGE,numerical_audit_pending=True,
        local_neural_calls=0,local_solver_calls=0,local_gradient_queries=0,local_optimizer_updates=0,VM_calls=0))
    from audit_cctv_dgp_decoder_feasibility_v1_return import audit_saved
    audit_saved(RETURN,activation,repair,source,parent,replay,p,ap,rp,old,data,ANALYSIS/'independent_audit.json')


if __name__=='__main__':main()
