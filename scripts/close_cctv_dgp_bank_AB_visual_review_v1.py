"""Bind separately recorded actual sheet observations to independent audits."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def close(name,kind):
    analysis=ROOT/'outputs'/name;dest=analysis/'visual_review'
    receipt=analysis/'visual_review.json'
    assert not receipt.exists()
    manifest=read(dest/'sheet_manifest.json');audit=read(analysis/'independent_audit.json')
    assert manifest['complete'] and audit['complete']
    bindings={}
    def bind(path):bindings[path.relative_to(ROOT).as_posix()]=sha(path)
    bind(analysis/'independent_audit.json');bind(dest/'sheet_manifest.json')
    sheets={row['name']:row['sha256'] for row in manifest['sheets']}
    notes=analysis/'visual_observations'
    assert {file.name for file in notes.iterdir()}=={name+'.json' for name in sheets}
    for name,expected in sheets.items():
        assert sha(dest/name)==expected
        observation=read(notes/(name+'.json'))
        assert observation['complete'] and observation['sheet']==name
        assert observation['sheet_sha256']==expected
        assert observation['sheet_manifest_sha256']==sha(dest/'sheet_manifest.json')
        assert observation['displayed_through']=='view_image at original resolution'
        assert observation['reviewer']=='implementing assistant' and not observation['independent_final_review']
        assert len(observation['notes'])>=30
        bind(dest/name);bind(notes/(name+'.json'))
    for name,expected in manifest['source_tile_bindings'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        assert sha(path)==expected
    if kind=='AB':
        assert len(sheets)==26 and manifest['paired_sheets']==20 and manifest['native_sheets']==6
        assert manifest['missing_tiles']==[]
        for name in ['import.json','route_import.json','execution_audit.json',
                     'preview_integrity_audit.json','matched_AB_comparison.json','saved_learning_diagnostic.json']:
            assert read(analysis/name)['complete'];bind(analysis/name)
        assert audit['saved_paired_raw_PNG_cases_checked']==7910 and audit['native_compositions_checked']==72
        assert audit['optimizer_updates_in_VM']==50 and not audit['model_qualified']
        assert all(not value['pass'] for value in audit['metric_and_gate_decisions'].values())
        paired,native=100,24
    else:
        assert len(sheets)==5 and manifest['paired_sheets']==4 and manifest['native_sheets']==1
        assert audit['paired_compositions_checked']==200 and audit['native_unpaired_compositions_checked']==48
        assert len(audit['independent_CPU_spot_replays'])==4
        assert not audit['trained_candidate_qualification']
        assert audit['local_gradient_queries']==audit['local_optimizer_updates']==0
        paired,native=20,4
    before=read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/local_protected_before.json')
    for name,expected in before['files_sha256'].items():assert sha(ROOT/name)==expected
    bind(ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/local_protected_before.json')
    for name,expected in bindings.items():assert sha(ROOT/name)==expected
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        closure_checker_sha256=sha(__file__),evidence_files_sha256=bindings,
        viewed_sheets=len(sheets),paired_sheets=manifest['paired_sheets'],native_sheets=manifest['native_sheets'],
        paired_TRAIN_preview_cases_viewed=paired,native_DEV_unpaired_cases_viewed=native,
        all_planned_sheets_viewed=True,source_tile_bindings_rechecked=len(manifest['source_tile_bindings']),
        observations_retained=True,reviewer='implementing assistant; original-resolution view_image displays',
        independent_saved_output_audit_complete=True,independent_final_review=False,
        limited_diagnostic_visual_scope=kind!='AB',local_gradient_queries=0,local_optimizer_updates=0,
        retained_original_checkpoint_and_app_bindings_unchanged=True,
        final_evaluation_identities_used=False,all_seven_completion_families_pending=True,
        model_qualified=False,app_promotion=False,goal_complete=False)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(kind=kind,viewed_sheets=len(sheets),qualified=False))


if __name__=='__main__':
    close('cctv_dgp_bank_comparison_v1_r4_B_analysis','AB')
    close('cctv_dgp_bank_feature_DC_probe_v1_analysis','DC')
