"""Retain an edited guide and recover its exact historical bytes for hash closure."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v34_return_v35_closure_basis_v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert not OUT.exists() and not (ROOT/'outputs/cctv_dgp_v34_return_v35_probe_milestone').exists()
    old=json.loads((ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone/milestone.json').read_text())
    guide=ROOT/'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md'
    expected=old['new_evidence_sha256'][guide.name]
    source=ROOT/'scripts/prepare_cctv_dgp_group_guard_grad_v34.py'
    assert sha(source)==old['new_evidence_sha256'][source.relative_to(ROOT).as_posix()]
    spec=importlib.util.spec_from_file_location('verified_V34_guide_generator_only',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.guide('d03050e18e4fd2369a6dd0803632bebfa0c709cc95cc8f7a6c8776785b0923a1',
        'db822ba0b17f941d8e20783e5ee73ed97586dfdfb0f53a4d4909d392bd4b2656').encode('utf-8')
    assert hashlib.sha256(original).hexdigest()==expected
    command='gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/cctv-dgp-group-guard-grad-v34-results.tar.gz.sha256" "."\n'
    assert original.decode().count(command)==1
    assert guide.read_text(encoding='utf-8')==original.decode().replace(command,command+command)
    OUT.mkdir()
    (OUT/'V34_original_runbook.md').write_bytes(original)
    (OUT/'V34_live_edited_runbook.md').write_bytes(guide.read_bytes())
    for name in ['record_cctv_dgp_v34_return_v35_probe_milestone.py','verify_cctv_dgp_v34_return_v35_probe_milestone.py']:
        path=ROOT/'scripts'/name
        with (OUT/name).open('xb') as stream:stream.write(path.read_bytes())
        text=path.read_text(encoding='utf-8')
        text=text.replace("PREVIOUS=ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone'",
            "PREVIOUS=ROOT/'outputs/cctv_dgp_v33_return_v34_diagnostic_milestone'\nOVERRIDES={'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md':'outputs/cctv_dgp_v34_return_v35_closure_basis_v1/V34_original_runbook.md'}")
        if name.startswith('record_'):
            text=text.replace("for name,digest in old['new_evidence_sha256'].items():assert sha(ROOT/name)==digest,name",
                "for name,digest in old['new_evidence_sha256'].items():assert sha(ROOT/OVERRIDES.get(name,name))==digest,name")
            text=text.replace("'record_cctv_dgp_v34_return_v35_probe_milestone.py','verify_cctv_dgp_v34_return_v35_probe_milestone.py'",
                "'record_cctv_dgp_v34_return_v35_probe_milestone.py','verify_cctv_dgp_v34_return_v35_probe_milestone.py',\n        'record_cctv_dgp_v34_return_v35_probe_milestone_r1.py','verify_cctv_dgp_v34_return_v35_probe_milestone_r1.py',\n        'preserve_cctv_dgp_v34_closure_basis_and_prepare_r1.py'")
            text=text.replace("'cctv_dgp_group_guard_probe_v35_r1_vm','cctv_dgp_group_guard_probe_v35_r1_preparation']:",
                "'cctv_dgp_group_guard_probe_v35_r1_vm','cctv_dgp_group_guard_probe_v35_r1_preparation',\n        'cctv_dgp_v34_return_v35_closure_basis_v1']:")
            text=text.replace("files=[handoff,saved,", "files=[handoff,saved,ROOT/'CCTV_DGP_GROUP_GUARD_GRAD_V34_VM.md',")
            text=text.replace("'previous_milestone_sha256':sha(PREVIOUS/'milestone.json'),", "'previous_file_overrides':OVERRIDES,'previous_milestone_sha256':sha(PREVIOUS/'milestone.json'),")
            text=text.replace("The entire previous handoff follows.",
                "The edited V34 runbook's duplicated checksum-download line is retained. Its\noriginal guide is recovered exactly from its hash-matched generator and bound\nseparately for historical closure; no live user file was overwritten.\n\nThe entire previous handoff follows.")
        else:
            text=text.replace("path=ROOT/m['previous_handoff_path'] if name=='PROJECT_HANDOFF.md' else ROOT/name",
                "path=ROOT/m['previous_handoff_path'] if name=='PROJECT_HANDOFF.md' else ROOT/m['previous_file_overrides'].get(name,name)")
            text=text.replace("'complete_previous_handoff_preserved':True,", "'complete_previous_handoff_preserved':True,'historical_runbook_recovered_to_exact_hash':True,'live_edited_runbook_preserved':True,")
        ast.parse(text,feature_version=(3,10))
        revised=path.with_name(path.stem+'_r1.py')
        with revised.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    receipt={'complete':True,'failure_location':'record_cctv_dgp_v34_return_v35_probe_milestone.py before writes',
        'failure_tool_chunk':'fbe450','diagnostic_tool_chunk':'0e1d34',
        'cause':'One duplicate checksum download command in live V34 runbook; other5960 prior bindings unchanged',
        'original_expected_sha256':expected,'recovered_original_sha256':sha(OUT/'V34_original_runbook.md'),
        'live_edited_guide_sha256':sha(guide),'verified_generator_sha256':sha(source),
        'live_guide_modified':False,'checkpoint_or_quality_gate_changes':False,'neural_or_VM_calls':0}
    with (OUT/'failure_and_recovery.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'complete':True,'original_hash_recovered':True,'live_guide_preserved':True}))


if __name__=='__main__':main()
