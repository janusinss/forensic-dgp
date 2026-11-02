from pathlib import Path
import ast,shutil
D=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\research\completion_codeformer_spatial_code_intervention_v1_development');backup=D/'prior_revision_003_before_failure_witness_metadata';backup.mkdir()
for name in ('cctv_completion_codeformer_spatial_code_intervention_v1_vm.py','source_tests.py','source_preparation_tests.json'):shutil.copyfile(D/name,backup/name)
p=D/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';s=p.read_text();s=s.replace("{'actual':actual,'equals_R1_before':actual==before,'counts':dict(counts)}", "{'actual':actual,'equals_R1_before':actual==before,'counts':dict(counts),'actual_path_witness':dict(witness)}")
s=s.replace("{'error':repr(state_error),'counts':dict(counts)}", "{'error':repr(state_error),'counts':dict(counts),'actual_path_witness':dict(witness)}")
s=s.replace("'progress':progress,'counts':counts,'typed_forward_slots':typed,'partial_outputs_retained':True", "'progress':progress,'counts':counts,'typed_forward_slots':typed,'actual_path_witness':dict(witness),'partial_outputs_retained':True")
ast.parse(s,feature_version=(3,10));p.write_text(s,encoding='utf-8',newline='\n')
t=D/'source_tests.py';s=t.read_text();at=s.index(' def test_18_root_design_and_manual_owned_supervisor(self):');s=s[:at]+''' def test_21_failure_metadata_preserves_actual_path_witness(self):
  for filename in ('state_on_failure.json','state_receipt_failure.json','failure.json'):
   lines=[line for line in SOURCE.splitlines() if "record(out/'"+filename+"'" in line];self.assertEqual(len(lines),1);self.assertIn("'actual_path_witness':dict(witness)",lines[0])
''' +s[at:];ast.parse(s,feature_version=(3,10));t.write_text(s,encoding='utf-8',newline='\n');(D/'source_preparation_tests.json').unlink()
print({'worker_bytes':p.stat().st_size,'failure_witness_preserved':True})
