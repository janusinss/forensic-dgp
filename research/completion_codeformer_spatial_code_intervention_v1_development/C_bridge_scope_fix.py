from pathlib import Path
import ast,shutil
D=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\research\completion_codeformer_spatial_code_intervention_v1_development');backup=D/'prior_revision_002_before_C_bridge_scope_fix';backup.mkdir()
for name in ('cctv_completion_codeformer_spatial_code_intervention_v1_vm.py','source_tests.py','source_preparation_tests.json'):shutil.copyfile(D/name,backup/name)
p=D/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';s=p.read_text();s=s.replace("                old_rgb=torch.from_numpy(old['input256']).to('cuda');old_mask=torch.from_numpy(old['mask256']).to('cuda');old_input=app_input(old_rgb,old_mask)\n",'')
s=s.replace("'input512_exact_from_exact_source_and_pinned_rule':bool(torch.equal(x,old_input)),'old_input512_not_stored_reconstructed_only':True,", "'conditioning512_evidence':'Four fresh owned-versus-original app parity pairs; historical C has no saved input512',")
s=s.replace(" and r['input512_exact_from_exact_source_and_pinned_rule']",'');ast.parse(s,feature_version=(3,10));p.write_text(s,encoding='utf-8',newline='\n')
t=D/'source_tests.py';s=t.read_text();at=s.index(' def test_18_root_design_and_manual_owned_supervisor(self):');s=s[:at]+''' def test_20_C_bridge_does_not_invent_historical512_layout(self):
  self.assertNotIn('old_input=app_input',SOURCE);self.assertNotIn('input512_exact_from_exact_source_and_pinned_rule',SOURCE);self.assertIn('historical C has no saved input512',SOURCE)
  self.assertIn("'input256_exact':np.array_equal",SOURCE);self.assertIn("'mask256_exact':np.array_equal",SOURCE);self.assertIn("'predicted_codes_exact':np.array_equal",SOURCE);self.assertIn("'used_codes_exact':np.array_equal",SOURCE);self.assertIn("'PNG_pixels_exact':np.array_equal",SOURCE)
''' +s[at:];ast.parse(s,feature_version=(3,10));t.write_text(s,encoding='utf-8',newline='\n');(D/'source_preparation_tests.json').unlink()
print({'worker_bytes':p.stat().st_size,'C_bridge_scope':'Stored256/mask/raw512/result/pred/used/PNGonly;conditioning512paritiesseparate'})
