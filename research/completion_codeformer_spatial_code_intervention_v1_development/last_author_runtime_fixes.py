from pathlib import Path
D=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\research\completion_codeformer_spatial_code_intervention_v1_development');p=D/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';s=p.read_text();s=s.replace("value.dtype==np.int64 and value.shape==(1,256),'Saved256indices", "value.dtype==np.int64 and value.shape==(1,256) and value.min()>=0 and value.max()<512,'Saved256indices")
s=s.replace("set(z.namelist())=={'prequant_latent.npy','nearest_codes.npy'} and z.testzip() is None", "len(z.infolist())==2 and set(z.namelist())=={'prequant_latent.npy','nearest_codes.npy'} and z.testzip() is None")
s=s.replace('with np.load(path,allow_pickle=False) as z:', 'with np.load(path,allow_pickle=False,max_header_size=512) as z:')
s=s.replace("r['max_raw256_error']<=1e-6 and r['input512_exact']", "r['max_raw256_error']<=1e-6 and r['model_raw512_max_error']<=1e-6 and r['input512_exact']")
p.write_text(s,encoding='utf-8',newline='\n')
