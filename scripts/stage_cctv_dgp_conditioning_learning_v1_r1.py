"""Preserve the first sealed draft; correct finite-stop and numeric guards."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def save(path,text):
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


def main():
    old=ROOT/'outputs/cctv_dgp_conditioning_learning_v1_vm'
    before=json.loads((old/'protocol.json').read_text())
    worker=(old/'scripts/cctv_dgp_conditioning_learning_v1_vm.py').read_text(encoding='utf-8')
    worker=worker.replace('conditioning_learning_v1','conditioning_learning_v1_r1').replace('conditioning-learning-v1','conditioning-learning-v1-r1')
    assert worker.count("        assert time.monotonic()-start<p['worker_seconds'],'Worker time stop'")==1
    worker=worker.replace("        assert time.monotonic()-start<p['worker_seconds'],'Worker time stop'",
        "        assert time.monotonic()-start<p['worker_seconds'],'Worker time stop'\n"
        "        assert time.monotonic()-start<p['measurement_seconds'],'Learning measurement stop'")
    assert worker.count('        clock();verify(root,parent,pin)')==1
    worker=worker.replace('        clock();verify(root,parent,pin)','        clock();verify(root,parent,pin);clock()')
    save(ROOT/'scripts/cctv_dgp_conditioning_learning_v1_r1_vm.py',worker)
    contract=(old/'conditioning_contract.py').read_text(encoding='utf-8')
    anchor='    stages={}\n'
    assert contract.count(anchor)==1
    contract=contract.replace(anchor,
        "    for collection in [initial,previous,after]:\n"
        "        assert set(collection)=={'raw','png'}\n"
        "        for groups in collection.values():\n"
        "            assert len(groups)==17\n"
        "            for row in groups.values():\n"
        "                assert row['cases']>0\n"
        "                assert all(np.isfinite(row[m]) for m in ['MSE','SSIM','ArcFace_observed_fixed',\n"
        "                    'landmark_high_frequency_MSE','constant_mean_shift_only_MSE']), 'Nonfinite scientific metric; retain stop'\n"
        "                assert row['MSE']>=0 and row['landmark_high_frequency_MSE']>0\n"
        +anchor)
    save(ROOT/'scripts/cctv_dgp_conditioning_learning_v1_r1_contract.py',contract)
    for source,target in [
        ('prepare_cctv_dgp_conditioning_learning_v1.py','prepare_cctv_dgp_conditioning_learning_v1_r1.py'),
        ('prepare_cctv_dgp_conditioning_learning_v1_return_checker.py','prepare_cctv_dgp_conditioning_learning_v1_r1_return_checker.py'),
        ('audit_cctv_dgp_conditioning_learning_v1_return.py','audit_cctv_dgp_conditioning_learning_v1_r1_return.py')]:
        text=(ROOT/'scripts'/source).read_text(encoding='utf-8')
        text=text.replace('conditioning_learning_v1','conditioning_learning_v1_r1').replace('conditioning-learning-v1','conditioning-learning-v1-r1')
        save(ROOT/'scripts'/target,text)
    stop=ROOT/'outputs/cctv_dgp_conditioning_learning_v1_preparation_review.json'
    record=dict(complete=True,status='First draft superseded before manual publication; preserve sealed packet',
        original_protocol_sha256=hashlib.sha256((old/'protocol.json').read_bytes()).hexdigest(),
        original_archive_sha256=hashlib.sha256((ROOT/'outputs/cctv-dgp-conditioning-learning-v1-execution.tar.gz').read_bytes()).hexdigest(),
        causes=['Internal1350s measurement cap was not checked at every clock or after final verification',
                'Reject nonfinite scientific group metrics explicitly, including SSIM/ArcFace'],
        scientific_thresholds_and_architecture_unchanged=True,actual_training_not_run_by_agent=True,
        original_packet_retained=True,new_R1_only_for_forthcoming_manual_commands=True)
    save(stop,json.dumps(record,indent=2)+'\n')
    print(dict(complete=True,original_sealed_draft_preserved=True,fresh_R1_staged=True))


if __name__=='__main__':main()
