"""Publish the prepared R1 tail using a distinct guide; preserve failed publication."""
from pathlib import Path
import importlib.util
import time
from cctv_dgp_finite_guard_v1_r1_contract import NAME,STEM,read,write,sha,verified_assets

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic(); bundle = ROOT/'outputs'/NAME
    prep = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_preparation'
    assert prep.is_dir() and not (prep/'prepared.json').exists()
    pin = sha(bundle/'protocol.json'); p = verified_assets(bundle,pin)
    for name,digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest,name
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz'); digest = sha(archive)
    assert Path(str(archive)+'.sha256').read_text().split() == [digest,archive.name]
    # Original guide remains exact, proving the exclusive-create guard preserved it.
    original_guide = ROOT/'CCTV_DGP_FINITE_GUARD_V1_VM.md'; assert original_guide.is_file()
    module_path = ROOT/'scripts/prepare_cctv_dgp_finite_guard_v1_r1.py'
    spec = importlib.util.spec_from_file_location('frozen_r1_guide',module_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    projection = int(10*(p['prior_baseline_bytes']+2*1024**2)*1.25+p['projected_gradient_and_snapshot_bytes'])
    guide = ROOT/'CCTV_DGP_FINITE_GUARD_V1_R1_VM.md'
    module.text(guide,module.guide(pin,digest,archive.stat().st_size,projection))
    write(prep/'publication_failure_retained.json',{'complete':True,
        'location':'prepare_cctv_dgp_finite_guard_v1_r1.py:207',
        'cause':'FileExistsError: exclusive creation of the already preserved V1 manual guide',
        'fix':'Publish the same frozen packet commands to a distinct V1_R1 guide; no packet/source changes',
        'frozen_builder_sha256':sha(module_path),'prior_guide_sha256':sha(original_guide),
        'published_guide_sha256':sha(guide),'packet_sha256':digest,'protocol_sha256':pin,
        'VM_training_launched':False,'completion_script_sha256':sha(Path(__file__))})
    write(prep/'prepared.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':digest,
        'archive_bytes':archive.stat().st_size,'assets':len(p['assets_sha256']),'cases':100,
        'maximum_training_changes':3,'optimizer_updates_executed':0,'neural_calls':0,
        'VM_training_launched':False,'independent_packet_audit_pending':True,
        'projected_return_bytes':projection,'publication_guard_failure_preserved':True,
        'manual_guide_sha256':sha(guide),'seconds_for_tail_publication':time.monotonic()-started})
    print({'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size})


if __name__ == '__main__': main()
