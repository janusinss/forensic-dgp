"""Exercise real packet staging on tiny synthetic bytes; no model/NN imports."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("completion_gradient_prepare",ROOT/"scripts/prepare_completion_codeformer_gradient_v1.py")
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)

class PacketStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="completion-gradient-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"source";self.root.mkdir()
        def file(relative,data):
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
        self.file=file
        sources=[]
        for relative in prep.SOURCE_FILES:
            path=file(relative,("fixture:"+relative).encode())
            if relative not in prep.NEW_RESEARCH_FILES:sources.append({'path':str(path),**prep.pin(path)})
        self.checkpoint=file('weights/test-inpainting.pth',b'not-a-real-model-no-neural-loading')
        self.admission=file('outputs/completion_codeformer_manual_v1/source_admission.json',json.dumps({'complete':True,'independent':True,'source_revision':'b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b','source_files':sources}).encode())
        rows=[];cases=[]
        for n in range(391):
            ref='tr_ffhq_'+str(n).zfill(5)
            row={'reference_id':ref,'original_role':'train','legacy_completion_split':'train'}
            if n<4:
                original=file('fixtures/'+ref+'.png',('source'+ref).encode())
                target=file('fixtures/'+ref+'_target.png',('target'+ref).encode())
                row.update({'source':{'path':str(original),**prep.pin(original)},'target256':{'path':str(target),**prep.pin(target)},'source_size':[1024,1024],'source_review_decision':'accept_clean_restoration','attribution':{'photo_url':'https://example.invalid/'+ref,'license':'fixture'}})
                for kind in ('lower_face','eye_band'):
                    mask=file('fixtures/'+ref+'__'+kind+'.png',('mask'+ref+kind).encode())
                    cases.append({'id':ref+'__'+kind,'reference_id':ref,'mask_kind':kind,'mask256':{'path':str(mask),**prep.pin(mask)},'mask_pixels':100})
            rows.append(row)
        self.readiness=file('readiness.json',json.dumps({'complete':True,'independent':True,'genuine_HQ_candidate_sources':{'training_sources':rows}}).encode())
        self.probe=file('probe.json',json.dumps({'selected_sources':rows[:4],'cases':cases}).encode())
        self.destination=Path(self.temp.name)/'staged'
        patches={'ROOT':self.root,'CHECKPOINT':self.checkpoint,'CHECKPOINT_BYTES':self.checkpoint.stat().st_size,'CHECKPOINT_SHA':prep.sha(self.checkpoint),'READINESS':self.readiness,'READINESS_SHA':prep.sha(self.readiness),'PROBE':self.probe,'ADMISSION_SHA':prep.sha(self.admission)}
        for key,value in patches.items():
            context=patch.object(prep,key,value);context.start();self.addCleanup(context.stop)
    def stage(self):
        with contextlib.redirect_stdout(io.StringIO()):prep.stage(self.destination,prep.sha(self.probe))
    def test_complete_tiny_staging_freezes_all_required_members(self):
        before=prep.pin(self.checkpoint)
        self.stage();protocol=json.loads((self.destination/'protocol.json').read_text())
        names={row['path'] for row in protocol['files']}
        self.assertTrue(prep.NEW_RESEARCH_FILES<=names)
        self.assertIn('evidence/source_admission.json',names)
        self.assertIn('evidence/provenance_and_terms.json',names)
        self.assertEqual(len(protocol['references']),4);self.assertEqual(len(protocol['cases']),8)
        self.assertEqual(protocol['optimizer_updates'],0)
        for row in protocol['files']:
            self.assertEqual(prep.pin(self.destination/row['path']),{k:row[k] for k in ('bytes','sha256')})
        self.assertEqual(before,prep.pin(self.checkpoint))
        self.assertEqual((self.destination.parent/'protocol.sha256').read_text().strip(),prep.sha(self.destination/'protocol.json'))
        self.assertNotIn('torch',sys.modules)
    def test_missing_exporter_stops_before_creating_packet(self):
        (self.root/'scripts/export_completion_codeformer_gradient_v1.py').unlink()
        with self.assertRaisesRegex(ValueError,'Missing runtime source'):self.stage()
        self.assertFalse(self.destination.exists())
    def test_changed_admission_stops_before_creating_packet(self):
        with self.admission.open('ab') as stream:stream.write(b' ')
        with self.assertRaisesRegex(ValueError,'Architecture source admission drift'):self.stage()
        self.assertFalse(self.destination.exists())

if __name__=='__main__':unittest.main(verbosity=2)
