"""CPU opaque-byte export/refusal fixtures; no framework or VM work."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import zlib
import struct

from research.codeformer_role_preserving_training_v1 import contract as c, transport as t
from scripts import supervise_codeformer_role_preserving_training_v1_vm as s

ROOT=Path(__file__).resolve().parents[1]


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    return {'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)}


def encoded(value):return (json.dumps(value,indent=2)+'\n').encode()


class TransportCPU(unittest.TestCase):
    def fixture(self,terminal='failure'):
        base=ROOT/'scratch'/('role_transport_'+uuid.uuid4().hex);root=base/'root';home=base/'home'
        root.mkdir(parents=True);home.mkdir()
        pin=put(root/'research/example.py',b'raise RuntimeError("never execute")\n')
        bundle={'training':[{'id':'train_u%04d_slot%d'%(u,slot)}for u in range(1,51)for slot in range(1,5)],
            'hq':[{'reference_id':'tr_ffhq_%05d'%n}for n in range(100)],
            'development':[{'id':'case_%03d'%n}for n in range(406)]}
        p={'files_sha256':{'research/example.py':pin['sha256']},'files_bytes':{'research/example.py':pin['bytes']},
            'parent_dir':c.PARENT_NAME,'parent_protocol_sha256':c.PARENT_PROTOCOL_SHA,
            'dependencies':{'protocol.json':{'sha256':'a'*64,'bytes':0}},'retained_output_dependencies':{},'extra_returned_dependencies':{}}
        sha=put(root/'protocol.json',encoded(p))['sha256']
        put(root/'outputs/supervision/run_receipt.json',encoded({'worker_terminal':True,'protocol_sha256':sha}))
        for name in ('preflight.log','diagnostic.log','export_invocation.json'):put(root/'outputs/supervision'/name,b'{}\n')
        put(root/'outputs/supervision/export.log',b'active stdout boundary')
        if terminal:
            counts=dict.fromkeys(c.COUNTS,0);counts.update(candidate_training_forwards=8,backward_calls=8,optimizer_step_attempts=2)
            value={'complete':True,'failure':True,'protocol_sha256':sha,'counts':counts,
                'completed_training_updates':2,'partial_microbatches':0,'optimizer_inflight':False}
            put(root/t.OUTPUTS/'failure.json',encoded(value))
        return SimpleNamespace(root=root,home=home,p=p,bundle=bundle,sha=sha)

    @contextmanager
    def environment(self,f):
        with ExitStack()as stack:
            stack.enter_context(mock.patch.object(c,'manual_root',side_effect=lambda p:Path(p)))
            stack.enter_context(mock.patch.object(t,'verify_transfer',return_value={'complete':True,'protocol':f.p,'data_bundle':f.bundle}))
            stack.enter_context(mock.patch.object(t,'_process_rows',return_value=[]))
            stack.enter_context(mock.patch.object(Path,'home',return_value=f.home))
            stack.enter_context(mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=32*1024**3)))
            yield

    def test_failure_export_actual2updates_and_missingcoverage(self):
        f=self.fixture()
        with self.environment(f):r=t.export_results(f.root,f.sha)
        self.assertEqual(r['completed_training_updates'],2);self.assertFalse(r['model_qualified'])
        self.assertTrue(r['failure_present']);self.assertFalse(r['output_coverage']['complete'])
        archive=f.home/(t.PREFIX+'-results.tar.gz')
        self.assertEqual(Path(str(archive)+'.sha256').read_bytes(),(c.sha256(archive)+'  '+archive.name+'\n').encode('ascii'))
        compressed=archive.read_bytes();decoder=zlib.decompressobj(31);raw=decoder.decompress(compressed)
        self.assertTrue(decoder.eof);self.assertFalse(decoder.unused_data)
        self.assertEqual(struct.unpack('<II',compressed[-8:]),(zlib.crc32(raw),len(raw)))
        position=0;members={}
        while raw[position:position+512]!=b'\0'*512:
            header=raw[position:position+512];name=header[:100].split(b'\0')[0].decode();prefix=header[345:500].split(b'\0')[0].decode()
            if prefix:name=prefix+'/'+name
            self.assertEqual(header[257:265],b'ustar\0' + b'00');self.assertEqual(header[156:157],b'0')
            size=int(header[124:136].rstrip(b'\0'),8);value=raw[position+512:position+512+size]
            self.assertEqual(header[100:108],b'0000644\0');self.assertEqual(header[108:124],b'0000000\0'*2)
            checksum=int(header[148:154],8);self.assertEqual(checksum,sum(header[:148])+8*32+sum(header[156:]))
            members[name]=hashlib.sha256(value).hexdigest();padding=(-size)%512
            self.assertEqual(raw[position+512+size:position+512+size+padding],b'\0'*padding)
            position+=512+size+padding
        self.assertGreaterEqual(len(raw)-position,1024);self.assertEqual(raw[position:],b'\0'*(len(raw)-position))
        manifest=c.read_json(f.root/t.OUTPUTS/'export_manifest.json')
        self.assertEqual(set(members),set(manifest['files_sha256'])|{t.OUTPUTS+'/export_manifest.json'})
        self.assertTrue(all(members[n]==h for n,h in manifest['files_sha256'].items()))
        with self.environment(f),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_absent_worker_activity_unknown(self):
        f=self.fixture(None)
        r=t._terminal_evidence(f.root,t._walk(f.root),f.sha,f.bundle)
        self.assertIsNone(r['completed_training_updates']);self.assertIsNone(r['worker_recorded_counts'])
        self.assertTrue(r['activity_unknown_without_worker_receipt'])

    def test_counter_boolean_overflow_and_commit_inconsistency_refused(self):
        for mutation in ('bool','overflow','commit'):
            f=self.fixture();path=f.root/t.OUTPUTS/'failure.json';v=c.read_json(path)
            if mutation=='bool':v['completed_training_updates']=True
            elif mutation=='overflow':v['counts']['gradient_queries']=6
            else:v['completed_training_updates']=3
            put(path,encoded(v))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):t._terminal_evidence(f.root,t._walk(f.root),f.sha,f.bundle)

    def test_checkpoint_exactnames_caps_temps_exclusive50stopped(self):
        cp=c.CHECKPOINT_PATHS
        self.assertEqual(t.checkpoint_kind(cp[0]),'final')
        self.assertEqual(t.checkpoint_kind(cp[2]+'.tmp-'+'a'*32),'temporary')
        self.assertIsNone(t.checkpoint_kind(cp[2]+'.tmp-nope'))
        with self.assertRaises(ValueError):t.checkpoint_accounting({cp[1]:1,cp[2]:1})
        with self.assertRaises(ValueError):t.checkpoint_accounting({cp[0]:1,cp[0]+'.tmp-'+'a'*32:1})
        with self.assertRaises(ValueError):t.checkpoint_accounting({cp[0]:c.LIMITS['initial_checkpoint_file_bytes']+1})
        self.assertEqual(t.checkpoint_accounting({cp[0]:1,cp[2]+'.tmp-'+'a'*32:1})['temporary_bytes'],1)

    def test_wrongcheckpoint_and_jsoncap_refuse(self):
        f=self.fixture();name=t.OUTPUTS+'/arbitrary.pth';put(f.root/name,b'opaque')
        with self.assertRaises(ValueError):t._measure_files(f.root,[name])
        destination=f.root/t.OUTPUTS/'oversize.json'
        with self.environment(f),self.assertRaises(ValueError):t._write_reserved_json(destination,{'payload':'x'*c.LIMITS['one_JSON_bytes']})
        self.assertFalse(destination.exists())

    def test_exact50_alias_and_mismatch_temp_refusal(self):
        f=self.fixture();cp=c.CHECKPOINT_PATHS[1];pin=put(f.root/cp,b'opaque actual saved state')
        alias={'complete':True,'protocol_sha256':f.sha,'completed_training_updates':50,'partial_microbatches':0,
            'optimizer_inflight':False,'source_checkpoint':cp,'source_sha256':pin['sha256'],'source_bytes':pin['bytes'],
            'committed_state_digest_equal':True,'committed_state_digest':'b'*64}
        put(f.root/c.STATE_POLICY['stopped_alias_file'],encoded(alias));files=t._walk(f.root)
        self.assertEqual(t._alias_evidence(f.root,files,f.sha,50,0,False),alias)
        # A kill after writing the alias but before terminal JSON must preserve
        # the artifact while every absent terminal activity field stays unknown.
        missing=self.fixture(None);pin2=put(missing.root/cp,b'opaque actual saved state')
        unknown_alias={**alias,'protocol_sha256':missing.sha,'source_sha256':pin2['sha256'],'source_bytes':pin2['bytes']}
        put(missing.root/c.STATE_POLICY['stopped_alias_file'],encoded(unknown_alias))
        unknown=t._terminal_evidence(missing.root,t._walk(missing.root),missing.sha,missing.bundle)
        self.assertIsNotNone(unknown['stopped_checkpoint_alias']);self.assertIsNone(unknown['completed_training_updates'])
        self.assertIsNone(unknown['worker_recorded_counts']);self.assertTrue(unknown['activity_unknown_without_worker_receipt'])
        for completed,partial,inflight in ((49,0,False),(50,1,False),(50,0,True)):
            with self.assertRaises(ValueError):t._alias_evidence(f.root,files,f.sha,completed,partial,inflight)
        put(f.root/(c.CHECKPOINT_PATHS[2]+'.tmp-'+'a'*32),b'x')
        with self.assertRaises(ValueError):t._alias_evidence(f.root,t._walk(f.root),f.sha,50,0,False)

    def test_source_ledger_currentcaps_census_and_initialdisk(self):
        r=t.resource_ledger();self.assertLess(r['projected_payload_bytes'],4*1024**3)
        self.assertLess(r['single_gzip_bound_bytes'],c.LIMITS['archive_bytes']);self.assertLess(r['projected_files'],7000)
        self.assertEqual((r['neural_forwards'],r['reservation_events']),(1397,1852))
        with mock.patch.dict(c.LIMITS,{'initial_free_bytes':8*1024**3}),self.assertRaises(ValueError):t.resource_ledger()

    def test_supervisor_failedrun_stillexport_once_currentcaps(self):
        f=self.fixture();clean=f.root/'manual';clean.mkdir()
        def result(root,sha,flag,seconds,grace,log):return {'stage':flag,'exit_code':1 if flag=='--run'else 0,'worker_terminal':True}
        with mock.patch.object(c,'manual_root',return_value=clean),mock.patch.dict(s.os.environ,{'TMUX':'fixture'}),\
            mock.patch.object(s.shutil,'which',return_value='/usr/bin/timeout'),mock.patch.object(s,'idle',return_value={}),\
            mock.patch.object(s,'stage',side_effect=result)as stages,mock.patch.object(s,'final_companion',return_value={}):
            self.assertEqual(s.main(['--root',str(clean),'--protocol-sha',f.sha]),1)
        self.assertEqual([(x.args[2],x.args[3],x.args[4])for x in stages.call_args_list],
            [('--verify-transfer',300,30),('--run',3000,30),('--export',600,30)])

    def test_liveworker_and_lowreserve_refused(self):
        f=self.fixture()
        with self.environment(f),mock.patch.object(t,'_process_rows',return_value=[(123,'python -m '+t.MODULE+' --run')]),self.assertRaises(ValueError):t.export_results(f.root,f.sha)
        with self.environment(f),mock.patch.object(t.shutil,'disk_usage',return_value=SimpleNamespace(free=t.RESERVE)),self.assertRaises(ValueError):t.export_results(f.root,f.sha)

    def test_portablepaths_and_framework_absent(self):
        import sys
        for name in ('../escape','x/CON.png','a\\b','a/./b','/abs','a.'):
            with self.assertRaises(ValueError):t.canonical(name)
        self.assertFalse(any(n in sys.modules for n in ('torch','tensorflow','jax')))

    def test_actual_data_plan_metadata_joins_and_role_mutation(self):
        folder=ROOT/'outputs/codeformer_role_preserving_training_v1_preparation/data_plan_v1'
        if not folder.exists():self.skipTest('Optional real read-only data plan not present')
        receipt=c.read_json(folder/'receipt.json');bundle=c.read_json(folder/'data_bundle.json')
        p={k:receipt[k]for k in ('dependencies','retained_output_dependencies','extra_returned_dependencies','selected_train_reference_ids')}
        resolver=lambda n:Path(receipt['local_dependency_resolution'][n])
        r=t.verify_data_bundle(None,p,None,resolver=resolver,bundle=bundle)
        self.assertEqual((r['training_observations'],r['HQ_selected_references'],r['development_cases']),(200,100,406))
        changed=deepcopy(bundle);changed['training'][0]['role']='dark'
        with self.assertRaises(ValueError):t.verify_data_bundle(None,p,None,resolver=resolver,bundle=changed)

    def test_namespace_unknown_witness_and_history_refused_by_allowlist(self):
        f=self.fixture();allowed=t.allowed_output_names(f.bundle)
        self.assertIn(t.OUTPUTS+'/call_trace/01852.json',allowed)
        self.assertNotIn(t.OUTPUTS+'/call_trace/01853.json',allowed)
        self.assertNotIn(t.OUTPUTS+'/unknown.npy',allowed)
        self.assertIn(t.OUTPUTS+'/history/update_0050_committed.json',allowed)

    def test_builder_compact_allfields_under2MiB(self):
        from scripts import prepare_codeformer_role_preserving_training_v1 as builder
        plan=ROOT/'outputs/codeformer_role_preserving_training_v1_preparation/data_plan_v1/data_bundle.json'
        if not plan.exists():self.skipTest('Optional retained real data plan missing')
        value=c.read_json(plan);payload=builder.compact(value)
        self.assertLessEqual(len(payload),c.LIMITS['one_JSON_bytes'])
        self.assertEqual(json.loads(payload),value)
        self.assertTrue(payload.endswith(b'\n'));self.assertLess(len(payload),plan.stat().st_size)

    def test_real_transfer_opaque_bytes_changedparent_refusal(self):
        f=self.fixture(None);parent=f.root.parent/'parent';parent.mkdir()
        manifest_name=c.RETURN_MANIFEST['path'];rawname='outputs/historical/raw.npy'
        pin=put(parent/rawname,b'opaque parent array; never load')
        m=put(parent/manifest_name,encoded({'complete':True,'protocol_sha256':c.PARENT_PROTOCOL_SHA,'files_sha256':{rawname:pin['sha256']}}))
        old=put(parent/'protocol.json',b'{}\n');bundlepin=put(f.root/'data_bundle.json',encoded(f.bundle))
        f.p.update(data_bundle_file='data_bundle.json',dependencies={'protocol.json':old},retained_output_dependencies={rawname:pin},extra_returned_dependencies={manifest_name:m})
        f.p['files_sha256']['data_bundle.json']=bundlepin['sha256'];f.p['files_bytes']['data_bundle.json']=bundlepin['bytes']
        f.sha=put(f.root/'protocol.json',encoded(f.p))['sha256']
        with mock.patch.object(c,'verify_protocol'),mock.patch.object(c,'resolve_parent',return_value=parent),mock.patch.object(t,'verify_data_bundle',return_value={'complete':True}):
            result=t.verify_transfer(f.root,f.p,f.sha);self.assertEqual(result['dependencies_verified'],3)
            put(parent/rawname,b'changed')
            with self.assertRaisesRegex(ValueError,'Read-only parent bytes'):t.verify_transfer(f.root,f.sha)


if __name__=='__main__':unittest.main()
