"""Copy two exact retained V1 weights into a fresh R1 stage; never run models."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys

OLD_STAGE='cctv_codeformer_u96_postfusion_preflight_v1_vm'
NEW_STAGE='cctv_codeformer_u96_postfusion_preflight_v1_r1_vm'
OLD_PROTOCOL_SHA='1ebf080ba7a6bae090df7c776ad17cf145ad43f3c0e090b5b401028b655bda07'
OLD_FAILURE={'path':'outputs/u96_postfusion_preflight_v1/failure.json','bytes':1532,
             'sha256':'fa46f3327b261eb79c29764286f594ef69551e84bb92e33e4f599849fc4d8f80'}


def install(root,expected_sha):
    if sys.platform!='linux' or sys.version_info<(3,10):
        raise ValueError('Install manually on the existing Linux VM with Python3.10+')
    spec=importlib.util.spec_from_file_location('r1_inert_packet',Path(__file__).with_name('packet.py'))
    packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)
    root=packet.directory(root)
    packet.need(root==packet.directory(Path.home()/'forensic-dgp'/NEW_STAGE) and
                root==packet.directory(Path.cwd()),'Exact fresh R1 root and cwd required')
    packet.need(Path(__file__).resolve().parents[2]==root,'Installer must belong to this R1 stage')
    packet.need(packet.sha256(root/'protocol.json')==expected_sha,'Exact R1 protocol SHA required')
    protocol=packet.strict_json((root/'protocol.json').read_bytes())
    packet.need(protocol['format']==packet.FORMAT and protocol['stage_name']==NEW_STAGE and
                protocol['limits']==packet.LIMITS and protocol['optimizer_updates']==0,
                'Known zero-optimizer R1 protocol required')
    old=packet.directory(Path.home()/'forensic-dgp'/OLD_STAGE)
    packet.need(packet.sha256(old/'protocol.json')==OLD_PROTOCOL_SHA,'Retained exact V1 protocol required')
    old_protocol=packet.strict_json((old/'protocol.json').read_bytes())
    packet.need(packet.pin(old,old/OLD_FAILURE['path'])==OLD_FAILURE,
                'Retain the audited original zero-call psutil failure')
    failure=packet.strict_json((old/OLD_FAILURE['path']).read_bytes())
    packet.need(failure['error']=="No module named 'psutil'" and not failure['model_construction_attempted'] and
                all(value==0 for value in failure['counters'].values()),'Expected no-model/no-update V1 failure required')
    for name,binding in old_protocol['files'].items():
        packet.relative(name)
        actual=packet.pin(old,old/name)
        packet.need(actual['bytes']==binding['bytes'] and actual['sha256']==binding['sha256'],
                    'Retained V1 static file changed: '+name)
    weight_names={row['path'] for row in protocol['model'].values()}
    packet.need(weight_names=={'weights/codeformer_r2.pth','weights/codeformer_u96_adapter.pth'},
                'Only the two exact incumbent weight files may be copied')
    for key,binding in protocol['model'].items():
        packet.need(binding==old_protocol['model'][key] and
                    (binding['bytes'],binding['sha256'])==packet.FIXED['model'][key],
                    'Exact retained initializer binding required')
        packet.need(not (root/binding['path']).exists() and not (root/binding['path']).is_symlink(),
                    'Never overwrite even a partial weight copy')
    packet.need(not (root/'outputs').exists() and not any((root/name).exists() for name in
                ('run.log','supervisor.json','export.json')),'Fresh R1 install only; preserve prior state')
    permitted_dirs={'weights'}
    for name,binding in protocol['files'].items():
        relative=packet.relative(name)
        permitted_dirs.update(str(p) for p in relative.parents if str(p)!='.')
        if name in weight_names:continue
        actual=packet.pin(root,root/name)
        packet.need(actual['bytes']==binding['bytes'] and actual['sha256']==binding['sha256'],
                    'R1 static source/input differs before copy: '+name)
    for item in root.rglob('*'):
        name=item.relative_to(root).as_posix()
        if item.is_dir():
            packet.directory(item);packet.need(name in permitted_dirs,'Unexpected install directory')
        else:
            packet.regular(item);packet.need(name in protocol['files'] or name=='protocol.json',
                                             'Unexpected install file')
    reserve=protocol['limits']['initial_free_disk_bytes']
    needed=sum(row['bytes'] for row in protocol['model'].values())
    packet.need(shutil.disk_usage(root).free>=needed+reserve,
                'Need room for the two weight copies plus 6GiB remaining free; no files copied')
    weights=root/'weights'
    if weights.exists():packet.directory(weights)
    else:weights.mkdir()
    for key in ('r2','u96'):
        binding=protocol['model'][key];source=packet.regular(old/binding['path']);target=root/binding['path']
        before=packet.pin(old,source)
        with source.open('rb') as reader,target.open('xb') as writer:
            shutil.copyfileobj(reader,writer,length=1024**2);writer.flush();os.fsync(writer.fileno())
        packet.need(packet.pin(old,source)==before and packet.pin(root,target)==binding,
                    'Copied weight differs; retain partial evidence and stop')
    packet.verify(root,expected_sha)
    packet.need('torch' not in sys.modules and 'numpy' not in sys.modules,'Installer must remain inert')
    print('{"complete":true,"weights_copied":2,"source_V1_preserved":true,"neural_or_optimizer_calls":0}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args();install(args.root,args.protocol_sha)
