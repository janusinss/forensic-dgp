"""Reject unsafe tar namespaces without extracting or executing anything."""
import argparse
import json
from pathlib import Path
import tarfile

from import_cctv_dgp_conditioning_response_v1_return import NAME, member_names


def item(name,kind=tarfile.REGTYPE):
    value=tarfile.TarInfo(name);value.type=kind;value.size=1;return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True,type=Path);a=parser.parse_args()
    record=dict(complete=False,neural_or_training_calls=0)
    try:
        good=member_names([item(NAME+'/protocol.json'),item(NAME+'/outputs/state.npy')]);assert len(good)==2
        cases={
            'parent_traversal':[item(NAME+'/../outside')],
            'absolute':[item('/'+NAME+'/file')],
            'wrong_root':[item('other/file')],
            'windows_drive':[item(NAME+'/C:/file')],
            'backslash':[item(NAME+'/folder\\file')],
            'reserved_device':[item(NAME+'/CON.json')],
            'trailing_space':[item(NAME+'/file ')],
            'noncanonical':[item(NAME+'/folder//file')],
            'windows_question_mark':[item(NAME+'/file?.json')],
            'control_character':[item(NAME+'/file\x01.json')],
            'case_duplicate':[item(NAME+'/FILE'),item(NAME+'/file')],
            'file_directory_collision':[item(NAME+'/folder'),item(NAME+'/folder/file')],
            'symlink':[item(NAME+'/link',tarfile.SYMTYPE)],
            'hardlink':[item(NAME+'/link',tarfile.LNKTYPE)],
            'directory':[item(NAME+'/folder',tarfile.DIRTYPE)],
        }
        rejected=[]
        for name,members in cases.items():
            try:member_names(members)
            except AssertionError:rejected.append(name)
            else:raise AssertionError('Unsafe namespace accepted: '+name)
        record.update(complete=True,valid_files_verified=2,unsafe_members_rejected=rejected,extracted_or_executed=False)
    except Exception as error:
        record['failure']=str(error)
        with a.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2)+'\n')
        raise
    with a.receipt.open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(dict(complete=True,unsafe_namespaces_rejected=len(rejected),extracted_or_executed=False))


if __name__=='__main__':main()
