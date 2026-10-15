"""Unsafe return paths must be rejected before any archive extraction."""
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert (ROOT/'scripts/import_cctv_dgp_appearance_input_response_v1_return.py').is_file(),'Safe input-response return importer is not implemented'
    from import_cctv_dgp_appearance_input_response_v1_return import NAME,member_names
    def member(name,kind=tarfile.REGTYPE):
        m=tarfile.TarInfo(name);m.type=kind;m.size=1;return m
    assert set(member_names([member(NAME+'/protocol.json'),member(NAME+'/outputs/repeat00/b036.npy')]))=={'protocol.json','outputs/repeat00/b036.npy'}
    invalid=[[member(n)] for n in ['/abs','../escape',NAME+'/../escape',NAME+'/x\\y',NAME+'/x:y',NAME+'/CON.json',
        NAME+'/x.',NAME+'/x ',NAME+'/a?b',NAME+'/a*b',NAME+'/a\nfile',NAME+'//x']]
    invalid+=[[member(NAME+'/a',tarfile.SYMTYPE)],[member(NAME+'/a',tarfile.LNKTYPE)],
              [member(NAME+'/a'),member(NAME+'/A')],[member(NAME+'/a'),member(NAME+'/a/b')]]
    for members in invalid:
        try:member_names(members)
        except AssertionError:pass
        else:raise AssertionError('Unsafe return member accepted')
    print(dict(complete=True,valid_members=2,unsafe_sets_rejected=len(invalid),archives_extracted=0,neural_or_training_calls=0))


if __name__=='__main__':main()
