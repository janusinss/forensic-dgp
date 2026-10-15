"""Strict archive path regressions on the actual prospective importer."""
import tarfile
from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import member_names,NAME,OUT,sha,write


def member(name):return tarfile.TarInfo(NAME+'/'+name)


def main():
    assert set(member_names([member('protocol.json'),member('outputs/a.npy')]))=={'protocol.json','outputs/a.npy'}
    bad=[[member('../escape')],[member('a:stream')],[member('a\\b')],[member('a?b')],[member('CON.txt')],
         [member('a.')],[member('a ')],[member('a'),member('A')],[member('dir'),member('dir/file')],
         [member('a'),member('a')],[member('sub/../../escape')],[tarfile.TarInfo('/absolute/file')]]
    link=member('link');link.type=tarfile.SYMTYPE;link.linkname='../../escape';bad.append([link])
    link=member('hardlink');link.type=tarfile.LNKTYPE;link.linkname='protocol.json';bad.append([link])
    rejected=0
    for members in bad:
        try:member_names(members)
        except AssertionError:rejected+=1
        else:raise AssertionError('Unsafe member accepted')
    assert rejected==14
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_archive.json',dict(complete=True,checker_sha256=sha(__file__),unsafe_member_sets_rejected=rejected,local_neural_or_training_calls=0))
    print(dict(complete=True,unsafe_member_sets_rejected=rejected,neural_calls=0))


if __name__=='__main__':main()
