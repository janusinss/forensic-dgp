"""Strict archive path regressions using real tar-member metadata."""
import tarfile
from import_cctv_dgp_recognizer_precision_v1_return import member_names, NAME


def member(name):
    return tarfile.TarInfo(NAME + '/' + name)


def main():
    assert set(member_names([member('protocol.json'), member('outputs/a.npy')])) == {'protocol.json', 'outputs/a.npy'}
    bad = [[member('../escape')], [member('a:stream')], [member('a\\b')],
           [member('a?b')], [member('CON.txt')], [member('a.')], [member('a ')],
           [member('a'), member('A')], [member('dir'), member('dir/file')],
           [member('a'), member('a')], [member('sub/../../escape')], [tarfile.TarInfo('/absolute/file')]]
    symlink = member('link'); symlink.type = tarfile.SYMTYPE; symlink.linkname = '../../escape'; bad.append([symlink])
    hardlink = member('hardlink'); hardlink.type = tarfile.LNKTYPE; hardlink.linkname = 'protocol.json'; bad.append([hardlink])
    rejected = 0
    for members in bad:
        try:
            member_names(members)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Unsafe tar member accepted')
    assert rejected == 14
    print(dict(complete=True, unsafe_member_sets_rejected=rejected, neural_calls=0))


if __name__ == '__main__':
    main()
