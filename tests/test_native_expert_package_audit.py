import unittest


class ArchiveMemberContracts(unittest.TestCase):
    def test_rejects_traversal_duplicate_members_and_links(self):
        from scripts.audit_native_expert_package import validate_members
        import tarfile
        member = tarfile.TarInfo('native_expert_vm_bundle/inputs/one.png')
        self.assertEqual(validate_members([member]), {'inputs/one.png': member})
        for names in (['native_expert_vm_bundle/../outside'], ['/absolute'],
                      ['native_expert_vm_bundle/C:/outside'], ['native_expert_vm_bundle/inputs\\one.png']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_members([tarfile.TarInfo(name) for name in names])
        with self.assertRaises(ValueError): validate_members([member, member])
        link = tarfile.TarInfo('native_expert_vm_bundle/link'); link.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError): validate_members([link])


if __name__ == '__main__': unittest.main()
