"""Reviewed installer experience wiring; executable fixture supplements these checks."""
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class InstallerExperience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.code=(ROOT/'packaging/installer.iss').read_text(encoding='utf8')

    def test_destination_is_visible_and_full_backend_root_is_forwarded(self):
        self.assertIn('DisableDirPage=no',self.code)
        self.assertIn(' -InstallRoot ',self.code)
        self.assertNotIn("Result := 'Use the fixed per-user application location.'",self.code)

    def test_activity_bar_is_always_visible_and_supports_measured_downloads(self):
        self.assertIn('ProgressPage.ProgressBar.Visible := True',self.code)
        self.assertIn('npbstMarquee',self.code)
        self.assertIn('npbstNormal',self.code)
        self.assertNotIn('ProgressPage.SetProgress(0, 0)',self.code)
        self.assertIn('DGP_PROGRESS ',self.code)

    def test_setup_explains_requirements_destination_and_finish(self):
        self.assertIn('Before you install',self.code)
        self.assertIn('Microsoft Edge',self.code)
        self.assertIn('runtime',self.code)
        self.assertIn('UpdateReadyMemo',self.code)
        self.assertIn('FinishedLabel',self.code)

    def test_relocation_does_not_leave_a_stale_registered_uninstaller(self):
        self.assertIn('PreviousInstallRoot',self.code)
        self.assertNotIn('uninstall the Windows launcher first',self.code)
        self.assertIn("RunState('CompleteMove')",self.code)
        self.assertIn("CurStep = ssPostInstall",self.code)
        self.assertIn('Remove verified files',self.code)
        self.assertIn('InspectInstallation',self.code)
        self.assertIn("if PreviousLauncherVerified and (PreviousInstallRoot <> '')",self.code)
        self.assertIn('RegisteredInstallRoot',self.code)

    def test_trusted_location_preflight_precedes_execution(self):
        wrapper=(ROOT/'packaging/gui_setup.ps1').read_text(encoding='utf8')
        self.assertIn('install_location.ps1',wrapper)
        self.assertLess(wrapper.index('Hash-File'),wrapper.index('-Action Prepare'))
        self.assertIn('install-location.json',wrapper)
        self.assertIn('$env:DGP_INSTALL_ROOT',wrapper)


if __name__=='__main__':
    unittest.main()
