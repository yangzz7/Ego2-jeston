"""Automatic entry creation must preserve user files and use local calibration."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml

from ego2.common.camera import load_camera
from ego2.common.paths import ROOT
from ego2.multi_vio.config import MULTI_CONFIG, load_multi_config
from ego2.multi_vio.profiles import ensure_camera_profile, prepare_camera_profiles
from ego2.multi_vio.selection import Device, scan_devices


class AutoProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patcher = patch('ego2.common.camera.CAMERAS', self.root)
        self.patcher.start()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.patcher.stop)
        self.sn = 's299999'
        self.folder = self.seed(self.sn)

    def seed(self, sn):
        folder = self.root/sn
        folder.mkdir(exist_ok=True)
        for name in ('imu.yaml','camchain.yaml','camchain-imucam.yaml'):
            shutil.copy2(ROOT/'cameras/s200097'/name, folder/name)
        return folder

    def test_generated_entry_loads_in_real_multi_vio_and_preserves_calibration(self):
        original = {p.name:p.read_bytes() for p in self.folder.iterdir()}
        with redirect_stdout(StringIO()) as output:
            path,created = ensure_camera_profile(self.sn)
        self.assertTrue(created)
        self.assertIn(str(path), output.getvalue())
        profile = load_camera(sn=self.sn)
        self.assertEqual(profile['input']['serial'], self.sn)
        self.assertEqual(profile['calibration'], str(self.folder/'camchain-imucam.yaml'))
        self.assertEqual(profile['imu_calibration'], str(self.folder/'imu.yaml'))
        cfg,devices = load_multi_config(selection={'chest':self.sn})
        self.assertEqual(devices['chest']['native']['serial'], self.sn)
        self.assertEqual(cfg['viewer']['row_order'], ['chest'])
        for name,content in original.items():
            self.assertEqual((self.folder/name).read_bytes(), content)
        self.assertEqual(list(self.folder.glob('.camera-*')), [])
        before = path.read_bytes(), path.stat().st_mtime_ns
        self.assertFalse(ensure_camera_profile(self.sn)[1])
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_existing_even_custom_or_invalid_entry_is_never_replaced(self):
        target = self.folder/'camera.yaml'
        for text in ('# custom\nserial: s299999\ninput: {fps: 10}\n','{malformed'):
            target.write_text(text)
            self.assertFalse(ensure_camera_profile(self.sn)[1])
            self.assertEqual(target.read_text(),text)

    def test_each_missing_file_reports_reason_without_publishing(self):
        for name in ('imu.yaml','camchain.yaml','camchain-imucam.yaml'):
            with self.subTest(name=name):
                self.seed(self.sn)
                (self.folder/name).unlink()
                with self.assertRaisesRegex(ValueError,name): ensure_camera_profile(self.sn)
                self.assertFalse((self.folder/'camera.yaml').exists())
                self.assertEqual(list(self.folder.glob('.camera-*')), [])

    def test_invalid_yaml_geometry_and_imu_do_not_generate_entry(self):
        cases = [
            ('imu.yaml',lambda data: data['imu0'].update(update_rate=float('nan'))),
            ('imu.yaml',lambda data: data['imu0'].update(gyroscope_noise_density=-1)),
            ('camchain.yaml',lambda data: data['cam0'].update(resolution=[640,480])),
            ('camchain-imucam.yaml',lambda data: data['cam1'].pop('T_cam_imu')),
            ('camchain-imucam.yaml',lambda data: data['cam1'].update(timeshift_cam_imu=float('inf'))),
            ('camchain-imucam.yaml',lambda data: data['cam0'].update(distortion_model='radtan')),
        ]
        for name,mutate in cases:
            with self.subTest(name=name,mutate=mutate):
                self.seed(self.sn)
                path=self.folder/name; data=yaml.safe_load(path.read_text()); mutate(data)
                path.write_text(yaml.safe_dump(data))
                with self.assertRaisesRegex(ValueError,'标定参数无效'): ensure_camera_profile(self.sn)
                self.assertFalse((self.folder/'camera.yaml').exists())
        self.seed(self.sn)
        (self.folder/'imu.yaml').write_text('{broken')
        with self.assertRaisesRegex(ValueError,'标定参数无效'): ensure_camera_profile(self.sn)
        self.assertFalse((self.folder/'camera.yaml').exists())

    def test_legacy_imu_filename_supported_but_primary_name_has_precedence(self):
        (self.folder/'imu.yaml').rename(self.folder/'imu-calibrated.yaml')
        path,_=ensure_camera_profile(self.sn)
        self.assertEqual(yaml.safe_load(path.read_text())['imu_calibration'],'imu-calibrated.yaml')
        path.unlink()
        self.seed(self.sn)
        path,_=ensure_camera_profile(self.sn)
        self.assertEqual(yaml.safe_load(path.read_text())['imu_calibration'],'imu.yaml')

    def test_startup_prepares_disconnected_units_without_blocking_on_bad_folder(self):
        bad=self.seed('s299998'); (bad/'imu.yaml').unlink()
        created,problems=prepare_camera_profiles()
        self.assertEqual(created,[self.folder/'camera.yaml'])
        self.assertEqual(set(problems),{'s299998'})
        self.assertIn('imu.yaml',problems['s299998'])
        self.assertFalse((bad/'camera.yaml').exists())

    def test_panel_refresh_retries_after_missing_files_are_supplied(self):
        (self.folder/'imu.yaml').unlink()
        with patch('ego2.multi_vio.selection.discover_devices',return_value=[Device(self.sn,'/dev/video42','/usb/test')]):
            first=scan_devices(MULTI_CONFIG)
            self.assertFalse(first[0].ready)
            self.assertIn('imu.yaml',first[0].problem)
            self.seed(self.sn)
            second=scan_devices(MULTI_CONFIG)
            self.assertTrue(second[0].ready,second[0].problem)
            self.assertTrue((self.folder/'camera.yaml').is_file())

    def test_simultaneous_creation_never_overwrites_another_entry(self):
        target=self.folder/'camera.yaml'
        original_link=os.link
        def competing_write(source,destination):
            target.write_text('# created by another launcher\n')
            original_link(source,destination)
        with patch('ego2.multi_vio.profiles.os.link',side_effect=competing_write):
            self.assertFalse(ensure_camera_profile(self.sn)[1])
        self.assertEqual(target.read_text(),'# created by another launcher\n')
        self.assertEqual(list(self.folder.glob('.camera-*')),[])

    def test_cli_defers_profiles_until_capture_and_prepares_noninteractive(self):
        from ego2.multi_vio.__main__ import main
        target=self.folder/'camera.yaml'
        def cancel_panel(config, *, mode=None):
            self.assertFalse(target.exists())  # Playback/cancel must not depend on calibration.
            return None
        with patch('ego2.multi_vio.selection.open_launcher',side_effect=cancel_panel), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(['--no-record']),0)
        with patch('ego2.multi_vio.selection.default_selection',return_value={'chest':self.sn}), \
             patch('ego2.multi_vio.app.run') as run, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(main(['--no-setup','--no-record']),0)
            self.assertEqual(run.call_args.args[1]['chest']['native']['serial'],self.sn)
        target.unlink()
        with patch('ego2.multi_vio.__main__.load_multi_config',return_value=({},{})), \
             patch('ego2.multi_vio.app.run'), redirect_stdout(StringIO()):
            self.assertEqual(main(['--demo']),0)
        self.assertFalse(target.exists())


if __name__=='__main__': unittest.main()
