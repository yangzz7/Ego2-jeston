"""USB enumeration, launcher lifecycle, and selected-device-only alignment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from ego2.multi_vio.selection import (Device, discover_devices, scan_devices,
    normalize_selection, validate_connected, preferences_path, read_preferences,
    write_preferences, default_selection, qt_environment)
from ego2.multi_vio.config import MULTI_CONFIG, MULTI_ROLES, load_multi_config
from ego2.multi_vio.alignment import Coordinator, inverse
from ego2.multi_vio.visualizer import build_blueprint
from tests.test_multi_vio_core import measured, transform


class SelectionTests(unittest.TestCase):
    def fake_usb(self, root, parent_name, sn, node_indexes, *, vid='04b4'):
        parent=root/parent_name; parent.mkdir()
        for key,value in [('idVendor',vid),('idProduct','00f9'),('serial',sn)]:
            (parent/key).write_text(value)
        interface=parent/'interface'; interface.mkdir()
        for name,index in node_indexes:
            node=root/'video4linux'/name; node.mkdir(parents=True)
            (node/'index').write_text(str(index)); (node/'device').symlink_to(interface)

    def test_physical_discovery_deduplicates_metadata_and_ignores_webcam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fake_usb(root,'usb1','s200100',[('video0',0),('video1',1)])
            self.fake_usb(root,'usb2','s200032',[('video4',0),('video5',1)])
            self.fake_usb(root,'webcam','other',[('video6',0)],vid='1234')
            devices=discover_devices(root/'video4linux')
            self.assertEqual([d.sn for d in devices],['s200032','s200100'])
            self.assertTrue(all(d.ready for d in devices))
            self.assertEqual(devices[0].node,'/dev/video4')

    def test_ambiguous_and_missing_serials_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.fake_usb(root,'usb1','s200100',[('video0',0)])
            self.fake_usb(root,'usb2','S200100',[('video2',0)])
            self.fake_usb(root,'usb3','',[('video3',0)])
            devices=discover_devices(root/'video4linux')
            self.assertEqual(len(devices),3)
            self.assertTrue(all(not d.ready for d in devices))
            with self.assertRaises(ValueError): validate_connected({'chest':'s200100'},devices)
        with self.assertRaisesRegex(ValueError,'同一 SN'):
            normalize_selection({'chest':'s200100','head':'S200100'})

    def test_profile_checks_both_optics_and_imu_and_reports_missing_bundle(self):
        devices=[Device(sn,'/dev/video0','/usb/'+sn) for sn in ('s200100','s200032','s200097','missing_test_sn')]
        with patch('ego2.multi_vio.selection.discover_devices',return_value=devices):
            checked=scan_devices(MULTI_CONFIG)
            self.assertTrue(all(d.ready for d in checked[:3]),checked)
            self.assertIn('camera.yaml',checked[-1].problem)
            with patch('ego2.reconstruction.calibration.write_orb_settings',side_effect=ValueError('invalid IMU')):
                failed=scan_devices(MULTI_CONFIG)
                self.assertTrue(all('invalid IMU' in d.problem for d in failed[:3]))

    def test_preferences_roundtrip_validation_and_saved_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'my-config.yaml'
            config.write_text('devices:\n  chest: {sn: s200100}\n  head: {sn: null}\n')
            path=preferences_path(config)
            self.assertIsNone(read_preferences(path))
            self.assertEqual(default_selection(config),{'chest':'s200100'})
            choice={'chest':'s200032','left_hand':'s200100'}
            write_preferences(path,choice)
            self.assertEqual(read_preferences(path),choice)
            self.assertEqual(default_selection(config),choice)
            original=path.read_bytes()
            with self.assertRaises(ValueError): write_preferences(path,{'head':'s200002','left_hand':'s200100'})
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(tmp).glob('.*')),[])
            path.write_text('{broken')
            with self.assertRaisesRegex(ValueError,'无法读取'): read_preferences(path)

    def test_qt_environment_only_removes_opencv_owned_settings(self):
        with patch.dict(os.environ,{'QT_QPA_PLATFORM_PLUGIN_PATH':'/site-packages/cv2/qt/plugins',
                                   'QT_QPA_FONTDIR':'/user/fonts','DISPLAY':':1'}):
            env=qt_environment()
            self.assertNotIn('QT_QPA_PLATFORM_PLUGIN_PATH',env)
            self.assertEqual(env['QT_QPA_FONTDIR'],'/user/fonts')
            self.assertEqual(env['DISPLAY'],':1')

    def test_fresh_gui_process_can_load_opencv_qt_and_discover_then_cancel(self):
        # A pre-imported cv2 in the main test process can hide a GLib/Qt load
        # order regression, so exercise the real launcher module in a new one.
        code='''
from ego2.multi_vio.launcher import PreparationDialog
from ego2.multi_vio.config import MULTI_CONFIG
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
app=QApplication([])
d=PreparationDialog(MULTI_CONFIG, initial_selection={})
def finish():
    if d.first_scan or d.worker is not None:
        QTimer.singleShot(30,finish)
        return
    print('SCAN_OK',len(d.devices),flush=True)
    d.reject()
QTimer.singleShot(50,finish)
QTimer.singleShot(5000,d.reject)
assert d.exec()==0 and d.selection is None
'''
        env=qt_environment(); env['QT_QPA_PLATFORM']='offscreen'
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('SCAN_OK',result.stdout)

    def test_cli_cancel_never_loads_or_starts_capture(self):
        from ego2.multi_vio.__main__ import main
        with patch('ego2.multi_vio.selection.open_launcher',return_value=None) as panel, \
             patch('ego2.multi_vio.__main__.load_multi_config') as config, \
             patch('ego2.multi_vio.app.run') as run:
            self.assertEqual(main(['--no-record']),0)
            panel.assert_called_once(); config.assert_not_called(); run.assert_not_called()

    def test_cli_accepts_exact_selection_and_noninteractive_skips_panel(self):
        from ego2.multi_vio.__main__ import main
        selection={'chest':'s200032','head':'s200100'}
        with patch('ego2.multi_vio.selection.open_launcher',return_value=selection) as panel, \
             patch('ego2.multi_vio.app.run') as run:
            self.assertEqual(main(['--no-record']),0)
            cfg,devices,args=run.call_args.args
            self.assertEqual({r:d['native']['serial'] for r,d in devices.items()},selection)
            self.assertEqual(cfg['viewer']['row_order'],['head','chest'])
            panel.assert_called_once()
        for arguments in (['check'],['--no-setup'],['--no-viz'],['--headless','--record']):
            with self.subTest(arguments=arguments), \
                 patch('ego2.multi_vio.selection.open_launcher') as panel, \
                 patch('ego2.multi_vio.selection.default_selection',return_value=selection), \
                 patch('ego2.multi_vio.selection.default_mode',return_value='vio'), \
                 patch('ego2.multi_vio.transport.check_devices'), patch('ego2.multi_vio.app.run'):
                self.assertEqual(main(arguments),0)
                panel.assert_not_called()


class VariableCameraTests(unittest.TestCase):
    def test_one_through_four_align_only_selected_roles_and_build_exact_rows(self):
        for count in range(1,5):
            with self.subTest(count=count):
                roles=MULTI_ROLES[:count]
                cfg,devices=load_multi_config(demo=True,selection={r:None for r in roles})
                cfg['alignment'].update(min_samples=5,min_duration_s=.3)
                coordinator=Coordinator(cfg,devices)
                poses={r:transform((.1*i,.04,-.1),(.2*i,0,.6)) for i,r in enumerate(roles)}
                anchors={r:transform((.2,-.3*i,.1),(1+i,.3,.2)) for i,r in enumerate(roles)}
                for index in range(6):
                    for role in roles:
                        packet=dict(frame_id=index,timestamp=index*.1,capture_time=index*.1,map_id='0',
                            initialization_complete=True,valid=True,
                            T_world_camera=(inverse(anchors[role]) @ poses[role]).tolist())
                        coordinator.ingest(role,packet,index*.1)
                        coordinator.observe_board(role,packet,measured(poses[role]),index*.1)
                    coordinator.relative(index*.1)
                self.assertTrue(coordinator.started)
                self.assertEqual(len(coordinator.initial_deltas),count-1)
                for role in roles:
                    np.testing.assert_allclose(coordinator.pose(role,.5),coordinator.shared_from_board @ poses[role],atol=1e-9)
                blueprint=build_blueprint(cfg)
                left,right=blueprint.root_container.contents
                self.assertEqual(len(right.contents),count)
                self.assertTrue(all(len(row.contents)==2 for row in right.contents))
                self.assertEqual(cfg['viewer']['initial_pose_panel'],count>1)
                with tempfile.TemporaryDirectory() as tmp:
                    blueprint.save('variable-camera-test',str(Path(tmp)/'layout.rbl'))


class DialogTests(unittest.TestCase):
    def test_legacy_attitude_preference_keeps_camera_selection_and_saves_current_options(self):
        from PySide6.QtWidgets import QCheckBox
        from ego2.multi_vio.selection import read_setup
        self.pref.write_text(json.dumps(dict(version=2, mode='gesture',
            devices={'head':'s200032'}, imu_attitude=True)))
        setup = read_setup(self.pref)
        d = self.dialog(initial_selection=setup['devices'], mode=setup['mode'])
        d.apply_devices(self.devices)
        self.assertEqual(d.current_selection(), {'head':'s200032'})
        self.assertFalse(any('IMU' in item.text() for item in d.findChildren(QCheckBox)))
        self.assertTrue(d.remember.isChecked())
        d.start_button.click()
        self.wait_for(lambda:d.selection is not None)
        self.assertEqual(json.loads(self.pref.read_text()),
                         dict(version=2, mode='gesture', devices={'head':'s200032'}))

    @classmethod
    def setUpClass(cls):
        os.environ['QT_QPA_PLATFORM']='offscreen'
        for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):
            if '/cv2/' in os.environ.get(key,''): os.environ.pop(key)
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont
        cls.app=QApplication.instance() or QApplication([])
        cls.app.setStyle('Fusion'); cls.app.setFont(QFont('Noto Sans CJK SC',11))

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.pref=Path(self.temporary.name)/'selection.json'
        self.devices=[Device(sn,f'/dev/video{i*2}',f'/usb/{i}') for i,sn in enumerate(('s200100','s200002','s200032'))]
        self.dialogs=[]

    def tearDown(self):
        for d in self.dialogs:
            d.reject()
            self.wait_for(lambda: d.worker is None)
            d.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def dialog(self, **kwargs):
        from ego2.multi_vio.launcher import PreparationDialog
        d=PreparationDialog(MULTI_CONFIG,scanner=kwargs.pop('scanner',lambda:self.devices),
            preference_file=self.pref,initial_selection=kwargs.pop('initial_selection',{}),
            mode=kwargs.pop('mode','vio'),auto_scan=False,**kwargs)
        self.dialogs.append(d); d.show(); self.app.processEvents()
        return d

    def wait_for(self, predicate):
        from PySide6.QtTest import QTest
        deadline=time.monotonic()+5
        while not predicate() and time.monotonic()<deadline:
            self.app.processEvents(); QTest.qWait(10)
        self.assertTrue(predicate(),'GUI operation did not finish')

    def choose(self,d,role,sn):
        combo=d.combos[role]
        combo.setCurrentIndex(combo.findData(sn))
        self.app.processEvents()

    def test_restore_only_available_serials_no_guessed_body_and_unique_choices(self):
        d=self.dialog(initial_selection={'chest':'s200100','head':'absent'})
        d.apply_devices(self.devices)
        self.assertEqual(d.current_selection(),{'chest':'s200100'})
        self.assertTrue(d.start_button.isEnabled())
        head=d.combos['head']
        self.assertFalse(head.model().item(head.findData('s200100')).isEnabled())
        self.choose(d,'left_hand','s200032')
        d.apply_devices(list(reversed(self.devices)))
        self.assertEqual(d.current_selection(),{'chest':'s200100','left_hand':'s200032'})
        self.assertEqual(d.start_button.text(),'启动 2 台相机')

    def test_unplug_does_not_silently_disable_selected_role_and_replug_recovers(self):
        d=self.dialog(initial_selection={'chest':'s200100','head':'s200002'})
        d.apply_devices(self.devices)
        d.apply_devices([self.devices[0]])
        self.assertEqual(d.current_selection()['head'],'s200002')
        self.assertIn('已断开',d.combos['head'].currentText())
        self.assertFalse(d.start_button.isEnabled())
        d.apply_devices(self.devices)
        self.assertTrue(d.start_button.isEnabled())

    def test_missing_profile_and_multiple_cameras_without_chest_disable_start(self):
        d=self.dialog()
        d.apply_devices([Device('s200098','/dev/video0','/usb/0',problem='缺少 camera.yaml')]+self.devices)
        chest=d.combos['chest']
        self.assertFalse(chest.model().item(chest.findData('s200098')).isEnabled())
        self.assertFalse(d.start_button.isEnabled())
        self.assertIn('至少选择一台',d.validation.text())
        self.choose(d,'head','s200002')
        self.assertTrue(d.start_button.isEnabled())
        self.choose(d,'left_hand','s200032')
        self.assertFalse(d.start_button.isEnabled())
        self.assertIn('未选择胸部时，仅支持单相机',d.validation.text())
        self.assertEqual(d.current_selection(),{'head':'s200002','left_hand':'s200032'})
        self.choose(d,'chest','s200100')
        self.assertTrue(d.start_button.isEnabled())
        self.choose(d,'chest',None)
        self.assertFalse(d.start_button.isEnabled())
        self.choose(d,'left_hand',None)
        self.assertTrue(d.start_button.isEnabled())

    def test_each_non_chest_role_can_start_and_restore_as_single_camera(self):
        for role in ('head','left_hand','right_hand'):
            with self.subTest(role=role):
                d=self.dialog(initial_selection={role:'s200032'})
                d.apply_devices(self.devices)
                self.assertEqual(d.current_selection(),{role:'s200032'})
                self.assertTrue(d.start_button.isEnabled())
                self.assertEqual(d.start_button.text(),'启动 1 台相机')
                d.start_button.click()
                self.wait_for(lambda:d.selection is not None)
                self.assertEqual(d.selection,{role:'s200032'})
                self.assertEqual(read_preferences(self.pref),d.selection)

    def test_start_rechecks_and_remembers_only_active_roles(self):
        d=self.dialog(); d.apply_devices(self.devices)
        self.choose(d,'chest','s200100'); self.choose(d,'left_hand','s200032')
        d.start_button.click()
        self.wait_for(lambda:d.selection is not None)
        self.assertEqual(d.selection,{'chest':'s200100','left_hand':'s200032'})
        self.assertEqual(read_preferences(self.pref),d.selection)

    def test_mode_switch_updates_rules_preserves_choices_and_saves_mode(self):
        from ego2.multi_vio.selection import read_setup
        for mode in ('gesture','aprilgrid'):
            with self.subTest(mode=mode):
                d=self.dialog(initial_selection={'head':'s200032','left_hand':'s200100'})
                d.apply_devices(self.devices)
                self.assertFalse(d.start_button.isEnabled())
                d.mode_combo.setCurrentIndex(d.mode_combo.findData(mode))
                self.wait_for(lambda:d.worker is None)
                self.assertTrue(d.start_button.isEnabled())
                self.assertNotIn('参考',d.part_labels['chest'].text())
                self.assertIn('无需选择胸部',d.policy.text())
                d.mode_combo.setCurrentIndex(d.mode_combo.findData('vio'))
                self.wait_for(lambda:d.worker is None)
                self.assertFalse(d.start_button.isEnabled())
                self.assertEqual(d.current_selection(),{'head':'s200032','left_hand':'s200100'})
                d.mode_combo.setCurrentIndex(d.mode_combo.findData(mode))
                self.wait_for(lambda:d.worker is None)
                d.start_button.click()
                self.wait_for(lambda:d.selection is not None)
                self.assertEqual(read_setup(self.pref)['mode'],mode)

    def test_reconstruction_requires_exactly_one_any_role(self):
        d=self.dialog(initial_selection={'head':'s200032','left_hand':'s200100'},mode='reconstruction')
        d.apply_devices(self.devices)
        self.assertFalse(d.start_button.isEnabled())
        self.assertIn('仅支持一台',d.validation.text())
        self.choose(d,'head',None)
        self.assertTrue(d.start_button.isEnabled())
        self.assertIn('3D 点云',d.hint.text())
        d.start_button.click()
        self.wait_for(lambda:d.selection is not None)
        self.assertEqual(d.selection,{'left_hand':'s200100'})

    def test_final_scan_detects_unplug_without_saving_or_accepting(self):
        d=self.dialog(scanner=lambda:[]); d.apply_devices(self.devices)
        self.choose(d,'chest','s200100'); d.start_button.click()
        self.wait_for(lambda:d.worker is None)
        self.assertIsNone(d.selection); self.assertFalse(self.pref.exists())
        self.assertFalse(d.start_button.isEnabled())

    def test_cancel_during_scan_preserves_preferences_and_does_not_accept(self):
        old={'chest':'s200032'}; write_preferences(self.pref,old)
        gate=threading.Event()
        d=self.dialog(scanner=lambda:(gate.wait(3),self.devices)[1])
        d.apply_devices(self.devices); self.choose(d,'chest','s200100')
        d.start_button.click(); d.reject(); gate.set()
        self.wait_for(lambda:d.worker is None)
        self.assertIsNone(d.selection); self.assertEqual(read_preferences(self.pref),old)
        self.assertFalse(d.isVisible())

    def test_unchecking_remember_removes_old_mapping_only_on_acceptance(self):
        write_preferences(self.pref,{'chest':'s200032'})
        d=self.dialog(); d.apply_devices(self.devices); self.choose(d,'chest','s200100')
        d.remember.setChecked(False); d.reject()
        self.assertTrue(self.pref.exists())
        d=self.dialog(); d.apply_devices(self.devices); self.choose(d,'chest','s200100')
        d.remember.setChecked(False); d.start_button.click()
        self.wait_for(lambda:d.selection is not None)
        self.assertFalse(self.pref.exists())


if __name__=='__main__': unittest.main()
