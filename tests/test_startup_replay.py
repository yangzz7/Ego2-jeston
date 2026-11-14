"""Offline startup routing, session inspection, and actual async panel behavior."""
from concurrent.futures import Future
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, PropertyMock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from ego2.multi_vio.startup import StartupDialog, QApplication
from ego2.multi_vio.config import MULTI_CONFIG
from ego2.multi_vio.session_catalog import inspect_session, list_sessions, duration_text
from ego2.multi_vio.storage import status, write_json, validate_rrd
from ego2.multi_vio.__main__ import main


def make_rrd(root, mode='aprilgrid'):
    import numpy as np
    import rerun as rr
    root.mkdir(parents=True, exist_ok=True)
    record = rr.RecordingStream('startup-test')
    record.save(str(root/'replay.rrd'))
    record.set_time('capture_time', duration=0.)
    record.log('/cameras/head/cam0/image', rr.Image(np.zeros((16,16,3),dtype=np.uint8)))
    record.flush(); record.disconnect()
    status(root,'complete',mode=mode,devices={'head':'test-camera'},
           record_start_monotonic=10.,record_stop_monotonic=12.5)
    return root


def make_lerobot(root, mode='gesture'):
    from ego2.multi_vio.dataset import assemble
    from ego2.common.config import read_yaml
    from tests.test_unified_storage import frame
    root.mkdir(parents=True, exist_ok=True)
    cfg=read_yaml(MULTI_CONFIG)
    cfg.update(mode=mode,devices={'head':{'sn':'offline-test'}},reference_role='head')
    cfg['viewer'].update(row_order=['head'],initial_pose_panel=False)
    assemble(root,mode,{'head':iter([frame(0.,0),frame(.1,1)])},10,
             dict(config=cfg,devices={'head':'offline-test'},start_time=None))
    status(root,'complete',mode=mode,format='lerobot-v3')
    return root


class CatalogTests(unittest.TestCase):
    def test_four_formats_sort_by_metadata_and_use_recorded_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a=make_rrd(root/'aprilgrid/custom-name')
            b=make_lerobot(root/'gesture/older-name')
            c=make_rrd(root/'reconstruction/map','reconstruction')
            d=make_lerobot(root/'trajectory/trace','vio')
            status(a,'complete',created_at='2026-01-01T00:00:00+08:00')
            status(b,'complete',created_at='2026-02-01T00:00:00+08:00')
            status(c,'complete',created_at='2026-03-01T00:00:00+08:00')
            status(d,'complete',created_at='2026-04-01T00:00:00+08:00')
            entries=list_sessions(root)
            self.assertEqual([s.path for s in entries],[d,c,b,a])
            self.assertTrue(all(s.playable for s in entries),entries)
            self.assertEqual(entries[2].devices,{'head':'offline-test'})
            self.assertAlmostEqual(entries[2].duration,.2)
            self.assertEqual(entries[3].duration,2.5)

    def test_missing_unfinished_and_corrupt_sessions_are_not_playable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertFalse(inspect_session(root).playable)
            make_rrd(root)
            status(root,'recording')
            self.assertIn('正在录制',inspect_session(root).error)
            status(root,'complete')
            (root/'replay.rrd').unlink()
            self.assertIn('replay.rrd',inspect_session(root).error)
            (root/'session.json').write_text('{bad')
            self.assertIn('元数据',inspect_session(root).error)
            write_json(root/'session.json',{'mode':[], 'status':[]})
            self.assertFalse(inspect_session(root).playable)
            self.assertEqual(inspect_session(root).title,'未知模式')

    def test_unknown_duration_not_derived_from_directory_age(self):
        with tempfile.TemporaryDirectory() as directory:
            root=make_rrd(Path(directory))
            write_json(root/'session.json',dict(mode='aprilgrid',status='complete'))
            self.assertEqual(duration_text(inspect_session(root).duration),'未记录')
            self.assertEqual(duration_text(3723),'01:02:03')

    def test_gui_replay_routing_never_prepares_cameras_or_starts_capture(self):
        for flags in ([],['--record']):
            with self.subTest(flags=flags), \
                 patch('ego2.multi_vio.selection.open_launcher',return_value={'action':'replay','exit_code':0}), \
                 patch('ego2.multi_vio.profiles.prepare_camera_profiles',side_effect=AssertionError('camera preparation')), \
                 patch('ego2.multi_vio.__main__.load_multi_config',side_effect=AssertionError('live config')):
                self.assertEqual(main(flags),0)

    def test_launcher_decodes_replay_without_camera_selection_validation(self):
        from ego2.multi_vio.selection import open_launcher
        def run(command, **kwargs):
            self.assertIn('ego2.multi_vio.startup',command)
            write_json(Path(command[command.index('--result')+1]),dict(action='replay',exit_code=0))
            return subprocess.CompletedProcess(command,0)
        with patch('ego2.multi_vio.selection.subprocess.run',side_effect=run), \
             patch('ego2.multi_vio.selection.normalize_selection',side_effect=AssertionError):
            self.assertEqual(open_launcher('/missing/config'),dict(action='replay',exit_code=0))

    def test_worker_replays_all_four_modes_without_modifying_sources(self):
        from ego2.multi_vio.replay_worker import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for mode in ('gesture','vio','aprilgrid','reconstruction'):
                session=(make_lerobot if mode in ('gesture','vio') else make_rrd)(root/mode,mode)
                before={str(p.relative_to(session)):p.read_bytes() for p in session.rglob('*') if p.is_file()}
                workspace=root/(mode+'-cache');workspace.mkdir()
                process=Mock(returncode=0);process.poll.return_value=0;process.wait.return_value=0
                with patch('ego2.multi_vio.replay_worker.subprocess.Popen',return_value=process) as popen:
                    self.assertEqual(run(session,workspace),0)
                rrd=Path(popen.call_args.args[0][1])
                self.assertIn('--new',popen.call_args.args[0])
                validate_rrd(rrd)
                self.assertEqual(json.loads((workspace/'state.json').read_text())['phase'],'playing')
                self.assertEqual(before,{str(p.relative_to(session)):p.read_bytes() for p in session.rglob('*') if p.is_file()})


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.dialogs=[]

    def tearDown(self):
        for d in self.dialogs:
            d.replay.shutdown()
            d.reject()
            self.until(lambda:d.worker is None)
            d.deleteLater()
        self.app.processEvents()
        self.temporary.cleanup()

    def until(self, predicate):
        end=time.monotonic()+8
        while not predicate() and time.monotonic()<end:
            self.app.processEvents();time.sleep(.01)
        self.assertTrue(predicate())

    def dialog(self, **kwargs):
        d=StartupDialog(MULTI_CONFIG,output=self.root,initial_selection={},auto_scan=False,**kwargs)
        self.dialogs.append(d)
        d.show();d.tabs.setCurrentIndex(1)
        return d

    def select(self,d,path):
        d.replay.path.setText(str(path))
        self.until(lambda:d.replay.selected is not None)

    def test_empty_and_external_folder_selection_preserved_across_tabs(self):
        with patch('ego2.multi_vio.launcher.scan_devices',side_effect=AssertionError('hardware')):
            d=self.dialog()
            self.until(lambda:not d.replay.jobs)
            self.assertEqual(d.windowTitle(),'Sensia')
            self.assertIn('暂无',d.replay.list.item(0).text())
            self.assertFalse(d.replay.start.isEnabled())
            path=make_rrd(self.root/'external 中文')
            self.select(d,path)
            self.assertTrue(d.replay.start.isEnabled())
            d.tabs.setCurrentIndex(0);d.tabs.setCurrentIndex(1)
            self.assertEqual(d.replay.path.text(),str(path))
            self.assertEqual(d.replay.values['mode'].text(),'AprilGrid 标定板跟踪与定位')

    def test_stale_details_cannot_enable_wrong_selected_folder(self):
        d=self.dialog()
        good=make_rrd(self.root/'good')
        stale=Future();fresh=Future()
        with patch.object(d.replay.pool,'submit',side_effect=[stale,fresh]):
            d.replay.path.setText(str(good));d.replay.inspect()
            d.replay.path.setText(str(self.root/'missing'));d.replay.inspect()
            d.replay.debounce.stop()
        fresh.set_result(inspect_session(self.root/'missing'));d.replay.poll()
        stale.set_result(inspect_session(good));d.replay.poll()
        self.assertFalse(d.replay.start.isEnabled())
        self.assertEqual(d.replay.selected.path,self.root/'missing')

    def test_corrupt_rrd_worker_failure_leaves_panel_open_and_retryable(self):
        root=make_rrd(self.root/'bad');(root/'replay.rrd').write_bytes(b'broken')
        d=self.dialog();self.select(d,root)
        d.replay.start.click()
        cache=Path(d.replay.temporary.name)
        self.until(lambda:not d.replay.busy)
        self.assertTrue(d.isVisible())
        self.assertTrue(d.replay.start.isEnabled())
        self.assertFalse(cache.exists())
        self.assertTrue(d.replay.notice.text())
        self.assertEqual((root/'replay.rrd').read_bytes(),b'broken')

    def test_cancel_real_background_process_cleans_cache_and_preserves_source(self):
        root=make_rrd(self.root/'original')
        d=self.dialog();self.select(d,root)
        real_popen=subprocess.Popen
        def slow(command, **kwargs):
            return real_popen([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
        with patch('ego2.multi_vio.replay_panel.subprocess.Popen',side_effect=slow):
            d.replay.start.click()
        process=d.replay.process;cache=Path(d.replay.temporary.name)
        self.assertFalse(d.tabs.tabBar().isEnabled())
        d.replay.cancel.click()
        self.until(lambda:not d.replay.busy)
        self.assertIsNotNone(process.poll())
        self.assertFalse(cache.exists())
        self.assertTrue(d.isVisible())
        self.assertTrue(d.tabs.tabBar().isEnabled())
        self.assertTrue(root.exists())

    def test_replay_survives_inflight_device_scan_without_restarting_it(self):
        gate=threading.Event()
        scan=Mock(side_effect=lambda:(gate.wait(3),[])[1])
        d=self.dialog(scanner=scan)
        d.tabs.setCurrentIndex(0);d.refresh_devices()
        self.assertIsNotNone(d.worker)
        d.tabs.setCurrentIndex(1)
        d.replay_launched()
        self.assertTrue(d.isVisible())
        gate.set()
        self.until(lambda:not d.isVisible())
        self.assertTrue(d.replay_ready)
        scan.assert_called_once()

    def test_cancel_revokes_replay_acceptance_waiting_for_a_device_scan(self):
        from ego2.multi_vio.replay_panel import ReplayPage
        gate=threading.Event()
        d=self.dialog(scanner=lambda:(gate.wait(3),[])[1])
        d.tabs.setCurrentIndex(0);d.refresh_devices();d.tabs.setCurrentIndex(1)
        d.replay_launched()
        self.assertTrue(d.replay_ready)
        with patch.object(ReplayPage,'busy',new_callable=PropertyMock,return_value=True), \
             patch.object(d.replay,'cancel_preparation') as cancel:
            d.cancel_replay()
            cancel.assert_called_once()
            self.assertFalse(d.replay_ready)
            gate.set()
            self.until(lambda:d.worker is None)
            self.assertTrue(d.isVisible())


if __name__=='__main__':unittest.main()
