"""Preview mode isolation, timing, camera selection and resource cleanup."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import rerun.blueprint as b

from ego2.common.sources import Source, Pair
from ego2.multi_vio.config import MULTI_CONFIG
from ego2.multi_vio.modes import load_mode_config, MODES
from ego2.multi_vio.roles import ROLES
from ego2.multi_vio.selection import write_preferences, read_setup
from ego2.multi_vio.rgb import RgbViewer, build_blueprint, enhanced_images, run


def args(**values):
    return SimpleNamespace(**dict(dict(demo=False, frames=3, seconds=0,
        no_viz=False, headless=False, connect=None, web=False, web_port=9090,
        grpc_port=9876), **values))


class RgbConfigTests(unittest.TestCase):
    def test_multiple_roles_without_chest_and_no_algorithm_profiles(self):
        with patch('ego2.gesture.config.load_config', side_effect=AssertionError('gesture loaded')), \
             patch('ego2.reconstruction.calibration.load_config', side_effect=AssertionError('VIO loaded')):
            cfg, devices = load_mode_config(MULTI_CONFIG, mode='rgb',
                selection={'head':'s200032', 'left_hand':'s200100'})
        self.assertEqual(MODES['rgb'], '纯RGB画面')
        self.assertEqual(set(devices), {'head', 'left_hand'})
        self.assertIsNone(cfg['reference_role'])
        self.assertEqual(cfg['viewer']['display_isp']['max_fps'], 30)
        for data in devices.values():
            self.assertNotIn('gesture', data['profile'])
            self.assertFalse(data['profile']['input']['preserve_raw'])
            self.assertTrue(data['profile']['input']['preserve_display_raw'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'selection.json'
            write_preferences(path, {'head':'s200032'}, mode='rgb')
            self.assertEqual(read_setup(path)['mode'], 'rgb')

    def test_blueprint_contains_only_selected_stereo_views_and_timeline(self):
        for count in (1, 4):
            cfg = {'devices': {r:{'sn':f'test{i}'} for i,r in enumerate(ROLES[:count])},
                   'viewer': {'row_order': list(ROLES[:count])}}
            blueprint = build_blueprint(cfg)
            self.assertEqual(len(blueprint.root_container.contents), count)
            for row in blueprint.root_container.contents:
                self.assertEqual(len(row.contents), 2)
                self.assertTrue(all(isinstance(v, b.Spatial2DView) for v in row.contents))
            with tempfile.TemporaryDirectory() as folder:
                blueprint.save('rgb-test', str(Path(folder)/'rgb.rbl'))

    def test_cli_dispatch_disables_recording_even_with_normal_default(self):
        from ego2.multi_vio.__main__ import main
        with redirect_stdout(io.StringIO()), \
             patch('ego2.multi_vio.selection.open_launcher', return_value={'mode':'rgb','devices':{'head':'s200032'}}), \
             patch('ego2.multi_vio.profiles.prepare_camera_profiles', return_value=([],{})), \
             patch('ego2.multi_vio.rgb.run') as preview, \
             patch('ego2.multi_vio.multi_mode.run') as algorithms:
            self.assertEqual(main([]), 0)
        self.assertFalse(preview.call_args.args[2].record)
        self.assertFalse(preview.call_args.args[2].manual_record)
        algorithms.assert_not_called()


class RawOnlyTests(unittest.TestCase):
    def test_no_demosaic_white_balance_or_cuda_open_and_owns_raw(self):
        raw = np.full((8,160,2), 37, np.uint8)
        raw[-1] = 0
        raw[-1,0,0], raw[-1,1,0] = 2,7
        profile = {'_base':'.','input':{'type':'cyper','device':'/dev/fake',
                    'width':160,'height':8,'fps':30,'preserve_display_raw':True}}
        source = Source(profile, {}, decode_images=False)
        cap = MagicMock()
        cap.get.side_effect = lambda key: 0 if key == 16 else 30
        cap.read.return_value = (True,raw)
        with patch('ego2.common.sources.cv2.VideoCapture', return_value=cap), \
             patch('ego2.common.cyper_cuda.CudaDecoder', side_effect=AssertionError('double decode')), \
             patch('ego2.common.sources.decode_cyper', side_effect=AssertionError('CPU decode')), \
             patch('ego2.common.sources.white_balance', side_effect=AssertionError('WB')):
            try:
                source.open()
                pair = source.read(threading.Event())
            finally:
                source.close()
        self.assertEqual(pair.images, ())
        self.assertIsNone(pair.raw_transport)
        np.testing.assert_array_equal(pair.display_transport, raw)
        raw[:] = 99
        self.assertFalse(np.all(pair.display_transport == 99))
        cap.release.assert_called_once()

    def test_raw_only_rejects_inputs_without_owned_display_raw(self):
        for kind, keep in [('video', True), ('cyper', False)]:
            with self.assertRaises(ValueError):
                Source({'input':{'type':kind, 'preserve_display_raw':keep}}, {}, decode_images=False)

    def test_isp_failure_is_visible_and_no_unenhanced_frame_is_substituted(self):
        pair = Pair((), 0, 0, 1, display_transport=np.zeros((8,8,2),np.uint8))
        isp = MagicMock()
        isp.process_transport.side_effect = RuntimeError('ISP failed')
        with self.assertRaisesRegex(RuntimeError, 'ISP failed'):
            enhanced_images(isp, pair, {'input': {'width':8,'height':8}})


class FakeSource:
    instances = []
    def __init__(self, profile, cameras, *, decode_images):
        assert not decode_images
        self.serial = profile['input']['serial']
        self.closed = False
        self.count = 0
        self.instances.append(self)
    def open(self): pass
    def read(self, stop):
        if stop.wait(.002): return None
        self.count += 1
        now = time.monotonic()
        raw = np.full((8,8,2), self.count % 255, np.uint8)
        return Pair((),now,now,self.count,display_transport=raw)
    def close(self): self.closed = True


class FakeIsp:
    backend = 'cuda'
    def __init__(self, fail=False): self.closed, self.fail = False, fail
    def process_transport(self, raw):
        if self.fail: raise RuntimeError('test ISP failure')
        return tuple(np.repeat(raw[:,:,i:i+1], 3, axis=2) for i in range(2))
    def close(self): self.closed = True


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        FakeSource.instances = []
        self.cfg = {'viewer': {'display_isp': {'max_fps': 100}}}
        self.devices = {r:{'native':{'serial':r}, 'cameras':{}, 'profile':{
            'input':{'serial':r,'type':'cyper','width':8,'height':8},
            'channels':[{'camera':'cam0'},{'camera':'cam1'}]}} for r in ROLES}

    def test_four_camera_preview_closes_resources_and_skips_old_frames(self):
        processors = [FakeIsp() for _ in ROLES]
        viewer = MagicMock()
        with patch('ego2.multi_vio.transport.check_devices'), \
             patch('ego2.multi_vio.rgb.Source', FakeSource), \
             patch('ego2.multi_vio.rgb.open_display_isp', side_effect=processors), \
             redirect_stdout(io.StringIO()):
            result = run(self.cfg, self.devices, args(frames=5), viewer_factory=lambda *_:viewer)
        self.assertEqual(set(result), set(ROLES))
        self.assertTrue(all(row['frames']==5 for row in result.values()))
        self.assertTrue(all(row['preview_dropped']>0 for row in result.values()))
        self.assertEqual(viewer.images.call_count, 20)
        self.assertTrue(all(source.closed for source in FakeSource.instances))
        self.assertTrue(all(isp.closed for isp in processors))
        viewer.close.assert_called_once()

    def test_failure_stops_other_cameras_and_closes_all_resources(self):
        processors = [FakeIsp(fail=i==0) for i in range(4)]
        viewer = MagicMock()
        with patch('ego2.multi_vio.transport.check_devices'), \
             patch('ego2.multi_vio.rgb.Source', FakeSource), \
             patch('ego2.multi_vio.rgb.open_display_isp', side_effect=processors), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, '失败'):
            run(self.cfg, self.devices, args(frames=0), viewer_factory=lambda *_:viewer)
        self.assertTrue(all(source.closed for source in FakeSource.instances))
        self.assertTrue(all(isp.closed for isp in processors))
        viewer.close.assert_called_once()

    def test_viewer_publishes_only_rgb_and_a_shared_pair_timestamp(self):
        viewer = RgbViewer.__new__(RgbViewer)
        viewer.recording, viewer.lock, viewer.quality = MagicMock(), threading.Lock(), 85
        images = (np.zeros((8,8,3), np.uint8),)*2
        viewer.images('head', 1.25, images, [{'camera':'cam0'},{'camera':'cam1'}])
        viewer.recording.set_time.assert_called_once_with('capture_time', duration=1.25)
        self.assertEqual([c.args[0] for c in viewer.recording.log.call_args_list],
                         ['/cameras/head/cam0/rgb', '/cameras/head/cam1/rgb'])


if __name__ == '__main__':
    unittest.main()
