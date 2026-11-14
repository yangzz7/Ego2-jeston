"""Live preview must not wait for board detection or mix frame identities."""
from contextlib import ExitStack
from concurrent.futures import Future
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ego2.common.display_isp import DisplayIsp
from ego2.multi_vio.app import Output, live
from ego2.multi_vio.live_preview import TrajectoryPreview
from ego2.multi_vio.visualizer import Visualizer
from tests.test_live_display_isp import FakeIsp


def packet(n, map_id='0'):
    return dict(frame_id=n, capture_time=n*.1, map_id=map_id)


class LivePreviewTests(unittest.TestCase):
    def setUp(self):
        self.images = tuple(np.full((24,32,3),v,np.uint8) for v in (11,22))
        for image in self.images:
            image.flags.writeable = False
        self.publish, self.annotate = MagicMock(), MagicMock()

    def preview(self, processors=None):
        preview = TrajectoryPreview(('head',), processors or {}, 85, 1000,
                                    self.publish, self.annotate)
        self.addCleanup(preview.close)
        return preview

    def test_eyes_encode_in_parallel_and_publish_only_as_complete_pair(self):
        preview = self.preview()
        right_done = threading.Event()
        def encode(image, quality):
            self.assertEqual(quality,85)
            if image is self.images[0]:
                self.assertTrue(right_done.wait(2), 'Encoding was serial')
            else:
                self.publish.assert_not_called()
                right_done.set()
            return int(image[0,0,0])
        with patch('ego2.multi_vio.live_preview.encode_image',side_effect=encode):
            preview._render('head',packet(1),self.images,'RUNNING')
        self.publish.assert_called_once_with('head',packet(1),[11,22],'RUNNING')
        self.annotate.assert_not_called()

    def test_encoding_failure_never_publishes_partial_stereo_frame(self):
        preview = self.preview()
        def encode(image, quality):
            if image is self.images[1]: raise RuntimeError('encoder failure')
            return 11
        with patch('ego2.multi_vio.live_preview.encode_image',side_effect=encode):
            with self.assertRaisesRegex(RuntimeError,'encoder failure'):
                preview._render('head',packet(1),self.images,'RUNNING')
        self.publish.assert_not_called()
        self.assertFalse(preview.displayed['head'])

    def test_board_before_image_is_snapshotted_and_added_after_matching_image(self):
        preview = self.preview()
        events = []
        self.publish.side_effect = lambda *args: events.append('image')
        self.annotate.side_effect = lambda *args: events.append('overlay')
        p = packet(1)
        obs = [dict(id=12)]
        track = SimpleNamespace(status='tracking')
        preview.board('head',p,obs,track,'RUNNING',25)
        p['frame_id']=999; obs[0]['id']=99; track.status='changed'
        self.assertFalse(events)
        preview._render('head',packet(1),self.images,'RUNNING')
        self.assertEqual(events,['image','overlay'])
        args=self.annotate.call_args.args
        self.assertEqual(args[1],packet(1))
        self.assertEqual(args[2],[dict(id=12)])
        self.assertEqual(args[3].status,'tracking')

    def test_late_board_keeps_original_time_and_dropped_frame_has_no_overlay(self):
        preview = self.preview()
        for n in (1,3): preview._render('head',packet(n),self.images,'RUNNING')
        preview.board('head',packet(1),[],None,'RUNNING',30)
        preview.board('head',packet(2),[],None,'RUNNING',30)
        preview.board('head',packet(3,'reset'),[],None,'RUNNING',30)
        self.annotate.assert_called_once_with('head',packet(1),[],None,'RUNNING',30)
        self.assertEqual(self.publish.call_count,2)  # no old-image re-send

    def test_history_and_pending_boards_are_bounded(self):
        preview = self.preview()
        preview.HISTORY=3; preview.PENDING_BOARDS=2
        for n in range(5): preview._render('head',packet(n),self.images,'RUNNING')
        for n in range(5,10): preview.board('head',packet(n),[],None,'RUNNING',30)
        self.assertEqual(len(preview.displayed['head']),3)
        self.assertEqual(len(preview.pending['head']),2)
        preview.board('head',packet(0),[],None,'RUNNING',30)
        self.annotate.assert_not_called()
        preview._render('head',packet(10),self.images,'RUNNING')
        self.assertFalse(preview.pending['head'])

    def test_matching_frame_numbers_in_different_cameras_never_share_overlays(self):
        preview=TrajectoryPreview(('head','chest'),{},85,1000,self.publish,self.annotate)
        self.addCleanup(preview.close)
        preview._render('chest',packet(1),self.images,'RUNNING')
        preview.board('head',packet(1),[],None,'RUNNING',30)
        self.annotate.assert_not_called()
        preview._render('head',packet(1),self.images,'RUNNING')
        self.annotate.assert_called_once_with('head',packet(1),[],None,'RUNNING',30)

    def test_slow_isp_drops_only_pending_display_and_preserves_original_pixels(self):
        isp=FakeIsp(block=True)
        processor=DisplayIsp(isp,max_fps=1000)
        preview=self.preview({'head':processor})
        raw=np.full((24,32,2),33,np.uint8)
        raw.flags.writeable=False
        try:
            for n in (1,2,3):
                preview.submit('head',packet(n),self.images,'RUNNING',raw)
                if n==1: self.assertTrue(isp.entered.wait(2))
            preview.board('head',packet(2),[],None,'RUNNING',40)
        finally:
            isp.release.set()
            preview.close()
        self.assertEqual([c.args[1]['frame_id'] for c in self.publish.call_args_list],[1,3])
        self.annotate.assert_not_called()
        self.assertFalse(preview.pending['head'])
        self.assertEqual(processor.dropped,1)
        for image,value in zip(self.images,(11,22)): np.testing.assert_array_equal(image,value)
        np.testing.assert_array_equal(raw,33)

    def test_no_isp_fallback_is_also_bounded_and_asynchronous(self):
        preview=self.preview()
        entered,release=threading.Event(),threading.Event()
        def publish(role,p,*args):
            if p['frame_id']==1:
                entered.set()
                if not release.wait(2): raise TimeoutError('test release')
        self.publish.side_effect=publish
        try:
            preview.submit('head',packet(1),self.images,'RUNNING')
            self.assertTrue(entered.wait(2))
            for n in (2,3): preview.submit('head',packet(n),self.images,'RUNNING')
        finally:
            release.set(); preview.close()
        self.assertEqual([c.args[1]['frame_id'] for c in self.publish.call_args_list],[1,3])
        self.assertEqual(preview.fallback['head'].dropped,1)

    def test_bgr_jpeg_keeps_color_and_does_not_copy_or_convert_algorithm_buffer(self):
        preview=self.preview()
        images=tuple(np.full((24,32,3),color,np.uint8) for color in ((255,0,0),(0,0,255)))
        for image in images: image.flags.writeable=False
        encoded=[]
        with patch('ego2.multi_vio.preview_transport.rr.EncodedImage',
                   side_effect=lambda **kw: encoded.append(kw['contents']) or 'jpeg'), \
             patch('ego2.multi_vio.preview_transport.cv2.cvtColor',side_effect=AssertionError('Extra color conversion')):
            preview._render('head',packet(1),images,'RUNNING')
        colors=[cv2.imdecode(buf,cv2.IMREAD_COLOR)[10,10] for buf in encoded]
        self.assertEqual(sorted(int(np.argmax(c)) for c in colors),[0,2])
        for image,color in zip(images,((255,0,0),(0,0,255))):
            np.testing.assert_array_equal(image[0,0],color)

    def test_each_published_image_clears_previous_frame_annotations(self):
        viewer=Visualizer.__new__(Visualizer); viewer.recording=MagicMock()
        viewer._preview_images('head',packet(3),['left-jpeg','right-jpeg'],'RUNNING')
        self.assertEqual(viewer.recording.set_time.call_args.kwargs,{'duration':3*.1})
        calls=viewer.recording.log.call_args_list
        self.assertEqual([c.args[0] for c in calls],[
            '/cameras/head/cam0/overlays','/cameras/head/cam0/image','/cameras/head/cam0/status',
            '/cameras/head/cam1/overlays','/cameras/head/cam1/image','/cameras/head/cam1/status'])
        self.assertEqual(calls[1].args[1],'left-jpeg')
        self.assertEqual(calls[4].args[1],'right-jpeg')


class CoordinatorIntegrationTests(unittest.TestCase):
    def test_busy_board_worker_does_not_suppress_new_camera_images(self):
        output=MagicMock()
        output.counts={'head':{'frames':0}}
        output.viewer.display_processors={'head':None}
        output.recording.session=None
        backend=MagicMock()
        backend.failed=False; backend.process.poll.return_value=None
        images=(np.zeros((8,8,3),np.uint8),)*2
        backend.receive.side_effect=[(packet(n),images) for n in (1,2,3)]
        def ingest(role,p,now): output.counts[role]['frames']+=1
        output.ingest.side_effect=ingest
        board=MagicMock(); board.future=None
        pool=MagicMock(); pool.submit.return_value=Future()  # never completes
        args=SimpleNamespace(seconds=0,frames=3,manual_record=False)
        with patch('ego2.multi_vio.app.zmq.Context'),patch('ego2.multi_vio.app.zmq.Poller'), \
             patch('ego2.multi_vio.app.Backend',return_value=backend), \
             patch('ego2.multi_vio.app.BoardWorker',return_value=board), \
             patch('ego2.multi_vio.app.ThreadPoolExecutor') as factory, ExitStack() as resources:
            factory.return_value.__enter__.return_value=pool
            live({'mode':'multi_vio','timing':{'startup_timeout_s':30}},
                 {'head':{'native':{}}},args,output,'unused',resources)
        self.assertEqual(output.viewer.preview.call_count,3)
        self.assertTrue(all(c.kwargs['need_images'] for c in backend.receive.call_args_list))
        self.assertEqual(pool.submit.call_count,1)
        self.assertIs(pool.submit.call_args.args[2],images)
        output.board.assert_not_called()

    def test_recording_receives_original_board_payload_while_preview_gets_overlay_only(self):
        output=Output.__new__(Output)
        output.args=SimpleNamespace(demo=False)
        output.devices={'head':{}}
        output.counts={'head':dict(board_frames=0,valid_board=0)}
        output.coordinator=MagicMock()
        output.coordinator.devices={'head':SimpleNamespace(map_id='0')}
        output.recording=MagicMock(); output.viewer=MagicMock()
        p=packet(1); images=(object(),object()); obs=[]; track=SimpleNamespace(measurement=None)
        output.board('head',(p,images,obs,track,40),.5)
        self.assertIs(output.recording.board.call_args.args[2],images)
        self.assertIs(output.recording.board.call_args.args[3],obs)
        self.assertIs(output.recording.board.call_args.args[4],track)
        output.viewer.images.assert_not_called()
        output.viewer.board_overlay.assert_called_once()


if __name__=='__main__':
    unittest.main()
