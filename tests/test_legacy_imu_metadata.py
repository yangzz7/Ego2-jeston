"""Old attitude fields remain readable after removing the optional attitude feature."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import queue
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun.blueprint as b

from ego2.common.cyper_metadata import CyperMetadataDecoder
from ego2.common.sources import Pair, Source, decode_cyper
from ego2.gesture.dataset import GestureDataset
from ego2.multi_vio.dataset import finish_gesture
from ego2.multi_vio.mode_viewer import BatchRecording, GestureSink, build_blueprint
from ego2.multi_vio.replay import Reader, export_dataset
from ego2.multi_vio.storage import status
from tests.test_cyper_metadata import metadata, imu, transport


def old_attitude():
    return dict(valid=True, euler_valid=True, status='running', epoch=0,
                timestamp_seconds=10., rpy_deg=[1., 2., 3.], quaternion_wxyz=[1., 0., 0., 0.])


def config(roles=('head',)):
    return dict(mode='gesture', imu_attitude=True, devices={r:dict(sn='test-'+r) for r in roles},
                viewer=dict(row_order=list(roles), memory_limit='100MiB'))


class LegacyMetadataTests(unittest.TestCase):
    def test_stale_profile_flag_does_not_change_raw_capture_images_or_imu_samples(self):
        words = metadata();imu(words, 0)
        raw = transport(words)
        profile = dict(input=dict(type='cyper', device='/dev/fake', width=160, height=8,
                                  white_balance=False, preserve_raw=True, preserve_display_raw=True),
                       _base='.', imu_attitude=dict(enabled=True))
        cap = MagicMock();cap.isOpened.return_value=True;cap.read.return_value=(True, raw)
        cap.get.side_effect=lambda key:0 if key == 16 else 30
        source = Source(profile, {})
        with patch('ego2.common.cyper_cuda.capture_backend', return_value='cpu'), \
             patch('cv2.VideoCapture', return_value=cap):
            try:
                source.open()
                pair = source.read(threading.Event())
            finally:
                source.close()
        np.testing.assert_array_equal(pair.raw_transport, raw)
        np.testing.assert_array_equal(pair.display_transport, raw)
        for actual, expected in zip(pair.images, decode_cyper(raw, 160, 8)[0]):
            np.testing.assert_array_equal(actual, expected)
        expected = CyperMetadataDecoder().decode(raw, 160, 8)
        for key, value in expected.items():
            self.assertEqual(pair.metadata[key], value)
        self.assertTrue(pair.metadata['imu_samples'])
        self.assertNotIn('imu_attitude', pair.metadata)

    def test_old_live_metadata_draws_stereo_images_without_attitude_charts(self):
        roles = ('chest', 'head', 'left_hand', 'right_hand')
        layout = build_blueprint(config(roles))
        self.assertEqual(len(layout.root_container.contents), 4)
        for row in layout.root_container.contents:
            self.assertEqual(len(row.contents), 2)
            self.assertTrue(all(isinstance(view, b.Spatial2DView) for view in row.contents))
        q = queue.Queue()
        record = BatchRecording(q, 'head', 0, demo=True, eye_names=('cam0', 'cam1'))
        sink = GestureSink(dict(viewer={}), record)
        image = np.zeros((24, 32, 3), np.uint8)
        sink.log(Pair((image, image), 0, 0, 1, dict(imu_attitude=old_attitude())), .1, [[], []])
        data = pickle.loads(q.get_nowait())
        paths = {entry[0] for entry in data['logs']}
        self.assertTrue(any('/cam0/' in path for path in paths))
        self.assertTrue(any('/cam1/' in path for path in paths))
        self.assertFalse(any('/sensors/attitude' in path for path in paths))

    def test_old_panel_result_starts_gesture_without_enabling_attitude_or_vio(self):
        from ego2.multi_vio.__main__ import main
        with redirect_stdout(io.StringIO()), \
             patch('ego2.multi_vio.profiles.prepare_camera_profiles', return_value=([], {})), \
             patch('ego2.multi_vio.selection.open_launcher', return_value=dict(
                 mode='gesture', devices={'head':'s200032'}, imu_attitude=True)), \
             patch('ego2.multi_vio.multi_mode.run') as run, patch('ego2.multi_vio.app.run') as vio:
            self.assertEqual(main(['--no-record']), 0)
            cfg, devices, _ = run.call_args.args
            self.assertNotIn('imu_attitude', cfg)
            self.assertNotIn('imu_attitude', devices['head']['profile'])
            vio.assert_not_called()

    def test_finalize_keeps_raw_imu_and_old_dataset_still_exports_matched_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'session';root.mkdir()
            work = root/'_work'
            profile = dict(input=dict(type='demo'), channels=[dict(camera='cam0'), dict(camera='cam1')],
                           imu_attitude=dict(enabled=True))
            writer = GestureDataset(work/'head/data', profile, {}, fps=10)
            pairs, samples = [], []
            for index in range(4):
                image = np.full((24, 32, 3), index*30, np.uint8)
                sample = dict(timestamp_seconds=10+index*.1,
                              acceleration_m_s2=[1., 2., 9.80665], angular_velocity_rad_s=[.1, .2, .3])
                samples.append(sample)
                pair = Pair((image, image.copy()), 100+index*.1, 200+index*.1, index,
                            dict(imu_attitude=old_attitude(), imu_samples=[sample]))
                pairs.append(pair);writer.capture(pair)
            for index in (0, 3):
                writer.result(pairs[index], index*.1, [[], []])
            writer.close()
            status(root, 'recording', mode='gesture')
            finish_gesture(root, work, config(), origin=200)
            self.assertEqual(pq.read_table(root/'sensors/head/imu.parquet').to_pylist(), samples)
            self.assertFalse((root/'sensors/head/attitude.csv').exists())
            frames = list(Reader(root).frames())
            self.assertEqual(len(frames), 4)
            for index, (_, images, row) in enumerate(frames):
                self.assertNotIn('imu_attitude', json.loads(row['observation.head.result_json']))
                np.testing.assert_array_equal(images['head']['cam0'], pairs[index].images[0])
            # Emulate an already-recorded session with the obsolete result field.
            for path in (root/'data').glob('chunk-*/file-*.parquet'):
                table = pq.read_table(path)
                rows = table.to_pylist()
                for row in rows:
                    result = json.loads(row['observation.head.result_json'])
                    result['imu_attitude'] = old_attitude()
                    row['observation.head.result_json'] = json.dumps(result)
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
            exported = export_dataset(root, root/'replay.rrd')
            from rerun.experimental import RrdReader
            paths = {chunk.entity_path for chunk in RrdReader(str(exported)).stream()}
            self.assertIn('/cameras/head/cam0/rgb', paths)
            self.assertIn('/cameras/head/cam1/rgb', paths)
            self.assertFalse(any('/sensors/attitude' in path for path in paths))


if __name__ == '__main__':
    unittest.main()
