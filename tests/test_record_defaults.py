"""Unified launcher defaults to manual recording without opening camera streams."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

from ego2.multi_vio.__main__ import main


class RecordingDefaultTests(unittest.TestCase):
    def test_all_modes_default_to_manual_recording_and_allow_preview_only(self):
        choice = {'head': 's200032'}
        devices = {'head': {'native': {'serial': 's200032'}, 'profile': {}}}
        for mode in ('vio', 'gesture', 'aprilgrid', 'reconstruction'):
            for flags, enabled in (([], True), (['--record'], True), (['--no-record'], False)):
                for launch_flags in ([], ['--no-setup'], ['--headless']):
                    if not enabled and launch_flags == ['--headless']:
                        continue
                    with self.subTest(mode=mode, flags=flags, launch=launch_flags), ExitStack() as stack:
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        stack.enter_context(patch('ego2.multi_vio.profiles.prepare_camera_profiles', return_value=([], {})))
                        stack.enter_context(patch('ego2.multi_vio.selection.open_launcher',
                            return_value={'mode': mode, 'devices': choice}))
                        stack.enter_context(patch('ego2.multi_vio.selection.default_selection', return_value=choice))
                        for loader in ('load_multi_config', 'load_mode_config'):
                            stack.enter_context(patch('ego2.multi_vio.__main__.' + loader, return_value=({}, devices)))
                        vio = stack.enter_context(patch('ego2.multi_vio.app.run'))
                        other = stack.enter_context(patch('ego2.multi_vio.multi_mode.run'))
                        self.assertEqual(main(['--mode', mode] + flags + launch_flags), 0)
                        called, unused = (vio, other) if mode == 'vio' else (other, vio)
                        called.assert_called_once()
                        unused.assert_not_called()
                        args = called.call_args.args[2]
                        self.assertEqual(args.record, enabled)
                        self.assertEqual(args.manual_record, enabled and '--headless' not in launch_flags)

    def test_preview_only_rejects_recording_destination_or_headless(self):
        for flags in (['--headless'], ['--session', '/tmp/recording-preset']):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as caught:
                    main(['--no-record'] + flags)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn('cannot be combined with --no-record', errors.getvalue())

    def test_cli_replay_bypasses_capture_even_with_recording_enabled_by_default(self):
        with patch('ego2.multi_vio.replay.replay') as replay, \
             patch('ego2.multi_vio.selection.open_launcher', side_effect=AssertionError), \
             patch('ego2.multi_vio.profiles.prepare_camera_profiles', side_effect=AssertionError):
            self.assertEqual(main(['replay', '--session', '/tmp/saved-session']), 0)
            replay.assert_called_once()


if __name__ == '__main__':
    unittest.main()
