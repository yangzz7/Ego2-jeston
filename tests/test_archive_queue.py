"""An encoder stall must not stall device acquisition or discard queued pairs."""
import threading
import signal
import unittest

from ego2.common.runtime import QueuedArchive, LatestPairs


class ArchiveQueueTests(unittest.TestCase):
    def test_camera_drains_while_encoder_is_busy_then_archive_preserves_order(self):
        busy, release = threading.Event(), threading.Event()
        written = []
        def write(pair):
            if pair == 1:
                busy.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release encoder")
            written.append(pair)
        class Source:
            def __init__(self):
                self.n, self.closed = 0, False
            def open(self):
                pass
            def read(self, stop):
                self.n += 1
                if self.n == 2:
                    if not busy.wait(5):
                        raise TimeoutError("encoder did not start")
                return self.n if self.n <= 12 else None
            def close(self):
                self.closed = True
        source = Source()
        writer = QueuedArchive(write, max_pending=16).start()
        capture = LatestPairs(source, threading.Event(), on_pair=writer.submit).start()
        try:
            self.assertTrue(busy.wait(5))
            capture.thread.join(2)
            self.assertFalse(capture.thread.is_alive(), "video writer blocked source reads")
            self.assertTrue(source.closed)
            self.assertEqual(written, [])
            self.assertEqual(capture.read(), 12)
        finally:
            release.set()
            capture.close()
            writer.close()
        self.assertEqual(written, list(range(1, 13)))
        self.assertGreater(writer.peak_pending, 1)

    def test_overflow_is_explicit_and_already_queued_frames_are_preserved(self):
        busy, release = threading.Event(), threading.Event()
        written = []
        def write(pair):
            busy.set()
            if not release.wait(5):
                raise TimeoutError("test did not release encoder")
            written.append(pair)
        writer = QueuedArchive(write, max_pending=1).start()
        try:
            writer.submit(1)
            self.assertTrue(busy.wait(5))
            writer.submit(2)
            with self.assertRaisesRegex(RuntimeError, "queue is full"):
                writer.submit(3)
        finally:
            release.set()
            writer.close()
        self.assertEqual(written, [1, 2])

    def test_disk_error_reaches_shutdown(self):
        def write(pair):
            raise OSError("disk full")
        writer = QueuedArchive(write).start()
        writer.submit(1)
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            writer.close()

    def test_shutdown_can_retry_after_timeout_without_losing_queued_frames(self):
        busy, release = threading.Event(), threading.Event()
        written = []

        def write(pair):
            if pair == 1:
                busy.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release encoder")
            written.append(pair)

        writer = QueuedArchive(write, max_pending=1).start()
        try:
            writer.submit(1)
            self.assertTrue(busy.wait(5))
            writer.submit(2)
            with self.assertRaisesRegex(RuntimeError, "did not drain"):
                writer.close(timeout=0)
            with self.assertRaisesRegex(RuntimeError, "queue is closed"):
                writer.submit(3)
        finally:
            release.set()
            writer.close(timeout=5)
        self.assertFalse(writer.thread.is_alive())
        self.assertEqual(written, [1, 2])
        writer.close()  # Repeated successful shutdown remains idempotent.

    def test_capture_cleanup_defers_second_interrupt_and_restores_handler(self):
        from ego2.gesture.app import _CaptureResources
        before = signal.getsignal(signal.SIGINT)
        observed = []
        with _CaptureResources() as resources:
            def finalize():
                observed.append(signal.getsignal(signal.SIGINT))
                signal.raise_signal(signal.SIGINT)
                observed.append("finished")
            resources.callback(finalize)
        self.assertEqual(observed, [signal.SIG_IGN, "finished"])
        self.assertEqual(signal.getsignal(signal.SIGINT), before)

    def test_capture_cleanup_restores_interrupt_handler_when_finalization_fails(self):
        from ego2.gesture.app import _CaptureResources
        before = signal.getsignal(signal.SIGINT)
        observed = []

        def finalize():
            signal.raise_signal(signal.SIGINT)
            raise OSError("finalization failed")

        with self.assertRaisesRegex(OSError, "finalization failed"):
            with _CaptureResources() as resources:
                resources.callback(lambda: observed.append("remaining cleanup ran"))
                resources.callback(finalize)
        self.assertEqual(observed, ["remaining cleanup ran"])
        self.assertEqual(signal.getsignal(signal.SIGINT), before)


if __name__ == "__main__":
    unittest.main()
