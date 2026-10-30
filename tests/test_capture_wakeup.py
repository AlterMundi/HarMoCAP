import threading
import numpy as np
from harmocap.capture import LatchingCamera


def test_capture_wakeup_preserves_single_latest_frame():
    camera = LatchingCamera.__new__(LatchingCamera)
    camera._lock = threading.Lock()
    camera._ready = threading.Event()
    camera._frame = None
    assert not camera.wait_for_frame(.001)
    with camera._lock:
        camera._frame = np.zeros((2, 2, 3))
        camera._frame_counter = 7
        camera._captured_at_us = 123
        camera._ready.set()
    assert camera.wait_for_frame(.001)
    _, fid, stamp = camera.get_latest()
    assert (fid, stamp) == (7, 123)
    assert not camera.wait_for_frame(.001)
    assert camera.get_latest() is None


def test_device_paths_remain_live_while_regular_files_are_paced(tmp_path, monkeypatch):
    """Named USB devices must behave like numeric cameras, including EOF retry."""
    import pytest
    import harmocap.capture as capture
    from pathlib import Path
    import subprocess

    media = tmp_path / 'synthetic.mp4'
    media.touch()
    media_link = tmp_path / 'media-link.mp4'
    media_link.symlink_to(media)
    device_link = tmp_path / 'usb-camera'
    device_link.symlink_to('/dev/null')
    for source, is_file in [(0, False), ('/dev/video98', False),
                            (str(device_link), False),
                            ('rtsp://example.invalid/camera', False),
                            (str(media), True), (str(media_link), True)]:
        camera = LatchingCamera.__new__(LatchingCamera)
        camera.source = source
        camera._running = True
        camera._lock = threading.Lock()
        camera._ready = threading.Event()
        camera._frame = None
        camera._frame_counter = 0
        camera._dropped = 0
        sleeps, controls = [], []

        class Capture:
            reads = 0
            def get(self, _): return 30.
            def read(self):
                self.reads += 1
                if self.reads == 1:
                    return True, np.zeros((2, 2, 3))
                camera._running = False
                return False, None
        camera.cap = Capture()
        monkeypatch.setattr(capture.time, 'monotonic', lambda: 100.)
        monkeypatch.setattr(capture, 'mono_us', lambda: 123)
        monkeypatch.setattr(capture.time, 'sleep', sleeps.append)
        monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: controls.append(a[0]))
        camera._reader()
        assert camera.wait_for_frame(.001)
        _, sequence, stamp = camera.get_latest()
        assert (sequence, stamp) == (1, 123)
        if is_file:
            assert sleeps == pytest.approx([1 / 30])
            assert not controls
        else:
            assert sleeps == [.005]  # transient read failure, no file pacing
            assert len(controls) == 1  # post-open hardware control reapply
