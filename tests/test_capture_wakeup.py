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
