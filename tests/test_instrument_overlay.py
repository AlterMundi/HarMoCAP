import json

import numpy as np

from harmocap.instrument_overlay import read_instrument_state, draw_instrument_state


def test_state_requires_fresh_timestamp_and_matching_stream(tmp_path):
    path = tmp_path / "state.json"
    assert read_instrument_state(path, "camera", now=10) is None
    path.write_text(json.dumps(dict(updated_at=9, stream_id="camera", zones=[])))
    assert read_instrument_state(path, "camera", now=10) is not None
    assert read_instrument_state(path, "other", now=10) is None
    assert read_instrument_state(path, "camera", now=12) is None
    path.write_text("{incomplete")
    assert read_instrument_state(path, "camera", now=10) is None


def test_zones_draw_only_for_visible_person_and_use_skeleton_coordinates():
    state = dict(mode="Consonancia kinetica", slot_id=2,
                 zones=[dict(n=9, x=.5, y=.5, observed=True,
                             active=True, freq=363.6)])
    absent = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_instrument_state(absent, state, visible_slots={1})
    assert not absent[300:450].any()
    visible = np.zeros_like(absent)
    draw_instrument_state(visible, state, visible_slots={2})
    # Mirrored x = 1280 - 1 - .5 * 720 = 919; y = 360.
    assert visible[350:371, 909:930].any()
    assert not visible[350:371, 350:371].any()


def test_waiting_state_can_render_without_producer():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_instrument_state(img, None, visible_slots=set())
    assert img.any()


def test_raw_pose_has_no_smoothing_or_held_coordinates():
    from harmocap.smoothing import raw_keypoint_sample
    from harmocap.schema import KpState
    first = raw_keypoint_sample([(.1, .2, 1.)] * 17)
    second = raw_keypoint_sample([(.9, .8, 1.)] * 17)
    assert first[0][:2] == (.1, .2)
    assert second[0] == (.9, .8, 1., int(KpState.OBSERVED), 0, 0)
    invalid = raw_keypoint_sample([(.3, .4, .1)] * 17)
    assert invalid[0] == (.3, .4, .1, int(KpState.INVALID), 0, 0)
