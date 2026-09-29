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


def test_onion_history_is_bounded_and_resets_with_person():
    from harmocap.instrument_overlay import InstrumentOverlay
    overlay = InstrumentOverlay()
    skeleton = [dict(x=.5, y=.5, observed=True)] * 17
    state = dict(mode='Consonancia kinetica', stream_id='a', slot_id=0,
                 skeleton=skeleton, zones=[], trail_ms=800)
    for frame in range(100):
        state['captured_at_us'] = frame
        overlay.draw(np.zeros((180, 320, 3), dtype=np.uint8), state,
                     visible_slots={0}, now=frame / 100)
    assert len(overlay.history) == 24
    state['slot_id'] = 1
    overlay.draw(np.zeros((180, 320, 3), dtype=np.uint8), state,
                 visible_slots={1}, now=1)
    assert len(overlay.history) == 1
    overlay.draw(np.zeros((180, 320, 3), dtype=np.uint8), None,
                 visible_slots=set(), now=1.1)
    assert not overlay.history


def test_joint_intensity_follows_gain_not_just_tracking():
    state = dict(mode='Consonancia kinetica', slot_id=0,
                 zones=[dict(n=3, x=.5, y=.5, observed=True, active=False,
                             color='#8ede92', gain=0, speed=1, freq=121.2)])
    quiet = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_instrument_state(quiet, state, visible_slots={0})
    state['zones'][0].update(active=True, gain=.4)
    loud = np.zeros_like(quiet)
    draw_instrument_state(loud, state, visible_slots={0})
    assert loud[345:376, 904:935].sum() > quiet[345:376, 904:935].sum()


def test_trajectory_connects_positions_but_breaks_at_missing_observation():
    from harmocap.instrument_overlay import InstrumentOverlay

    def render(missing):
        overlay = InstrumentOverlay()
        for frame, x in enumerate((.3, .5, .7)):
            pose = [dict(x=x, y=.7, observed=False) for _ in range(17)]
            pose[9]['observed'] = not (missing and frame == 1)
            state = dict(slot_id=0, stream_id='a', captured_at_us=frame,
                         skeleton=pose, trail_ms=800,
                         zones=[dict(n=6, color='#cf98f9', observed=True,
                                     active=False, points=[dict(index=9, **pose[9])])])
            img = np.zeros((200, 400, 3), dtype=np.uint8)
            overlay.draw(img, state, visible_slots={0}, now=frame*.1)
        return img

    # Between old and current positions, away from nodes, labels, skeleton edges.
    assert render(False)[137:144, 305:315].any()
    assert not render(True)[137:144, 305:315].any()
