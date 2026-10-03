"""Camera feedback for an external instrument; no musical inference here."""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import time

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
SEGMENTS = {1: ((11, 12),), 2: ((5, 6),),
            3: ((11, 13), (12, 14)), 4: ((5, 7), (6, 8)),
            5: ((13, 15), (14, 16)), 6: ((7, 9), (8, 10))}


def read_instrument_state(path, stream_id, *, now=None):
    now = time.time() if now is None else now
    try:
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(state, dict):
            return None
        age = now - float(state['updated_at'])
        if not 0 <= age < 2.0 or state.get('stream_id') not in (None, stream_id):
            return None
        return state
    except (OSError, ValueError, TypeError, KeyError):
        return None


class InstrumentOverlay:
    """Bounded pose-only onion history; one effects layer, no image history."""
    def __init__(self):
        self.history = deque(maxlen=24)
        self.identity = None
        self.last_capture = None

    def draw(self, img, state, *, visible_slots, now=None):
        import cv2

        now = time.monotonic() if now is None else now
        identity = (state.get('stream_id'), state.get('slot_id')) if state else None
        if identity != self.identity:
            self.history.clear()
            self.last_capture = None
            self.identity = identity
        slot = state.get('slot_id') if state else None
        title = state.get('mode', 'Instrumento') if state else 'Instrumento: esperando datos'
        title += f' | slot {slot}' if slot is not None else ' | sin persona activa'
        cv2.putText(img, title, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (230, 230, 230), 2, cv2.LINE_AA)
        if not state or slot not in visible_slots:
            self.history.clear()
            return
        h, w = img.shape[:2]
        def pixel(p):
            return w - 1 - int(p['x'] * h), int(p['y'] * h)
        skeleton = state.get('skeleton', [])
        cap = state.get('captured_at_us')
        if skeleton and cap != self.last_capture:
            self.history.append((now, skeleton))
            self.last_capture = cap
        trail_s = max(0, min(800, state.get('trail_ms', 320))) / 1000
        while self.history and now - self.history[0][0] > trail_s:
            self.history.popleft()
        effects = img.copy()
        # Only stored coordinates, decimated to at most 12 ghost skeletons.
        if trail_s:
            for stamp, points in list(self.history)[:-1:2]:
                fade = max(0, 1 - (now - stamp) / trail_s)
                color = tuple(int(c * fade) for c in (180, 170, 135))
                for a, b in EDGES:
                    if points[a]['observed'] and points[b]['observed']:
                        cv2.line(effects, pixel(points[a]), pixel(points[b]), color, 1, cv2.LINE_AA)
        zones = state.get('zones', [])
        for zone in zones:
            if not zone.get('observed'):
                continue
            color_hex = zone.get('color', '#ffe000').lstrip('#')
            color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            energy = min(1.0, zone.get('gain', 0) / .45) if zone.get('active') else 0.0
            points = zone.get('points') or [zone]
            for point in points:
                if not point.get('observed'):
                    continue
                contribution = min(1.0, point.get('speed', zone.get('speed', 1)) /
                                   max(zone.get('speed', 1), 1e-9))
                e = energy * contribution
                if e > 0:
                    cv2.circle(effects, pixel(point), 12 + int(18 * math.sqrt(e)),
                               color, -1, cv2.LINE_AA)
        cv2.addWeighted(effects, .22, img, .78, 0, dst=img)
        # Trajectories connect actual consecutive measurements in space and time.
        # Missing observations break a trail instead of inventing a connecting path.
        for zone in zones:
            color_hex = zone.get('color', '#ffe000').lstrip('#')
            full = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            for joint in zone.get('points', []):
                index = joint.get('index')
                previous = None
                for stamp, pose in self.history:
                    point = pose[index] if isinstance(index, int) and 0 <= index < len(pose) else None
                    if not point or not point.get('observed'):
                        previous = None
                        continue
                    if previous is not None and trail_s:
                        fade = max(0, 1 - (now - stamp) / trail_s)
                        color = tuple(int(c * .75 * fade) for c in full)
                        cv2.line(img, pixel(previous), pixel(point), color,
                                 1 + int(3 * fade), cv2.LINE_AA)
                    previous = point
        for zone in zones:
            if not zone.get('observed'):
                continue
            color_hex = zone.get('color', '#ffe000').lstrip('#')
            full = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            energy = min(1., zone.get('gain', 0) / .45) if zone.get('active') else 0.
            for j, point in enumerate(zone.get('points') or [zone]):
                if not point.get('observed'):
                    continue
                e = energy * min(1., point.get('speed', zone.get('speed', 1)) /
                                 max(zone.get('speed', 1), 1e-9))
                color = tuple(int(c * (.35 + .65 * e)) for c in full)
                x, y = pixel(point)
                radius = 5 + int(9 * math.sqrt(e))
                segments = SEGMENTS.get(zone['n'], ())
                if skeleton and j < len(segments) and e > 0:
                    a, b = segments[j]
                    if skeleton[a]['observed'] and skeleton[b]['observed']:
                        cv2.line(img, pixel(skeleton[a]), pixel(skeleton[b]), color,
                                 2 + int(4 * e), cv2.LINE_AA)
                cv2.circle(img, (x, y), radius, color, -1 if e > 0 else 1, cv2.LINE_AA)
                label = f"F{zone['n']} {zone['freq']:.0f}Hz" if e > 0 else f"F{zone['n']}"
                cv2.putText(img, label, (max(0, min(w - 140, x + radius + 9)),
                                        max(65, min(h - 30, y))),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def draw_instrument_state(img, state, *, visible_slots):
    """Stateless rendering convenience for consumers without onion history."""
    InstrumentOverlay().draw(img, state, visible_slots=visible_slots)
