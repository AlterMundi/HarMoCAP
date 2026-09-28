"""Display an external instrument's state over the existing camera viewport.

The instrument owns the mapping and sends normalized image coordinates; this
module only renders them. The JSON file is local and atomically replaced by
its producer. Stale or foreign-stream states never light up the skeleton.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def read_instrument_state(path, stream_id, *, now=None):
    now = time.time() if now is None else now
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return None
        age = now - float(state["updated_at"])
        if not 0 <= age < 2.0 or state.get("stream_id") not in (None, stream_id):
            return None
        return state
    except (OSError, ValueError, TypeError, KeyError):
        return None


def draw_instrument_state(img, state, *, visible_slots):
    import cv2

    title = state.get("mode", "Instrumento") if state else "Instrumento: esperando datos"
    slot = state.get("slot_id") if state else None
    if state and slot is None:
        title += " | sin persona activa"
    elif slot is not None:
        title += f" | slot {slot}"
    cv2.putText(img, title, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (230, 230, 230), 2, cv2.LINE_AA)
    if not state or slot not in visible_slots:
        return
    h, w = img.shape[:2]
    for zone in state.get("zones", []):
        if not zone.get("observed"):
            continue
        # Same height-normalized coordinates and mirror as the skeleton.
        x, y = w - 1 - int(zone["x"] * h), int(zone["y"] * h)
        active = zone.get("active", False)
        color = (0, 235, 255) if active else (150, 150, 150)
        cv2.circle(img, (x, y), 9 if active else 5, color, 2, cv2.LINE_AA)
        label = f"H{zone['n']} {zone['freq']:.1f}Hz" if active else f"H{zone['n']}"
        cv2.putText(img, label, (max(0, min(w - 160, x + 12)), max(65, min(h - 10, y))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
