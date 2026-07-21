#!/usr/bin/env python
"""Compara los PRESETS de la interfaz web sobre uno o más videos.

Mide, para cada preset, el costo (fps de proceso) y la estabilidad de identidad
(IDs únicos del tracker y slot-switches por minuto — proxy SIN ground truth,
igual que eval_tracking.py). Sirve para elegir preset con números y no de oído.

    python scripts/eval_presets.py <video> [<video> ...] [--presets a,b]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from harmocap.webapp.processing import (  # noqa: E402
    PRESET_ORDER, RunConfig, _PipelineState, preset_config,
)


def eval_preset(video: Path, run: RunConfig) -> dict:
    st = _PipelineState(run)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ids_seen: set[int] = set()
    slot_switches = frames = src_i = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        src_i += 1
        if (src_i - 1) % max(1, run.frame_stride):
            continue
        frames += 1
        t_us = int((src_i - 1) * 1e6 / fps)
        h, w = frame.shape[:2]
        dets, raw, _sp, _wh = st.backend.track_frame(frame)
        ids_seen.update(d.track_id for d in dets)
        for ev in st.slots.update(dets, t_us, aspect=w / h if h else 16 / 9):
            if ev.slot_reset and ev.detection is not None and not ev.rebound:
                slot_switches += 1
            if ev.detection is not None:
                st.person(ev, t_us)
        if st.crowd is not None:
            st.crowd.update(raw, t_us, aspect=w / h if h else 16 / 9)
    cap.release()
    dt = max(time.time() - t0, 1e-6)
    minutes = src_i / fps / 60.0
    return {"unique_track_ids": len(ids_seen),
            "slot_switches": slot_switches,
            "slot_switches_per_min": round(slot_switches / max(minutes, 1e-6), 1),
            "rebinds": st.slots.rebind_count,
            "frames": frames,
            "proc_fps": round(frames / dt, 1),
            "realtime": round(frames / dt, 1) >= fps * 0.95,
            "loaded": st.backend.info()}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sel = [a.split("=", 1)[1].split(",") for a in sys.argv[1:]
           if a.startswith("--presets=")]
    presets = sel[0] if sel else list(PRESET_ORDER)
    videos = [Path(a) for a in args]
    assert videos, __doc__
    out: dict = {}
    for v in videos:
        out[v.name] = {}
        for name in presets:
            run = preset_config(name)
            print(f"[presets] {v.name} / {name} …", flush=True)
            out[v.name][name] = {"config": asdict(run),
                                 **eval_preset(v, run)}
    dest = REPO / "reports" / "preset_comparison.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n{'video':28} {'preset':10} {'fps':>7} {'IDs':>5} "
          f"{'switch/min':>11}")
    for vn, per in out.items():
        for name, r in per.items():
            print(f"{vn[:28]:28} {name:10} {r['proc_fps']:7.1f} "
                  f"{r['unique_track_ids']:5} {r['slot_switches_per_min']:11.1f}")
    print(f"\n→ {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
