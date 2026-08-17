#!/usr/bin/env python
"""M2/M4 — corre el pipeline de tiempo real (plan M2).

Fuente: webcam (índice) o archivo de video. Emite OSC según el contrato v1 y
reporta al final las métricas GO/NO-GO (latencia software p50/p95/p99, jitter
|Δsent−Δcaptured|, drops) que se versionan en reports/<run_id>/.

Uso:
    python scripts/run_realtime.py --source 0 --show
    python scripts/run_realtime.py --source video.mp4 --record outputs/sessions/s1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from harmocap.pipeline import HarmocapPipeline  # noqa: E402


# COCO-17 connections in the same order used by YOLO-pose.
SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

COLS, ROWS = 4, 8
N_PADS = COLS * ROWS  # 32 harmonic pads in 4×8 serpentine grid


def pad_index(col: int, row: int) -> int:
    """Serpentine: even cols bottom→top, odd cols top→bottom."""
    if col % 2 == 0:
        return col * ROWS + row
    return col * ROWS + (ROWS - 1 - row)


def pad_from_xy(kp_x: float, kp_y: float, w: int, h: int) -> int | None:
    """Map keypoint to pad index 0..31, or None if invalid.
    
    HarMoCAP normalises X relative to height (not width), so kp_x * h
    gives the pixel X position.  Y is unit-normalised (kp_y * h = pixel Y).
    """
    if not (0.0 <= kp_y <= 1.0):
        return None
    # Mirror X for flipped display (same transform as skeleton drawing)
    px = w - 1 - int(kp_x * h)
    py = int(kp_y * h)
    col = max(0, min(COLS - 1, px * COLS // w))
    row = max(0, min(ROWS - 1, py * ROWS // h))
    # Flip row: HarMoCAP Y=0 at top → grid row=7 (top), Y=1 at bottom → row=0
    row = ROWS - 1 - row
    return pad_index(col, row)


# Harmonic hues for active pads (warm→cool spectrum)
PAD_HUES = [
    (0, 140, 255), (0, 160, 255), (0, 180, 255), (0, 200, 255),
    (0, 220, 255), (0, 240, 200), (0, 255, 150), (80, 255, 80),
    (140, 255, 0), (200, 240, 0), (240, 200, 0), (255, 160, 0),
    (255, 120, 0), (255, 80, 40), (255, 0, 80), (200, 0, 200),
    (160, 0, 255), (120, 0, 255), (80, 0, 255), (40, 40, 255),
    (0, 140, 255), (0, 160, 255), (0, 180, 255), (0, 200, 255),
    (0, 220, 255), (0, 240, 200), (0, 255, 150), (80, 255, 80),
    (140, 255, 0), (200, 240, 0), (240, 200, 0), (255, 160, 0),
]


def visible_people_map(persons):
    """Mapea teclas 1-8 a personas visibles ordenadas de izquierda a derecha."""
    visible = [p for p in persons if p.present]
    visible.sort(key=lambda p: (p.bbox[0], p.slot_id))
    by_key = {i + 1: p.slot_id for i, p in enumerate(visible)}
    display_number = {p.slot_id: i + 1 for i, p in enumerate(visible)}
    return by_key, display_number


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0",
                    help="índice de webcam o ruta de video")
    ap.add_argument("--seconds", type=float, default=None,
                    help="duración; indefinido por defecto")
    ap.add_argument("--warmup", type=float, default=10.0,
                    help="warmup excluido de métricas (plan r3 #12)")
    ap.add_argument("--record", default=None, help="grabar sesión a .jsonl")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--mode", default="group", choices=("group", "crowd"),
                    help="group: identidad sagrada (BoT-SORT+ReID+reasoc) | "
                         "crowd: masa, recall (imgsz alto, agregados)")
    ap.add_argument("--checkpoint", default=None,
                    help="Explicit local pose checkpoint for this run")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Override inference resolution (default from config)")
    ap.add_argument("--show", action="store_true",
                    help="ventana con esqueletos + selección de foco: teclas "
                         "1-N = persona visible (izq→der), 0/a = auto, q/ESC = salir")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    dests = [(args.host, args.port)] if args.host and args.port else None
    pipe = HarmocapPipeline(REPO, source=source, record_to=args.record,
                            osc_destinations=dests, mode=args.mode,
                            checkpoint=args.checkpoint, imgsz_override=args.imgsz,
                            camera_width=1280, camera_height=720)
    pipe.camera.start()
    print(f"[run] backend: {pipe.backend.info()}")
    print(f"[run] captura: {pipe.camera.profile()}")
    print(f"[run] stream_id={pipe.stream_id} contract_id={pipe.contract_id}")

    show = args.show
    fullscreen = False
    if show:
        import cv2
        cv2.namedWindow("HarMoCAP", cv2.WINDOW_NORMAL)
    t0 = time.monotonic()
    warmup_done = False
    try:
        deadline = (None if args.seconds is None
                    else t0 + args.warmup + args.seconds)
        while deadline is None or time.monotonic() < deadline:
            if not warmup_done and time.monotonic() - t0 >= args.warmup:
                pipe.metrics["lat_sw_ms"].clear()   # descartar warmup
                pipe.metrics["jitter_ms"].clear()
                warmup_done = True
            if not pipe.step():
                print("[run] fuente agotada")
                break
            if show and getattr(pipe, "last_frame_img", None) is not None:
                img = pipe.last_frame_img.copy()
                img = cv2.flip(img, 1)  # mirror
                h, w = img.shape[0], img.shape[1]
                cell_w = w // COLS
                cell_h = h // ROWS
                gap = 2

                # ── Compute active pads from focused-body wrists ──
                active_pads: set[int] = set()
                hand_pads: dict[str, int] = {}  # 'L'/'R' → pad 0..31
                for p in pipe.last_persons:
                    if not p.present or not p.focused:
                        continue
                    for kp_idx, label in ((9, "L"), (10, "R")):
                        kp = p.keypoints[kp_idx]
                        if kp.state != 2:
                            pid = pad_from_xy(kp.x, kp.y, w, h)
                            if pid is not None:
                                active_pads.add(pid)
                                hand_pads[label] = pid

                # ── 4×8 serpentine pad grid ──
                for col in range(COLS):
                    for grid_row in range(ROWS):
                        pid = pad_index(col, grid_row)
                        # grid_row 0 = bottom of model → renders at bottom of canvas
                        row = ROWS - 1 - grid_row
                        x0 = col * cell_w
                        y0 = row * cell_h
                        x1 = x0 + cell_w
                        y1 = y0 + cell_h
                        active = pid in active_pads
                        hue = PAD_HUES[pid]
                        # Fill
                        alpha = 0.30 if active else 0.06
                        roi = img[y0:y1, x0:x1]
                        rect = roi.copy()
                        rect[:, :] = hue if active else (30, 30, 40)
                        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, rect, alpha, 0)
                        # Border
                        border_col = hue if active else (80, 90, 110)
                        border_thick = 2 if active else 1
                        cv2.rectangle(img, (x0, y0), (x1, y1), border_col, border_thick)
                        # Label
                        label = f"H{pid + 1}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                        cv2.putText(img, label,
                                    (x0 + (cell_w - tw) // 2, y0 + (cell_h + th) // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255) if active else (160, 170, 190), 1)

                # ── Skeletons ──
                key_to_slot, display_number = visible_people_map(pipe.last_persons)
                for p in pipe.last_persons:
                    if not p.present:
                        continue
                    col = (0, 220, 255) if p.focused else (0, 180, 0)
                    points = {
                        i: (w - 1 - int(k.x * h), int(k.y * h))
                        for i, k in enumerate(p.keypoints)
                        if k.state != 2
                    }
                    for left, right in SKELETON_EDGES:
                        if left in points and right in points:
                            cv2.line(img, points[left], points[right], col, 3, cv2.LINE_AA)
                    for i, k in enumerate(p.keypoints):
                        if k.state != 2:
                            if i in (9, 10):
                                cv2.circle(img, points[i], 8, (0, 255, 200), -1)
                                cv2.circle(img, points[i], 10, (0, 255, 200), 2)
                            else:
                                cv2.circle(img, points[i], 4, col, -1)
                    # Wrist → pad labels
                    for kp_idx, label in ((9, "L"), (10, "R")):
                        kp = p.keypoints[kp_idx]
                        if kp.state != 2:
                            pid = pad_from_xy(kp.x, kp.y, w, h)
                            if pid is not None:
                                px = w - 1 - int(kp.x * h)
                                py = int(kp.y * h)
                                hue = PAD_HUES[pid]
                                cv2.putText(img, f"{label}→H{pid+1}",
                                            (px + 12, py - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, hue, 1)
                    xs = [point[0] for point in points.values()]
                    ys = [point[1] for point in points.values()]
                    if xs:
                        person_number = display_number[p.slot_id]
                        cv2.putText(img, f"P{person_number} / slot {p.slot_id}" +
                                    (" *FOCO*" if p.focused else ""),
                                    (int(min(xs)), max(14, int(min(ys)) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

                # ── Bottom bar ──
                focused_number = display_number.get(pipe.slots.focused_slot, "-")
                cv2.putText(img,
                            f"f=fullscreen | foco: {pipe.slots.focus_mode} (P{focused_number})"
                            f" [1-{len(key_to_slot)}]=persona 0/a=auto q=salir",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # ── Fullscreen toggle ──
                if fullscreen:
                    cv2.setWindowProperty("HarMoCAP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("HarMoCAP", img)
                if cv2.getWindowProperty("HarMoCAP", cv2.WND_PROP_VISIBLE) < 1:
                    show = False
                    continue
                key = cv2.waitKey(1) & 0xFF
                if key == ord("f"):
                    fullscreen = not fullscreen
                    if not fullscreen:
                        cv2.setWindowProperty("HarMoCAP", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                elif key in (ord("q"), 27):
                    break
                elif key in (ord("0"), ord("a")):
                    pipe.slots.select_auto()
                elif ord("1") <= key <= ord("8"):
                    slot = key_to_slot.get(key - ord("0"))
                    if slot is not None:
                        pipe.slots.select_focus(slot)
    except KeyboardInterrupt:
        print("\n[run] interrumpido")
    finally:
        if show:
            cv2.destroyAllWindows()
        report = pipe.report()
        pipe.close()

    print(json.dumps(report, indent=2, default=str))
    run_file = REPO / "reports" / "CURRENT_RUN"
    if run_file.exists():
        run_id = run_file.read_text().strip().split("=", 1)[1]
        out = REPO / "reports" / run_id / "realtime_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"[run] métricas → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
