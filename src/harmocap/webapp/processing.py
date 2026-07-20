"""Núcleo de procesamiento de la interfaz web — una sola pasada por el video.

Corre el pipeline REAL de features (percepción → identidad → suavizado →
features + multitud/densidad), dibuja los overlays elegidos y, en la misma
pasada, graba la sesión a .jsonl y acumula las variables para exportar a CSV y
graficar. Reutiliza los mismos componentes que el pipeline en vivo, así que lo
que se exporta son las variables del contrato, no detecciones crudas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from harmocap.crowd import CrowdAggregator
from harmocap.features import CalibrationManager, FeatureExtractor
from harmocap.identity import SlotManager
from harmocap.perception import PoseBackend, resolve_device
from harmocap.schema import CROWD_FIELDS, FEATURE_ORDER, KpState
from harmocap.smoothing import KeypointSmoother

REPO = Path(__file__).resolve().parent.parent.parent.parent

PALETTE = [(66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
           (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
         (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16))
KPT_CONF = 0.35

# opciones de overlay que la UI ofrece
OVERLAYS = ("points", "skeleton", "bbox", "ids", "homunculus", "density")
OVERLAY_LABELS = {
    "points": "Puntos", "skeleton": "Esqueleto (líneas)", "bbox": "Caja (bbox)",
    "ids": "Número de identidad", "homunculus": "Silueta (homúnculo)",
    "density": "Mapa de densidad (masa)",
}


@dataclass
class ProcessResult:
    video_path: Path            # mp4 con overlay
    jsonl_path: Path            # sesión grabada
    frames: int
    seconds: float
    fps_proc: float
    device: str
    mode: str
    feature_series: dict        # {feature: [(t_s, valor por slot presente)]}
    crowd_series: dict          # {campo: [(t_s, valor)]}


def hardware_info() -> dict:
    dev = resolve_device("auto")
    label = {"cuda:0": "Placa NVIDIA (rápido)",
             "mps": "Apple Silicon (moderado)",
             "cpu": "Procesador, sin placa (lento)"}.get(dev, dev)
    return {"device": dev, "label": label}


def _load_cfgs():
    def cfg(n):
        return yaml.safe_load((REPO / "configs" / f"{n}.yaml").read_text())
    return cfg("model"), cfg("smoothing"), cfg("features"), cfg("identity")


def _build(mode: str, want_density: bool):
    cfg_model, cfg_smooth, cfg_feat, cfg_ident = _load_cfgs()
    mode_cfg = yaml.safe_load((REPO / "configs" / "modes" / f"{mode}.yaml").read_text())
    ov = mode_cfg.get("overrides", {})
    m = dict(cfg_model["model"])
    if "model" in ov:
        m.update({k: v for k, v in ov["model"].items() if k != "imgsz_fallback"})
    dev = resolve_device(m["device"])
    eff_imgsz = m["imgsz"]
    if ov.get("model", {}).get("imgsz_fallback") and not dev.startswith("cuda"):
        eff_imgsz = ov["model"]["imgsz_fallback"]
    tracker = ov.get("tracker", {}).get("name", cfg_model["tracker"]["name"])
    if "/" in tracker:
        tracker = str(REPO / tracker)
    backend = PoseBackend(
        realtime_checkpoint=str(REPO / m["realtime_checkpoint"]),
        fallback_checkpoint=m["fallback_checkpoint"], device=m["device"],
        imgsz=eff_imgsz, conf=m["conf"], max_det=m["max_det"], tracker=tracker)
    reacq = dict(cfg_ident.get("reacquisition", {}))
    reacq["enabled"] = ov.get("identity", {}).get(
        "reacquisition_enabled", reacq.get("enabled", True))
    slots = SlotManager(
        max_slots=cfg_ident["slot"].get("max_slots", 8),
        occlusion_grace_ms=cfg_ident["slot"]["occlusion_grace_ms"],
        release_timeout_ms=cfg_ident["slot"]["release_timeout_ms"],
        acquire_rule=cfg_ident["slot"]["acquire_rule"],
        auto_focus_switch_ratio=cfg_ident["slot"].get("auto_focus_switch_ratio", 1.2),
        tombstone_repeat_frames=cfg_ident["slot"]["tombstone_repeat_frames"],
        reacquisition=reacq)
    calib = CalibrationManager(cfg_feat["calibration"]["fallback"],
                               period_ms=cfg_feat["calibration"]["period_ms"])
    crowd = CrowdAggregator()

    density = density_crowd = None
    dcfg = ov.get("density", {})
    if (want_density or dcfg.get("enabled")):
        onnx = REPO / dcfg.get("onnx", "outputs/density/zip_qnrf_n.onnx")
        if onnx.is_file():
            from harmocap.density import DensityBackend
            from harmocap.density_crowd import DensityCrowdAggregator
            density = DensityBackend(onnx, min_edge=dcfg.get("min_edge", 448))
            density_crowd = DensityCrowdAggregator()
    return (backend, slots, calib, crowd, density, density_crowd,
            cfg_smooth, cfg_feat, dev)


def _iso_to_px(kps_iso, h):
    """Isotrópico (x/h, y/h) → píxeles. Devuelve [(x_px, y_px, conf)]."""
    return [(k[0] * h, k[1] * h, k[2]) for k in kps_iso]


def _draw_person(frame, slot_id, det, overlays, w, h, focused):
    col = PALETTE[slot_id % 8]
    kpx = _iso_to_px(det.keypoints_iso, h)
    pts = {i: (int(x), int(y)) for i, (x, y, c) in enumerate(kpx) if c >= KPT_CONF}
    if "homunculus" in overlays and len(pts) >= 3:
        hull = cv2.convexHull(np.array(list(pts.values()), np.int32))
        ov = frame.copy()
        cv2.fillConvexPoly(ov, hull, col)
        cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
        cv2.polylines(frame, [hull], True, col, 2, cv2.LINE_AA)
    if "skeleton" in overlays:
        for a, b in EDGES:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], col, 3, cv2.LINE_AA)
    if "points" in overlays or "skeleton" in overlays:
        for p in pts.values():
            cv2.circle(frame, p, 5, col, -1, cv2.LINE_AA)
            cv2.circle(frame, p, 5, (255, 255, 255), 1, cv2.LINE_AA)
    cx, cy, bw, bh = det.bbox_xywhn
    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
    if "bbox" in overlays:
        cv2.rectangle(frame, (x1, y1), (x2, y2), col,
                      4 if focused else 2)
    if "ids" in overlays:
        tag = f"P{slot_id}" + (" *" if focused else "")
        cv2.putText(frame, tag, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)


class StreamProcessor:
    """Procesa la webcam en vivo, un cuadro por llamada, manteniendo el estado
    del pipeline entre cuadros (identidad, suavizado, tempo, multitud).

    Uso: `annotated_bgr, readout = sp.step(frame_bgr)`. El tiempo avanza con el
    reloj real, así las features time-aware (velocidades, tempo) son correctas
    aunque la tasa de cuadros del stream sea irregular.
    """

    # features que se muestran en vivo (las más legibles de un vistazo)
    LIVE_FEATS = ("qom", "expansion", "contraction", "verticality",
                  "vel_center", "tempo_bpm", "tempo_conf")

    def __init__(self, mode: str = "group", overlays: list[str] | None = None):
        import time
        self._time = time.monotonic
        overlays = overlays or ["skeleton", "bbox", "ids"]
        self.overlays = set(overlays)
        self.mode = mode
        want_density = "density" in self.overlays or mode == "crowd"
        (self.backend, self.slots, self.calib, self.crowd, self.density,
         self.density_crowd, cfg_smooth, self.cfg_feat, self.device) = _build(
            mode, want_density)
        self.oe, self.ks = cfg_smooth["one_euro"], cfg_smooth["keypoint_state"]
        self.smoothers: dict[int, KeypointSmoother] = {}
        self.extractors: dict[int, FeatureExtractor] = {}
        self._t0 = self._time()
        self._density_i = 0
        self._last_mass = {"mass_present": 0.0, "mass_active": 0.0}

    def step(self, frame_bgr):
        t_us = int((self._time() - self._t0) * 1e6)
        h, w = frame_bgr.shape[:2]
        aspect = w / h if h else 16 / 9
        dets, raw, _sp, _wh = self.backend.track_frame(frame_bgr)

        if self.density is not None and ("density" in self.overlays
                                         or self.mode == "crowd"):
            if self._density_i % 5 == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                dmap = self.density.infer(rgb)
                dc = self.density_crowd.update(dmap, rgb, t_us)
                self._last_mass = {"mass_present": dc["mass_present"],
                                   "mass_active": dc["mass_active"]}
                if "density" in self.overlays:
                    hm = cv2.resize(dmap, (w, h))
                    hm = np.clip(hm / (hm.max() + 1e-9), 0, 1)
                    hm = cv2.applyColorMap((hm * 255).astype(np.uint8),
                                           cv2.COLORMAP_JET)
                    frame_bgr = cv2.addWeighted(frame_bgr, 0.6, hm, 0.4, 0)
            self._density_i += 1

        events = self.slots.update(dets, t_us, aspect=aspect)
        focused = self.slots.focused_slot
        crowd_out = self.crowd.update(raw, t_us, aspect=aspect)
        crowd_out.update(self._last_mass)

        focal_readout = {}
        n_present = 0
        for ev in events:
            if ev.detection is None:
                continue
            n_present += 1
            sid = ev.slot_id
            if ev.slot_reset or sid not in self.smoothers:
                self.smoothers[sid] = KeypointSmoother(
                    mincutoff=self.oe["mincutoff"], beta=self.oe["beta"],
                    dcutoff=self.oe["dcutoff"],
                    conf_threshold=self.ks["conf_threshold"],
                    held_timeout_ms=self.ks["held_timeout_ms"],
                    conf_decay_per_s=self.ks["conf_decay_per_s"])
                self.extractors[sid] = FeatureExtractor(
                    self.cfg_feat["windows"], self.calib)
            sm = self.smoothers[sid].update(ev.detection.keypoints_iso, t_us)
            vals, states = self.extractors[sid].extract(sm, t_us)
            _draw_person(frame_bgr, sid, ev.detection, self.overlays, w, h,
                         sid == focused)
            if sid == focused:
                idx = {f: i for i, f in enumerate(FEATURE_ORDER)}
                for f in self.LIVE_FEATS:
                    i = idx[f]
                    focal_readout[f] = (None if states[i] == int(KpState.INVALID)
                                        else round(vals[i], 3))

        readout = {"personas": n_present, "focal": focused,
                   "focal_features": focal_readout,
                   "multitud": {"crowd_count": crowd_out["crowd_count"],
                                "crowd_qom": crowd_out["crowd_qom"],
                                "mass_present": crowd_out["mass_present"],
                                "mass_active": crowd_out["mass_active"]}}
        return frame_bgr, readout


def process_video(video: str | Path, mode: str, overlays: list[str],
                  out_dir: str | Path, progress=None) -> ProcessResult:
    """Procesa el video en una pasada. `progress`: callable(frac, texto) o None."""
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays = set(overlays)
    want_density = "density" in overlays or mode == "crowd"

    (backend, slots, calib, crowd, density, density_crowd,
     cfg_smooth, cfg_feat, dev) = _build(mode, want_density)
    oe, ks = cfg_smooth["one_euro"], cfg_smooth["keypoint_state"]
    smoothers: dict[int, KeypointSmoother] = {}
    extractors: dict[int, FeatureExtractor] = {}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out_video = out_dir / f"{video.stem}__{mode}.mp4"
    vw = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    jsonl = out_dir / f"{video.stem}__{mode}.jsonl"
    fj = jsonl.open("w", encoding="utf-8")

    feat_series = {f: [] for f in FEATURE_ORDER}
    crowd_series = {c: [] for c in CROWD_FIELDS}
    import time
    t0, t_us, frames = time.time(), 0, 0
    density_i, last_mass = 0, {"mass_present": 0.0, "mass_active": 0.0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t_us += int(1e6 / fps)
        t_s = t_us / 1e6
        dets, raw, _sp, _wh = backend.track_frame(frame)

        # densidad primero (overlay de fondo), con stride
        if density is not None and ("density" in overlays or mode == "crowd"):
            if density_i % 5 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dmap = density.infer(rgb)
                dc = density_crowd.update(dmap, rgb, t_us)
                last_mass = {"mass_present": dc["mass_present"],
                             "mass_active": dc["mass_active"]}
                if "density" in overlays:
                    hm = cv2.resize(dmap, (w, h))
                    hm = np.clip(hm / (hm.max() + 1e-9), 0, 1)
                    hm = cv2.applyColorMap((hm * 255).astype(np.uint8),
                                           cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(frame, 0.6, hm, 0.4, 0)
            density_i += 1

        events = slots.update(dets, t_us, aspect=w / h if h else 16 / 9)
        focused = slots.focused_slot
        crowd_out = crowd.update(raw, t_us, aspect=w / h if h else 16 / 9)
        crowd_out.update(last_mass)

        frame_persons = []
        for ev in events:
            if ev.detection is None:
                continue
            sid = ev.slot_id
            if ev.slot_reset or sid not in smoothers:
                smoothers[sid] = KeypointSmoother(
                    mincutoff=oe["mincutoff"], beta=oe["beta"],
                    dcutoff=oe["dcutoff"], conf_threshold=ks["conf_threshold"],
                    held_timeout_ms=ks["held_timeout_ms"],
                    conf_decay_per_s=ks["conf_decay_per_s"])
                extractors[sid] = FeatureExtractor(cfg_feat["windows"], calib)
            sm = smoothers[sid].update(ev.detection.keypoints_iso, t_us)
            vals, states = extractors[sid].extract(sm, t_us)
            self_present = {"slot": sid, "focused": sid == focused,
                            "features": vals, "feature_states": states}
            frame_persons.append(self_present)
            _draw_person(frame, sid, ev.detection, overlays, w, h,
                         sid == focused)
            for fi, fname in enumerate(FEATURE_ORDER):
                if states[fi] != int(KpState.INVALID):
                    feat_series[fname].append((round(t_s, 3), sid, vals[fi]))

        for ci, cname in enumerate(CROWD_FIELDS):
            crowd_series[cname].append((round(t_s, 3), crowd_out[cname]))

        fj.write(json.dumps({"t_s": round(t_s, 3), "frame": frames,
                             "persons": frame_persons, "crowd": crowd_out}) + "\n")
        vw.write(frame)
        if progress and frames % 10 == 0:
            progress(frames / total, f"Procesando… {frames}/{total} cuadros")

    cap.release(); vw.release(); fj.close()
    dt = max(time.time() - t0, 1e-6)
    return ProcessResult(
        video_path=out_video, jsonl_path=jsonl, frames=frames,
        seconds=round(dt, 1), fps_proc=round(frames / dt, 1), device=dev,
        mode=mode, feature_series=feat_series, crowd_series=crowd_series)
