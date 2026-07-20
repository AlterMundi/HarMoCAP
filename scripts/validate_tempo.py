#!/usr/bin/env python
"""H5 — validación del tempo corporal contra el BPM de la música (ground truth).

Primer contraste del proyecto contra una referencia externa real: el BPM del
audio del mismo video. El criterio NO es igualdad — la gente se mueve a la
mitad, al doble o al triple del pulso musical — sino que la razón entre tempo
corporal y tempo musical sea una razón simple (1/3, 1/2, 1, 2, 3).

Requiere librosa (dependencia SOLO de validación, no de runtime).
Uso: python scripts/validate_tempo.py video1.mp4 [video2.mp4 …]
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from harmocap.features import CalibrationManager, FeatureExtractor  # noqa: E402
from harmocap.identity import SlotManager  # noqa: E402
from harmocap.perception import PoseBackend  # noqa: E402
from harmocap.schema import FEATURE_ORDER  # noqa: E402
from harmocap.smoothing import KeypointSmoother  # noqa: E402

I_BPM = FEATURE_ORDER.index("tempo_bpm")
I_CONF = FEATURE_ORDER.index("tempo_conf")
RATIOS = (1 / 3, 1 / 2, 1.0, 2.0, 3.0)


def music_bpm(video: Path) -> float | None:
    """BPM de la pista de audio del video (librosa)."""
    try:
        import librosa
    except ImportError:
        print("  (librosa no instalado: sin referencia de audio)")
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                        "-ac", "1", "-ar", "22050", wav], check=True)
        y, sr = librosa.load(wav, sr=22050)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo if not hasattr(tempo, "__len__") else tempo[0])
    except Exception as e:                       # audio ausente o corrupto
        print(f"  (sin BPM de audio: {e})")
        return None
    finally:
        Path(wav).unlink(missing_ok=True)


def body_tempo(video: Path, max_seconds: float = 60.0) -> dict:
    """Tempo corporal por slot sobre el video (modo grupo)."""
    cfgs = yaml.safe_load((REPO / "configs/smoothing.yaml").read_text())
    feat_cfg = yaml.safe_load((REPO / "configs/features.yaml").read_text())
    ident = yaml.safe_load((REPO / "configs/identity.yaml").read_text())
    backend = PoseBackend(
        realtime_checkpoint=str(REPO / "outputs/harmocap-m-pose-ft2.engine"),
        fallback_checkpoint="harmocap-m-pose-ft2.pt", imgsz=640, conf=0.05,
        max_det=300, tracker=str(REPO / "configs/tracker_group.yaml"))
    slots = SlotManager(
        max_slots=8,
        occlusion_grace_ms=ident["slot"]["occlusion_grace_ms"],
        release_timeout_ms=ident["slot"]["release_timeout_ms"],
        acquire_rule=ident["slot"]["acquire_rule"],
        tombstone_repeat_frames=ident["slot"]["tombstone_repeat_frames"],
        reacquisition=dict(ident.get("reacquisition", {}), enabled=True))
    calib = CalibrationManager(feat_cfg["calibration"]["fallback"],
                               period_ms=feat_cfg["calibration"]["period_ms"])
    oe, ks = cfgs["one_euro"], cfgs["keypoint_state"]
    smoothers: dict[int, KeypointSmoother] = {}
    extractors: dict[int, FeatureExtractor] = {}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    per_slot: dict[int, list[float]] = {}
    t_us, frames = 0, 0
    while frames < max_seconds * fps:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t_us += int(1e6 / fps)
        dets, _raw, _sp, (w, h) = backend.track_frame(frame)
        for ev in slots.update(dets, t_us, aspect=w / h if h else 16 / 9):
            if ev.detection is None:
                continue
            sid = ev.slot_id
            if ev.slot_reset or sid not in smoothers:
                smoothers[sid] = KeypointSmoother(
                    mincutoff=oe["mincutoff"], beta=oe["beta"],
                    dcutoff=oe["dcutoff"],
                    conf_threshold=ks["conf_threshold"],
                    held_timeout_ms=ks["held_timeout_ms"],
                    conf_decay_per_s=ks["conf_decay_per_s"])
                extractors[sid] = FeatureExtractor(feat_cfg["windows"], calib)
            sm = smoothers[sid].update(ev.detection.keypoints_iso, t_us)
            vals, states = extractors[sid].extract(sm, t_us)
            if states[I_BPM] != 2 and vals[I_CONF] >= 0.5:
                per_slot.setdefault(sid, []).append(vals[I_BPM])
    cap.release()
    return {sid: {"mediana_bpm": round(statistics.median(v), 1),
                  "muestras": len(v),
                  "estabilidad": round(statistics.pstdev(v), 1) if len(v) > 1 else 0.0}
            for sid, v in per_slot.items() if len(v) >= 30}


def closest_ratio(body: float, music: float) -> tuple[float, float]:
    """Razón simple más cercana y su error relativo."""
    best, err = None, 1e9
    for r in RATIOS:
        e = abs(body - music * r) / (music * r)
        if e < err:
            best, err = r, e
    return best, err


def main() -> int:
    videos = [Path(a) for a in sys.argv[1:]]
    assert videos, "uso: validate_tempo.py <video> [...]"
    out = {}
    for v in videos:
        print(f"\n[tempo] {v.name}")
        m = music_bpm(v)
        print(f"  música: {m:.1f} BPM" if m else "  música: n/d")
        b = body_tempo(v)
        rows = []
        for sid, st in sorted(b.items()):
            line = f"  slot {sid}: {st['mediana_bpm']:.1f} BPM (±{st['estabilidad']:.1f}, n={st['muestras']})"
            if m:
                r, err = closest_ratio(st["mediana_bpm"], m)
                st["razon_vs_musica"] = r
                st["error_rel"] = round(err, 3)
                line += f"  → {r:g}× música, error {err*100:.0f}%"
            print(line)
            rows.append(st)
        out[v.name] = {"music_bpm": m, "slots": b}
    dest = REPO / "reports" / "20260717_e71e14a" / "tempo_validation.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[tempo] → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
