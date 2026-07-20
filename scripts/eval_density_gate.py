#!/usr/bin/env python
"""H6-F0 — compuerta de decisión para el conteo por mapas de densidad.

NO integra nada: mide si un modelo de densidad (ZIP) es utilizable en NUESTRO
punto de operación antes de escribir código de producción. La investigación
advierte que acumulamos cuatro degradaciones (resolución bajo el codo de la
curva, cambio de dominio, cero ajuste fino, cabezas de ~10 px) que nadie midió
juntas: sin medir, no se distingue una señal de densidad de ruido caro.

Qué mide, sin anotación humana:
  1. **Acuerdo en régimen ralo**: donde las personas son grandes, la detección
     es confiable; la correlación y el sesgo entre densidad y detección ahí
     calibran el modelo sin ground truth.
  2. **Control cruzado entre checkpoints** (QNRF vs NWPU): si dos modelos
     entrenados en datasets distintos coinciden, hay señal; si divergen,
     estamos fuera de distribución y el número no significa nada.
  3. **Efecto de la resolución**: nativa vs reescalada al mínimo de
     entrenamiento, que la literatura señala como el factor dominante.
  4. **Latencia real** en la GPU de trabajo.

Uso: python scripts/eval_density_gate.py video1.mp4 [...] [--frames 60]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
ZIP_DIR = REPO / "outputs" / "zip_eval" / "ZIP"
sys.path.insert(0, str(ZIP_DIR))
sys.path.insert(0, str(REPO / "src"))

# El paquete `models` de ZIP importa la rama CLIP (open_clip, peft) al cargarse,
# aunque acá solo usamos la rama EBC/MobileNetV4. Se stubean esas dependencias:
# instalarlas traería el andamiaje de CLIP entero para código que no ejecutamos.
# Si alguna vez se evalúa CLIP-EBC como control, hay que instalarlas de verdad.
import types  # noqa: E402

def _stub(dep: str):
    """Módulo vacío que solo falla si se usa de verdad (los dunder pasan: la
    maquinaria de import e `inspect` los consultan al cargar el paquete)."""
    m = types.ModuleType(dep)
    m.__file__ = f"<stub {dep}>"
    m.__path__ = []                       # type: ignore[attr-defined]

    def _getattr(name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        # los módulos CLIP hacen `from peft import get_peft_model, LoraConfig`
        # a nivel de módulo: hay que devolver algo importable que solo explote
        # si de verdad se ejecuta la rama que no usamos.
        def _placeholder(*_a, **_k):
            raise RuntimeError(f"{dep}.{name} requerido: instalar {dep} de verdad")

        return _placeholder

    m.__getattr__ = _getattr              # type: ignore[attr-defined]
    return m


for _dep in ("open_clip", "peft"):
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except ImportError:
            sys.modules[_dep] = _stub(_dep)

CHECKPOINTS = {
    "qnrf_nano": REPO / "outputs/zip_eval/qnrf_n/ebc_n_best/best_mae.pth",
    "nwpu_nano": REPO / "outputs/zip_eval/nwpu_n/best_mae.pth",
}
TRAIN_MIN_EDGE = 672      # input_size con que se entrenaron los checkpoints


def load_zip(ckpt: Path, device: str = "cuda:0"):
    from models import get_model
    model = get_model(model_info_path=str(ckpt))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["weights"])
    return model.to(device).eval()


@torch.no_grad()
def density_count(model, frame_bgr, device: str = "cuda:0",
                  min_edge: int | None = None) -> tuple[float, np.ndarray, float]:
    """Devuelve (conteo, mapa de densidad, latencia_ms)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if min_edge:
        h, w = img.shape[:2]
        s = max(min_edge / min(h, w), 1.0)
        if s > 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32    # múltiplo del bloque
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    x = ((x - mean) / std).unsqueeze(0).to(device)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    out = model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    ms = (time.perf_counter() - t0) * 1000.0
    dm = out[0] if isinstance(out, (tuple, list)) else out
    dm = dm.squeeze().float().cpu().numpy()
    return float(dm.sum()), dm, ms


def yolo_counts(video: Path, idxs: list[int]) -> dict[int, tuple[int, float]]:
    """Conteo de detección y tamaño mediano de bbox (para separar regímenes)."""
    from harmocap.perception import PoseBackend
    be = PoseBackend(
        realtime_checkpoint=str(REPO / "outputs/harmocap-m-pose-ft2.engine"),
        fallback_checkpoint="harmocap-m-pose-ft2.pt", imgsz=1280, conf=0.05,
        max_det=300, tracker="bytetrack.yaml")
    cap = cv2.VideoCapture(str(video))
    out, i, want = {}, 0, set(idxs)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            _d, raw, _s, _wh = be.track_frame(frame)
            med_h = statistics.median([b[3] for b in raw]) if raw else 0.0
            out[i] = (len(raw), med_h)
        i += 1
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--frames", type=int, default=40, help="frames por video")
    args = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = {k: load_zip(p, dev) for k, p in CHECKPOINTS.items() if p.exists()}
    print(f"[gate] modelos: {list(models)} | device {dev}")

    report: dict = {}
    for video in args.videos:
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idxs = [int(total * (i + 0.5) / args.frames) for i in range(args.frames)]
        print(f"\n[gate] {video.name}: {args.frames} frames de {total}")
        yc = yolo_counts(video, idxs)

        rows = []
        cap = cv2.VideoCapture(str(video))
        i = 0
        want = set(idxs)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want:
                row = {"frame": i, "yolo_count": yc.get(i, (0, 0))[0],
                       "yolo_med_h": round(yc.get(i, (0, 0))[1], 4)}
                for name, m in models.items():
                    c_nat, _dm, ms_nat = density_count(m, frame, dev)
                    c_up, _dm2, ms_up = density_count(m, frame, dev, TRAIN_MIN_EDGE)
                    row[f"{name}_nativo"] = round(c_nat, 1)
                    row[f"{name}_upscale"] = round(c_up, 1)
                    row[f"{name}_ms_nativo"] = round(ms_nat, 1)
                    row[f"{name}_ms_upscale"] = round(ms_up, 1)
                rows.append(row)
            i += 1
        cap.release()

        # --- análisis ---
        ralo = [r for r in rows if r["yolo_med_h"] >= 0.25 and r["yolo_count"] > 0]
        denso = [r for r in rows if r["yolo_med_h"] < 0.12 and r["yolo_count"] > 0]
        res: dict = {"n_frames": len(rows), "n_ralo": len(ralo), "n_denso": len(denso)}
        for name in models:
            for var in ("nativo", "upscale"):
                key = f"{name}_{var}"
                if ralo:
                    d = [r[key] for r in ralo]
                    y = [r["yolo_count"] for r in ralo]
                    sesgo = statistics.median([a / b for a, b in zip(d, y) if b > 0])
                    corr = float(np.corrcoef(d, y)[0, 1]) if len(d) > 2 else float("nan")
                    res[f"{key}_ralo_sesgo_vs_yolo"] = round(sesgo, 2)
                    res[f"{key}_ralo_correlacion"] = round(corr, 3)
                if denso:
                    d = [r[key] for r in denso]
                    y = [r["yolo_count"] for r in denso]
                    res[f"{key}_denso_mediana_densidad"] = round(statistics.median(d), 1)
                    res[f"{key}_denso_mediana_yolo"] = round(statistics.median(y), 1)
                res[f"{key}_ms_p50"] = round(
                    statistics.median([r[f"{name}_ms_{var}"] for r in rows]), 1)
        # acuerdo entre checkpoints (control cruzado)
        if len(models) == 2:
            a, b = list(models)
            for var in ("nativo", "upscale"):
                xa = [r[f"{a}_{var}"] for r in rows]
                xb = [r[f"{b}_{var}"] for r in rows]
                res[f"acuerdo_{var}_correlacion"] = round(float(np.corrcoef(xa, xb)[0, 1]), 3)
                res[f"acuerdo_{var}_razon_mediana"] = round(
                    statistics.median([p / q for p, q in zip(xa, xb) if q > 0.5]), 2)
        report[video.name] = {"resumen": res, "frames": rows}
        for k, v in res.items():
            print(f"    {k}: {v}")

    dest = REPO / "reports" / "20260717_e71e14a" / "density_gate.json"
    dest.write_text(json.dumps(report, indent=2))
    print(f"\n[gate] → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
