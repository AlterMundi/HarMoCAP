"""H5 — tempo y fase: recuperación, robustez y semántica del contrato."""
import math

import pytest

from harmocap.tempo import BPM_MAX, BPM_MIN, TempoEstimator

FPS = 30.0


def _run(signal_fn, seconds=10.0, fps=FPS, te=None):
    te = te or TempoEstimator()
    out = (0.0, 0.0, 0.0)
    for i in range(int(seconds * fps)):
        t = i / fps
        out = te.update(signal_fn(t), int(t * 1e6))
    return out, te


@pytest.mark.parametrize("bpm", [48, 60, 90, 120, 144, 180, 210])
def test_recupera_bpm_de_senoide(bpm):
    """Error < 2% sobre todo el rango declarado."""
    hz = bpm / 60.0
    (got, _phase, conf), _ = _run(lambda t: 0.5 + 0.4 * math.sin(2 * math.pi * hz * t))
    assert abs(got - bpm) / bpm < 0.02, f"esperaba {bpm}, dio {got}"
    assert conf > 0.5


def test_qom_rectificado_da_tasa_de_eventos_no_frecuencia():
    """SEMÁNTICA DEL CONTRATO: qom es magnitud de velocidad, así que una
    oscilación de 1.2 Hz produce eventos a 2.4 Hz = 144 BPM, no 72."""
    (got, _p, _c), _ = _run(lambda t: abs(math.cos(2 * math.pi * 1.2 * t)) * 0.6 + 0.05)
    assert abs(got - 144.0) / 144.0 < 0.05, f"dio {got} (¿error de octava?)"


def test_sin_octava_baja_en_senoide_pura():
    """El submúltiplo también correlaciona: debe ganar el lag MÁS CORTO."""
    (got, _p, _c), _ = _run(lambda t: 0.5 + 0.4 * math.sin(2 * math.pi * 2.4 * t))
    assert got > 100.0, f"dio {got}: reportó un submúltiplo"


def test_cuerpo_quieto_no_declara_tempo():
    (bpm, phase, conf), _ = _run(lambda t: 0.5)
    assert bpm == 0.0 and phase == 0.0 and conf == 0.0


@pytest.mark.parametrize("seed", [1, 5, 11, 13, 17])
def test_ruido_no_declara_tempo(seed):
    """El piso de confianza debe estar por encima del máximo de la ACF del
    ruido: barrer ~40 lags eleva ese máximo a ~3σ y con un piso bajo aparecía
    tempo inventado (seeds 5, 11 y 13 lo disparaban con piso 0.15)."""
    import random
    random.seed(seed)
    (bpm, _p, conf), _ = _run(lambda t: random.random())
    assert bpm == 0.0, f"declaró {bpm} BPM sobre ruido (conf {conf})"


def test_historia_insuficiente_no_declara_tempo():
    """Antes de llenar la ventana no hay tempo: el consumidor lo ve invalid."""
    (bpm, _p, _c), _ = _run(lambda t: 0.5 + 0.4 * math.sin(2 * math.pi * 2.0 * t),
                            seconds=2.0)
    assert bpm == 0.0


def test_fase_es_rampa_monotona_y_envuelve():
    """La fase se integra: avance uniforme ≈ f/fps por frame, sin saltos."""
    te = TempoEstimator()
    hz, phases = 2.0, []
    for i in range(int(12 * FPS)):
        t = i / FPS
        bpm, ph, _ = te.update(0.5 + 0.4 * math.sin(2 * math.pi * hz * t), int(t * 1e6))
        if i > 10 * FPS:
            phases.append(ph)
    deltas = [(phases[i + 1] - phases[i]) % 1.0 for i in range(len(phases) - 1)]
    esperado = hz / FPS
    assert all(0.0 < d < 0.5 for d in deltas), "la fase retrocede o salta"
    # tolerancia amplia: en las re-estimaciones el lazo corrige la fase
    assert abs(sum(deltas) / len(deltas) - esperado) < 0.02


def test_rango_declarado_se_respeta():
    """Nunca se emite BPM fuera del rango del contrato."""
    for hz in (0.2, 0.5, 5.0, 8.0):    # 12, 30, 300, 480 BPM: fuera de rango
        (bpm, _p, _c), _ = _run(lambda t: 0.5 + 0.4 * math.sin(2 * math.pi * hz * t))
        assert bpm == 0.0 or BPM_MIN <= bpm <= BPM_MAX, f"{hz*60} BPM → {bpm}"


def test_reset_borra_la_historia():
    _out, te = _run(lambda t: 0.5 + 0.4 * math.sin(2 * math.pi * 2.0 * t))
    assert te.update(0.5, int(11 * 1e6))[0] > 0.0     # todavía sabe
    te.reset()
    assert te.update(0.5, int(12 * 1e6)) == (0.0, 0.0, 0.0)


def test_tolera_jitter_de_captura():
    """Los t_us reales no son uniformes; el remuestreo debe absorberlo."""
    import random
    random.seed(3)
    te = TempoEstimator()
    hz, t = 2.0, 0.0
    out = (0.0, 0.0, 0.0)
    for _ in range(300):
        t += (1.0 / FPS) * random.uniform(0.7, 1.3)     # ±30% de jitter
        out = te.update(0.5 + 0.4 * math.sin(2 * math.pi * hz * t), int(t * 1e6))
    assert abs(out[0] - 120.0) / 120.0 < 0.05, f"con jitter dio {out[0]}"


def test_features_expone_tempo_con_estado_invalid_al_principio():
    """Integración: las 3 features viajan y arrancan invalid (sin historia)."""
    from harmocap.features import CalibrationManager, FeatureExtractor
    from harmocap.schema import FEATURE_ORDER, N_FEATURES

    assert N_FEATURES == 24
    for name in ("tempo_bpm", "beat_phase", "tempo_conf"):
        assert name in FEATURE_ORDER

    calib = CalibrationManager({"torso_height_norm": 0.28, "vmax_hand": 3.0,
                                "vmax_center": 1.5, "jerk_ref": 40.0,
                                "energy_ref": 4.0, "accel_ref": 8.0})
    fx = FeatureExtractor({"velocity_ms": 120, "accel_ms": 200, "jerk_ms": 300,
                           "aggregate_ms": 1000, "qom_ms": 400}, calib)
    # (x, y, conf, estado, age_frames, age_us) — salida del KeypointSmoother
    kps = [(0.5 + 0.01 * i, 0.2 + 0.03 * i, 0.9, 0, 0, 0) for i in range(17)]
    vals, states = fx.extract(kps, 1_000_000)
    assert len(vals) == 24 and len(states) == 24
    i = FEATURE_ORDER.index("tempo_bpm")
    assert states[i] == 2 and vals[i] == 0.0    # invalid + sentinel
