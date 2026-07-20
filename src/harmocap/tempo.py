"""Tempo y fase del movimiento (H5, contrato 1.3) — estimación CAUSAL.

Qué mide
--------
La **tasa de eventos de movimiento**, no la frecuencia de oscilación del cuerpo.
La señal de entrada es la magnitud de velocidad (`qom`), y una oscilación de
1,2 Hz produce picos de velocidad a 2,4 Hz: cada medio ciclo es un evento
(una pisada, un rebote). Musicalmente esa es la lectura correcta, pero el
consumidor debe saberlo — está declarado en el contrato.

Cómo
----
Autocorrelación normalizada sobre una ventana trailing, remuestreada a grilla
uniforme (los tiempos de captura tienen jitter y la autocorrelación necesita
paso constante). Solo mira hacia atrás: nunca usa muestras futuras.

El BPM se re-estima a ~6 Hz (el tempo no cambia rápido y la autocorrelación es
lo caro). La **fase es un acumulador**: avanza cada frame integrando el tempo
vigente, y en cada re-estimación se corrige hacia la fase medida por el camino
angular corto y con ganancia baja, a la manera de un lazo de fase. Medirla por
correlación en cada frame la volvía irregular (±30 % de wobble en la velocidad
instantánea); integrada, el consumidor recibe una rampa monótona apta para
sincronizar audio.

Salidas
-------
tempo_bpm    BPM crudo en [0, 240]; 0 = desconocido (estado invalid)
beat_phase   [0,1) fase dentro del período actual, continua entre frames
tempo_conf   [0,1] altura del pico de autocorrelación = cuán periódica es la señal
"""
from __future__ import annotations

import math
from collections import deque

BPM_MIN = 40.0
BPM_MAX = 240.0
BPM_RANGE = (0.0, BPM_MAX)     # 0 reservado para "desconocido"

_WINDOW_S = 6.0                # historia para la autocorrelación
_GRID_HZ = 30.0                # grilla de remuestreo
_ESTIMATE_EVERY_S = 1.0 / 6.0  # re-estimación de BPM a ~6 Hz
_MIN_FILL = 0.75               # fracción mínima de ventana llena para estimar
_MIN_STD = 1e-3                # varianza mínima: cuerpo quieto → sin tempo
_CONF_FLOOR = 0.30             # piso de confianza. Con n≈180 muestras la ACF del
                               # ruido blanco tiene σ≈1/√n≈0.075, y barrer ~40 lags
                               # eleva el máximo esperado a ~3σ: un piso de 0.15
                               # dejaba pasar tempo espurio sobre ruido puro. 0.30
                               # ≈4σ y las señales periódicas reales dan 0.8-0.9.
_CONTINUITY_TOL = 0.90         # un pico ≥90% del máximo cerca del BPM previo gana
_CONTINUITY_CENTS = 0.12       # "cerca" = ±12% en BPM
_HARMONIC_TOL = 0.85           # pico "comparable al máximo" para elegir el lag corto
_DETREND_S = 1.0               # ventana de la media móvil que se resta (pasaalto)
_MEDIAN_S = 3.0                # ventana de la mediana móvil de BPM
_MEDIAN_MIN = 5                # estimaciones mínimas para declarar tempo
_PHASE_LOCK_GAIN = 0.15        # ganancia del enganche de fase (0=libre, 1=salto):
                               # engancha en ~1 s y deja correcciones imperceptibles


class TempoEstimator:
    """Estimador causal de tempo/fase sobre una señal escalar de movimiento.

    Uso: `update(value, t_us)` por frame → (bpm, phase, conf). Con historia
    insuficiente o señal no periódica devuelve (0.0, 0.0, conf) y el llamador
    marca la feature como invalid.
    """

    def __init__(self, window_s: float = _WINDOW_S, grid_hz: float = _GRID_HZ):
        self.window_s = window_s
        self.grid_hz = grid_hz
        self._buf: deque[tuple[int, float]] = deque()   # (t_us, valor) crudo
        self._bpm = 0.0
        self._conf = 0.0
        self._last_estimate_us: int | None = None
        self._last_frame_us: int | None = None
        self._recent: deque[tuple[int, float]] = deque()
        self._phase = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._bpm = 0.0
        self._conf = 0.0
        self._last_estimate_us = None
        self._last_frame_us = None
        self._recent.clear()
        self._phase = 0.0

    # ------------------------------------------------------------------ API
    def update(self, value: float, t_us: int) -> tuple[float, float, float]:
        self._buf.append((t_us, float(value)))
        win_us = self.window_s * 1e6
        while self._buf and t_us - self._buf[0][0] > win_us:
            self._buf.popleft()

        # 1) la fase avanza SIEMPRE por integración del tempo vigente: es un
        #    acumulador, no una medición por frame. Así el consumidor recibe una
        #    rampa monótona y suave, que es lo que un motor de audio necesita.
        if self._bpm > 0.0 and self._last_frame_us is not None:
            dt = (t_us - self._last_frame_us) / 1e6
            if dt > 0:
                self._phase = (self._phase + (self._bpm / 60.0) * dt) % 1.0
        self._last_frame_us = t_us

        # 2) el tempo se re-estima a ~6 Hz; ahí también se corrige la fase hacia
        #    la medida, por el camino angular corto y con ganancia baja (lazo de
        #    fase): engancha en ~1 s sin producir saltos audibles.
        due = (self._last_estimate_us is None
               or (t_us - self._last_estimate_us) / 1e6 >= _ESTIMATE_EVERY_S)
        if due:
            self._last_estimate_us = t_us
            raw_bpm, self._conf = self._estimate(t_us)
            # el tempo varía lento: una mediana móvil de las estimaciones
            # recientes recorta la dispersión (medida en ±30 BPM cuadro a
            # cuadro sobre video real) sin agregar latencia perceptible
            if raw_bpm > 0.0:
                self._recent.append((t_us, raw_bpm))
            while self._recent and (t_us - self._recent[0][0]) / 1e6 > _MEDIAN_S:
                self._recent.popleft()
            vals = sorted(b for _t, b in self._recent)
            self._bpm = vals[len(vals) // 2] if len(vals) >= _MEDIAN_MIN else 0.0
            if self._bpm > 0.0:
                measured = self._phase_at(t_us)
                err = (measured - self._phase + 0.5) % 1.0 - 0.5   # ∈ [-0.5, 0.5)
                self._phase = (self._phase + _PHASE_LOCK_GAIN * err) % 1.0
            else:
                self._phase = 0.0
        return self._bpm, self._phase, self._conf

    # -------------------------------------------------------------- interno
    def _grid(self, t_us: int) -> list[float] | None:
        """Remuestreo lineal causal de la ventana a paso fijo. None si falta historia."""
        if len(self._buf) < 8:
            return None
        t0 = self._buf[0][0]
        span_s = (t_us - t0) / 1e6
        if span_s < self.window_s * _MIN_FILL:
            return None
        n = int(span_s * self.grid_hz)
        if n < 16:
            return None
        step_us = 1e6 / self.grid_hz
        out: list[float] = []
        j = 0
        items = list(self._buf)
        for k in range(n):
            tk = t0 + k * step_us
            while j + 1 < len(items) and items[j + 1][0] < tk:
                j += 1
            if j + 1 >= len(items):
                out.append(items[-1][1])
                continue
            (ta, va), (tb, vb) = items[j], items[j + 1]
            f = 0.0 if tb == ta else (tk - ta) / (tb - ta)
            out.append(va + (vb - va) * max(0.0, min(1.0, f)))
        return out

    def _estimate(self, t_us: int) -> tuple[float, float]:
        xs = self._grid(t_us)
        if xs is None:
            return 0.0, 0.0
        n = len(xs)
        # QUITA DE TENDENCIA: restar la media de toda la ventana no alcanza —
        # una deriva lenta (la persona que se desplaza, el encuadre que cambia)
        # domina la autocorrelación en los lags largos y tapa el pulso. Se resta
        # una media móvil corta (~1 s), que es un pasaalto por debajo de 60 BPM:
        # medido, sin esto los slots con deriva daban confianza alta y ningún
        # tempo declarable.
        w = max(3, int(self.grid_hz * _DETREND_S))
        acc, trend = 0.0, []
        for i, x in enumerate(xs):
            acc += x
            if i >= w:
                acc -= xs[i - w]
            trend.append(acc / min(i + 1, w))
        xs = [x - m for x, m in zip(xs, trend)]
        var = sum(x * x for x in xs) / n
        if var <= _MIN_STD * _MIN_STD:
            return 0.0, 0.0        # cuerpo quieto: no hay tempo que declarar

        # ceil/floor, no truncamiento: con int() el lag mínimo daba 7, que a
        # 30 Hz equivale a 257 BPM — fuera del rango declarado. La regla del
        # lag más corto lo elegía y la estimación se descartaba en silencio,
        # produciendo "confianza alta y ningún tempo" (medido en video real).
        lag_min = max(2, math.ceil(self.grid_hz * 60.0 / BPM_MAX))
        lag_max = min(n - 4, math.floor(self.grid_hz * 60.0 / BPM_MIN))
        if lag_max <= lag_min:
            return 0.0, 0.0

        denom = var * n
        max_r = 0.0
        prev_bpm = self._bpm
        acf: list[tuple[int, float]] = []
        for lag in range(lag_min, lag_max + 1):
            s = 0.0
            for i in range(n - lag):
                s += xs[i] * xs[i + lag]
            # normalización SESGADA (divide por n, sin corregir solapamiento):
            # la corrección insesgada n/(n-lag) amplifica los lags largos y
            # empuja sistemáticamente a errores de octava hacia abajo — para una
            # senoide pura el lag 2T ganaba sobre el fundamental T.
            r = s / denom
            acf.append((lag, r))
            if r > max_r:
                max_r = r

        # Solo se consideran MÁXIMOS LOCALES. Tomar cualquier lag por encima de
        # un umbral funciona con una senoide limpia, pero en señal real (jitter
        # de keypoints, banda ancha) los lags cortos conservan correlación
        # residual y la regla elegía siempre el lag mínimo: el estimador
        # reportaba 225 BPM constantes sobre cualquier video.
        peaks = [(lag, r) for i, (lag, r) in enumerate(acf)
                 if 0 < i < len(acf) - 1 and r > acf[i - 1][1] and r >= acf[i + 1][1]]
        if not peaks:
            return 0.0, max(0.0, max_r)
        peak_max = max(r for _l, r in peaks)
        if peak_max < _CONF_FLOOR:
            return 0.0, max(0.0, peak_max)

        # entre los picos comparables al mayor se elige el lag MÁS CORTO: la
        # señal es tasa de eventos y todo múltiplo entero del período también
        # correlaciona; sin esta regla se reportan submúltiplos.
        best_lag, best_r = 0, 0.0
        cont_lag, cont_r = 0, 0.0
        for lag, r in peaks:
            if r >= peak_max * _HARMONIC_TOL:
                best_lag, best_r = lag, r
                break
        if best_lag == 0:
            return 0.0, max(0.0, peak_max)
        max_r = peak_max

        # sesgo de continuidad: si hay un pico casi tan bueno cerca del BPM
        # previo, se prefiere — mitiga los saltos de octava entre frames
        if prev_bpm > 0.0:
            for lag, r in acf:
                bpm = 60.0 * self.grid_hz / lag
                if abs(bpm - prev_bpm) / prev_bpm <= _CONTINUITY_CENTS and \
                        r >= best_r * _CONTINUITY_TOL and r > cont_r:
                    cont_lag, cont_r = lag, r
            if cont_lag:
                best_lag, best_r = cont_lag, cont_r

        # refinamiento sub-muestra por parábola sobre los vecinos del pico
        lag_f = float(best_lag)
        idx = best_lag - lag_min
        if 0 < idx < len(acf) - 1:
            y0, y1, y2 = acf[idx - 1][1], acf[idx][1], acf[idx + 1][1]
            d = y0 - 2 * y1 + y2
            if abs(d) > 1e-12:
                # acotado al rango barrido: sin el clamp la parábola extrapolaba
                # fuera de [lag_min, lag_max] y el BPM caía del rango declarado,
                # descartando estimaciones válidas
                lag_f = min(max(best_lag + 0.5 * (y0 - y2) / d,
                                float(lag_min)), float(lag_max))

        bpm = 60.0 * self.grid_hz / lag_f
        if not (BPM_MIN <= bpm <= BPM_MAX):
            return 0.0, max(0.0, best_r)
        return bpm, max(0.0, min(1.0, best_r))

    def _phase_at(self, t_us: int) -> float:
        """Fase de la componente periódica evaluada AHORA (correlación compleja).

        Se pondera hacia las muestras recientes: la fase debe seguir al presente,
        no al promedio de la ventana.
        """
        f = self._bpm / 60.0
        acc_re = acc_im = w_sum = 0.0
        tail_us = min(self.window_s, 2.0 / max(f, 1e-6)) * 1e6   # ~2 períodos
        for t_i, v in self._buf:
            dt = (t_us - t_i) / 1e6
            if dt * 1e6 > tail_us:
                continue
            w = 1.0 - (dt * 1e6) / tail_us      # lineal, más peso a lo reciente
            ang = 2.0 * math.pi * f * dt
            acc_re += w * v * math.cos(ang)
            acc_im += w * v * math.sin(ang)
            w_sum += w
        if w_sum <= 0.0 or (acc_re == 0.0 and acc_im == 0.0):
            return 0.0
        # atan2 da la fase del máximo más reciente; se normaliza a [0,1)
        ph = math.atan2(acc_im, acc_re) / (2.0 * math.pi)
        return ph % 1.0
