"""Interfaz web local de HarMoCAP — sube un video o usá la webcam en vivo.

Corre en la máquina del usuario (localhost), procesa con el hardware local hasta
donde alcance. No sube nada a ningún lado. Lanzar con: python scripts/webapp.py

La interfaz no decide nada: arma un `RunConfig` (lo que cuesta hardware) y un
`RenderConfig` (lo que se ve) y se los pasa al procesamiento. Los presets son
puntos conocidos de ese espacio; "Personalizado" abre todas las perillas.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import gradio as gr

from harmocap.schema import CROWD_FIELDS, FEATURE_ORDER
from harmocap.webapp.exports import export_csv, plot_crowd, plot_features
from harmocap.webapp.processing import (
    BACKGROUNDS, IMGSZ_CHOICES, MODELS, OVERLAY_LABELS, OVERLAYS, PRESET_ORDER,
    PRESETS, TRACKERS, RenderConfig, RunConfig, StreamProcessor, benchmark,
    hardware_info, preset_config, process_video, suggested_preset,
)

FEATURE_GROUPS = {
    "Energía y movimiento": ["qom", "vel_center", "vel_hand_l", "vel_hand_r",
                             "smoothness_l", "smoothness_r"],
    "Forma del cuerpo": ["contraction", "expansion", "symmetry", "verticality"],
    "Ángulos": ["angle_elbow_l", "angle_elbow_r", "angle_knee_l", "angle_knee_r",
                "angle_shoulder_l", "angle_shoulder_r", "angle_hip_l", "angle_hip_r"],
    "Cualidad (Laban)": ["laban_weight_proxy", "laban_time_proxy", "laban_space_proxy"],
    "Ritmo": ["tempo_bpm", "beat_phase", "tempo_conf"],
}

_OUT = Path(tempfile.gettempdir()) / "harmocap_webapp"
_OUT.mkdir(parents=True, exist_ok=True)

PRESET_CHOICES = [(PRESETS[k][0], k) for k in PRESET_ORDER] + \
                 [("Personalizado — todas las perillas a mano", "custom")]
MODEL_CHOICES = [(v[0], k) for k, v in MODELS.items()]
TRACKER_CHOICES = [(v[0], k) for k, v in TRACKERS.items()]
BG_CHOICES = [(v, k) for k, v in BACKGROUNDS.items()]

# Orden de los controles de RunConfig en la UI. La misma tupla se usa para armar
# el RunConfig desde los valores, así no se pueden desincronizar.
RUN_FIELDS = ("model_size", "imgsz", "conf", "max_det", "max_slots", "tracker",
              "reacquisition", "crowd", "density", "density_stride",
              "frame_stride", "mincutoff", "beta")
RENDER_FIELDS = ("background", "dim", "line_w", "point_r", "trail_ms",
                 "color_by_slot", "hud", "scale", "write_video")


def _est(hw_label: str) -> str:
    return {"Placa NVIDIA (rápido)": "Rápido: cerca del tiempo real.",
            "Apple Silicon (moderado)": "Moderado: para tiempo real usá el preset Esencial.",
            "Procesador, sin placa (lento)": "Lento: usá el preset Esencial."}\
        .get(hw_label, "")


def _mk_run(preset, vals, t_start=0.0, t_end=0.0) -> RunConfig:
    """Preset + valores de los controles → RunConfig."""
    base = preset_config("grupo" if preset == "custom" else preset)
    kw = dict(zip(RUN_FIELDS, vals))
    for k in ("imgsz", "max_det", "max_slots", "density_stride", "frame_stride"):
        kw[k] = int(kw[k])
    for k in ("conf", "mincutoff", "beta"):
        kw[k] = float(kw[k])
    for k in ("reacquisition", "crowd", "density"):
        kw[k] = bool(kw[k])
    return replace(base, preset=preset, t_start=float(t_start or 0),
                   t_end=float(t_end or 0), **kw)


def _mk_render(overlay_labels, vals) -> RenderConfig:
    ov = [k for k in OVERLAYS if OVERLAY_LABELS[k] in (overlay_labels or [])]
    kw = dict(zip(RENDER_FIELDS, vals))
    for k in ("line_w", "point_r", "trail_ms"):
        kw[k] = int(kw[k])
    for k in ("dim", "scale"):
        kw[k] = float(kw[k])
    for k in ("color_by_slot", "hud", "write_video"):
        kw[k] = bool(kw[k])
    return RenderConfig(overlays=ov or ["skeleton"], **kw)


def _run_controls(defaults: RunConfig):
    """Controles de procesamiento. Devuelve la lista en el orden de RUN_FIELDS."""
    with gr.Row():
        model_size = gr.Radio(MODEL_CHOICES, value=defaults.model_size,
                              label="Modelo de pose")
        tracker = gr.Radio(TRACKER_CHOICES, value=defaults.tracker,
                           label="Seguimiento de identidad")
    with gr.Row():
        imgsz = gr.Dropdown([str(s) for s in IMGSZ_CHOICES],
                            value=str(defaults.imgsz),
                            label="Resolución de proceso (px)")
        max_slots = gr.Slider(1, 8, value=defaults.max_slots, step=1,
                              label="Personas a seguir (slots)")
    with gr.Row():
        conf = gr.Slider(0.01, 0.8, value=defaults.conf, step=0.01,
                         label="Confianza mínima de detección")
        max_det = gr.Slider(4, 300, value=defaults.max_det, step=1,
                            label="Detecciones máximas por cuadro")
    with gr.Row():
        reacq = gr.Checkbox(value=defaults.reacquisition,
                            label="Reasociar identidad tras oclusión")
        crowd = gr.Checkbox(value=defaults.crowd, label="Variables de multitud")
        density = gr.Checkbox(value=defaults.density,
                              label="Mapa de densidad (caro)")
    with gr.Row():
        d_stride = gr.Slider(1, 20, value=defaults.density_stride, step=1,
                             label="Densidad: 1 de cada N cuadros")
        f_stride = gr.Slider(1, 10, value=defaults.frame_stride, step=1,
                             label="Procesar 1 de cada N cuadros")
    with gr.Row():
        mincutoff = gr.Slider(0.1, 5.0, value=defaults.mincutoff, step=0.1,
                              label="Suavizado: corte mínimo (Hz)")
        beta = gr.Slider(0.0, 1.0, value=defaults.beta, step=0.01,
                         label="Suavizado: respuesta (beta)")
    return [model_size, imgsz, conf, max_det, max_slots, tracker, reacq,
            crowd, density, d_stride, f_stride, mincutoff, beta]


def _render_controls(defaults: RenderConfig, live: bool):
    """Controles de render. Devuelve (overlays, [en orden de RENDER_FIELDS])."""
    overlays = gr.CheckboxGroup(
        [OVERLAY_LABELS[k] for k in OVERLAYS],
        value=[OVERLAY_LABELS[k] for k in defaults.overlays],
        label="Qué dibujar")
    with gr.Row():
        bg = gr.Radio(BG_CHOICES, value=defaults.background, label="Fondo")
        dim = gr.Slider(0.0, 1.0, value=defaults.dim, step=0.05,
                        label="Cuánto oscurecer el fondo")
    with gr.Row():
        line_w = gr.Slider(1, 12, value=defaults.line_w, step=1,
                           label="Grosor de línea")
        point_r = gr.Slider(0, 14, value=defaults.point_r, step=1,
                            label="Tamaño de punto (0 = sin puntos)")
    with gr.Row():
        trail = gr.Slider(0, 3000, value=defaults.trail_ms, step=50,
                          label="Estelas: duración (ms, 0 = sin estelas)")
        scale = gr.Slider(0.25, 1.0, value=defaults.scale, step=0.05,
                          label="Escala de la imagen de salida")
    with gr.Row():
        color_slot = gr.Checkbox(value=defaults.color_by_slot,
                                 label="Un color por persona")
        hud = gr.Checkbox(value=defaults.hud, label="Datos sobre la imagen")
        write = gr.Checkbox(value=defaults.write_video,
                            label="Generar video de salida", visible=not live)
    return overlays, [bg, dim, line_w, point_r, trail, color_slot, hud, scale, write]


# ────────────────────────────── callbacks ───────────────────────────────────
def on_preset(name):
    """Al elegir un preset se cargan sus valores en los controles."""
    if name == "custom":
        return [gr.update() for _ in RUN_FIELDS]
    c = preset_config(name)
    return [gr.update(value=str(c.imgsz) if f == "imgsz" else getattr(c, f))
            for f in RUN_FIELDS]


def on_benchmark(video, preset, *run_vals):
    if not video:
        return "Cargá un video primero para estimar."
    b = benchmark(video, _mk_run(preset, run_vals))
    if "error" in b:
        return b["error"]
    eta = (f" · el video completo tardaría ~{b['eta_s'] / 60:.1f} min"
           if b["eta_s"] else "")
    rt = ("**alcanza para tiempo real**" if b["realtime"]
          else "no alcanza para tiempo real")
    return (f"{b['fps_proc']} fps de proceso ({b['ms_frame']} ms/cuadro) en "
            f"`{b['device']}` — {rt}{eta}.\n\n"
            f"Modelo cargado: `{Path(b['loaded']['loaded_checkpoint']).name}` · "
            f"imgsz {b['loaded']['imgsz']} · tracker "
            f"`{Path(b['loaded']['tracker']).name}`")


def run(video, preset, t_start, t_end, feat_labels, crowd_labels, do_csv,
        do_plots, *ctrl_vals, progress=gr.Progress()):
    if not video:
        raise gr.Error("Cargá un video primero.")
    n = len(RUN_FIELDS)
    run_cfg = _mk_run(preset, ctrl_vals[:n], t_start=t_start, t_end=t_end)
    render_cfg = _mk_render(ctrl_vals[n], ctrl_vals[n + 1:])

    progress(0.02, desc="Preparando el modelo…")
    res = process_video(video, run_cfg, render_cfg, _OUT,
                        progress=lambda f, t: progress(f, desc=t))

    files = [str(res.jsonl_path), str(res.config_path)]
    fig_p = fig_c = None
    sel_feats = list(feat_labels or FEATURE_ORDER)
    if do_csv:
        p_csv, c_csv = export_csv(res.jsonl_path, _OUT, features=sel_feats)
        files += [str(p_csv), str(c_csv)]
    if do_plots:
        fig_p = plot_features(res.feature_series, sel_feats[:8])
        if run_cfg.crowd or crowd_labels:
            fig_c = plot_crowd(res.crowd_series,
                               list(crowd_labels) or ["crowd_count", "crowd_qom",
                                                      "mass_present", "mass_active"])
    ck = Path(res.backend_info["loaded_checkpoint"]).name
    resumen = (f"✅ {res.frames} cuadros en {res.seconds}s · "
               f"**{res.fps_proc} fps** de proceso · `{res.device}`\n\n"
               f"Preset `{run_cfg.preset}` · modelo `{ck}` · imgsz "
               f"{run_cfg.imgsz} · conf {run_cfg.conf} · max_det "
               f"{run_cfg.max_det} · {run_cfg.max_slots} slots · tracker "
               f"`{run_cfg.tracker}`\n\n"
               f"Config exacta de la corrida: `{res.config_path.name}`")
    return (str(res.video_path) if res.video_path else None,
            resumen, files, fig_p, fig_c)


def live_step(frame_rgb, preset, state, *ctrl_vals):
    """Un cuadro de la webcam en vivo. `state` = StreamProcessor por sesión."""
    if frame_rgb is None:
        return None, {}, state
    n = len(RUN_FIELDS)
    run_cfg = _mk_run(preset, ctrl_vals[:n])
    render_cfg = _mk_render(ctrl_vals[n], ctrl_vals[n + 1:])
    if state is None or state.run != run_cfg:
        state = StreamProcessor(run_cfg, render_cfg)   # recarga el modelo
    elif state.render_cfg != render_cfg:
        state.render_cfg = render_cfg                  # barato: solo dibujo
        state.renderer.cfg = render_cfg
    frame_bgr = frame_rgb[:, :, ::-1].copy()   # Gradio da RGB; el pipeline usa BGR
    annotated_bgr, readout = state.step(frame_bgr)
    return annotated_bgr[:, :, ::-1], readout, state


def build() -> gr.Blocks:
    hw = hardware_info()
    default_preset = suggested_preset(hw["device"])
    d_run = preset_config(default_preset)

    with gr.Blocks(title="HarMoCAP — captura de movimiento local") as demo:
        gr.Markdown(
            "# HarMoCAP\n"
            "Medí el movimiento del cuerpo y de la multitud. **Todo se procesa "
            "en esta máquina y no sale de acá.**\n\n"
            f"Hardware detectado: **{hw['label']}** — {_est(hw['label'])}")

        # ═══════════════════════ pestaña 1: procesar un video ════════════════
        with gr.Tab("Procesar un video"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1 · Cargar")
                    video = gr.Video(label="Video (o grabá con la webcam)",
                                     sources=["upload", "webcam"])
                    with gr.Row():
                        t_start = gr.Number(value=0, label="Desde (s)")
                        t_end = gr.Number(value=0, label="Hasta (s, 0 = final)")

                    gr.Markdown("### 2 · Rendimiento y calidad")
                    preset = gr.Radio(PRESET_CHOICES, value=default_preset,
                                      label="Preset")
                    with gr.Accordion("Parámetros de procesamiento", open=False):
                        gr.Markdown(
                            "Definen **cuánto cuesta** cada cuadro. Las que más "
                            "pesan: el modelo, la resolución de proceso y el "
                            "seguimiento con ReID (corre una segunda red por "
                            "cada caja detectada).")
                        run_ctrls = _run_controls(d_run)
                    bench_btn = gr.Button("Estimar velocidad con esta config",
                                          size="sm")
                    bench_out = gr.Markdown()

                    gr.Markdown("### 3 · Qué ver y qué exportar")
                    with gr.Accordion("Render del video de salida", open=True):
                        overlays, render_ctrls = _render_controls(
                            RenderConfig(), live=False)
                    with gr.Accordion("Variables a exportar", open=False):
                        feat_sel = gr.CheckboxGroup(
                            FEATURE_ORDER, value=list(FEATURE_ORDER),
                            label="Por persona (24)")
                        crowd_sel = gr.CheckboxGroup(
                            list(CROWD_FIELDS), value=[],
                            label="De multitud (13)")
                    with gr.Row():
                        do_csv = gr.Checkbox(value=True, label="Exportar CSV")
                        do_plots = gr.Checkbox(value=True, label="Graficar variables")
                    go = gr.Button("Procesar", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 4 · Resultado")
                    out_video = gr.Video(label="Video con el render", autoplay=False)
                    resumen = gr.Markdown()
                    out_files = gr.File(label="Descargas (sesión + config + CSV)",
                                        file_count="multiple")
                    plot_p = gr.Plot(label="Variables por persona")
                    plot_c = gr.Plot(label="Variables de multitud")

            preset.change(on_preset, inputs=[preset], outputs=run_ctrls)
            bench_btn.click(on_benchmark, inputs=[video, preset] + run_ctrls,
                            outputs=[bench_out])
            go.click(run,
                     inputs=([video, preset, t_start, t_end, feat_sel, crowd_sel,
                              do_csv, do_plots] + run_ctrls + [overlays]
                             + render_ctrls),
                     outputs=[out_video, resumen, out_files, plot_p, plot_c])

        # ═══════════════════════ pestaña 2: en vivo (webcam) ═════════════════
        with gr.Tab("En vivo (webcam)"):
            gr.Markdown(
                "Procesa la **cámara de este navegador** en tiempo real. La "
                "cámara es la de tu máquina aunque el procesamiento corra en "
                "otra. Si va lento: preset **Esencial**, y bajá la resolución "
                "de proceso y la escala de salida.")
            with gr.Row():
                with gr.Column(scale=1):
                    live_preset = gr.Radio(PRESET_CHOICES, value=default_preset,
                                           label="Preset")
                    live_cam = gr.Image(sources=["webcam"], streaming=True,
                                        type="numpy", label="Webcam")
                    with gr.Accordion("Parámetros de procesamiento", open=False):
                        live_run = _run_controls(d_run)
                    with gr.Accordion("Render", open=False):
                        live_ov, live_render = _render_controls(
                            RenderConfig(hud=True), live=True)
                with gr.Column(scale=1):
                    live_out = gr.Image(label="En vivo (con el render)", type="numpy")
                    live_readout = gr.JSON(
                        label="Variables en vivo (persona focal + multitud)")
            live_state = gr.State(None)
            live_preset.change(on_preset, inputs=[live_preset], outputs=live_run)
            live_cam.stream(
                live_step,
                inputs=([live_cam, live_preset, live_state] + live_run
                        + [live_ov] + live_render),
                outputs=[live_out, live_readout, live_state],
                stream_every=0.1, concurrency_limit=1, show_progress="hidden")

    return demo


def main(share: bool = False, port: int = 7860, host: str = "127.0.0.1"):
    # host por defecto 127.0.0.1: solo esta máquina. Para exponer en una red
    # confiable (p. ej. una VPN) se pasa la IP de esa interfaz; nunca 0.0.0.0
    # a la ligera, que incluiría la interfaz pública.
    build().launch(server_name=host, server_port=port,
                   inbrowser=(host == "127.0.0.1"), share=share)


if __name__ == "__main__":
    main()
