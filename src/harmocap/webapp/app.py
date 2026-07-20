"""Interfaz web local de HarMoCAP — sube un video, elegí qué medir/ver/exportar.

Corre en la máquina del usuario (localhost), procesa con el hardware local hasta
donde alcance. No sube nada a ningún lado. Lanzar con: python scripts/webapp.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from harmocap.schema import CROWD_FIELDS, FEATURE_ORDER
from harmocap.webapp.exports import export_csv, plot_crowd, plot_features
from harmocap.webapp.processing import (
    OVERLAY_LABELS, OVERLAYS, StreamProcessor, hardware_info, process_video,
)

# --- agrupación de features para la selección (por familia) -----------------
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


def _est(hw_label: str) -> str:
    return {"Placa NVIDIA (rápido)": "Rápido: cerca del tiempo real.",
            "Apple Silicon (moderado)": "Moderado: algo más lento que el video.",
            "Procesador, sin placa (lento)": "Lento: puede tardar varios minutos."}\
        .get(hw_label, "")


def run(video, mode, overlay_labels, feat_labels, crowd_labels,
        do_csv, do_plots, progress=gr.Progress()):
    if not video:
        raise gr.Error("Cargá un video primero.")
    overlays = [k for k in OVERLAYS if OVERLAY_LABELS[k] in (overlay_labels or [])]
    if not overlays:
        overlays = ["skeleton"]
    sel_feats = list(feat_labels or FEATURE_ORDER)
    sel_crowd = list(crowd_labels or [])

    def pcb(frac, txt):
        progress(frac, desc=txt)

    progress(0.02, desc="Preparando el modelo…")
    res = process_video(video, mode, overlays, _OUT, progress=pcb)

    files = [str(res.jsonl_path)]
    fig_p = fig_c = None
    if do_csv:
        p_csv, c_csv = export_csv(res.jsonl_path, _OUT, features=sel_feats)
        files += [str(p_csv), str(c_csv)]
    if do_plots:
        fig_p = plot_features(res.feature_series, sel_feats)
        if mode == "crowd" or sel_crowd:
            fig_c = plot_crowd(res.crowd_series,
                               sel_crowd or ["crowd_count", "crowd_qom",
                                             "mass_present", "mass_active"])

    resumen = (f"✅ {res.frames} cuadros · {res.fps_proc} fps de proceso · "
               f"modo **{res.mode}** · hardware `{res.device}`\n\n"
               f"Sesión grabada: `{res.jsonl_path.name}`")
    return str(res.video_path), resumen, files, fig_p, fig_c


def live_step(frame_rgb, mode, state):
    """Un cuadro de la webcam en vivo. `state` = StreamProcessor por sesión."""
    if frame_rgb is None:
        return None, {}, state
    if state is None or state.mode != mode:
        overlays = ["skeleton", "bbox", "ids"] + (["density"] if mode == "crowd" else [])
        state = StreamProcessor(mode=mode, overlays=overlays)
    frame_bgr = frame_rgb[:, :, ::-1].copy()      # Gradio da RGB; el pipeline usa BGR
    annotated_bgr, readout = state.step(frame_bgr)
    return annotated_bgr[:, :, ::-1], readout, state


def build() -> gr.Blocks:
    hw = hardware_info()
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
                    gr.Markdown("### 2 · Configurar")
                    mode = gr.Radio(
                        [("Grupo — identidad firme, hasta 8 personas", "group"),
                         ("Masa — multitud, señales de masa", "crowd")],
                        value="group", label="Modo")
                    overlays = gr.CheckboxGroup(
                        [OVERLAY_LABELS[k] for k in OVERLAYS],
                        value=[OVERLAY_LABELS["skeleton"], OVERLAY_LABELS["ids"]],
                        label="Qué dibujar en el video de salida")
                    with gr.Accordion("Variables a exportar", open=False):
                        feat_sel = gr.CheckboxGroup(
                            FEATURE_ORDER, value=list(FEATURE_ORDER),
                            label="Por persona (24)")
                        crowd_sel = gr.CheckboxGroup(
                            list(CROWD_FIELDS), value=[],
                            label="De multitud (13) — útiles en modo masa")
                    with gr.Row():
                        do_csv = gr.Checkbox(value=True, label="Exportar CSV")
                        do_plots = gr.Checkbox(value=True, label="Graficar variables")
                    gr.Markdown("### 3 · Procesar")
                    go = gr.Button("Procesar", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 4 · Ver y exportar")
                    out_video = gr.Video(label="Video con overlay", autoplay=False)
                    resumen = gr.Markdown()
                    out_files = gr.File(label="Descargas (sesión + CSV)",
                                        file_count="multiple")
                    plot_p = gr.Plot(label="Variables por persona")
                    plot_c = gr.Plot(label="Variables de multitud")

            go.click(run,
                     inputs=[video, mode, overlays, feat_sel, crowd_sel,
                             do_csv, do_plots],
                     outputs=[out_video, resumen, out_files, plot_p, plot_c])

        # ═══════════════════════ pestaña 2: en vivo (webcam) ═════════════════
        with gr.Tab("En vivo (webcam)"):
            gr.Markdown(
                "Procesa la **cámara de este navegador** en tiempo real y muestra "
                "los esqueletos y las variables mientras se mueve. La cámara es la "
                "de tu máquina aunque el procesamiento corra en otra.")
            with gr.Row():
                with gr.Column(scale=1):
                    live_mode = gr.Radio(
                        [("Grupo — identidad firme", "group"),
                         ("Masa — multitud", "crowd")],
                        value="group", label="Modo")
                    live_cam = gr.Image(sources=["webcam"], streaming=True,
                                        type="numpy", label="Webcam")
                with gr.Column(scale=1):
                    live_out = gr.Image(label="En vivo (con overlay)", type="numpy")
                    live_readout = gr.JSON(label="Variables en vivo (persona focal + multitud)")
            live_state = gr.State(None)
            live_cam.stream(
                live_step, inputs=[live_cam, live_mode, live_state],
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
