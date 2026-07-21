"""La configuración de la interfaz web llega intacta al pipeline.

La UI arma un RunConfig; si un preset o un control se desincronizara del
backend, el usuario creería estar corriendo algo que no corre. Estos tests
fijan ese contrato sin cargar modelos ni gradio.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from harmocap.webapp.processing import (
    IMGSZ_CHOICES, MODELS, PRESET_ORDER, PRESETS, TRACKERS, RenderConfig,
    RunConfig, preset_config, suggested_preset,
)


def test_presets_declarados_y_coherentes():
    assert set(PRESET_ORDER) == set(PRESETS)
    for name in PRESET_ORDER:
        label, cfg = PRESETS[name]
        assert label and cfg.preset == name
        assert cfg.model_size in MODELS
        assert cfg.tracker in TRACKERS
        assert cfg.imgsz in IMGSZ_CHOICES
        assert 1 <= cfg.max_slots <= 8
        assert 0 < cfg.conf < 1
        assert cfg.max_det >= cfg.max_slots


def test_preset_config_devuelve_copia():
    a = preset_config("esencial")
    a.imgsz = 1280
    assert preset_config("esencial").imgsz == 512


def test_esencial_es_el_mas_barato():
    """El preset de tiempo real no debe arrastrar nada caro."""
    c = preset_config("esencial")
    assert c.model_size == "n"          # modelo nano
    assert c.tracker == "bytetrack"     # sin ReID: no hay 2da red por caja
    assert not c.density
    assert c.max_det <= 8               # ReID/asociación no ven cientos de cajas
    assert c.imgsz <= 640


def test_masa_activa_densidad_y_recall():
    c = preset_config("masa")
    assert c.density and c.crowd
    assert c.max_det >= 200 and c.conf <= 0.1


def test_sugerencia_por_hardware():
    assert suggested_preset("cuda:0") == "grupo"
    assert suggested_preset("mps") == "esencial"
    assert suggested_preset("cpu") == "esencial"


def test_ui_cubre_todas_las_perillas_de_costo():
    """Todo campo de RunConfig que cambie el costo tiene control en la UI."""
    from harmocap.webapp.app import RENDER_FIELDS, RUN_FIELDS
    sin_control = {"preset", "t_start", "t_end", "density_min_edge"}
    assert set(RUN_FIELDS) | sin_control == {f.name for f in fields(RunConfig)}
    assert set(RENDER_FIELDS) | {"overlays", "solid_color"} == \
        {f.name for f in fields(RenderConfig)}


@pytest.mark.parametrize("preset", PRESET_ORDER)
def test_mk_run_reconstruye_el_preset(preset):
    """Los valores por defecto de los controles regeneran el mismo RunConfig."""
    from harmocap.webapp.app import RUN_FIELDS, _mk_run
    c = preset_config(preset)
    vals = [str(c.imgsz) if f == "imgsz" else getattr(c, f) for f in RUN_FIELDS]
    assert _mk_run(preset, vals) == c


def test_mk_render_traduce_etiquetas():
    from harmocap.webapp.app import RENDER_FIELDS, _mk_render
    d = RenderConfig()
    vals = [getattr(d, f) for f in RENDER_FIELDS]
    r = _mk_render(["Esqueleto (líneas)", "Caja (bbox)"], vals)
    assert r.overlays == ["skeleton", "bbox"]
    # sin nada seleccionado no se devuelve un render vacío
    assert _mk_render([], vals).overlays == ["skeleton"]
