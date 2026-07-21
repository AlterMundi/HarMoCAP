"""El render de la interfaz web: fondo, estelas, escala. Sin modelos."""
from __future__ import annotations

import numpy as np

from harmocap.identity import Detection
from harmocap.webapp.processing import RenderConfig, Renderer


def _frame(v=120):
    return np.full((200, 320, 3), v, np.uint8)


def _det(track_id=1):
    # persona centrada, keypoints con confianza alta en coords isotrópicas
    kps = [(0.5 + 0.02 * i, 0.2 + 0.05 * i, 0.9) for i in range(17)]
    return Detection(track_id=track_id, bbox_xywhn=(0.5, 0.5, 0.3, 0.6),
                     keypoints_iso=kps)


def test_fondo_negro_no_deja_nada_del_video():
    r = Renderer(RenderConfig(background="black", overlays=[]))
    out = r.background(_frame(200))
    assert out.max() == 0


def test_fondo_oscurecido_conserva_proporcion():
    r = Renderer(RenderConfig(background="dim", dim=0.75))
    assert int(r.background(_frame(200)).mean()) == 50   # 200 * (1 - 0.75)


def test_fondo_video_no_toca_el_cuadro():
    r = Renderer(RenderConfig(background="video"))
    f = _frame(200)
    assert r.background(f) is f


def test_dibuja_solo_lo_seleccionado():
    """Con el fondo en negro, el render es exactamente lo que se pidió ver."""
    vacio = Renderer(RenderConfig(background="black", overlays=[], point_r=0))
    c = vacio.background(_frame())
    vacio.draw_person(c, 0, _det(), 320, 200, False)
    assert c.max() == 0                       # nada seleccionado, nada dibujado

    con = Renderer(RenderConfig(background="black", overlays=["skeleton"]))
    c2 = con.background(_frame())
    con.draw_person(c2, 0, _det(), 320, 200, False)
    assert c2.max() > 0


def test_estelas_persisten_y_decaen():
    cfg = RenderConfig(background="black", overlays=["skeleton"], trail_ms=1000)
    r = Renderer(cfg)
    c = r.background(_frame())
    layer = r.new_layer(c)
    r.draw_person(c, 0, _det(), 320, 200, False, layer)
    r.compose_trail(c, layer, 0.033)
    pico = r._trail.max()
    assert pico > 0
    # cuadros siguientes sin nadie: la estela sigue ahí pero más tenue
    c2 = r.background(_frame())
    out = r.compose_trail(c2, r.new_layer(c2), 0.5)
    assert 0 < r._trail.max() < pico
    assert out.max() > 0


def test_sin_estelas_no_hay_lienzo():
    r = Renderer(RenderConfig(trail_ms=0))
    c = r.background(_frame())
    assert r.new_layer(c) is None
    assert r.compose_trail(c, None, 0.033) is c


def test_escala_de_salida():
    r = Renderer(RenderConfig(scale=0.5))
    assert r.rescale(_frame()).shape[:2] == (100, 160)
    assert Renderer(RenderConfig(scale=1.0)).rescale(_frame()).shape[:2] == (200, 320)


def test_color_unico_vs_por_persona():
    por_slot = Renderer(RenderConfig())
    assert por_slot.color(0) != por_slot.color(1)
    unico = Renderer(RenderConfig(color_by_slot=False, solid_color=(1, 2, 3)))
    assert unico.color(0) == unico.color(5) == (1, 2, 3)
