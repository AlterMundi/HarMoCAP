"""SlotManager — N slots estables + foco seleccionable (Hito 2, contrato 1.1).

Generaliza el slot principal del MVP a `max_slots` (8) slots estables:
- Asignación: track nuevo → slot libre MÁS BAJO; el incumbente de cada slot lo
  retiene con histéresis (occlusion_grace → releasing → timeout → tombstones
  repetidos), nadie le roba el slot mientras su track siga válido.
- boxes.id is None → no se emite slot provisional (se espera al tracker).
- Foco: modo `auto` (mayor bbox con histéresis de ratio: un nuevo mayor toma
  el foco solo si supera al actual por auto_focus_switch_ratio, evitando
  parpadeo) o `manual` (pineado por /control/select o teclado). Si el slot
  focal muere en modo manual → revierte a auto (documentado en el spec).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class Detection:
    """Una detección trackeada de un frame (ya extraída de Ultralytics)."""
    track_id: int
    bbox_xywhn: tuple[float, float, float, float]   # xc, yc, w, h normalizados
    keypoints_iso: list[tuple[float, float, float]]  # coords isotrópicas + conf YOLO

    @property
    def area(self) -> float:
        return self.bbox_xywhn[2] * self.bbox_xywhn[3]

    @property
    def centrality(self) -> float:
        dx = abs(self.bbox_xywhn[0] - 0.5)
        dy = abs(self.bbox_xywhn[1] - 0.5)
        return 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5)


class SlotState:
    EMPTY = "empty"
    ACTIVE = "active"
    OCCLUDED = "occluded"
    RELEASING = "releasing"
    TOMBSTONE = "tombstone"


@dataclass
class _Slot:
    state: str = SlotState.EMPTY
    track_id: int | None = None
    lost_since_us: int | None = None
    tombstones_left: int = 0


@dataclass
class SlotEvent:
    """Resultado por slot de un update(): qué emitir y si resetear estado."""
    slot_id: int
    detection: Detection | None    # datos de la persona (None si no hay este frame)
    emit_tombstone: bool           # emitir present=0 este frame
    slot_reset: bool               # el llamador resetea smoother+features del slot


class SlotManager:
    """Gestor de N slots estables + foco. Thread-safe para la selección."""

    def __init__(self, *, max_slots: int = 8,
                 occlusion_grace_ms: float = 1500.0,
                 release_timeout_ms: float = 3000.0,
                 acquire_rule: str = "largest_bbox",
                 auto_focus_switch_ratio: float = 1.20,
                 tombstone_repeat_frames: int = 15):
        self.max_slots = max_slots
        self.occlusion_grace_us = occlusion_grace_ms * 1000.0
        self.release_timeout_us = release_timeout_ms * 1000.0
        self.acquire_rule = acquire_rule
        self.auto_focus_switch_ratio = auto_focus_switch_ratio
        self.tombstone_repeat_frames = tombstone_repeat_frames
        self._slots = [_Slot() for _ in range(max_slots)]
        self._focus_lock = threading.Lock()
        self._manual_focus: int | None = None    # None = modo auto
        self._auto_focus: int | None = None

    # ------------------------------------------------------------------- foco
    def select_focus(self, slot: int) -> bool:
        """Pinea el foco a un slot (por /control/select o teclado). -1 = auto."""
        with self._focus_lock:
            if slot < 0:
                self._manual_focus = None
                return True
            if 0 <= slot < self.max_slots:
                self._manual_focus = slot
                return True
            return False

    def select_auto(self) -> None:
        self.select_focus(-1)

    @property
    def focused_slot(self) -> int | None:
        with self._focus_lock:
            if self._manual_focus is not None:
                return self._manual_focus
            return self._auto_focus

    @property
    def focus_mode(self) -> str:
        with self._focus_lock:
            return "manual" if self._manual_focus is not None else "auto"

    # ----------------------------------------------------------------- update
    def update(self, detections: list[Detection], t_us: int) -> list[SlotEvent]:
        by_id = {d.track_id: d for d in detections if d.track_id is not None}
        events: list[SlotEvent] = []
        claimed: set[int] = set()

        # 1) incumbentes: retienen su slot con histéresis
        for sid, slot in enumerate(self._slots):
            if slot.state in (SlotState.ACTIVE, SlotState.OCCLUDED, SlotState.RELEASING):
                det = by_id.get(slot.track_id)
                if det is not None:
                    slot.state = SlotState.ACTIVE
                    slot.lost_since_us = None
                    claimed.add(slot.track_id)
                    events.append(SlotEvent(sid, det, False, False))
                    continue
                if slot.lost_since_us is None:
                    slot.lost_since_us = t_us
                lost_for = t_us - slot.lost_since_us
                if lost_for <= self.occlusion_grace_us:
                    slot.state = SlotState.OCCLUDED
                elif lost_for <= self.release_timeout_us:
                    slot.state = SlotState.RELEASING
                else:
                    slot.state = SlotState.TOMBSTONE
                    slot.track_id = None
                    slot.lost_since_us = None
                    slot.tombstones_left = max(0, self.tombstone_repeat_frames - 1)
                    events.append(SlotEvent(sid, None, True, True))
            elif slot.state == SlotState.TOMBSTONE:
                if slot.tombstones_left > 0:
                    slot.tombstones_left -= 1
                    events.append(SlotEvent(sid, None, True, False))
                else:
                    slot.state = SlotState.EMPTY

        # 2) tracks no reclamados → slots libres (más bajo primero, orden
        #    determinista por regla de adquisición)
        free = [i for i, s in enumerate(self._slots) if s.state == SlotState.EMPTY]
        unclaimed = [d for tid, d in by_id.items() if tid not in claimed]
        key = (lambda d: -d.area) if self.acquire_rule == "largest_bbox" \
            else (lambda d: -d.centrality)
        for det in sorted(unclaimed, key=key):
            if not free:
                break                     # más personas que slots: se ignoran
            sid = free.pop(0)
            slot = self._slots[sid]
            slot.state = SlotState.ACTIVE
            slot.track_id = det.track_id
            slot.lost_since_us = None
            events.append(SlotEvent(sid, det, False, True))

        # 3) foco automático con histéresis + reversión del manual muerto
        active = {e.slot_id: e.detection for e in events
                  if e.detection is not None}
        with self._focus_lock:
            if self._manual_focus is not None and \
                    self._slots[self._manual_focus].state in (
                        SlotState.EMPTY, SlotState.TOMBSTONE):
                self._manual_focus = None     # el focal murió → revertir a auto
            if not active:
                self._auto_focus = None
            else:
                best = max(active, key=lambda s: active[s].area)
                if self._auto_focus not in active:
                    self._auto_focus = best
                elif best != self._auto_focus:
                    current_area = active[self._auto_focus].area
                    if active[best].area >= current_area * self.auto_focus_switch_ratio:
                        self._auto_focus = best

        events.sort(key=lambda e: e.slot_id)
        return events
