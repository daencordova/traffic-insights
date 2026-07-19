"""
Máquina de estados para tracks.

Gestiona las transiciones de estado de los tracks basándose
en hits, pérdidas y otras métricas.
"""

from typing import Dict, Any, List
import time

from models.enums import TrackStatus
from core.constants import MIN_HITS_TO_CONFIRM, MAX_FRAMES_MISSED
from utils.logger import LoggerMixin


class TrackStateMachine(LoggerMixin):
    """
    Máquina de estados para tracks.

    Estados posibles:
    - TENTATIVE: Track recién creado, necesita confirmación
    - CONFIRMED: Track confirmado, seguimiento activo
    - LOST: Track perdido temporalmente
    - DEAD: Track muerto, será eliminado

    Transiciones:
    TENTATIVE -> CONFIRMED: hits >= min_hits_to_confirm
    CONFIRMED -> LOST: no_losses > max_frames_missed // 2
    CONFIRMED -> DEAD: no_losses > max_frames_missed
    LOST -> CONFIRMED: hits >= min_hits_to_confirm y no_losses == 0
    LOST -> DEAD: no_losses > max_frames_missed
    """

    def __init__(
        self,
        min_hits_to_confirm: int = MIN_HITS_TO_CONFIRM,
        max_frames_missed: int = MAX_FRAMES_MISSED
    ):
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_frames_missed = max_frames_missed

        self._transitions: List[Dict[str, Any]] = []
        self._stats = {
            "tentative_to_confirmed": 0,
            "confirmed_to_lost": 0,
            "confirmed_to_dead": 0,
            "lost_to_confirmed": 0,
            "lost_to_dead": 0,
        }

        self.logger.info(
            "TrackStateMachine inicializado",
            min_hits_to_confirm=min_hits_to_confirm,
            max_frames_missed=max_frames_missed
        )

    def transition(
        self,
        current_status: TrackStatus,
        hits: int,
        no_losses: int
    ) -> TrackStatus:
        """
        Calcula el siguiente estado basado en el estado actual.

        Args:
            current_status: Estado actual del track
            hits: Número de detecciones asociadas
            no_losses: Número de frames consecutivos sin pérdida

        Returns:
            TrackStatus: Nuevo estado del track
        """
        if current_status == TrackStatus.DEAD:
            return TrackStatus.DEAD

        if current_status == TrackStatus.TENTATIVE:
            if hits >= self.min_hits_to_confirm:
                self._record_transition("tentative_to_confirmed")
                return TrackStatus.CONFIRMED

        elif current_status == TrackStatus.CONFIRMED:
            if no_losses > self.max_frames_missed:
                self._record_transition("confirmed_to_dead")
                return TrackStatus.DEAD
            elif no_losses > self.max_frames_missed // 2:
                self._record_transition("confirmed_to_lost")
                return TrackStatus.LOST

        elif current_status == TrackStatus.LOST:
            if no_losses > self.max_frames_missed:
                self._record_transition("lost_to_dead")
                return TrackStatus.DEAD
            elif hits >= self.min_hits_to_confirm and no_losses == 0:
                self._record_transition("lost_to_confirmed")
                return TrackStatus.CONFIRMED

        return current_status

    def should_promote_to_confirmed(self, hits: int) -> bool:
        """Verifica si un track tentativo debe ser confirmado."""
        return hits >= self.min_hits_to_confirm

    def should_mark_lost(self, no_losses: int) -> bool:
        """Verifica si un track debe ser marcado como perdido."""
        return no_losses > self.max_frames_missed // 2

    def should_mark_dead(self, no_losses: int) -> bool:
        """Verifica si un track debe ser marcado como muerto."""
        return no_losses > self.max_frames_missed

    def _record_transition(self, transition_type: str) -> None:
        """Registra una transición de estado."""
        if transition_type in self._stats:
            self._stats[transition_type] += 1
            self._transitions.append({
                "type": transition_type,
                "timestamp": time.time()
            })

            if len(self._transitions) > 1000:
                self._transitions = self._transitions[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la máquina de estados."""
        total = sum(self._stats.values())
        return {
            **self._stats,
            "total_transitions": total,
            "recent_transitions": self._transitions[-10:] if self._transitions else [],
        }

    def reset(self) -> None:
        """Reinicia las estadísticas."""
        self._transitions.clear()
        self._stats = {
            "tentative_to_confirmed": 0,
            "confirmed_to_lost": 0,
            "confirmed_to_dead": 0,
            "lost_to_confirmed": 0,
            "lost_to_dead": 0,
        }
