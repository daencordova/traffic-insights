"""
Estado del pipeline y gestión de transiciones.
"""

import time
from enum import Enum, auto


class PipelineStatus(Enum):
    """Estados posibles del pipeline."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

    def is_active(self) -> bool:
        """Verifica si el estado es activo (procesando)."""
        return self in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]

    def is_terminal(self) -> bool:
        """Verifica si el estado es terminal."""
        return self in [PipelineStatus.STOPPED, PipelineStatus.ERROR]


class PipelineState:
    """
    Gestor de estado del pipeline.

    Responsabilidades:
    - Mantener el estado actual
    - Validar transiciones de estado
    - Registrar métricas de tiempo
    - Gestionar recuperación de errores
    """

    def __init__(self):
        self._status = PipelineStatus.IDLE
        self._previous_status = PipelineStatus.IDLE
        self._status_changed_at = time.time()
        self._error_count = 0
        self._last_error_time = 0.0
        self._max_errors = 3
        self._error_window = 60.0
        self._start_time = time.time()

    def set_status(self, new_status: PipelineStatus) -> bool:
        """
        Cambia el estado del pipeline si la transición es válida.

        Args:
            new_status: Nuevo estado deseado

        Returns:
            bool: True si el cambio fue exitoso
        """
        if not self._is_valid_transition(new_status):
            self._previous_status = self._status
            return False

        self._previous_status = self._status
        self._status = new_status
        self._status_changed_at = time.time()
        return True

    def _is_valid_transition(self, new_status: PipelineStatus) -> bool:
        """Verifica si la transición de estado es válida."""
        valid_transitions = {
            PipelineStatus.IDLE: [PipelineStatus.RUNNING, PipelineStatus.STOPPED],
            PipelineStatus.RUNNING: [PipelineStatus.PAUSED, PipelineStatus.STOPPING,
                                     PipelineStatus.ERROR, PipelineStatus.STOPPED],
            PipelineStatus.PAUSED: [PipelineStatus.RUNNING, PipelineStatus.STOPPING,
                                    PipelineStatus.ERROR, PipelineStatus.STOPPED],
            PipelineStatus.STOPPING: [PipelineStatus.STOPPED],
            PipelineStatus.STOPPED: [PipelineStatus.IDLE, PipelineStatus.RUNNING],
            PipelineStatus.ERROR: [PipelineStatus.IDLE, PipelineStatus.RUNNING,
                                   PipelineStatus.STOPPED],
        }

        return new_status in valid_transitions.get(self._status, [])

    def get_status(self) -> PipelineStatus:
        """Obtiene el estado actual."""
        return self._status

    def is_running(self) -> bool:
        """Verifica si el pipeline está en ejecución."""
        return self._status == PipelineStatus.RUNNING

    def is_paused(self) -> bool:
        """Verifica si el pipeline está pausado."""
        return self._status == PipelineStatus.PAUSED

    def is_stopped(self) -> bool:
        """Verifica si el pipeline está detenido."""
        return self._status in [PipelineStatus.STOPPED, PipelineStatus.STOPPING]

    def is_error(self) -> bool:
        """Verifica si el pipeline está en estado de error."""
        return self._status == PipelineStatus.ERROR

    def get_uptime(self) -> float:
        """Obtiene el tiempo de ejecución en segundos."""
        return time.time() - self._start_time

    def get_status_duration(self) -> float:
        """Obtiene el tiempo en el estado actual."""
        return time.time() - self._status_changed_at

    def record_error(self) -> None:
        """Registra un error para el mecanismo de recuperación."""
        current_time = time.time()

        if current_time - self._last_error_time > self._error_window:
            self._error_count = 0

        self._error_count += 1
        self._last_error_time = current_time

        if self._error_count >= self._max_errors:
            self.set_status(PipelineStatus.ERROR)

    def can_recover(self) -> bool:
        """Verifica si es posible recuperarse de un error."""
        if self._status != PipelineStatus.ERROR:
            return True

        if time.time() - self._status_changed_at > 5.0:
            self._error_count = 0
            return True

        return False

    def reset(self) -> None:
        """Reinicia el estado."""
        self._status = PipelineStatus.IDLE
        self._previous_status = PipelineStatus.IDLE
        self._status_changed_at = time.time()
        self._error_count = 0
        self._last_error_time = 0.0
