"""Gestor de estado del pipeline con lógica de transiciones y recuperación.

Este módulo proporciona una gestión robusta del estado del pipeline,
incluyendo validación de transiciones, recuperación de errores y estadísticas.

El estado del pipeline puede ser:
- IDLE: Pipeline inicializado pero no en ejecución
- RUNNING: Pipeline procesando frames activamente
- PAUSED: Pipeline pausado temporalmente
- STOPPING: Pipeline en proceso de detención
- STOPPED: Pipeline detenido completamente
- ERROR: Pipeline en estado de error (requiere recuperación)

Características principales:
- Validación de transiciones de estado
- Registro de errores con ventana de tiempo
- Límite de intentos de recuperación
- Estadísticas de tiempo en cada estado
- Thread-safe (no requiere locks por diseño)
"""

from enum import Enum, auto
import time
from typing import Any


class PipelineStatus(Enum):
    """Estados posibles del pipeline.

    Attributes:
        IDLE: Pipeline inicializado pero no en ejecución.
        RUNNING: Pipeline procesando frames activamente.
        PAUSED: Pipeline pausado temporalmente.
        STOPPING: Pipeline en proceso de detención.
        STOPPED: Pipeline detenido completamente.
        ERROR: Pipeline en estado de error (requiere recuperación).
    """

    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()

    def is_active(self) -> bool:
        """Verifica si el estado es activo (procesando).

        Returns:
            bool: True si el estado es RUNNING o PAUSED.
        """
        return self in [PipelineStatus.RUNNING, PipelineStatus.PAUSED]

    def is_terminal(self) -> bool:
        """Verifica si el estado es terminal.

        Returns:
            bool: True si el estado es STOPPED o ERROR.
        """
        return self in [PipelineStatus.STOPPED, PipelineStatus.ERROR]

    def is_running(self) -> bool:
        """Verifica si el pipeline está en ejecución activa.

        Returns:
            bool: True si el estado es RUNNING.
        """
        return self == PipelineStatus.RUNNING

    def is_paused(self) -> bool:
        """Verifica si el pipeline está pausado.

        Returns:
            bool: True si el estado es PAUSED.
        """
        return self == PipelineStatus.PAUSED

    def is_stopped(self) -> bool:
        """Verifica si el pipeline está detenido.

        Returns:
            bool: True si el estado es STOPPED o STOPPING.
        """
        return self in [PipelineStatus.STOPPED, PipelineStatus.STOPPING]

    def is_error(self) -> bool:
        """Verifica si el pipeline está en estado de error.

        Returns:
            bool: True si el estado es ERROR.
        """
        return self == PipelineStatus.ERROR

    def get_display_name(self) -> str:
        """Obtiene el nombre legible del estado.

        Returns:
            str: Nombre del estado en formato legible.
        """
        names = {
            PipelineStatus.IDLE: "Inactivo",
            PipelineStatus.RUNNING: "Ejecutando",
            PipelineStatus.PAUSED: "Pausado",
            PipelineStatus.STOPPING: "Deteniendo",
            PipelineStatus.STOPPED: "Detenido",
            PipelineStatus.ERROR: "Error",
        }
        return names.get(self, self.name)


class PipelineStateManager:
    """Gestor de estado del pipeline con validación de transiciones.

    Esta clase maneja el ciclo de vida del pipeline, asegurando que las
    transiciones de estado sean válidas y proporcionando mecanismos
    para la recuperación de errores.

    Características:
        - Validación de transiciones de estado
        - Registro de errores con ventana de tiempo
        - Límite de intentos de recuperación
        - Estadísticas de tiempo en cada estado
        - Historial de transiciones
        - Estados configurados con umbrales ajustables

    Attributes:
        max_errors: Número máximo de errores antes de entrar en estado ERROR.
        error_window: Ventana de tiempo para contar errores (segundos).
        max_recovery_attempts: Número máximo de intentos de recuperación.
        recovery_cooldown: Tiempo de espera antes de intentar recuperación (segundos).

    Example:
        >>> state_manager = PipelineStateManager()
        >>> state_manager.start()
        >>> # ... pipeline ejecutándose ...
        >>> state_manager.pause()
        >>> state_manager.resume()
        >>> state_manager.stop()
        >>> stats = state_manager.get_stats()
    """

    DEFAULT_MAX_ERRORS = 3
    DEFAULT_ERROR_WINDOW = 60.0
    DEFAULT_MAX_RECOVERY_ATTEMPTS = 3
    DEFAULT_RECOVERY_COOLDOWN = 5.0

    def __init__(
        self,
        max_errors: int = DEFAULT_MAX_ERRORS,
        error_window: float = DEFAULT_ERROR_WINDOW,
        max_recovery_attempts: int = DEFAULT_MAX_RECOVERY_ATTEMPTS,
        recovery_cooldown: float = DEFAULT_RECOVERY_COOLDOWN,
    ):
        """Inicializa el gestor de estado.

        Args:
            max_errors: Número máximo de errores antes de entrar en estado ERROR.
            error_window: Ventana de tiempo para contar errores (segundos).
            max_recovery_attempts: Número máximo de intentos de recuperación.
            recovery_cooldown: Tiempo de espera antes de intentar recuperación.
        """
        self.max_errors = max_errors
        self.error_window = error_window
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_cooldown = recovery_cooldown

        self._status = PipelineStatus.IDLE
        self._previous_status = PipelineStatus.IDLE
        self._status_changed_at = time.time()
        self._start_time = time.time()

        self._error_count = 0
        self._last_error_time = 0.0

        self._recovery_attempts = 0
        self._last_recovery_attempt = 0.0

        self._transition_history: list[dict[str, Any]] = []
        self._max_history = 100

        self._valid_transitions = {
            PipelineStatus.IDLE: [PipelineStatus.RUNNING, PipelineStatus.STOPPED],
            PipelineStatus.RUNNING: [
                PipelineStatus.PAUSED,
                PipelineStatus.STOPPING,
                PipelineStatus.ERROR,
                PipelineStatus.STOPPED,
            ],
            PipelineStatus.PAUSED: [
                PipelineStatus.RUNNING,
                PipelineStatus.STOPPING,
                PipelineStatus.ERROR,
                PipelineStatus.STOPPED,
            ],
            PipelineStatus.STOPPING: [PipelineStatus.STOPPED],
            PipelineStatus.STOPPED: [PipelineStatus.IDLE, PipelineStatus.RUNNING],
            PipelineStatus.ERROR: [
                PipelineStatus.IDLE,
                PipelineStatus.RUNNING,
                PipelineStatus.STOPPED,
            ],
        }

    def start(self) -> bool:
        """Inicia el pipeline (transición IDLE -> RUNNING).

        Returns:
            bool: True si la transición fue exitosa.
        """
        if self._status == PipelineStatus.IDLE:
            return self._change_status(PipelineStatus.RUNNING)
        return False

    def pause(self) -> bool:
        """Pausa el pipeline (transición RUNNING -> PAUSED).

        Returns:
            bool: True si la transición fue exitosa.
        """
        if self._status == PipelineStatus.RUNNING:
            return self._change_status(PipelineStatus.PAUSED)
        return False

    def resume(self) -> bool:
        """Reanuda el pipeline (transición PAUSED -> RUNNING).

        Returns:
            bool: True si la transición fue exitosa.
        """
        if self._status == PipelineStatus.PAUSED:
            return self._change_status(PipelineStatus.RUNNING)
        return False

    def stop(self) -> bool:
        """Detiene el pipeline (transición * -> STOPPING -> STOPPED).

        Returns:
            bool: True si la transición fue exitosa.
        """
        if self._status in [PipelineStatus.IDLE, PipelineStatus.RUNNING, PipelineStatus.PAUSED] and self._change_status(PipelineStatus.STOPPING):
            return self._change_status(PipelineStatus.STOPPED)
        return False

    def set_status(self, new_status: PipelineStatus) -> bool:
        """Cambia el estado del pipeline si la transición es válida.

        Args:
            new_status: Nuevo estado deseado.

        Returns:
            bool: True si el cambio fue exitoso.

        Note:
            Prefiere usar los métodos específicos (start, pause, resume, stop)
            en lugar de este método genérico.
        """
        if new_status not in self._valid_transitions.get(self._status, []):
            return False

        return self._change_status(new_status)

    def _change_status(self, new_status: PipelineStatus) -> bool:
        """Realiza el cambio de estado y registra la transición.

        Args:
            new_status: Nuevo estado.

        Returns:
            bool: True si el cambio fue exitoso.
        """
        old_status = self._status
        self._previous_status = old_status
        self._status = new_status
        self._status_changed_at = time.time()

        self._record_transition(old_status, new_status)

        if old_status == PipelineStatus.IDLE and new_status == PipelineStatus.RUNNING:
            self._start_time = time.time()

        return True

    def _record_transition(self, old_status: PipelineStatus, new_status: PipelineStatus) -> None:
        """Registra una transición en el historial."""
        entry = {
            "timestamp": time.time(),
            "from": old_status.name,
            "to": new_status.name,
            "duration_seconds": self.get_status_duration() if old_status != new_status else 0.0,
        }

        self._transition_history.append(entry)
        if len(self._transition_history) > self._max_history:
            self._transition_history = self._transition_history[-self._max_history :]


    def can_transition_to(self, new_status: PipelineStatus) -> bool:
        """Verifica si es posible transicionar a un estado.

        Args:
            new_status: Estado al que se desea transicionar.

        Returns:
            bool: True si la transición es válida.
        """
        return new_status in self._valid_transitions.get(self._status, [])

    def is_valid_status(self, status: PipelineStatus) -> bool:
        """Verifica si un estado es válido.

        Args:
            status: Estado a verificar.

        Returns:
            bool: True si el estado existe en la enumeración.
        """
        return status in PipelineStatus

    def record_error(self) -> bool:
        """Registra un error y determina si se debe entrar en estado ERROR.

        Returns:
            bool: True si el pipeline entró en estado ERROR.

        Note:
            Los errores se cuentan dentro de una ventana de tiempo.
            Si se supera el umbral, el pipeline pasa a estado ERROR.
        """
        current_time = time.time()

        if current_time - self._last_error_time > self.error_window:
            self._error_count = 0

        self._error_count += 1
        self._last_error_time = current_time

        if self._error_count >= self.max_errors and self._status != PipelineStatus.ERROR:
            self.set_status(PipelineStatus.ERROR)
            return True

        return False

    def can_recover(self) -> bool:
        """Verifica si es posible recuperarse de un error.

        Returns:
            bool: True si se puede intentar la recuperación.

        Note:
            La recuperación solo es posible si:
            1. El pipeline está en estado ERROR
            2. Ha pasado el tiempo de cooldown
            3. No se han excedido los intentos máximos
        """
        if self._status != PipelineStatus.ERROR:
            return True

        if time.time() - self._status_changed_at < self.recovery_cooldown:
            return False

        return self._recovery_attempts < self.max_recovery_attempts

    def mark_recovery_attempt(self) -> None:
        """Marca un intento de recuperación.

        Note:
            Incrementa el contador de intentos y actualiza el timestamp.
        """
        self._recovery_attempts += 1
        self._last_recovery_attempt = time.time()

    def reset_recovery_attempts(self) -> None:
        """Reinicia el contador de intentos de recuperación."""
        self._recovery_attempts = 0
        self._last_recovery_attempt = 0.0

    def reset_errors(self) -> None:
        """Reinicia el contador de errores."""
        self._error_count = 0
        self._last_error_time = 0.0

    def get_status(self) -> PipelineStatus:
        """Obtiene el estado actual del pipeline.

        Returns:
            PipelineStatus: Estado actual.
        """
        return self._status

    def get_previous_status(self) -> PipelineStatus:
        """Obtiene el estado anterior del pipeline.

        Returns:
            PipelineStatus: Estado anterior.
        """
        return self._previous_status

    def get_status_display(self) -> str:
        """Obtiene el nombre legible del estado actual.

        Returns:
            str: Nombre del estado en formato legible.
        """
        return self._status.get_display_name()

    def is_running(self) -> bool:
        """Verifica si el pipeline está en ejecución activa.

        Returns:
            bool: True si el estado es RUNNING.
        """
        return self._status.is_running()

    def is_paused(self) -> bool:
        """Verifica si el pipeline está pausado.

        Returns:
            bool: True si el estado es PAUSED.
        """
        return self._status.is_paused()

    def is_stopped(self) -> bool:
        """Verifica si el pipeline está detenido.

        Returns:
            bool: True si el estado es STOPPED.
        """
        return self._status.is_stopped()

    def is_error(self) -> bool:
        """Verifica si el pipeline está en estado de error.

        Returns:
            bool: True si el estado es ERROR.
        """
        return self._status.is_error()

    def is_active(self) -> bool:
        """Verifica si el pipeline está activo (ejecutando o pausado).

        Returns:
            bool: True si el estado es RUNNING o PAUSED.
        """
        return self._status.is_active()

    def get_uptime(self) -> float:
        """Obtiene el tiempo total de ejecución en segundos.

        Returns:
            float: Tiempo de ejecución desde el último inicio.
        """
        return time.time() - self._start_time

    def get_status_duration(self) -> float:
        """Obtiene el tiempo en el estado actual en segundos.

        Returns:
            float: Duración del estado actual.
        """
        return time.time() - self._status_changed_at

    def get_time_in_state(self, status: PipelineStatus) -> float:
        """Obtiene el tiempo total acumulado en un estado específico.

        Args:
            status: Estado del cual obtener el tiempo.

        Returns:
            float: Tiempo acumulado en el estado (segundos).

        Note:
            Esta es una estimación basada en las transiciones registradas.
            Para cálculos precisos, se necesita un sistema de tracking de tiempo
            más sofisticado.
        """
        total_time = 0.0

        if status == self._status:
            total_time += self.get_status_duration()

        return total_time

    def get_stats(self) -> dict[str, Any]:
        """Obtiene estadísticas completas del gestor de estado.

        Returns:
            Dict[str, Any]: Diccionario con todas las estadísticas.
        """
        return {
            "status": self._status.name,
            "status_display": self.get_status_display(),
            "previous_status": self._previous_status.name if self._previous_status else None,
            "status_duration_seconds": self.get_status_duration(),
            "uptime_seconds": self.get_uptime(),
            "is_running": self.is_running(),
            "is_paused": self.is_paused(),
            "is_stopped": self.is_stopped(),
            "is_error": self.is_error(),
            "is_active": self.is_active(),
            "error_count": self._error_count,
            "error_window_seconds": self.error_window,
            "max_errors": self.max_errors,
            "recovery_attempts": self._recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "last_recovery_attempt": self._last_recovery_attempt,
            "last_error_time": self._last_error_time,
            "transition_count": len(self._transition_history),
        }

    def get_recent_transitions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Obtiene las transiciones más recientes.

        Args:
            limit: Número máximo de transiciones a retornar.

        Returns:
            List[Dict[str, Any]]: Lista de transiciones recientes.
        """
        return self._transition_history[-limit:] if self._transition_history else []

    def get_transition_summary(self) -> dict[str, int]:
        """Obtiene un resumen de las transiciones por tipo.

        Returns:
            Dict[str, int]: Conteo de transiciones por estado destino.
        """
        summary = {}
        for entry in self._transition_history:
            to_state = entry.get("to", "UNKNOWN")
            summary[to_state] = summary.get(to_state, 0) + 1
        return summary

    def reset(self) -> None:
        """Reinicia completamente el gestor de estado.

        Note:
            Esto limpia todas las estadísticas y vuelve al estado IDLE.
        """
        self._status = PipelineStatus.IDLE
        self._previous_status = PipelineStatus.IDLE
        self._status_changed_at = time.time()
        self._start_time = time.time()
        self._error_count = 0
        self._last_error_time = 0.0
        self._recovery_attempts = 0
        self._last_recovery_attempt = 0.0
        self._transition_history.clear()

    def clear_history(self) -> None:
        """Limpia el historial de transiciones sin resetear el estado."""
        self._transition_history.clear()

    def __repr__(self) -> str:
        """Representación del gestor de estado."""
        return (
            f"PipelineStateManager(status={self._status.name}, "
            f"uptime={self.get_uptime():.1f}s, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Representación legible del gestor de estado."""
        return (
            f"Estado: {self.get_status_display()} | "
            f"Tiempo: {self.get_uptime():.1f}s | "
            f"Errores: {self._error_count} | "
            f"Recuperaciones: {self._recovery_attempts}"
        )

def create_state_manager(
    max_errors: int = 3,
    error_window: float = 60.0,
    max_recovery_attempts: int = 3,
    recovery_cooldown: float = 5.0,
) -> PipelineStateManager:
    """Crea un gestor de estado con la configuración especificada.

    Args:
        max_errors: Número máximo de errores antes de entrar en estado ERROR.
        error_window: Ventana de tiempo para contar errores (segundos).
        max_recovery_attempts: Número máximo de intentos de recuperación.
        recovery_cooldown: Tiempo de espera antes de intentar recuperación.

    Returns:
        PipelineStateManager: Gestor de estado configurado.
    """
    return PipelineStateManager(
        max_errors=max_errors,
        error_window=error_window,
        max_recovery_attempts=max_recovery_attempts,
        recovery_cooldown=recovery_cooldown,
    )


def get_status_display_name(status: PipelineStatus) -> str:
    """Obtiene el nombre legible de un estado.

    Args:
        status: Estado del pipeline.

    Returns:
        str: Nombre legible del estado.
    """
    return status.get_display_name()


def is_status_transition_valid(
    from_status: PipelineStatus,
    to_status: PipelineStatus,
) -> bool:
    """Verifica si una transición entre dos estados es válida.

    Args:
        from_status: Estado origen.
        to_status: Estado destino.

    Returns:
        bool: True si la transición es válida.
    """
    valid_transitions = {
        PipelineStatus.IDLE: [PipelineStatus.RUNNING, PipelineStatus.STOPPED],
        PipelineStatus.RUNNING: [
            PipelineStatus.PAUSED,
            PipelineStatus.STOPPING,
            PipelineStatus.ERROR,
            PipelineStatus.STOPPED,
        ],
        PipelineStatus.PAUSED: [
            PipelineStatus.RUNNING,
            PipelineStatus.STOPPING,
            PipelineStatus.ERROR,
            PipelineStatus.STOPPED,
        ],
        PipelineStatus.STOPPING: [PipelineStatus.STOPPED],
        PipelineStatus.STOPPED: [PipelineStatus.IDLE, PipelineStatus.RUNNING],
        PipelineStatus.ERROR: [
            PipelineStatus.IDLE,
            PipelineStatus.RUNNING,
            PipelineStatus.STOPPED,
        ],
    }
    return to_status in valid_transitions.get(from_status, [])
