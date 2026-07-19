"""
Controlador de estado y flujo del pipeline.

Gestiona el ciclo de vida del pipeline y aplica control de flujo
para evitar saturación del sistema.
"""

import time
import threading
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum, auto


class PipelineState(Enum):
    """Estados del pipeline."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class FlowControlConfig:
    """Configuración del control de flujo."""
    drop_threshold: float = 0.8
    recovery_threshold: float = 0.3
    max_frame_skip: int = 2
    consecutive_skip_limit: int = 5
    min_capture_fps: float = 5.0
    max_capture_fps: float = 15.0


class PipelineController:
    """
    Controlador de estado y flujo del pipeline.

    Responsabilidades:
    - Gestionar el estado del pipeline (RUNNING, PAUSED, etc.)
    - Aplicar control de flujo basado en el buffer
    - Mantener estadísticas de frames
    - Gestionar eventos de pausa/detención
    """

    def __init__(
        self,
        flow_config: Optional[FlowControlConfig] = None,
        is_cpu_mode: bool = True,
    ):
        self.flow_config = flow_config or FlowControlConfig()
        self.is_cpu_mode = is_cpu_mode

        self._state = PipelineState.IDLE
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        self._frame_skip_counter = 0
        self._consecutive_skips = 0
        self._dropped_count = 0
        self._frame_count = 0

        self._capture_fps_target = (
            FlowControlConfig().min_capture_fps if is_cpu_mode else 30.0
        )
        self._capture_interval = 1.0 / self._capture_fps_target

        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0
        self._last_capture_time = time.time()

        self._health_issues: List[str] = []

    @property
    def state(self) -> PipelineState:
        """Estado actual del pipeline."""
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == PipelineState.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._state == PipelineState.PAUSED

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    @property
    def pause_event(self) -> threading.Event:
        return self._pause_event

    @property
    def capture_interval(self) -> float:
        return self._capture_interval

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def start(self) -> None:
        """Inicia el pipeline."""
        if self._state == PipelineState.RUNNING:
            return

        self._state = PipelineState.RUNNING
        self._stop_event.clear()
        self._pause_event.clear()
        self._health_issues.clear()
        self._dropped_count = 0
        self._frame_count = 0
        self._frame_skip_counter = 0
        self._consecutive_skips = 0

        if self.is_cpu_mode:
            self._capture_fps_target = FlowControlConfig().min_capture_fps
            self._capture_interval = 1.0 / self._capture_fps_target

    def stop(self) -> None:
        """Detiene el pipeline."""
        self._state = PipelineState.STOPPING
        self._stop_event.set()
        self._pause_event.set()

    def pause(self) -> None:
        """Pausa el pipeline."""
        if self._state == PipelineState.RUNNING:
            self._state = PipelineState.PAUSED
            self._pause_event.set()

    def resume(self) -> None:
        """Reanuda el pipeline."""
        if self._state == PipelineState.PAUSED:
            self._state = PipelineState.RUNNING
            self._pause_event.clear()

    def mark_stopped(self) -> None:
        """Marca el pipeline como detenido."""
        self._state = PipelineState.STOPPED

    def mark_error(self) -> None:
        """Marca el pipeline en estado de error."""
        self._state = PipelineState.ERROR

    def can_capture_frame(self) -> bool:
        """
        Determina si se puede capturar un frame basado en el control de flujo.

        Returns:
            bool: True si se debe capturar un frame.
        """
        if self._stop_event.is_set():
            return False

        if self._pause_event.is_set():
            return False

        current_time = time.time()
        if current_time - self._last_capture_time < self._capture_interval:
            return False

        self._last_capture_time = current_time
        return True

    def should_process_frame(self, buffer_usage: float) -> bool:
        """
        Aplica control de flujo para decidir si procesar un frame.

        Args:
            buffer_usage: Uso del buffer (0-1)

        Returns:
            bool: True si el frame debe ser procesado.
        """
        if buffer_usage > self.flow_config.drop_threshold:
            self._frame_skip_counter += 1
            if self._frame_skip_counter < self.flow_config.max_frame_skip:
                self._dropped_count += 1
                self._consecutive_skips += 1

                if self._consecutive_skips > self.flow_config.consecutive_skip_limit:
                    self._add_health_issue(
                        f"Buffer crítico: {buffer_usage*100:.1f}%"
                    )
                    if self.is_cpu_mode:
                        self._reduce_capture_fps()

                return False
            else:
                self._frame_skip_counter = 0
                self._consecutive_skips = max(0, self._consecutive_skips - 2)

        elif buffer_usage < self.flow_config.recovery_threshold:
            self._frame_skip_counter = 0
            self._consecutive_skips = max(0, self._consecutive_skips - 2)
            if self._capture_fps_target < self.flow_config.max_capture_fps:
                self._capture_fps_target = min(
                    self.flow_config.max_capture_fps,
                    self._capture_fps_target + 0.5
                )
                self._capture_interval = 1.0 / self._capture_fps_target

        elif buffer_usage < 0.6:
            self._consecutive_skips = max(0, self._consecutive_skips - 1)

        return True

    def _reduce_capture_fps(self) -> None:
        """Reduce el FPS de captura para aliviar el buffer."""
        self._capture_fps_target = max(
            self.flow_config.min_capture_fps,
            self._capture_fps_target * 0.9
        )
        self._capture_interval = 1.0 / self._capture_fps_target

    def increment_frame_count(self) -> None:
        """Incrementa el contador de frames."""
        self._frame_count += 1

    def update_fps(self) -> None:
        """Actualiza el FPS actual."""
        self._fps_counter += 1
        if time.time() - self._fps_timer >= 1.0:
            self._current_fps = self._fps_counter
            self._fps_counter = 0
            self._fps_timer = time.time()

    def _add_health_issue(self, issue: str) -> None:
        """Registra un problema de salud."""
        timestamp = time.strftime("%H:%M:%S")
        self._health_issues.append(f"[{timestamp}] {issue}")
        if len(self._health_issues) > 100:
            self._health_issues = self._health_issues[-50:]

    def get_health_status(self) -> dict:
        """Obtiene el estado de salud del pipeline."""
        return {
            "healthy": len(self._health_issues) == 0,
            "issues": self._health_issues[-10:],
            "state": self._state.name,
        }

    def get_stats(self) -> dict:
        """Obtiene estadísticas del controlador."""
        return {
            "state": self._state.name,
            "fps": self._current_fps,
            "frame_count": self._frame_count,
            "dropped_count": self._dropped_count,
            "capture_interval": self._capture_interval,
            "capture_fps_target": self._capture_fps_target,
            "is_paused": self._pause_event.is_set(),
            "is_stopped": self._stop_event.is_set(),
        }
