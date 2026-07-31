"""Módulo para obtener información del sistema (CPU, Memoria, Estado). Minimalista y optimizado para mostrar en una sola línea."""

from collections import deque
from dataclasses import dataclass
import threading
import time

from core.constants.values import (
    FPS_LOW,
    MEMORY_HIGH_PERCENT,
    MEMORY_SAFE_PERCENT,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


TARGET_FPS = 30.0
MIN_ACCEPTABLE_FPS = 15.0


@dataclass(slots=True)
class SystemInfo:
    """Información minimalista del sistema."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    fps: float = 0.0
    processing_time_ms: float = 0.0
    active_tracks: int = 0
    status: str = "RUNNING"
    timestamp: float = 0.0


class SystemInfoCollector:
    """Recolector de información del sistema con caché para evitar
    llamadas frecuentes a psutil.
    """

    def __init__(self, cache_seconds: float = 0.5):
        """Args:
        cache_seconds: Tiempo de caché para la información del sistema
        """
        self.cache_seconds = cache_seconds
        self._cached_info: SystemInfo | None = None
        self._last_update: float = 0.0
        self._lock = threading.Lock()

        self._cpu_history = deque(maxlen=5)
        self._memory_history = deque(maxlen=5)

        self._pipeline_status: str = "RUNNING"

    def get_info(
        self, fps: float = 0.0, processing_time_ms: float = 0.0, active_tracks: int = 0
    ) -> SystemInfo:
        """Obtiene información del sistema con caché.

        Args:
            fps: FPS actual
            processing_time_ms: Tiempo de procesamiento en ms
            active_tracks: Número de tracks activos

        Returns:
            SystemInfo: Información del sistema
        """
        current_time = time.time()

        if (current_time - self._last_update) < self.cache_seconds and self._cached_info:
            self._cached_info.fps = fps
            self._cached_info.processing_time_ms = processing_time_ms
            self._cached_info.active_tracks = active_tracks
            self._cached_info.status = self._pipeline_status
            return self._cached_info

        with self._lock:
            cpu_percent, memory_percent, memory_used_mb = self._get_system_metrics()

            self._cpu_history.append(cpu_percent)
            self._memory_history.append(memory_percent)

            cpu_smooth = sum(self._cpu_history) / len(self._cpu_history) if self._cpu_history else 0
            memory_smooth = (
                sum(self._memory_history) / len(self._memory_history) if self._memory_history else 0
            )

            info = SystemInfo(
                cpu_percent=round(cpu_smooth, 1),
                memory_percent=round(memory_smooth, 1),
                memory_used_mb=round(memory_used_mb, 1),
                fps=round(fps, 1),
                processing_time_ms=round(processing_time_ms, 1),
                active_tracks=active_tracks,
                status=self._pipeline_status,
                timestamp=current_time,
            )

            self._cached_info = info
            self._last_update = current_time

            return info

    def _get_system_metrics(self) -> tuple:
        """Obtiene métricas del sistema.

        Returns:
            tuple: (cpu_percent, memory_percent, memory_used_mb)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0, 0.0

        try:
            cpu_percent = psutil.cpu_percent(interval=None)

            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)

            return cpu_percent, memory_percent, memory_used_mb

        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return 0.0, 0.0, 0.0

    def set_status(self, status: str) -> None:
        """Actualiza el estado del pipeline.

        Args:
            status: Estado del pipeline (RUNNING, PAUSED, STOPPED, ERROR, IDLE)
        """
        self._pipeline_status = status

    def get_status_icon(self, status: str | None = None) -> str:
        """Obtiene el icono correspondiente al estado.

        Args:
            status: Estado opcional (si no se proporciona, usa el actual)

        Returns:
            str: Icono del estado
        """
        if status is None:
            status = self._pipeline_status

        status_icons = {
            "RUNNING": "🟢",
            "PAUSED": "⏸️",
            "STOPPED": "⏹️",
            "ERROR": "🔴",
            "IDLE": "🟡",
        }
        return status_icons.get(status, "⚪")

    def get_color_for_performance(self, fps: float, cpu_percent: float) -> tuple:
        """Determina el color según el rendimiento.

        Returns:
            tuple: Color en formato (B, G, R)
        """
        if fps >= TARGET_FPS and cpu_percent < MEMORY_SAFE_PERCENT:
            return (0, 255, 0)
        if fps >= FPS_LOW and cpu_percent < MEMORY_HIGH_PERCENT:
            return (0, 255, 255)
        return (0, 0, 255)

    def get_color_for_memory(self, memory_percent: float) -> tuple:
        """Determina el color según el uso de memoria.

        Returns:
            tuple: Color en formato (B, G, R)
        """
        if memory_percent < MEMORY_SAFE_PERCENT:
            return (0, 255, 0)
        if memory_percent < MEMORY_HIGH_PERCENT:
            return (0, 255, 255)
        return (0, 0, 255)

    def get_status_text(self) -> str:
        """Obtiene el texto del estado actual.

        Returns:
            str: Texto del estado
        """
        status_texts = {
            "RUNNING": "RUNNING",
            "PAUSED": "PAUSED",
            "STOPPED": "STOPPED",
            "ERROR": "ERROR",
            "IDLE": "IDLE",
        }
        return status_texts.get(self._pipeline_status, "UNKNOWN")


_system_info_collector = SystemInfoCollector(cache_seconds=0.5)


def get_system_info(
    fps: float = 0.0, processing_time_ms: float = 0.0, active_tracks: int = 0
) -> SystemInfo:
    """Función de conveniencia para obtener información del sistema.

    Args:
        fps: FPS actual
        processing_time_ms: Tiempo de procesamiento en ms
        active_tracks: Número de tracks activos

    Returns:
        SystemInfo: Información del sistema
    """
    return _system_info_collector.get_info(fps, processing_time_ms, active_tracks)


def set_system_status(status: str) -> None:
    """Actualiza el estado del sistema globalmente.

    Args:
        status: Estado del sistema (RUNNING, PAUSED, STOPPED, ERROR, IDLE)
    """
    _system_info_collector.set_status(status)


def get_system_status() -> str:
    """Obtiene el estado actual del sistema.

    Returns:
        str: Estado actual del sistema
    """
    return _system_info_collector._pipeline_status


def get_system_info_collector() -> SystemInfoCollector:
    """Obtiene la instancia global del recolector de información.

    Returns:
        SystemInfoCollector: Instancia global
    """
    return _system_info_collector
