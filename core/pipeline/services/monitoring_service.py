"""
Servicio de monitoreo y métricas.

Responsable de:
- Recolectar métricas del sistema
- Monitorear la salud del pipeline
- Generar estadísticas de rendimiento
- Detectar problemas y alertar
"""

import time
import threading
from typing import Optional, Dict, Any, List
from collections import deque

from utils.logger import LoggerMixin
from utils.helpers import get_memory_usage


class MonitoringService(LoggerMixin):
    """
    Servicio especializado en monitoreo y métricas.
    """

    def __init__(
        self,
        config,
        interval: float = 5.0,
        max_history: int = 60,
    ):
        self.config = config
        self.interval = interval
        self.max_history = max_history

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._metrics_history: deque = deque(maxlen=max_history)

        self._current_metrics = {
            'fps': 0.0,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used_mb': 0.0,
            'active_tracks': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'errors': 0,
            'uptime_seconds': 0.0,
        }

        self._start_time = time.time()
        self._frame_counter = 0
        self._dropped_counter = 0
        self._error_counter = 0

        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

        self.logger.info(
            "MonitoringService inicializado",
            interval=interval,
            max_history=max_history
        )

    def start(self) -> None:
        """Inicia el servicio de monitoreo."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="MonitoringService",
            daemon=True
        )
        self._thread.start()
        self.logger.info("Servicio de monitoreo iniciado")

    def stop(self) -> None:
        """Detiene el servicio de monitoreo."""
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self.logger.info("Servicio de monitoreo detenido")

    def _monitor_loop(self) -> None:
        """Bucle principal de monitoreo."""
        self.logger.info("Bucle de monitoreo iniciado")

        while self._running:
            try:
                self._collect_metrics()
                time.sleep(self.interval)
            except Exception as e:
                self.logger.error(f"Error en monitoreo: {e}", exc_info=True)
                time.sleep(1.0)

        self.logger.info("Bucle de monitoreo terminado")

    def _collect_metrics(self) -> None:
        """Recolecta métricas del sistema."""
        self._current_fps = self._fps_counter / (time.time() - self._fps_timer + 0.001)
        self._fps_counter = 0
        self._fps_timer = time.time()

        mem = get_memory_usage()

        self._current_metrics.update({
            'fps': self._current_fps,
            'cpu_percent': mem.get('percent', 0.0),
            'memory_percent': mem.get('system_percent', 0.0),
            'memory_used_mb': mem.get('rss_mb', 0.0),
            'active_tracks': self._current_metrics.get('active_tracks', 0),
            'processed_frames': self._frame_counter,
            'dropped_frames': self._dropped_counter,
            'errors': self._error_counter,
            'uptime_seconds': time.time() - self._start_time,
        })

        self._metrics_history.append(self._current_metrics.copy())

        self._check_health()

    def _check_health(self) -> None:
        """Verifica la salud del sistema."""
        if self._current_metrics['memory_percent'] > 80:
            self.logger.warning(
                f"Memoria alta: {self._current_metrics['memory_percent']:.1f}%"
            )

        if self._current_metrics['fps'] < 5 and self._current_metrics['uptime_seconds'] > 30:
            self.logger.warning(
                f"FPS bajo: {self._current_metrics['fps']:.1f}"
            )

        if self._error_counter > 10:
            self.logger.warning(
                f"Demasiados errores: {self._error_counter}"
            )

    def update_active_tracks(self, count: int) -> None:
        """Actualiza el número de tracks activos."""
        self._current_metrics['active_tracks'] = count

    def record_processed_frame(self) -> None:
        """Registra un frame procesado."""
        self._frame_counter += 1
        self._fps_counter += 1

    def record_dropped_frame(self) -> None:
        """Registra un frame descartado."""
        self._dropped_counter += 1

    def record_error(self) -> None:
        """Registra un error."""
        self._error_counter += 1

    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtiene las métricas actuales."""
        return self._current_metrics.copy()

    def get_metrics_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de métricas.

        Args:
            limit: Número máximo de registros a retornar

        Returns:
            List[Dict[str, Any]]: Historial de métricas
        """
        return list(self._metrics_history)[-limit:]

    def get_average_fps(self, window: int = 10) -> float:
        """
        Obtiene el FPS promedio en la ventana especificada.

        Args:
            window: Número de registros a considerar

        Returns:
            float: FPS promedio
        """
        history = list(self._metrics_history)[-window:]
        if not history:
            return 0.0

        fps_values = [m.get('fps', 0.0) for m in history if m.get('fps', 0.0) > 0]
        if not fps_values:
            return 0.0

        return sum(fps_values) / len(fps_values)

    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene el estado de salud del sistema."""
        metrics = self._current_metrics

        issues = []

        if metrics['memory_percent'] > 80:
            issues.append('high_memory')

        if metrics['fps'] < 5 and metrics['uptime_seconds'] > 30:
            issues.append('low_fps')

        if metrics['dropped_frames'] > 100:
            issues.append('high_drop_rate')

        if metrics['errors'] > 5:
            issues.append('too_many_errors')

        status = 'healthy' if not issues else 'unhealthy'

        return {
            'status': status,
            'issues': issues,
            'metrics': metrics,
        }

    def get_stats(self) -> dict:
        """Obtiene estadísticas del servicio."""
        return {
            'current_metrics': self._current_metrics.copy(),
            'history_size': len(self._metrics_history),
            'is_running': self._running,
            'uptime_seconds': time.time() - self._start_time,
            'average_fps': self.get_average_fps(),
            'health': self.get_health_status(),
        }

    @property
    def is_running(self) -> bool:
        return self._running
