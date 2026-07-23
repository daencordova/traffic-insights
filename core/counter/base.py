"""
Contador inteligente de vehículos.

Este módulo implementa el contador principal del sistema que orquesta
la gestión de líneas de conteo, detección de cruces y recolección
de estadísticas de vehículos.

El contador es responsable de:
- Gestionar líneas de conteo virtuales
- Detectar cruces de vehículos a través de líneas
- Recolectar estadísticas de conteo
- Mantener historial de eventos
- Calcular métricas de tráfico

Componentes principales:
- LineManager: Gestión de líneas de conteo
- CrossingDetector: Detección de cruces de líneas
- StatisticsCollector: Recolección de estadísticas
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from utils.logger import LoggerMixin
from core.interfaces import ICounter
from core.counter.line_manager import LineManager, CountingLine
from core.counter.crossing_detector import CrossingDetector
from core.counter.statistics_collector import StatisticsCollector, VehicleEvent


class VehicleCounter(ICounter, LoggerMixin):
    """
    Contador de vehículos con soporte para múltiples líneas.

    Esta clase orquesta los componentes de gestión de líneas,
    detección de cruces y recolección de estadísticas para
    proporcionar un sistema completo de conteo de vehículos.

    Características:
        - Múltiples líneas de conteo virtuales
        - Detección de dirección de cruce (up/down)
        - Estadísticas por línea y por clase de vehículo
        - Historial de eventos con timestamp
        - Cálculo de velocidades promedio
        - Conteo por minuto

    Attributes:
        line_manager: Gestor de líneas de conteo.
        crossing_detector: Detector de cruces de líneas.
        stats_collector: Recolector de estadísticas.
        config: Configuración del sistema.
        _frame_counter: Contador de frames procesados.
        _last_process_time: Tiempo de la última actualización.

    Example:
        >>> counter = VehicleCounter()
        >>> tracks = tracker.update(detections, frame)
        >>> stats = counter.process(tracks, frame)
        >>> print(f"Total vehículos: {stats['total']}")
        >>> print(f"Conteo por línea: {stats['line_counts']}")
    """

    def __init__(self, config=None) -> None:
        """
        Inicializa el contador de vehículos.

        Args:
            config: Configuración del sistema. Si es None, se usa
                la configuración global.

        Note:
            Si no hay líneas de conteo configuradas, el contador
            funcionará sin realizar conteos (solo estadísticas básicas).
        """
        from config.manager import config_manager

        self.config = config or config_manager.config
        self.logger.info("Inicializando VehicleCounter")

        self.line_manager = LineManager(self.config.counting_lines)
        self.crossing_detector = CrossingDetector()
        self.stats_collector = StatisticsCollector()

        line_count = self.line_manager.get_line_count()
        self.logger.info(
            "Contador inicializado",
            lines=line_count,
            line_ids=[line.id for line in self.line_manager.get_all_lines()]
        )

        if line_count == 0:
            self.logger.warning("No hay líneas de conteo configuradas")

        self._frame_counter = 0
        self._last_process_time = 0.0
        self._processed_tracks_count = 0

    def process(self, tracks: Dict[int, Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        """
        Procesa los tracks y actualiza los conteos.

        Args:
            tracks: Diccionario de tracks activos del tracker.
            frame: Frame actual (necesario para dimensiones y contexto).

        Returns:
            Dict[str, Any]: Estadísticas actualizadas incluyendo:
                - total: Total de vehículos contados
                - line_counts: Conteo por línea
                - class_counts: Conteo por clase
                - avg_speed: Velocidad promedio
                - active_objects: Número de objetos activos
                - frame_counter: Número de frame procesado

        Example:
            >>> tracks = tracker.update(detections, frame)
            >>> stats = counter.process(tracks, frame)
            >>> if stats['total'] > 100:
            ...     print("Alto volumen de tráfico detectado")

        Note:
            El procesamiento se realiza por cada track activo,
            verificando si ha cruzado alguna línea de conteo.
        """
        start_time = time.perf_counter()
        self._frame_counter += 1
        self._processed_tracks_count = 0

        if not self._is_valid_input(frame, tracks):
            return self.get_stats()

        if not self.line_manager.has_active_lines():
            return self.get_stats()

        height = frame.shape[0]
        self._process_all_tracks(tracks, height)

        self._update_minute_stats()

        self._log_performance(start_time)

        return self.get_stats()

    def _is_valid_input(self, frame: np.ndarray, tracks: Dict[int, Dict[str, Any]]) -> bool:
        """
        Valida la entrada del método process.

        Args:
            frame: Frame a validar.
            tracks: Tracks a validar.

        Returns:
            bool: True si la entrada es válida.
        """
        if frame is None or frame.size == 0:
            self.logger.debug("Frame inválido recibido")
            return False

        if not isinstance(tracks, dict):
            self.logger.debug("Tracks inválidos (no es diccionario)")
            return False

        return True

    def _process_all_tracks(self, tracks: Dict[int, Dict[str, Any]], height: int) -> None:
        """
        Procesa todos los tracks para detección de cruces.

        Args:
            tracks: Diccionario de tracks activos.
            height: Alto del frame.
        """
        for object_id, track_data in tracks.items():
            try:
                if self._process_single_track(object_id, track_data, height):
                    self._processed_tracks_count += 1
            except Exception as e:
                self.logger.debug(
                    "Error procesando track",
                    object_id=object_id,
                    error=str(e)
                )
                continue

    def _process_single_track(self, object_id: int, track_data: Dict[str, Any], height: int) -> bool:
        """
        Procesa un track individual para detección de cruces.

        Args:
            object_id: ID del objeto.
            track_data: Datos del track.
            height: Alto del frame (para coordenadas).

        Returns:
            bool: True si se procesó correctamente (cruzó alguna línea).
        """
        if not self._validate_track_data(track_data):
            return False

        centroid = track_data["centroid"]
        crossed_any = False

        for line in self.line_manager.get_all_lines():
            if self._check_line_crossing(object_id, centroid, line, height):
                crossed_any = True

        self._record_track_velocity(object_id, track_data)

        return crossed_any

    def _validate_track_data(self, track_data: Dict[str, Any]) -> bool:
        """
        Valida los datos de un track.

        Args:
            track_data: Datos del track a validar.

        Returns:
            bool: True si los datos son válidos.
        """
        if not isinstance(track_data, dict):
            return False

        centroid = track_data.get("centroid")
        if centroid is None:
            return False

        if not isinstance(centroid, (tuple, list)) or len(centroid) != 2:
            return False

        x, y = centroid
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False

        if x < 0 or y < 0:
            return False

        return True

    def _check_line_crossing(
        self,
        object_id: int,
        centroid: Tuple[int, int],
        line: CountingLine,
        height: int
    ) -> bool:
        """
        Verifica si un objeto ha cruzado una línea específica.

        Args:
            object_id: ID del objeto.
            centroid: Centroide actual.
            line: Línea de conteo.
            height: Alto del frame.

        Returns:
            bool: True si el objeto cruzó la línea.
        """
        crossed = self.crossing_detector.detect_crossing(
            object_id=object_id,
            current_position=centroid,
            line=line,
            height=height
        )

        if crossed:
            self._handle_crossing(object_id, line, centroid)
            return True

        return False

    def _handle_crossing(
        self,
        object_id: int,
        line: CountingLine,
        centroid: Tuple[int, int]
    ) -> None:
        """
        Maneja un evento de cruce de línea.

        Args:
            object_id: ID del objeto.
            line: Línea de conteo cruzada.
            centroid: Posición del objeto.
        """
        pass

    def _record_crossing(
        self,
        object_id: int,
        line: CountingLine,
        track_data: Dict[str, Any],
        centroid: Tuple[int, int]
    ) -> None:
        """
        Registra un cruce en el recolector de estadísticas.

        Args:
            object_id: ID del objeto.
            line: Línea de conteo.
            track_data: Datos del track.
            centroid: Posición del objeto.
        """
        self.stats_collector.record_crossing(
            object_id=object_id,
            line_id=line.id,
            line_name=line.name,
            track_data=track_data,
            centroid=centroid
        )

        self.logger.debug(
            "Vehículo contado",
            object_id=object_id,
            line=line.id,
            total=self.stats_collector.get_total_count()
        )

    def _record_track_velocity(self, object_id: int, track_data: Dict[str, Any]) -> None:
        """
        Registra la velocidad de un track.

        Args:
            object_id: ID del objeto.
            track_data: Datos del track.
        """
        velocity = track_data.get("velocity", (0, 0))
        if isinstance(velocity, (tuple, list)) and len(velocity) == 2:
            self.stats_collector.record_speed(object_id, velocity)

    def _update_minute_stats(self) -> None:
        """Actualiza los conteos por minuto."""
        total = self.stats_collector.get_total_count()
        self.stats_collector.update_minute_counts(total)

    def _log_performance(self, start_time: float) -> None:
        """
        Registra métricas de rendimiento.

        Args:
            start_time: Timestamp de inicio del procesamiento.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._last_process_time = elapsed_ms

        if self._frame_counter % 100 == 0 and self._processed_tracks_count > 0:
            self.logger.debug(
                "Procesamiento de conteo",
                frames=self._frame_counter,
                tracks_processed=self._processed_tracks_count,
                total=self.stats_collector.get_total_count(),
                time_ms=f"{elapsed_ms:.2f}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas detalladas del conteo.

        Returns:
            Dict[str, Any]: Estadísticas actuales incluyendo:
                - total: Total de vehículos contados
                - line_counts: Conteo por línea
                - class_counts: Conteo por clase
                - avg_speed: Velocidad promedio
                - max_speed: Velocidad máxima
                - min_speed: Velocidad mínima
                - avg_per_minute: Promedio por minuto
                - count_rate: Tasa de conteo
                - total_events: Total de eventos
                - runtime_seconds: Tiempo de ejecución
                - active_objects: Objetos activos
                - frame_counter: Número de frame
                - last_process_time_ms: Tiempo de procesamiento

        Example:
            >>> stats = counter.get_stats()
            >>> print(f"Total: {stats['total']}")
            >>> print(f"Velocidad promedio: {stats['avg_speed']:.1f} px/frame")
        """
        stats = self.stats_collector.get_stats()
        stats["active_objects"] = self.crossing_detector.get_active_objects()
        stats["frame_counter"] = self._frame_counter
        stats["last_process_time_ms"] = self._last_process_time
        stats["processed_tracks"] = self._processed_tracks_count

        return stats

    def get_recent_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Obtiene los últimos eventos de conteo.

        Args:
            limit: Número máximo de eventos a retornar.

        Returns:
            List[Dict[str, Any]]: Eventos recientes en formato diccionario.
        """
        return self.stats_collector.get_recent_events(limit)

    def get_line_count(self, line_id: str) -> int:
        """
        Obtiene el conteo de una línea específica.

        Args:
            line_id: ID de la línea.

        Returns:
            int: Conteo de la línea.
        """
        return self.stats_collector.get_line_count(line_id)

    def get_class_count(self, class_name: str) -> int:
        """
        Obtiene el conteo de una clase específica.

        Args:
            class_name: Nombre de la clase (ej: 'car', 'truck').

        Returns:
            int: Conteo de la clase.
        """
        return self.stats_collector.get_class_count(class_name)

    def get_crossed_lines(self, object_id: int) -> Set[str]:
        """
        Obtiene las líneas que un objeto ha cruzado.

        Args:
            object_id: ID del objeto.

        Returns:
            Set[str]: Conjunto de IDs de líneas cruzadas.
        """
        return self.crossing_detector.get_crossed_lines(object_id)

    def get_line_manager(self) -> LineManager:
        """Obtiene el gestor de líneas."""
        return self.line_manager

    def get_crossing_detector(self) -> CrossingDetector:
        """Obtiene el detector de cruces."""
        return self.crossing_detector

    def get_statistics_collector(self) -> StatisticsCollector:
        """Obtiene el recolector de estadísticas."""
        return self.stats_collector

    def get_events(self, limit: int = 100) -> List[VehicleEvent]:
        """
        Obtiene los eventos de conteo.

        Args:
            limit: Número máximo de eventos.

        Returns:
            List[VehicleEvent]: Eventos de conteo.
        """
        return self.stats_collector.events[-limit:] if self.stats_collector.events else []

    def reset_line(self, line_id: str) -> None:
        """
        Reinicia el conteo de una línea específica.

        Args:
            line_id: ID de la línea a reiniciar.

        Example:
            >>> counter.reset_line("line_1")
            >>> # El conteo de la línea 1 se reinicia a 0
        """
        self.crossing_detector.reset_line(line_id)
        self.stats_collector.line_counts[line_id] = 0
        self.logger.info(f"Línea {line_id} reiniciada")

    def reset(self) -> None:
        """
        Reinicia todos los contadores y estadísticas.

        Example:
            >>> counter.reset()
            >>> # Todos los conteos se reinician a 0
        """
        self.logger.info(
            "Reiniciando contador",
            total=self.stats_collector.get_total_count(),
            lines=self.line_manager.get_total_lines()
        )

        self.crossing_detector.clear()
        self.stats_collector.reset()
        self._frame_counter = 0
        self._last_process_time = 0.0
        self._processed_tracks_count = 0

        self.logger.info("Contador reiniciado")

    def __len__(self) -> int:
        """Retorna el número total de conteos."""
        return self.stats_collector.get_total_count()

    def __str__(self) -> str:
        """Representación en string del contador."""
        return (
            f"VehicleCounter(total={self.stats_collector.get_total_count()}, "
            f"lines={self.line_manager.get_line_count()})"
        )
