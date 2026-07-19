"""
Contador inteligente de vehículos.

Orquesta los componentes de gestión de líneas, detección de cruces
y recolección de estadísticas.
"""

from typing import Any, Dict, List, Set

import numpy as np

from utils.logger import LoggerMixin
from core.interfaces import ICounter
from core.counter.line_manager import LineManager
from core.counter.crossing_detector import CrossingDetector
from core.counter.statistics_collector import StatisticsCollector, VehicleEvent


class VehicleCounter(ICounter, LoggerMixin):
    """
    Contador de vehículos con soporte para múltiples líneas.

    Orquesta los componentes de gestión de líneas, detección de cruces
    y recolección de estadísticas.

    Attributes:
        line_manager: Gestor de líneas de conteo
        crossing_detector: Detector de cruces
        stats_collector: Recolector de estadísticas
        config: Configuración del sistema
    """

    def __init__(self, config=None) -> None:
        """
        Inicializa el contador de vehículos.

        Args:
            config: Configuración del sistema (opcional)
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

    def process(self, tracks: Dict[int, Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        """
        Procesa los tracks y actualiza los conteos.

        Args:
            tracks: Diccionario de tracks activos
            frame: Frame actual

        Returns:
            Dict[str, Any]: Estadísticas actualizadas
        """
        import time
        start_time = time.perf_counter()

        self._frame_counter += 1

        if frame is None or frame.size == 0:
            self.logger.debug("Frame inválido recibido")
            return self.get_stats()

        if not self.line_manager.has_active_lines():
            return self.get_stats()

        height = frame.shape[0]

        processed_count = 0
        for object_id, track_data in tracks.items():
            try:
                if self._process_track(object_id, track_data, height):
                    processed_count += 1
            except Exception as e:
                self.logger.debug(
                    "Error procesando track",
                    object_id=object_id,
                    error=str(e)
                )
                continue

        total = self.stats_collector.get_total_count()
        self.stats_collector.update_minute_counts(total)

        self._last_process_time = (time.perf_counter() - start_time) * 1000

        if self._frame_counter % 100 == 0 and processed_count > 0:
            self.logger.debug(
                "Procesamiento de conteo",
                frames=self._frame_counter,
                tracks=len(tracks),
                processed=processed_count,
                total=total
            )

        return self.get_stats()

    def _process_track(self, object_id: int, track_data: Dict[str, Any], height: int) -> bool:
        """
        Procesa un track individual.

        Args:
            object_id: ID del objeto
            track_data: Datos del track
            height: Alto del frame

        Returns:
            bool: True si se procesó correctamente
        """
        if not self._validate_track_data(track_data):
            return False

        centroid = track_data["centroid"]
        crossed_any = False

        for line in self.line_manager.get_all_lines():
            crossed = self.crossing_detector.detect_crossing(
                object_id=object_id,
                current_position=centroid,
                line=line,
                height=height
            )

            if crossed:
                self.stats_collector.record_crossing(
                    object_id=object_id,
                    line_id=line.id,
                    line_name=line.name,
                    track_data=track_data,
                    centroid=centroid
                )

                crossed_any = True

                self.logger.debug(
                    "Vehículo contado",
                    object_id=object_id,
                    line=line.id,
                    total=self.stats_collector.get_total_count()
                )

        velocity = track_data.get("velocity", (0, 0))
        if isinstance(velocity, (tuple, list)) and len(velocity) == 2:
            self.stats_collector.record_speed(object_id, velocity)

        return crossed_any

    def _validate_track_data(self, track_data: Dict[str, Any]) -> bool:
        """
        Valida los datos de un track.

        Args:
            track_data: Datos del track a validar

        Returns:
            bool: True si los datos son válidos
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

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas detalladas.

        Returns:
            Dict[str, Any]: Estadísticas actuales
        """
        stats = self.stats_collector.get_stats()
        stats["active_objects"] = self.crossing_detector.get_active_objects()
        stats["frame_counter"] = self._frame_counter
        stats["last_process_time_ms"] = self._last_process_time

        return stats

    def get_recent_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Obtiene los últimos eventos de conteo.

        Args:
            limit: Número máximo de eventos

        Returns:
            List[Dict[str, Any]]: Eventos recientes
        """
        return self.stats_collector.get_recent_events(limit)

    def get_line_count(self, line_id: str) -> int:
        """
        Obtiene el conteo de una línea específica.

        Args:
            line_id: ID de la línea

        Returns:
            int: Conteo de la línea
        """
        return self.stats_collector.get_line_count(line_id)

    def get_class_count(self, class_name: str) -> int:
        """
        Obtiene el conteo de una clase específica.

        Args:
            class_name: Nombre de la clase

        Returns:
            int: Conteo de la clase
        """
        return self.stats_collector.get_class_count(class_name)

    def get_crossed_lines(self, object_id: int) -> Set[str]:
        """
        Obtiene las líneas que un objeto ha cruzado.

        Args:
            object_id: ID del objeto

        Returns:
            Set[str]: Conjunto de IDs de líneas cruzadas
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
            limit: Número máximo de eventos

        Returns:
            List[VehicleEvent]: Eventos de conteo
        """
        return self.stats_collector.events[-limit:] if self.stats_collector.events else []

    def reset_line(self, line_id: str) -> None:
        """
        Reinicia el conteo de una línea específica.

        Args:
            line_id: ID de la línea a reiniciar
        """
        self.crossing_detector.reset_line(line_id)
        self.stats_collector.line_counts[line_id] = 0
        self.logger.info(f"Línea {line_id} reiniciada")

    def reset(self) -> None:
        """Reinicia todos los contadores y estadísticas."""
        self.logger.info(
            "Reiniciando contador",
            total=self.stats_collector.get_total_count(),
            lines=self.line_manager.get_total_lines()
        )

        self.crossing_detector.clear()
        self.stats_collector.reset()
        self._frame_counter = 0
        self._last_process_time = 0.0

        self.logger.info("Contador reiniciado")

    def __len__(self) -> int:
        """Retorna el número total de conteos."""
        return self.stats_collector.get_total_count()

    def __str__(self) -> str:
        """Representación en string del contador."""
        return f"VehicleCounter(total={self.stats_collector.get_total_count()}, lines={self.line_manager.get_line_count()})"
