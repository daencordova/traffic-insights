"""
Contador inteligente de vehículos con análisis de trayectoria
"""

import time
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np

from utils.logger import LoggerMixin
from utils.geometry import check_crossing
from .interfaces import ICounter

Point = Tuple[int, int]
LineId = str
ObjectId = int
TrackDict = Dict[ObjectId, Dict[str, Any]]
StatsDict = Dict[str, Any]
LogEntry = Dict[str, Any]
LineConfig = Dict[str, Any]


class VehicleCounter(ICounter, LoggerMixin):
    """Contador de vehículos con soporte para múltiples líneas y análisis"""

    def __init__(self) -> None:
        from config.manager import config_manager
        self.config = config_manager.config
        self.logger.info("Inicializando VehicleCounter")

        self._total: int = 0
        self._line_counts: Dict[LineId, int] = {}
        self._line_crossed: Dict[ObjectId, Set[LineId]] = {}
        self._previous_positions: Dict[ObjectId, Point] = {}

        self._vehicle_classes = defaultdict(int)
        self._speed_history: Dict[ObjectId, List[float]] = {}
        self._vehicle_log: List[LogEntry] = []

        self._validate_line_configs()

        for line in self.config.counting_lines:
            line_id = line.get("id", f"line_{len(self._line_counts)}")
            self._line_counts[line_id] = 0

        self._last_count_time: float = time.time()
        self._counts_per_minute: List[int] = []

        self._start_time: float = time.time()

        self.logger.info(
            "Contador inicializado",
            lines=len(self.config.counting_lines),
            line_ids=list(self._line_counts.keys())
        )

    def _validate_line_configs(self) -> None:
        """Valida las configuraciones de líneas de conteo"""
        if not self.config.counting_lines:
            self.logger.warning("No hay líneas de conteo configuradas")
            return

        for idx, line in enumerate(self.config.counting_lines):
            if not isinstance(line, dict):
                self.logger.warning("Línea con configuración inválida", index=idx)
                continue

            points = line.get("points", [])
            if len(points) < 1:
                self.logger.warning("Línea sin puntos suficientes", index=idx, name=line.get("name", ""))
                continue

            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    self.logger.warning("Punto inválido en línea", index=idx, point=point)
                    continue

                x, y = point
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    self.logger.warning("Coordenadas no numéricas", index=idx, point=point)
                    continue

    def _validate_track_data(self, track_data: Dict[str, Any]) -> bool:
        """Valida los datos de un track"""
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

    def _validate_line(self, line: LineConfig) -> bool:
        """Valida una línea de conteo"""
        if not isinstance(line, dict):
            return False

        points = line.get("points", [])
        if not points:
            return False

        if not isinstance(points, (list, tuple)) or len(points) < 1:
            return False

        first_point = points[0]
        if not isinstance(first_point, (list, tuple)) or len(first_point) != 2:
            return False

        x, y = first_point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False

        return True

    def process(self, tracks: TrackDict, frame: np.ndarray) -> StatsDict:
        """Procesa los tracks y actualiza los conteos"""
        if frame is None or frame.size == 0:
            self.logger.debug("Frame inválido recibido")
            return self.get_stats()

        height = frame.shape[0]
        current_time = time.time()

        if not self.config.counting_lines:
            return self.get_stats()

        for obj_id, track_data in tracks.items():
            try:
                if not self._validate_track_data(track_data):
                    continue

                centroid = track_data["centroid"]

                if obj_id not in self._line_crossed:
                    self._line_crossed[obj_id] = set()

                prev_pos = self._previous_positions.get(obj_id)

                if prev_pos is not None:
                    prev_y, current_y = prev_pos[1], centroid[1]

                    for idx, line in enumerate(self.config.counting_lines):
                        if not self._validate_line(line):
                            continue

                        line_id = line.get("id", f"line_{idx}")

                        if line_id in self._line_crossed[obj_id]:
                            continue

                        points = line.get("points", [])
                        if not points:
                            continue

                        line_y = points[0][1] if points else height // 2
                        direction = line.get("direction", "down")

                        if check_crossing(prev_y, current_y, line_y, direction):
                            self._line_counts[line_id] = self._line_counts.get(line_id, 0) + 1
                            self._total += 1
                            self._line_crossed[obj_id].add(line_id)

                            label = track_data.get("label", "vehicle")
                            self._vehicle_classes[label] = self._vehicle_classes.get(label, 0) + 1

                            log_entry: LogEntry = {
                                "timestamp": time.strftime("%H:%M:%S"),
                                "object_id": obj_id,
                                "line_id": line_id,
                                "line_name": line.get("name", line_id),
                                "label": label,
                                "class_id": track_data.get("class_id", -1),
                                "centroid": centroid,
                                "velocity": track_data.get("velocity", (0, 0)),
                            }
                            self._vehicle_log.append(log_entry)

                            self.logger.debug(
                                "Vehículo contado",
                                object_id=obj_id,
                                line=line_id,
                                label=label,
                                total=self._total
                            )

                self._previous_positions[obj_id] = centroid

                velocity = track_data.get("velocity", (0, 0))
                if isinstance(velocity, (tuple, list)) and len(velocity) == 2:
                    if obj_id not in self._speed_history:
                        self._speed_history[obj_id] = []
                    speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                    self._speed_history[obj_id].append(speed)

                    if len(self._speed_history[obj_id]) > 30:
                        self._speed_history[obj_id] = self._speed_history[obj_id][-30:]

            except Exception as e:
                self.logger.debug("Error procesando track", object_id=obj_id, error=str(e))
                continue

        if current_time - self._last_count_time >= 60:
            self._counts_per_minute.append(self._total)
            self._last_count_time = current_time

            if len(self._counts_per_minute) > 60:
                self._counts_per_minute = self._counts_per_minute[-60:]

        return self.get_stats()

    def get_stats(self) -> StatsDict:
        """Retorna estadísticas detalladas"""
        avg_speed = 0
        all_speeds = []
        for speeds in self._speed_history.values():
            if speeds:
                all_speeds.extend(speeds)
        if all_speeds:
            try:
                avg_speed = float(np.mean(all_speeds))
            except Exception:
                avg_speed = 0

        stats: StatsDict = {
            "total": self._total,
            "line_counts": dict(self._line_counts),
            "vehicle_classes": dict(self._vehicle_classes),
            "active_objects": len(self._previous_positions),
            "total_logged": len(self._vehicle_log),
            "avg_speed": avg_speed,
            "avg_per_minute": float(np.mean(self._counts_per_minute)) if self._counts_per_minute else 0.0,
            "runtime_seconds": time.time() - self._start_time,
            "timestamp": time.time(),
        }

        self.logger.debug("Estadísticas actuales", total=stats["total"], active=stats["active_objects"])
        return stats

    def reset(self) -> None:
        """Reinicia todos los contadores"""
        self.logger.info(
            "Reiniciando contador",
            total=self._total,
            lines=len(self._line_counts)
        )

        self._total = 0
        for key in self._line_counts:
            self._line_counts[key] = 0
        self._line_crossed.clear()
        self._previous_positions.clear()
        self._vehicle_classes.clear()
        self._speed_history.clear()
        self._vehicle_log.clear()
        self._counts_per_minute.clear()
        self._start_time = time.time()
        self._last_count_time = time.time()

        self.logger.info("Contador reiniciado")

    def get_recent_log(self, limit: int = 20) -> List[LogEntry]:
        """Obtiene los últimos eventos de conteo"""
        return self._vehicle_log[-limit:] if self._vehicle_log else []

    def get_line_count(self, line_id: str) -> int:
        """Obtiene el conteo de una línea específica"""
        return self._line_counts.get(line_id, 0)
